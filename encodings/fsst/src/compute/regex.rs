// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

//! Regex matching on FSST-compressed string arrays.
//!
//! The kernel runs in three tiers, picking the cheapest one that can
//! handle the pattern:
//!
//! 1. **Literal Prefix / Contains** (cheapest). When
//!    [`RegexLiteralShape::analyze`] classifies the pattern as `^literal`
//!    or unanchored `literal` (no metacharacters or escapes), we feed
//!    the literal directly into the existing LIKE infrastructure
//!    ([`FlatPrefixDfa`] / [`FlatContainsDfa`]). These DFAs are
//!    hand-tuned for substring matching and have the lowest overhead.
//!
//! 2. **General regex DFA over FSST symbols**
//!    ([`FsstMatcher::try_new_regex`]). For arbitrary patterns we
//!    compile a byte-level DFA via `regex_automata`, BFS its reachable
//!    states, and lift the byte transitions into a per-symbol table.
//!    The runtime scan is the same per-row loop, just driven by a
//!    larger table. Bounded by [`RegexFsstDfa::MAX_STATES`] to keep the
//!    table size reasonable.
//!
//! 3. **Canonical fallback**. Patterns whose DFA is too large, that use
//!    regex features `regex_automata` rejects, or that hit an internal
//!    quit state return `Ok(None)` from the kernel. The executor then
//!    canonicalises to `VarBinViewArray` and runs the compiled regex
//!    against decompressed strings.

use vortex_array::ArrayRef;
use vortex_array::ArrayView;
use vortex_array::ExecutionCtx;
use vortex_array::IntoArray;
use vortex_array::arrays::BoolArray;
use vortex_array::arrays::PrimitiveArray;
use vortex_array::arrays::varbin::VarBinArrayExt;
use vortex_array::match_each_integer_ptype;
use vortex_array::scalar_fn::fns::regex::RegexKernel;
use vortex_array::scalar_fn::fns::regex::RegexLiteralShape;
use vortex_array::scalar_fn::fns::regex::RegexOptions;
use vortex_error::VortexResult;

use crate::FSST;
use crate::FSSTArrayExt;
use crate::dfa::FsstMatcher;
use crate::dfa::dfa_scan_to_bitbuf;

impl RegexKernel for FSST {
    fn regex(
        array: ArrayView<'_, Self>,
        pattern: &ArrayRef,
        options: RegexOptions,
        ctx: &mut ExecutionCtx,
    ) -> VortexResult<Option<ArrayRef>> {
        let Some(pattern_scalar) = pattern.as_constant() else {
            return Ok(None);
        };
        let Some(utf8) = pattern_scalar.as_utf8_opt() else {
            return Ok(None);
        };
        let Some(pattern_str) = utf8.value() else {
            return Ok(None);
        };

        let symbols = array.symbols();
        let symbol_lengths = array.symbol_lengths();

        // The literal-shape fast paths use byte-exact DFAs, so they only
        // apply when the case-sensitivity matches the encoding. Force the
        // analyzer to ignore literal shapes if the caller asked for case
        // folding.
        let literal_shape = if options.case_insensitive {
            None
        } else {
            RegexLiteralShape::analyze(pattern_str.as_str())
        };

        // Tier 1: literal Prefix / Contains hit the dedicated DFAs.
        //
        // Tier 2: anything else compiles to a general regex DFA lifted
        // over the symbol table. Exact / Suffix shapes go through the
        // general DFA because the literal DFAs only model "starts with"
        // and "contains anywhere", which isn't equivalent to "equals
        // exactly" or "ends with".
        //
        // Tier 3: anything the general DFA can't build (oversized state
        // space, unsupported regex feature) returns `Ok(None)` and the
        // canonical execution path takes over.
        let matcher = match literal_shape {
            Some(RegexLiteralShape::Prefix(prefix)) => FsstMatcher::try_new_prefix_literal(
                symbols.as_slice(),
                symbol_lengths.as_slice(),
                prefix.as_bytes(),
            )?,
            Some(RegexLiteralShape::Contains(needle)) => FsstMatcher::try_new_contains_literal(
                symbols.as_slice(),
                symbol_lengths.as_slice(),
                needle.as_bytes(),
            )?,
            // Exact / Suffix / non-literal / case-insensitive patterns
            // all go through the general DFA, which honours
            // case-insensitivity via the regex syntax config.
            _ => FsstMatcher::try_new_regex(
                symbols.as_slice(),
                symbol_lengths.as_slice(),
                pattern_str.as_str(),
                options.case_insensitive,
            )?,
        };
        let Some(matcher) = matcher else {
            return Ok(None);
        };

        let negated = options.negated;
        let codes = array.codes();
        let offsets = codes.offsets().clone().execute::<PrimitiveArray>(ctx)?;
        let all_bytes = codes.bytes();
        let all_bytes = all_bytes.as_slice();
        let n = codes.len();

        let result = match_each_integer_ptype!(offsets.ptype(), |T| {
            let off = offsets.as_slice::<T>();
            dfa_scan_to_bitbuf(n, off, all_bytes, negated, |codes| matcher.matches(codes))
        });

        let validity = array
            .codes()
            .validity()?
            .union_nullability(pattern_scalar.dtype().nullability());

        Ok(Some(BoolArray::new(result, validity).into_array()))
    }
}

#[cfg(test)]
mod tests {
    use std::sync::LazyLock;

    use vortex_array::ArrayView;
    use vortex_array::Canonical;
    use vortex_array::IntoArray;
    use vortex_array::VortexSessionExecute;
    use vortex_array::arrays::BoolArray;
    use vortex_array::arrays::ConstantArray;
    use vortex_array::arrays::VarBinArray;
    use vortex_array::arrays::scalar_fn::ScalarFnFactoryExt;
    use vortex_array::assert_arrays_eq;
    use vortex_array::dtype::DType;
    use vortex_array::dtype::Nullability;
    use vortex_array::scalar_fn::fns::regex::Regex;
    use vortex_array::scalar_fn::fns::regex::RegexKernel;
    use vortex_array::scalar_fn::fns::regex::RegexOptions;
    use vortex_array::session::ArraySession;
    use vortex_error::VortexResult;
    use vortex_session::VortexSession;

    use crate::FSST;
    use crate::FSSTArray;
    use crate::fsst_compress;
    use crate::fsst_train_compressor;

    static SESSION: LazyLock<VortexSession> =
        LazyLock::new(|| VortexSession::empty().with::<ArraySession>());

    fn make_fsst(strings: &[Option<&str>], nullability: Nullability) -> FSSTArray {
        let varbin = VarBinArray::from_iter(strings.iter().copied(), DType::Utf8(nullability));
        let compressor = fsst_train_compressor(&varbin);
        let len = varbin.len();
        let dtype = varbin.dtype().clone();
        fsst_compress(
            varbin,
            len,
            &dtype,
            &compressor,
            &mut SESSION.create_execution_ctx(),
        )
    }

    fn run_regex(array: FSSTArray, pattern: &str, opts: RegexOptions) -> VortexResult<BoolArray> {
        let len = array.len();
        let arr = array.into_array();
        let pattern = ConstantArray::new(pattern, len).into_array();
        let result = Regex
            .try_new_array(len, opts, [arr, pattern])?
            .into_array()
            .execute::<Canonical>(&mut SESSION.create_execution_ctx())?;
        Ok(result.into_bool())
    }

    #[test]
    fn fsst_regex_anchored() -> VortexResult<()> {
        let fsst = make_fsst(
            &[
                Some("http://example.com"),
                Some("http://test.org"),
                Some("ftp://files.net"),
                Some("http://vortex.dev"),
                Some("ssh://server.io"),
            ],
            Nullability::NonNullable,
        );
        let result = run_regex(fsst, "^http://", RegexOptions::default())?;
        assert_arrays_eq!(
            &result,
            &BoolArray::from_iter([true, true, false, true, false])
        );
        Ok(())
    }

    #[test]
    fn fsst_regex_contains_alternation() -> VortexResult<()> {
        let fsst = make_fsst(
            &[
                Some("hello world"),
                Some("say hello"),
                Some("goodbye"),
                Some("worldview"),
            ],
            Nullability::NonNullable,
        );
        let result = run_regex(fsst, "(hello|world)", RegexOptions::default())?;
        assert_arrays_eq!(&result, &BoolArray::from_iter([true, true, false, true]));
        Ok(())
    }

    #[test]
    fn fsst_regex_case_insensitive_negated() -> VortexResult<()> {
        // Combines the negated and case-insensitive flags; the easiest one to
        // accidentally flip a sign on.
        let fsst = make_fsst(
            &[Some("AbC"), Some("xyz"), Some("abZ"), Some("abcdef")],
            Nullability::NonNullable,
        );
        let opts = RegexOptions {
            negated: true,
            case_insensitive: true,
        };
        let result = run_regex(fsst, "^ab", opts)?;
        // Matches: AbC, abZ, abcdef -> three match, one doesn't -> negated: 0,1,0,0
        assert_arrays_eq!(&result, &BoolArray::from_iter([false, true, false, false]));
        Ok(())
    }

    /// Anchored prefixes should be handled directly by the FSST DFA without
    /// falling back to canonical decompression.
    #[test]
    fn fsst_regex_anchored_prefix_kernel_handles() -> VortexResult<()> {
        let fsst = make_fsst(
            &[Some("http://a.com"), Some("ftp://b.com")],
            Nullability::NonNullable,
        );
        let pattern = ConstantArray::new("^http://", fsst.len()).into_array();
        let mut ctx = SESSION.create_execution_ctx();

        let view: ArrayView<'_, FSST> = fsst.as_view();
        let result =
            <FSST as RegexKernel>::regex(view, &pattern, RegexOptions::default(), &mut ctx)?;
        assert!(result.is_some(), "FSST RegexKernel should handle ^prefix");
        assert_arrays_eq!(result.unwrap(), BoolArray::from_iter([true, false]));
        Ok(())
    }

    /// Unanchored literals lower to the contains DFA.
    #[test]
    fn fsst_regex_contains_literal_kernel_handles() -> VortexResult<()> {
        let fsst = make_fsst(
            &[Some("hello world"), Some("goodbye")],
            Nullability::NonNullable,
        );
        let pattern = ConstantArray::new("world", fsst.len()).into_array();
        let mut ctx = SESSION.create_execution_ctx();

        let view: ArrayView<'_, FSST> = fsst.as_view();
        let result =
            <FSST as RegexKernel>::regex(view, &pattern, RegexOptions::default(), &mut ctx)?;
        assert!(
            result.is_some(),
            "FSST RegexKernel should handle literal contains"
        );
        assert_arrays_eq!(result.unwrap(), BoolArray::from_iter([true, false]));
        Ok(())
    }

    /// Patterns with regex metacharacters now go through the general DFA
    /// instead of falling back. Verify the kernel returns a result and
    /// that the result is correct.
    #[test]
    fn fsst_regex_dfa_handles_metacharacters() -> VortexResult<()> {
        let fsst = make_fsst(
            &[Some("abc"), Some("axc"), Some("def"), Some("abcdef")],
            Nullability::NonNullable,
        );

        // Each (pattern, expected) tuple: the general DFA should handle
        // the pattern and return the indicated bool mask.
        let cases: &[(&str, [bool; 4])] = &[
            // any char between a and c
            ("a.c", [true, true, false, true]),
            // alternation
            ("(abc|def)", [true, false, true, true]),
            // anchored exact (only `abc` itself, not `abcdef`)
            ("^abc$", [true, false, false, false]),
            // anchored suffix
            ("def$", [false, false, true, true]),
            // character class with quantifier
            ("a[bx]c+", [true, true, false, true]),
        ];

        let mut ctx = SESSION.create_execution_ctx();
        for &(pattern, expected) in cases {
            let view: ArrayView<'_, FSST> = fsst.as_view();
            let pat = ConstantArray::new(pattern, fsst.len()).into_array();
            let result =
                <FSST as RegexKernel>::regex(view, &pat, RegexOptions::default(), &mut ctx)?
                    .unwrap_or_else(|| panic!("pattern `{pattern}` should hit the FSST DFA"));
            assert_arrays_eq!(result, BoolArray::from_iter(expected));
        }
        Ok(())
    }

    /// Case-insensitive matching now flows through the general DFA via
    /// the regex syntax config.
    #[test]
    fn fsst_regex_dfa_handles_case_insensitive() -> VortexResult<()> {
        let fsst = make_fsst(
            &[Some("ABC"), Some("abc"), Some("aBcD"), Some("xyz")],
            Nullability::NonNullable,
        );
        let mut ctx = SESSION.create_execution_ctx();

        let view: ArrayView<'_, FSST> = fsst.as_view();
        let pat = ConstantArray::new("^abc", fsst.len()).into_array();
        let result = <FSST as RegexKernel>::regex(
            view,
            &pat,
            RegexOptions {
                negated: false,
                case_insensitive: true,
            },
            &mut ctx,
        )?
        .expect("case-insensitive should hit the general DFA");
        assert_arrays_eq!(result, BoolArray::from_iter([true, true, true, false]));
        Ok(())
    }

    /// A regex whose DFA blows past the FSST DFA state cap must fall
    /// back to canonical execution. `(a|b)*a(a|b){n}` is the classic
    /// NFA→DFA exponential blow-up: the DFA needs `2^(n+1)` states.
    /// With n=10 we ask for ~2048 states, well past the 512 cap.
    #[test]
    fn fsst_regex_falls_back_when_dfa_too_large() -> VortexResult<()> {
        let fsst = make_fsst(&[Some("aaa"), Some("bbb")], Nullability::NonNullable);
        let pattern = "(a|b)*a(a|b){10}";

        let mut ctx = SESSION.create_execution_ctx();
        let view: ArrayView<'_, FSST> = fsst.as_view();
        let pat = ConstantArray::new(pattern, fsst.len()).into_array();
        let result = <FSST as RegexKernel>::regex(view, &pat, RegexOptions::default(), &mut ctx)?;
        assert!(
            result.is_none(),
            "oversized regex DFA should fall back to canonical"
        );
        Ok(())
    }

    /// End-to-end through `Regex.try_new_array` + optimize for a few
    /// non-literal regex shapes, making sure validity carries through.
    #[test]
    fn fsst_regex_general_with_nulls() -> VortexResult<()> {
        let fsst = make_fsst(
            &[Some("alpha"), None, Some("alphabet"), Some("beta"), None],
            Nullability::Nullable,
        );
        // `a.+a` matches "alpha" (a..a — yes via greedy) and "alphabet"
        // (a..a — yes), but not "beta".
        let result = run_regex(fsst, "a.+a", RegexOptions::default())?;
        assert_arrays_eq!(
            &result,
            &BoolArray::from_iter([Some(true), None, Some(true), Some(false), None])
        );
        Ok(())
    }

    #[test]
    fn fsst_regex_with_nulls() -> VortexResult<()> {
        let fsst = make_fsst(
            &[Some("hello"), None, Some("help"), None, Some("goodbye")],
            Nullability::Nullable,
        );
        let result = run_regex(fsst, "^hel", RegexOptions::default())?;
        assert_arrays_eq!(
            &result,
            &BoolArray::from_iter([Some(true), None, Some(true), None, Some(false)])
        );
        Ok(())
    }
}
