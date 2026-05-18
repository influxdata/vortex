// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

//! Regex matching on FSST-compressed string arrays.
//!
//! For regex patterns whose body is a literal (after stripping leading `^`
//! and trailing `$`), we lift the LIKE DFA infrastructure: a `^prefix`
//! pattern feeds the `FlatPrefixDfa` and an unanchored `needle` feeds the
//! `FlatContainsDfa`. Both run directly on FSST symbol codes without
//! decompressing.
//!
//! Patterns that fall outside that classification — anchored exact
//! matches, suffix anchors, character classes, escapes, alternation, or
//! anything else with metacharacters — return `None` from the kernel and
//! fall back to the default `Regex` execution path, which canonicalizes
//! to `VarBinViewArray` and runs the compiled regex over the
//! decompressed strings.

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
        // Case-insensitive matching would need a separate DFA that folds
        // both halves of every transition; defer to the canonical path.
        if options.case_insensitive {
            return Ok(None);
        }

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

        // Map the regex shape onto an existing FSST DFA. Exact and suffix
        // patterns aren't covered by the prefix/contains DFAs — the
        // prefix DFA accepts as soon as the literal matches and the
        // contains DFA accepts at any position — so we leave those to
        // the fallback canonical path along with any non-literal pattern.
        let Some(shape) = RegexLiteralShape::analyze(pattern_str.as_str()) else {
            return Ok(None);
        };
        let matcher = match shape {
            RegexLiteralShape::Prefix(prefix) => FsstMatcher::try_new_prefix_literal(
                symbols.as_slice(),
                symbol_lengths.as_slice(),
                prefix.as_bytes(),
            )?,
            RegexLiteralShape::Contains(needle) => FsstMatcher::try_new_contains_literal(
                symbols.as_slice(),
                symbol_lengths.as_slice(),
                needle.as_bytes(),
            )?,
            RegexLiteralShape::Exact(_) | RegexLiteralShape::Suffix(_) => return Ok(None),
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

    /// Patterns with regex metacharacters must fall back to canonical
    /// decompression — the FSST DFA only models literal prefix / contains.
    #[test]
    fn fsst_regex_falls_back_for_metacharacters() -> VortexResult<()> {
        let fsst = make_fsst(&[Some("abc"), Some("def")], Nullability::NonNullable);
        let mut ctx = SESSION.create_execution_ctx();

        for pattern in ["a.c", "(a|b)c", "^a$", "abc$"] {
            let view: ArrayView<'_, FSST> = fsst.as_view();
            let pat = ConstantArray::new(pattern, fsst.len()).into_array();
            let result =
                <FSST as RegexKernel>::regex(view, &pat, RegexOptions::default(), &mut ctx)?;
            assert!(
                result.is_none(),
                "pattern `{pattern}` should fall back to canonical"
            );
        }

        // Case-insensitive also falls back.
        let view: ArrayView<'_, FSST> = fsst.as_view();
        let pat = ConstantArray::new("abc", fsst.len()).into_array();
        let result = <FSST as RegexKernel>::regex(
            view,
            &pat,
            RegexOptions {
                negated: false,
                case_insensitive: true,
            },
            &mut ctx,
        )?;
        assert!(
            result.is_none(),
            "case-insensitive should fall back to canonical"
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
