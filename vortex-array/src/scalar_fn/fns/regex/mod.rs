// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

mod kernel;

use std::fmt::Display;
use std::fmt::Formatter;

pub use kernel::*;
use prost::Message;
use regex::RegexBuilder;
use vortex_buffer::BitBuffer;
use vortex_error::VortexResult;
use vortex_error::vortex_bail;
use vortex_error::vortex_err;
use vortex_proto::expr as pb;
use vortex_session::VortexSession;

use crate::ArrayRef;
use crate::ExecutionCtx;
use crate::IntoArray;
use crate::arrays::BoolArray;
use crate::arrays::VarBinViewArray;
use crate::arrays::varbinview::VarBinViewArrayExt;
use crate::dtype::DType;
use crate::expr::Expression;
use crate::expr::StatsCatalog;
use crate::expr::and;
use crate::expr::gt;
use crate::expr::gt_eq;
use crate::expr::lit;
use crate::expr::lt;
use crate::expr::or;
use crate::scalar::StringLike;
use crate::scalar_fn::Arity;
use crate::scalar_fn::ChildName;
use crate::scalar_fn::ExecutionArgs;
use crate::scalar_fn::ScalarFnId;
use crate::scalar_fn::ScalarFnVTable;
use crate::scalar_fn::fns::literal::Literal;
use crate::validity::Validity;

/// Options for the [`Regex`] scalar function.
#[derive(Default, Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct RegexOptions {
    /// If true, the result is inverted (i.e. matches return false, non-matches return true).
    pub negated: bool,
    /// If true, the regex is matched case-insensitively.
    pub case_insensitive: bool,
}

impl Display for RegexOptions {
    fn fmt(&self, f: &mut Formatter<'_>) -> std::fmt::Result {
        if self.negated {
            write!(f, "NOT ")?;
        }
        if self.case_insensitive {
            write!(f, "REGEXP_IMATCH")
        } else {
            write!(f, "REGEXP_MATCH")
        }
    }
}

/// Expression that matches each row of a string input against a regular expression.
///
/// The pattern uses the syntax accepted by the [`regex`] crate and is unanchored:
/// a match anywhere in the input is considered a match. To anchor the pattern,
/// callers must include `^` and/or `$` themselves.
#[derive(Clone)]
pub struct Regex;

impl ScalarFnVTable for Regex {
    type Options = RegexOptions;

    fn id(&self) -> ScalarFnId {
        ScalarFnId::new("vortex.regex")
    }

    fn serialize(&self, instance: &Self::Options) -> VortexResult<Option<Vec<u8>>> {
        Ok(Some(
            pb::RegexOpts {
                negated: instance.negated,
                case_insensitive: instance.case_insensitive,
            }
            .encode_to_vec(),
        ))
    }

    fn deserialize(
        &self,
        metadata: &[u8],
        _session: &VortexSession,
    ) -> VortexResult<Self::Options> {
        let opts = pb::RegexOpts::decode(metadata)?;
        Ok(RegexOptions {
            negated: opts.negated,
            case_insensitive: opts.case_insensitive,
        })
    }

    fn arity(&self, _options: &Self::Options) -> Arity {
        Arity::Exact(2)
    }

    fn child_name(&self, _instance: &Self::Options, child_idx: usize) -> ChildName {
        match child_idx {
            0 => ChildName::from("child"),
            1 => ChildName::from("pattern"),
            _ => unreachable!("Invalid child index {} for Regex expression", child_idx),
        }
    }

    fn fmt_sql(
        &self,
        options: &Self::Options,
        expr: &Expression,
        f: &mut Formatter<'_>,
    ) -> std::fmt::Result {
        expr.child(0).fmt_sql(f)?;
        if options.negated {
            write!(f, " !~")?;
        } else {
            write!(f, " ~")?;
        }
        if options.case_insensitive {
            write!(f, "* ")?;
        } else {
            write!(f, " ")?;
        }
        expr.child(1).fmt_sql(f)
    }

    fn return_dtype(&self, _options: &Self::Options, arg_dtypes: &[DType]) -> VortexResult<DType> {
        let input = &arg_dtypes[0];
        let pattern = &arg_dtypes[1];

        if !input.is_utf8() {
            vortex_bail!("REGEX expression requires UTF8 input dtype, got {}", input);
        }
        if !pattern.is_utf8() {
            vortex_bail!(
                "REGEX expression requires UTF8 pattern dtype, got {}",
                pattern
            );
        }

        Ok(DType::Bool(
            (input.is_nullable() || pattern.is_nullable()).into(),
        ))
    }

    fn execute(
        &self,
        options: &Self::Options,
        args: &dyn ExecutionArgs,
        ctx: &mut ExecutionCtx,
    ) -> VortexResult<ArrayRef> {
        let child = args.get(0)?;
        let pattern = args.get(1)?;
        regex_execute(&child, &pattern, *options, ctx)
    }

    fn validity(
        &self,
        _options: &Self::Options,
        expression: &Expression,
    ) -> VortexResult<Option<Expression>> {
        let child_validity = expression.child(0).validity()?;
        let pattern_validity = expression.child(1).validity()?;
        Ok(Some(and(child_validity, pattern_validity)))
    }

    fn is_null_sensitive(&self, _instance: &Self::Options) -> bool {
        false
    }

    fn stat_falsification(
        &self,
        options: &Self::Options,
        expr: &Expression,
        catalog: &dyn StatsCatalog,
    ) -> Option<Expression> {
        // Attempt min/max pruning for `^literal$` (exact) and `^literal`
        // (prefix). Case-insensitive and negated forms aren't safely
        // representable as lexicographic bounds, so we leave them alone.
        if options.negated || options.case_insensitive {
            return None;
        }

        let pat = expr.child(1).as_::<Literal>();
        let pat_str = pat.as_utf8().value()?;
        let src = expr.child(0).clone();
        let src_min = src.stat_min(catalog)?;
        let src_max = src.stat_max(catalog)?;

        match RegexLiteralShape::analyze(pat_str.as_str())? {
            RegexLiteralShape::Exact(text) => {
                // col ~ '^exact$' ⟹ col.min > 'exact' || col.max < 'exact'
                Some(or(gt(src_min, lit(text)), lt(src_max, lit(text))))
            }
            RegexLiteralShape::Prefix(prefix) => {
                // col ~ '^prefix' ⟹ col.min >= succ(prefix) || col.max < prefix
                let succ = prefix.to_string().increment().ok()?;
                Some(or(gt_eq(src_min, lit(succ)), lt(src_max, lit(prefix))))
            }
            // Suffix/contains patterns can't be pruned by min/max bounds.
            RegexLiteralShape::Suffix(_) | RegexLiteralShape::Contains(_) => None,
        }
    }
}

/// Classification of a regex pattern that contains only literal bytes
/// (no metacharacters or escapes).
///
/// The classifier strips an optional leading `^` and trailing `$` and
/// inspects the remaining body for any regex metacharacter. Patterns that
/// use escapes (e.g. `\d`, `\.`) are conservatively rejected — we'd need
/// a full regex parser to know whether each backslash makes the
/// surrounding bytes literal or special.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum RegexLiteralShape<'a> {
    /// `^literal$` — match the input exactly.
    Exact(&'a str),
    /// `^literal` — match strings that start with `literal`.
    Prefix(&'a str),
    /// `literal$` — match strings that end with `literal`.
    Suffix(&'a str),
    /// `literal` (no anchors) — match strings that contain `literal`.
    Contains(&'a str),
}

impl<'a> RegexLiteralShape<'a> {
    /// Attempt to classify `pattern` as one of the literal shapes.
    ///
    /// Returns `None` if the pattern contains any regex metacharacter or
    /// backslash in its body, since those can change the semantics of
    /// otherwise-literal characters.
    pub fn analyze(pattern: &'a str) -> Option<Self> {
        let (start_anchor, after_start) = match pattern.strip_prefix('^') {
            Some(rest) => (true, rest),
            None => (false, pattern),
        };
        let (end_anchor, body) = match after_start.strip_suffix('$') {
            Some(rest) => (true, rest),
            None => (false, after_start),
        };
        if !is_literal_body(body.as_bytes()) {
            return None;
        }
        Some(match (start_anchor, end_anchor) {
            (true, true) => Self::Exact(body),
            (true, false) => Self::Prefix(body),
            (false, true) => Self::Suffix(body),
            (false, false) => Self::Contains(body),
        })
    }

    /// The literal substring that the pattern is built around.
    pub fn literal(&self) -> &'a str {
        match self {
            Self::Exact(s) | Self::Prefix(s) | Self::Suffix(s) | Self::Contains(s) => s,
        }
    }
}

/// Returns true iff every byte in `body` is a literal regex character.
///
/// Rejects all regex metacharacters and backslash escapes. Bytes outside
/// ASCII are accepted as literals — non-ASCII regex syntax only enters
/// via `\u{...}` escapes, which the backslash check already catches.
fn is_literal_body(body: &[u8]) -> bool {
    !body.iter().any(|b| {
        matches!(
            b,
            b'.' | b'*'
                | b'+'
                | b'?'
                | b'('
                | b')'
                | b'['
                | b']'
                | b'{'
                | b'}'
                | b'|'
                | b'\\'
                | b'^'
                | b'$'
        )
    })
}

/// Default execution path for the [`Regex`] scalar function.
///
/// Canonicalizes the input array to a [`VarBinViewArray`], compiles the pattern
/// once, then walks the views producing a [`BoolArray`]. Validity is the
/// conjunction of the input validity and the pattern validity (a null pattern
/// produces an all-null result).
pub(crate) fn regex_execute(
    array: &ArrayRef,
    pattern: &ArrayRef,
    options: RegexOptions,
    ctx: &mut ExecutionCtx,
) -> VortexResult<ArrayRef> {
    if !array.dtype().is_utf8() {
        vortex_bail!(
            "regex_execute requires a UTF8 input array, got {}",
            array.dtype()
        );
    }
    if !pattern.dtype().is_utf8() {
        vortex_bail!(
            "regex_execute requires a UTF8 pattern, got {}",
            pattern.dtype()
        );
    }
    assert_eq!(
        array.len(),
        pattern.len(),
        "Regex execute: length mismatch for {}",
        array.encoding_id()
    );

    let len = array.len();
    let result_nullability = (array.dtype().is_nullable() || pattern.dtype().is_nullable()).into();

    let canonical = array.clone().execute::<VarBinViewArray>(ctx)?;
    let canonical = canonical.into_array();
    let canonical_view = canonical
        .as_opt::<crate::arrays::VarBinView>()
        .ok_or_else(|| vortex_err!("expected VarBinView after canonical execute"))?;
    let input_validity = canonical_view.varbinview_validity();

    let Some(pattern_scalar) = pattern.as_constant() else {
        // Per-row patterns are not supported. Compile a single regex per row
        // would be O(n) compiles which defeats pushdown; treat as an error so
        // callers know to provide a constant pattern.
        vortex_bail!("REGEX pattern must be a constant expression");
    };

    let pattern_value = pattern_scalar.as_utf8().value();
    let Some(pattern_str) = pattern_value else {
        // Pattern is NULL: result is all-null with the right validity shape.
        let bits = BitBuffer::new_unset(len);
        let validity = Validity::AllInvalid;
        return Ok(BoolArray::new(bits, validity).into_array());
    };

    let compiled = RegexBuilder::new(pattern_str.as_str())
        .case_insensitive(options.case_insensitive)
        .build()
        .map_err(|e| {
            vortex_err!(
                "Failed to compile regex pattern '{}': {}",
                pattern_str.as_str(),
                e
            )
        })?;

    let bits = BitBuffer::collect_bool(len, |i| {
        let bytes = canonical_view.bytes_at(i);
        let m = match std::str::from_utf8(bytes.as_slice()) {
            Ok(s) => compiled.is_match(s),
            Err(_) => false,
        };
        m ^ options.negated
    });

    let validity = input_validity.union_nullability(result_nullability);
    Ok(BoolArray::new(bits, validity).into_array())
}

#[cfg(test)]
mod tests {
    use vortex_buffer::BitBuffer;
    use vortex_session::VortexSession;

    use super::*;
    use crate::IntoArray;
    use crate::VortexSessionExecute;
    use crate::arrays::BoolArray;
    use crate::arrays::ConstantArray;
    use crate::arrays::VarBinArray;
    use crate::arrays::VarBinViewArray;
    use crate::arrays::scalar_fn::ScalarFnFactoryExt;
    use crate::assert_arrays_eq;
    use crate::dtype::Nullability;
    use crate::expr::iregex;
    use crate::expr::lit;
    use crate::expr::not_iregex;
    use crate::expr::not_regex;
    use crate::expr::regex as regex_expr;
    use crate::expr::root;
    use crate::optimizer::ArrayOptimizer;
    use crate::session::ArraySession;

    fn session() -> VortexSession {
        VortexSession::empty().with::<ArraySession>()
    }

    fn run_regex(array: ArrayRef, pattern: &str, opts: RegexOptions) -> VortexResult<BoolArray> {
        let len = array.len();
        let pattern_arr = ConstantArray::new(pattern, len).into_array();
        let session = session();
        let mut ctx = session.create_execution_ctx();
        let result = Regex
            .try_new_array(len, opts, [array, pattern_arr])?
            .optimize()?
            .execute::<crate::Canonical>(&mut ctx)?;
        Ok(result.into_bool())
    }

    #[test]
    fn dtype() {
        let dtype = DType::Utf8(Nullability::NonNullable);
        let expr = regex_expr(root(), lit("^foo"));
        assert_eq!(
            expr.return_dtype(&dtype).unwrap(),
            DType::Bool(Nullability::NonNullable)
        );
    }

    #[test]
    fn proto_roundtrip() -> VortexResult<()> {
        for opts in [
            RegexOptions {
                negated: false,
                case_insensitive: false,
            },
            RegexOptions {
                negated: true,
                case_insensitive: false,
            },
            RegexOptions {
                negated: false,
                case_insensitive: true,
            },
            RegexOptions {
                negated: true,
                case_insensitive: true,
            },
        ] {
            let serialized = Regex.serialize(&opts)?.expect("regex always serializable");
            let deserialized = Regex.deserialize(&serialized, &session())?;
            assert_eq!(deserialized, opts);
        }
        Ok(())
    }

    #[test]
    fn varbin_basic_match() -> VortexResult<()> {
        let arr = VarBinArray::from(vec!["hello world", "goodbye", "say hello"]).into_array();
        let opts = RegexOptions::default();
        let result = run_regex(arr, "^hello", opts)?;
        assert_arrays_eq!(&result, &BoolArray::from_iter([true, false, false]));
        Ok(())
    }

    #[test]
    fn varbinview_contains_unanchored() -> VortexResult<()> {
        let arr = VarBinViewArray::from_iter_str(["alphabet", "beta", "gamma"]).into_array();
        let opts = RegexOptions::default();
        let result = run_regex(arr, "et", opts)?;
        assert_arrays_eq!(&result, &BoolArray::from_iter([true, true, false]));
        Ok(())
    }

    #[test]
    fn case_insensitive() -> VortexResult<()> {
        let arr = VarBinArray::from(vec!["HELLO", "Hello", "world"]).into_array();
        let opts = RegexOptions {
            negated: false,
            case_insensitive: true,
        };
        let result = run_regex(arr, "^hello$", opts)?;
        assert_arrays_eq!(&result, &BoolArray::from_iter([true, true, false]));
        Ok(())
    }

    #[test]
    fn negated() -> VortexResult<()> {
        let arr = VarBinArray::from(vec!["abc", "xyz", "abz"]).into_array();
        let opts = RegexOptions {
            negated: true,
            case_insensitive: false,
        };
        let result = run_regex(arr, "^ab", opts)?;
        assert_arrays_eq!(&result, &BoolArray::from_iter([false, true, false]));
        Ok(())
    }

    #[test]
    fn negated_case_insensitive() -> VortexResult<()> {
        // The combined case is the easiest to flip a sign on accidentally.
        let arr = VarBinArray::from(vec!["AbC", "xyz", "abz"]).into_array();
        let opts = RegexOptions {
            negated: true,
            case_insensitive: true,
        };
        let result = run_regex(arr, "^ab", opts)?;
        assert_arrays_eq!(&result, &BoolArray::from_iter([false, true, false]));
        Ok(())
    }

    #[test]
    fn nullable_input_propagates() -> VortexResult<()> {
        let arr =
            VarBinViewArray::from_iter_nullable_str([Some("foo"), None, Some("bar")]).into_array();
        let result = run_regex(arr, "^fo", RegexOptions::default())?;
        assert_arrays_eq!(
            &result,
            &BoolArray::from_iter([Some(true), None, Some(false)])
        );
        Ok(())
    }

    #[test]
    fn invalid_pattern_errors() {
        let arr = VarBinArray::from(vec!["abc"]).into_array();
        let err = run_regex(arr, "(", RegexOptions::default()).unwrap_err();
        let msg = err.to_string();
        assert!(msg.contains("regex"), "unexpected error: {msg}");
    }

    #[test]
    fn literal_shape_analyzer() {
        use RegexLiteralShape::*;
        assert_eq!(RegexLiteralShape::analyze("^foo$"), Some(Exact("foo")));
        assert_eq!(RegexLiteralShape::analyze("^foo"), Some(Prefix("foo")));
        assert_eq!(RegexLiteralShape::analyze("foo$"), Some(Suffix("foo")));
        assert_eq!(RegexLiteralShape::analyze("foo"), Some(Contains("foo")));
        // Empty literal forms are still literal.
        assert_eq!(RegexLiteralShape::analyze("^$"), Some(Exact("")));
        assert_eq!(RegexLiteralShape::analyze("^"), Some(Prefix("")));
        assert_eq!(RegexLiteralShape::analyze("$"), Some(Suffix("")));
        assert_eq!(RegexLiteralShape::analyze(""), Some(Contains("")));

        // Metacharacters in the body disqualify the pattern.
        assert!(RegexLiteralShape::analyze("^a.b").is_none());
        assert!(RegexLiteralShape::analyze("^a*").is_none());
        assert!(RegexLiteralShape::analyze("^(a|b)$").is_none());
        // Escapes are conservatively rejected: \. is a literal '.', but
        // distinguishing that from \d requires parsing the regex.
        assert!(RegexLiteralShape::analyze("^a\\.b").is_none());
        // Anchors inside the body (after the leading/trailing strip) are
        // not allowed either.
        assert!(RegexLiteralShape::analyze("^a^b").is_none());
        assert!(RegexLiteralShape::analyze("a$b$").is_none());
    }

    #[test]
    fn regex_stat_falsification() {
        use crate::expr::col;
        use crate::expr::pruning::pruning_expr::TrackingStatsCatalog;

        let catalog = TrackingStatsCatalog::default();

        // Anchored prefix produces the same shape as LIKE 'prefix%' pruning.
        let pruning = regex_expr(col("a"), lit("^prefix"))
            .stat_falsification(&catalog)
            .expect("anchored prefix should prune");
        insta::assert_snapshot!(pruning, @r#"(($.a_min >= "prefiy") or ($.a_max < "prefix"))"#);

        // Anchored exact mirrors LIKE exact pruning.
        let pruning = regex_expr(col("a"), lit("^exactly$"))
            .stat_falsification(&catalog)
            .expect("anchored exact should prune");
        insta::assert_snapshot!(pruning, @r#"(($.a_min > "exactly") or ($.a_max < "exactly"))"#);

        // Contains and suffix patterns have no min/max pruning.
        assert!(
            regex_expr(col("a"), lit("suffix$"))
                .stat_falsification(&catalog)
                .is_none()
        );
        assert!(
            regex_expr(col("a"), lit("contains"))
                .stat_falsification(&catalog)
                .is_none()
        );

        // Negated and case-insensitive variants don't prune even on
        // otherwise-literal patterns.
        assert!(
            not_regex(col("a"), lit("^prefix"))
                .stat_falsification(&catalog)
                .is_none()
        );
        assert!(
            iregex(col("a"), lit("^prefix"))
                .stat_falsification(&catalog)
                .is_none()
        );

        // Non-literal regex doesn't prune.
        assert!(
            regex_expr(col("a"), lit("^a.b"))
                .stat_falsification(&catalog)
                .is_none()
        );
    }

    #[test]
    fn display_includes_pattern() {
        let e1 = regex_expr(root(), lit("^a"));
        assert_eq!(e1.to_string(), "$ ~ \"^a\"");
        let e2 = iregex(root(), lit("b"));
        assert_eq!(e2.to_string(), "$ ~* \"b\"");
        let e3 = not_regex(root(), lit("c"));
        assert_eq!(e3.to_string(), "$ !~ \"c\"");
        let e4 = not_iregex(root(), lit("d"));
        assert_eq!(e4.to_string(), "$ !~* \"d\"");
    }

    // Silence unused warnings on BitBuffer import in tests-only context.
    #[allow(dead_code)]
    fn _ensure_bitbuffer_used() -> BitBuffer {
        BitBuffer::new_unset(0)
    }
}
