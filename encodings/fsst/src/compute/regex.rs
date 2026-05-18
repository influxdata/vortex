// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

//! Regex matching on FSST-compressed string arrays.
//!
//! There is no specialized regex kernel for FSST today: full regular-expression
//! evaluation on compressed FSST symbols would require translating an arbitrary
//! regex into a DFA over the FSST symbol table, which is well beyond what we
//! need for SQL pushdown. Instead, the default `Regex` execution path
//! canonicalizes FSST arrays to `VarBinViewArray` and runs the regex over the
//! decompressed strings.
//!
//! The tests below exercise that path so we can be confident regex predicates
//! produce correct results on FSST-encoded data.

#[cfg(test)]
mod tests {
    use std::sync::LazyLock;

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
    use vortex_array::scalar_fn::fns::regex::RegexOptions;
    use vortex_array::session::ArraySession;
    use vortex_error::VortexResult;
    use vortex_session::VortexSession;

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
