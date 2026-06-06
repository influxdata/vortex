// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use std::collections::BTreeSet;
use std::ops::BitAnd;
use std::ops::Range;
use std::sync::Arc;
use std::sync::OnceLock;

use futures::FutureExt;
use futures::TryFutureExt;
use futures::future::BoxFuture;
use futures::try_join;
use vortex_array::ArrayRef;
use vortex_array::IntoArray;
use vortex_array::MaskFuture;
use vortex_array::VortexSessionExecute;
use vortex_array::arrays::DictArray;
use vortex_array::arrays::SharedArray;
use vortex_array::dtype::DType;
use vortex_array::dtype::FieldMask;
use vortex_array::dtype::Nullability;
use vortex_array::dtype::PType;
use vortex_array::expr::Expression;
use vortex_array::expr::eq;
use vortex_array::expr::lit;
use vortex_array::expr::or_collect;
use vortex_array::expr::root;
use vortex_array::optimizer::ArrayOptimizer;
use vortex_array::scalar::Scalar;
use vortex_array::scalar_fn::fns::binary::Binary;
use vortex_array::scalar_fn::fns::literal::Literal;
use vortex_array::scalar_fn::fns::operators::Operator;
use vortex_array::scalar_fn::fns::root::Root;
use vortex_error::VortexError;
use vortex_error::VortexExpect;
use vortex_error::VortexResult;
use vortex_error::vortex_bail;
use vortex_mask::AllOr;
use vortex_mask::Mask;
use vortex_session::VortexSession;
use vortex_utils::aliases::dash_map::DashMap;

use super::DictLayout;
use crate::LayoutReader;
use crate::LayoutReaderRef;
use crate::SplitRange;
use crate::layouts::SharedArrayFuture;
use crate::segments::SegmentSource;

pub struct DictReader {
    layout: DictLayout,
    name: Arc<str>,
    session: VortexSession,

    /// Length of the values array
    values_len: usize,
    /// Cached dict values array
    values_array: OnceLock<SharedArrayFuture>,
    /// Cache of expression evaluation results on the values array by expression
    values_evals: DashMap<Expression, SharedArrayFuture>,

    values: LayoutReaderRef,
    codes: LayoutReaderRef,
}

impl DictReader {
    pub(super) fn try_new(
        layout: DictLayout,
        name: Arc<str>,
        segment_source: Arc<dyn SegmentSource>,
        session: VortexSession,
    ) -> VortexResult<Self> {
        let values_len = usize::try_from(layout.values.row_count())?;
        let values = layout.values.new_reader(
            format!("{name}.values").into(),
            Arc::clone(&segment_source),
            &session,
        )?;
        let codes =
            layout
                .codes
                .new_reader(format!("{name}.codes").into(), segment_source, &session)?;

        Ok(Self {
            layout,
            name,
            session,
            values_len,
            values_array: Default::default(),
            values_evals: Default::default(),
            values,
            codes,
        })
    }

    fn values_array(&self) -> SharedArrayFuture {
        // We capture the name, so it may be wrong if we re-use the same reader within multiple
        // different parent readers. But that's rare...
        let values_len = self.values_len;
        self.values_array
            .get_or_init(move || {
                self.values
                    .projection_evaluation(
                        &(0..values_len as u64),
                        &root(),
                        MaskFuture::new_true(values_len),
                    )
                    .vortex_expect("must construct dict values array evaluation")
                    .map_err(Arc::new)
                    .map(move |array| {
                        let array = array?;
                        Ok(SharedArray::new(array).into_array())
                    })
                    .boxed()
                    .shared()
            })
            .clone()
    }

    // This is the dict values array without canonicalization, if not already canonical
    fn values_array_uncanonical(&self) -> SharedArrayFuture {
        // We capture the name, so it may be wrong if we re-use the same reader within multiple
        // different parent readers. But that's rare...
        let values_len = self.values_len;
        self.values_array.get().cloned().unwrap_or_else(|| {
            self.values
                .projection_evaluation(
                    &(0..values_len as u64),
                    &root(),
                    MaskFuture::new_true(values_len),
                )
                .vortex_expect("must construct dict values array evaluation")
                .map_err(Arc::new)
                .boxed()
                .shared()
        })
    }

    fn values_eval(&self, expr: Expression) -> SharedArrayFuture {
        // This is unsound since we cannot be sure that all the values are referenced in the query
        // after applying the filter, so if the expression is fallible this might fail when it
        // shouldn't.
        // TODO(joe): fixme

        // Check cache first with read-only lock
        if let Some(fut) = self.values_evals.get(&expr) {
            return fut.clone();
        }

        self.values_evals
            .entry(expr.clone())
            .or_insert_with(|| {
                self.values_array_uncanonical()
                    .map(move |array| {
                        let array = array?.apply(&expr)?;
                        Ok(SharedArray::new(array).into_array())
                    })
                    .boxed()
                    .shared()
            })
            .clone()
    }
}

impl LayoutReader for DictReader {
    fn name(&self) -> &Arc<str> {
        &self.name
    }

    fn dtype(&self) -> &DType {
        self.layout.dtype()
    }

    fn row_count(&self) -> u64 {
        self.layout.row_count()
    }

    fn register_splits(
        &self,
        field_mask: &[FieldMask],
        split_range: &SplitRange,
        splits: &mut BTreeSet<u64>,
    ) -> VortexResult<()> {
        self.codes.register_splits(field_mask, split_range, splits)
    }

    fn pruning_evaluation(
        &self,
        row_range: &Range<u64>,
        expr: &Expression,
        mask: Mask,
    ) -> VortexResult<MaskFuture> {
        // Translate a value-domain equality / IN predicate into the *code* domain
        // and push it down to the codes child. This is the only expression shape
        // for which dict-code pruning is both sound and selective:
        //
        //   * a value maps to a single code, so `value == lit` becomes
        //     `code == code_of(lit)` — and a zone whose stored code min/max
        //     excludes that code provably holds no matching row;
        //   * when the codes are storage-monotonic (e.g. a trailing series-key
        //     tag, whose distinct values each occupy one contiguous run), the
        //     per-zone code min/max are tight and disjoint, so a point lookup
        //     localizes to ~1 zone where the decoded-value min/max would admit
        //     every zone.
        //
        // Range predicates are excluded (codes are assigned in first-appearance
        // / storage order, not value order, so a value range is not a code
        // range), as are null-sensitive predicates (`IS NULL`): zone min/max
        // ignore nulls, so they must never drive a skip.
        if !is_prunable_dict_eq(expr) {
            return Ok(MaskFuture::ready(mask));
        }

        // The rewritten predicate must compare against the codes child's own
        // primitive type. If that can't be determined, fall through to no-op.
        let Ok(codes_ptype) = PType::try_from(self.codes.dtype()) else {
            return Ok(MaskFuture::ready(mask));
        };

        // Evaluate the predicate against the (small, cached) dictionary values to
        // find which entries — i.e. which codes — satisfy it. This reuses the
        // exact values-domain evaluation the filter path uses.
        let values_eval = self.values_eval(expr.clone());
        let codes = Arc::clone(&self.codes);
        let row_range = row_range.clone();
        let session = self.session.clone();
        let mask_len = mask.len();

        Ok(MaskFuture::new(mask_len, async move {
            let bool_over_values = values_eval.map_err(VortexError::from).await?;
            let mut ctx = session.create_execution_ctx();
            let satisfied = bool_over_values.execute::<Mask>(&mut ctx)?;

            let code_indices: Vec<usize> = match satisfied.indices() {
                AllOr::All => (0..satisfied.len()).collect(),
                AllOr::None => Vec::new(),
                AllOr::Some(idxs) => idxs.to_vec(),
            };

            // No dictionary entry matches: the literal(s) are absent from this
            // dictionary run, so no row in its range can match. Prune it all.
            if code_indices.is_empty() {
                return Ok(Mask::new_false(mask_len));
            }

            // Rewrite into the code domain: `code == c0 OR code == c1 OR ...`.
            let mut code_terms = Vec::with_capacity(code_indices.len());
            for code in code_indices {
                code_terms.push(eq(root(), lit(code_scalar(code, codes_ptype)?)));
            }
            let rewritten = or_collect(code_terms).vortex_expect("code_terms is non-empty");

            // Delegate to the codes child. When the codes are zoned (i.e. the
            // writer nested `Dict(Zoned(codes), values)`), this prunes via the
            // tight per-zone code min/max. Otherwise the codes child carries no
            // zone stats and this is a no-op beyond the membership check above.
            codes
                .pruning_evaluation(&row_range, &rewritten, mask)?
                .await
        }))
    }

    fn filter_evaluation(
        &self,
        row_range: &Range<u64>,
        expr: &Expression,
        mask: MaskFuture,
    ) -> VortexResult<MaskFuture> {
        // TODO(joe): fix up expr partitioning with fallible & null sensitive annotations
        let values_eval = self.values_eval(expr.clone());

        // We register interest on the entire codes row_range for now, there
        // is no straightforward shift into the codes domain we can do to the expression
        // without reading values.
        let codes_eval = self.codes.projection_evaluation(
            row_range,
            &root(),
            MaskFuture::new_true(mask.len()),
        )?;

        let session = self.session.clone();

        Ok(MaskFuture::new(mask.len(), async move {
            // Join on the I/O futures first, before the mask.
            let (codes, values) = try_join!(codes_eval, values_eval.map_err(VortexError::from))?;
            let mask = mask.await?;

            let mut ctx = session.create_execution_ctx();
            let dict_mask = values.take(codes)?.execute::<Mask>(&mut ctx)?;

            Ok(mask.bitand(&dict_mask))
        }))
    }

    fn projection_evaluation(
        &self,
        row_range: &Range<u64>,
        expr: &Expression,
        mask: MaskFuture,
    ) -> VortexResult<BoxFuture<'static, VortexResult<ArrayRef>>> {
        // TODO: fix up expr partitioning with fallible & null sensitive annotations
        let values_eval = self.values_array();
        let codes_eval = self
            .codes
            .projection_evaluation(row_range, &root(), mask)
            .map_err(|err| err.with_context("While evaluating projection on codes"))?;
        let expr = expr.clone();

        let all_values_referenced = self.layout.has_all_values_referenced();
        Ok(async move {
            let (values, codes) = try_join!(values_eval.map_err(VortexError::from), codes_eval)?;

            // SAFETY: Layout was validated at write time.
            //  * The codes dtype is guaranteed to be an integer type from the layout
            //  * The codes child reader ensures the correct dtype.
            //  * The layout stores `all_values_referenced` and if this is malicious then it must
            //    only affect correctness not memory safety.
            let array = unsafe {
                DictArray::new_unchecked(codes, values)
                    .set_all_values_referenced(all_values_referenced)
            }
            .into_array()
            .optimize()?;

            array.apply(&expr)
        }
        .boxed())
    }

    fn as_any(&self) -> &dyn std::any::Any {
        self
    }
}

/// Whether `expr` is a dict-root equality (`root == lit`), or an OR-tree of such
/// equalities (the `IN`-list shape), with non-null literals.
///
/// These are exactly the predicates whose value-domain membership maps cleanly
/// onto specific dictionary codes, making code-domain zone pruning sound. Range
/// and null-sensitive predicates are intentionally rejected (see
/// [`DictReader::pruning_evaluation`]).
fn is_prunable_dict_eq(expr: &Expression) -> bool {
    if !expr.is::<Binary>() {
        return false;
    }
    match *expr.as_::<Binary>() {
        Operator::Eq => {
            let lhs = expr.child(0);
            let rhs = expr.child(1);
            (lhs.is::<Root>() && is_non_null_literal(rhs))
                || (rhs.is::<Root>() && is_non_null_literal(lhs))
        }
        Operator::Or => is_prunable_dict_eq(expr.child(0)) && is_prunable_dict_eq(expr.child(1)),
        _ => false,
    }
}

/// Whether `expr` is a literal whose scalar value is non-null.
fn is_non_null_literal(expr: &Expression) -> bool {
    expr.is::<Literal>() && !expr.as_::<Literal>().is_null()
}

/// Build a code-domain literal scalar for dict code `code`, typed to match the
/// codes child's primitive type so the rewritten `code == lit` compares cleanly.
fn code_scalar(code: usize, codes_ptype: PType) -> VortexResult<Scalar> {
    let n = Nullability::NonNullable;
    Ok(match codes_ptype {
        PType::U8 => Scalar::primitive(code as u8, n),
        PType::U16 => Scalar::primitive(code as u16, n),
        PType::U32 => Scalar::primitive(code as u32, n),
        PType::U64 => Scalar::primitive(code as u64, n),
        PType::I8 => Scalar::primitive(code as i8, n),
        PType::I16 => Scalar::primitive(code as i16, n),
        PType::I32 => Scalar::primitive(code as i32, n),
        PType::I64 => Scalar::primitive(code as i64, n),
        other => vortex_bail!("unexpected dict codes ptype {other:?} for code-domain pruning"),
    })
}

#[cfg(test)]
mod tests {
    use std::sync::Arc;

    use rstest::rstest;
    use vortex_array::ArrayContext;
    use vortex_array::Canonical;
    use vortex_array::IntoArray as _;
    use vortex_array::LEGACY_SESSION;
    use vortex_array::MaskFuture;
    use vortex_array::VortexSessionExecute;
    use vortex_array::arrays::BoolArray;
    use vortex_array::arrays::StructArray;
    use vortex_array::arrays::VarBinArray;
    use vortex_array::assert_arrays_eq;
    use vortex_array::dtype::DType;
    use vortex_array::dtype::FieldName;
    use vortex_array::dtype::FieldNames;
    use vortex_array::dtype::Nullability;
    use vortex_array::expr::eq;
    use vortex_array::expr::is_not_null;
    use vortex_array::expr::lit;
    use vortex_array::expr::pack;
    use vortex_array::expr::root;
    use vortex_array::scalar_fn::session::ScalarFnSession;
    use vortex_array::session::ArraySession;
    use vortex_array::validity::Validity;
    use vortex_error::VortexExpect;
    use vortex_io::runtime::Handle;
    use vortex_io::runtime::single::block_on;
    use vortex_io::session::RuntimeSession;
    use vortex_io::session::RuntimeSessionExt;
    use vortex_session::VortexSession;

    use crate::LayoutId;
    use crate::LayoutRef;
    use crate::LayoutStrategy;
    use crate::layouts::dict::writer::DictLayoutOptions;
    use crate::layouts::dict::writer::DictStrategy;
    use crate::layouts::flat::writer::FlatLayoutStrategy;
    use crate::segments::TestSegments;
    use crate::sequence::SequenceId;
    use crate::sequence::SequentialArrayStreamExt;
    use crate::sequence::SequentialStreamAdapter;
    use crate::sequence::SequentialStreamExt;
    use crate::session::LayoutSession;

    // FIXME(ngates): Deprecate the global `runtime::single::block_on` helper and require tests
    // to call `block_on` on an explicit runtime instance.
    fn session_with_handle(handle: Handle) -> VortexSession {
        VortexSession::empty()
            .with::<ArraySession>()
            .with::<LayoutSession>()
            .with::<ScalarFnSession>()
            .with::<RuntimeSession>()
            .with_handle(handle)
    }

    /// The soundness gate for code-domain pruning: only `root == lit` and OR-trees of those
    /// (with non-null literals) may prune; range and null-sensitive predicates must not.
    #[test]
    fn is_prunable_dict_eq_gate() {
        use vortex_array::dtype::PType;
        use vortex_array::expr::get_item;
        use vortex_array::expr::is_null;
        use vortex_array::expr::lt;
        use vortex_array::expr::or;
        use vortex_array::scalar::Scalar;

        // `root == lit` (either operand order) is prunable.
        assert!(super::is_prunable_dict_eq(&eq(root(), lit(1i32))));
        assert!(super::is_prunable_dict_eq(&eq(lit(1i32), root())));
        // An OR-tree of root-equalities (the IN-list shape) is prunable.
        assert!(super::is_prunable_dict_eq(&or(
            eq(root(), lit(1i32)),
            or(eq(root(), lit(2i32)), eq(root(), lit(3i32))),
        )));

        // IS NULL must never drive a skip (min/max ignore nulls).
        assert!(!super::is_prunable_dict_eq(&is_null(root())));
        // Range predicates: dict codes are storage- not value-ordered.
        assert!(!super::is_prunable_dict_eq(&lt(root(), lit(1i32))));
        // Equality whose column side isn't the dict root (un-peeled `get_item`).
        assert!(!super::is_prunable_dict_eq(&eq(get_item("a", root()), lit(1i32))));
        // A null literal is excluded.
        let null_lit = lit(Scalar::null(DType::Primitive(PType::I32, Nullability::Nullable)));
        assert!(!super::is_prunable_dict_eq(&eq(root(), null_lit)));
        // An OR with a non-equality branch is not (wholly) prunable.
        assert!(!super::is_prunable_dict_eq(&or(
            eq(root(), lit(1i32)),
            lt(root(), lit(2i32))
        )));
    }

    #[test]
    fn reading_nested_packs_works() {
        block_on(|handle| async move {
            let session = session_with_handle(handle);
            let strategy = DictStrategy::new(
                FlatLayoutStrategy::default(),
                FlatLayoutStrategy::default(),
                FlatLayoutStrategy::default(),
                DictLayoutOptions::default(),
            );

            let array = VarBinArray::from_iter(
                [
                    Some("abc"),
                    Some("def"),
                    None,
                    Some("abc"),
                    Some("def"),
                    None,
                    Some("abc"),
                    Some("def"),
                    None,
                ],
                DType::Utf8(Nullability::Nullable),
            )
            .into_array();
            let array_to_write = array.clone();
            let ctx = ArrayContext::empty();
            let segments = Arc::new(TestSegments::default());
            let (ptr, eof) = SequenceId::root().split();
            let layout: LayoutRef = strategy
                .write_stream(
                    ctx,
                    Arc::<TestSegments>::clone(&segments),
                    SequentialStreamAdapter::new(
                        DType::Utf8(Nullability::Nullable),
                        array_to_write.to_array_stream().sequenced(ptr),
                    )
                    .sendable(),
                    eof,
                    &session,
                )
                .await
                .unwrap();

            let expression = pack(
                [(
                    "top",
                    pack([("one", root()), ("two", root())], Nullability::NonNullable),
                )],
                Nullability::NonNullable,
            );
            assert!(layout.encoding_id() == LayoutId::new("vortex.dict"));
            let actual = layout
                .new_reader("".into(), segments, &session)
                .unwrap()
                .projection_evaluation(
                    &(0..layout.row_count()),
                    &expression,
                    MaskFuture::new_true(layout.row_count().try_into().unwrap()),
                )
                .unwrap()
                .await
                .unwrap();
            let expected = StructArray::try_new(
                FieldNames::from([FieldName::from("top")]),
                vec![
                    StructArray::try_new(
                        FieldNames::from([FieldName::from("one"), FieldName::from("two")]),
                        vec![array.clone(), array],
                        9,
                        Validity::NonNullable,
                    )
                    .unwrap()
                    .into_array(),
                ],
                9,
                Validity::NonNullable,
            )
            .unwrap()
            .into_array();
            assert_arrays_eq!(actual, expected);
        })
    }

    #[rstest]
    #[case::all_true_case(
        vec![Some(""), None, Some("")], // Dict values: [""]
        "", // Filter for empty string
        vec![true, false, true], // Expected: nulls excluded, all dict values match
    )]
    #[case::all_false_case(
        vec![Some("x"), None, Some("x")], // Dict values: ["x"]
        "", // Filter for empty string
        vec![false, false, false], // Expected: all false, no dict values match
    )]
    fn shortpathes_filtering(
        #[case] data: Vec<Option<&str>>,
        #[case] filter_value: &str,
        #[case] expected: Vec<bool>,
    ) {
        block_on(|handle| async move {
            let session = session_with_handle(handle);
            let strategy = DictStrategy::new(
                FlatLayoutStrategy::default(),
                FlatLayoutStrategy::default(),
                FlatLayoutStrategy::default(),
                DictLayoutOptions::default(),
            );

            let array =
                VarBinArray::from_iter(data, DType::Utf8(Nullability::Nullable)).into_array();
            let ctx = ArrayContext::empty();
            let segments = Arc::new(TestSegments::default());
            let (ptr, eof) = SequenceId::root().split();
            let layout: LayoutRef = strategy
                .write_stream(
                    ctx,
                    Arc::<TestSegments>::clone(&segments),
                    SequentialStreamAdapter::new(
                        DType::Utf8(Nullability::Nullable),
                        array.to_array_stream().sequenced(ptr),
                    )
                    .sendable(),
                    eof,
                    &session,
                )
                .await
                .unwrap();

            let filter = eq(
                root(),
                lit(vortex_array::scalar::Scalar::utf8(
                    filter_value,
                    Nullability::Nullable,
                )),
            );
            let mask = layout
                .new_reader("".into(), segments, &session)
                .unwrap()
                .filter_evaluation(&(0..3), &filter, MaskFuture::new_true(3))
                .unwrap()
                .await
                .unwrap();

            assert_arrays_eq!(mask.into_array(), BoolArray::from_iter(expected));
        })
    }

    #[test]
    fn reading_is_null_works() {
        block_on(|handle| async move {
            let mut ctx_exec = LEGACY_SESSION.create_execution_ctx();
            let session = session_with_handle(handle);
            let strategy = DictStrategy::new(
                FlatLayoutStrategy::default(),
                FlatLayoutStrategy::default(),
                FlatLayoutStrategy::default(),
                DictLayoutOptions::default(),
            );

            let array = VarBinArray::from_iter(
                [
                    Some("abc"),
                    Some("def"),
                    None,
                    Some("abc"),
                    Some("def"),
                    None,
                    Some("abc"),
                    Some("def"),
                    None,
                ],
                DType::Utf8(Nullability::Nullable),
            )
            .into_array();
            let array_to_write = array.clone();
            let ctx = ArrayContext::empty();

            let segments = Arc::new(TestSegments::default());
            let (ptr, eof) = SequenceId::root().split();
            let layout: LayoutRef = strategy
                .write_stream(
                    ctx,
                    Arc::<TestSegments>::clone(&segments),
                    SequentialStreamAdapter::new(
                        DType::Utf8(Nullability::Nullable),
                        array_to_write.to_array_stream().sequenced(ptr),
                    )
                    .sendable(),
                    eof,
                    &session,
                )
                .await
                .unwrap();

            let expression = is_not_null(root());
            assert_eq!(layout.encoding_id(), LayoutId::new("vortex.dict"));
            let actual = layout
                .new_reader("".into(), segments, &session)
                .unwrap()
                .projection_evaluation(
                    &(0..layout.row_count()),
                    &expression,
                    MaskFuture::new_true(layout.row_count().try_into().unwrap()),
                )
                .unwrap()
                .await
                .unwrap();
            let expected = array
                .validity()
                .unwrap()
                .execute_mask(array.len(), &mut ctx_exec)
                .unwrap()
                .into_array();
            let actual_canonical = actual
                .execute::<Canonical>(&mut ctx_exec)
                .vortex_expect("to_canonical failed")
                .into_array();
            assert_arrays_eq!(actual_canonical, expected);
        })
    }
}
