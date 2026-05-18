// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use vortex_error::VortexExpect;
use vortex_error::VortexResult;

use crate::ArrayRef;
use crate::ExecutionCtx;
use crate::array::ArrayView;
use crate::array::VTable;
use crate::arrays::ScalarFn;
use crate::arrays::scalar_fn::ExactScalarFn;
use crate::arrays::scalar_fn::ScalarFnArrayExt;
use crate::arrays::scalar_fn::ScalarFnArrayView;
use crate::kernel::ExecuteParentKernel;
use crate::optimizer::rules::ArrayParentReduceRule;
use crate::scalar_fn::fns::regex::Regex as RegexExpr;
use crate::scalar_fn::fns::regex::RegexOptions;

/// Regex pattern matching on an array without reading buffers.
///
/// This trait is for regex implementations that can operate purely on array
/// metadata and structure (for example, dictionary-encoded strings where the
/// regex only needs to evaluate against the unique values). Implementations
/// should return `None` if the operation requires buffer access.
///
/// Dispatch is on child 0 (the input). The `pattern` and `options` are
/// extracted from the parent `ScalarFnArray`.
pub trait RegexReduce: VTable {
    fn regex(
        array: ArrayView<'_, Self>,
        pattern: &ArrayRef,
        options: RegexOptions,
    ) -> VortexResult<Option<ArrayRef>>;
}

/// Regex pattern matching on an array, potentially reading buffers.
///
/// Unlike [`RegexReduce`], this trait is for regex implementations that may
/// need to read and execute on the underlying buffers to produce the result.
///
/// Dispatch is on child 0 (the input). The `pattern` and `options` are
/// extracted from the parent `ScalarFnArray`.
pub trait RegexKernel: VTable {
    fn regex(
        array: ArrayView<'_, Self>,
        pattern: &ArrayRef,
        options: RegexOptions,
        ctx: &mut ExecutionCtx,
    ) -> VortexResult<Option<ArrayRef>>;
}

/// Adaptor that wraps a [`RegexReduce`] impl as an [`ArrayParentReduceRule`].
#[derive(Default, Debug)]
pub struct RegexReduceAdaptor<V>(pub V);

impl<V> ArrayParentReduceRule<V> for RegexReduceAdaptor<V>
where
    V: RegexReduce,
{
    type Parent = ExactScalarFn<RegexExpr>;

    fn reduce_parent(
        &self,
        array: ArrayView<'_, V>,
        parent: ScalarFnArrayView<'_, RegexExpr>,
        child_idx: usize,
    ) -> VortexResult<Option<ArrayRef>> {
        if child_idx != 0 {
            return Ok(None);
        }
        let scalar_fn_array = parent
            .as_opt::<ScalarFn>()
            .vortex_expect("ExactScalarFn matcher confirmed ScalarFnArray");
        let pattern = scalar_fn_array.get_child(1);
        let options = *parent.options;
        <V as RegexReduce>::regex(array, pattern, options)
    }
}

/// Adaptor that wraps a [`RegexKernel`] impl as an [`ExecuteParentKernel`].
#[derive(Default, Debug)]
pub struct RegexExecuteAdaptor<V>(pub V);

impl<V> ExecuteParentKernel<V> for RegexExecuteAdaptor<V>
where
    V: RegexKernel,
{
    type Parent = ExactScalarFn<RegexExpr>;

    fn execute_parent(
        &self,
        array: ArrayView<'_, V>,
        parent: ScalarFnArrayView<'_, RegexExpr>,
        child_idx: usize,
        ctx: &mut ExecutionCtx,
    ) -> VortexResult<Option<ArrayRef>> {
        if child_idx != 0 {
            return Ok(None);
        }
        let scalar_fn_array = parent
            .as_opt::<ScalarFn>()
            .vortex_expect("ExactScalarFn matcher confirmed ScalarFnArray");
        let pattern = scalar_fn_array.get_child(1);
        let options = *parent.options;
        <V as RegexKernel>::regex(array, pattern, options, ctx)
    }
}
