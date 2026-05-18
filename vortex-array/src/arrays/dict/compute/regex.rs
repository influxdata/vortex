// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use vortex_error::VortexResult;

use super::Dict;
use super::DictArray;
use crate::ArrayRef;
use crate::IntoArray;
use crate::array::ArrayView;
use crate::arrays::ConstantArray;
use crate::arrays::dict::DictArrayExt;
use crate::arrays::dict::DictArraySlotsExt;
use crate::arrays::scalar_fn::ScalarFnFactoryExt;
use crate::optimizer::ArrayOptimizer;
use crate::scalar_fn::fns::regex::Regex;
use crate::scalar_fn::fns::regex::RegexOptions;
use crate::scalar_fn::fns::regex::RegexReduce;

impl RegexReduce for Dict {
    fn regex(
        array: ArrayView<'_, Dict>,
        pattern: &ArrayRef,
        options: RegexOptions,
    ) -> VortexResult<Option<ArrayRef>> {
        // If we have more values than codes, it is faster to canonicalize first.
        if array.values().len() > array.codes().len() {
            return Ok(None);
        }
        if let Some(pattern) = pattern.as_constant() {
            let pattern = ConstantArray::new(pattern, array.values().len()).into_array();

            let values = Regex
                .try_new_array(pattern.len(), options, [array.values().clone(), pattern])?
                .optimize()?;

            // SAFETY: REGEX preserves the length of the values, so codes still index into
            // valid positions. Preserve all_values_referenced since codes are unchanged.
            unsafe {
                Ok(Some(
                    DictArray::new_unchecked(array.codes().clone(), values)
                        .set_all_values_referenced(array.has_all_values_referenced())
                        .into_array(),
                ))
            }
        } else {
            Ok(None)
        }
    }
}

#[cfg(test)]
mod tests {
    use vortex_buffer::buffer;
    use vortex_error::VortexResult;

    use crate::IntoArray;
    use crate::arrays::BoolArray;
    use crate::arrays::ConstantArray;
    use crate::arrays::DictArray;
    use crate::arrays::VarBinArray;
    use crate::arrays::scalar_fn::ScalarFnFactoryExt;
    use crate::assert_arrays_eq;
    use crate::optimizer::ArrayOptimizer;
    use crate::scalar_fn::fns::regex::Regex;
    use crate::scalar_fn::fns::regex::RegexOptions;

    #[test]
    fn regex_reduce_dict_match() -> VortexResult<()> {
        let dict = DictArray::try_new(
            buffer![0u8, 1, 0, 2].into_array(),
            VarBinArray::from(vec!["hello", "world", "help"]).into_array(),
        )?
        .into_array();

        let pattern = ConstantArray::new("^hel", 4).into_array();
        let result = Regex
            .try_new_array(4, RegexOptions::default(), [dict, pattern])?
            .optimize()?;

        assert_arrays_eq!(result, BoolArray::from_iter([true, false, true, true]));
        Ok(())
    }

    #[test]
    fn regex_reduce_dict_case_insensitive() -> VortexResult<()> {
        let dict = DictArray::try_new(
            buffer![0u8, 1, 0, 2].into_array(),
            VarBinArray::from(vec!["Hello", "WORLD", "help"]).into_array(),
        )?
        .into_array();

        let pattern = ConstantArray::new("^hel", 4).into_array();
        let opts = RegexOptions {
            negated: false,
            case_insensitive: true,
        };
        let result = Regex.try_new_array(4, opts, [dict, pattern])?.optimize()?;

        assert_arrays_eq!(result, BoolArray::from_iter([true, false, true, true]));
        Ok(())
    }

    #[test]
    fn regex_reduce_dict_negated() -> VortexResult<()> {
        let dict = DictArray::try_new(
            buffer![0u8, 1, 0, 2].into_array(),
            VarBinArray::from(vec!["hello", "world", "help"]).into_array(),
        )?
        .into_array();

        let pattern = ConstantArray::new("^hel", 4).into_array();
        let opts = RegexOptions {
            negated: true,
            case_insensitive: false,
        };
        let result = Regex.try_new_array(4, opts, [dict, pattern])?.optimize()?;

        assert_arrays_eq!(result, BoolArray::from_iter([false, true, false, false]));
        Ok(())
    }
}
