# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0


FieldId = nvml.FieldId


cdef class FieldValue:
    """
    Represents the data from a single field value.

    Use :meth:`Device.get_field_values` to get multiple field values at once.
    """
    cdef object _field_value

    def __init__(self, field_value: nvml.FieldValue):
        assert len(field_value) == 1
        self._field_value = field_value

    @property
    def field_id(self) -> FieldId:
        """
        The field ID.
        """
        return FieldId(self._field_value.field_id)

    @property
    def scope_id(self) -> int:
        """
        The scope ID.
        """
        # Explicit int() cast required because this is a Numpy type
        return int(self._field_value.scope_id)

    @property
    def timestamp(self) -> int:
        """
        The CPU timestamp (in microseconds since 1970) at which the value was
        sampled.
        """
        # Explicit int() cast required because this is a Numpy type
        return int(self._field_value.timestamp)

    @property
    def latency_usec(self) -> int:
        """
        How long this field value took to update (in usec) within NVML. This may
        be averaged across several fields that are serviced by the same driver
        call.
        """
        # Explicit int() cast required because this is a Numpy type
        return int(self._field_value.latency_usec)

    @property
    def value(self) -> int | float:
        """
        The field value.

        Raises
        ------
        :class:`cuda.core.system.NvmlError`
            If there was an error retrieving the field value.
        """
        nvml.check_status(self._field_value.nvml_return)

        cdef int value_type = self._field_value.value_type
        value = self._field_value.value

        ValueType = nvml.ValueType

        if value_type == ValueType.DOUBLE:
            return float(value.d_val[0])
        elif value_type == ValueType.UNSIGNED_INT:
            return int(value.ui_val[0])
        elif value_type == ValueType.UNSIGNED_LONG:
            return int(value.ul_val[0])
        elif value_type == ValueType.UNSIGNED_LONG_LONG:
            return int(value.ull_val[0])
        elif value_type == ValueType.SIGNED_LONG_LONG:
            return int(value.ll_val[0])
        elif value_type == ValueType.SIGNED_INT:
            return int(value.si_val[0])
        elif value_type == ValueType.UNSIGNED_SHORT:
            return int(value.us_val[0])
        else:
            raise AssertionError("Unexpected value type")


cdef class FieldValues:
    """
    Container of multiple field values.
    """
    cdef object _field_values

    def __init__(self, field_values: nvml.FieldValue):
        self._field_values = field_values

    def __getitem__(self, idx: int) -> FieldValue:
        return FieldValue(self._field_values[idx])

    def __len__(self) -> int:
        return len(self._field_values)

    def validate(self) -> None:
        """
        Validate that there are no issues in any of the contained field values.

        Raises an exception for the first issue found, if any.

        Raises
        ------
        :class:`cuda.core.system.NvmlError`
            If any of the contained field values has an associated exception.
        """
        # TODO: This is a classic use case for an `ExceptionGroup`, but those
        # are only available in Python 3.11+.
        return_values = self._field_values.nvml_return
        if len(self._field_values) == 1:
            return_values = [return_values]
        for return_value in return_values:
            nvml.check_status(return_value)

    def get_all_values(self) -> list[int | float]:
        """
        Get all field values as a list.

        This will validate each of the values and include just the core value in
        the list.

        Returns
        -------
        list[int | float]
            List of all field values.

        Raises
        ------
        :class:`cuda.core.system.NvmlError`
            If any of the contained field values has an associated exception.
        """
        return [x.value for x in self]
