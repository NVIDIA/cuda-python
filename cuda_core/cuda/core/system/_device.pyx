# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

from libc.stdint cimport intptr_t
from libc.math cimport ceil

from multiprocessing import cpu_count
from typing import Iterable

from cuda.bindings import _nvml as nvml

from ._nvml_context cimport initialize

include "_device_utils.pxi"


FieldId = nvml.FieldId


class DeviceArchitecture:
    """
    Device architecture enumeration.
    """

    def __init__(self, architecture: int):
        try:
            self._architecture = nvml.DeviceArch(architecture)
        except ValueError:
            self._architecture = None

    @property
    def id(self) -> int:
        """
        The numeric id of the device architecture.

        Returns -1 if the device is unknown.
        """
        if self._architecture is None:
            return -1
        return int(self._architecture)

    @property
    def name(self) -> str:
        """
        The name of the device architecture.

        Returns "Unlisted" if the device is unknown.
        """
        if self._architecture is None:
            return "Unlisted"
        name = self._architecture.name
        return name[name.rfind("_") + 1 :].title()


cdef class MemoryInfo:
    """
    Memory allocation information for a device.
    """
    cdef object _memory_info

    def __init__(self, memory_info: nvml.Memory_v2):
        self._memory_info = memory_info

    @property
    def free(self) -> int:
        """
        Unallocated device memory (in bytes)
        """
        return self._memory_info.free

    @property
    def total(self) -> int:
        """
        Total physical device memory (in bytes)
        """
        return self._memory_info.total

    @property
    def used(self) -> int:
        """
        Allocated device memory (in bytes)
        """
        return self._memory_info.used

    @property
    def reserved(self) -> int:
        """
        Device memory (in bytes) reserved for system use (driver or firmware)
        """
        return self._memory_info.reserved


cdef class BAR1MemoryInfo(MemoryInfo):
    """
    BAR1 Memory allocation information for a device.
    """
    cdef object _memory_info

    def __init__(self, memory_info: nvml.BAR1Memory):
        self._memory_info = memory_info

    @property
    def free(self) -> int:
        """
        Unallocated BAR1 memory (in bytes)
        """
        return self._memory_info.bar1_free

    @property
    def total(self) -> int:
        """
        Total BAR1 memory (in bytes)
        """
        return self._memory_info.bar1_total

    @property
    def used(self) -> int:
        """
        Allocated used memory (in bytes)
        """
        return self._memory_info.bar1_used


cdef class PciInfo:
    """
    PCI information about a GPU device.
    """
    cdef object _pci_info

    def __init__(self, pci_info: nvml.PciInfo):
        self._pci_info = pci_info

    @property
    def bus(self) -> int:
        """
        The bus on which the device resides, 0 to 255
        """
        return self._pci_info.bus

    @property
    def bus_id(self) -> str:
        """
        The tuple domain:bus:device.function PCI identifier string
        """
        return self._pci_info.bus_id

    @property
    def device(self) -> int:
        """
        The device's id on the bus, 0 to 31
        """
        return self._pci_info.device_

    @property
    def domain(self) -> int:
        """
        The PCI domain on which the device's bus resides, 0 to 0xffffffff
        """
        return self._pci_info.domain

    @property
    def vendor_id(self) -> int:
        """
        The PCI vendor id of the device
        """
        return self._pci_info.pci_device_id & 0xFFFF

    @property
    def device_id(self) -> int:
        """
        The PCI device id of the device
        """
        return self._pci_info.pci_device_id >> 16


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


cdef class Device:
    """
    Representation of a device.

    :class:`cuda.core.system.Device` provides access to various pieces of metadata
    about devices and their topology, as provided by the NVIDIA Management
    Library (NVML).  To use CUDA with a device, use :class:`cuda.core.Device`.

    Parameters
    ----------
    index: int, optional
        Integer representing the CUDA device index to get a handle to.
    uuid: bytes or str, optional
        UUID of a CUDA device to get a handle to.

    Raises
    ------
    ValueError
        If neither `index` nor `uuid` are specified or if both are specified.
    """

    cdef intptr_t _handle

    def __init__(self, index: int | None = None, uuid: bytes | str | None = None):
        initialize()

        if index is not None and uuid is not None:
            raise ValueError("Handle requires only one of either device `index` or `uuid`.")
        if index is None and uuid is None:
            raise ValueError("Handle requires either a device `index` or `uuid`.")

        if index is not None:
            self._handle = nvml.device_get_handle_by_index_v2(index)
        else:
            if isinstance(uuid, bytes):
                uuid = uuid.decode("ascii")
            self._handle = nvml.device_get_handle_by_uuid(uuid)

    @property
    def handle(self) -> int:
        return self._handle

    @classmethod
    def get_all_devices(cls) -> Iterable[Device]:
        """
        Query the available device instances.

        Returns
        -------
        Iterator of Device
            An iterator over available devices.
        """
        total = nvml.device_get_count_v2()
        for device_id in range(total):
            yield cls(device_id)

    @property
    def architecture(self) -> DeviceArchitecture:
        """
        Device architecture. For example, a Tesla V100 will report
        ``DeviceArchitecture.name == "Volta"``, and RTX A6000 will report
        ``DeviceArchitecture.name == "Ampere"``. If the device returns an
        architecture that is unknown to NVML then ``DeviceArchitecture.name ==
        "Unknown"`` is reported, whereas an architecture that is unknown to
        cuda.core.system is reported as ``DeviceArchitecture.name == "Unlisted"``.
        """
        return DeviceArchitecture(nvml.device_get_architecture(self._handle))

    @property
    def bar1_memory_info(self) -> BAR1MemoryInfo:
        """
        Get information about BAR1 memory.

        BAR1 is used to map the FB (device memory) so that it can be directly
        accessed by the CPU or by 3rd party devices (peer-to-peer on the PCIE
        bus).
        """
        return BAR1MemoryInfo(nvml.device_get_bar1_memory_info(self._handle))

    @property
    def cpu_affinity(self) -> list[int]:
        """
        Get a list containing the CPU indices to which the GPU is directly connected.

        Examples
        --------
        >>> Device(index=0).cpu_affinity
        [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19,
         40, 41, 42, 43, 44, 45, 46, 47, 48, 49, 50, 51, 52, 53, 54, 55, 56, 57, 58, 59]
        """
        return _unpack_bitmask(nvml.device_get_cpu_affinity(
            self._handle,
            <unsigned int>ceil(cpu_count() / 64),
        ))

    @property
    def cuda_compute_capability(self) -> tuple[int, int]:
        """
        CUDA compute capability of the device, e.g.: `(7, 0)` for a Tesla V100.

        Returns a tuple `(major, minor)`.
        """
        return nvml.device_get_cuda_compute_capability(self._handle)

    @property
    def memory_info(self) -> MemoryInfo:
        """
        Object with memory information.
        """
        return MemoryInfo(nvml.device_get_memory_info_v2(self._handle))

    @property
    def name(self) -> str:
        """
        Name of the device, e.g.: `"Tesla V100-SXM2-32GB"`
        """
        return nvml.device_get_name(self._handle)

    @property
    def pci_info(self) -> PciInfo:
        """
        The PCI attributes of this device.
        """
        return PciInfo(nvml.device_get_pci_info_v3(self._handle))

    @property
    def serial(self) -> str:
        """
        Retrieves the globally unique board serial number associated with this
        device's board.
        """
        return nvml.device_get_serial(self._handle)

    @property
    def uuid(self) -> str:
        """
        Retrieves the globally unique immutable UUID associated with this
        device, as a 5 part hexadecimal string, that augments the immutable,
        board serial identifier.
        """
        return nvml.device_get_uuid(self._handle)

    def get_field_values(self, field_ids: list[int | tuple[int, int]]) -> FieldValues:
        """
        Get multiple field values from the device.

        Each value specified can raise its own exception.  That exception will
        be raised when attempting to access the corresponding ``value`` from the
        returned :class:`FieldValues` container.

        To confirm that there are no exceptions in the entire container, call
        :meth:`FieldValues.validate`.

        Parameters
        ----------
        field_ids: list of int or tuple of (int, int)
            List of field IDs to query.

            Each item may be either a single from the :class:`FieldId` enum, or a pair
            of (:class:`FieldId`, scope ID).

        Returns
        -------
        :class:`FieldValues`
            Container of field values corresponding to the requested field IDs.
        """
        return FieldValues(nvml.device_get_field_values(self._handle, field_ids))

    def clear_field_values(self, field_ids: list[int | tuple[int, int]]) -> None:
        """
        Clear multiple field values from the device.

        Parameters
        ----------
        field_ids: list of int or tuple of (int, int)
            List of field IDs to clear.

            Each item may be either a single from the :class:`FieldId` enum, or
            a pair of (:class:`FieldId`, scope ID).
        """
        nvml.device_clear_field_values(self._handle, field_ids)


__all__ = [
    "BAR1MemoryInfo",
    "Device",
    "DeviceArchitecture",
    "FieldId",
    "FieldValue",
    "FieldValues",
    "MemoryInfo",
    "PciInfo",
]
