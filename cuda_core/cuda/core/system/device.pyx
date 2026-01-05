# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

from libc.stdint cimport intptr_t
from libc.math cimport ceil

from multiprocessing import cpu_count
from typing import Iterable

from cuda.bindings import _nvml as nvml

from ._nvml_context import validate
from .utils cimport unpack_bitmask


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


cdef class Device:
    """
    Representation of a CUDA device.

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
        validate()

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
        return unpack_bitmask(nvml.device_get_cpu_affinity(
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
