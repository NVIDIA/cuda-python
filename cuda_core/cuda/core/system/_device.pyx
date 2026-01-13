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
include "_inforom.pxi"


AddressingMode = nvml.DeviceAddressingModeType
BrandType = nvml.BrandType
FieldId = nvml.FieldId
GpuP2PCapsIndex = nvml.GpuP2PCapsIndex
GpuP2PStatus = nvml.GpuP2PStatus
GpuTopologyLevel = nvml.GpuTopologyLevel
InforomObject = nvml.InforomObject
PcieUtilCounter = nvml.PcieUtilCounter


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

    cdef object _pci_info_ext
    cdef intptr_t _handle

    def __init__(self, pci_info_ext: nvml.PciInfoExt_v1, handle: int):
        self._pci_info_ext = pci_info_ext
        self._handle = handle

    @property
    def bus(self) -> int:
        """
        The bus on which the device resides, 0 to 255
        """
        return self._pci_info_ext.bus

    @property
    def bus_id(self) -> str:
        """
        The tuple domain:bus:device.function PCI identifier string
        """
        return self._pci_info_ext.bus_id

    @property
    def device(self) -> int:
        """
        The device's id on the bus, 0 to 31
        """
        return self._pci_info_ext.device_

    @property
    def domain(self) -> int:
        """
        The PCI domain on which the device's bus resides, 0 to 0xffffffff
        """
        return self._pci_info_ext.domain

    @property
    def vendor_id(self) -> int:
        """
        The PCI vendor id of the device
        """
        return self._pci_info_ext.pci_device_id & 0xFFFF

    @property
    def device_id(self) -> int:
        """
        The PCI device id of the device
        """
        return self._pci_info_ext.pci_device_id >> 16

    @property
    def subsystem_id(self) -> int:
        """
        The subsystem device ID
        """
        return self._pci_info_ext.pci_sub_system_id

    @property
    def base_class(self) -> int:
        """
        The 8-bit PCI base class code
        """
        return self._pci_info_ext.base_class

    @property
    def sub_class(self) -> int:
        """
        The 8-bit PCI sub class code
        """
        return self._pci_info_ext.sub_class

    def get_max_pcie_link_generation(self) -> int:
        """
        Retrieve the maximum PCIe link generation possible with this device and system.

        For Fermi™ or newer fully supported devices.

        For example, for a generation 2 PCIe device attached to a generation 1
        PCIe bus, the max link generation this function will report is
        generation 1.
        """
        return nvml.device_get_max_pcie_link_generation(self._handle)

    def get_gpu_max_pcie_link_generation(self) -> int:
        """
        Retrieve the maximum PCIe link generation supported by this GPU device.

        For Fermi™ or newer fully supported devices.
        """
        return nvml.device_get_gpu_max_pcie_link_generation(self._handle)

    def get_max_pcie_link_width(self) -> int:
        """
        Retrieve the maximum PCIe link width possible with this device and system.

        For Fermi™ or newer fully supported devices.

        For example, for a device with a 16x PCIe bus width attached to a 8x
        PCIe system bus this function will report
        a max link width of 8.
        """
        return nvml.device_get_max_pcie_link_width(self._handle)

    def get_current_pcie_link_generation(self) -> int:
        """
        Retrieve the current PCIe link generation.

        For Fermi™ or newer fully supported devices.
        """
        return nvml.device_get_curr_pcie_link_generation(self._handle)

    def get_current_pcie_link_width(self) -> int:
        """
        Retrieve the current PCIe link width.

        For Fermi™ or newer fully supported devices.
        """
        return nvml.device_get_curr_pcie_link_width(self._handle)

    def get_pcie_throughput(self, counter: PcieUtilCounter) -> int:
        """
        Retrieve PCIe utilization information, in KB/s.

        This function is querying a byte counter over a 20ms interval, and thus
        is the PCIe throughput over that interval.

        For Maxwell™ or newer fully supported devices.

        This method is not supported in virtual machines running virtual GPU
        (vGPU).
        """
        return nvml.device_get_pcie_throughput(self._handle, counter)

    def get_pcie_replay_counter(self) -> int:
        """
        Retrieve the PCIe replay counter.

        For Kepler™ or newer fully supported devices.
        """
        return nvml.device_get_pcie_replay_counter(self._handle)


cdef class DeviceAttributes:
    """
    Various device attributes.
    """
    def __init__(self, attributes: nvml.DeviceAttributes):
        self._attributes = attributes

    @property
    def multiprocessor_count(self) -> int:
        """
        The streaming multiprocessor count
        """
        return self._attributes.multiprocessor_count

    @property
    def shared_copy_engine_count(self) -> int:
        """
        The shared copy engine count
        """
        return self._attributes.shared_copy_engine_count

    @property
    def shared_decoder_count(self) -> int:
        """
        The shared decoder engine count
        """
        return self._attributes.shared_decoder_count

    @property
    def shared_encoder_count(self) -> int:
        """
        The shared encoder engine count
        """
        return self._attributes.shared_encoder_count

    @property
    def shared_jpeg_count(self) -> int:
        """
        The shared JPEG engine count
        """
        return self._attributes.shared_jpeg_count

    @property
    def shared_ofa_count(self) -> int:
        """
        The shared optical flow accelerator (OFA) engine count
        """
        return self._attributes.shared_ofa_count

    @property
    def gpu_instance_slice_count(self) -> int:
        """
        The GPU instance slice count
        """
        return self._attributes.gpu_instance_slice_count

    @property
    def compute_instance_slice_count(self) -> int:
        """
        The compute instance slice count
        """
        return self._attributes.compute_instance_slice_count

    @property
    def memory_size_mb(self) -> int:
        """
        Device memory size in MiB
        """
        return self._attributes.memory_size_mb


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


cdef class RepairStatus:
    """
    Repair status for TPC/Channel repair.
    """
    cdef object _repair_status

    def __init__(self, handle: int):
        self._repair_status = nvml.device_get_repair_status(handle)

    @property
    def channel_repair_pending(self) -> bool:
        """
        `True` if a channel repair is pending.
        """
        return bool(self._repair_status.b_channel_repair_pending)

    @property
    def tpc_repair_pending(self) -> bool:
        """
        `True` if a TPC repair is pending.
        """
        return bool(self._repair_status.b_tpc_repair_pending)


cdef class Device:
    """
    Representation of a device.

    :class:`cuda.core.system.Device` provides access to various pieces of metadata
    about devices and their topology, as provided by the NVIDIA Management
    Library (NVML).  To use CUDA with a device, use :class:`cuda.core.Device`.

    Creating a device instance causes NVML to initialize the target GPU.
    NVML may initialize additional GPUs if the target GPU is an SLI slave.

    Parameters
    ----------
    index: int, optional
        Integer representing the CUDA device index to get a handle to.  Valid
        values are between ``0`` and ``cuda.core.system.get_num_devices() - 1``.

        The order in which devices are enumerated has no guarantees of
        consistency between reboots.  For that reason, it is recommended that
        devices are looked up by their PCI ids or UUID.

    uuid: bytes or str, optional
        UUID of a CUDA device to get a handle to.

    pci_bus_id: bytes or str, optional
        PCI bus ID of a CUDA device to get a handle to.

    Raises
    ------
    ValueError
        If anything other than a single `index`, `uuid` or `pci_bus_id` are specified.
    """

    cdef intptr_t _handle

    def __init__(
        self,
        *,
        index: int | None = None,
        uuid: bytes | str | None = None,
        pci_bus_id: bytes | str | None = None,
    ):
        args = [index, uuid, pci_bus_id]
        cdef int arg_count = sum(arg is not None for arg in args)

        if arg_count > 1:
            raise ValueError("Handle requires only one of `index`, `uuid`, or `pci_bus_id`.")
        if arg_count == 0:
            raise ValueError("Handle requires either a device `index`, `uuid`, or `pci_bus_id`.")

        initialize()

        if index is not None:
            self._handle = nvml.device_get_handle_by_index_v2(index)
        elif uuid is not None:
            if isinstance(uuid, bytes):
                uuid = uuid.decode("ascii")
            self._handle = nvml.device_get_handle_by_uuid(uuid)
        elif pci_bus_id is not None:
            if isinstance(pci_bus_id, bytes):
                pci_bus_id = pci_bus_id.decode("ascii")
            self._handle = nvml.device_get_handle_by_pci_bus_id_v2(pci_bus_id)

    @classmethod
    def get_device_count(cls) -> int:
        """
        Get the number of available devices.

        Returns
        -------
        int
            The number of available devices.
        """
        return nvml.device_get_count_v2()

    @classmethod
    def get_all_devices(cls) -> Iterable[Device]:
        """
        Query the available device instances.

        Returns
        -------
        Iterator of Device
            An iterator over available devices.
        """
        for device_id in range(nvml.device_get_count_v2()):
            yield cls(index=device_id)

    @classmethod
    def get_all_devices_with_cpu_affinity(cls, cpu_index: int) -> Iterable[Device]:
        """
        Retrieve the set of GPUs that have a CPU affinity with the given CPU number.

        Supported on Linux only.

        Parameters
        ----------
        cpu_index: int
            The CPU index.

        Returns
        -------
        Iterator of Device
            An iterator over available devices.
        """
        cdef Device device
        for handle in nvml.system_get_topology_gpu_set(cpu_index):
            device = Device.__new__(Device)
            device._handle = handle
            yield device

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
    def brand(self) -> BrandType:
        """
        Brand of the device
        """
        return BrandType(nvml.device_get_brand(self._handle))

    @property
    def index(self) -> int:
        """
        The NVML index of this device.

        The order in which NVML enumerates devices has no guarantees of
        consistency between reboots. For that reason it is recommended that
        devices be looked up by their PCI ids or GPU UUID.
        """
        return nvml.device_get_index(self._handle)

    @property
    def pci_info(self) -> PciInfo:
        """
        The PCI attributes of this device.
        """
        return PciInfo(nvml.device_get_pci_info_ext(self._handle), self._handle)

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

    @property
    def index(self) -> int:
        """
        The NVML index of this device.

        Valid indices are derived from the count returned by
        :meth:`Device.get_device_count`.  For example, if ``get_device_count()``
        returns 2, the valid indices are 0 and 1, corresponding to GPU 0 and GPU
        1.

        The order in which NVML enumerates devices has no guarantees of
        consistency between reboots. For that reason, it is recommended that
        devices be looked up by their PCI ids or GPU UUID.

        Note: The NVML index may not correlate with other APIs, such as the CUDA
        device index.
        """
        return nvml.device_get_index(self._handle)

    @property
    def module_id(self) -> int:
        """
        Get a unique identifier for the device module on the baseboard.

        This API retrieves a unique identifier for each GPU module that exists
        on a given baseboard.  For non-baseboard products, this ID would always
        be 0.
        """
        return nvml.device_get_module_id(self._handle)

    @property
    def minor_number(self) -> int:
        """
        The minor number of this device.

        For Linux only.

        The minor number is used by the Linux device driver to identify the
        device node in ``/dev/nvidiaX``.
        """
        return nvml.device_get_minor_number(self._handle)

    @property
    def addressing_mode(self) -> AddressingMode:
        """
        Get the addressing mode of the device.

        Addressing modes can be one of:

        - :attr:`AddressingMode.DEVICE_ADDRESSING_MODE_HMM`: System allocated
          memory (``malloc``, ``mmap``) is addressable from the device (GPU), via
          software-based mirroring of the CPU's page tables, on the GPU.
        - :attr:`AddressingMode.DEVICE_ADDRESSING_MODE_ATS`: System allocated
          memory (``malloc``, ``mmap``) is addressable from the device (GPU), via
          Address Translation Services. This means that there is (effectively) a
          single set of page tables, and the CPU and GPU both use them.
        - :attr:`AddressingMode.DEVICE_ADDRESSING_MODE_NONE`: Neither HMM nor ATS
          is active.
        """
        return AddressingMode(nvml.device_get_addressing_mode(self._handle).value)

    @property
    def display_mode(self) -> bool:
        """
        The display mode for this device.

        Indicates whether a physical display (e.g. monitor) is currently connected to
        any of the device's connectors.
        """
        return True if nvml.device_get_display_mode(self._handle) == nvml.EnableState.FEATURE_ENABLED else False

    @property
    def display_active(self) -> bool:
        """
        The display active status for this device.

        Indicates whether a display is initialized on the device.  For example,
        whether X Server is attached to this device and has allocated memory for
        the screen.

        Display can be active even when no monitor is physically attached.
        """
        return True if nvml.device_get_display_active(self._handle) == nvml.EnableState.FEATURE_ENABLED else False

    @property
    def repair_status(self) -> RepairStatus:
        """
        Get the repair status for TPC/Channel repair.

        For Ampere™ or newer fully supported devices.
        """
        return RepairStatus(self._handle)

    @property
    def inforom(self) -> InforomInfo:
        """
        Accessor for InfoROM information.

        For all products with an InfoROM.
        """
        return InforomInfo(self)

    def get_topology_nearest_gpus(self, level: GpuTopologyLevel) -> Iterable[Device]:
        """
        Retrieve the GPUs that are nearest to this device at a specific interconnectivity level.

        Supported on Linux only.

        Parameters
        ----------
        level: :class:`GpuTopologyLevel`
            The topology level.

        Returns
        -------
        Iterable of :class:`Device`
            The nearest devices at the given topology level.
        """
        cdef Device device
        for handle in nvml.device_get_topology_nearest_gpus(self._handle, level):
            device = Device.__new__(Device)
            device._handle = handle
            yield device

    @property
    def attributes(self) -> DeviceAttributes:
        """
        Get various device attributes.

        For Ampere™ or newer fully supported devices.  Only available on Linux
        systems.
        """
        return DeviceAttributes(nvml.device_get_attributes_v2(self._handle))

    @property
    def is_c2c_mode_enabled(self) -> bool:
        """
        Whether the C2C (Chip-to-Chip) mode is enabled for this device.
        """
        return bool(nvml.device_get_c2c_mode_info_v(self._handle).is_c2c_enabled)

    @property
    def persistence_mode_enabled(self) -> bool:
        """
        Whether persistence mode is enabled for this device.

        For Linux only.
        """
        return nvml.device_get_persistence_mode(self._handle) == nvml.EnableState.FEATURE_ENABLED

    @persistence_mode_enabled.setter
    def persistence_mode_enabled(self, enabled: bool) -> None:
        nvml.device_set_persistence_mode(
            self._handle,
            nvml.EnableState.FEATURE_ENABLED if enabled else nvml.EnableState.FEATURE_DISABLED
        )

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

            Each item may be either a single value from the :class:`FieldId`
            enum, or a pair of (:class:`FieldId`, scope ID).

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

            Each item may be either a single value from the :class:`FieldId`
            enum, or a pair of (:class:`FieldId`, scope ID).
        """
        nvml.device_clear_field_values(self._handle, field_ids)


def get_topology_common_ancestor(device1: Device, device2: Device) -> GpuTopologyLevel:
    """
    Retrieve the common ancestor for two devices.

    For Linux only.

    Parameters
    ----------
    device1: :class:`Device`
        The first device.
    device2: :class:`Device`
        The second device.

    Returns
    -------
    :class:`GpuTopologyLevel`
        The common ancestor level of the two devices.
    """
    return GpuTopologyLevel(
        nvml.device_get_topology_common_ancestor(
            device1._handle,
            device2._handle,
        )
    )


def get_p2p_status(device1: Device, device2: Device, index: GpuP2PCapsIndex) -> GpuP2PStatus:
    """
    Retrieve the P2P status between two devices.

    Parameters
    ----------
    device1: :class:`Device`
        The first device.
    device2: :class:`Device`
        The second device.
    index: :class:`GpuP2PCapsIndex`
        The P2P capability index being looked for between ``device1`` and ``device2``.

    Returns
    -------
    :class:`GpuP2PStatus`
        The P2P status between the two devices.
    """
    return GpuP2PStatus(
        nvml.device_get_p2p_status(
            device1._handle,
            device2._handle,
            index,
        )
    )


__all__ = [
    "AddressingMode",
    "BAR1MemoryInfo",
    "BrandType",
    "Device",
    "DeviceArchitecture",
    "DeviceAttributes",
    "FieldId",
    "FieldValue",
    "FieldValues",
    "GpuP2PCapsIndex",
    "GpuP2PStatus",
    "GpuTopologyLevel",
    "InforomInfo",
    "InforomObject",
    "MemoryInfo",
    "PcieUtilCounter",
    "PciInfo",
    "RepairStatus",
    "get_p2p_status",
    "get_topology_common_ancestor",
]
