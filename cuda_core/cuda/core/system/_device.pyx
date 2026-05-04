# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

from libc.stdint cimport intptr_t, uint64_t
from libc.math cimport ceil

import sys
if sys.version_info >= (3, 11):
    from enum import StrEnum
else:
    from backports.strenum import StrEnum
from multiprocessing import cpu_count
from typing import Iterable
import warnings

from cuda.bindings import nvml
from cuda.bindings._internal._fast_enum import FastEnum

from ._nvml_context cimport initialize


cdef object _pstate_to_int(object pstate):
    if pstate == nvml.Pstates.PSTATE_UNKNOWN:
        return None
    return int(pstate) - int(nvml.Pstates.PSTATE_0)


cdef int _pstate_to_enum(int pstate):
    if pstate < 0 or pstate > 15:
        raise ValueError(f"Invalid P-state: {pstate}. Must be between 0 and 15 inclusive.")
    return int(pstate) + int(nvml.Pstates.PSTATE_0)


include "_clock.pxi"
include "_cooler.pxi"
include "_device_attributes.pxi"
include "_device_utils.pxi"
include "_event.pxi"
include "_fan.pxi"
include "_field_values.pxi"
include "_inforom.pxi"
include "_memory.pxi"
include "_mig.pxi"
include "_nvlink.pxi"
include "_pci_info.pxi"
include "_performance.pxi"
include "_process.pxi"
include "_repair_status.pxi"
include "_temperature.pxi"
include "_utilization.pxi"


class AddressingMode(StrEnum):
    """
    Addressing mode of a device.

    For Kepler™ or newer fully supported devices.
    """
    HMM = "hmm"
    ATS = "ats"
AddressingMode.HMM.__doc__ = """
    System allocated memory (``malloc``, ``mmap``) is addressable from the device
    (GPU), via software-based mirroring of the CPU's page tables, on the GPU.
"""
AddressingMode.ATS.__doc__ = """
    System allocated memory (``malloc``, ``mmap``) is addressable from the device
    (GPU), via Address Translation Services. This means that there is (effectively)
    a single set of page tables, and the CPU and GPU both use them.
"""
cdef dict _ADDRESSING_MODE_MAPPING = {
    nvml.DeviceAddressingModeType.DEVICE_ADDRESSING_MODE_HMM: AddressingMode.HMM,
    nvml.DeviceAddressingModeType.DEVICE_ADDRESSING_MODE_ATS: AddressingMode.ATS,
}


class AffinityScope(StrEnum):
    """
    Scope for affinity queries.
    """
    NODE = "node"
    SOCKET = "socket"
AffinityScope.NODE.__doc__ = """
    The NUMA node is the scope of the affinity query.  This is the default scope.
"""
AffinityScope.SOCKET.__doc__ = """
    The CPU socket is the scope of the affinity query.
"""
cdef dict _AFFINITY_SCOPE_MAPPING = {
    AffinityScope.NODE: nvml.AffinityScope.NODE,
    AffinityScope.SOCKET: nvml.AffinityScope.SOCKET,
}


cdef dict _BRAND_TYPE_MAPPING = {
    nvml.BrandType.BRAND_UNKNOWN: "Unknown",
    nvml.BrandType.BRAND_QUADRO: "Quadro",
    nvml.BrandType.BRAND_TESLA: "Tesla",
    nvml.BrandType.BRAND_NVS: "NVS",
    nvml.BrandType.BRAND_GRID: "GRID",
    nvml.BrandType.BRAND_GEFORCE: "GeForce",
    nvml.BrandType.BRAND_TITAN: "Titan",
    nvml.BrandType.BRAND_NVIDIA_VAPPS: "NVIDIA vApps",
    nvml.BrandType.BRAND_NVIDIA_VPC: "NVIDIA VPC",
    nvml.BrandType.BRAND_NVIDIA_VCS: "NVIDIA VCS",
    nvml.BrandType.BRAND_NVIDIA_VWS: "NVIDIA VWS",
    nvml.BrandType.BRAND_NVIDIA_CLOUD_GAMING: "NVIDIA Cloud Gaming",
    nvml.BrandType.BRAND_NVIDIA_VGAMING: "NVIDIA vGaming",
    nvml.BrandType.BRAND_QUADRO_RTX: "Quadro RTX",
    nvml.BrandType.BRAND_NVIDIA_RTX: "NVIDIA RTX",
    nvml.BrandType.BRAND_NVIDIA: "NVIDIA",
    nvml.BrandType.BRAND_GEFORCE_RTX: "GeForce RTX",
    nvml.BrandType.BRAND_TITAN_RTX: "Titan RTX",
}


# This uses FastEnum instead of StrEnum because the ordering of the values is
# meaningful, e.g. Kepler "or later"
class DeviceArch(FastEnum):
    """
    Device architecture.
    """
    KEPLER = int(nvml.DeviceArch.KEPLER)
    MAXWELL = int(nvml.DeviceArch.MAXWELL)
    PASCAL = int(nvml.DeviceArch.PASCAL)
    VOLTA = int(nvml.DeviceArch.VOLTA)
    TURING = int(nvml.DeviceArch.TURING)
    AMPERE = int(nvml.DeviceArch.AMPERE)
    ADA = int(nvml.DeviceArch.ADA)
    HOPPER = int(nvml.DeviceArch.HOPPER)
    BLACKWELL = int(nvml.DeviceArch.BLACKWELL)
    UNKNOWN = int(nvml.DeviceArch.UNKNOWN)


class GpuP2PCapsIndex(StrEnum):
    """
    GPU peer-to-peer capabilities index.
    """
    READ = "read"
    WRITE = "write"
    NVLINK = "nvlink"
    ATOMICS = "atomics"
    PCI = "pci"
    PROP = "prop"
    UNKNOWN = "unknown"
cdef dict _GPU_P2P_CAPS_INDEX_MAPPING = {
    GpuP2PCapsIndex.READ: nvml.GpuP2PCapsIndex.P2P_CAPS_INDEX_READ,
    GpuP2PCapsIndex.WRITE: nvml.GpuP2PCapsIndex.P2P_CAPS_INDEX_WRITE,
    GpuP2PCapsIndex.NVLINK: nvml.GpuP2PCapsIndex.P2P_CAPS_INDEX_NVLINK,
    GpuP2PCapsIndex.ATOMICS: nvml.GpuP2PCapsIndex.P2P_CAPS_INDEX_ATOMICS,
    GpuP2PCapsIndex.PCI: nvml.GpuP2PCapsIndex.P2P_CAPS_INDEX_PCI,
    GpuP2PCapsIndex.PROP: nvml.GpuP2PCapsIndex.P2P_CAPS_INDEX_PROP,
}


class GpuP2PStatus(StrEnum):
    """
    GPU peer-to-peer status.
    """
    OK = "ok"
    CHIPSET_NOT_SUPPORTED = "chipset not supported"
    GPU_NOT_SUPPORTED = "GPU not supported"
    IOH_TOPOLOGY_NOT_SUPPORTED = "IOH topology not supported"
    DISABLED_BY_REGKEY = "disabled by regkey"
    NOT_SUPPORTED = "not supported"
    UNKNOWN = "unknown"
cdef dict _GPU_P2P_STATUS_MAPPING = {
    nvml.GpuP2PStatus.P2P_STATUS_OK: GpuP2PStatus.OK,
    # Typo in upstream library
    nvml.GpuP2PStatus.P2P_STATUS_CHIPSET_NOT_SUPPORED: GpuP2PStatus.CHIPSET_NOT_SUPPORTED,
    nvml.GpuP2PStatus.P2P_STATUS_CHIPSET_NOT_SUPPORTED: GpuP2PStatus.CHIPSET_NOT_SUPPORTED,
    nvml.GpuP2PStatus.P2P_STATUS_GPU_NOT_SUPPORTED: GpuP2PStatus.GPU_NOT_SUPPORTED,
    nvml.GpuP2PStatus.P2P_STATUS_IOH_TOPOLOGY_NOT_SUPPORTED: GpuP2PStatus.IOH_TOPOLOGY_NOT_SUPPORTED,
    nvml.GpuP2PStatus.P2P_STATUS_DISABLED_BY_REGKEY: GpuP2PStatus.DISABLED_BY_REGKEY,
    nvml.GpuP2PStatus.P2P_STATUS_NOT_SUPPORTED: GpuP2PStatus.NOT_SUPPORTED,
    nvml.GpuP2PStatus.P2P_STATUS_UNKNOWN: GpuP2PStatus.UNKNOWN,
}


class GpuTopologyLevel(StrEnum):
    """
    Represents level relationships within a system between two GPUs.
    """
    INTERNAL = "internal"
    SINGLE = "single"
    MULTIPLE = "multiple"
    HOSTBRIDGE = "hostbridge"
    NODE = "node"
    SYSTEM = "system"
cdef dict _GPU_TOPOLOGY_LEVEL_MAPPING = {
    GpuTopologyLevel.INTERNAL: nvml.GpuTopologyLevel.TOPOLOGY_INTERNAL,
    GpuTopologyLevel.SINGLE: nvml.GpuTopologyLevel.TOPOLOGY_SINGLE,
    GpuTopologyLevel.MULTIPLE: nvml.GpuTopologyLevel.TOPOLOGY_MULTIPLE,
    GpuTopologyLevel.HOSTBRIDGE: nvml.GpuTopologyLevel.TOPOLOGY_HOSTBRIDGE,
    GpuTopologyLevel.NODE: nvml.GpuTopologyLevel.TOPOLOGY_NODE,
    GpuTopologyLevel.SYSTEM: nvml.GpuTopologyLevel.TOPOLOGY_SYSTEM,
}
cdef dict _GPU_TOPOLOGY_LEVEL_INV_MAPPING = {v: k for k, v in _GPU_TOPOLOGY_LEVEL_MAPPING.items()}



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

    # This is made public for testing purposes only
    cdef public intptr_t _handle

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

    #########################################################################
    # BASIC PROPERTIES

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
    def uuid(self) -> str:
        """
        Retrieves the globally unique immutable UUID associated with this
        device, as a 5 part hexadecimal string, that augments the immutable,
        board serial identifier.

        In the upstream NVML C++ API, the UUID includes a ``gpu-`` or ``mig-``
        prefix.  If you need a `uuid` without that prefix (for example, to
        interact with CUDA), use the `uuid_without_prefix` property.
        """
        return nvml.device_get_uuid(self._handle)

    @property
    def uuid_without_prefix(self) -> str:
        """
        Retrieves the globally unique immutable UUID associated with this
        device, as a 5 part hexadecimal string, that augments the immutable,
        board serial identifier.

        In the upstream NVML C++ API, the UUID includes a ``gpu-`` or ``mig-``
        prefix.  This property returns it without the prefix, to match the UUIDs
        used in CUDA.  If you need the prefix, use the `uuid` property.
        """
        # NVML UUIDs have a `gpu-` or `mig-` prefix.  We remove that here.
        return nvml.device_get_uuid(self._handle)[4:]

    @property
    def pci_bus_id(self) -> str:
        """
        Retrieves the PCI bus ID of this device.
        """
        return self.pci_info.bus_id

    @property
    def numa_node_id(self) -> int:
        """
        The NUMA node of the given GPU device.

        This only applies to platforms where the GPUs are NUMA nodes.
        """
        return nvml.device_get_numa_node_id(self._handle)

    @property
    def arch(self) -> DeviceArch:
        """
        :obj:`~DeviceArch` device architecture.

        For example, a Tesla V100 will report ``DeviceArchitecture.name ==
        "VOLTA"``, and RTX A6000 will report ``DeviceArchitecture.name ==
        "AMPERE"``.
        """
        arch = nvml.device_get_architecture(self._handle)
        try:
            return DeviceArch(arch)
        except ValueError:
            return DeviceArch.UNKNOWN

    @property
    def name(self) -> str:
        """
        Name of the device, e.g.: `"Tesla V100-SXM2-32GB"`
        """
        return nvml.device_get_name(self._handle)

    @property
    def brand(self) -> str:
        """
        The brand of the device.

        Returns "Unknown" if the brand is unknown.
        """
        return _BRAND_TYPE_MAPPING.get(nvml.device_get_brand(self._handle), "Unknown")

    @property
    def serial(self) -> str:
        """
        Retrieves the globally unique board serial number associated with this
        device's board.

        For all products with an InfoROM.
        """
        return nvml.device_get_serial(self._handle)

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
    def is_c2c_enabled(self) -> bool:
        """
        Whether the C2C (Chip-to-Chip) mode is enabled for this device.
        """
        return bool(nvml.device_get_c2c_mode_info_v(self._handle).is_c2c_enabled)

    @property
    def is_persistence_mode_enabled(self) -> bool:
        """
        Whether persistence mode is enabled for this device.

        For Linux only.
        """
        return nvml.device_get_persistence_mode(self._handle) == nvml.EnableState.FEATURE_ENABLED

    @is_persistence_mode_enabled.setter
    def is_persistence_mode_enabled(self, enabled: bool) -> None:
        nvml.device_set_persistence_mode(
            self._handle,
            nvml.EnableState.FEATURE_ENABLED if enabled else nvml.EnableState.FEATURE_DISABLED
        )

    @property
    def cuda_compute_capability(self) -> tuple[int, int]:
        """
        CUDA compute capability of the device, e.g.: `(7, 0)` for a Tesla V100.

        Returns a tuple `(major, minor)`.
        """
        return nvml.device_get_cuda_compute_capability(self._handle)

    def to_cuda_device(self) -> "cuda.core.Device":
        """
        Get the corresponding :class:`cuda.core.Device` (which is used for CUDA
        access) for this :class:`cuda.core.system.Device` (which is used for
        NVIDIA machine library (NVML) access).

        The devices are mapped to one another by their UUID.

        Returns
        -------
        cuda.core.Device
            The corresponding CUDA device.
        """
        from cuda.core import Device as CudaDevice

        # CUDA does not have an API to get a device by its UUID, so we just
        # search all the devices for one with a matching UUID.

        for cuda_device in CudaDevice.get_all_devices():
            if cuda_device.uuid == self.uuid_without_prefix:
                return cuda_device

        raise RuntimeError("No corresponding CUDA device found for this NVML device.")

    @classmethod
    def get_device_count(cls) -> int:
        """
        Get the number of available devices.

        Returns
        -------
        int
            The number of available devices.
        """
        initialize()

        return nvml.device_get_count_v2()

    @classmethod
    def get_all_devices(cls) -> Iterable[Device]:
        """
        Query the available device instances.

        Returns
        -------
        Iterator over :obj:`~Device`
            An iterator over available devices.
        """
        initialize()

        for device_id in range(nvml.device_get_count_v2()):
            yield cls(index=device_id)

    #########################################################################
    # ADDRESSING MODE

    @property
    def addressing_mode(self) -> AddressingMode | None:
        """
        Get the :obj:`~AddressingMode` of the device.
        """
        return _ADDRESSING_MODE_MAPPING.get(nvml.device_get_addressing_mode(self._handle).value, None)

    #########################################################################
    # MIG (MULTI-INSTANCE GPU) DEVICES

    @property
    def mig(self) -> MigInfo:
        """
        Get :obj:`~MigInfo` accessor for MIG (Multi-Instance GPU) information.

        For Ampere™ or newer fully supported devices.
        """
        return MigInfo(self)

    #########################################################################
    # AFFINITY

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
        Iterator of :obj:`~Device`
            An iterator over available devices.
        """
        cdef Device device
        for handle in nvml.system_get_topology_gpu_set(cpu_index):
            device = Device.__new__(Device)
            device._handle = handle
            yield device

    def get_memory_affinity(self, scope: AffinityScope | str=AffinityScope.NODE) -> list[int]:
        """
        Retrieves a list of indices of NUMA nodes or CPU sockets with the ideal
        memory affinity for the device.

        For Kepler™ or newer fully supported devices.

        Supported on Linux only.

        If requested scope is not applicable to the target topology, the API
        will fall back to reporting the memory affinity for the immediate non-I/O
        ancestor of the device.

        Parameters
        ----------
        scope: AffinityScope | str, optional
            The scope of the affinity query.  Must be one of the values of
            :class:`AffinityScope`.  Default is :attr:`AffinityScope.NODE`.

        Returns
        -------
        list[int]
            A list of indices of NUMA nodes or CPU sockets with the ideal memory
            affinity for the device.
        """
        try:
            scope = _AFFINITY_SCOPE_MAPPING[scope]
        except KeyError:
            raise ValueError(
                f"Invalid affinity scope: {scope}. "
                f"Must be one of {list(AffinityScope.__members__.values())}"
            ) from None
        return _unpack_bitmask(
            nvml.device_get_memory_affinity(
                self._handle,
                <unsigned int>ceil(cpu_count() / 64),
                scope,
            )
        )

    def get_cpu_affinity(self, scope: AffinityScope | str=AffinityScope.NODE) -> list[int]:
        """
        Retrieves a list of indices of NUMA nodes or CPU sockets with the ideal
        CPU affinity for the device.

        For Kepler™ or newer fully supported devices.

        Supported on Linux only.

        If requested scope is not applicable to the target topology, the API
        will fall back to reporting the memory affinity for the immediate non-I/O
        ancestor of the device.

        Parameters
        ----------
        scope: AffinityScope | str, optional
            The scope of the affinity query.  Must be one of the values of
            :class:`AffinityScope`.  Default is :attr:`AffinityScope.NODE`.

        Returns
        -------
        list[int]
            A list of indices of NUMA nodes or CPU sockets with the ideal memory
            affinity for the device.
        """
        try:
            scope = _AFFINITY_SCOPE_MAPPING[scope]
        except KeyError:
            raise ValueError(
                f"Invalid affinity scope: {scope}. "
                f"Must be one of {list(AffinityScope.__members__.values())}"
            ) from None
        return _unpack_bitmask(
            nvml.device_get_cpu_affinity_within_scope(
                self._handle,
                <unsigned int>ceil(cpu_count() / 64),
                scope,
            )
        )

    def set_cpu_affinity(self):
        """
        Sets the ideal affinity for the calling thread and device.

        For Kepler™ or newer fully supported devices.

        Supported on Linux only.
        """
        nvml.device_set_cpu_affinity(self._handle)

    def clear_cpu_affinity(self):
        """
        Clear all affinity bindings for the calling thread.

        For Kepler™ or newer fully supported devices.

        Supported on Linux only.
        """
        nvml.device_clear_cpu_affinity(self._handle)

    #########################################################################
    # CLOCK
    # See external class definitions in _clock.pxi

    def get_clock(self, clock_type: ClockType | str) -> ClockInfo:
        """
        :obj:`~_device.ClockInfo` object to get information about and manage a specific clock on a device.
        """
        return ClockInfo(self._handle, clock_type)

    @property
    def is_auto_boosted_clocks_enabled(self) -> tuple[bool, bool]:
        """
        Retrieve the current state of auto boosted clocks on a device.

        For Kepler™ or newer fully supported devices.

        Auto Boosted clocks are enabled by default on some hardware, allowing
        the GPU to run at higher clock rates to maximize performance as thermal
        limits allow.

        On Pascal™ and newer hardware, Auto Boosted clocks are controlled
        through application clocks. Use :meth:`set_application_clocks` and
        :meth:`reset_application_clocks` to control Auto Boost behavior.

        Returns
        -------
        bool
            The current state of Auto Boosted clocks
        bool
            The default Auto Boosted clocks behavior

        """
        current, default = nvml.device_get_auto_boosted_clocks_enabled(self._handle)
        return current == nvml.EnableState.FEATURE_ENABLED, default == nvml.EnableState.FEATURE_ENABLED

    @property
    def current_clock_event_reasons(self) -> list[ClocksEventReasons]:
        """
        Retrieves the current :obj:`~ClocksEventReasons`.

        For all fully supported products.
        """
        cdef uint64_t[1] reasons
        reasons[0] = nvml.device_get_current_clocks_event_reasons(self._handle)
        output_reasons = []
        for reason in _unpack_bitmask(reasons):
            try:
                output_reason = _CLOCKS_EVENT_REASONS_MAPPING[1 << reason]
            except KeyError:
                raise ValueError(f"Unknown clock event reason bit: {1 << reason}")
            output_reasons.append(output_reason)
        return output_reasons

    @property
    def supported_clock_event_reasons(self) -> list[ClocksEventReasons]:
        """
        Retrieves supported :obj:`~ClocksEventReasons` that can be returned by
        :meth:`get_current_clock_event_reasons`.

        For all fully supported products.

        This method is not supported in virtual machines running virtual GPU (vGPU).
        """
        cdef uint64_t[1] reasons
        reasons[0] = nvml.device_get_supported_clocks_event_reasons(self._handle)
        output_reasons = []
        for reason in _unpack_bitmask(reasons):
            try:
                output_reason = _CLOCKS_EVENT_REASONS_MAPPING[1 << reason]
            except KeyError:
                raise ValueError(f"Unknown clock event reason bit: {1 << reason}")
            output_reasons.append(output_reason)
        return output_reasons

    ##########################################################################
    # COOLER
    # See external class definitions in _cooler.pxi

    @property
    def cooler(self) -> CoolerInfo:
        """
        :obj:`~_device.CoolerInfo` object with cooler information for the device.
        """
        return CoolerInfo(nvml.device_get_cooler_info(self._handle))

    ##########################################################################
    # DEVICE ATTRIBUTES
    # See external class definitions in _device_attributes.pxi

    @property
    def attributes(self) -> DeviceAttributes:
        """
        :obj:`~_device.DeviceAttributes` object with various device attributes.

        For Ampere™ or newer fully supported devices.  Only available on Linux
        systems.
        """
        return DeviceAttributes(nvml.device_get_attributes_v2(self._handle))

    #########################################################################
    # DISPLAY

    @property
    def is_display_connected(self) -> bool:
        """
        The display mode for this device.

        Indicates whether a physical display (e.g. monitor) is currently connected to
        any of the device's connectors.
        """
        return nvml.device_get_display_mode(self._handle) == nvml.EnableState.FEATURE_ENABLED

    @property
    def is_display_active(self) -> bool:
        """
        The display active status for this device.

        Indicates whether a display is initialized on the device.  For example,
        whether X Server is attached to this device and has allocated memory for
        the screen.

        Display can be active even when no monitor is physically attached.
        """
        return nvml.device_get_display_active(self._handle) == nvml.EnableState.FEATURE_ENABLED

    ##########################################################################
    # EVENTS
    # See external class definitions in _event.pxi

    def register_events(self, events: EventType | str | list[EventType | str]) -> DeviceEvents:
        """
        Starts recording events on this device.

        For Fermi™ or newer fully supported devices.  For Linux only.

        ECC events are available only on ECC-enabled devices (see
        :meth:`Device.get_total_ecc_errors`).  Power capping events are
        available only on Power Management enabled devices (see
        :meth:`Device.get_power_management_mode`).

        This call starts recording of events on specific device.  All events
        that occurred before this call are not recorded.  Wait for events using
        the :meth:`DeviceEvents.wait` method on the result.

        Examples
        --------
        >>> device = Device(index=0)
        >>> events = device.register_events([
        ...     EventType.XID_CRITICAL_ERROR,
        ... ])
        >>> while event := events.wait(timeout_ms=10000):
        ...     print(f"Event {event.event_type} occurred on device {event.device.uuid}")

        Parameters
        ----------
        events: EventType, str, or list of EventType or str
            The event type or list of event types to register for this device.

        Returns
        -------
        :obj:`~_device.DeviceEvents`
            An object representing the registered events.  Call
            :meth:`~_device.DeviceEvents.wait` on this object to wait for events.

        Raises
        ------
        :class:`cuda.core.system.NotSupportedError`
            None of the requested event types are registered.
        """
        return DeviceEvents(self._handle, events)

    def get_supported_event_types(self) -> list[EventType]:
        """
        Get the list of event types supported by this device.

        For Fermi™ or newer fully supported devices.  For Linux only (returns an
        empty list on Windows).

        Returns
        -------
        list[EventType]
            The list of supported event types.
        """
        cdef uint64_t[1] bitmask
        bitmask[0] = nvml.device_get_supported_event_types(self._handle)
        events = []
        for ev in _unpack_bitmask(bitmask):
            try:
                ev_enum = _EVENT_TYPE_MAPPING[1 << ev]
            except KeyError:
                raise ValueError(f"Unknown event type bit: {1 << ev}")
            events.append(ev_enum)
        return events

    ##########################################################################
    # FAN
    # See external class definitions in _fan.pxi

    def get_fan(self, fan: int = 0) -> FanInfo:
        """
        :obj:`~_device.FanInfo` object to get information and manage a specific fan on a device.
        """
        if fan < 0 or fan >= self.num_fans:
            raise ValueError(f"Fan index {fan} is out of range [0, {self.num_fans})")
        return FanInfo(self._handle, fan)

    @property
    def num_fans(self) -> int:
        """
        The number of fans on the device.
        """
        return nvml.device_get_num_fans(self._handle)

    ##########################################################################
    # FIELD VALUES
    # See external class definitions in _field_values.pxi

    def get_field_values(self, field_ids: list[int | tuple[int, int]]) -> FieldValues:
        """
        Get multiple field values from the device.

        Each value specified can raise its own exception.  That exception will
        be raised when attempting to access the corresponding ``value`` from the
        returned :obj:`~_device.FieldValues` container.

        To confirm that there are no exceptions in the entire container, call
        :meth:`~_device.FieldValues.validate`.

        Parameters
        ----------
        field_ids: list[int | tuple[int, int]]
            List of field IDs to query.

            Each item may be either a single value from the :class:`FieldId`
            enum, or a pair of (:class:`FieldId`, scope ID).

        Returns
        -------
        :obj:`~_device.FieldValues`
            Container of field values corresponding to the requested field IDs.
        """
        # Passing a field_ids array of length 0 raises an InvalidArgumentError,
        # so avoid that.
        if len(field_ids) == 0:
            return FieldValues(nvml.FieldValue(0))

        return FieldValues(nvml.device_get_field_values(self._handle, field_ids))

    def clear_field_values(self, field_ids: list[int | tuple[int, int]]) -> None:
        """
        Clear multiple field values from the device.

        Parameters
        ----------
        field_ids: list[int | tuple[int, int]]
            List of field IDs to clear.

            Each item may be either a single value from the :class:`FieldId`
            enum, or a pair of (:class:`FieldId`, scope ID).
        """
        # Passing a field_ids array of length 0 raises an InvalidArgumentError,
        # so avoid that.
        if len(field_ids) == 0:
            return

        nvml.device_clear_field_values(self._handle, field_ids)

    ##########################################################################
    # INFOROM
    # See external class definitions in _inforom.pxi

    @property
    def inforom(self) -> InforomInfo:
        """
        :obj:`~_device.InforomInfo` object with InfoROM information.

        For all products with an InfoROM.
        """
        return InforomInfo(self)

    ##########################################################################
    # MEMORY
    # See external class definitions in _memory.pxi

    @property
    def bar1_memory_info(self) -> BAR1MemoryInfo:
        """
        :obj:`~_device.BAR1MemoryInfo` object with BAR1 memory information.

        BAR1 is used to map the FB (device memory) so that it can be directly
        accessed by the CPU or by 3rd party devices (peer-to-peer on the PCIE
        bus).
        """
        return BAR1MemoryInfo(nvml.device_get_bar1_memory_info(self._handle))

    @property
    def memory_info(self) -> MemoryInfo:
        """
        :obj:`~_device.MemoryInfo` object with memory information.
        """
        return MemoryInfo(nvml.device_get_memory_info_v2(self._handle))

    ##########################################################################
    # NVLINK
    # See external class definitions in _nvlink.pxi

    def get_nvlink(self, link: int) -> NvlinkInfo:
        """
        Get :obj:`~NvlinkInfo` about this device.

        For devices with NVLink support.
        """
        if link < 0 or link >= NvlinkInfo.max_links:
            raise ValueError(f"Link index {link} is out of range [0, {NvlinkInfo.max_links})")
        return NvlinkInfo(self, link)

    ##########################################################################
    # PCI INFO
    # See external class definitions in _pci_info.pxi

    @property
    def pci_info(self) -> PciInfo:
        """
        :obj:`~_device.PciInfo` object with the PCI attributes of this device.
        """
        return PciInfo(nvml.device_get_pci_info_ext(self._handle), self._handle)

    ##########################################################################
    # PERFORMANCE
    # See external class definitions in _performance.pxi

    @property
    def performance_state(self) -> int | None:
        """
        The current performance state of the device.

        For Fermi™ or newer fully supported devices.

        Returns
        -------
        int | None
            The current performance state of the device, as an integer between 0 and 15,
            where 0 is maximum performance and higher numbers are lower performance.
            Returns `None` if the performance state is unknown.
        """
        return _pstate_to_int(nvml.device_get_performance_state(self._handle))

    @property
    def dynamic_pstates_info(self) -> GpuDynamicPstatesInfo:
        """
        :obj:`~_device.GpuDynamicPstatesInfo` object with performance monitor samples from the associated subdevice.
        """
        return GpuDynamicPstatesInfo(nvml.device_get_dynamic_pstates_info(self._handle))

    @property
    def supported_pstates(self) -> list[int]:
        """
        Get all supported Performance States (P-States) for the device.

        The returned list contains a contiguous list of valid P-States supported by
        the device.

        Return
        ------
        list[int]
            A list of supported performance state of the device, as an integer
            between 0 and 15, where 0 is maximum performance and higher numbers
            are lower performance.
        """
        # From nvml.h:
        # The returned array would contain a contiguous list of valid P-States
        # supported by the device. If the number of supported P-States is fewer
        # than the size of the array supplied missing elements would contain \a
        # NVML_PSTATE_UNKNOWN.

        pstates = []
        for pstate in nvml.device_get_supported_performance_states(self._handle):
            pstate_value = _pstate_to_int(pstate)
            if pstate_value is not None:
                pstates.append(pstate_value)
        return pstates

    ##########################################################################
    # PROCESS
    # See external class definitions in _process.pxi

    @property
    def compute_running_processes(self) -> list[ProcessInfo]:
        """
        Get information about processes with a compute context on a device

        For Fermi™ or newer fully supported devices.

        This function returns information only about compute running processes
        (e.g. CUDA application which have active context). Any graphics
        applications (e.g. using OpenGL, DirectX) won't be listed by this
        function.

        Keep in mind that information returned by this call is dynamic and the
        number of elements might change in time.

        In MIG mode, if device handle is provided, the API returns aggregate
        information, only if the caller has appropriate privileges. Per-instance
        information can be queried by using specific MIG device handles.
        Querying per-instance information using MIG device handles is not
        supported if the device is in vGPU Host virtualization mode.
        """
        return [ProcessInfo(self, proc) for proc in nvml.device_get_compute_running_processes_v3(self._handle)]

    ##########################################################################
    # REPAIR STATUS
    # See external class definitions in _repair_status.pxi

    @property
    def repair_status(self) -> RepairStatus:
        """
        :obj:`~_device.RepairStatus` object with TPC/Channel repair status.

        For Ampere™ or newer fully supported devices.
        """
        return RepairStatus(self._handle)

    ##########################################################################
    # TEMPERATURE
    # See external class definitions in _temperature.pxi

    @property
    def temperature(self) -> Temperature:
        """
        :obj:`~_device.Temperature` object with temperature information for the device.
        """
        return Temperature(self._handle)

    #######################################################################
    # TOPOLOGY

    def get_topology_nearest_gpus(self, level: GpuTopologyLevel | str) -> Iterable[Device]:
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
        try:
            level = _GPU_TOPOLOGY_LEVEL_MAPPING[level]
        except KeyError:
            raise ValueError(
                f"Invalid topology level: {level}. "
                f"Must be one of {list(GpuTopologyLevel.__members__.values())}"
            ) from None
        for handle in nvml.device_get_topology_nearest_gpus(self._handle, level):
            device = Device.__new__(Device)
            device._handle = handle
            yield device

    #######################################################################
    # UTILIZATION

    @property
    def utilization(self) -> Utilization:
        """
        Retrieves the current :obj:`~Utilization` rates for the device's major
        subsystems.

        For Fermi™ or newer fully supported devices.

        Note: During driver initialization when ECC is enabled one can see high
        GPU and Memory Utilization readings.  This is caused by ECC Memory
        Scrubbing mechanism that is performed during driver initialization.

        Note: On MIG-enabled GPUs, querying device utilization rates is not
        currently supported.

        Returns
        -------
        Utilization
            An object containing the current utilization rates for the device.
        """
        return Utilization(nvml.device_get_utilization_rates(self._handle))


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
    return _GPU_TOPOLOGY_LEVEL_INV_MAPPING[
        nvml.device_get_topology_common_ancestor(
            device1._handle,
            device2._handle,
        )
    ]


def get_p2p_status(device1: Device, device2: Device, index: GpuP2PCapsIndex | str) -> GpuP2PStatus:
    """
    Retrieve the P2P status between two devices.

    Parameters
    ----------
    device1: :class:`Device`
        The first device.
    device2: :class:`Device`
        The second device.
    index: :class:`GpuP2PCapsIndex` | str
        The P2P capability index being looked for between ``device1`` and ``device2``.

    Returns
    -------
    :class:`GpuP2PStatus`
        The P2P status between the two devices.
    """
    try:
        index_enum = _GPU_P2P_CAPS_INDEX_MAPPING[index]
    except KeyError:
        raise ValueError(
            f"Invalid P2P caps index: {index}. "
            f"Must be one of {list(GpuP2PCapsIndex.__members__.values())}"
        ) from None
    return _GPU_P2P_STATUS_MAPPING.get(
        nvml.device_get_p2p_status(
            device1._handle,
            device2._handle,
            index_enum,
        ),
        GpuP2PStatus.UNKNOWN
    )


__all__ = [
    "AddressingMode",
    "AffinityScope",
    "ClockId",
    "ClocksEventReasons",
    "ClockType",
    "CoolerControl",
    "CoolerTarget",
    "Device",
    "DeviceArch",
    "EventType",
    "FanControlPolicy",
    "FieldId",
    "get_p2p_status",
    "get_topology_common_ancestor",
    "GpuP2PCapsIndex",
    "GpuP2PStatus",
    "GpuTopologyLevel",
    "InforomObject",
    "NvlinkInfo",
    "TemperatureThresholds",
    "ThermalController",
    "ThermalTarget",
    "Utilization",
]
