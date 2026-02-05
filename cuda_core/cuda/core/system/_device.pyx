# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

from libc.stdint cimport intptr_t, uint64_t
from libc.math cimport ceil

from multiprocessing import cpu_count
from typing import Iterable

from cuda.bindings import nvml

from ._nvml_context cimport initialize


AddressingMode = nvml.DeviceAddressingModeType
AffinityScope = nvml.AffinityScope
BrandType = nvml.BrandType
DeviceArch = nvml.DeviceArch
GpuP2PCapsIndex = nvml.GpuP2PCapsIndex
GpuP2PStatus = nvml.GpuP2PStatus
GpuTopologyLevel = nvml.GpuTopologyLevel
Pstates = nvml.Pstates


include "_clock.pxi"
include "_cooler.pxi"
include "_device_attributes.pxi"
include "_device_utils.pxi"
include "_event.pxi"
include "_fan.pxi"
include "_field_values.pxi"
include "_inforom.pxi"
include "_memory.pxi"
include "_pci_info.pxi"
include "_performance.pxi"
include "_repair_status.pxi"
include "_temperature.pxi"


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
        prefix.  That is not included in ``cuda.core.system``.
        """
        # NVML UUIDs have a `GPU-` or `MIG-` prefix.  We remove that here.

        # TODO: If the user cares about the prefix, we will expose that in the
        # future using the MIG-related APIs in NVML.
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
        Device architecture.

        For example, a Tesla V100 will report ``DeviceArchitecture.name ==
        "VOLTA"``, and RTX A6000 will report ``DeviceArchitecture.name ==
        "AMPERE"``.
        """
        return DeviceArch(nvml.device_get_architecture(self._handle))

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
            if cuda_device.uuid == self.uuid:
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

    #########################################################################
    # ADDRESSING MODE

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
        Iterator of Device
            An iterator over available devices.
        """
        cdef Device device
        for handle in nvml.system_get_topology_gpu_set(cpu_index):
            device = Device.__new__(Device)
            device._handle = handle
            yield device

    def get_memory_affinity(self, scope: AffinityScope=AffinityScope.NODE) -> list[int]:
        """
        Retrieves a list of indices of NUMA nodes or CPU sockets with the ideal
        memory affinity for the device.

        For Kepler™ or newer fully supported devices.

        Supported on Linux only.

        If requested scope is not applicable to the target topology, the API
        will fall back to reporting the memory affinity for the immediate non-I/O
        ancestor of the device.
        """
        return _unpack_bitmask(
            nvml.device_get_memory_affinity(
                self._handle,
                <unsigned int>ceil(cpu_count() / 64),
                scope
            )
        )

    def get_cpu_affinity(self, scope: AffinityScope=AffinityScope.NODE) -> list[int]:
        """
        Retrieves a list of indices of NUMA nodes or CPU sockets with the ideal
        CPU affinity for the device.

        For Kepler™ or newer fully supported devices.

        Supported on Linux only.

        If requested scope is not applicable to the target topology, the API
        will fall back to reporting the memory affinity for the immediate non-I/O
        ancestor of the device.
        """
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

    def clock(self, clock_type: ClockType) -> ClockInfo:
        """
        Get information about and manage a specific clock on a device.
        """
        return ClockInfo(self._handle, clock_type)

    def get_auto_boosted_clocks_enabled(self) -> tuple[bool, bool]:
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

    def get_current_clock_event_reasons(self) -> list[ClocksEventReasons]:
        """
        Retrieves the current clocks event reasons.

        For all fully supported products.
        """
        cdef uint64_t[1] reasons
        reasons[0] = nvml.device_get_current_clocks_event_reasons(self._handle)
        return [ClocksEventReasons(1 << reason) for reason in _unpack_bitmask(reasons)]

    def get_supported_clock_event_reasons(self) -> list[ClocksEventReasons]:
        """
        Retrieves supported clocks event reasons that can be returned by
        :meth:`get_current_clock_event_reasons`.

        For all fully supported products.

        This method is not supported in virtual machines running virtual GPU (vGPU).
        """
        cdef uint64_t[1] reasons
        reasons[0] = nvml.device_get_supported_clocks_event_reasons(self._handle)
        return [ClocksEventReasons(1 << reason) for reason in _unpack_bitmask(reasons)]

    ##########################################################################
    # COOLER
    # See external class definitions in _cooler.pxi

    @property
    def cooler(self) -> CoolerInfo:
        """
        Get information about cooler on a device.
        """
        return CoolerInfo(nvml.device_get_cooler_info(self._handle))

    ##########################################################################
    # DEVICE ATTRIBUTES
    # See external class definitions in _device_attributes.pxi

    @property
    def attributes(self) -> DeviceAttributes:
        """
        Get various device attributes.

        For Ampere™ or newer fully supported devices.  Only available on Linux
        systems.
        """
        return DeviceAttributes(nvml.device_get_attributes_v2(self._handle))

    #########################################################################
    # DISPLAY

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

    ##########################################################################
    # EVENTS
    # See external class definitions in _event.pxi

    def register_events(self, events: EventType | int | list[EventType | int]) -> DeviceEvents:
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
        ...     EventType.EVENT_TYPE_XID_CRITICAL_ERROR,
        ... ])
        >>> while event := events.wait(timeout_ms=10000):
        ...     print(f"Event {event.event_type} occurred on device {event.device.uuid}")

        Parameters
        ----------
        events: EventType, int, or list of EventType or int
            The event type or list of event types to register for this device.

        Returns
        -------
        :class:`DeviceEvents`
            An object representing the registered events.  Call
            :meth:`DeviceEvents.wait` on this object to wait for events.

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
        return [EventType(1 << ev) for ev in _unpack_bitmask(bitmask)]

    ##########################################################################
    # FAN
    # See external class definitions in _fan.pxi

    def fan(self, fan: int = 0) -> FanInfo:
        """
        Get information and manage a specific fan on a device.
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

    ##########################################################################
    # INFOROM
    # See external class definitions in _inforom.pxi

    @property
    def inforom(self) -> InforomInfo:
        """
        Accessor for InfoROM information.

        For all products with an InfoROM.
        """
        return InforomInfo(self)

    ##########################################################################
    # MEMORY
    # See external class definitions in _memory.pxi

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
    def memory_info(self) -> MemoryInfo:
        """
        Object with memory information.
        """
        return MemoryInfo(nvml.device_get_memory_info_v2(self._handle))

    ##########################################################################
    # PCI INFO
    # See external class definitions in _pci_info.pxi

    @property
    def pci_info(self) -> PciInfo:
        """
        The PCI attributes of this device.
        """
        return PciInfo(nvml.device_get_pci_info_ext(self._handle), self._handle)

    ##########################################################################
    # PERFORMANCE
    # See external class definitions in _performance.pxi

    @property
    def performance_state(self) -> Pstates:
        """
        The current performance state of the device.

        For Fermi™ or newer fully supported devices.

        See :class:`Pstates` for possible performance states.
        """
        return Pstates(nvml.device_get_performance_state(self._handle))

    @property
    def dynamic_pstates_info(self) -> GpuDynamicPstatesInfo:
        """
        Retrieve performance monitor samples from the associated subdevice.
        """
        return GpuDynamicPstatesInfo(nvml.device_get_dynamic_pstates_info(self._handle))

    def get_supported_pstates(self) -> list[Pstates]:
        """
        Get all supported Performance States (P-States) for the device.

        The returned list contains a contiguous list of valid P-States supported by
        the device.
        """
        return [Pstates(x) for x in nvml.device_get_supported_performance_states(self._handle)]

    ##########################################################################
    # REPAIR STATUS
    # See external class definitions in _repair_status.pxi

    @property
    def repair_status(self) -> RepairStatus:
        """
        Get the repair status for TPC/Channel repair.

        For Ampere™ or newer fully supported devices.
        """
        return RepairStatus(self._handle)

    ##########################################################################
    # TEMPERATURE
    # See external class definitions in _temperature.pxi

    @property
    def temperature(self) -> Temperature:
        """
        Get information about temperatures on a device.
        """
        return Temperature(self._handle)

    #######################################################################
    # TOPOLOGY

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
    "AffinityScope",
    "BAR1MemoryInfo",
    "BrandType",
    "ClockId",
    "ClockInfo",
    "ClockOffsets",
    "ClocksEventReasons",
    "ClockType",
    "CoolerControl",
    "CoolerInfo",
    "CoolerTarget",
    "Device",
    "DeviceArch",
    "DeviceAttributes",
    "DeviceEvents",
    "EventData",
    "EventType",
    "FanControlPolicy",
    "FanInfo",
    "FieldId",
    "FieldValue",
    "FieldValues",
    "get_p2p_status",
    "get_topology_common_ancestor",
    "GpuDynamicPstatesInfo",
    "GpuDynamicPstatesUtilization",
    "GpuP2PCapsIndex",
    "GpuP2PStatus",
    "GpuTopologyLevel",
    "InforomInfo",
    "InforomObject",
    "MemoryInfo",
    "PcieUtilCounter",
    "PciInfo",
    "Pstates",
    "RepairStatus",
    "Temperature",
    "TemperatureSensors",
    "TemperatureThresholds",
    "ThermalController",
    "ThermalSensor",
    "ThermalSettings",
    "ThermalTarget",
]
