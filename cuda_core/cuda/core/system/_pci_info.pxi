# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0


PcieUtilCounter = nvml.PcieUtilCounter


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
