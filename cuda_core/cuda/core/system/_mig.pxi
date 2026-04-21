# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0


from typing import Iterable


cdef class MigInfo:
    cdef Device _device

    def __init__(self, device: Device):
        self._device = device

    @property
    def is_mig_device(self) -> bool:
        """
        Whether this device is a MIG (Multi-Instance GPU) device.

        A MIG device handle is an NVML abstraction which maps to a MIG compute
        instance.  These overloaded references can be used (with some
        restrictions) interchangeably with a GPU device handle to execute
        queries at a per-compute instance granularity.

        For Ampere™ or newer fully supported devices.
        """
        return bool(nvml.device_is_mig_device_handle(self._device._handle))

    @property
    def mode(self) -> bool:
        """
        Get current MIG mode for the device.

        For Ampere™ or newer fully supported devices.

        Changing MIG modes may require device unbind or reset. The "pending" MIG
        mode refers to the target mode following the next activation trigger.

        Returns
        -------
        bool
            `True` if current MIG mode is enabled.
        """
        current, _ = nvml.device_get_mig_mode(self._device._handle)
        return current == nvml.EnableState.FEATURE_ENABLED

    @mode.setter
    def mode(self, mode: bool):
        """
        Set the MIG mode for the device.

        For Ampere™ or newer fully supported devices.

        Changing MIG modes may require device unbind or reset. The "pending" MIG
        mode refers to the target mode following the next activation trigger.

        Parameters
        ----------
        mode: bool
            `True` to enable MIG mode, `False` to disable MIG mode.
        """
        nvml.device_set_mig_mode(
            self._device._handle,
            nvml.EnableState.FEATURE_ENABLED if mode else nvml.EnableState.FEATURE_DISABLED
        )

    @property
    def pending_mode(self) -> bool:
        """
        Get pending MIG mode for the device.

        For Ampere™ or newer fully supported devices.

        Changing MIG modes may require device unbind or reset. The "pending" MIG
        mode refers to the target mode following the next activation trigger.

        If the device is not a MIG device, returns `False`.

        Returns
        -------
        bool
            `True` if pending MIG mode is enabled.
        """
        _, pending = nvml.device_get_mig_mode(self._device._handle)
        return pending == nvml.EnableState.FEATURE_ENABLED

    @property
    def device_count(self) -> int:
        """
        Get the maximum number of MIG devices that can exist under this device.

        Returns zero if MIG is not supported or enabled.

        For Ampere™ or newer fully supported devices.

        Returns
        -------
        int
            The number of MIG devices (compute instances) on this GPU.
        """
        return nvml.device_get_max_mig_device_count(self._device._handle)

    @property
    def parent(self) -> Device:
        """
        For MIG devices, get the parent GPU device.

        For Ampere™ or newer fully supported devices.

        Returns
        -------
        Device
            The parent GPU device for this MIG device.
        """
        parent_handle = nvml.device_get_handle_from_mig_device_handle(self._device._handle)
        parent_device = Device.__new__(Device)
        parent_device._handle = parent_handle
        return parent_device

    def get_device_by_index(self, index: int) -> Device:
        """
        Get MIG device for the given index under its parent device.

        If the compute instance is destroyed either explicitly or by destroying,
        resetting or unbinding the parent GPU instance or the GPU device itself
        the MIG device handle would remain invalid and must be requested again
        using this API. Handles may be reused and their properties can change in
        the process.

        For Ampere™ or newer fully supported devices.

        Parameters
        ----------
        index: int
            The index of the MIG device (compute instance) to retrieve.  Must be
            between 0 and the value returned by `device_count - 1`.

        Returns
        -------
        Device
            The MIG device corresponding to the given index.
        """
        mig_device_handle = nvml.device_get_mig_device_handle_by_index(self._device._handle, index)
        mig_device = Device.__new__(Device)
        mig_device._handle = mig_device_handle
        return mig_device

    def get_all_devices(self) -> Iterable[Device]:
        """
        Get all MIG devices under its parent device.

        If the compute instance is destroyed either explicitly or by destroying,
        resetting or unbinding the parent GPU instance or the GPU device itself
        the MIG device handle would remain invalid and must be requested again
        using this API. Handles may be reused and their properties can change in
        the process.

        For Ampere™ or newer fully supported devices.

        Returns
        -------
        list[Device]
            A list of all MIG devices corresponding to this GPU.
        """
        for i in range(self.device_count):
            yield self.get_device_by_index(i)
