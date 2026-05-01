# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0


NvlinkVersion = nvml.NvlinkVersion


cdef class NvlinkInfo:
    """
    Nvlink information for a device.
    """
    cdef Device _device
    cdef int _link

    def __init__(self, device: Device, link: int):
        self._device = device
        self._link = link

    @property
    def version(self) -> NvlinkVersion:
        """
        Retrieves the :obj:`~NvlinkVersion` for the device and link.

        For all products with NvLink support.

        Returns
        -------
        NvlinkVersion
            The Nvlink version.
        """
        return NvlinkVersion(nvml.device_get_nvlink_version(self._device._handle, self._link))

    @property
    def state(self) -> bool:
        """
        Retrieves the state of the device's Nvlink for the device and link specified.

        For Pascal™ or newer fully supported devices.

        For all products with Nvlink support.

        Returns
        -------
        bool
            `True` if the Nvlink is active.
        """
        return (
            nvml.device_get_nvlink_state(self._device._handle, self._link) == nvml.EnableState.FEATURE_ENABLED
        )

    max_links = nvml.NVLINK_MAX_LINKS
