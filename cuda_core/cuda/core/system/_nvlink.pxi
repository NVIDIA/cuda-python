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
    def version(self) -> NvLinkVersion:
        """
        Retrieves the :obj:`~NvLinkVersion` for the device and link.

        For all products with NvLink support.

        Returns
        -------
        NvLinkVersion
            The NvLink version.
        """
        return NvlinkVersion(nvml.device_get_nvlink_version(self._device._handle, self._link))

    @property
    def state(self) -> bool:
        """
        Retrieves the state of the device's NvLink for the device and link specified.

        For Pascal™ or newer fully supported devices.

        For all products with NvLink support.

        Returns
        -------
        bool
            `True` if the NvLink is active.
        """
        return (
            nvml.device_get_nvlink_state(self._device._handle, self._link) == nvml.EnableState.FEATURE_ENABLED
        )

    max_links = nvml.NVLINK_MAX_LINKS
