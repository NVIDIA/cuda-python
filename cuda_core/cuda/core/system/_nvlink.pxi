# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0


_NVLINK_VERSION_MAPPING = {
    nvml.NvlinkVersion.VERSION_1_0: (1, 0),
    nvml.NvlinkVersion.VERSION_2_0: (2, 0),
    nvml.NvlinkVersion.VERSION_2_2: (2, 2),
    nvml.NvlinkVersion.VERSION_3_0: (3, 0),
    nvml.NvlinkVersion.VERSION_3_1: (3, 1),
    nvml.NvlinkVersion.VERSION_4_0: (4, 0),
    nvml.NvlinkVersion.VERSION_5_0: (5, 0),
}


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
    def version(self) -> tuple[int, int]:
        """
        Retrieves the NvLink version for the device and link.

        For all products with NvLink support.

        Returns
        -------
        tuple[int, int]
            The Nvlink version as a tuple of (major, minor).
        """
        version = nvml.device_get_nvlink_version(self._device._handle, self._link)
        if version == nvml.NvlinkVersion.VERSION_INVALID:
            raise RuntimeError(f"Invalid NvLink version returned for device")
        try:
            return _NVLINK_VERSION_MAPPING[version]
        except KeyError:
            raise RuntimeError(f"Unknown NvLink version {version} returned for device") from None

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
