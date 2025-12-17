# SPDX-FileCopyrightText: Copyright (c) 2024-2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

import warnings

from cuda.core.experimental._device import Device
from cuda.core.experimental._utils.cuda_utils import driver, handle_return, runtime


class System:
    """Provide information about the cuda system.
    This class is a singleton and should not be instantiated directly.
    """

    _instance = None

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance

    def __init__(self):
        if hasattr(self, "_initialized") and self._initialized:
            return
        self._initialized = True

    def get_driver_version(self) -> tuple[int, int]:
        """
        Query the CUDA driver version.

        Returns
        -------
        tuple of int
            A 2-tuple of (major, minor) version numbers.
        """
        version = handle_return(driver.cuDriverGetVersion())
        major = version // 1000
        minor = (version % 1000) // 10
        return (major, minor)

    @property
    def driver_version(self) -> tuple[int, int]:
        """
        Query the CUDA driver version.

        Returns
        -------
        tuple of int
            A 2-tuple of (major, minor) version numbers.

        .. deprecated:: 0.5.0
          `cuda.core.system.driver_version` will be removed in 0.6.0.
          Use `cuda.core.system.get_driver_version()` instead.
        """
        warnings.warn(
            "cuda.core.system.driver_version is deprecated. Use cuda.core.system.get_driver_version() instead.",
            DeprecationWarning,
            stacklevel=1,
        )
        return self.get_driver_version()

    def get_num_devices(self) -> int:
        """
        Query the number of available GPUs.

        Returns
        -------
        int
            The number of available GPU devices.
        """
        return handle_return(runtime.cudaGetDeviceCount())

    @property
    def num_devices(self) -> int:
        """
        Query the number of available GPUs.

        Returns
        -------
        int
            The number of available GPU devices.

        .. deprecated:: 0.5.0
          `cuda.core.system.num_devices` will be removed in 0.6.0.
          Use `cuda.core.system.get_num_devices()` instead.
        """
        warnings.warn(
            "cuda.core.system.num_devices is deprecated. Use cuda.core.system.get_num_devices() instead.",
            DeprecationWarning,
            stacklevel=1,
        )
        return self.get_num_devices()

    @property
    def devices(self) -> tuple:
        """
        Query the available device instances.

        Returns
        -------
        tuple of Device
            A tuple containing instances of available devices.

        .. deprecated:: 0.5.0
          `cuda.core.system.devices` will be removed in 0.6.0.
          Use `cuda.core.Device.get_all_devices()` instead.
        """
        warnings.warn(
            "cuda.core.system.devices is deprecated. Use cuda.core.Device.get_all_devices() instead.",
            DeprecationWarning,
            stacklevel=1,
        )
        return Device.get_all_devices()
