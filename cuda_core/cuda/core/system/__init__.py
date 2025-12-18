# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

# ruff: noqa: F403, F405


__all__ = [
    "get_driver_version",
    "get_driver_version_full",
    "get_gpu_driver_version",
    "get_num_devices",
    "get_process_name",
]


import cuda.bindings

from .system import *

# We need both the existence of cuda.bindings._nvml and a sufficient version
# with the APIs implemented as we need them.

_BINDINGS_VERSION = tuple(int(x) for x in cuda.bindings.__version__.split("."))

_HAS_WORKING_NVML = _BINDINGS_VERSION >= (13, 1, 2) or (_BINDINGS_VERSION[0] == 12 and _BINDINGS_VERSION[1:3] >= (9, 6)) or True

if _HAS_WORKING_NVML:
    from cuda.bindings import _nvml

    from ._nvml_context import initialize
    from .device import Device, DeviceArchitecture

    initialize()

    __all__.extend(["Device", "DeviceArchitecture"])
