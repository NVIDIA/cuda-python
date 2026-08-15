# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from contextlib import contextmanager
from functools import cache

import pytest


@cache
def hardware_supports_nvml():
    """Try the simplest NVML API to verify basic functionality.

    Returns False on platforms where NVML is unsupported (e.g. Jetson Orin).
    """
    from cuda.bindings import nvml
    from cuda.bindings._internal.utils import FunctionNotFoundError as NvmlSymbolNotFoundError  # noqa: F401

    nvml.init_v2()
    try:
        nvml.system_get_driver_branch()
    except (nvml.NotSupportedError, nvml.UnknownError):
        return False
    else:
        return True
    finally:
        nvml.shutdown()


def _should_skip_nvml_tests() -> bool:
    """Return True if NVML tests should be skipped on this system.

    Checks cuda.core's compatibility gate first (if cuda.core is installed),
    then falls back to a hardware-level NVML probe.
    """
    try:
        from cuda.core import system

        if not system.CUDA_BINDINGS_NVML_IS_COMPATIBLE:
            return True
    except ImportError:
        pass  # cuda.core not installed; skip the compat gate
    return not hardware_supports_nvml()


skip_if_nvml_unsupported = pytest.mark.skipif(
    _should_skip_nvml_tests(),
    reason="NVML support requires cuda.bindings version 12.9.6+ for CUDA 12.x or 13.2.0+ for CUDA 13.x, and hardware that supports NVML",
)


@contextmanager
def unsupported_before(device, expected_device_arch):
    """Context manager that skips or xfails when an NVML API is not supported on this device.

    ``device`` may be a raw NVML device handle (int) or any object that exposes
    the handle via a ``._handle`` attribute (e.g. ``cuda.core.system.Device``).
    """
    from cuda.bindings import nvml
    from cuda.bindings._internal.utils import FunctionNotFoundError as NvmlSymbolNotFoundError

    handle = getattr(device, "_handle", device)
    device_arch = nvml.device_get_architecture(handle)

    if isinstance(expected_device_arch, nvml.DeviceArch):
        expected_device_arch_int = int(expected_device_arch)
    elif expected_device_arch == "FERMI":
        expected_device_arch_int = 1
    else:
        expected_device_arch_int = 0

    if expected_device_arch is None or expected_device_arch == "HAS_INFOROM" or device_arch == nvml.DeviceArch.UNKNOWN:
        # We don't know if it will fail, so we tolerate either outcome.
        #
        # TODO: There are APIs that are documented as supported only if the
        # device has an InfoROM, but I couldn't find a way to detect that.  For
        # now, they are just handled as "possibly failing".
        try:
            yield
        except (nvml.NotSupportedError, nvml.FunctionNotFoundError, NvmlSymbolNotFoundError):
            pytest.skip(
                f"Unsupported call for device architecture {nvml.DeviceArch(device_arch).name} "
                f"on device '{nvml.device_get_name(handle)}'"
            )
    elif int(device_arch) < expected_device_arch_int:
        # We know it will fail; assert that it does.
        with pytest.raises(nvml.NotSupportedError):
            yield
        pytest.skip(f"Unsupported before {expected_device_arch.name}, got {nvml.device_get_name(handle)}")
    else:
        yield
