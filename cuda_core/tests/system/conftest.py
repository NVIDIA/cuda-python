# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0


from contextlib import contextmanager

import pytest
from cuda.core import system

skip_if_nvml_unsupported = pytest.mark.skipif(
    not system.CUDA_BINDINGS_NVML_IS_COMPATIBLE, reason="NVML support requires cuda.bindings version 12.9.6+ or 13.1.2+"
)


@contextmanager
def unsupported_before(device: system.Device, expected_device_arch: system.DeviceArch | str | None):
    device_arch = device.arch

    if isinstance(expected_device_arch, system.DeviceArch):
        expected_device_arch_int = int(expected_device_arch)
    elif expected_device_arch == "FERMI":
        expected_device_arch_int = 1
    else:
        expected_device_arch_int = 0

    if (
        expected_device_arch is None
        or expected_device_arch == "HAS_INFOROM"
        or device_arch == system.DeviceArch.UNKNOWN
    ):
        # In this case, we don't /know/ if it will fail, but we are ok if it
        # does or does not.

        # TODO: There are APIs that are documented as supported only if the
        # device has an InfoROM, but I couldn't find a way to detect that.  For now, they
        # are just handled as "possibly failing".

        try:
            yield
        except system.NotSupportedError:
            pytest.skip(f"Unsupported call for device architecture {device_arch.name} on device '{device.name}'")
    elif int(device_arch) < expected_device_arch_int:
        # In this case, we /know/ if will fail, and we want to assert that it does.
        with pytest.raises(system.NotSupportedError):
            yield
        pytest.skip("Unsupported before {expected_device_arch.name}, got {device_arch.name}")
    else:
        # In this case, we /know/ it should work, and if it fails, the test should fail.
        yield
