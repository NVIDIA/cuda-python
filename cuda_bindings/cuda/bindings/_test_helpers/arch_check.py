# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: LicenseRef-NVIDIA-SOFTWARE-LICENSE


from contextlib import contextmanager

import pytest
from cuda.bindings import _nvml as nvml


@contextmanager
def unsupported_before(device: int, expected_device_arch: nvml.DeviceArch | str | None):
    device_arch = nvml.device_get_architecture(device)

    if isinstance(expected_device_arch, nvml.DeviceArch):
        expected_device_arch_int = int(expected_device_arch)
    elif expected_device_arch == "FERMI":
        expected_device_arch_int = 1
    else:
        expected_device_arch_int = 0

    if expected_device_arch is None or expected_device_arch == "HAS_INFOROM" or device_arch == nvml.DeviceArch.UNKNOWN:
        # In this case, we don't /know/ if it will fail, but we are ok if it
        # does or does not.

        # TODO: There are APIs that are documented as supported only if the
        # device has an InfoROM, but I couldn't find a way to detect that.  For
        # now, they are just handled as "possibly failing".

        try:
            yield
        except nvml.NotSupportedError:
            # The API call raised NotSupportedError, so we skip the test, but
            # don't fail it
            pytest.skip(
                f"Unsupported call for device architecture {nvml.DeviceArch(device_arch).name} "
                f"on device '{nvml.device_get_name(device)}'"
            )
        # If the API call worked, just continue
    elif int(device_arch) < expected_device_arch_int:
        # In this case, we /know/ if will fail, and we want to assert that it does.
        with pytest.raises(nvml.NotSupportedError):
            yield
        # The above call was unsupported, so the rest of the test is skipped
        pytest.skip(f"Unsupported before {expected_device_arch.name}, got {nvml.device_get_name(device)}")
    else:
        # In this case, we /know/ it should work, and if it fails, the test should fail.
        yield
