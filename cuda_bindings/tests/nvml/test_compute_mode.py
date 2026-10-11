# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0


import sys

import pytest

from cuda.bindings import nvml

from .conftest import unsupported_before

COMPUTE_MODES = [
    nvml.ComputeMode.COMPUTEMODE_DEFAULT,
    nvml.ComputeMode.COMPUTEMODE_PROHIBITED,
    nvml.ComputeMode.COMPUTEMODE_EXCLUSIVE_PROCESS,
]


@pytest.mark.skipif(sys.platform == "win32", reason="Test not supported on Windows")
def test_compute_mode_supported_nonroot(all_devices, subtests):
    for device in all_devices:
        device_index = nvml.device_get_index(device)
        original_compute_mode = None
        with (
            subtests.test(device_index=device_index, compute_mode_api="get_compute_mode"),
            unsupported_before(device, None),
        ):
            original_compute_mode = nvml.device_get_compute_mode(device)
        if original_compute_mode is None:
            continue

        for cm in COMPUTE_MODES:
            with subtests.test(device_index=device_index, compute_mode=cm.name):
                try:
                    nvml.device_set_compute_mode(device, cm)
                except nvml.NoPermissionError:
                    pytest.skip("Insufficient permissions to set compute mode")
                except nvml.NvmlError:
                    nvml.device_set_compute_mode(device, original_compute_mode)
                    raise
                nvml.device_set_compute_mode(device, original_compute_mode)
                assert original_compute_mode == nvml.device_get_compute_mode(device), (
                    "Compute mode shouldn't have changed"
                )
