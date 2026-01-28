# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: LicenseRef-NVIDIA-SOFTWARE-LICENSE


import sys

import pytest

from cuda.bindings import _nvml as nvml

from .conftest import unsupported_before

COMPUTE_MODES = [
    nvml.ComputeMode.COMPUTEMODE_DEFAULT,
    nvml.ComputeMode.COMPUTEMODE_PROHIBITED,
    nvml.ComputeMode.COMPUTEMODE_EXCLUSIVE_PROCESS,
]


@pytest.mark.skipif(sys.platform == "win32", reason="Test not supported on Windows")
def test_compute_mode_supported_nonroot(all_devices):
    for device in all_devices:
        with unsupported_before(device, None):
            original_compute_mode = nvml.device_get_compute_mode(device)

        for cm in COMPUTE_MODES:
            with pytest.raises(nvml.NoPermissionError):
                nvml.device_set_compute_mode(device, cm)
            assert original_compute_mode == nvml.device_get_compute_mode(device), "Compute mode shouldn't have changed"
