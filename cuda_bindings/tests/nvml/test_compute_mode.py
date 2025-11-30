# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: LicenseRef-NVIDIA-SOFTWARE-LICENSE

import pytest
from cuda.bindings import nvml

COMPUTE_MODES = [
    nvml.ComputeMode.COMPUTEMODE_DEFAULT,
    nvml.ComputeMode.COMPUTEMODE_PROHIBITED,
    nvml.ComputeMode.COMPUTEMODE_EXCLUSIVE_PROCESS,
]


def test_compute_mode_supported_nonroot(for_all_devices):
    device = for_all_devices

    try:
        original_compute_mode = nvml.device_get_compute_mode(device)
    except nvml.NotSupportedError:
        pytest.skip("nvmlDeviceGetComputeMode not supported")

    for cm in COMPUTE_MODES:
        with pytest.raises(nvml.NoPermissionError):
            nvml.device_set_compute_mode(device, cm)
        assert original_compute_mode == nvml.device_get_compute_mode(device), "Compute mode shouldn't have changed"
