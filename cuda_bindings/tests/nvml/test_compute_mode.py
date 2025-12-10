# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: LicenseRef-NVIDIA-SOFTWARE-LICENSE


import sys

import pytest

from cuda.bindings import _nvml as nvml

COMPUTE_MODES = [
    nvml.ComputeMode.COMPUTEMODE_DEFAULT,
    nvml.ComputeMode.COMPUTEMODE_PROHIBITED,
    nvml.ComputeMode.COMPUTEMODE_EXCLUSIVE_PROCESS,
]


@pytest.mark.skipif(sys.platform == "win32", reason="Test not supported on Windows")
def test_compute_mode_supported_nonroot(all_devices):
    skip_reasons = set()
    for device in all_devices:
        try:
            original_compute_mode = nvml.device_get_compute_mode(device)
        except nvml.NotSupportedError:
            skip_reasons.add(f"nvmlDeviceGetComputeMode not supported for device {device}")
            continue

        for cm in COMPUTE_MODES:
            with pytest.raises(nvml.NoPermissionError):
                nvml.device_set_compute_mode(device, cm)
            assert original_compute_mode == nvml.device_get_compute_mode(device), "Compute mode shouldn't have changed"

    if skip_reasons:
        pytest.skip(" ; ".join(skip_reasons))
