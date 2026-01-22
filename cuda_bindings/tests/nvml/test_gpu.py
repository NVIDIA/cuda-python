# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: LicenseRef-NVIDIA-SOFTWARE-LICENSE

import pytest
from cuda.bindings import _nvml as nvml

from . import util
from .conftest import unsupported_before


def test_gpu_get_module_id(nvml_init):
    # Unique module IDs cannot exceed the number of GPUs on the system
    device_count = nvml.device_get_count_v2()

    for i in range(device_count):
        device = nvml.device_get_handle_by_index_v2(i)
        uuid = nvml.device_get_uuid(device)

        if util.is_vgpu(device):
            continue

        module_id = nvml.device_get_module_id(device)
        assert isinstance(module_id, int)


def test_gpu_get_platform_info(all_devices):
    for device in all_devices:
        if util.is_vgpu(device):
            pytest.skip(f"Not supported on vGPU device {device}")

        # Documentation says Blackwell or newer only, but this does seem to pass
        # on some newer GPUs.

        with unsupported_before(device, None):
            platform_info = nvml.device_get_platform_info(device)

        assert isinstance(platform_info, nvml.PlatformInfo_v2)
