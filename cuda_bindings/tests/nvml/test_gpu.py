# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: LicenseRef-NVIDIA-SOFTWARE-LICENSE

import pytest
from cuda.bindings import _nvml as nvml

from . import util


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
    skip_reasons = set()
    for device in all_devices:
        if util.is_vgpu(device):
            skip_reasons.add(f"Not supported on vGPU device {device}")
            continue

        # TODO
        # if device.feature_dict.board.chip < board_class.Architecture.Blackwell:
        #     test_utils.skip_test("Not supported on chip before Blackwell")

        try:
            platform_info = nvml.device_get_platform_info(device)
        except nvml.NotSupportedError:
            skip_reasons.add(f"Not supported returned, linkely NVLink is disable for {device}")
            continue

        assert isinstance(platform_info, nvml.PlatformInfo_v2)

    if skip_reasons:
        pytest.skip(" ; ".join(skip_reasons))
