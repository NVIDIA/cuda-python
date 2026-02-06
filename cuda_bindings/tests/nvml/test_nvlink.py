# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: LicenseRef-NVIDIA-SOFTWARE-LICENSE


import pytest
from cuda.bindings import nvml


def test_nvlink_get_link_count(all_devices):
    """
    Checks that the link count of the device is same.
    """
    for device in all_devices:
        fields = nvml.FieldValue(1)
        fields[0].field_id = nvml.FieldId.DEV_NVLINK_LINK_COUNT
        value = nvml.device_get_field_values(device, fields)[0]
        if value.nvml_return not in (nvml.Return.SUCCESS, nvml.Return.ERROR_NOT_SUPPORTED):
            pytest.skip(f"NVLink link count query unsupported (return {value.nvml_return})")

        # Use the alternative argument to device_get_field_values
        value = nvml.device_get_field_values(device, [nvml.FieldId.DEV_NVLINK_LINK_COUNT])[0]
        if value.nvml_return not in (nvml.Return.SUCCESS, nvml.Return.ERROR_NOT_SUPPORTED):
            pytest.skip(f"NVLink link count query unsupported (return {value.nvml_return})")

        # The feature_nvlink_supported detection is not robust, so we
        # can't be more specific about how many links we should find.
        if value.nvml_return == nvml.Return.SUCCESS:
            try:
                link_count = int(value.value.ui_val)
            except (TypeError, ValueError):
                pytest.skip("NVLink link count value unavailable")
            if link_count > nvml.NVLINK_MAX_LINKS:
                pytest.skip(f"NVLink link count value out of range: {link_count}")
            assert link_count <= nvml.NVLINK_MAX_LINKS, f"Unexpected link count {link_count}"
