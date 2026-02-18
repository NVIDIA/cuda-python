# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: LicenseRef-NVIDIA-SOFTWARE-LICENSE


from cuda.bindings import nvml


def test_nvlink_get_link_count(all_devices):
    """
    Checks that the link count of the device is same.
    """
    for device in all_devices:
        fields = nvml.FieldValue(1)
        fields[0].field_id = nvml.FieldId.DEV_NVLINK_LINK_COUNT
        value = nvml.device_get_field_values(device, fields)[0]
        assert value.nvml_return == nvml.Return.SUCCESS or value.nvml_return == nvml.Return.ERROR_NOT_SUPPORTED, (
            f"Unexpected return {value.nvml_return} for link count field query"
        )

        # Use the alternative argument to device_get_field_values
        value = nvml.device_get_field_values(device, [nvml.FieldId.DEV_NVLINK_LINK_COUNT])[0]
        assert value.nvml_return == nvml.Return.SUCCESS or value.nvml_return == nvml.Return.ERROR_NOT_SUPPORTED, (
            f"Unexpected return {value.nvml_return} for link count field query"
        )

        # The feature_nvlink_supported detection is not robust, so we
        # can't be more specific about how many links we should find.
        if value.nvml_return == nvml.Return.SUCCESS:
            assert value.value.ui_val <= nvml.NVLINK_MAX_LINKS, f"Unexpected link count {value.value.ui_val}"
