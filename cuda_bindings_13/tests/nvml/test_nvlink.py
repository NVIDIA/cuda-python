# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import pytest

from cuda.bindings import nvml


@pytest.mark.agent_authored(model="gpt-6")
def test_nvlink_get_link_count(all_devices, subtests):
    """
    Checks that the link count of the device is same.
    """
    for device in all_devices:
        fields = nvml.FieldValue(0)
        assert len(nvml.device_get_field_values(device, fields)) == 0

        assert len(nvml.device_get_field_values(device, [])) == 0

        fields = nvml.FieldValue(1)
        fields[0].field_id = nvml.FieldId.DEV_NVLINK_LINK_COUNT
        fields[0].scope_id = 0
        expected_statuses = {nvml.Return.SUCCESS, nvml.Return.ERROR_NOT_SUPPORTED}
        if nvml.device_get_name(device) == "Orin (nvgpu)":
            # nvbug 6952048: the binding reports UNKNOWN for untouched native output.
            expected_statuses.add(nvml.Return.ERROR_UNKNOWN)

        for input_kind, values in (("buffer", fields), ("field_ids", [nvml.FieldId.DEV_NVLINK_LINK_COUNT])):
            with subtests.test(device=device, input_kind=input_kind):
                value = nvml.device_get_field_values(device, values)[0]
                assert value.nvml_return in expected_statuses, (
                    f"Unexpected return {value.nvml_return} for link count field query"
                )
