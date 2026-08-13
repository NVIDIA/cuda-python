# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0


import pytest

from cuda.bindings import nvml

from . import util


class _FakeFieldValue:
    nvml_return = nvml.Return.SUCCESS


@pytest.mark.agent_authored(model="claude-opus-5")
def test_supports_nvlink_queries_a_real_field_id(monkeypatch):
    """The helper has to name an enum that exists; nvml.FI never did."""
    queried = {}

    def fake_device_get_field_values(device, fields):
        queried["field_id"] = fields[0].field_id
        return [_FakeFieldValue()]

    monkeypatch.setattr(nvml, "device_get_field_values", fake_device_get_field_values)

    assert util.supports_nvlink(object()) is True
    assert queried["field_id"] == nvml.FieldId.DEV_NVLINK_GET_STATE
