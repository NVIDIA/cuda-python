# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import pytest

from cuda.bindings import nvml
from cuda.core import system


def _nvlink_count_field(count, status):
    field = nvml.FieldValue()
    field.field_id = int(nvml.FieldId.DEV_NVLINK_LINK_COUNT)
    field.nvml_return = int(status)
    field.value_type = int(nvml.ValueType.UNSIGNED_INT)
    field.value.ui_val[0] = count
    return field


@pytest.mark.thread_unsafe(reason="Temporarily replaces process-global NVML functions")
@pytest.mark.agent_authored(model="gpt-6")
def test_nvlink_zero_count_does_not_require_link_state(monkeypatch):
    def unsupported_state(_handle, _link):
        raise nvml.NotSupportedError(nvml.Return.ERROR_NOT_SUPPORTED)

    field = _nvlink_count_field(0, nvml.Return.SUCCESS)
    monkeypatch.setattr(nvml, "device_get_nvlink_state", unsupported_state)
    monkeypatch.setattr(nvml, "device_get_field_values", lambda _handle, _fields: field)
    device = system.Device.__new__(system.Device)

    assert device.get_nvlink_count() == 0
    assert list(device.get_nvlinks()) == []


@pytest.mark.thread_unsafe(reason="Temporarily replaces process-global NVML functions")
@pytest.mark.parametrize("method", ["get_nvlink_count", "get_nvlinks"])
@pytest.mark.parametrize(
    ("status_name", "exception_name"),
    [("ERROR_NOT_SUPPORTED", "NotSupportedError"), ("ERROR_UNKNOWN", "UnknownError")],
)
@pytest.mark.agent_authored(model="gpt-6")
def test_nvlink_queries_propagate_field_error(monkeypatch, method, status_name, exception_name):
    field = _nvlink_count_field(0, getattr(nvml.Return, status_name))
    monkeypatch.setattr(nvml, "device_get_field_values", lambda _handle, _fields: field)
    device = system.Device.__new__(system.Device)

    with pytest.raises(getattr(system, exception_name)):
        result = getattr(device, method)()
        if method == "get_nvlinks":
            list(result)
