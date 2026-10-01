# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import pytest

from cuda.core import system


def _nvlink_count_field(nvml, count):
    field = nvml.FieldValue()
    field.field_id = int(nvml.FieldId.DEV_NVLINK_LINK_COUNT)
    field.nvml_return = int(nvml.Return.SUCCESS)
    field.value_type = int(nvml.ValueType.UNSIGNED_INT)
    field.value.ui_val[0] = count
    return field


@pytest.mark.skipif(not system.CUDA_BINDINGS_NVML_IS_COMPATIBLE, reason="Compatible NVML bindings are required")
@pytest.mark.thread_unsafe(reason="Temporarily replaces process-global NVML functions")
@pytest.mark.parametrize("method", ["get_nvlink_count", "get_nvlinks"])
@pytest.mark.agent_authored(model="gpt-6")
def test_nvlink_queries_propagate_not_supported(monkeypatch, method):
    from cuda.bindings import nvml

    def unsupported_state(_handle, _link):
        raise nvml.NotSupportedError(nvml.Return.ERROR_NOT_SUPPORTED)

    # A successful field result alone does not establish NVLink support.
    field = _nvlink_count_field(nvml, 0)
    monkeypatch.setattr(nvml, "device_get_nvlink_state", unsupported_state)
    monkeypatch.setattr(nvml, "device_get_field_values", lambda _handle, _fields: field)
    device = system.Device.__new__(system.Device)

    with pytest.raises(system.NotSupportedError):
        result = getattr(device, method)()
        if method == "get_nvlinks":
            list(result)


@pytest.mark.skipif(not system.CUDA_BINDINGS_NVML_IS_COMPATIBLE, reason="Compatible NVML bindings are required")
@pytest.mark.thread_unsafe(reason="Temporarily replaces process-global NVML functions")
@pytest.mark.parametrize(("state", "count"), [(False, 3), (True, 3), (None, 0)])
@pytest.mark.agent_authored(model="gpt-6")
def test_nvlink_count_handles_disabled_or_absent_link(monkeypatch, state, count):
    from cuda.bindings import nvml

    def link_state(_handle, _link):
        if state is None:
            raise nvml.InvalidArgumentError(nvml.Return.ERROR_INVALID_ARGUMENT)
        return state

    field = _nvlink_count_field(nvml, count)
    monkeypatch.setattr(nvml, "device_get_nvlink_state", link_state)
    monkeypatch.setattr(nvml, "device_get_field_values", lambda _handle, _fields: field)
    device = system.Device.__new__(system.Device)

    assert device.get_nvlink_count() == count
