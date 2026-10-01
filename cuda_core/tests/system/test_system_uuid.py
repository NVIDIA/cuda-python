# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import pytest

from cuda.core import system


@pytest.mark.skipif(not system.CUDA_BINDINGS_NVML_IS_COMPATIBLE, reason="Compatible NVML bindings are required")
@pytest.mark.thread_unsafe(reason="Temporarily replaces a process-global NVML function")
@pytest.mark.parametrize("prefix", ["GPU-", "MIG-", "DLA-", ""])
@pytest.mark.agent_authored(model="gpt-6")
def test_device_uuid_preserves_unprefixed_value(monkeypatch, prefix):
    from cuda.bindings import nvml

    expected_uuid = "abcdef12-abcd-0123-4567-1234567890ab"
    raw_uuid = prefix + expected_uuid
    monkeypatch.setattr(nvml, "device_get_uuid", lambda _handle: raw_uuid)
    device = system.Device.__new__(system.Device)

    assert device.uuid == raw_uuid
    assert device.uuid_without_prefix == expected_uuid


@pytest.mark.skipif(not system.CUDA_BINDINGS_NVML_IS_COMPATIBLE, reason="Compatible NVML bindings are required")
@pytest.mark.thread_unsafe(reason="Temporarily replaces a process-global NVML function")
@pytest.mark.parametrize(
    ("exception_name", "status_name"),
    [("NotFoundError", "ERROR_NOT_FOUND"), ("NotSupportedError", "ERROR_NOT_SUPPORTED")],
)
@pytest.mark.agent_authored(model="gpt-6")
def test_to_system_device_propagates_uuid_lookup_error(init_cuda, monkeypatch, exception_name, status_name):
    from cuda.bindings import nvml

    def unavailable_lookup(_uuid):
        raise getattr(nvml, exception_name)(getattr(nvml.Return, status_name))

    monkeypatch.setattr(nvml, "device_get_handle_by_uuid", unavailable_lookup)

    with pytest.raises(getattr(system, exception_name)):
        init_cuda.to_system_device()
