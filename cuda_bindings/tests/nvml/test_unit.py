# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import pytest

from cuda.bindings import nvml


@pytest.fixture
def unit_handles(nvml_init):
    """Yield all unit handles, or skip if no units are present."""
    count = nvml.unit_get_count()
    if count == 0:
        pytest.skip("No NVML units present on this system")
    return [nvml.unit_get_handle_by_index(i) for i in range(count)]


@pytest.mark.agent_authored(model="claude-sonnet-4-6")
def test_unit_get_devices_returns_array(unit_handles):
    """Regression test: unit_get_devices must not crash due to double-pointer cast.

    Previously nvmlUnitGetDevices was called with <nvmlUnit_t *>unit instead of
    <nvmlUnit_t>unit, causing undefined behaviour because nvmlUnit_t is itself an
    opaque pointer handle.
    """
    for unit in unit_handles:
        devices = nvml.unit_get_devices(unit)
        # Result is a memoryview / cython array of intptr_t device handles.
        assert hasattr(devices, "__len__")
        # Each element must be a non-zero pointer (valid device handle).
        for dev in devices:
            assert dev != 0
