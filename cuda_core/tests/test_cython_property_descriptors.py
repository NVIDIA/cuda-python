# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

import inspect
import types

import pytest

from cuda.core import (
    Buffer,
    Context,
    DeviceMemoryResource,
    DeviceResources,
    Event,
    PinnedMemoryResource,
)
from cuda.core.system import CUDA_BINDINGS_NVML_IS_COMPATIBLE

_PYTHON_PROPERTIES = [
    pytest.param(DeviceMemoryResource, "allocation_handle", id="DeviceMemoryResource.allocation_handle"),
    pytest.param(PinnedMemoryResource, "allocation_handle", id="PinnedMemoryResource.allocation_handle"),
    pytest.param(Event, "ipc_descriptor", id="Event.ipc_descriptor"),
    pytest.param(Buffer, "ipc_descriptor", id="Buffer.ipc_descriptor"),
    pytest.param(Context, "resources", id="Context.resources"),
    pytest.param(DeviceResources, "workqueue", id="DeviceResources.workqueue"),
]

if CUDA_BINDINGS_NVML_IS_COMPATIBLE:
    from cuda.core.system import Device as SystemDevice

    _PYTHON_PROPERTIES.extend(
        [
            pytest.param(SystemDevice, "attributes", id="system.Device.attributes"),
            pytest.param(
                SystemDevice,
                "is_auto_boosted_clocks_enabled",
                id="system.Device.is_auto_boosted_clocks_enabled",
            ),
            pytest.param(SystemDevice, "is_c2c_enabled", id="system.Device.is_c2c_enabled"),
            pytest.param(SystemDevice, "numa_node_id", id="system.Device.numa_node_id"),
            pytest.param(SystemDevice, "current_clock_event_reasons", id="system.Device.current_clock_event_reasons"),
            pytest.param(
                SystemDevice,
                "supported_clock_event_reasons",
                id="system.Device.supported_clock_event_reasons",
            ),
        ]
    )


@pytest.mark.parametrize("cls, name", _PYTHON_PROPERTIES)
def test_known_public_cython_properties_are_python_properties(cls, name):
    descriptor = inspect.getattr_static(cls, name)

    assert isinstance(descriptor, property)
    assert not isinstance(descriptor, types.GetSetDescriptorType)
