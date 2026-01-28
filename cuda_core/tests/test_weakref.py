# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

import weakref

import pytest
from cuda.core import Device


@pytest.fixture(scope="module")
def device():
    dev = Device()
    dev.set_current()
    return dev


@pytest.fixture
def stream(device):
    return device.create_stream()


@pytest.fixture
def event(device):
    return device.create_event()


@pytest.fixture
def context(device):
    return device.context


@pytest.fixture
def buffer(device):
    return device.allocate(1024)


@pytest.fixture
def launch_config():
    from cuda.core import LaunchConfig

    return LaunchConfig(grid=(1,), block=(1,))


@pytest.fixture
def object_code():
    from cuda.core import Program

    prog = Program('extern "C" __global__ void test_kernel() {}', "c++")
    return prog.compile("cubin")


@pytest.fixture
def kernel(object_code):
    return object_code.get_kernel("test_kernel")


WEAK_REFERENCEABLE = [
    "device",
    "stream",
    "event",
    "context",
    "buffer",
    "launch_config",
    "object_code",
    "kernel",
]


@pytest.mark.parametrize("fixture_name", WEAK_REFERENCEABLE)
def test_weakref(fixture_name, request):
    """Core API classes should be weak-referenceable."""
    obj = request.getfixturevalue(fixture_name)
    ref = weakref.ref(obj)
    assert ref() is obj
