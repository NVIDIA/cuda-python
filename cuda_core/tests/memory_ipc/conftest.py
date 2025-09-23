# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import pytest

from cuda.core.experimental import Device, DeviceMemoryResource

POOL_SIZE = 2097152


@pytest.fixture(scope="function")
def device():
    """Obtains a device suitable for IPC-enabled mempool tests, or skips."""
    # Check if IPC is supported on this platform/device
    device = Device()
    device.set_current()

    if not device.properties.memory_pools_supported:
        pytest.skip("Device does not support mempool operations")

    # Note: Linux specific. Once Windows support for IPC is implemented, this
    # test should be updated.
    if not device.properties.handle_type_posix_file_descriptor_supported:
        pytest.skip("Device does not support IPC")

    return device


@pytest.fixture(scope="function")
def ipc_memory_resource(device):
    mr = DeviceMemoryResource(device, dict(max_size=POOL_SIZE, ipc_enabled=True))
    assert mr.is_ipc_enabled
    return mr
