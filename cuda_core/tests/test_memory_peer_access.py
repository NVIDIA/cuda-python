# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import cuda.core
import pytest
from cuda.core import Device

NBYTES = 1024


def _mempool_device_impl(num):
    num_devices = len(cuda.core.system.devices)
    if num_devices < num:
        pytest.skip("Test requires at least {num} GPUs")

    devs = [Device(i) for i in range(num)]
    for i in reversed(range(num)):
        devs[i].set_current()

    if not all(devs[i].can_access_peer(j) for i in range(num) for j in range(num)):
        pytest.skip("Test requires GPUs with peer access")

    if not all(devs[i].properties.memory_pools_supported for i in range(num)):
        pytest.skip("Device does not support mempool operations")

    return devs


@pytest.fixture
def mempool_device_x2():
    """Fixture that provides two devices if available, otherwise skips test."""
    return _mempool_device_impl(2)


@pytest.fixture
def mempool_device_x3():
    """Fixture that provides three devices if available, otherwise skips test."""
    return _mempool_device_impl(3)
