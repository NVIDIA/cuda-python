# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Backward-compatibility checks for undocumented dict options in MR constructors."""

import pytest
from conftest import (
    create_managed_memory_resource_or_skip,
    create_pinned_memory_resource_or_xfail,
    skip_if_managed_memory_unsupported,
    skip_if_pinned_memory_unsupported,
)
from helpers.constants import POOL_SIZE

from cuda.core import Device, DeviceMemoryResource


@pytest.mark.agent_authored(model="gpt-5.3-codex")
def test_device_mr_accepts_dict_keyword(init_cuda):
    device = Device()
    if not device.properties.memory_pools_supported:
        pytest.skip("Device does not support memory pool operations")
    device.set_current()
    mr = DeviceMemoryResource(device, options={"max_size": POOL_SIZE})
    buf = mr.allocate(64, stream=device.default_stream)
    buf.close(stream=device.default_stream)
    mr.close()


@pytest.mark.agent_authored(model="gpt-5.3-codex")
def test_pinned_mr_accepts_dict_keyword(init_cuda):
    device = Device()
    skip_if_pinned_memory_unsupported(device)
    device.set_current()
    mr = create_pinned_memory_resource_or_xfail(options={"max_size": POOL_SIZE}, xfail_device=device)
    buf = mr.allocate(64, stream=device.default_stream)
    buf.close(stream=device.default_stream)
    mr.close()


@pytest.mark.agent_authored(model="gpt-5.3-codex")
def test_managed_mr_accepts_dict_keyword(init_cuda):
    device = Device()
    skip_if_managed_memory_unsupported(device)
    device.set_current()
    mr = create_managed_memory_resource_or_skip(options={})
    buf = mr.allocate(64, stream=device.default_stream)
    buf.close(stream=device.default_stream)
    mr.close()
