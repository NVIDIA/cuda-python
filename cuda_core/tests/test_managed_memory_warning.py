# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""
Test that a warning is emitted when ManagedMemoryResource is created on a
platform without concurrent managed memory access.

These tests only run on affected platforms (concurrent_managed_access is False).
"""

import warnings

import cuda.bindings
import pytest
from cuda.core import Device, ManagedMemoryResource, ManagedMemoryResourceOptions
from cuda.core._memory._managed_memory_resource import reset_concurrent_access_warning

_cuda_major = int(cuda.bindings.__version__.split(".")[0])

requires_cuda_13 = pytest.mark.skipif(
    _cuda_major < 13,
    reason="ManagedMemoryResource requires CUDA 13.0 or later",
)


def _make_managed_mr(device_id):
    """Create a ManagedMemoryResource with an explicit device preference."""
    return ManagedMemoryResource(options=ManagedMemoryResourceOptions(preferred_location=device_id))


@pytest.fixture
def device_without_concurrent_managed_access(init_cuda):
    """Return a device that lacks concurrent managed access, or skip."""
    device = Device()
    device.set_current()

    if not device.properties.memory_pools_supported:
        pytest.skip("Device does not support memory pools")

    if device.properties.concurrent_managed_access:
        pytest.skip("Device supports concurrent managed access; warning not applicable")

    return device


@requires_cuda_13
def test_warning_emitted(device_without_concurrent_managed_access):
    """ManagedMemoryResource emits a warning when concurrent managed access is unsupported."""
    dev_id = device_without_concurrent_managed_access.device_id
    reset_concurrent_access_warning()

    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        mr = _make_managed_mr(dev_id)

        concurrent_warnings = [
            warning for warning in w if "concurrent managed memory access" in str(warning.message).lower()
        ]
        assert len(concurrent_warnings) == 1
        assert concurrent_warnings[0].category is UserWarning
        assert "segfault" in str(concurrent_warnings[0].message).lower()

    mr.close()


@requires_cuda_13
def test_warning_emitted_only_once(device_without_concurrent_managed_access):
    """Warning fires only once even when multiple ManagedMemoryResources are created."""
    dev_id = device_without_concurrent_managed_access.device_id
    reset_concurrent_access_warning()

    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        mr1 = _make_managed_mr(dev_id)
        mr2 = _make_managed_mr(dev_id)

        concurrent_warnings = [
            warning for warning in w if "concurrent managed memory access" in str(warning.message).lower()
        ]
        assert len(concurrent_warnings) == 1

    mr1.close()
    mr2.close()
