# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Memory-related test helpers (skip guards and resource factories)."""

import pytest

from cuda.core import Device, ManagedMemoryResource, PinnedMemoryResource
from cuda.core._utils.cuda_utils import CUDAError


def skip_if_pinned_memory_unsupported(device):
    try:
        if not device.properties.host_memory_pools_supported:
            pytest.skip("Device does not support host mempool operations")
    except AttributeError:
        pytest.skip("PinnedMemoryResource requires CUDA 13.0 or later")


def skip_if_managed_memory_unsupported(device):
    try:
        if not device.properties.memory_pools_supported or not device.properties.concurrent_managed_access:
            pytest.skip("Device does not support managed memory pool operations")
    except AttributeError:
        pytest.skip("ManagedMemoryResource requires CUDA 13.0 or later")
    try:
        ManagedMemoryResource()
    except RuntimeError as e:
        if "requires CUDA 13.0" in str(e):
            pytest.skip("ManagedMemoryResource requires CUDA 13.0 or later")
        raise


def create_managed_memory_resource_or_skip(*args, xfail_device=None, **kwargs):
    # Keep the established "skip" helper name for call-site readability.
    if not args and kwargs.get("options") is None and not Device().properties.concurrent_managed_access:
        # Without options this looks up the device's default managed pool, which
        # supports managed allocations only with concurrent managed access. Check
        # the property instead of the lookup's error: the driver returns
        # CUDA_ERROR_NOT_SUPPORTED on some machines and CUDA_ERROR_OUT_OF_MEMORY
        # on others. Dedicated pools (options given) work without it.
        pytest.skip("Device does not support concurrent managed memory access")
    try:
        return ManagedMemoryResource(*args, **kwargs)
    except CUDAError as e:
        if "CUDA_ERROR_NOT_SUPPORTED" in str(e):
            pytest.skip("ManagedMemoryResource is not supported on this platform/device")
        raise
    except RuntimeError as e:
        if "requires CUDA 13.0" in str(e):
            pytest.skip("ManagedMemoryResource requires CUDA 13.0 or later")
        if "concurrent managed access is not available" in str(e).lower():
            pytest.skip("Device does not support concurrent managed memory access")
        raise


def create_pinned_memory_resource_or_xfail(*args, xfail_device=None, **kwargs):
    return PinnedMemoryResource(*args, **kwargs)
