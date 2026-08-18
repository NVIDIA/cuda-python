# SPDX-FileCopyrightText: Copyright (c) 2024-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Helpers for tests that need CUDA device and memory-resource setup."""

from contextlib import contextmanager

import pytest
from cuda_python_test_helpers.mempool import xfail_if_mempool_oom

from cuda.core import (
    ManagedMemoryResource,
    PinnedMemoryResource,
)
from cuda.core._utils.cuda_utils import CUDAError


def _device_id_from_resource_options(device, args, kwargs):
    if device is not None:
        return device
    options = kwargs.get("options")
    if options is None and args:
        options = args[0]
    if options is None:
        return 0
    if isinstance(options, dict):
        preferred_location = options.get("preferred_location")
        preferred_location_type = options.get("preferred_location_type")
    else:
        preferred_location = getattr(options, "preferred_location", None)
        preferred_location_type = getattr(options, "preferred_location_type", None)
    if preferred_location_type in (None, "device") and isinstance(preferred_location, int) and preferred_location >= 0:
        return preferred_location
    return 0


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
    except CUDAError as e:
        xfail_if_mempool_oom(e, device)
        raise
    except RuntimeError as e:
        if "requires CUDA 13.0" in str(e):
            pytest.skip("ManagedMemoryResource requires CUDA 13.0 or later")
        raise


def create_managed_memory_resource_or_skip(*args, xfail_device=None, **kwargs):
    # Keep the established "skip" helper name for call-site readability, even though
    # Windows MCDM mempool OOM setup failures are xfailed instead of skipped.
    try:
        return ManagedMemoryResource(*args, **kwargs)
    except CUDAError as e:
        xfail_if_mempool_oom(e, _device_id_from_resource_options(xfail_device, args, kwargs))
        if "CUDA_ERROR_NOT_SUPPORTED" in str(e):
            pytest.skip("ManagedMemoryResource is not supported on this platform/device")
        raise
    except RuntimeError as e:
        if "requires CUDA 13.0" in str(e):
            pytest.skip("ManagedMemoryResource requires CUDA 13.0 or later")
        raise


def create_pinned_memory_resource_or_xfail(*args, xfail_device=None, **kwargs):
    try:
        return PinnedMemoryResource(*args, **kwargs)
    except CUDAError as e:
        xfail_if_mempool_oom(e, xfail_device)
        raise


@contextmanager
def xfail_on_graph_mempool_oom(device=0):
    try:
        yield
    except CUDAError as e:
        xfail_if_mempool_oom(e, "cuGraphAddMemAllocNode", device)
        raise
