# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Per-directory conftest for the ``copy_batch`` test modules.

Provides the device, stream and buffer fixtures shared by
``test_copy_batch.py`` (data movement) and ``test_copy_batch_options.py``
(options and validation).
"""

import pytest
from helpers.copy_batch import COPY_BATCH_COUNT, COPY_BATCH_SIZE

from cuda.core import Device, LegacyPinnedMemoryResource
from cuda.core._memory._copy_ops import _batch_entry_point_in_use


@pytest.fixture
def copy_batch_device(init_cuda):
    """``copy_batch`` works on every supported toolkit, so this never skips."""
    device = Device()
    device.set_current()
    return device


@pytest.fixture
def requires_copy_options():
    """Skip when ``CopyOptions`` cannot reach the driver.

    Options travel only through ``cuMemcpyBatchAsync``. Where that is
    unavailable, ``copy_batch`` falls back to per-copy ``cuMemcpyAsync``,
    which cannot convey attributes, so non-default options are rejected.
    """
    if not _batch_entry_point_in_use():
        pytest.skip(
            "non-default CopyOptions requires cuda.core built against CUDA 13, "
            "cuda.bindings 13.0+, and a driver reporting CUDA 13.0 or newer"
        )


@pytest.fixture
def copy_stream(copy_batch_device):
    """The single stream used for both allocation and copies in a test.

    Stream-ordered pool allocations are only guaranteed usable on the
    stream that allocated them, so tests allocate and copy on this one
    stream rather than mixing it with ``device.default_stream``.
    """
    s = copy_batch_device.create_stream()
    yield s
    s.close()


@pytest.fixture
def h2d_bufs(copy_batch_device, copy_stream):
    """Pinned-host source / device destination pairs."""
    pinned_mr = LegacyPinnedMemoryResource()
    device_mr = copy_batch_device.memory_resource

    srcs = [pinned_mr.allocate(COPY_BATCH_SIZE) for _ in range(COPY_BATCH_COUNT)]
    dsts = [device_mr.allocate(COPY_BATCH_SIZE, stream=copy_stream) for _ in range(COPY_BATCH_COUNT)]

    yield srcs, dsts

    for buf in srcs + dsts:
        buf.close(copy_stream)
    copy_stream.sync()


@pytest.fixture
def device_bufs(copy_batch_device, copy_stream):
    """Device source / device destination pairs."""
    device_mr = copy_batch_device.memory_resource

    srcs = [device_mr.allocate(COPY_BATCH_SIZE, stream=copy_stream) for _ in range(COPY_BATCH_COUNT)]
    dsts = [device_mr.allocate(COPY_BATCH_SIZE, stream=copy_stream) for _ in range(COPY_BATCH_COUNT)]

    yield srcs, dsts

    for buf in srcs + dsts:
        buf.close(copy_stream)
    copy_stream.sync()
