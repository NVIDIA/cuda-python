# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

"""Shared constants and helpers for the ``copy_batch`` tests.

Fixtures live in ``tests/memory/conftest.py``; this module holds the
pieces that tests import by name.
"""

from cuda.core import LegacyPinnedMemoryResource
from helpers.buffers import compare_equal_buffers, make_scratch_buffer

COPY_BATCH_SIZE = 4096
COPY_BATCH_COUNT = 4


def assert_managed_holds(dev, buf, value, *, stream):
    """Assert a managed buffer holds ``value``.

    Reads via an explicit device-to-host copy rather than dereferencing
    the managed pointer from the host. Managed pages carry residency and
    ``cuMemAdvise`` state that earlier tests in the suite can leave
    behind, which makes direct host reads order-dependent. Also avoids
    ``compare_buffer_to_constant``, which resolves a ``Device`` from
    ``memory_resource.device_id`` -- that is -1 for
    ``ManagedMemoryResource``.
    """
    host = LegacyPinnedMemoryResource().allocate(buf.size)
    expected = make_scratch_buffer(dev, value, buf.size)
    try:
        buf.copy_to(host, stream=stream)
        stream.sync()
        assert compare_equal_buffers(expected, host)
    finally:
        expected.close()
        host.close(stream)
        stream.sync()
