# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

"""Shared constants and helpers for the ``copy_batch`` tests.

Fixtures live in ``tests/memory/conftest.py``; this module holds the
pieces that tests import by name.
"""

from cuda.core import LegacyPinnedMemoryResource
from cuda.core._memory._copy_ops import _batch_entry_point_in_use
from helpers.buffers import compare_equal_buffers, make_scratch_buffer

COPY_BATCH_SIZE = 4096
COPY_BATCH_COUNT = 4

# Matches the UserWarning raised when prefer_overlap_with_compute is
# requested on a discrete GPU. Tests that are not about the warning
# silence it so they stay green under -W error.
OVERLAP_WARNING_FILTER = "ignore:overlap_mode:UserWarning"


def uses_batch_entry_point() -> bool:
    """Whether copy_batch reaches ``cuMemcpyBatchAsync`` on this system.

    Delegates to the implementation's own dispatch predicate rather than
    re-deriving it, so the tests cannot drift from it. ``copy_batch``
    itself works either way -- only non-default ``CopyOptions`` need the
    batched entry point.
    """
    return _batch_entry_point_in_use()


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
