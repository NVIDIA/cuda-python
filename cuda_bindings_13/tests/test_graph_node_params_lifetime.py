# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Memory-safety and correctness tests for AUTO_LOWPP_CLASS wrappers around
CUDA_MEMCPY_BATCH_NODE_PARAMS_V1 / CUgraphNodeParams and similar embedded/batch
structs.

These do not require a CUDA device: they exercise the pure-Python/Cython
wrapper objects (buffer allocation, getters/setters, __dealloc__ ordering)
without calling into the driver. Tests that would previously segfault use a
fresh subprocess (via run_python_snippet) so a crash is reported as a failed
assertion instead of taking down the whole pytest run.

dsts and srcs share CUDA_MEMCPY_BATCH_NODE_PARAMS_V1's `count` field and so
cannot be set independently (setting one would silently invalidate the
other's length); they're set together via set_dsts_srcs(dsts, srcs) instead.
"""

import pytest
from cuda_python_test_helpers.subprocess_runner import run_python_snippet

import cuda.bindings.driver as cuda


@pytest.mark.agent_authored(model="claude-sonnet-5")
def test_embedded_batch_member_survives_parent_deletion():
    """Deleting a CUgraphNodeParams whose embedded memcpyBatch was populated
    must not crash: the parent's __dealloc__ must release its cached
    memcpyBatch wrapper before freeing its own buffer, since that wrapper's
    own __dealloc__ writes through the same memory."""
    result = run_python_snippet(
        """
import gc
import cuda.bindings.driver as cuda

params = cuda.CUgraphNodeParams()
params.memcpyBatch.set_dsts_srcs([1, 2, 3], [4, 5, 6])
del params
gc.collect()
print("ok")
"""
    )
    assert result.stdout.strip() == "ok"


@pytest.mark.agent_authored(model="claude-sonnet-5")
def test_dsts_and_srcs_cannot_be_set_independently():
    """dsts and srcs share the same public `count` field but each keeps its
    own private backing buffer. Setting them independently would let one
    setter overwrite the shared count field written by the other, so each
    property setter raises, pointing at set_dsts_srcs instead."""
    batch = cuda.CUDA_MEMCPY_BATCH_NODE_PARAMS_V1()
    with pytest.raises(AttributeError, match="set_dsts_srcs"):
        batch.dsts = [1, 2, 3]
    with pytest.raises(AttributeError, match="set_dsts_srcs"):
        batch.srcs = [1, 2, 3]


@pytest.mark.agent_authored(model="claude-sonnet-5")
def test_set_dsts_srcs_requires_matching_lengths():
    batch = cuda.CUDA_MEMCPY_BATCH_NODE_PARAMS_V1()
    with pytest.raises(ValueError, match="same length"):
        batch.set_dsts_srcs([1, 2], [3, 4, 5])


@pytest.mark.agent_authored(model="claude-sonnet-5")
def test_set_dsts_srcs_updates_both_and_shared_count():
    batch = cuda.CUDA_MEMCPY_BATCH_NODE_PARAMS_V1()
    batch.set_dsts_srcs([1, 2, 3], [4, 5, 6])
    assert [int(x) for x in batch.dsts] == [1, 2, 3]
    assert [int(x) for x in batch.srcs] == [4, 5, 6]
    assert batch.count == 3

    # Reassigning to a different length must not leave dsts and srcs (or the
    # count they share) inconsistent with each other.
    batch.set_dsts_srcs([7], [8])
    assert [int(x) for x in batch.dsts] == [7]
    assert [int(x) for x in batch.srcs] == [8]
    assert batch.count == 1


@pytest.mark.agent_authored(model="claude-sonnet-5")
def test_sibling_members_sharing_a_count_field_do_not_crash_in_subprocess():
    """Regression check for the NULL-pointer write this used to be: before
    set_dsts_srcs existed, setting dsts then srcs independently would have
    srcs's setter see the count field dsts's setter just wrote, wrongly
    conclude its own (still-NULL) buffer didn't need allocating, and write
    through NULL. Run out-of-process since this is exactly the kind of bug
    that segfaults instead of raising."""
    result = run_python_snippet(
        """
import cuda.bindings.driver as cuda

batch = cuda.CUDA_MEMCPY_BATCH_NODE_PARAMS_V1()
batch.set_dsts_srcs([1, 2, 3], [4, 5, 6])
assert [int(x) for x in batch.dsts] == [1, 2, 3]
assert [int(x) for x in batch.srcs] == [4, 5, 6]
print("ok")
"""
    )
    assert result.stdout.strip() == "ok"


@pytest.mark.agent_authored(model="claude-sonnet-5")
def test_memcpy_batch_assignment_is_a_shallow_copy():
    """params.memcpyBatch = batch does a raw memcpy of batch's bytes, same as
    every other embedded-struct/struct-array assignment in these bindings
    (e.g. CUgraphNodeParams.conditional, CUDA_MEMCPY_BATCH_NODE_PARAMS_V1.attrs):
    none of them deep-copy or retain a source object's own owned buffers.
    So params.memcpyBatch.dsts ends up aliasing batch's dsts buffer, and
    freeing batch (its buffer, not just the Python object) invalidates it.
    This documents that as accepted, existing behavior, not something this
    struct's setter is expected to protect against."""
    params = cuda.CUgraphNodeParams()
    batch = cuda.CUDA_MEMCPY_BATCH_NODE_PARAMS_V1()
    batch.set_dsts_srcs([10, 20, 30], [40, 50, 60])
    params.memcpyBatch = batch
    assert [int(x) for x in params.memcpyBatch.dsts] == [10, 20, 30]
    assert int(params.memcpyBatch.dsts[0]) == int(batch.dsts[0])


@pytest.mark.agent_authored(model="claude-sonnet-5")
def test_dsts_getter_returns_value_copies_not_aliases():
    """Elements returned by the dsts getter must be independent copies: if
    they instead alias the backing buffer, replacing dsts (which frees the
    old buffer) leaves previously-returned elements pointing at freed
    memory."""
    result = run_python_snippet(
        """
import gc
import cuda.bindings.driver as cuda

batch = cuda.CUDA_MEMCPY_BATCH_NODE_PARAMS_V1()
batch.set_dsts_srcs([100, 200], [1, 2])
elems = batch.dsts
batch.set_dsts_srcs([999], [1])  # frees and replaces the old backing buffer
gc.collect()
assert [int(x) for x in elems] == [100, 200]
print("ok")
"""
    )
    assert result.stdout.strip() == "ok"


@pytest.mark.agent_authored(model="claude-sonnet-5")
def test_dsts_getter_reads_public_count_field():
    """The dsts/srcs getters must read the struct's own public `count` field
    rather than a private, Python-setter-only length. A driver call that
    populates a CUDA_MEMCPY_BATCH_NODE_PARAMS_V1 in place (e.g. reading back
    node params) only ever sets `count`, never any private bookkeeping. This
    simulates that by creating a second wrapper around the same memory
    address, whose own private state was never touched."""
    batch = cuda.CUDA_MEMCPY_BATCH_NODE_PARAMS_V1()
    batch.set_dsts_srcs([11, 22, 33], [1, 2, 3])

    aliased = cuda.CUDA_MEMCPY_BATCH_NODE_PARAMS_V1(_ptr=batch.getPtr())
    assert aliased.count == 3
    assert [int(x) for x in aliased.dsts] == [11, 22, 33]


@pytest.mark.agent_authored(model="claude-sonnet-5")
def test_attrs_struct_list_reads_public_num_attrs_field():
    """Same as test_dsts_getter_reads_public_count_field, but for the
    struct-list (attrs / numAttrs) getter path rather than the primitive-list
    (dsts / count) path. attrs is independently settable (it doesn't share
    numAttrs with any other member with real list semantics), so it keeps
    its own setter rather than needing a combined one."""
    batch = cuda.CUDA_MEMCPY_BATCH_NODE_PARAMS_V1()
    batch.attrs = [cuda.CUmemcpyAttributes(), cuda.CUmemcpyAttributes()]

    aliased = cuda.CUDA_MEMCPY_BATCH_NODE_PARAMS_V1(_ptr=batch.getPtr())
    assert aliased.numAttrs == 2
    assert len(aliased.attrs) == 2


@pytest.mark.agent_authored(model="claude-sonnet-5")
def test_set_dsts_srcs_raises_without_leaking_or_mutating_on_bad_element():
    """A conversion failure partway through set_dsts_srcs must not leave
    dsts/srcs partially overwritten, and must not leak the freshly-allocated
    replacement buffers."""
    batch = cuda.CUDA_MEMCPY_BATCH_NODE_PARAMS_V1()
    batch.set_dsts_srcs([1, 2, 3], [4, 5, 6])
    before_dsts = [int(x) for x in batch.dsts]
    before_srcs = [int(x) for x in batch.srcs]
    before_count = batch.count

    with pytest.raises((ValueError, TypeError)):
        batch.set_dsts_srcs([1, 2, "not-an-int"], [4, 5, 6])

    assert batch.count == before_count
    assert [int(x) for x in batch.dsts] == before_dsts
    assert [int(x) for x in batch.srcs] == before_srcs


@pytest.mark.agent_authored(model="claude-sonnet-5")
def test_set_dsts_srcs_exception_safety_in_subprocess():
    """Same as the test above, but run out-of-process: pre-fix, the resize
    path allocated the new buffers and started writing into them before
    validating every element, which is memory-unsafe enough to be worth
    checking under a fresh interpreter too."""
    result = run_python_snippet(
        """
import cuda.bindings.driver as cuda

batch = cuda.CUDA_MEMCPY_BATCH_NODE_PARAMS_V1()
batch.set_dsts_srcs([1, 2, 3], [4, 5, 6])
try:
    batch.set_dsts_srcs([1, 2, 3, 4, "not-an-int"], [4, 5, 6, 7, 8])
    raise SystemExit("expected an exception")
except (ValueError, TypeError):
    pass
assert batch.count == 3
assert [int(x) for x in batch.dsts] == [1, 2, 3]
assert [int(x) for x in batch.srcs] == [4, 5, 6]
print("ok")
"""
    )
    assert result.stdout.strip() == "ok"
