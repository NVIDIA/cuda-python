# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

import pytest

from cuda.core import (
    ContextOptions,
    DeviceResources,
    SMResource,
    SMResourceOptions,
    WorkqueueResource,
    WorkqueueResourceOptions,
)
from cuda.core._utils.cuda_utils import CUDAError


def _sm_resource_or_skip(dev):
    try:
        return dev.resources.sm
    except (NotImplementedError, CUDAError) as exc:
        pytest.skip(str(exc))


def _split_or_skip(sm, options, **kwargs):
    try:
        return sm.split(options, **kwargs)
    except (NotImplementedError, CUDAError) as exc:
        pytest.skip(str(exc))


def _green_context_or_skip(dev):
    sm = _sm_resource_or_skip(dev)
    groups, _ = _split_or_skip(sm, SMResourceOptions(count=None))
    try:
        return dev.create_context(ContextOptions(resources=[groups[0]]))
    except CUDAError as exc:
        pytest.skip(str(exc))


def test_not_user_constructible():
    with pytest.raises(RuntimeError):
        DeviceResources()
    with pytest.raises(RuntimeError):
        SMResource()
    with pytest.raises(RuntimeError):
        WorkqueueResource()


def test_create_context_without_resources_stays_unimplemented(init_cuda):
    with pytest.raises(NotImplementedError):
        init_cuda.create_context()
    with pytest.raises(NotImplementedError):
        init_cuda.create_context(ContextOptions(resources=None))
    with pytest.raises(TypeError):
        init_cuda.create_context(object())


def test_sm_resource_query(init_cuda):
    sm = _sm_resource_or_skip(init_cuda)

    assert sm.handle != 0
    assert sm.sm_count > 0
    assert sm.min_partition_size > 0
    assert sm.coscheduled_alignment > 0
    assert isinstance(sm.flags, int)
    assert not hasattr(sm, "memory_node_id")


def test_workqueue_resource_query_and_configure(init_cuda):
    try:
        wq = init_cuda.resources.workqueue
    except (NotImplementedError, CUDAError) as exc:
        pytest.skip(str(exc))

    assert wq.handle != 0
    assert wq.configure(WorkqueueResourceOptions(sharing_scope=None)) is None
    assert wq.configure(WorkqueueResourceOptions(sharing_scope="green_ctx_balanced")) is None
    with pytest.raises(ValueError, match="Unknown sharing_scope"):
        wq.configure(WorkqueueResourceOptions(sharing_scope="bogus"))


def test_sm_resource_split_validation(init_cuda):
    sm = _sm_resource_or_skip(init_cuda)

    with pytest.raises(ValueError, match="count is scalar"):
        sm.split(SMResourceOptions(count=4, coscheduled_sm_count=(2, 2)))

    with pytest.raises(ValueError, match="expected 2"):
        sm.split(SMResourceOptions(count=(4, 4), coscheduled_sm_count=(2, 2, 2)))

    with pytest.raises(NotImplementedError, match="min_count"):
        sm.split(SMResourceOptions(count=4, min_count=2))


def test_sm_resource_split_dry_run_cannot_create_context(init_cuda):
    sm = _sm_resource_or_skip(init_cuda)
    groups, _ = _split_or_skip(sm, SMResourceOptions(count=None), dry_run=True)

    assert len(groups) == 1
    with pytest.raises(ValueError, match="dry-run SMResource"):
        init_cuda.create_context(ContextOptions(resources=[groups[0]]))


def test_create_green_context(init_cuda):
    ctx = _green_context_or_skip(init_cuda)

    assert ctx.is_green
    assert ctx.handle is not None
    ctx.close()


def test_set_current_swap_preserves_green_context(init_cuda):
    dev = init_cuda
    green_ctx = _green_context_or_skip(dev)

    prev = dev.set_current(green_ctx)
    assert prev is not None

    restored = dev.set_current(prev)
    assert restored is green_ctx
    assert restored.is_green
    restored.close()
