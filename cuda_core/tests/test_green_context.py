# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0


import contextlib

import numpy as np
import pytest

from cuda.core import (
    ContextOptions,
    DeviceResources,
    LaunchConfig,
    LegacyPinnedMemoryResource,
    Program,
    ProgramOptions,
    SMResource,
    SMResourceOptions,
    WorkqueueResource,
    WorkqueueResourceOptions,
    launch,
)
from cuda.core._utils.cuda_utils import CUDAError

# ---------------------------------------------------------------------------
# Kernel source
# ---------------------------------------------------------------------------

_FILL_KERNEL = r"""
extern "C" __global__ void fill(int* out, int value, int n) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid < n) {
        out[tid] = value;
    }
}
"""


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def sm_resource(init_cuda):
    """Query SM resources from the device, skip if unsupported."""
    try:
        return init_cuda.resources.sm
    except (RuntimeError, ValueError, CUDAError) as exc:
        pytest.skip(str(exc))


@pytest.fixture
def wq_resource(init_cuda):
    """Query workqueue resources from the device, skip if unsupported."""
    try:
        return init_cuda.resources.workqueue
    except (RuntimeError, ValueError, CUDAError) as exc:
        pytest.skip(str(exc))


@pytest.fixture
def green_ctx(init_cuda, sm_resource):
    """Create a single-group green context with proper teardown."""
    groups, _ = sm_resource.split(SMResourceOptions(count=None))
    try:
        ctx = init_cuda.create_context(ContextOptions(resources=[groups[0]]))
    except CUDAError as exc:
        pytest.skip(str(exc))
    yield ctx
    ctx.close()


@pytest.fixture
def fill_kernel(init_cuda):
    """Compile the fill kernel for the current device."""
    dev = init_cuda
    opts = ProgramOptions(std="c++17", arch=f"sm_{dev.arch}")
    prog = Program(_FILL_KERNEL, code_type="c++", options=opts)
    mod = prog.compile("cubin")
    return mod.get_kernel("fill")


def _is_invalid_resource_configuration(exc):
    return "CUDA_ERROR_INVALID_RESOURCE_CONFIGURATION" in str(exc)


def _iter_requested_sm_counts(sm, n_groups=1, *, descending=False):
    """Yield even per-group SM counts worth probing on this device."""
    start = max(2, sm.min_partition_size)
    if start % 2:
        start += 1
    stop = sm.sm_count // n_groups
    counts = range(start, stop + 1, 2)
    return reversed(counts) if descending else counts


def _try_sm_split(sm, *, count, backfill=False):
    try:
        return sm.split(SMResourceOptions(count=count, backfill=backfill))
    except CUDAError as exc:
        if _is_invalid_resource_configuration(exc):
            return None
        raise


def _find_supported_split(sm, *, n_groups=1, backfill=False, descending=False):
    """Return a supported explicit split request for this device, if any."""
    for count in _iter_requested_sm_counts(sm, n_groups=n_groups, descending=descending):
        request = count if n_groups == 1 else (count,) * n_groups
        split = _try_sm_split(sm, count=request, backfill=backfill)
        if split is not None:
            groups, rem = split
            return count, groups, rem
    return None


def _find_any_two_group_split(sm):
    split = _find_supported_split(sm, n_groups=2)
    if split is not None:
        return split
    return _find_supported_split(sm, n_groups=2, backfill=True)


def _find_backfill_only_two_group_split(sm):
    """Return a 2-group split size that needs backfill, if the device has one."""
    for count in _iter_requested_sm_counts(sm, n_groups=2, descending=True):
        request = (count, count)
        if _try_sm_split(sm, count=request) is not None:
            continue
        split = _try_sm_split(sm, count=request, backfill=True)
        if split is not None:
            groups, rem = split
            return count, groups, rem
    return None


@contextlib.contextmanager
def _use_green_ctx(dev, ctx):
    """Context manager: set green ctx current, restore previous on exit."""
    prev = dev.set_current(ctx)
    try:
        yield
    finally:
        dev.set_current(prev)


# ---------------------------------------------------------------------------
# Construction / type tests
# ---------------------------------------------------------------------------


def test_not_user_constructible():
    with pytest.raises(RuntimeError):
        DeviceResources()
    with pytest.raises(RuntimeError):
        SMResource()
    with pytest.raises(RuntimeError):
        WorkqueueResource()


def test_create_context_requires_resources(init_cuda):
    with pytest.raises(ValueError, match="resources must be provided"):
        init_cuda.create_context()
    with pytest.raises(ValueError, match="resources must be provided"):
        init_cuda.create_context(ContextOptions(resources=None))
    with pytest.raises(TypeError):
        init_cuda.create_context(object())


# ---------------------------------------------------------------------------
# SM resource query
# ---------------------------------------------------------------------------


class TestSMResourceQuery:
    def test_properties(self, sm_resource):
        assert sm_resource.handle != 0
        assert sm_resource.sm_count > 0
        assert sm_resource.min_partition_size > 0
        assert sm_resource.coscheduled_alignment > 0
        assert isinstance(sm_resource.flags, int)

    def test_no_memory_node_id_in_v1(self, sm_resource):
        """memory_node_id is deferred to v1.1 (CUDA 13.4)."""
        assert not hasattr(sm_resource, "memory_node_id")

    def test_arch_constraints_pre_hopper(self, init_cuda, sm_resource):
        if init_cuda.compute_capability >= (9, 0):
            pytest.skip("Test is for pre-Hopper architectures")
        assert sm_resource.min_partition_size >= 2
        assert sm_resource.coscheduled_alignment >= 2

    def test_arch_constraints_hopper_plus(self, init_cuda, sm_resource):
        if init_cuda.compute_capability < (9, 0):
            pytest.skip("Test is for Hopper+ architectures")
        assert sm_resource.min_partition_size >= 2
        assert sm_resource.coscheduled_alignment >= 2
        assert sm_resource.min_partition_size % 2 == 0
        assert sm_resource.coscheduled_alignment % 2 == 0


# ---------------------------------------------------------------------------
# Workqueue resource
# ---------------------------------------------------------------------------


class TestWorkqueueResource:
    def test_query(self, wq_resource):
        assert wq_resource.handle != 0

    def test_configure_none_is_noop(self, wq_resource):
        assert wq_resource.configure(WorkqueueResourceOptions(sharing_scope=None)) is None

    def test_configure_valid_scope(self, wq_resource):
        wq_resource.configure(WorkqueueResourceOptions(sharing_scope="green_ctx_balanced"))

    def test_configure_invalid_scope_raises(self, wq_resource):
        with pytest.raises(ValueError, match="Unknown sharing_scope"):
            wq_resource.configure(WorkqueueResourceOptions(sharing_scope="bogus"))

    def test_query_concurrency_limit_nonzero(self, wq_resource):
        # driver populates from CUDA_DEVICE_MAX_CONNECTIONS
        assert wq_resource.concurrency_limit >= 1

    def test_configure_concurrency_limit(self, wq_resource):
        wq_resource.configure(WorkqueueResourceOptions(concurrency_limit=4))
        assert wq_resource.concurrency_limit == 4

    def test_configure_concurrency_and_scope(self, wq_resource):
        wq_resource.configure(WorkqueueResourceOptions(
            sharing_scope="green_ctx_balanced",
            concurrency_limit=2,
        ))
        assert wq_resource.concurrency_limit == 2

    def test_configure_concurrency_limit_zero_raises(self, wq_resource):
        with pytest.raises(ValueError, match="concurrency_limit must be >= 1"):
            wq_resource.configure(WorkqueueResourceOptions(concurrency_limit=0))

    def test_configure_concurrency_limit_negative_raises(self, wq_resource):
        with pytest.raises(ValueError, match="concurrency_limit must be >= 1"):
            wq_resource.configure(WorkqueueResourceOptions(concurrency_limit=-3))


# ---------------------------------------------------------------------------
# SM resource split — validation
# ---------------------------------------------------------------------------


class TestSMResourceSplitValidation:
    def test_scalar_count_with_sequence_field_raises(self, sm_resource):
        count = sm_resource.min_partition_size
        with pytest.raises(ValueError, match="count is scalar"):
            sm_resource.split(
                SMResourceOptions(
                    count=count,
                    coscheduled_sm_count=(count, count),
                )
            )

    def test_sequence_length_mismatch_raises(self, sm_resource):
        count = sm_resource.min_partition_size
        with pytest.raises(ValueError, match="expected 2"):
            sm_resource.split(
                SMResourceOptions(
                    count=(count, count),
                    coscheduled_sm_count=(count, count, count),
                )
            )

    def test_negative_count_raises(self, sm_resource):
        with pytest.raises(ValueError, match="count must be non-negative"):
            sm_resource.split(SMResourceOptions(count=-1))

    def test_dry_run_cannot_create_context(self, init_cuda, sm_resource):
        groups, _ = sm_resource.split(SMResourceOptions(count=None), dry_run=True)
        assert len(groups) == 1
        with pytest.raises(ValueError, match="dry-run SMResource"):
            init_cuda.create_context(ContextOptions(resources=[groups[0]]))


# ---------------------------------------------------------------------------
# SM resource split — functional
# ---------------------------------------------------------------------------


class TestSMResourceSplit:
    def test_single_group_counts(self, sm_resource):
        """Single-group split: group gets at least a supported requested size."""
        split = _find_supported_split(sm_resource)
        if split is None:
            pytest.skip("Device does not expose a valid explicit single-group split")
        requested, groups, rem = split

        assert len(groups) == 1
        assert groups[0].sm_count >= requested
        assert groups[0].sm_count + rem.sm_count <= sm_resource.sm_count

    def test_discovery_mode(self, sm_resource):
        """count=None auto-detects a valid SM count."""
        groups, _ = sm_resource.split(SMResourceOptions(count=None))

        assert len(groups) == 1
        assert groups[0].sm_count >= sm_resource.min_partition_size

    def test_discovery_respects_alignment(self, sm_resource):
        groups, _ = sm_resource.split(SMResourceOptions(count=None))

        if sm_resource.coscheduled_alignment > 0:
            assert groups[0].sm_count % sm_resource.coscheduled_alignment == 0

    def test_two_groups(self, sm_resource):
        """Two-group split succeeds for a supported explicit request."""
        split = _find_supported_split(sm_resource, n_groups=2)
        if split is None:
            pytest.skip("Device does not expose a valid 2-group split without backfill")
        count, groups, rem = split

        assert len(groups) == 2
        assert groups[0].sm_count >= count
        assert groups[1].sm_count >= count
        total = groups[0].sm_count + groups[1].sm_count + rem.sm_count
        assert total <= sm_resource.sm_count

    def test_two_groups_backfill(self, sm_resource):
        """Backfill unlocks a 2-group split size that default placement rejects."""
        split = _find_backfill_only_two_group_split(sm_resource)
        if split is None:
            pytest.skip("Device does not expose a backfill-only 2-group split")
        requested, groups, rem = split

        assert len(groups) == 2
        assert groups[0].sm_count >= requested
        assert groups[1].sm_count >= requested
        assert groups[0].sm_count + groups[1].sm_count + rem.sm_count <= sm_resource.sm_count

    def test_dry_run_matches_real(self, sm_resource):
        """Dry-run reports the same SM counts as a real split."""
        opts = SMResourceOptions(count=None)

        dry_groups, _ = sm_resource.split(opts, dry_run=True)
        real_groups, _ = sm_resource.split(opts, dry_run=False)

        assert len(dry_groups) == len(real_groups)
        for dg, rg in zip(dry_groups, real_groups):
            assert dg.sm_count == rg.sm_count


# ---------------------------------------------------------------------------
# Green context lifecycle
# ---------------------------------------------------------------------------


class TestGreenContextLifecycle:
    def test_is_green(self, green_ctx):
        assert green_ctx.is_green
        assert green_ctx.handle is not None

    def test_create_stream_on_primary_raises(self, init_cuda):
        """create_stream is only for green contexts."""
        # The init_cuda fixture sets the primary context
        # Get the primary context via device internals
        ctx = init_cuda._context
        with pytest.raises(RuntimeError, match="only supported on green contexts"):
            ctx.create_stream()

    def test_create_stream_blocking_raises(self, green_ctx):
        """Green context streams must be non-blocking."""
        from cuda.core import StreamOptions

        with pytest.raises(ValueError, match="must be non-blocking"):
            green_ctx.create_stream(StreamOptions(nonblocking=False))

    def test_create_stream_explicit(self, green_ctx):
        """Create a stream directly from the green context (no set_current)."""
        stream = green_ctx.create_stream()
        assert stream is not None
        assert stream.context.is_green
        assert stream.context == green_ctx

    def test_stream_and_event_track_green_context(self, green_ctx):
        stream = green_ctx.create_stream()
        event = stream.record()
        assert stream.context.is_green
        assert stream.context == green_ctx
        assert event.context.is_green
        assert event.context == green_ctx
        stream.sync()
        event.sync()

    def test_close_while_current_raises(self, init_cuda, green_ctx):
        """close() on a current context raises — test via set_current."""
        dev = init_cuda
        with _use_green_ctx(dev, green_ctx), pytest.raises(RuntimeError, match="while it is current"):
            green_ctx.close()

    def test_set_current_swap_regression(self, init_cuda, green_ctx):
        """set_current still works (backward compat) and preserves identity."""
        dev = init_cuda
        with _use_green_ctx(dev, green_ctx):
            pass  # just verify push/pop works
        # Swap again and check identity round-trip
        prev = dev.set_current(green_ctx)
        try:
            assert prev is not None
        finally:
            restored = dev.set_current(prev)
        assert restored is green_ctx
        assert restored.is_green


# ---------------------------------------------------------------------------
# Context.resources
# ---------------------------------------------------------------------------


class TestContextResources:
    def test_green_ctx_sm_resources(self, green_ctx, sm_resource):
        """Green context's SM resources should be a subset of device SMs."""
        ctx_sm = green_ctx.resources.sm
        assert ctx_sm.sm_count > 0
        assert ctx_sm.sm_count <= sm_resource.sm_count

    def test_green_ctx_resources_reflect_partition(self, init_cuda, sm_resource):
        """Two green contexts should have disjoint SM partitions."""
        split = _find_any_two_group_split(sm_resource)
        if split is None:
            pytest.skip("Device does not expose a valid 2-group split")
        _, groups, _ = split

        ctx_a = ctx_b = None
        try:
            ctx_a = init_cuda.create_context(ContextOptions(resources=[groups[0]]))
            ctx_b = init_cuda.create_context(ContextOptions(resources=[groups[1]]))

            sm_a = ctx_a.resources.sm.sm_count
            sm_b = ctx_b.resources.sm.sm_count
            assert sm_a > 0
            assert sm_b > 0
            assert sm_a + sm_b <= sm_resource.sm_count
        finally:
            if ctx_b is not None:
                ctx_b.close()
            if ctx_a is not None:
                ctx_a.close()

    def test_stream_resources_match_context(self, green_ctx, sm_resource):
        """stream.resources should return the same as ctx.resources."""
        stream = green_ctx.create_stream()

        stream_sm = stream.resources.sm
        ctx_sm = green_ctx.resources.sm
        assert stream_sm.sm_count == ctx_sm.sm_count
        assert stream_sm.sm_count > 0
        assert stream_sm.sm_count <= sm_resource.sm_count

        try:
            stream_wq = stream.resources.workqueue
            ctx_wq = green_ctx.resources.workqueue
            assert stream_wq.handle != 0
            assert ctx_wq.handle != 0
        except (RuntimeError, ValueError, CUDAError):
            pass  # workqueue not available on this driver/build


# ---------------------------------------------------------------------------
# Kernel launch in green context (explicit model)
# ---------------------------------------------------------------------------


def _launch_fill_and_verify(dev, stream, kernel, n, value):
    """Launch the fill kernel and verify results on host."""
    dev_buf = dev.allocate(n * np.dtype(np.int32).itemsize, stream=stream)

    config = LaunchConfig(grid=(n + 31) // 32, block=32)
    launch(stream, config, kernel, dev_buf, np.int32(value), np.int32(n))

    host_mr = LegacyPinnedMemoryResource()
    host_buf = host_mr.allocate(n * np.dtype(np.int32).itemsize)
    host_arr = np.from_dlpack(host_buf).view(np.int32)
    host_arr[:] = 0

    dev_buf.copy_to(host_buf, stream=stream)
    stream.sync()

    np.testing.assert_array_equal(host_arr, np.full(n, value, dtype=np.int32))


class TestGreenContextKernelLaunch:
    def test_launch_and_verify(self, init_cuda, green_ctx, fill_kernel):
        """Launch kernel via ctx.create_stream (explicit model, no set_current)."""
        stream = green_ctx.create_stream()
        _launch_fill_and_verify(init_cuda, stream, fill_kernel, n=64, value=42)

    def test_two_green_contexts_independent(self, init_cuda, sm_resource, fill_kernel):
        """Two SM groups -> two green contexts -> two independent kernels."""
        dev = init_cuda
        split = _find_any_two_group_split(sm_resource)
        if split is None:
            pytest.skip("Device does not expose a valid 2-group split")
        _, groups, _ = split
        assert len(groups) == 2

        ctx_a = ctx_b = None
        try:
            ctx_a = dev.create_context(ContextOptions(resources=[groups[0]]))
            ctx_b = dev.create_context(ContextOptions(resources=[groups[1]]))

            for ctx, value in [(ctx_a, 10), (ctx_b, 20)]:
                stream = ctx.create_stream()
                _launch_fill_and_verify(dev, stream, fill_kernel, n=64, value=value)
        finally:
            if ctx_b is not None:
                ctx_b.close()
            if ctx_a is not None:
                ctx_a.close()

    def test_with_workqueue_resource(self, init_cuda, sm_resource, wq_resource, fill_kernel):
        """Green context with SM + workqueue resources can launch a kernel."""
        dev = init_cuda
        groups, _ = sm_resource.split(SMResourceOptions(count=None))

        try:
            ctx = dev.create_context(ContextOptions(resources=[groups[0], wq_resource]))
        except CUDAError as exc:
            pytest.skip(str(exc))

        assert ctx.is_green

        try:
            stream = ctx.create_stream()
            _launch_fill_and_verify(dev, stream, fill_kernel, n=32, value=99)
        finally:
            ctx.close()
