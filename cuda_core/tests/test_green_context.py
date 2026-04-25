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
    except (NotImplementedError, CUDAError) as exc:
        pytest.skip(str(exc))


@pytest.fixture
def wq_resource(init_cuda):
    """Query workqueue resources from the device, skip if unsupported."""
    try:
        return init_cuda.resources.workqueue
    except (NotImplementedError, CUDAError) as exc:
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
def green_ctx_active(init_cuda, green_ctx):
    """Set a green context as current and restore the previous on teardown.

    Yields (dev, green_ctx, stream) for use in kernel launch tests.
    """
    dev = init_cuda
    prev = dev.set_current(green_ctx)
    try:
        stream = dev.create_stream()
        yield dev, green_ctx, stream
    finally:
        dev.set_current(prev)


@pytest.fixture
def fill_kernel(init_cuda):
    """Compile the fill kernel for the current device."""
    dev = init_cuda
    opts = ProgramOptions(std="c++17", arch=f"sm_{dev.arch}")
    prog = Program(_FILL_KERNEL, code_type="c++", options=opts)
    mod = prog.compile("cubin")
    return mod.get_kernel("fill")


def _aligned_half(sm):
    """Compute half the SM count, rounded down to min_partition_size alignment."""
    min_size = sm.min_partition_size
    half = (sm.sm_count // 2 // min_size) * min_size
    return half


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


def test_create_context_without_resources_stays_unimplemented(init_cuda):
    with pytest.raises(NotImplementedError):
        init_cuda.create_context()
    with pytest.raises(NotImplementedError):
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
        assert sm_resource.min_partition_size >= 8
        assert sm_resource.coscheduled_alignment >= 8


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
        """Single-group split: group gets at least requested SMs."""
        requested = sm_resource.min_partition_size
        groups, rem = sm_resource.split(SMResourceOptions(count=requested))

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
        """Two-group split with explicit aligned counts."""
        half = _aligned_half(sm_resource)
        if half < sm_resource.min_partition_size:
            pytest.skip("Not enough SMs for a 2-group split")

        groups, rem = sm_resource.split(SMResourceOptions(count=(half, half)))

        assert len(groups) == 2
        assert groups[0].sm_count > 0
        assert groups[1].sm_count > 0
        total = groups[0].sm_count + groups[1].sm_count + rem.sm_count
        assert total <= sm_resource.sm_count

    def test_two_groups_each_meets_request(self, sm_resource):
        min_size = sm_resource.min_partition_size
        half = _aligned_half(sm_resource)
        if half < min_size:
            pytest.skip("Not enough SMs for a 2-group split")

        groups, _ = sm_resource.split(SMResourceOptions(count=(min_size, min_size)))

        assert len(groups) == 2
        assert groups[0].sm_count >= min_size
        assert groups[1].sm_count >= min_size

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

    def test_set_current_swap_preserves_identity(self, init_cuda, green_ctx):
        dev = init_cuda
        with _use_green_ctx(dev, green_ctx):
            pass  # just verify push/pop works
        # After exiting, primary context is restored.
        # Verify we can swap in and get identity back:
        prev = dev.set_current(green_ctx)
        try:
            pass
        finally:
            restored = dev.set_current(prev)
        assert restored is green_ctx
        assert restored.is_green

    def test_stream_and_event_track_green_context(self, green_ctx_active):
        dev, green_ctx, stream = green_ctx_active

        event = stream.record()
        assert stream.context.is_green
        assert stream.context == green_ctx
        assert event.context.is_green
        assert event.context == green_ctx
        stream.sync()
        event.sync()

    def test_close_while_current_raises(self, init_cuda, green_ctx):
        dev = init_cuda
        with _use_green_ctx(dev, green_ctx), pytest.raises(RuntimeError, match="while it is current"):
            green_ctx.close()


# ---------------------------------------------------------------------------
# Kernel launch in green context
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
    def test_launch_and_verify(self, green_ctx_active, fill_kernel):
        """Compile, launch in green context, verify results on host."""
        dev, _, stream = green_ctx_active
        _launch_fill_and_verify(dev, stream, fill_kernel, n=64, value=42)

    def test_two_green_contexts_independent(self, init_cuda, sm_resource, fill_kernel):
        """Two SM groups -> two green contexts -> two independent kernels."""
        dev = init_cuda
        half = _aligned_half(sm_resource)
        if half < sm_resource.min_partition_size:
            pytest.skip("Not enough SMs for a 2-group split")

        groups, _ = sm_resource.split(SMResourceOptions(count=(half, half)))
        assert len(groups) == 2

        ctx_a = ctx_b = None
        try:
            ctx_a = dev.create_context(ContextOptions(resources=[groups[0]]))
            ctx_b = dev.create_context(ContextOptions(resources=[groups[1]]))

            for ctx, value in [(ctx_a, 10), (ctx_b, 20)]:
                with _use_green_ctx(dev, ctx):
                    stream = dev.create_stream()
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
            with _use_green_ctx(dev, ctx):
                stream = dev.create_stream()
                _launch_fill_and_verify(dev, stream, fill_kernel, n=32, value=99)
        finally:
            ctx.close()
