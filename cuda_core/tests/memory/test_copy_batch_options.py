# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""``CopyOptions`` handling and argument validation for ``copy_batch``.

Covers how options are encoded into the driver's attribute runs, how each
option field behaves, and every rejection path. The data-movement
behaviour itself lives in ``test_copy_batch.py``.
"""

import warnings

import pytest
from helpers.buffers import compare_buffer_to_constant, compare_equal_buffers, set_buffer
from helpers.copy_batch import (
    COPY_BATCH_SIZE,
    OVERLAP_WARNING_FILTER,
    assert_managed_holds,
    managed_mr_or_skip,
)

from cuda.core import Device, Host, LegacyPinnedMemoryResource
from cuda.core._memory._copy_enums import _attr_run_starts
from cuda.core._utils.version import binding_version
from cuda.core.utils import (
    CopyOptions,
    MemcpyOverlapMode,
    MemcpySrcAccessOrder,
    copy_batch,
)


class TestAttrRunStarts:
    """Unit tests for the attrsIdxs run-length encoding.

    Pure logic, no CUDA: ``attrs[k]`` applies to the copies in
    ``[starts[k], starts[k + 1])``.
    """

    @pytest.mark.agent_authored(model="Claude Opus 5")
    def test_broadcast_collapses_to_one_run(self):
        attrs = [CopyOptions(src_access_order=MemcpySrcAccessOrder.ANY)] * 4
        assert _attr_run_starts(attrs) == [0]

    @pytest.mark.agent_authored(model="Claude Opus 5")
    def test_equal_but_distinct_instances_collapse(self):
        # Structural equality, not identity, drives the collapse.
        attrs = [CopyOptions(src_access_order="stream") for _ in range(3)]
        assert len({id(a) for a in attrs}) == 3
        assert _attr_run_starts(attrs) == [0]

    @pytest.mark.agent_authored(model="Claude Opus 5")
    def test_all_distinct_yields_one_run_each(self):
        attrs = [
            CopyOptions(src_access_order=MemcpySrcAccessOrder.STREAM),
            CopyOptions(src_access_order=MemcpySrcAccessOrder.ANY),
            CopyOptions(src_access_order=MemcpySrcAccessOrder.DURING_API_CALL),
        ]
        assert _attr_run_starts(attrs) == [0, 1, 2]

    @pytest.mark.agent_authored(model="Claude Opus 5")
    def test_adjacent_runs_are_grouped(self):
        stream_attr = CopyOptions(src_access_order=MemcpySrcAccessOrder.STREAM)
        any_attr = CopyOptions(src_access_order=MemcpySrcAccessOrder.ANY)
        attrs = [stream_attr, stream_attr, any_attr, any_attr, stream_attr]
        # Runs start at 0 (stream), 2 (any) and 4 (stream again).
        assert _attr_run_starts(attrs) == [0, 2, 4]

    @pytest.mark.agent_authored(model="Claude Opus 5")
    def test_single_element(self):
        assert _attr_run_starts([CopyOptions()]) == [0]


class TestCopyBatchOptions:
    """Each ``CopyOptions`` field is accepted and does not corrupt the copy."""

    @pytest.mark.agent_authored(model="Claude Opus 5")
    @pytest.mark.parametrize(
        ("order", "marker"),
        [
            (MemcpySrcAccessOrder.STREAM, 31),
            (MemcpySrcAccessOrder.DURING_API_CALL, 32),
            (MemcpySrcAccessOrder.ANY, 33),
        ],
    )
    def test_src_access_order(self, h2d_bufs, copy_stream, order, marker):
        srcs, dsts = h2d_bufs
        for i, src in enumerate(srcs):
            set_buffer(src, i + marker)

        copy_batch(copy_stream, srcs, dsts, options=CopyOptions(src_access_order=order))
        copy_stream.sync()

        for i, dst in enumerate(dsts):
            assert compare_buffer_to_constant(dst, i + marker)

    @pytest.mark.agent_authored(model="Claude Opus 5")
    def test_per_copy_options(self, h2d_bufs, copy_stream):
        srcs, dsts = h2d_bufs
        for i, src in enumerate(srcs):
            set_buffer(src, i + 40)

        per_copy_options = [
            CopyOptions(src_access_order=MemcpySrcAccessOrder.STREAM),
            CopyOptions(src_access_order=MemcpySrcAccessOrder.ANY),
            CopyOptions(src_access_order=MemcpySrcAccessOrder.DURING_API_CALL),
            CopyOptions(src_access_order=MemcpySrcAccessOrder.STREAM),
        ]
        assert _attr_run_starts(per_copy_options) == [0, 1, 2, 3]

        copy_batch(copy_stream, srcs, dsts, options=per_copy_options)
        copy_stream.sync()

        for i, dst in enumerate(dsts):
            assert compare_buffer_to_constant(dst, i + 40)

    @pytest.mark.agent_authored(model="Claude Opus 5")
    def test_scalar_options_broadcast(self, copy_batch_device, h2d_bufs, copy_stream):
        """A scalar option must apply to every copy.

        Verified three ways: the scalar collapses to a single driver
        attribute, a scalar and an equivalent explicit per-copy list give
        identical bytes, and a short list is *not* silently broadcast.
        """
        srcs, scalar_dsts = h2d_bufs
        device_mr = copy_batch_device.memory_resource
        pinned_mr = LegacyPinnedMemoryResource()
        n = len(srcs)
        scalar_option = CopyOptions(src_access_order=MemcpySrcAccessOrder.ANY)

        for i, src in enumerate(srcs):
            set_buffer(src, i + 95)

        # A scalar is expanded internally to n copies of one option, which
        # the encoder then collapses back to a single driver entry.
        assert _attr_run_starts([scalar_option] * n) == [0]

        copy_batch(copy_stream, srcs, scalar_dsts, options=scalar_option)

        listed_dsts = [device_mr.allocate(COPY_BATCH_SIZE, stream=copy_stream) for _ in srcs]
        copy_batch(copy_stream, srcs, listed_dsts, options=[scalar_option] * n)
        copy_stream.sync()

        # Every copy received the option, and both spellings agree.
        for i, (scalar_dst, listed_dst) in enumerate(zip(scalar_dsts, listed_dsts)):
            assert compare_buffer_to_constant(scalar_dst, i + 95)
            scalar_host = pinned_mr.allocate(COPY_BATCH_SIZE)
            listed_host = pinned_mr.allocate(COPY_BATCH_SIZE)
            scalar_dst.copy_to(scalar_host, stream=copy_stream)
            listed_dst.copy_to(listed_host, stream=copy_stream)
            copy_stream.sync()
            assert compare_equal_buffers(scalar_host, listed_host)
            scalar_host.close(copy_stream)
            listed_host.close(copy_stream)

        # A sequence is paired by index and never broadcast, so a
        # one-element list is a length error rather than a scalar.
        with pytest.raises(ValueError, match="options length"):
            copy_batch(copy_stream, srcs, listed_dsts, options=[scalar_option])

        for buf in listed_dsts:
            buf.close(copy_stream)
        copy_stream.sync()

    @pytest.mark.agent_authored(model="Claude Opus 5")
    def test_location_hints(self, copy_batch_device, copy_stream):
        dev = copy_batch_device
        mr = managed_mr_or_skip()
        srcs = [mr.allocate(COPY_BATCH_SIZE, stream=copy_stream) for _ in range(2)]
        dsts = [mr.allocate(COPY_BATCH_SIZE, stream=copy_stream) for _ in range(2)]

        for i, src in enumerate(srcs):
            src.fill(i + 80, stream=copy_stream)

        options = CopyOptions(
            src_access_order=MemcpySrcAccessOrder.STREAM,
            src_location_hint=dev,
            dst_location_hint=Host(),
        )
        copy_batch(copy_stream, srcs, dsts, options=options)
        copy_stream.sync()

        for i, dst in enumerate(dsts):
            assert_managed_holds(dev, dst, i + 80, stream=copy_stream)

        for buf in srcs + dsts:
            buf.close(copy_stream)
        copy_stream.sync()
        mr.close()

    @pytest.mark.agent_authored(model="Claude Opus 5")
    def test_host_numa_location_hint(self, copy_batch_device, copy_stream):
        """NUMA host hints round-trip on CUDA 13 and are rejected on CUDA 12."""
        dev = copy_batch_device
        mr = managed_mr_or_skip()
        srcs = [mr.allocate(COPY_BATCH_SIZE, stream=copy_stream) for _ in range(2)]
        dsts = [mr.allocate(COPY_BATCH_SIZE, stream=copy_stream) for _ in range(2)]
        for i, src in enumerate(srcs):
            src.fill(i + 85, stream=copy_stream)

        options = CopyOptions(dst_location_hint=Host(numa_id=0))

        if binding_version() < (13, 0, 0):
            with pytest.raises(TypeError, match="CUDA 13"):
                copy_batch(copy_stream, srcs, dsts, options=options)
        else:
            copy_batch(copy_stream, srcs, dsts, options=options)
            copy_stream.sync()
            for i, dst in enumerate(dsts):
                assert_managed_holds(dev, dst, i + 85, stream=copy_stream)

        for buf in srcs + dsts:
            buf.close(copy_stream)
        copy_stream.sync()
        mr.close()

    @pytest.mark.agent_authored(model="Claude Opus 5")
    @pytest.mark.filterwarnings(OVERLAP_WARNING_FILTER)
    def test_overlap_mode_copies_correctly(self, h2d_bufs, copy_stream):
        """The overlap hint is advisory and must not change the bytes copied."""
        srcs, dsts = h2d_bufs
        for i, src in enumerate(srcs):
            set_buffer(src, i + 90)

        copy_batch(
            copy_stream,
            srcs,
            dsts,
            options=CopyOptions(overlap_mode=MemcpyOverlapMode.PREFER_OVERLAP_WITH_COMPUTE),
        )
        copy_stream.sync()

        for i, dst in enumerate(dsts):
            assert compare_buffer_to_constant(dst, i + 90)

    @pytest.mark.agent_authored(model="Claude Opus 5")
    def test_overlap_mode_warns_only_on_discrete_gpu(self, copy_batch_device, h2d_bufs, copy_stream):
        srcs, dsts = h2d_bufs
        options = CopyOptions(overlap_mode=MemcpyOverlapMode.PREFER_OVERLAP_WITH_COMPUTE)

        if copy_batch_device.properties.integrated:
            # Tegra honours the hint, so no warning should be emitted.
            with warnings.catch_warnings():
                warnings.simplefilter("error", UserWarning)
                copy_batch(copy_stream, srcs, dsts, options=options)
        else:
            with pytest.warns(UserWarning, match="non-integrated"):
                copy_batch(copy_stream, srcs, dsts, options=options)
        copy_stream.sync()

    @pytest.mark.agent_authored(model="Claude Opus 5")
    @pytest.mark.filterwarnings(OVERLAP_WARNING_FILTER)
    def test_default_overlap_mode_does_not_warn(self, h2d_bufs, copy_stream):
        srcs, dsts = h2d_bufs
        with warnings.catch_warnings():
            warnings.simplefilter("error", UserWarning)
            copy_batch(copy_stream, srcs, dsts, options=CopyOptions())
        copy_stream.sync()


class TestCopyOptionsValidation:
    """``CopyOptions`` rejects invalid enum values at construction."""

    @pytest.mark.agent_authored(model="Claude Opus 5")
    def test_invalid_access_order(self):
        with pytest.raises(ValueError, match="invalid src_access_order"):
            CopyOptions(src_access_order="invalid_order")

    @pytest.mark.agent_authored(model="Claude Opus 5")
    def test_invalid_overlap_mode(self):
        with pytest.raises(ValueError, match="invalid overlap_mode"):
            CopyOptions(overlap_mode="invalid_mode")


class TestCopyBatchValidation:
    """``copy_batch`` rejects malformed buffer and option arguments."""

    @pytest.mark.agent_authored(model="Claude Opus 5")
    def test_rejects_single_buffer(self, h2d_bufs, copy_stream):
        srcs, dsts = h2d_bufs

        with pytest.raises(TypeError, match="sequence of Buffers"):
            copy_batch(copy_stream, srcs[0], dsts)

        with pytest.raises(TypeError, match="sequence of Buffers"):
            copy_batch(copy_stream, srcs, dsts[0])

    @pytest.mark.agent_authored(model="Claude Opus 5")
    def test_rejects_empty_sequence(self, h2d_bufs, copy_stream):
        srcs, _ = h2d_bufs

        with pytest.raises(ValueError, match="empty buffers sequence"):
            copy_batch(copy_stream, [], [])

        with pytest.raises(ValueError, match="empty buffers sequence"):
            copy_batch(copy_stream, srcs, [])

    @pytest.mark.agent_authored(model="Claude Opus 5")
    def test_rejects_non_buffer_element(self, h2d_bufs, copy_stream):
        srcs, dsts = h2d_bufs

        with pytest.raises(TypeError, match="expected Buffer, got int"):
            copy_batch(copy_stream, [srcs[0], 42], dsts[:2])

        with pytest.raises(TypeError, match="expected Buffer, got NoneType"):
            copy_batch(copy_stream, srcs[:2], [dsts[0], None])

    @pytest.mark.agent_authored(model="Claude Opus 5")
    def test_rejects_non_sequence(self, h2d_bufs, copy_stream):
        srcs, dsts = h2d_bufs

        with pytest.raises(TypeError, match="must be a sequence of Buffer"):
            copy_batch(copy_stream, 42, dsts)

        with pytest.raises(TypeError, match="must be a sequence of Buffer"):
            copy_batch(copy_stream, srcs, None)

    @pytest.mark.agent_authored(model="Claude Opus 5")
    def test_length_mismatch(self, h2d_bufs, copy_stream):
        srcs, dsts = h2d_bufs

        with pytest.raises(ValueError, match="does not match dsts length"):
            copy_batch(copy_stream, srcs[:2], dsts[:3])

    @pytest.mark.agent_authored(model="Claude Opus 5")
    @pytest.mark.parametrize(("src_size", "dst_size"), [(1024, 2048), (2048, 1024)])
    def test_size_mismatch(self, copy_batch_device, copy_stream, src_size, dst_size):
        """Sizes come from the buffers, so any inequality is an error."""
        pinned_mr = LegacyPinnedMemoryResource()
        src = pinned_mr.allocate(src_size)
        dst = copy_batch_device.memory_resource.allocate(dst_size, stream=copy_stream)

        with pytest.raises(ValueError, match="size mismatch at index 0"):
            copy_batch(copy_stream, [src], [dst])

        src.close(copy_stream)
        dst.close(copy_stream)
        copy_stream.sync()

    @pytest.mark.agent_authored(model="Claude Opus 5")
    def test_options_length_mismatch(self, h2d_bufs, copy_stream):
        srcs, dsts = h2d_bufs

        with pytest.raises(ValueError, match="options length"):
            copy_batch(copy_stream, srcs, dsts, options=[CopyOptions()] * 3)

    @pytest.mark.agent_authored(model="Claude Opus 5")
    def test_rejects_bad_options_type(self, h2d_bufs, copy_stream):
        srcs, dsts = h2d_bufs

        with pytest.raises(TypeError, match="options must be CopyOptions"):
            copy_batch(copy_stream, srcs, dsts, options=42)

    @pytest.mark.agent_authored(model="Claude Opus 5")
    def test_rejects_bad_options_element(self, h2d_bufs, copy_stream):
        srcs, dsts = h2d_bufs
        bad = [CopyOptions()] * (len(srcs) - 1) + ["nope"]

        with pytest.raises(TypeError, match="each options element must be CopyOptions"):
            copy_batch(copy_stream, srcs, dsts, options=bad)


@pytest.mark.agent_authored(model="Claude Opus 5")
def test_cuda12_raises_not_implemented(init_cuda):
    """cuMemcpyBatchAsync is CUDA 13+; single copies use Buffer.copy_to."""
    if binding_version() >= (13, 0, 0):
        pytest.skip("Only relevant on CUDA 12 builds")

    device = Device()
    device.set_current()
    stream = device.create_stream()
    pinned_mr = LegacyPinnedMemoryResource()
    src = pinned_mr.allocate(1024)
    dst = device.memory_resource.allocate(1024, stream=stream)

    with pytest.raises(NotImplementedError, match="CUDA 13"):
        copy_batch(stream, [src], [dst])

    src.close(stream)
    dst.close(stream)
    stream.sync()
    stream.close()
