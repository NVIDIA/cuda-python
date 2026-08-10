# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Data movement behaviour of ``copy_batch``.

Covers that the right bytes reach the right destination, that batched
results agree with the per-buffer ``Buffer.copy_to`` path, and that the
batch is correctly ordered on its stream. Options handling and argument
validation live in ``test_copy_batch_options.py``.
"""

import pytest
from helpers.buffers import (
    compare_buffer_to_constant,
    compare_equal_buffers,
    make_scratch_buffer,
    set_buffer,
)
from helpers.copy_batch import COPY_BATCH_SIZE

from cuda.core import LegacyPinnedMemoryResource
from cuda.core._utils.cuda_utils import CUDAError
from cuda.core.utils import copy_batch


class TestCopyBatchCore:
    """Each transfer direction moves the expected bytes."""

    @pytest.mark.agent_authored(model="Claude Opus 5")
    def test_h2d_batch(self, h2d_bufs, copy_stream):
        srcs, dsts = h2d_bufs
        for i, src in enumerate(srcs):
            set_buffer(src, i + 1)

        copy_batch(copy_stream, srcs, dsts)
        copy_stream.sync()

        for i, dst in enumerate(dsts):
            assert compare_buffer_to_constant(dst, i + 1)

    @pytest.mark.agent_authored(model="Claude Opus 5")
    def test_d2h_batch(self, copy_batch_device, h2d_bufs, copy_stream):
        dev = copy_batch_device
        _, device_dsts = h2d_bufs
        pinned_mr = LegacyPinnedMemoryResource()

        for i, buf in enumerate(device_dsts):
            buf.fill(i + 10, stream=copy_stream)

        host_bufs = [pinned_mr.allocate(COPY_BATCH_SIZE) for _ in device_dsts]
        copy_batch(copy_stream, device_dsts, host_bufs)
        copy_stream.sync()

        for i, host_buf in enumerate(host_bufs):
            expected = make_scratch_buffer(dev, i + 10, COPY_BATCH_SIZE)
            assert compare_equal_buffers(expected, host_buf)
            expected.close()
            host_buf.close(copy_stream)
        copy_stream.sync()

    @pytest.mark.agent_authored(model="Claude Opus 5")
    def test_d2d_batch(self, device_bufs, copy_stream):
        srcs, dsts = device_bufs
        for i, src in enumerate(srcs):
            src.fill(i + 20, stream=copy_stream)

        copy_batch(copy_stream, srcs, dsts)
        copy_stream.sync()

        for i, dst in enumerate(dsts):
            assert compare_buffer_to_constant(dst, i + 20)

    @pytest.mark.agent_authored(model="Claude Opus 5")
    def test_various_sizes(self, copy_batch_device, copy_stream):
        pinned_mr = LegacyPinnedMemoryResource()
        device_mr = copy_batch_device.memory_resource
        sizes = [1024, 2048, 512, 4096]

        srcs = [pinned_mr.allocate(size) for size in sizes]
        dsts = [device_mr.allocate(size, stream=copy_stream) for size in sizes]
        for i, src in enumerate(srcs):
            set_buffer(src, i + 1)

        copy_batch(copy_stream, srcs, dsts)
        copy_stream.sync()

        for i, dst in enumerate(dsts):
            assert compare_buffer_to_constant(dst, i + 1)

        for buf in srcs + dsts:
            buf.close(copy_stream)
        copy_stream.sync()

    @pytest.mark.agent_authored(model="Claude Opus 5")
    def test_single_element_batch(self, copy_batch_device, copy_stream):
        """A one-element batch is legal; only a bare Buffer is rejected."""
        pinned_mr = LegacyPinnedMemoryResource()
        src = pinned_mr.allocate(COPY_BATCH_SIZE)
        dst = copy_batch_device.memory_resource.allocate(COPY_BATCH_SIZE, stream=copy_stream)
        set_buffer(src, 7)

        copy_batch(copy_stream, [src], [dst])
        copy_stream.sync()

        assert compare_buffer_to_constant(dst, 7)
        src.close(copy_stream)
        dst.close(copy_stream)
        copy_stream.sync()


class TestCopyBatchEquivalence:
    """Batched results must agree with the already-tested per-buffer path.

    ``Buffer.copy_to`` and ``Buffer.copy_from`` have their own coverage in
    ``tests/test_memory.py``, so agreement between the two paths is the
    property under test here.
    """

    @pytest.mark.agent_authored(model="Claude Opus 5")
    def test_batch_matches_sequential_copy_to(self, copy_batch_device, h2d_bufs, copy_stream):
        srcs, _ = h2d_bufs
        device_mr = copy_batch_device.memory_resource
        pinned_mr = LegacyPinnedMemoryResource()

        for i, src in enumerate(srcs):
            set_buffer(src, i + 50)

        seq_dsts = [device_mr.allocate(COPY_BATCH_SIZE, stream=copy_stream) for _ in srcs]
        for src, dst in zip(srcs, seq_dsts):
            src.copy_to(dst, stream=copy_stream)

        batch_dsts = [device_mr.allocate(COPY_BATCH_SIZE, stream=copy_stream) for _ in srcs]
        copy_batch(copy_stream, srcs, batch_dsts)
        copy_stream.sync()

        for seq_dst, batch_dst in zip(seq_dsts, batch_dsts):
            seq_host = pinned_mr.allocate(COPY_BATCH_SIZE)
            batch_host = pinned_mr.allocate(COPY_BATCH_SIZE)
            seq_dst.copy_to(seq_host, stream=copy_stream)
            batch_dst.copy_to(batch_host, stream=copy_stream)
            copy_stream.sync()
            assert compare_equal_buffers(seq_host, batch_host)
            seq_host.close(copy_stream)
            batch_host.close(copy_stream)

        for buf in seq_dsts + batch_dsts:
            buf.close(copy_stream)
        copy_stream.sync()

    @pytest.mark.agent_authored(model="Claude Opus 5")
    def test_batch_matches_sequential_varied_sizes(self, copy_batch_device, copy_stream):
        device_mr = copy_batch_device.memory_resource
        pinned_mr = LegacyPinnedMemoryResource()
        sizes = [1024, 2048, 512]

        srcs = [pinned_mr.allocate(size) for size in sizes]
        for i, src in enumerate(srcs):
            set_buffer(src, i + 60)

        seq_dsts = [device_mr.allocate(size, stream=copy_stream) for size in sizes]
        for src, dst in zip(srcs, seq_dsts):
            src.copy_to(dst, stream=copy_stream)

        batch_dsts = [device_mr.allocate(size, stream=copy_stream) for size in sizes]
        copy_batch(copy_stream, srcs, batch_dsts)
        copy_stream.sync()

        for size, seq_dst, batch_dst in zip(sizes, seq_dsts, batch_dsts):
            seq_host = pinned_mr.allocate(size)
            batch_host = pinned_mr.allocate(size)
            seq_dst.copy_to(seq_host, stream=copy_stream)
            batch_dst.copy_to(batch_host, stream=copy_stream)
            copy_stream.sync()
            assert compare_equal_buffers(seq_host, batch_host)
            seq_host.close(copy_stream)
            batch_host.close(copy_stream)

        for buf in srcs + seq_dsts + batch_dsts:
            buf.close(copy_stream)
        copy_stream.sync()

    @pytest.mark.agent_authored(model="Claude Opus 5")
    def test_batch_matches_sequential_d2d(self, copy_batch_device, device_bufs, copy_stream):
        srcs, seq_dsts = device_bufs
        device_mr = copy_batch_device.memory_resource
        pinned_mr = LegacyPinnedMemoryResource()

        for i, src in enumerate(srcs):
            src.fill(i + 70, stream=copy_stream)

        for src, dst in zip(srcs, seq_dsts):
            src.copy_to(dst, stream=copy_stream)

        batch_dsts = [device_mr.allocate(COPY_BATCH_SIZE, stream=copy_stream) for _ in srcs]
        copy_batch(copy_stream, srcs, batch_dsts)
        copy_stream.sync()

        for seq_dst, batch_dst in zip(seq_dsts, batch_dsts):
            seq_host = pinned_mr.allocate(COPY_BATCH_SIZE)
            batch_host = pinned_mr.allocate(COPY_BATCH_SIZE)
            seq_dst.copy_to(seq_host, stream=copy_stream)
            batch_dst.copy_to(batch_host, stream=copy_stream)
            copy_stream.sync()
            assert compare_equal_buffers(seq_host, batch_host)
            seq_host.close(copy_stream)
            batch_host.close(copy_stream)

        for buf in batch_dsts:
            buf.close(copy_stream)
        copy_stream.sync()


class TestCopyBatchStreamSemantics:
    """Where the batch sits in stream order, and what it cannot be part of."""

    @pytest.mark.agent_authored(model="Claude Opus 5")
    def test_ordered_between_prior_and_later_stream_work(self, device_bufs, copy_stream):
        """The batch must observe prior stream work and precede later work.

        Each source is filled with ``before``, copied, then refilled with
        ``after`` -- all enqueued on one stream with no intervening sync.
        Destinations holding ``before`` prove the copy ran after the first
        fill and before the second, rather than racing either.
        """
        srcs, dsts = device_bufs
        before, after = 11, 22

        for src in srcs:
            src.fill(before, stream=copy_stream)
        copy_batch(copy_stream, srcs, dsts)
        for src in srcs:
            src.fill(after, stream=copy_stream)

        copy_stream.sync()

        for dst in dsts:
            assert compare_buffer_to_constant(dst, before)
        for src in srcs:
            assert compare_buffer_to_constant(src, after)

    @pytest.mark.agent_authored(model="Claude Opus 5")
    def test_visible_on_other_stream_after_explicit_wait(self, copy_batch_device, device_bufs, copy_stream):
        """A batch on one stream is not ordered against an unrelated stream.

        The second stream must be made to wait explicitly; once it does,
        it observes the copied bytes.
        """
        srcs, dsts = device_bufs
        other = copy_batch_device.create_stream()
        try:
            for src in srcs:
                src.fill(33, stream=copy_stream)
            copy_batch(copy_stream, srcs, dsts)

            # Explicit cross-stream dependency, then observe from `other`.
            other.wait(copy_stream)
            probes = [copy_batch_device.memory_resource.allocate(COPY_BATCH_SIZE, stream=other) for _ in dsts]
            for dst, probe in zip(dsts, probes):
                dst.copy_to(probe, stream=other)
            other.sync()

            for probe in probes:
                assert compare_buffer_to_constant(probe, 33)
            for probe in probes:
                probe.close(other)
            other.sync()
        finally:
            other.close()

    @pytest.mark.agent_authored(model="Claude Opus 5")
    def test_graph_builder_capture_is_unsupported(self, copy_batch_device, device_bufs, copy_stream):
        """Batched memcpy cannot be captured into a graph.

        ``Stream_accept`` takes a ``GraphBuilder``, so the batch reaches
        the driver, but ``cuMemcpyBatchAsync`` has no graph-node form and
        the driver rejects it mid-capture. Pinned here so the limitation
        is asserted rather than rediscovered; use ``GraphNode.memcpy`` or
        per-buffer ``Buffer.copy_to`` to build copies into a graph.
        """
        srcs, dsts = device_bufs
        gb = copy_batch_device.create_graph_builder().begin_building()
        try:
            with pytest.raises(CUDAError, match="STREAM_CAPTURE_UNSUPPORTED"):
                copy_batch(gb, srcs, dsts)
        finally:
            # The rejection leaves the capture intact, so it still ends cleanly.
            gb.end_building()
            gb.close()
