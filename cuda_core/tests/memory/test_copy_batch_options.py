# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""``CopyOptions`` handling and argument validation for ``copy_batch``.

Covers how options are encoded into the driver's attribute runs, how each
option field behaves, and every rejection path.
"""

import pytest

# Shared with test_managed_ops.py: handles the CUDA 13 requirement, mempool
# OOM, and CUDA_ERROR_NOT_SUPPORTED (managed pools are unavailable on
# Windows), so the location-hint tests skip rather than error there.
from conftest import create_managed_memory_resource_or_skip
from helpers.buffers import compare_buffer_to_constant, set_buffer
from helpers.copy_batch import (
    COPY_BATCH_SIZE,
    assert_managed_holds,
)

from cuda.core import Host, LegacyPinnedMemoryResource
from cuda.core._memory._copy_enums import _attr_run_starts, _reject_unsupported_during_api_call
from cuda.core._memory._copy_ops import (
    _normalize_copy_options,
)
from cuda.core._stream import PER_THREAD_DEFAULT_STREAM
from cuda.core._utils.version import binding_version, driver_version
from cuda.core.utils import (
    CopyOptions,
    MemcpyOverlapMode,
    MemcpySrcAccessOrder,
    copy_batch,
)


def _batch_native_available():
    """True when copy_batch will actually use cuMemcpyBatchAsync."""
    return binding_version() >= (13, 0, 0) and driver_version() >= (13, 0, 0)


class TestOptionsEncoding:
    """How ``options`` becomes the driver's ``attrs`` / ``attrsIdxs`` pair.

    Pure logic, no CUDA. This is the only place the effect of ``options``
    is observable: they are hints that change how the driver stages a
    transfer, never the bytes it produces, so no data comparison can
    distinguish an option that was applied from one that was dropped.
    """

    @pytest.mark.agent_authored(model="Claude Opus 5")
    def test_scalar_broadcasts_to_every_copy(self):
        """A scalar must reach all N copies, not just the first."""
        n = 4
        scalar = CopyOptions(src_access_order=MemcpySrcAccessOrder.ANY)

        # copy_batch expands the scalar to one entry per copy...
        assert _normalize_copy_options(scalar, n) == (scalar,) * n
        # ...and the encoder collapses those to a single driver attribute.
        assert _attr_run_starts(_normalize_copy_options(scalar, n)) == [0]

        # An explicit list of the same option is indistinguishable.
        assert _normalize_copy_options([scalar] * n, n) == _normalize_copy_options(scalar, n)

    @pytest.mark.agent_authored(model="Claude Opus 5")
    def test_none_broadcasts_defaults(self):
        assert _normalize_copy_options(None, 3) == (CopyOptions(),) * 3

    @pytest.mark.agent_authored(model="Claude Opus 5")
    def test_sequence_is_never_broadcast(self):
        """A sequence pairs by index, so a short one is an error."""
        scalar = CopyOptions(src_access_order=MemcpySrcAccessOrder.ANY)
        with pytest.raises(ValueError, match="options length"):
            _normalize_copy_options([scalar], 4)

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


class TestRejectUnsupportedDuringApiCall:
    """``_reject_unsupported_during_api_call`` guards the one hazardous fallback.

    Pure logic, no CUDA: this is what both ``Buffer.copy_to``/``copy_from``
    and ``copy_batch`` call before falling back to a plain ``cuMemcpyAsync``
    when the native attributes path is unavailable. STREAM and ANY never
    promise access sooner than stream order, so cuMemcpyAsync satisfies them
    silently; DURING_API_CALL promises all source reads complete before the
    call returns, which cuMemcpyAsync cannot provide, so it must raise
    instead of silently downgrading that guarantee.
    """

    @pytest.mark.agent_authored(model="Claude Sonnet 4.6")
    def test_during_api_call_raises(self):
        with pytest.raises(RuntimeError, match="src_access_order=DURING_API_CALL"):
            _reject_unsupported_during_api_call(MemcpySrcAccessOrder.DURING_API_CALL, "some requirement")

    @pytest.mark.agent_authored(model="Claude Sonnet 4.6")
    def test_during_api_call_message_names_requirement_and_index(self):
        with pytest.raises(RuntimeError, match="requires some requirement") as exc_info:
            _reject_unsupported_during_api_call(MemcpySrcAccessOrder.DURING_API_CALL, "some requirement", index=5)
        assert "at index 5" in str(exc_info.value)

    @pytest.mark.agent_authored(model="Claude Sonnet 4.6")
    def test_during_api_call_message_omits_index_when_not_given(self):
        with pytest.raises(RuntimeError) as exc_info:
            _reject_unsupported_during_api_call(MemcpySrcAccessOrder.DURING_API_CALL, "some requirement")
        assert "at index" not in str(exc_info.value)

    @pytest.mark.agent_authored(model="Claude Sonnet 4.6")
    @pytest.mark.parametrize("order", [MemcpySrcAccessOrder.STREAM, MemcpySrcAccessOrder.ANY])
    def test_stream_and_any_do_not_raise(self, order):
        """Stream-ordered access satisfies both, so no fallback hazard exists."""
        _reject_unsupported_during_api_call(order, "some requirement")
        _reject_unsupported_during_api_call(order, "some requirement", index=0)


class TestCopyBatchOptions:
    """Each ``CopyOptions`` field is accepted and does not corrupt the copy."""

    @pytest.mark.agent_authored(model="Claude Opus 5")
    @pytest.mark.parametrize(
        ("order", "marker"),
        [
            (MemcpySrcAccessOrder.STREAM, 31),
            (MemcpySrcAccessOrder.ANY, 33),
        ],
    )
    def test_src_access_order(self, h2d_bufs, copy_stream, order, marker):
        """STREAM and ANY are accepted and never corrupt the copy.

        Both are satisfied by stream-ordered access at worst, so this holds
        whether the native cuMemcpyBatchAsync path is used or the copy falls
        back to a per-copy cuMemcpyAsync loop. DURING_API_CALL is different
        (see test_during_api_call): its stronger guarantee cannot be
        silently downgraded, so it is tested separately.
        """
        srcs, dsts = h2d_bufs
        for i, src in enumerate(srcs):
            set_buffer(src, i + marker)

        copy_batch(copy_stream, srcs, dsts, options=CopyOptions(src_access_order=order))
        copy_stream.sync()

        for i, dst in enumerate(dsts):
            assert compare_buffer_to_constant(dst, i + marker)

    @pytest.mark.agent_authored(model="Claude Sonnet 4.6")
    def test_during_api_call(self, h2d_bufs, copy_stream):
        """DURING_API_CALL is honored on the native cuMemcpyBatchAsync path.

        On the per-copy cuMemcpyAsync fallback (pre-CUDA-13 build, or
        driver/bindings older than 13.0) it must raise RuntimeError instead
        of silently downgrading to stream-ordered access, which cannot honor
        the guarantee that all source reads complete before the call
        returns (see TestRejectUnsupportedDuringApiCall). CI runs both
        generations (see ci/test-matrix.yml), so this test must handle both
        outcomes rather than assuming the native path is available.
        """
        srcs, dsts = h2d_bufs
        marker = 32
        for i, src in enumerate(srcs):
            set_buffer(src, i + marker)

        opts = CopyOptions(src_access_order=MemcpySrcAccessOrder.DURING_API_CALL)
        if _batch_native_available():
            copy_batch(copy_stream, srcs, dsts, options=opts)
            copy_stream.sync()
            for i, dst in enumerate(dsts):
                assert compare_buffer_to_constant(dst, i + marker)
        else:
            with pytest.raises(RuntimeError, match="DURING_API_CALL"):
                copy_batch(copy_stream, srcs, dsts, options=opts)

    @pytest.mark.agent_authored(model="Claude Opus 5")
    def test_per_copy_options(self, h2d_bufs, copy_stream):
        srcs, dsts = h2d_bufs
        for i, src in enumerate(srcs):
            set_buffer(src, i + 40)

        # DURING_API_CALL is deliberately excluded here: it raises RuntimeError
        # rather than silently falling back on pre-CUDA-13 driver/bindings (see
        # test_during_api_call), which CI also exercises (ci/test-matrix.yml).
        # STREAM and ANY are enough to prove distinct per-copy options don't
        # corrupt the data; the encoding itself is covered by TestOptionsEncoding.
        per_copy_options = [
            CopyOptions(src_access_order=MemcpySrcAccessOrder.STREAM),
            CopyOptions(src_access_order=MemcpySrcAccessOrder.ANY),
            CopyOptions(src_access_order=MemcpySrcAccessOrder.ANY),
            CopyOptions(src_access_order=MemcpySrcAccessOrder.STREAM),
        ]
        copy_batch(copy_stream, srcs, dsts, options=per_copy_options)
        copy_stream.sync()

        for i, dst in enumerate(dsts):
            assert compare_buffer_to_constant(dst, i + 40)

    @pytest.mark.agent_authored(model="Claude Opus 5")
    def test_location_hints_do_not_corrupt_copy(self, copy_batch_device, copy_stream):
        """Device and host hints are accepted and leave the bytes intact.

        Hints only steer how the driver stages a transfer, so no data
        comparison can show one was *applied*; what this catches is a hint
        that errors or corrupts. It is also the only test that drives the
        ``device`` and ``host`` branches of ``to_cumemlocation`` and the
        ``src_location_hint`` path through ``copy_batch``.
        """
        dev = copy_batch_device
        mr = create_managed_memory_resource_or_skip()
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

    @pytest.mark.agent_authored(model="Claude Sonnet 4.6")
    def test_host_numa_location_hint(self, copy_batch_device, copy_stream):
        """A NUMA-specific host hint is accepted and does not corrupt the copy."""
        dev = copy_batch_device
        numa_id = dev.properties.host_numa_id
        if numa_id < 0:
            pytest.skip("System does not report a host NUMA node for this device")
        mr = create_managed_memory_resource_or_skip()
        srcs = [mr.allocate(COPY_BATCH_SIZE, stream=copy_stream) for _ in range(2)]
        dsts = [mr.allocate(COPY_BATCH_SIZE, stream=copy_stream) for _ in range(2)]
        for i, src in enumerate(srcs):
            src.fill(i + 85, stream=copy_stream)

        copy_batch(copy_stream, srcs, dsts, options=CopyOptions(dst_location_hint=Host(numa_id=numa_id)))
        copy_stream.sync()
        for i, dst in enumerate(dsts):
            assert_managed_holds(dev, dst, i + 85, stream=copy_stream)

        for buf in srcs + dsts:
            buf.close(copy_stream)
        copy_stream.sync()
        mr.close()

    @pytest.mark.agent_authored(model="Claude Sonnet 4.6")
    def test_host_numa_current_location_hint(self, copy_batch_device, copy_stream):
        """Host.numa_current() as a location hint is accepted and does not corrupt the copy."""
        dev = copy_batch_device
        if dev.properties.host_numa_id < 0:
            pytest.skip("System does not report a host NUMA node for this device")
        mr = create_managed_memory_resource_or_skip()
        srcs = [mr.allocate(COPY_BATCH_SIZE, stream=copy_stream) for _ in range(2)]
        dsts = [mr.allocate(COPY_BATCH_SIZE, stream=copy_stream) for _ in range(2)]
        for i, src in enumerate(srcs):
            src.fill(i + 86, stream=copy_stream)

        copy_batch(copy_stream, srcs, dsts, options=CopyOptions(dst_location_hint=Host.numa_current()))
        copy_stream.sync()
        for i, dst in enumerate(dsts):
            assert_managed_holds(dev, dst, i + 86, stream=copy_stream)

        for buf in srcs + dsts:
            buf.close(copy_stream)
        copy_stream.sync()
        mr.close()

    @pytest.mark.agent_authored(model="Claude Opus 5")
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
    def test_default_overlap_mode_does_not_warn(self, h2d_bufs, copy_stream):
        srcs, dsts = h2d_bufs
        copy_batch(copy_stream, srcs, dsts, options=CopyOptions())
        copy_stream.sync()

    @pytest.mark.agent_authored(model="Claude Sonnet 4.6")
    def test_options_on_per_thread_default_stream(self, copy_batch_device):
        """CopyOptions work on PER_THREAD_DEFAULT_STREAM like any explicit stream.

        Unlike LEGACY_DEFAULT_STREAM (rejected outright, see
        TestCopyBatchStreamSemantics in test_copy_batch.py),
        PER_THREAD_DEFAULT_STREAM is a real stream to cuMemcpyBatchAsync.
        """
        pinned_mr = LegacyPinnedMemoryResource()
        device_mr = copy_batch_device.memory_resource
        src = pinned_mr.allocate(COPY_BATCH_SIZE)
        dst = device_mr.allocate(COPY_BATCH_SIZE, stream=PER_THREAD_DEFAULT_STREAM)
        set_buffer(src, 44)

        copy_batch(
            PER_THREAD_DEFAULT_STREAM,
            [src],
            [dst],
            options=CopyOptions(src_access_order=MemcpySrcAccessOrder.ANY),
        )
        copy_batch_device.sync()

        assert compare_buffer_to_constant(dst, 44)

        src.close(PER_THREAD_DEFAULT_STREAM)
        dst.close(PER_THREAD_DEFAULT_STREAM)
        copy_batch_device.sync()


class TestCopyOptionsValidation:
    """``CopyOptions`` rejects invalid enum values at construction."""

    @pytest.mark.agent_authored(model="Claude Sonnet 4.6")
    def test_type_hints_resolvable(self):
        """All annotations on CopyOptions must resolve without NameError."""
        import typing

        typing.get_type_hints(CopyOptions)

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
