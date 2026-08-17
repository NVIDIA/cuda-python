# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""CopyOptions support for Buffer.copy_to / Buffer.copy_from (issue #2365)."""

import pytest
from conftest import create_managed_memory_resource_or_skip
from helpers.buffers import compare_equal_buffers, make_scratch_buffer, set_buffer
from helpers.copy_batch import assert_managed_holds

from cuda.core import Device, Host, LegacyPinnedMemoryResource
from cuda.core._stream import LEGACY_DEFAULT_STREAM, PER_THREAD_DEFAULT_STREAM
from cuda.core._utils.version import binding_version, driver_version
from cuda.core.utils import CopyOptions, MemcpyOverlapMode, MemcpySrcAccessOrder

SIZE = 4096


def _options_honored():
    """True when cuMemcpyWithAttributesAsync will actually be used for options.

    Mirrors _with_attributes_available() in _buffer.pyx. CI runs a matrix
    that includes pre-CUDA-13.2 driver/bindings combinations (see
    ci/test-matrix.yml), where this is False and the DURING_API_CALL tests
    below must expect a RuntimeError instead of a successful copy.
    """
    return driver_version() >= (13, 2, 0) and binding_version() >= (13, 2, 0)


@pytest.fixture
def single_copy_device(init_cuda):
    device = Device()
    device.set_current()
    return device


@pytest.fixture
def single_copy_stream(single_copy_device):
    s = single_copy_device.create_stream()
    yield s
    s.close()


@pytest.fixture
def pinned_mr():
    return LegacyPinnedMemoryResource()


@pytest.mark.agent_authored(model="Claude Sonnet 4.6")
def test_options_none_copy_to_data_correct(single_copy_device, single_copy_stream, pinned_mr):
    """options=None (default) continues to copy the right bytes."""
    src = make_scratch_buffer(single_copy_device, 0x55, SIZE)
    dst = pinned_mr.allocate(SIZE)

    src.copy_to(dst, stream=single_copy_stream)
    single_copy_stream.sync()

    assert compare_equal_buffers(src, dst)

    src.close(single_copy_stream)
    single_copy_stream.sync()
    dst.close()


@pytest.mark.agent_authored(model="Claude Sonnet 4.6")
def test_options_none_copy_from_data_correct(single_copy_device, single_copy_stream, pinned_mr):
    """copy_from with options=None copies the right bytes."""
    src = make_scratch_buffer(single_copy_device, 0xAA, SIZE)
    dst = pinned_mr.allocate(SIZE)

    dst.copy_from(src, stream=single_copy_stream)
    single_copy_stream.sync()

    assert compare_equal_buffers(src, dst)

    src.close(single_copy_stream)
    single_copy_stream.sync()
    dst.close()


@pytest.mark.agent_authored(model="Claude Sonnet 4.6")
@pytest.mark.parametrize(
    ("order", "marker"),
    [
        (MemcpySrcAccessOrder.STREAM, 0x31),
        (MemcpySrcAccessOrder.ANY, 0x33),
    ],
)
def test_src_access_order_copy_to(single_copy_device, single_copy_stream, pinned_mr, order, marker):
    """STREAM and ANY are accepted and never corrupt copy_to.

    Both are satisfied by stream-ordered access at worst, so whether
    cuMemcpyWithAttributesAsync actually honors the hint (CUDA 13.2+ driver
    and cuda.bindings) or the call silently falls back to cuMemcpyAsync, the
    copied bytes must be identical either way. DURING_API_CALL is different
    (see test_during_api_call_copy_to): its stronger guarantee cannot be
    silently downgraded, so it is tested separately.
    """
    src = make_scratch_buffer(single_copy_device, marker, SIZE)
    dst = pinned_mr.allocate(SIZE)
    opts = CopyOptions(src_access_order=order)

    src.copy_to(dst, stream=single_copy_stream, options=opts)
    single_copy_stream.sync()

    assert compare_equal_buffers(src, dst)

    src.close(single_copy_stream)
    single_copy_stream.sync()
    dst.close()


@pytest.mark.agent_authored(model="Claude Sonnet 4.6")
@pytest.mark.parametrize(
    ("order", "marker"),
    [
        (MemcpySrcAccessOrder.STREAM, 0x41),
        (MemcpySrcAccessOrder.ANY, 0x43),
    ],
)
def test_src_access_order_copy_from(single_copy_device, single_copy_stream, pinned_mr, order, marker):
    """STREAM and ANY are accepted and never corrupt copy_from. See
    test_src_access_order_copy_to for why DURING_API_CALL is tested
    separately.
    """
    src = make_scratch_buffer(single_copy_device, marker, SIZE)
    dst = pinned_mr.allocate(SIZE)
    opts = CopyOptions(src_access_order=order)

    dst.copy_from(src, stream=single_copy_stream, options=opts)
    single_copy_stream.sync()

    assert compare_equal_buffers(src, dst)

    src.close(single_copy_stream)
    single_copy_stream.sync()
    dst.close()


@pytest.mark.agent_authored(model="Claude Sonnet 4.6")
def test_during_api_call_copy_to(single_copy_device, single_copy_stream, pinned_mr):
    """DURING_API_CALL is honored on the native (CUDA 13.2+) path.

    On the pre-13.2 fallback it must raise RuntimeError instead of silently
    downgrading to stream-ordered cuMemcpyAsync, which cannot honor the
    guarantee that all source reads complete before the call returns (see
    TestRejectUnsupportedDuringApiCall in test_copy_batch_options.py). CI
    runs both driver generations (see ci/test-matrix.yml), so this test must
    handle both outcomes rather than assuming the native path is available.
    """
    src = make_scratch_buffer(single_copy_device, 0x32, SIZE)
    dst = pinned_mr.allocate(SIZE)
    opts = CopyOptions(src_access_order=MemcpySrcAccessOrder.DURING_API_CALL)

    if _options_honored():
        src.copy_to(dst, stream=single_copy_stream, options=opts)
        single_copy_stream.sync()
        assert compare_equal_buffers(src, dst)
    else:
        with pytest.raises(RuntimeError, match="DURING_API_CALL"):
            src.copy_to(dst, stream=single_copy_stream, options=opts)

    src.close(single_copy_stream)
    single_copy_stream.sync()
    dst.close()


@pytest.mark.agent_authored(model="Claude Sonnet 4.6")
def test_during_api_call_copy_from(single_copy_device, single_copy_stream, pinned_mr):
    """Same as test_during_api_call_copy_to, exercising copy_from instead."""
    src = make_scratch_buffer(single_copy_device, 0x42, SIZE)
    dst = pinned_mr.allocate(SIZE)
    opts = CopyOptions(src_access_order=MemcpySrcAccessOrder.DURING_API_CALL)

    if _options_honored():
        dst.copy_from(src, stream=single_copy_stream, options=opts)
        single_copy_stream.sync()
        assert compare_equal_buffers(src, dst)
    else:
        with pytest.raises(RuntimeError, match="DURING_API_CALL"):
            dst.copy_from(src, stream=single_copy_stream, options=opts)

    src.close(single_copy_stream)
    single_copy_stream.sync()
    dst.close()


@pytest.mark.agent_authored(model="Claude Sonnet 4.6")
def test_overlap_mode_copies_correctly(single_copy_device, single_copy_stream, pinned_mr):
    """The overlap hint is advisory and must not change the bytes copied."""
    src = make_scratch_buffer(single_copy_device, 0x77, SIZE)
    dst = pinned_mr.allocate(SIZE)
    opts = CopyOptions(overlap_mode=MemcpyOverlapMode.PREFER_OVERLAP_WITH_COMPUTE)

    src.copy_to(dst, stream=single_copy_stream, options=opts)
    single_copy_stream.sync()

    assert compare_equal_buffers(src, dst)

    src.close(single_copy_stream)
    single_copy_stream.sync()
    dst.close()


@pytest.mark.agent_authored(model="Claude Sonnet 4.6")
def test_legacy_default_stream_token_rejected_with_options(single_copy_device):
    """LEGACY_DEFAULT_STREAM with options raises TypeError, matching copy_batch.

    cuMemcpyWithAttributesAsync rejects the legacy default-stream token
    outright with CUDA_ERROR_INVALID_VALUE on every driver version, so
    copy_to / copy_from surface this before ever calling the driver, just
    like copy_batch does. options=None is unaffected: it never touches the
    attributes path, so LEGACY_DEFAULT_STREAM keeps working as it always has.
    """
    pinned_mr = LegacyPinnedMemoryResource()
    src = pinned_mr.allocate(SIZE)
    dst = pinned_mr.allocate(SIZE)
    opts = CopyOptions(src_access_order=MemcpySrcAccessOrder.ANY)

    with pytest.raises(TypeError, match="LEGACY_DEFAULT_STREAM"):
        src.copy_to(dst, stream=LEGACY_DEFAULT_STREAM, options=opts)

    with pytest.raises(TypeError, match="LEGACY_DEFAULT_STREAM"):
        dst.copy_from(src, stream=LEGACY_DEFAULT_STREAM, options=opts)

    # options=None never reaches the attributes path, so this keeps working.
    src.copy_to(dst, stream=LEGACY_DEFAULT_STREAM)
    single_copy_device.sync()

    src.close()
    dst.close()


@pytest.mark.agent_authored(model="Claude Sonnet 4.6")
def test_per_thread_default_stream_token_accepted_with_options(single_copy_device):
    """PER_THREAD_DEFAULT_STREAM is a real stream to the driver, so options are
    honored on it just like an explicit stream (subject to the usual CUDA
    13.2+ attributes gate), unlike LEGACY_DEFAULT_STREAM.
    """
    pinned_mr = LegacyPinnedMemoryResource()
    opts = CopyOptions(src_access_order=MemcpySrcAccessOrder.ANY)

    src = pinned_mr.allocate(SIZE)
    set_buffer(src, 0x22)
    dst = pinned_mr.allocate(SIZE)
    src.copy_to(dst, stream=PER_THREAD_DEFAULT_STREAM, options=opts)
    single_copy_device.sync()
    assert compare_equal_buffers(src, dst)

    set_buffer(src, 0x23)
    dst.copy_from(src, stream=PER_THREAD_DEFAULT_STREAM, options=opts)
    single_copy_device.sync()
    assert compare_equal_buffers(src, dst)

    src.close()
    dst.close()


@pytest.mark.agent_authored(model="Claude Sonnet 4.6")
def test_location_hints_do_not_corrupt_copy(single_copy_device, single_copy_stream):
    """Device and host location hints are accepted and leave the bytes intact.

    Hints are only honored by the driver for managed memory; for other
    allocation types they are silently ignored. This exercises the
    src_location_hint / dst_location_hint → to_cumemlocation path through
    cuMemcpyWithAttributesAsync rather than cuMemcpyBatchAsync.
    """
    dev = single_copy_device
    mr = create_managed_memory_resource_or_skip()
    src = mr.allocate(SIZE, stream=single_copy_stream)
    dst = mr.allocate(SIZE, stream=single_copy_stream)

    src.fill(0x88, stream=single_copy_stream)

    opts = CopyOptions(
        src_access_order=MemcpySrcAccessOrder.STREAM,
        src_location_hint=dev,
        dst_location_hint=Host(),
    )
    src.copy_to(dst, stream=single_copy_stream, options=opts)

    assert_managed_holds(dev, dst, 0x88, stream=single_copy_stream)

    src.close(single_copy_stream)
    dst.close(single_copy_stream)
    single_copy_stream.sync()
    mr.close()


@pytest.mark.agent_authored(model="Claude Sonnet 4.6")
def test_host_numa_location_hint(single_copy_device, single_copy_stream):
    """A NUMA-specific host hint is accepted and does not corrupt the copy."""
    dev = single_copy_device
    numa_id = dev.properties.host_numa_id
    if numa_id < 0:
        pytest.skip("System does not report a host NUMA node for this device")
    mr = create_managed_memory_resource_or_skip()
    src = mr.allocate(SIZE, stream=single_copy_stream)
    dst = mr.allocate(SIZE, stream=single_copy_stream)

    src.fill(0x99, stream=single_copy_stream)

    opts = CopyOptions(dst_location_hint=Host(numa_id=numa_id))
    src.copy_to(dst, stream=single_copy_stream, options=opts)

    assert_managed_holds(dev, dst, 0x99, stream=single_copy_stream)

    src.close(single_copy_stream)
    dst.close(single_copy_stream)
    single_copy_stream.sync()
    mr.close()


@pytest.mark.agent_authored(model="Claude Sonnet 4.6")
def test_host_numa_current_location_hint(single_copy_device, single_copy_stream):
    """Host.numa_current() as a location hint is accepted and does not corrupt the copy."""
    dev = single_copy_device
    if dev.properties.host_numa_id < 0:
        pytest.skip("System does not report a host NUMA node for this device")
    mr = create_managed_memory_resource_or_skip()
    src = mr.allocate(SIZE, stream=single_copy_stream)
    dst = mr.allocate(SIZE, stream=single_copy_stream)

    src.fill(0xAB, stream=single_copy_stream)

    opts = CopyOptions(dst_location_hint=Host.numa_current())
    src.copy_to(dst, stream=single_copy_stream, options=opts)

    assert_managed_holds(dev, dst, 0xAB, stream=single_copy_stream)

    src.close(single_copy_stream)
    dst.close(single_copy_stream)
    single_copy_stream.sync()
    mr.close()


@pytest.mark.agent_authored(model="Claude Sonnet 4.6")
def test_options_copy_to_data_correct(single_copy_device, single_copy_stream, pinned_mr):
    """copy_to with non-None options copies the right bytes on all driver versions."""
    src = make_scratch_buffer(single_copy_device, 0x77, SIZE)
    dst = pinned_mr.allocate(SIZE)
    opts = CopyOptions(src_access_order=MemcpySrcAccessOrder.ANY)

    src.copy_to(dst, stream=single_copy_stream, options=opts)
    single_copy_stream.sync()

    assert compare_equal_buffers(src, dst)

    src.close(single_copy_stream)
    single_copy_stream.sync()
    dst.close()


@pytest.mark.agent_authored(model="Claude Sonnet 4.6")
def test_options_copy_from_data_correct(single_copy_device, single_copy_stream, pinned_mr):
    """copy_from with non-None options copies the right bytes on all driver versions."""
    src = make_scratch_buffer(single_copy_device, 0x33, SIZE)
    dst = pinned_mr.allocate(SIZE)
    opts = CopyOptions(src_access_order=MemcpySrcAccessOrder.STREAM)

    dst.copy_from(src, stream=single_copy_stream, options=opts)
    single_copy_stream.sync()

    assert compare_equal_buffers(src, dst)

    src.close(single_copy_stream)
    single_copy_stream.sync()
    dst.close()


@pytest.mark.agent_authored(model="Claude Sonnet 4.6")
def test_options_copy_to_rejected_under_graph_capture(single_copy_stream, pinned_mr):
    """copy_to with options raises TypeError when the stream is capturing,
    matching copy_batch. Use GraphNode.memcpy to build attributed copies
    into a graph instead; options=None keeps working under capture as it
    always has (captured as a plain cuMemcpyAsync node).
    """
    src = pinned_mr.allocate(SIZE)
    dst = pinned_mr.allocate(SIZE)
    opts = CopyOptions(src_access_order=MemcpySrcAccessOrder.ANY)

    gb = single_copy_stream.create_graph_builder().begin_building()
    try:
        with pytest.raises(TypeError, match="graph capture"):
            src.copy_to(dst, stream=gb, options=opts)
    finally:
        gb.end_building()
        gb.close()

    src.close()
    dst.close()


@pytest.mark.agent_authored(model="Claude Sonnet 4.6")
def test_options_copy_from_rejected_under_graph_capture(single_copy_stream, pinned_mr):
    """Same as the copy_to variant, exercising copy_from instead."""
    src = pinned_mr.allocate(SIZE)
    dst = pinned_mr.allocate(SIZE)
    opts = CopyOptions(src_access_order=MemcpySrcAccessOrder.STREAM)

    gb = single_copy_stream.create_graph_builder().begin_building()
    try:
        with pytest.raises(TypeError, match="graph capture"):
            dst.copy_from(src, stream=gb, options=opts)
    finally:
        gb.end_building()
        gb.close()

    src.close()
    dst.close()


@pytest.mark.agent_authored(model="Claude Sonnet 4.6")
def test_options_none_copy_to_still_works_under_graph_capture(single_copy_stream, pinned_mr):
    """options=None never touches the attributes path, so copy_to keeps
    working under graph capture exactly as it did before options existed.
    """
    src = pinned_mr.allocate(SIZE)
    set_buffer(src, 0xBB)
    dst = pinned_mr.allocate(SIZE)

    gb = single_copy_stream.create_graph_builder().begin_building()
    src.copy_to(dst, stream=gb)
    graph = gb.end_building().complete()
    graph.launch(single_copy_stream)
    single_copy_stream.sync()

    assert compare_equal_buffers(src, dst)

    src.close()
    dst.close()


@pytest.mark.agent_authored(model="Claude Sonnet 5")
@pytest.mark.parametrize("bad_options", [42, "not-copyoptions", object()])
def test_copy_to_rejects_invalid_options_type(single_copy_stream, pinned_mr, bad_options):
    src = pinned_mr.allocate(SIZE)
    dst = pinned_mr.allocate(SIZE)

    with pytest.raises(TypeError, match="options must be CopyOptions"):
        src.copy_to(dst, stream=single_copy_stream, options=bad_options)

    with pytest.raises(TypeError, match="options must be CopyOptions"):
        dst.copy_from(src, stream=single_copy_stream, options=bad_options)

    with pytest.raises(TypeError, match="options must be CopyOptions"):
        src.copy_to(dst, stream=LEGACY_DEFAULT_STREAM, options="not-copyoptions")

    src.close()
    dst.close()


@pytest.mark.agent_authored(model="Claude Sonnet 4.6")
def test_dst_none_with_options(single_copy_device, single_copy_stream, pinned_mr):
    """dst=None auto-allocation works correctly with options on all driver versions."""
    mr = single_copy_device.memory_resource
    src = mr.allocate(SIZE, stream=single_copy_stream)
    src.fill(0xF0, stream=single_copy_stream)
    opts = CopyOptions(src_access_order=MemcpySrcAccessOrder.ANY)

    dst = src.copy_to(stream=single_copy_stream, options=opts)

    # Read back via pinned buffer to verify bytes.
    host = pinned_mr.allocate(SIZE)
    dst.copy_to(host, stream=single_copy_stream)
    single_copy_stream.sync()

    ref = make_scratch_buffer(single_copy_device, 0xF0, SIZE)
    assert compare_equal_buffers(ref, host)

    src.close(single_copy_stream)
    dst.close(single_copy_stream)
    single_copy_stream.sync()
    host.close()
    ref.close(single_copy_stream)
    single_copy_stream.sync()
