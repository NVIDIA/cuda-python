# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""CopyOptions support for Buffer.copy_to / Buffer.copy_from (issue #2365)."""

import pytest
from conftest import create_managed_memory_resource_or_skip
from helpers.buffers import compare_equal_buffers, make_scratch_buffer
from helpers.copy_batch import assert_managed_holds

from cuda.core import Device, Host, LegacyPinnedMemoryResource
from cuda.core._stream import LEGACY_DEFAULT_STREAM, PER_THREAD_DEFAULT_STREAM
from cuda.core._utils.version import binding_version, driver_version
from cuda.core.utils import CopyOptions, MemcpyOverlapMode, MemcpySrcAccessOrder

SIZE = 4096


def _options_honored():
    """True when cuMemcpyWithAttributesAsync will be used for options."""
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
        (MemcpySrcAccessOrder.DURING_API_CALL, 0x32),
        (MemcpySrcAccessOrder.ANY, 0x33),
    ],
)
def test_src_access_order_copy_to(single_copy_device, single_copy_stream, pinned_mr, order, marker):
    """Every src_access_order value is accepted and does not corrupt copy_to."""
    src = make_scratch_buffer(single_copy_device, marker, SIZE)
    dst = pinned_mr.allocate(SIZE)
    opts = CopyOptions(src_access_order=order)

    if _options_honored():
        src.copy_to(dst, stream=single_copy_stream, options=opts)
    else:
        with pytest.warns(UserWarning, match="CopyOptions are not honored"):
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
        (MemcpySrcAccessOrder.DURING_API_CALL, 0x42),
        (MemcpySrcAccessOrder.ANY, 0x43),
    ],
)
def test_src_access_order_copy_from(single_copy_device, single_copy_stream, pinned_mr, order, marker):
    """Every src_access_order value is accepted and does not corrupt copy_from."""
    src = make_scratch_buffer(single_copy_device, marker, SIZE)
    dst = pinned_mr.allocate(SIZE)
    opts = CopyOptions(src_access_order=order)

    if _options_honored():
        dst.copy_from(src, stream=single_copy_stream, options=opts)
    else:
        with pytest.warns(UserWarning, match="CopyOptions are not honored"):
            dst.copy_from(src, stream=single_copy_stream, options=opts)
    single_copy_stream.sync()

    assert compare_equal_buffers(src, dst)

    src.close(single_copy_stream)
    single_copy_stream.sync()
    dst.close()


@pytest.mark.agent_authored(model="Claude Sonnet 4.6")
def test_overlap_mode_copies_correctly(single_copy_device, single_copy_stream, pinned_mr):
    """The overlap hint is advisory and must not change the bytes copied."""
    src = make_scratch_buffer(single_copy_device, 0x77, SIZE)
    dst = pinned_mr.allocate(SIZE)
    opts = CopyOptions(overlap_mode=MemcpyOverlapMode.PREFER_OVERLAP_WITH_COMPUTE)

    if _options_honored():
        src.copy_to(dst, stream=single_copy_stream, options=opts)
    else:
        with pytest.warns(UserWarning, match="CopyOptions are not honored"):
            src.copy_to(dst, stream=single_copy_stream, options=opts)
    single_copy_stream.sync()

    assert compare_equal_buffers(src, dst)

    src.close(single_copy_stream)
    single_copy_stream.sync()
    dst.close()


@pytest.mark.agent_authored(model="Claude Sonnet 4.6")
@pytest.mark.parametrize(
    "default_stream_token",
    [LEGACY_DEFAULT_STREAM, PER_THREAD_DEFAULT_STREAM],
    ids=["legacy", "per_thread"],
)
def test_default_stream_token_accepted_with_options(single_copy_device, default_stream_token):
    """Default-stream tokens warn+fallback with options (cuMemcpyWithAttributesAsync rejects them).

    Unlike copy_batch (which raises TypeError), single-copy accepts the token but
    falls back to cuMemcpyAsync with a UserWarning because the attributes API does
    not support default-stream sentinels.
    """
    pinned_mr = LegacyPinnedMemoryResource()
    src = pinned_mr.allocate(SIZE)
    dst = pinned_mr.allocate(SIZE)
    opts = CopyOptions(src_access_order=MemcpySrcAccessOrder.ANY)

    # A warning is always emitted: on 13.2+ because the attributes API rejects
    # default-stream tokens; on older drivers for the version-gate reason.
    with pytest.warns(UserWarning, match="CopyOptions are not honored"):
        src.copy_to(dst, stream=default_stream_token, options=opts)
    single_copy_device.sync()
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
    if _options_honored():
        src.copy_to(dst, stream=single_copy_stream, options=opts)
    else:
        with pytest.warns(UserWarning, match="CopyOptions are not honored"):
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
    if _options_honored():
        src.copy_to(dst, stream=single_copy_stream, options=opts)
    else:
        with pytest.warns(UserWarning, match="CopyOptions are not honored"):
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
    if _options_honored():
        src.copy_to(dst, stream=single_copy_stream, options=opts)
    else:
        with pytest.warns(UserWarning, match="CopyOptions are not honored"):
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

    if _options_honored():
        src.copy_to(dst, stream=single_copy_stream, options=opts)
    else:
        with pytest.warns(UserWarning, match="CopyOptions are not honored"):
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

    if _options_honored():
        dst.copy_from(src, stream=single_copy_stream, options=opts)
    else:
        with pytest.warns(UserWarning, match="CopyOptions are not honored"):
            dst.copy_from(src, stream=single_copy_stream, options=opts)
    single_copy_stream.sync()

    assert compare_equal_buffers(src, dst)

    src.close(single_copy_stream)
    single_copy_stream.sync()
    dst.close()


@pytest.mark.agent_authored(model="Claude Sonnet 4.6")
def test_options_copy_to_warns_under_graph_capture(single_copy_device, single_copy_stream, pinned_mr):
    """copy_to warns and falls back to cuMemcpyAsync when the stream is capturing."""
    src = make_scratch_buffer(single_copy_device, 0xBB, SIZE)
    dst = pinned_mr.allocate(SIZE)

    gb = single_copy_stream.create_graph_builder().begin_building()
    with pytest.warns(UserWarning, match="CopyOptions are not honored"):
        src.copy_to(dst, stream=gb, options=CopyOptions(src_access_order=MemcpySrcAccessOrder.ANY))
    graph = gb.end_building().complete()
    graph.launch(single_copy_stream)
    single_copy_stream.sync()

    assert compare_equal_buffers(src, dst)

    src.close(single_copy_stream)
    single_copy_stream.sync()
    dst.close()


@pytest.mark.agent_authored(model="Claude Sonnet 4.6")
def test_options_copy_from_warns_under_graph_capture(single_copy_device, single_copy_stream, pinned_mr):
    """copy_from warns and falls back to cuMemcpyAsync when the stream is capturing."""
    src = make_scratch_buffer(single_copy_device, 0xCC, SIZE)
    dst = pinned_mr.allocate(SIZE)

    gb = single_copy_stream.create_graph_builder().begin_building()
    with pytest.warns(UserWarning, match="CopyOptions are not honored"):
        dst.copy_from(src, stream=gb, options=CopyOptions(src_access_order=MemcpySrcAccessOrder.STREAM))
    graph = gb.end_building().complete()
    graph.launch(single_copy_stream)
    single_copy_stream.sync()

    assert compare_equal_buffers(src, dst)

    src.close(single_copy_stream)
    single_copy_stream.sync()
    dst.close()


@pytest.mark.agent_authored(model="Claude Sonnet 4.6")
def test_dst_none_with_options(single_copy_device, single_copy_stream, pinned_mr):
    """dst=None auto-allocation works correctly with options on all driver versions."""
    mr = single_copy_device.memory_resource
    src = mr.allocate(SIZE, stream=single_copy_stream)
    src.fill(0xF0, stream=single_copy_stream)
    opts = CopyOptions(src_access_order=MemcpySrcAccessOrder.ANY)

    if _options_honored():
        dst = src.copy_to(stream=single_copy_stream, options=opts)
    else:
        with pytest.warns(UserWarning, match="CopyOptions are not honored"):
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
