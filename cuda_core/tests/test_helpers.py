# SPDX-FileCopyrightText: Copyright (c) 2024-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

import time

import pytest
from helpers import va_reservation
from helpers.buffers import PatternGen, compare_equal_buffers, make_scratch_buffer
from helpers.latch import LatchKernel
from helpers.logging import TimestampedLogger

from cuda.core import Device
from cuda_python_test_helpers import under_compute_sanitizer

ENABLE_LOGGING = False  # Set True for test debugging and development
NBYTES = 64


@pytest.mark.skipif(Device().compute_capability.major < 7, reason="__nanosleep is only available starting Volta (sm70)")
def test_latchkernel():
    """Test LatchKernel."""
    log = TimestampedLogger(enabled=ENABLE_LOGGING)
    log("begin")
    device = Device()
    device.set_current()
    stream = device.create_stream()
    target = make_scratch_buffer(device, 0, NBYTES)
    zeros = make_scratch_buffer(device, 0, NBYTES)
    ones = make_scratch_buffer(device, 1, NBYTES)
    latch = LatchKernel(device)
    log("launching latch kernel")
    latch.launch(stream)
    log("launching copy (0->1) kernel")
    target.copy_from(ones, stream=stream)
    log("going to sleep")
    time.sleep(1)
    if device.properties.concurrent_managed_access:
        # Host access to managed memory while a kernel is active is unsafe on
        # devices without concurrent managed access.
        log("checking target == 0")
        assert compare_equal_buffers(target, zeros)
    log("releasing latch and syncing")
    latch.release()
    stream.sync()
    log("checking target == 1")
    assert compare_equal_buffers(target, ones)
    log("done")


@pytest.mark.skipif(
    under_compute_sanitizer(),
    reason="Too slow under compute-sanitizer (UVM-heavy test).",
)
def test_patterngen_seeds():
    """Test PatternGen with seed argument."""
    device = Device()
    device.set_current()
    buffer = make_scratch_buffer(device, 0, NBYTES)

    # All seeds are pairwise different.
    # We test a sampling of values because exhaustive testing is too slow,
    # especially on Windows. See https://github.com/NVIDIA/cuda-python/issues/1455
    pgen = PatternGen(device, NBYTES)
    for i in (ii for ii in range(256) if ii < 5 or ii % 17 == 0):
        pgen.fill_buffer(buffer, seed=i)
        pgen.verify_buffer(buffer, seed=i)
        for j in (jj for jj in range(i + 1, 256) if jj < 5 or jj % 19 == 0):
            with pytest.raises(AssertionError):
                pgen.verify_buffer(buffer, seed=j)


def test_patterngen_values():
    """Test PatternGen with value argument, also compare_equal_buffers."""
    device = Device()
    device.set_current()
    ones = make_scratch_buffer(device, 1, NBYTES)
    twos = make_scratch_buffer(device, 2, NBYTES)
    assert compare_equal_buffers(ones, ones)
    assert not compare_equal_buffers(ones, twos)
    pgen = PatternGen(device, NBYTES)
    pgen.verify_buffer(ones, value=1)
    pgen.verify_buffer(twos, value=2)


# helpers.va_reservation (issue #2381). The GPU-dependent part runs once per
# session in an autouse fixture, so only the pure logic is covered here.


def _fake_reserve(limit, log=None):
    """Grants any reservation up to ``limit``, recording what was asked."""

    def reserve(size):
        if log is not None:
            log.append(size)
        return size <= limit

    return reserve


@pytest.mark.agent_authored(model="claude-opus-5")
def test_va_probe_finds_the_boundary():
    limit = 700 * va_reservation.GIB
    found = va_reservation.largest_reservable(reserve=_fake_reserve(limit))

    # Refinement should land within a few percent, never above the real limit.
    assert found <= limit
    assert found > limit * 0.9


@pytest.mark.agent_authored(model="claude-opus-5")
def test_va_probe_descends_so_only_one_grant_is_paid_for():
    # Releasing a granted reservation is the expensive half; a refused one costs
    # nothing. Ascending from the granularity would pay for every rung.
    asked = []
    limit = 700 * va_reservation.GIB
    va_reservation.largest_reservable(reserve=_fake_reserve(limit, asked), refine_steps=0)

    assert sum(1 for size in asked if size <= limit) == 1
    assert asked[0] == va_reservation.MAX_PROBE_BYTES


@pytest.mark.agent_authored(model="claude-opus-5")
def test_va_probe_reports_zero_when_nothing_can_be_reserved():
    assert va_reservation.largest_reservable(reserve=_fake_reserve(0)) == 0


@pytest.mark.agent_authored(model="claude-opus-5")
def test_va_probe_only_asks_for_aligned_sizes():
    # Device memory is not a multiple of the 2 MiB granularity, and an unaligned
    # ask fails with CUDA_ERROR_INVALID_VALUE at every size -- which would read
    # as an exhausted address space rather than as a bug here.
    asked = []
    va_reservation.largest_reservable(reserve=_fake_reserve(3 * 25650855936, asked))

    assert asked and all(size % va_reservation.VA_ALIGNMENT == 0 for size in asked)


@pytest.mark.agent_authored(model="claude-opus-5")
def test_reservation_records_failure_without_raising():
    def boom():
        raise RuntimeError("CUDA_ERROR_OUT_OF_MEMORY: nope")

    item = va_reservation.Reservation("default device mempool", "cuDeviceGetMemPool", boom)

    assert item.run() is False
    assert item.succeeded is False
    assert "CUDA_ERROR_OUT_OF_MEMORY" in item.error


@pytest.mark.agent_authored(model="claude-opus-5")
def test_failure_message_names_the_pool_and_the_size_it_needed():
    failed = va_reservation.Reservation("graph memory pool", "cuGraphAddMemAllocNode", lambda: 1 / 0)
    failed.run()
    report = va_reservation.ReservationReport(
        "NVIDIA Graphics Device", 191998918656, 300 * va_reservation.GIB, 300 * va_reservation.GIB, [failed], True
    )

    message = va_reservation.build_failure_message(report)

    assert "cannot run on this machine" in message
    assert "graph memory pool" in message
    assert "357.63 GiB" in message  # 2x installed device memory
    assert "address space" in message
    assert "#2381" in message


@pytest.mark.agent_authored(model="claude-opus-5")
def test_report_lines_cover_both_driver_pools():
    ok_pools = [
        va_reservation.Reservation("default device mempool", "cuDeviceGetMemPool", lambda: None),
        va_reservation.Reservation("graph memory pool", "cuGraphAddMemAllocNode", lambda: None),
    ]
    for item in ok_pools:
        item.run()
    report = va_reservation.ReservationReport(
        "dev", 25650855936, 900 * va_reservation.GIB, 800 * va_reservation.GIB, ok_pools, True
    )

    text = "\n".join(report.lines())

    assert report.failed == []
    assert "default device mempool" in text
    assert "graph memory pool" in text
    assert "shrank by 100.00 GiB" in text
    assert "more pool-sized" in text
