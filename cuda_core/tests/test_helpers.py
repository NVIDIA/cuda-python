# SPDX-FileCopyrightText: Copyright (c) 2024-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

import time
import types

import pytest
from helpers.buffers import PatternGen, compare_equal_buffers, make_scratch_buffer
from helpers.latch import LatchKernel
from helpers.logging import TimestampedLogger
from helpers.oom_diagnostics import (
    DEFAULT_FILENAME,
    OOM_MARKER,
    OomDiagnosticsRecorder,
    ProbeSnapshot,
    classify,
    probe_basics,
    record_if_oom,
    report_terminal_summary,
)
from helpers.oom_diagnostics import _round_up as oom_round_up  # white-box test of the alignment guard

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


# helpers.oom_diagnostics (issue #2381): a pytest harness plus an OOM reason
# checker that tells host virtual-address exhaustion apart from physical
# device memory exhaustion. Both halves are non-trivial infrastructure (see
# discussion on #2471), so they get tests. The harness and classifier tests
# below never touch the driver: they inject a `ProbeSnapshot` (or none of the
# recorder needs one at all) so a broken assertion here can't be the thing
# that materializes a session's first ~2x-device-memory pool reservation.
# Only test_oom_diagnostics_probe_basics_is_live_and_cheap talks to the
# driver, and only through the side-effect-free prefix of the checker.


def _fake_report(failed):
    return types.SimpleNamespace(failed=failed)


def _fake_call(exc_text, when="call"):
    excinfo = None if exc_text is None else types.SimpleNamespace(value=RuntimeError(exc_text))
    return types.SimpleNamespace(excinfo=excinfo, when=when)


def _fake_item(rootpath, nodeid="tests/test_x.py::test_a"):
    # get_plugin returns None so record_if_oom skips the terminal write.
    config = types.SimpleNamespace(
        rootpath=rootpath,
        pluginmanager=types.SimpleNamespace(get_plugin=lambda _: None),
    )
    return types.SimpleNamespace(nodeid=nodeid, config=config)


# A snapshot that short-circuits classify()/format_probe_log() at the very
# first check, so harness tests never depend on -- or need to fake -- driver
# call results.
_NO_CONTEXT_SNAPSHOT = ProbeSnapshot(has_context=False)


@pytest.mark.agent_authored(model="claude-sonnet-5")
@pytest.mark.parametrize(
    ("failed", "exc_text", "should_capture"),
    [
        (True, f"boom {OOM_MARKER}", True),
        (True, "CUDA_ERROR_INVALID_CONTEXT", False),
        (False, f"boom {OOM_MARKER}", False),
        (True, None, False),
    ],
)
def test_oom_diagnostics_fire_only_on_a_failing_oom(tmp_path, failed, exc_text, should_capture):
    recorder = OomDiagnosticsRecorder()
    result = record_if_oom(
        _fake_item(tmp_path),
        _fake_call(exc_text),
        _fake_report(failed),
        recorder=recorder,
        snapshot=_NO_CONTEXT_SNAPSHOT,
    )

    assert (result is not None) == should_capture
    assert recorder.captured == should_capture


@pytest.mark.agent_authored(model="claude-sonnet-5")
def test_oom_diagnostics_write_an_artifact_naming_the_failing_test(tmp_path, monkeypatch):
    # Omitting `recorder` also pins the call signature conftest.py relies on.
    # Patching the singleton keeps this from latching diagnostics for the run.
    recorder = OomDiagnosticsRecorder()
    monkeypatch.setattr("helpers.oom_diagnostics._default_recorder", recorder)

    text = record_if_oom(
        _fake_item(tmp_path, nodeid="tests/test_x.py::test_first"),
        _fake_call(f"boom {OOM_MARKER}"),
        _fake_report(True),
        snapshot=_NO_CONTEXT_SNAPSHOT,
    )

    written = (tmp_path / DEFAULT_FILENAME).read_text(encoding="utf-8")
    assert "tests/test_x.py::test_first" in written
    assert OOM_MARKER in written
    assert "tests/test_x.py::test_first" in text


@pytest.mark.agent_authored(model="claude-sonnet-5")
def test_oom_diagnostics_summary_points_at_the_artifact(tmp_path):
    # The report is emitted beside the failing test, thousands of lines above
    # the summary; without this line the artifact is effectively invisible.
    recorder = OomDiagnosticsRecorder()
    lines = []
    reporter = types.SimpleNamespace(write_sep=lambda *_, **__: None, write_line=lines.append)

    assert report_terminal_summary(reporter, recorder=recorder) is None
    assert lines == []

    recorder.capture("tests/test_x.py::test_first", "call", OOM_MARKER, tmp_path, snapshot=_NO_CONTEXT_SNAPSHOT)
    emitted = report_terminal_summary(reporter, recorder=recorder)

    assert "tests/test_x.py::test_first" in emitted
    assert str(tmp_path) in emitted
    assert emitted in lines


@pytest.mark.agent_authored(model="claude-sonnet-5")
def test_oom_diagnostics_latch_to_the_first_oom(tmp_path):
    # A failing run produces ~190 OOMs; capturing each would bury the log.
    recorder = OomDiagnosticsRecorder()
    assert recorder.capture("first", "call", OOM_MARKER, tmp_path, snapshot=_NO_CONTEXT_SNAPSHOT) is not None
    assert recorder.captured
    assert recorder.capture("second", "call", OOM_MARKER, tmp_path, snapshot=_NO_CONTEXT_SNAPSHOT) is None

    assert "first" in (tmp_path / DEFAULT_FILENAME).read_text(encoding="utf-8")


@pytest.mark.agent_authored(model="claude-sonnet-5")
def test_oom_diagnostics_report_names_the_failing_test_and_verdict(tmp_path):
    # build_report is what actually assembles the artifact text; exercise it
    # directly (rather than only through capture()) so a broken verdict line
    # is caught here instead of by reading a real OOM log after the fact.
    recorder = OomDiagnosticsRecorder()
    text = recorder.build_report("tests/test_x.py::test_first", "call", OOM_MARKER, snapshot=_NO_CONTEXT_SNAPSHOT)

    assert "tests/test_x.py::test_first" in text
    assert "verdict: no current CUDA context" in text


@pytest.mark.agent_authored(model="claude-sonnet-5")
@pytest.mark.parametrize(
    ("snapshot", "expected_substring"),
    [
        (ProbeSnapshot(has_context=False), "no current CUDA context"),
        (
            ProbeSnapshot(has_context=True, mem_get_info_error="CUDA_ERROR_INVALID_CONTEXT"),
            "cuMemGetInfo itself failed",
        ),
        (
            ProbeSnapshot(has_context=True, mem_free=10 * (1 << 30), mem_total=10 * (1 << 30), small_alloc_ok=False),
            "likely physical device memory exhaustion",
        ),
        (
            # Plenty of device memory reported free, but even a tiny host VA
            # reservation fails outright.
            ProbeSnapshot(
                has_context=True,
                mem_free=140 * (1 << 30),
                mem_total=180 * (1 << 30),
                small_alloc_ok=True,
                small_va_ok=False,
            ),
            "host virtual-address space is exhausted",
        ),
        (
            ProbeSnapshot(
                has_context=True,
                mem_free=140 * (1 << 30),
                mem_total=180 * (1 << 30),
                small_alloc_ok=True,
                mempools_supported=False,
            ),
            "mempools are not supported",
        ),
        (
            # The #2381 signature itself: device mostly free, but the
            # pool-sized reservation -- the same size cuDeviceGetMemPool
            # needs -- fails outright.
            ProbeSnapshot(
                has_context=True,
                mem_free=140 * (1 << 30),
                mem_total=180 * (1 << 30),
                small_alloc_ok=True,
                small_va_ok=True,
                mempools_supported=True,
                pool_va_ok=False,
                capped_pool_create_ok=False,
            ),
            "likely host VA exhaustion: a pool-sized reservation failed",
        ),
        (
            # Same, but a capped pool still fits: only the ~2x default
            # window is the problem, not pools in general.
            ProbeSnapshot(
                has_context=True,
                mem_free=140 * (1 << 30),
                mem_total=180 * (1 << 30),
                small_alloc_ok=True,
                small_va_ok=True,
                mempools_supported=True,
                pool_va_ok=False,
                capped_pool_create_ok=True,
            ),
            "likely host VA exhaustion for the pool-sized window only",
        ),
        (
            ProbeSnapshot(
                has_context=True,
                mem_free=140 * (1 << 30),
                mem_total=180 * (1 << 30),
                small_alloc_ok=True,
                small_va_ok=True,
                mempools_supported=True,
                pool_va_ok=True,
                get_mem_pool_ok=False,
            ),
            "default mempool materialization failed",
        ),
        (
            ProbeSnapshot(
                has_context=True,
                mem_free=140 * (1 << 30),
                mem_total=180 * (1 << 30),
                small_alloc_ok=True,
                small_va_ok=True,
                mempools_supported=True,
                pool_va_ok=True,
                get_mem_pool_ok=True,
                get_default_mem_pool_ok=True,
                capped_pool_create_ok=True,
            ),
            "inconclusive: all probes succeeded",
        ),
    ],
)
def test_oom_diagnostics_classify_names_the_right_bucket(snapshot, expected_substring):
    assert expected_substring in classify(snapshot)


@pytest.mark.agent_authored(model="claude-sonnet-5")
@pytest.mark.parametrize(
    ("value", "alignment", "expected"),
    [
        (0, 2 * (1 << 20), 0),
        (1, 2 * (1 << 20), 2 * (1 << 20)),
        (2 * (1 << 20), 2 * (1 << 20), 2 * (1 << 20)),
        # One byte short of aligned; an unaligned reserve would return
        # CUDA_ERROR_INVALID_VALUE, which looks like exhaustion.
        (24 * (1 << 30) - 1, 2 * (1 << 20), 24 * (1 << 30)),
    ],
)
def test_oom_diagnostics_round_up_keeps_va_reserves_aligned(value, alignment, expected):
    assert oom_round_up(value, alignment) == expected


@pytest.mark.agent_authored(model="claude-sonnet-5")
def test_oom_diagnostics_probe_basics_is_live_and_cheap(init_cuda):
    # Only the side-effect-free prefix: no cuDeviceGetMemPool, no
    # cuMemPoolCreate, no pool-sized cuMemAddressReserve. A healthy device
    # must report a context and free/total memory.
    snapshot, dev = probe_basics()

    assert snapshot.has_context
    assert snapshot.mem_get_info_error is None
    assert snapshot.mem_total > 0
    assert dev is not None
    # probe_basics never sets any field past the physical-allocator check.
    assert snapshot.small_va_ok is None
    assert snapshot.pool_va_ok is None
    assert snapshot.get_mem_pool_ok is None
    assert snapshot.capped_pool_create_ok is None
