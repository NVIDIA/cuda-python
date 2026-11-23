# SPDX-FileCopyrightText: Copyright (c) 2024-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

import ctypes
import os
import pathlib
import sys
import time
import types
from contextlib import closing
from unittest.mock import Mock

import pytest
from helpers.buffers import PatternGen, compare_equal_buffers, make_scratch_buffer, thread_unsafe_on_windows
from helpers.latch import LatchKernel
from helpers.logging import TimestampedLogger
from helpers.memory import create_managed_memory_resource_or_skip
from helpers.oom_diagnostics import (
    DEFAULT_FILENAME,
    OOM_MARKER,
    OomDiagnosticsRecorder,
    ProbeSnapshot,
    classify,
    format_probe_log,
    probe_basics,
    record_if_oom,
    report_terminal_summary,
    run_probe,
)
from helpers.oom_diagnostics import _round_up as oom_round_up  # white-box test of the alignment guard
from helpers.oom_diagnostics import (
    artifact_directory as oom_artifact_directory,
)

from cuda.bindings import driver
from cuda.core import Device
from cuda.core._utils.cuda_utils import handle_return
from cuda_python_test_helpers import under_compute_sanitizer

ENABLE_LOGGING = False  # Set True for test debugging and development
NBYTES = 64


@pytest.mark.skipif(Device().compute_capability.major < 7, reason="__nanosleep is only available starting Volta (sm70)")
@pytest.mark.thread_unsafe(reason="requires a barrier wait to avoid overlapping pinned latch allocations")
def test_latchkernel(init_cuda):
    """Test LatchKernel."""
    log = TimestampedLogger(enabled=ENABLE_LOGGING)
    log("begin")
    device = init_cuda
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
    assert not latch.timed_out
    log("checking target == 1")
    assert compare_equal_buffers(target, ones)
    log("done")


@pytest.mark.skipif(Device().compute_capability.major < 7, reason="__nanosleep is only available starting Volta (sm70)")
@pytest.mark.thread_unsafe(reason="requires a barrier wait to avoid overlapping pinned latch allocations")
@pytest.mark.agent_authored(model="claude-opus-5-5")
def test_latchkernel_reports_timeout(init_cuda):
    """A latch that is never released ends by timeout and reports it; relaunching clears the report."""
    device = init_cuda
    with closing(device.create_stream()) as stream, closing(LatchKernel(device, timeout_sec=0.1)) as latch:
        latch.launch(stream)
        stream.sync()
        assert latch.timed_out

        latch.launch(stream)
        latch.release()
        stream.sync()
        assert not latch.timed_out


@pytest.mark.skipif(
    under_compute_sanitizer(),
    reason="Too slow under compute-sanitizer (UVM-heavy test).",
)
@thread_unsafe_on_windows
def test_patterngen_seeds():
    """Test PatternGen with seed argument."""
    device = Device()
    device.set_current()
    buffer = make_scratch_buffer(device, 0, NBYTES)
    stream = device.default_stream

    # All seeds are pairwise different.
    # We test a sampling of values because exhaustive testing is too slow,
    # especially on Windows. See https://github.com/NVIDIA/cuda-python/issues/1455
    pgen = PatternGen(device, NBYTES, stream=stream)
    for i in (ii for ii in range(256) if ii < 5 or ii % 17 == 0):
        pgen.fill_buffer(buffer, seed=i)
        pgen.verify_buffer(buffer, seed=i)
        for j in (jj for jj in range(i + 1, 256) if jj < 5 or jj % 19 == 0):
            with pytest.raises(AssertionError):
                pgen.verify_buffer(buffer, seed=j)


@thread_unsafe_on_windows
def test_patterngen_values():
    """Test PatternGen with value argument, also compare_equal_buffers."""
    device = Device()
    device.set_current()
    ones = make_scratch_buffer(device, 1, NBYTES)
    twos = make_scratch_buffer(device, 2, NBYTES)
    assert compare_equal_buffers(ones, ones)
    assert not compare_equal_buffers(ones, twos)
    pgen = PatternGen(device, NBYTES, stream=device.default_stream)
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


def _fake_item(rootpath, nodeid="tests/test_x.py::test_a", xmlpath=None):
    # get_plugin returns None so record_if_oom skips the terminal write.
    config = types.SimpleNamespace(
        rootpath=rootpath,
        option=types.SimpleNamespace(xmlpath=xmlpath),
        invocation_params=types.SimpleNamespace(dir=pathlib.Path(rootpath)),
        pluginmanager=types.SimpleNamespace(get_plugin=lambda _: None),
    )
    return types.SimpleNamespace(nodeid=nodeid, config=config)


# A snapshot that short-circuits classify()/format_probe_log() at the very
# first check, so harness tests never depend on -- or need to fake -- driver
# call results.
_NO_CONTEXT_SNAPSHOT = ProbeSnapshot(has_context=False)
# The smallest snapshot that counts as probed; its probe log stops at "VMM unsupported".
_PROBED_SNAPSHOT = ProbeSnapshot(has_context=True, mem_free=1 << 30, mem_total=1 << 30, small_alloc_ok=True)


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


@pytest.mark.agent_authored(model="claude-opus-5-5")
def test_oom_diagnostics_report_without_context_keeps_the_latch_open(tmp_path):
    recorder = OomDiagnosticsRecorder()
    assert recorder.capture("first", "teardown", OOM_MARKER, tmp_path, snapshot=_NO_CONTEXT_SNAPSHOT) is not None
    assert not recorder.probed
    # A second unprobed report would only repeat the first.
    assert recorder.capture("second", "call", OOM_MARKER, tmp_path, snapshot=_NO_CONTEXT_SNAPSHOT) is None

    text = recorder.capture("third", "call", OOM_MARKER, tmp_path, snapshot=_PROBED_SNAPSHOT)
    assert "the probe at first had no context" in text
    assert recorder.probed
    assert recorder.capture("fourth", "call", OOM_MARKER, tmp_path, snapshot=_PROBED_SNAPSHOT) is None

    written = (tmp_path / DEFAULT_FILENAME).read_text(encoding="utf-8")
    assert "test:      first" in written
    assert "test:      third" in written
    assert "second" not in written
    assert "fourth" not in written

    lines = []
    reporter = types.SimpleNamespace(write_sep=lambda *_, **__: None, write_line=lines.append)
    assert "first CUDA OOM at first; probed at third" in report_terminal_summary(reporter, recorder=recorder)


@pytest.mark.agent_authored(model="claude-opus-5-5")
def test_oom_diagnostics_artifact_lands_beside_the_junit_file(tmp_path, monkeypatch):
    # CI uploads only the JUnit directory, which may not exist yet if the run
    # crashes before pytest writes the JUnit file.
    recorder = OomDiagnosticsRecorder()
    monkeypatch.setattr("helpers.oom_diagnostics._default_recorder", recorder)
    junit_dir = tmp_path / "test-results"

    record_if_oom(
        _fake_item(tmp_path / "rootdir", xmlpath=str(junit_dir / "junit-core.xml")),
        _fake_call(f"boom {OOM_MARKER}"),
        _fake_report(True),
        snapshot=_NO_CONTEXT_SNAPSHOT,
    )

    assert recorder.artifact_path == junit_dir / DEFAULT_FILENAME
    assert OOM_MARKER in recorder.artifact_path.read_text(encoding="utf-8")


@pytest.mark.agent_authored(model="claude-sonnet-5-5")
def test_oom_diagnostics_relative_junit_path_is_anchored_at_the_invocation_dir(tmp_path, monkeypatch):
    # A test may have changed directory by the time an OOM is captured.
    invocation = tmp_path / "invocation"
    elsewhere = tmp_path / "elsewhere"
    invocation.mkdir()
    elsewhere.mkdir()
    monkeypatch.chdir(elsewhere)
    item = _fake_item(invocation, xmlpath=os.path.join("test-results", "junit-core.xml"))

    assert oom_artifact_directory(item.config) == invocation / "test-results"


@pytest.mark.agent_authored(model="claude-sonnet-5-5")
def test_oom_diagnostics_does_not_overwrite_an_earlier_invocations_artifact(tmp_path):
    # tests/ and tests/cython run as separate pytest processes into one directory.
    first = OomDiagnosticsRecorder()
    first.capture("first", "call", OOM_MARKER, tmp_path, snapshot=_PROBED_SNAPSHOT)
    second = OomDiagnosticsRecorder()
    second.capture("second", "call", OOM_MARKER, tmp_path, snapshot=_PROBED_SNAPSHOT)

    assert first.artifact_path == tmp_path / DEFAULT_FILENAME
    assert second.artifact_path != first.artifact_path
    assert "test:      first" in first.artifact_path.read_text(encoding="utf-8")
    assert "test:      second" in second.artifact_path.read_text(encoding="utf-8")


@pytest.mark.agent_authored(model="claude-sonnet-5-5")
def test_oom_diagnostics_failed_borrow_names_the_failing_call(monkeypatch):
    # No GPU needed: the driver calls are faked so that no context is current
    # and the primary-context retain fails.
    def fake_call(fn, *args):
        if fn is driver.cuCtxGetCurrent:
            return True, None, (0,)
        if fn is driver.cuDeviceGet:
            return True, None, (0,)
        if fn is driver.cuDevicePrimaryCtxRetain:
            return False, "CUDA_ERROR_OUT_OF_MEMORY", ()
        raise AssertionError(f"unexpected driver call {fn}")

    monkeypatch.setattr("helpers.oom_diagnostics._call", fake_call)
    snapshot = run_probe()

    assert not snapshot.has_context
    assert snapshot.context_error is None
    assert "cuDevicePrimaryCtxRetain(dev 0): CUDA_ERROR_OUT_OF_MEMORY" in snapshot.borrow_error
    log = format_probe_log(snapshot)
    assert "cuCtxGetCurrent() (no context)" in log
    assert "borrowing device 0's primary context -> <failed: cuDevicePrimaryCtxRetain" in log
    assert "borrowing device 0's primary context failed" in classify(snapshot)


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


def _current_context():
    return int(handle_return(driver.cuCtxGetCurrent()))


@pytest.mark.agent_authored(model="claude-opus-5-5")
def test_oom_diagnostics_probe_borrows_and_restores_the_primary_context(init_cuda, monkeypatch):
    # The real probes are replaced: only the context handling is under test,
    # and the full probe reserves a ~2x-device-memory VA window.
    seen = []

    def fake_probe():
        seen.append(_current_context())
        return ProbeSnapshot(has_context=True)

    monkeypatch.setattr("helpers.oom_diagnostics._probe_current_context", fake_probe)
    dev = handle_return(driver.cuDeviceGet(0))
    primary = int(handle_return(driver.cuDevicePrimaryCtxRetain(dev)))
    handle_return(driver.cuDevicePrimaryCtxRelease(dev))
    current = _current_context()

    snapshot = run_probe()
    assert not snapshot.borrowed_primary_context
    assert seen == [current]

    popped = handle_return(driver.cuCtxPopCurrent())
    try:
        assert _current_context() == 0
        snapshot = run_probe()
        assert _current_context() == 0
    finally:
        handle_return(driver.cuCtxPushCurrent(popped))

    assert snapshot.borrowed_primary_context
    assert snapshot.context_restore_error is None
    assert seen[1] == primary


# ---------------------------------------------------------------------------
# GL helper tests
# ---------------------------------------------------------------------------

import pytest
from cuda_python_test_helpers.graphics import (
    gl_context_not_on_nvidia_gpu_reason,
    is_gl_context_unavailable,
    open_gl_window,
)


@pytest.mark.thread_unsafe(reason="patches the process-wide pyglet module")
@pytest.mark.agent_authored(model="gpt-5.6-sol")
def test_open_gl_window_cleans_up_failed_construction(monkeypatch):
    windows = set()
    partial_window = Mock()
    previous_context = Mock()

    def fail_window(**_kwargs):
        windows.add(partial_window)
        raise RuntimeError

    pyglet = types.ModuleType("pyglet")
    pyglet.options = {}
    pyglet.app = types.SimpleNamespace(windows=windows)
    pyglet.gl = types.SimpleNamespace(Config=Mock(), current_context=previous_context)
    pyglet.window = types.SimpleNamespace(Window=fail_window)
    monkeypatch.setitem(sys.modules, "pyglet", pyglet)

    with pytest.raises(RuntimeError):
        open_gl_window()

    assert partial_window.close.called
    assert previous_context.set_current.called


class _PygletError(Exception):
    pass


# Simulate a pyglet-namespaced exception by name.
def _make_pyglet_exc(name, module="pyglet.window"):
    cls = type(name, (_PygletError,), {})
    cls.__module__ = module
    return cls


@pytest.mark.human_reviewed
@pytest.mark.parametrize(
    "exc",
    [
        _make_pyglet_exc("NoSuchDisplayException")("x"),
        _make_pyglet_exc("NoSuchConfigException")("x"),
        _make_pyglet_exc("NoSuchScreenModeException")("x"),
        _make_pyglet_exc("WindowException")("x"),
        _make_pyglet_exc("ContextException")("x"),
        _make_pyglet_exc("MissingFunctionException", module="pyglet.gl.lib")("x"),
        FileNotFoundError("Could not find module 'opengl32' (or one of its dependencies)."),
        AttributeError("opengl32"),
        ImportError('Library "GL" not found.'),
        ImportError('Library "EGL" not found.'),
    ],
)
def test_is_gl_context_unavailable_accepts_genuine(exc):
    assert is_gl_context_unavailable(exc) is True


@pytest.mark.human_reviewed
@pytest.mark.parametrize(
    "exc",
    [
        # pyglet exception names that are not context-creation failures
        _make_pyglet_exc("GLException")("GL_INVALID_ENUM"),
        _make_pyglet_exc("ImageException")("x"),
        # Built-in exceptions that do not mention opengl32 / GL library
        TypeError("bug"),
        AttributeError("'NoneType' object has no attribute 'Config'"),
        FileNotFoundError("No such file: /tmp/missing"),
        ImportError("No module named 'foo'"),
        OSError("disk full"),
        RuntimeError("bug"),
    ],
)
def test_is_gl_context_unavailable_rejects_unrelated(exc):
    assert is_gl_context_unavailable(exc) is False


def _patch_gl_vendor(monkeypatch, vendor):
    # Only GL_VENDOR is answered; any other query (e.g. GL_RENDERER) returns NULL.
    gl = types.SimpleNamespace(
        GL_VENDOR=1,
        GL_RENDERER=2,
        glGetString=lambda name: ctypes.c_char_p(vendor if name == 1 else None),
    )
    monkeypatch.setitem(sys.modules, "pyglet.gl", types.SimpleNamespace(gl=gl))


@pytest.mark.thread_unsafe(reason="patches the process-wide pyglet.gl module")
def test_gl_context_not_on_nvidia_gpu_reason_accepts_nvidia(monkeypatch):
    _patch_gl_vendor(monkeypatch, b"NVIDIA Corporation")
    assert gl_context_not_on_nvidia_gpu_reason() is None


@pytest.mark.thread_unsafe(reason="patches the process-wide pyglet.gl module")
@pytest.mark.parametrize("vendor", [b"AMD", b"Mesa", None])
def test_gl_context_not_on_nvidia_gpu_reason_rejects_other_vendors(monkeypatch, vendor):
    _patch_gl_vendor(monkeypatch, vendor)
    reason = gl_context_not_on_nvidia_gpu_reason()
    assert reason is not None
    assert "not on an NVIDIA GPU" in reason


def _patch_managed_mr_factory(monkeypatch, concurrent_managed_access):
    # Stand in for the helper's Device and ManagedMemoryResource so the skip
    # decision does not depend on the local device.
    device = types.SimpleNamespace(
        properties=types.SimpleNamespace(concurrent_managed_access=concurrent_managed_access)
    )
    monkeypatch.setattr("helpers.memory.Device", lambda: device)
    factory = Mock(name="ManagedMemoryResource")
    monkeypatch.setattr("helpers.memory.ManagedMemoryResource", factory)
    return factory


@pytest.mark.thread_unsafe(reason="patches module globals of helpers.memory")
@pytest.mark.agent_authored(model="claude-opus-5.5")
def test_managed_mr_helper_skips_default_pool_without_concurrent_access(monkeypatch):
    factory = _patch_managed_mr_factory(monkeypatch, concurrent_managed_access=False)
    with pytest.raises(pytest.skip.Exception, match="concurrent managed memory access"):
        create_managed_memory_resource_or_skip()
    factory.assert_not_called()


@pytest.mark.thread_unsafe(reason="patches module globals of helpers.memory")
@pytest.mark.parametrize(
    ("concurrent_managed_access", "args", "kwargs"),
    [
        pytest.param(False, ("options",), {}, id="positional-options"),
        pytest.param(False, (), {"options": "options"}, id="keyword-options"),
        pytest.param(True, (), {}, id="default-pool-with-concurrent-access"),
    ],
)
@pytest.mark.agent_authored(model="claude-opus-5.5")
def test_managed_mr_helper_constructs_when_pool_is_usable(monkeypatch, concurrent_managed_access, args, kwargs):
    factory = _patch_managed_mr_factory(monkeypatch, concurrent_managed_access)
    assert create_managed_memory_resource_or_skip(*args, **kwargs) is factory.return_value
    factory.assert_called_once_with(*args, **kwargs)
