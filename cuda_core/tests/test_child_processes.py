# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Behavioral checks for the shared multiprocessing test helpers."""

import contextlib
import gc
import multiprocessing as mp
import os
import pathlib
import threading

import pytest
from helpers import child_processes

pytestmark = pytest.mark.thread_unsafe(
    reason="changes multiprocessing globals and counts process-wide file descriptors"
)


def _write_arguments(path, *args):
    path.write_text(repr(args))


def _exit_failure():
    raise SystemExit(7)


def _wait_forever(ready):
    ready.set()
    threading.Event().wait()


@pytest.fixture(autouse=True)
def _spawn_context():
    previous = mp.get_start_method(allow_none=True)
    mp.set_start_method("spawn", force=True)
    child_processes.warmup_multiprocessing.cache_clear()
    try:
        yield
    finally:
        child_processes.warmup_multiprocessing.cache_clear()
        mp.set_start_method(previous, force=True)


@pytest.fixture
def processes(monkeypatch):
    original_process = mp.Process
    tracked = []

    def make_process(*args, **kwargs):
        process = original_process(*args, **kwargs)
        tracked.append(process)
        return process

    monkeypatch.setattr(mp, "Process", make_process)
    try:
        yield tracked
    finally:
        child_processes.kill_subprocesses(*tracked)
        for process in tracked:
            with contextlib.suppress(ValueError):
                process.close()


def _assert_closed(processes):
    assert processes
    for process in processes:
        with pytest.raises(ValueError):
            process.is_alive()


def _timeout_after_start(monkeypatch, ready):
    startup_timeout = child_processes.child_timeout_sec()

    def short_timeout():
        assert ready.wait(timeout=startup_timeout), "child did not reach its target"
        return 0.05

    monkeypatch.setattr(child_processes, "child_timeout_sec", short_timeout)


@pytest.mark.agent_authored(model="gpt-6")
def test_run_child_passes_arguments_and_closes_process(tmp_path, processes):
    output = tmp_path / "arguments.txt"
    child_processes.run_child(_write_arguments, output, 42, "payload", None)
    assert output.read_text() == repr((42, "payload", None))
    _assert_closed(processes)


@pytest.mark.agent_authored(model="gpt-6")
def test_run_child_reports_failed_exit_and_closes_process(processes):
    with pytest.raises(AssertionError):
        child_processes.run_child(_exit_failure)
    _assert_closed(processes)


@pytest.mark.agent_authored(model="gpt-6")
def test_run_child_kills_hung_child(monkeypatch, processes):
    ready = mp.Event()
    _timeout_after_start(monkeypatch, ready)
    with pytest.raises(AssertionError):
        child_processes.run_child(_wait_forever, ready)
    assert ready.is_set()
    _assert_closed(processes)


@pytest.mark.agent_authored(model="gpt-6")
def test_run_child_kills_child_when_join_is_interrupted(monkeypatch, processes):
    ready = mp.Event()
    _timeout_after_start(monkeypatch, ready)
    original_join = mp.process.BaseProcess.join

    def interrupted_join(process, timeout=None):
        if timeout is not None:
            raise InterruptedError("join interrupted")
        return original_join(process)

    monkeypatch.setattr(mp.process.BaseProcess, "join", interrupted_join)
    with pytest.raises(InterruptedError):
        child_processes.run_child(_wait_forever, ready)
    _assert_closed(processes)


@pytest.mark.agent_authored(model="gpt-6")
def test_warmup_retries_failure_and_caches_success(monkeypatch, processes):
    noop = child_processes.noop
    monkeypatch.setattr(child_processes, "noop", _exit_failure)
    with pytest.raises(AssertionError):
        child_processes.warmup_multiprocessing()

    monkeypatch.setattr(child_processes, "noop", noop)
    child_processes.warmup_multiprocessing()

    monkeypatch.setattr(child_processes, "noop", _exit_failure)
    child_processes.warmup_multiprocessing()
    _assert_closed(processes)


@pytest.mark.skipif(not pathlib.Path("/proc/self/fd").is_dir(), reason="requires Linux file descriptor accounting")
@pytest.mark.agent_authored(model="gpt-6")
def test_repeated_warmup_and_run_child_leave_fd_count_stable(processes):
    child_processes.warmup_multiprocessing()
    gc.collect()
    initial_fds = len(os.listdir("/proc/self/fd"))
    for _ in range(3):
        child_processes.warmup_multiprocessing()
        child_processes.run_child(child_processes.noop, 42, "discarded", None)
    assert len(os.listdir("/proc/self/fd")) == initial_fds
    _assert_closed(processes)
