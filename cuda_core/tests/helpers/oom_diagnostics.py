# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Machine-state capture for the first CUDA OOM of a test session (issue #2381).

A failing ``cuda_core`` run reports ~190 ``CUDA_ERROR_OUT_OF_MEMORY`` failures
that all descend from a single earlier event, so only the first is worth
capturing; running ``nvidia-smi`` on every one would add minutes and bury the
log. Hence the latch in :class:`OomDiagnosticsRecorder`.
"""

import os
import pathlib
import subprocess
import sys
import threading

from cuda.bindings import driver

OOM_MARKER = "CUDA_ERROR_OUT_OF_MEMORY"
DEFAULT_FILENAME = "cuda_core_oom_diagnostics.txt"

_BANNER = "=" * 78


def format_probe(label, fn, *args):
    """Call a driver API and render its raw result."""
    try:
        return f"{label} -> {fn(*args)!r}"
    except Exception as exc:  # diagnostics must never mask the original test failure
        return f"{label} -> <raised {exc!r}>"


def probe_driver_state():
    """Query the driver directly, bypassing cuda.core's error reporting.

    cuda.core surfaces handle-creation failures through a thread-local "last
    error" slot (see cuda/core/_cpp/DESIGN.md). Reading the driver directly
    shows whether the device is genuinely out of memory and whether the default
    mempool is actually unavailable, which is what distinguishes real
    exhaustion from a mempool setup failure.
    """
    lines = ["--- direct driver probe (bypasses cuda.core error reporting) ---"]
    lines.append(format_probe("cuCtxGetCurrent()", driver.cuCtxGetCurrent))
    lines.append(format_probe("cuMemGetInfo()", driver.cuMemGetInfo))

    try:
        err, count = driver.cuDeviceGetCount()
    except Exception as exc:  # see format_probe
        lines.append(f"cuDeviceGetCount() -> <raised {exc!r}>")
        return "\n".join(lines)

    lines.append(f"cuDeviceGetCount() -> ({err!r}, {count})")
    if err != driver.CUresult.CUDA_SUCCESS:
        return "\n".join(lines)

    for ordinal in range(count):
        try:
            err, dev = driver.cuDeviceGet(ordinal)
        except Exception as exc:  # see format_probe
            lines.append(f"cuDeviceGet({ordinal}) -> <raised {exc!r}>")
            continue
        if err != driver.CUresult.CUDA_SUCCESS:
            lines.append(f"cuDeviceGet({ordinal}) -> {err!r}")
            continue
        lines.append(format_probe(f"cuDeviceGetMemPool(dev {ordinal})", driver.cuDeviceGetMemPool, dev))
        lines.append(format_probe(f"cuDeviceGetDefaultMemPool(dev {ordinal})", driver.cuDeviceGetDefaultMemPool, dev))

    return "\n".join(lines)


def run_nvidia_smi(*args):
    cmd = ["nvidia-smi", *args]
    printable = " ".join(cmd)
    try:
        proc = subprocess.run(cmd, capture_output=True, text=True, timeout=60, check=False)  # noqa: S603
    except (OSError, subprocess.SubprocessError) as exc:
        return f"$ {printable}\n<could not run: {exc!r}>"
    output = proc.stdout.strip() or proc.stderr.strip() or "<no output>"
    return f"$ {printable}\n(exit {proc.returncode})\n{output}"


class OomDiagnosticsRecorder:
    """Captures machine state the first time a CUDA OOM is seen, and only then."""

    def __init__(self, filename=DEFAULT_FILENAME):
        self._filename = filename
        self._lock = threading.Lock()
        self._captured = False
        self._nodeid = None
        self._artifact_path = None
        self._artifact_written = False

    @property
    def captured(self):
        return self._captured

    @property
    def nodeid(self):
        """Node id of the test that triggered capture, or None."""
        return self._nodeid

    @property
    def artifact_path(self):
        """Where the report was written, or None if nothing was captured."""
        return self._artifact_path

    @property
    def artifact_written(self):
        return self._artifact_written

    @staticmethod
    def matches(exc_text):
        return OOM_MARKER in exc_text

    def build_report(self, nodeid, phase, exc_text):
        return "\n".join(
            [
                _BANNER,
                "cuda_core diagnostics: first CUDA_ERROR_OUT_OF_MEMORY of this session",
                _BANNER,
                f"test:      {nodeid}",
                f"phase:     {phase}",
                f"pid:       {os.getpid()}",
                f"platform:  {sys.platform}",
                f"exception: {exc_text}",
                "",
                probe_driver_state(),
                "",
                run_nvidia_smi("-q"),
                "",
                run_nvidia_smi("--query-compute-apps=timestamp,pid,process_name,used_memory", "--format=csv"),
                _BANNER,
            ]
        )

    def capture(self, nodeid, phase, exc_text, directory):
        """Build and persist the report. Returns None if already captured."""
        with self._lock:
            if self._captured:
                return None
            self._captured = True

        report = self.build_report(nodeid, phase, exc_text)
        destination = pathlib.Path(directory) / self._filename
        self._nodeid = nodeid
        self._artifact_path = destination
        try:
            destination.write_text(report, encoding="utf-8")
            self._artifact_written = True
            return f"{report}\n(diagnostics also written to {destination})"
        except OSError as exc:
            return f"{report}\n(could not write {destination}: {exc!r})"


_default_recorder = OomDiagnosticsRecorder()


def record_if_oom(item, call, report, recorder=None):
    """Capture diagnostics when ``report`` is the session's first CUDA OOM.

    ``recorder`` defaults to a module-level singleton so the conftest hook does
    not have to hold session state; tests pass their own to stay isolated.

    Returns the emitted text, or None when nothing was captured.
    """
    if recorder is None:
        recorder = _default_recorder

    if recorder.captured or not report.failed or call.excinfo is None:
        return None

    exc_text = str(call.excinfo.value)
    if not recorder.matches(exc_text):
        return None

    text = recorder.capture(item.nodeid, call.when, exc_text, item.config.rootpath)
    if text is None:
        return None

    # terminalreporter writes outside pytest's stdout capture, so this survives
    # into a redirected log; a bare print() would not.
    terminal_reporter = item.config.pluginmanager.get_plugin("terminalreporter")
    if terminal_reporter is not None:
        terminal_reporter.write_line("")
        terminal_reporter.write_line(text)
    return text


def report_terminal_summary(terminalreporter, recorder=None):
    """Point at the diagnostics artifact from pytest's end-of-run summary.

    The report itself is emitted beside the failing test, which in a real
    failing run is thousands of lines above the summary people actually read.

    Returns the emitted line, or None when nothing was captured.
    """
    if recorder is None:
        recorder = _default_recorder

    if not recorder.captured:
        return None

    verb = "written to" if recorder.artifact_written else "could NOT be written to"
    line = f"first CUDA OOM at {recorder.nodeid}; diagnostics {verb} {recorder.artifact_path}"
    terminalreporter.write_sep("=", "cuda_core OOM diagnostics", red=True)
    terminalreporter.write_line(line)
    return line
