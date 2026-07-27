# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

"""Record session progress to a file so a hard crash names where it hit.

When the interpreter dies from a native fault there is no report, no
summary, and often no console output at all: pytest's terminal writer goes
through the capture manager and through a block-buffered pipe, both of
which lose whatever had not been flushed.  Writing progress to a
line-buffered file instead pushes each line to the OS before the next step
begins, so the record survives the process.

The log covers the whole session, not just the test loop, because a crash
during collection looks exactly like a crash before startup otherwise:

    (file absent)          crashed before open_log(), i.e. during interpreter
                           startup or `import pytest`
    open ... entering      crashed while pytest loaded its initial conftests
                           -- for cuda_core that means importing cuda.core
    COLLECT <module>       crashed importing or collecting that module
    START <nodeid>         crashed inside that test
    closed cleanly         crashed after the session finished, e.g. during
                           interpreter shutdown or coverage's atexit write

Enabled by setting PYTEST_CRASHLOG to the destination path; a no-op
otherwise.  ci/tools/run_pytest_with_stack.py calls open_log() before
handing control to pytest, then loads this module as a plugin.
"""

import os
import sys

_log = None
_disabled = False


def open_log(path=None):
    """Open the crash log.  Safe to call before pytest starts, and twice."""
    global _log, _disabled
    if _log is not None or _disabled:
        return
    path = path or os.environ.get("PYTEST_CRASHLOG")
    if not path:
        return
    try:
        # buffering=1 is line buffering: every newline reaches the OS, so a
        # crash cannot take the preceding lines down with it.  The handle
        # deliberately outlives this function, so no context manager.
        _log = open(  # noqa: SIM115
            os.path.abspath(path), "w", buffering=1, encoding="utf-8", errors="backslashreplace"
        )
    except OSError as exc:
        # A diagnostic must never break the run it is diagnosing.
        _disabled = True
        print(f"[crashlog] disabled, cannot write {path}: {exc}", file=sys.stderr, flush=True)
        return
    _log.write(f"open pid={os.getpid()}\n")


def note(message):
    """Append one line to the crash log, if it is open."""
    if _log is not None:
        _log.write(f"{message}\n")


def pytest_configure():
    # Covers direct `-p pytest_crashlog` use, where nothing called open_log().
    # By the time this runs the initial conftests have already been imported,
    # which is why the wrapper opens the log earlier.
    open_log()
    note("configured")


def pytest_unconfigure():
    global _log
    if _log is not None:
        _log.write("closed cleanly\n")
        _log.close()
        _log = None


def pytest_collectstart(collector):
    note(f"COLLECT {collector.nodeid or '<session>'}")


def pytest_collection_finish(session):
    note(f"collected {len(session.items)} items")


def pytest_runtest_logstart(nodeid):
    note(f"START {nodeid}")


def pytest_runtest_logfinish(nodeid):
    note(f"END   {nodeid}")
