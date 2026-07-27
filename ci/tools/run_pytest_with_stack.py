#!/usr/bin/env python3

# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

"""Run pytest on a thread with a larger stack size.

Cython linetrace instrumentation under coverage on Windows can exceed the
default 1 MB thread stack.  This helper spawns a single worker thread with
a configurable stack (default 8 MB) so the rest of the CI workflow stays
readable.

With --isolate, pytest runs in a child process instead and the parent
reports the child's raw exit status.  On Windows the shell only ever
shows a translation of that status -- Git Bash turns an access violation
into "Segmentation fault" and exit 139 -- which loses the distinction
between, say, a use-after-free and a blown stack.  A surviving parent
names the NTSTATUS outright, so nothing has to be inferred.

Usage:
    python run_pytest_with_stack.py [--stack-mb N] [--cwd DIR] [--isolate]
                                    [pytest args ...]
"""

import argparse
import concurrent.futures
import os
import subprocess
import sys
import threading

import pytest

# Windows exception codes worth naming in a CI log.  Anything else is
# printed as a bare hex status.
_NTSTATUS = {
    0xC0000005: "EXCEPTION_ACCESS_VIOLATION",
    0xC000001D: "EXCEPTION_ILLEGAL_INSTRUCTION",
    0xC0000094: "EXCEPTION_INT_DIVIDE_BY_ZERO",
    0xC00000FD: "EXCEPTION_STACK_OVERFLOW",
    0xC0000374: "STATUS_HEAP_CORRUPTION",
    0xC000041D: "STATUS_FATAL_USER_CALLBACK_EXCEPTION",
    0xC0000409: "STATUS_STACK_BUFFER_OVERRUN",
}


def _describe_exit(code):
    """Render an abnormal child exit status, or None if it looks normal."""
    if 0 <= code <= 255:
        return None
    if code < 0:  # POSIX: killed by signal -code
        return f"killed by signal {-code}"
    status = code & 0xFFFFFFFF
    return f"0x{status:08X} ({_NTSTATUS.get(status, 'unrecognized status')})"


def _run_isolated(stack_mb, cwd, pytest_args):
    """Re-run this script as a child so the parent survives a native crash."""
    cmd = [sys.executable, os.path.abspath(__file__), f"--stack-mb={stack_mb}"]
    if cwd:
        cmd += ["--cwd", cwd]
    cmd += pytest_args

    print(f"[stack-wrapper] launching child: {cmd}", flush=True)
    code = subprocess.call(cmd)  # noqa: S603 - cmd is this interpreter re-running this script
    described = _describe_exit(code)
    if described is None:
        print(f"[stack-wrapper] child exited with {code}", flush=True)
    else:
        print(f"[stack-wrapper] child died abnormally: {described}", flush=True)
    return code


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--stack-mb",
        type=int,
        default=8,
        help="Thread stack size in megabytes (default: 8)",
    )
    parser.add_argument(
        "--cwd",
        default=None,
        help="Working directory for the test run",
    )
    parser.add_argument(
        "--isolate",
        action="store_true",
        help="Run pytest in a child process and report its raw exit status",
    )
    args, pytest_args = parser.parse_known_args()

    if args.isolate:
        sys.exit(_run_isolated(args.stack_mb, args.cwd, pytest_args))

    plugins = []
    if os.environ.get("PYTEST_CRASHLOG"):
        # Load the progress logger from this script's directory, which is not
        # otherwise importable from the install root we chdir into below.
        sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
        import pytest_crashlog

        # Open the log here rather than from the plugin's pytest_configure:
        # pytest imports the initial conftests before configure runs, and for
        # cuda_core that import pulls in cuda.core itself.  A crash there
        # would otherwise leave no file at all, indistinguishable from a crash
        # before pytest ever started.
        pytest_crashlog.open_log()
        pytest_crashlog.note("entering pytest.main")
        # Hand pytest the imported module rather than "-p pytest_crashlog":
        # naming it would make pytest import it a second time and warn that
        # it can no longer rewrite the module's assertions.
        plugins.append(pytest_crashlog)

    if args.cwd:
        os.chdir(args.cwd)

    threading.stack_size(args.stack_mb * 1024 * 1024)

    with concurrent.futures.ThreadPoolExecutor(max_workers=1) as pool:
        code = pool.submit(pytest.main, pytest_args, plugins).result()

    sys.exit(code)


if __name__ == "__main__":
    main()
