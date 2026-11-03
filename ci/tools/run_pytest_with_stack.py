#!/usr/bin/env python3

# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

"""Run pytest on Python's main thread with a larger stack.

The Windows coverage workflow first raises the base interpreter's PE stack
reserve to 8 MB, covering the main thread and test subprocesses. Set the
stack size for threads started during pytest to match. Running pytest on the
main thread lets CPython process pending calls used by CUDA graph cleanup.

Usage:
    python run_pytest_with_stack.py [--stack-mb N] [--cwd DIR] [pytest args ...]
"""

import argparse
import os
import sys
import threading


def _ensure_main_stack(stack_mb):
    stack_bytes = stack_mb * 1024 * 1024
    if sys.platform == "win32":
        import ctypes

        low, high = ctypes.c_size_t(), ctypes.c_size_t()
        get_limits = ctypes.WinDLL("kernel32", use_last_error=True).GetCurrentThreadStackLimits
        get_limits.argtypes = [
            ctypes.POINTER(ctypes.c_size_t),
            ctypes.POINTER(ctypes.c_size_t),
        ]
        get_limits.restype = None
        get_limits(ctypes.byref(low), ctypes.byref(high))
        actual = high.value - low.value
        if actual < stack_bytes:
            raise RuntimeError(f"Main thread stack is {actual} bytes, requested {stack_bytes}")
        print(f"[stack-wrapper] main thread stack: {actual} bytes", flush=True)
    else:
        import resource

        soft, hard = resource.getrlimit(resource.RLIMIT_STACK)
        if soft != resource.RLIM_INFINITY and soft < stack_bytes:
            resource.setrlimit(resource.RLIMIT_STACK, (stack_bytes, hard))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--stack-mb",
        type=int,
        default=8,
        help="Main thread stack size in megabytes (default: 8)",
    )
    parser.add_argument(
        "--cwd",
        default=None,
        help="Working directory for the test run",
    )
    args, pytest_args = parser.parse_known_args()
    if args.stack_mb <= 0:
        parser.error("--stack-mb must be positive")

    _ensure_main_stack(args.stack_mb)
    threading.stack_size(args.stack_mb * 1024 * 1024)
    if args.cwd:
        os.chdir(args.cwd)
    import pytest

    sys.exit(pytest.main(pytest_args))


if __name__ == "__main__":
    main()
