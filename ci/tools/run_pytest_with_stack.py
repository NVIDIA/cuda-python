#!/usr/bin/env python3

# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

"""Run pytest on Python's main thread with a larger stack.

On Windows, start a temporary copy of the base Python executable with an
8 MB PE stack reservation. This preserves the virtual environment without
modifying the installed executable. Running pytest on the main thread lets
CPython process pending calls used by CUDA graph cleanup.

Usage:
    python run_pytest_with_stack.py [--stack-mb N] [--cwd DIR] [pytest args ...]
"""

import argparse
import os
import struct
import subprocess
import sys
import tempfile
from pathlib import Path


def _python_with_stack(executable, stack_bytes):
    """Return a PE executable image with at least the requested stack reserve."""
    image = bytearray(executable.read_bytes())
    if len(image) < 64 or image[:2] != b"MZ":
        raise ValueError(f"Not a PE executable: {executable}")
    pe_offset = struct.unpack_from("<I", image, 0x3C)[0]
    if image[pe_offset : pe_offset + 4] != b"PE\0\0":
        raise ValueError(f"Invalid PE signature: {executable}")
    optional_offset = pe_offset + 24
    if len(image) < optional_offset + 88:
        raise ValueError(f"Truncated PE optional header: {executable}")
    magic = struct.unpack_from("<H", image, optional_offset)[0]
    if magic not in (0x10B, 0x20B):
        raise ValueError(f"Unsupported PE optional header: {magic:#x}")
    size_format = "<Q" if magic == 0x20B else "<I"
    reserve_offset = optional_offset + 72
    reserve = struct.unpack_from(size_format, image, reserve_offset)[0]
    struct.pack_into(size_format, image, reserve_offset, max(reserve, stack_bytes))
    return image


def _run_windows(stack_mb, cwd, pytest_args):
    # A venv's python.exe may be a redirector, so patch the base interpreter.
    base_executable = Path(sys._base_executable)
    image = _python_with_stack(base_executable, stack_mb * 1024 * 1024)
    with tempfile.TemporaryDirectory(prefix="pytest-main-stack-") as directory:
        executable = Path(directory) / "python.exe"
        executable.write_bytes(image)
        env = os.environ.copy()
        env["__PYVENV_LAUNCHER__"] = sys.executable
        env["PATH"] = str(base_executable.parent) + os.pathsep + env.get("PATH", "")
        cmd = [str(executable), os.path.abspath(__file__), f"--stack-mb={stack_mb}"]
        cmd.append("--stack-child")
        if cwd:
            cmd.extend(("--cwd", cwd))
        cmd.extend(pytest_args)
        return subprocess.call(cmd, env=env)  # noqa: S603 - trusted Python copy


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
    parser.add_argument("--stack-child", action="store_true", help=argparse.SUPPRESS)
    args, pytest_args = parser.parse_known_args()
    if args.stack_mb <= 0:
        parser.error("--stack-mb must be positive")

    if sys.platform == "win32" and not args.stack_child:
        sys.exit(_run_windows(args.stack_mb, args.cwd, pytest_args))

    _ensure_main_stack(args.stack_mb)
    if args.cwd:
        os.chdir(args.cwd)
    import pytest

    sys.exit(pytest.main(pytest_args))


if __name__ == "__main__":
    main()
