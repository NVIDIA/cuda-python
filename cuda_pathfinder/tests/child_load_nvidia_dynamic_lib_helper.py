# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import subprocess
import sys
import tempfile
from pathlib import Path

from cuda.pathfinder._testing.load_nvidia_dynamic_lib_subprocess import DYNAMIC_LIB_NOT_FOUND_MARKER

LOAD_NVIDIA_DYNAMIC_LIB_SUBPROCESS_MODULE = "cuda.pathfinder._testing.load_nvidia_dynamic_lib_subprocess"
# Launch the child from a neutral directory so `python -m cuda.pathfinder...`
# resolves the installed package instead of the source checkout. In CI the
# checkout does not contain the generated `_version.py` file.
LOAD_NVIDIA_DYNAMIC_LIB_SUBPROCESS_CWD = Path(tempfile.gettempdir())
PROCESS_TIMED_OUT = -9


def build_child_process_failed_for_libname_message(libname: str, result: subprocess.CompletedProcess[str]) -> str:
    return (
        f"Child process failed for {libname=!r} with exit code {result.returncode}\n"
        f"--- stdout-from-child-process ---\n{result.stdout}<end-of-stdout-from-child-process>\n"
        f"--- stderr-from-child-process ---\n{result.stderr}<end-of-stderr-from-child-process>\n"
    )


def child_process_reported_dynamic_lib_not_found(result: subprocess.CompletedProcess[str]) -> bool:
    return result.stdout.startswith(DYNAMIC_LIB_NOT_FOUND_MARKER)


def run_load_nvidia_dynamic_lib_in_subprocess(
    libname: str,
    *,
    timeout: float,
) -> subprocess.CompletedProcess[str]:
    command = [sys.executable, "-m", LOAD_NVIDIA_DYNAMIC_LIB_SUBPROCESS_MODULE, libname]
    try:
        return subprocess.run(  # noqa: S603 - trusted argv: current interpreter + internal test helper module
            command,
            capture_output=True,
            text=True,
            timeout=timeout,
            check=False,
            cwd=LOAD_NVIDIA_DYNAMIC_LIB_SUBPROCESS_CWD,
        )
    except subprocess.TimeoutExpired:
        return subprocess.CompletedProcess(
            args=command,
            returncode=PROCESS_TIMED_OUT,
            stdout="",
            stderr=f"Process timed out after {timeout} seconds and was terminated.",
        )
