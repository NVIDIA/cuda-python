# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import subprocess
import tempfile
from pathlib import Path

from cuda.pathfinder._dynamic_libs.subprocess_protocol import (
    DYNAMIC_LIB_SUBPROCESS_MODULE,
    MODE_LOAD,
    STATUS_NOT_FOUND,
    build_dynamic_lib_subprocess_command,
    parse_dynamic_lib_subprocess_payload,
)

LOAD_NVIDIA_DYNAMIC_LIB_SUBPROCESS_MODULE = DYNAMIC_LIB_SUBPROCESS_MODULE
LOAD_NVIDIA_DYNAMIC_LIB_SUBPROCESS_MODE = MODE_LOAD
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


def parse_dynamic_lib_subprocess_result(
    result: subprocess.CompletedProcess[str],
    *,
    libname: str,
):
    return parse_dynamic_lib_subprocess_payload(
        result.stdout,
        libname=libname,
        error_label="Load subprocess child process",
    )


def child_process_reported_dynamic_lib_not_found(
    result: subprocess.CompletedProcess[str],
    *,
    libname: str,
) -> bool:
    payload = parse_dynamic_lib_subprocess_result(result, libname=libname)
    return payload.status == STATUS_NOT_FOUND


def run_load_nvidia_dynamic_lib_in_subprocess(
    libname: str,
    *,
    timeout: float,
) -> subprocess.CompletedProcess[str]:
    command = build_dynamic_lib_subprocess_command(LOAD_NVIDIA_DYNAMIC_LIB_SUBPROCESS_MODE, libname)
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
