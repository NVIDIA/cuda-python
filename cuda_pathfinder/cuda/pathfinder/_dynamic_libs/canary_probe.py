# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""The CTK-root canary probe, shared by library, header and binary discovery.

A "canary" library is loaded by the OS dynamic loader in a fresh Python
process, and the CUDA Toolkit root is derived from where the loader found it.
"""

from __future__ import annotations

import functools
import subprocess

from cuda.pathfinder._dynamic_libs.search_steps import derive_ctk_root
from cuda.pathfinder._dynamic_libs.subprocess_protocol import (
    DYNAMIC_LIB_SUBPROCESS_CWD,
    MODE_CANARY,
    STATUS_OK,
    DynamicLibSubprocessPayload,
    build_dynamic_lib_subprocess_command,
    parse_dynamic_lib_subprocess_payload,
)

_CANARY_PROBE_TIMEOUT_SECONDS = 10.0


def _coerce_subprocess_output(output: str | bytes | None) -> str:
    if isinstance(output, bytes):
        return output.decode(errors="replace")
    return "" if output is None else output


def _raise_canary_probe_child_process_error(
    *,
    returncode: int | None = None,
    timeout: float | None = None,
    stderr: str | bytes | None = None,
) -> None:
    if timeout is None:
        error_line = f"Canary probe child process exited with code {returncode}."
    else:
        error_line = f"Canary probe child process timed out after {timeout} seconds."
    raise ChildProcessError(
        f"{error_line}\n"
        "--- stderr-from-child-process ---\n"
        f"{_coerce_subprocess_output(stderr)}"
        "<end-of-stderr-from-child-process>\n"
    )


@functools.cache
def _resolve_system_loaded_abs_path_in_subprocess(
    libname: str,
    *,
    timeout: float = _CANARY_PROBE_TIMEOUT_SECONDS,
) -> str | None:
    """Resolve a canary library's absolute path in a fresh Python subprocess."""
    try:
        result = subprocess.run(  # noqa: S603 - trusted argv: current interpreter + internal probe module
            build_dynamic_lib_subprocess_command(MODE_CANARY, libname),
            capture_output=True,
            text=True,
            timeout=timeout,
            check=False,
            cwd=DYNAMIC_LIB_SUBPROCESS_CWD,
        )
    except subprocess.TimeoutExpired as exc:
        _raise_canary_probe_child_process_error(timeout=exc.timeout, stderr=exc.stderr)

    if result.returncode != 0:
        _raise_canary_probe_child_process_error(returncode=result.returncode, stderr=result.stderr)

    payload: DynamicLibSubprocessPayload = parse_dynamic_lib_subprocess_payload(
        result.stdout,
        libname=libname,
        error_label="Canary probe child process",
    )
    abs_path: str | None = payload.abs_path
    if payload.status == STATUS_OK:
        return abs_path
    return None


def resolve_ctk_root_via_canary(canary_libname: str) -> str | None:
    """Resolve the CUDA Toolkit root from a system-loadable canary library.

    The canary library's absolute path is resolved by the OS dynamic loader in
    an isolated subprocess, which honors ``LD_LIBRARY_PATH`` on Linux and the
    native DLL search on Windows. The toolkit root is then derived from that
    path. Returns ``None`` if the canary cannot be resolved or no root can be
    derived.
    """
    canary_abs_path = _resolve_system_loaded_abs_path_in_subprocess(canary_libname)
    if canary_abs_path is None:
        return None
    ctk_root: str | None = derive_ctk_root(canary_abs_path)
    return ctk_root
