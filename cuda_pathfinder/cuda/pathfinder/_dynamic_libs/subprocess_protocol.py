# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import json
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Literal, NoReturn

MODE_CANARY: Literal["canary"] = "canary"
MODE_LOAD: Literal["load"] = "load"
MODE_FIND: Literal["find"] = "find"
VALID_MODES: tuple[Literal["canary"], Literal["load"], Literal["find"]] = (MODE_CANARY, MODE_LOAD, MODE_FIND)

STATUS_OK: Literal["ok"] = "ok"
STATUS_NOT_FOUND: Literal["not-found"] = "not-found"

DYNAMIC_LIB_SUBPROCESS_MODULE = "cuda.pathfinder._dynamic_libs.dynamic_lib_subprocess"
DYNAMIC_LIB_SUBPROCESS_CWD = Path(__file__).resolve().parents[3]


@dataclass(frozen=True)
class DynamicLibSubprocessPayload:
    status: Literal["ok", "not-found"]
    abs_path: str | None
    error: dict[str, str] | None = None


def format_dynamic_lib_subprocess_payload(
    status: Literal["ok", "not-found"],
    abs_path: str | None,
    *,
    error: dict[str, str] | None = None,
) -> str:
    payload: dict[str, object] = {"status": status, "abs_path": abs_path}
    if error is not None:
        payload["error"] = error
    return json.dumps(payload)


def build_dynamic_lib_subprocess_command(mode: str, libname: str) -> list[str]:
    return [sys.executable, "-m", DYNAMIC_LIB_SUBPROCESS_MODULE, mode, libname]


def parse_dynamic_lib_subprocess_payload(
    stdout: str,
    *,
    libname: str,
    error_label: str,
) -> DynamicLibSubprocessPayload:
    # Use the final non-empty line in case earlier output lines are emitted.
    lines = [line for line in stdout.splitlines() if line.strip()]
    if not lines:
        raise RuntimeError(f"{error_label} produced no stdout payload for {libname!r}")
    try:
        payload = json.loads(lines[-1])
    except json.JSONDecodeError:
        raise RuntimeError(f"{error_label} emitted invalid JSON payload for {libname!r}: {lines[-1]!r}") from None
    if not isinstance(payload, dict):
        raise RuntimeError(f"{error_label} emitted unexpected payload for {libname!r}: {payload!r}")
    status = payload.get("status")
    abs_path = payload.get("abs_path")
    error = payload.get("error")

    def reject() -> NoReturn:
        raise RuntimeError(f"{error_label} emitted unexpected payload for {libname!r}: {payload!r}")

    if error is not None and not (
        isinstance(error, dict) and all(isinstance(k, str) and isinstance(v, str) for k, v in error.items())
    ):
        reject()
    if status == STATUS_OK and isinstance(abs_path, str):
        return DynamicLibSubprocessPayload(status=STATUS_OK, abs_path=abs_path, error=error)
    if status == STATUS_NOT_FOUND and abs_path is None:
        return DynamicLibSubprocessPayload(status=STATUS_NOT_FOUND, abs_path=None, error=error)
    reject()


def _coerce_subprocess_output(output: str | bytes | None) -> str:
    if isinstance(output, bytes):
        return output.decode(errors="replace")
    return "" if output is None else output


def raise_subprocess_child_process_error(
    error_label: str,
    *,
    returncode: int | None = None,
    timeout: float | None = None,
    stdout: str | bytes | None = None,
    stderr: str | bytes | None = None,
) -> NoReturn:
    if timeout is not None:
        first_line = f"{error_label} timed out after {timeout} seconds."
    else:
        first_line = f"{error_label} exited with code {returncode}."
    raise ChildProcessError(
        f"{first_line}\n"
        "--- stdout-from-child-process ---\n"
        f"{_coerce_subprocess_output(stdout)}<end-of-stdout-from-child-process>\n"
        "--- stderr-from-child-process ---\n"
        f"{_coerce_subprocess_output(stderr)}<end-of-stderr-from-child-process>\n"
    )


def run_dynamic_lib_subprocess(
    mode: str,
    libname: str,
    *,
    timeout: float,
    error_label: str,
) -> DynamicLibSubprocessPayload:
    """Run the dynamic-lib subprocess and parse its payload.

    Raises ``ChildProcessError`` if the child times out or exits non-zero;
    otherwise returns the parsed payload (which may itself be ``STATUS_NOT_FOUND``).
    """
    try:
        result = subprocess.run(  # noqa: S603 - trusted argv: current interpreter + internal probe module
            build_dynamic_lib_subprocess_command(mode, libname),
            capture_output=True,
            text=True,
            timeout=timeout,
            check=False,
            cwd=DYNAMIC_LIB_SUBPROCESS_CWD,
        )
    except subprocess.TimeoutExpired as exc:
        raise_subprocess_child_process_error(
            error_label, timeout=exc.timeout, stdout=exc.stdout, stderr=exc.stderr
        )

    if result.returncode != 0:
        raise_subprocess_child_process_error(
            error_label, returncode=result.returncode, stdout=result.stdout, stderr=result.stderr
        )

    return parse_dynamic_lib_subprocess_payload(result.stdout, libname=libname, error_label=error_label)
