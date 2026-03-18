# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import json
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

MODE_CANARY: Literal["canary"] = "canary"
MODE_LOAD: Literal["load"] = "load"
VALID_MODES: tuple[Literal["canary"], Literal["load"]] = (MODE_CANARY, MODE_LOAD)

STATUS_OK: Literal["ok"] = "ok"
STATUS_NOT_FOUND: Literal["not-found"] = "not-found"

DYNAMIC_LIB_SUBPROCESS_MODULE = "cuda.pathfinder._dynamic_libs.dynamic_lib_subprocess"
DYNAMIC_LIB_SUBPROCESS_CWD = Path(__file__).resolve().parents[3]


@dataclass(frozen=True)
class DynamicLibSubprocessPayload:
    status: Literal["ok", "not-found"]
    abs_path: str | None


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
    if status == STATUS_OK:
        if not isinstance(abs_path, str):
            raise RuntimeError(f"{error_label} emitted unexpected payload for {libname!r}: {payload!r}")
        return DynamicLibSubprocessPayload(status=STATUS_OK, abs_path=abs_path)
    if status == STATUS_NOT_FOUND:
        if abs_path is not None:
            raise RuntimeError(f"{error_label} emitted unexpected payload for {libname!r}: {payload!r}")
        return DynamicLibSubprocessPayload(status=STATUS_NOT_FOUND, abs_path=None)
    raise RuntimeError(f"{error_label} emitted unexpected payload for {libname!r}: {payload!r}")
