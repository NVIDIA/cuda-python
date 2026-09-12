# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Shared runner for tests that must execute a snippet in a fresh interpreter."""

from __future__ import annotations

import os
import subprocess
import sys
import tempfile
from collections.abc import Mapping

__all__ = ["run_python_snippet"]


def run_python_snippet(
    code: str,
    *,
    cwd: str | os.PathLike[str] | None = None,
    timeout: float | None = None,
    extra_env: Mapping[str, str] | None = None,
    unset_env: tuple[str, ...] = (),
    check: bool = True,
) -> subprocess.CompletedProcess[str]:
    """Run ``code`` with ``sys.executable -c`` and return the completed process.

    Output is captured as text. ``stdin`` is closed, so a child left
    interactive by ``PYTHONINSPECT`` exits instead of waiting.

    Args:
        code: Python source for the child interpreter.
        cwd: Directory to run from. When omitted, an empty temporary directory
            is created and cleaned up after the child exits.
        timeout: Seconds before ``subprocess.TimeoutExpired`` is raised.
        extra_env: Environment entries to set on top of the parent environment.
        unset_env: Environment variable names to remove from the child. Applied
            before ``extra_env``.
        check: Fail the calling test if the child exits non-zero, quoting the
            exit code and both streams. Pass ``False`` when the caller asserts
            on the exit code itself.

    Returns:
        The completed process, with ``stdout`` and ``stderr`` as ``str``.
    """
    env = os.environ.copy()
    for name in unset_env:
        env.pop(name, None)
    if extra_env:
        env.update(extra_env)

    def run(cwd: str | os.PathLike[str]) -> subprocess.CompletedProcess[str]:
        return subprocess.run(  # noqa: S603
            [sys.executable, "-c", code],
            capture_output=True,
            text=True,
            env=env,
            check=False,
            cwd=os.fspath(cwd),
            timeout=timeout,
            stdin=subprocess.DEVNULL,
        )

    if cwd is None:
        with tempfile.TemporaryDirectory() as tmpdir:
            result = run(tmpdir)
    else:
        result = run(cwd)
    if check:
        assert result.returncode == 0, (
            f"subprocess exited with code {result.returncode}\nstdout: {result.stdout}\nstderr: {result.stderr}"
        )
    return result
