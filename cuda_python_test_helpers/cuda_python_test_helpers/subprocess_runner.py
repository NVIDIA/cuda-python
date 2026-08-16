# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Run a Python snippet in an isolated interpreter subprocess.

Several tests must run their body in a fresh interpreter: to keep a
process-global side effect (the CPython pending-call queue, an import-time
monkeypatch, a deliberate glibc abort) from reaching the rest of the suite.
Each one had grown its own ``subprocess.run([sys.executable, "-c", code], ...)``
call, and they disagreed on which of the surrounding hazards they handled.

:func:`run_python_snippet` is the single place those hazards are handled:

* ``python -c`` puts the parent's working directory at the head of
  ``sys.path``. Run from ``cuda_core/`` (which holds a ``cuda/core/`` source
  tree) the child imports the source tree instead of the installed package, so
  ``cwd`` is a required argument rather than something a caller can forget.
* Output is always captured as text, so a failure message can quote both
  streams.
* ``stdin`` is closed. ``PYTHONINSPECT`` keeps the interpreter alive after
  ``-c``, and an inherited stdin would leave the implicit REPL waiting.
"""

from __future__ import annotations

import os
import subprocess
import sys
from collections.abc import Mapping

__all__ = ["run_python_snippet"]


def run_python_snippet(
    code: str,
    *,
    cwd: str | os.PathLike[str],
    timeout: float | None = None,
    extra_env: Mapping[str, str] | None = None,
    unset_env: tuple[str, ...] = (),
    check: bool = True,
) -> subprocess.CompletedProcess[str]:
    """Run ``code`` with ``sys.executable -c`` and return the completed process.

    Args:
        code: Python source for the child interpreter.
        cwd: Directory to run from. Required, because the parent's working
            directory would otherwise land on the child's ``sys.path`` and can
            shadow the installed package with a source tree. Pass a
            ``tmp_path`` fixture or another directory that holds no importable
            ``cuda`` package.
        timeout: Seconds before ``subprocess.TimeoutExpired`` is raised. ``None``
            waits indefinitely.
        extra_env: Environment entries to set on top of the parent environment.
        unset_env: Environment variable names to remove from the child. Use for
            values that must not leak in from the parent, such as
            ``PYTHONPATH`` when the child has to import the installed wheel.
        check: When true, fail the calling test if the child exits non-zero. The
            assertion message quotes the exit code and both output streams. Pass
            ``False`` when a non-zero exit is itself the expected result.

    Returns:
        The completed process, with ``stdout`` and ``stderr`` as ``str``.
    """
    env = os.environ.copy()
    for name in unset_env:
        env.pop(name, None)
    if extra_env:
        env.update(extra_env)

    result = subprocess.run(  # noqa: S603 - fixed argv, interpreter is sys.executable
        [sys.executable, "-c", code],
        capture_output=True,
        text=True,
        env=env,
        check=False,
        cwd=os.fspath(cwd),
        timeout=timeout,
        stdin=subprocess.DEVNULL,
    )
    if check:
        assert result.returncode == 0, (
            f"subprocess exited with code {result.returncode}\nstdout: {result.stdout}\nstderr: {result.stderr}"
        )
    return result
