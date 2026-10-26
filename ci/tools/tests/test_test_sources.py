# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import os
import subprocess
import sys
import zipfile
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[3]
RUN_TESTS = REPO_ROOT / "ci" / "tools" / "run-tests"


def _write_command(directory: Path, name: str, body: str) -> None:
    path = directory / name
    path.write_text(f"#!/usr/bin/env bash\nset -eu\n{body}\n", encoding="utf-8")
    path.chmod(0o755)


def _run_tests_env(tmp_path: Path) -> tuple[dict[str, str], Path]:
    fake_bin = tmp_path / "bin"
    fake_bin.mkdir()
    command_log = tmp_path / "commands.log"
    _write_command(
        fake_bin,
        "python",
        """
if [[ "${1:-}" == "-c" ]]; then
  exit 1
fi
if [[ "${1:-}" == "ci/tools/cuda_core_bindings_floor.py" ]]; then
  exec "$REAL_PYTHON" "$@"
fi
printf 'python %s\\n' "$*" >> "$COMMAND_LOG"
""".strip(),
    )
    _write_command(fake_bin, "pip", 'printf \'pip %s\\n\' "$*" >> "$COMMAND_LOG"')
    _write_command(fake_bin, "pytest", ":")
    env = {
        **os.environ,
        "COMMAND_LOG": str(command_log),
        "REAL_PYTHON": sys.executable,
        "PATH": f"{fake_bin}{os.pathsep}{os.environ['PATH']}",
        "CUDA_PATHFINDER_TEST_LOAD_NVIDIA_DYNAMIC_LIB_STRICTNESS": "see_what_works",
        "CUDA_PATHFINDER_TEST_FIND_NVIDIA_HEADERS_STRICTNESS": "see_what_works",
        "CUDA_PATHFINDER_TEST_FIND_NVIDIA_BITCODE_LIB_STRICTNESS": "see_what_works",
    }
    (tmp_path / "cuda_pathfinder").mkdir()
    return env, command_log


@pytest.mark.agent_authored(model="gpt-6-astra")
@pytest.mark.parametrize(("cuda_version", "wheel_floor"), [("12.6.3", "12.9.9"), ("13.0.2", "13.4.2")])
def test_core_install_uses_wheel_floor_with_older_toolkit(tmp_path: Path, cuda_version: str, wheel_floor: str) -> None:
    env, command_log = _run_tests_env(tmp_path)
    (tmp_path / "cuda_pathfinder" / "cuda_pathfinder-1.0-py3-none-any.whl").touch()
    core_dist = tmp_path / "cuda_core" / "dist"
    core_dist.mkdir(parents=True)
    wheel = core_dist / "cuda_core-1.3.0-cp313-cp313-linux_x86_64.whl"
    major = int(cuda_version.split(".")[0])
    with zipfile.ZipFile(wheel, "w") as archive:
        archive.writestr(
            "cuda_core-1.3.0.dist-info/METADATA",
            f'Requires-Dist: cuda-bindings[all]>={wheel_floor},<{major + 1}; extra == "cu{major}"\n',
        )
    tools = tmp_path / "ci" / "tools"
    tools.mkdir(parents=True)
    floor_script = REPO_ROOT / "ci" / "tools" / "cuda_core_bindings_floor.py"
    (tools / floor_script.name).write_text(floor_script.read_text(encoding="utf-8"), encoding="utf-8")
    env.update(
        BINDINGS_SOURCE="floor",
        CUDA_CORE_ARTIFACTS_DIR=str(core_dist),
        CUDA_VER=cuda_version,
        LOCAL_CTK="0",
        PATHFINDER_SOURCE="artifact",
        SANITIZER_CMD="",
        SKIP_CYTHON_TEST="1",
    )

    result = subprocess.run(  # noqa: S603 - repository script with install/test commands replaced by recording stubs
        [str(RUN_TESTS), "core"],
        cwd=tmp_path,
        env=env,
        check=False,
        capture_output=True,
        text=True,
    )

    assert result.returncode == 0, result.stderr
    calls = command_log.read_text(encoding="utf-8").splitlines()
    assert f"pip install cuda-bindings=={wheel_floor}" in calls
    toolkit_minor = ".".join(cuda_version.split(".")[:2])
    assert any(f"cuda-toolkit=={toolkit_minor}.*" in call for call in calls)
