# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import os
import subprocess
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[3]
RUN_TESTS = REPO_ROOT / "ci" / "tools" / "run-tests"
ENV_VARS = REPO_ROOT / "ci" / "tools" / "env-vars"


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
printf 'python %s\\n' "$*" >> "$COMMAND_LOG"
""".strip(),
    )
    _write_command(fake_bin, "pip", 'printf \'pip %s\\n\' "$*" >> "$COMMAND_LOG"')
    _write_command(fake_bin, "pytest", ":")
    env = {
        **os.environ,
        "COMMAND_LOG": str(command_log),
        "PATH": f"{fake_bin}{os.pathsep}{os.environ['PATH']}",
        "CUDA_PATHFINDER_TEST_LOAD_NVIDIA_DYNAMIC_LIB_STRICTNESS": "see_what_works",
        "CUDA_PATHFINDER_TEST_FIND_NVIDIA_HEADERS_STRICTNESS": "see_what_works",
        "CUDA_PATHFINDER_TEST_FIND_NVIDIA_BITCODE_LIB_STRICTNESS": "see_what_works",
    }
    (tmp_path / "cuda_pathfinder").mkdir()
    return env, command_log


def _run_env_vars(
    tmp_path: Path,
    *,
    bindings_source: str,
    pathfinder_source: str,
) -> subprocess.CompletedProcess[str]:
    for relative in (
        "cuda_bindings/dist",
        "cuda_bindings/tests/cython",
        "cuda_core/dist",
        "cuda_core/tests/cython",
        "cuda_core/tests/test_binaries",
    ):
        (tmp_path / relative).mkdir(parents=True, exist_ok=True)
    github_env = tmp_path / "github-env"
    github_path = tmp_path / "github-path"
    env = {
        **os.environ,
        "BINDINGS_SOURCE": bindings_source,
        "CUDA_VER": "13.3.0",
        "GITHUB_ENV": str(github_env),
        "GITHUB_PATH": str(github_path),
        "HOST_PLATFORM": "linux-64",
        "LOCAL_CTK": "0",
        "PATHFINDER_SOURCE": pathfinder_source,
        "PY_VER": "3.13",
        "SHA": "abcdef",
        "SKIP_BINDINGS_TEST_OVERRIDE": "0",
    }
    if bindings_source == "local":
        env["BINDINGS_SOURCE_DIR"] = "cuda_bindings"
    else:
        env["DEFAULT_BINDINGS_SOURCE_DIR"] = "cuda_bindings"
    return subprocess.run(  # noqa: S603 - invokes the repository script under test
        [str(ENV_VARS), "test"],
        cwd=tmp_path,
        env=env,
        check=False,
        capture_output=True,
        text=True,
    )


@pytest.mark.parametrize("wheel_count", [0, 2])
@pytest.mark.agent_authored(model="gpt-5.6")
def test_artifact_pathfinder_requires_exactly_one_wheel(tmp_path: Path, wheel_count: int) -> None:
    env, _ = _run_tests_env(tmp_path)
    env["PATHFINDER_SOURCE"] = "artifact"
    for index in range(wheel_count):
        (tmp_path / "cuda_pathfinder" / f"cuda_pathfinder-{index}.whl").touch()

    result = subprocess.run(  # noqa: S603 - invokes the repository script under test
        [str(RUN_TESTS), "pathfinder"],
        cwd=tmp_path,
        env=env,
        check=False,
        capture_output=True,
        text=True,
    )

    assert result.returncode != 0
    assert "Expected exactly one cuda-pathfinder wheel" in result.stderr


@pytest.mark.agent_authored(model="gpt-5.6")
def test_artifact_pathfinder_installs_the_only_wheel(tmp_path: Path) -> None:
    env, command_log = _run_tests_env(tmp_path)
    env["PATHFINDER_SOURCE"] = "artifact"
    wheel = tmp_path / "cuda_pathfinder" / "cuda_pathfinder-1.0-py3-none-any.whl"
    wheel.touch()

    subprocess.run(  # noqa: S603 - invokes the repository script under test
        [str(RUN_TESTS), "pathfinder"],
        cwd=tmp_path,
        env=env,
        check=True,
        capture_output=True,
        text=True,
    )

    assert f"pip install ./{wheel.name} --group test" in command_log.read_text(encoding="utf-8")


@pytest.mark.agent_authored(model="gpt-5.6")
def test_published_pathfinder_is_limited_to_bindings_release_tests(tmp_path: Path) -> None:
    env, _ = _run_tests_env(tmp_path)
    env["PATHFINDER_SOURCE"] = "published"

    result = subprocess.run(  # noqa: S603 - invokes the repository script under test
        [str(RUN_TESTS), "pathfinder"],
        cwd=tmp_path,
        env=env,
        check=False,
        capture_output=True,
        text=True,
    )

    assert result.returncode != 0
    assert "published for bindings release tests" in result.stderr


@pytest.mark.agent_authored(model="gpt-5.6")
def test_published_pathfinder_supports_bindings_release_tests(tmp_path: Path) -> None:
    env, command_log = _run_tests_env(tmp_path)
    env.update(
        CUDA_BINDINGS_ARTIFACTS_DIR=str(tmp_path / "bindings-artifacts"),
        CUDA_BINDINGS_ROOT="cuda_bindings",
        LOCAL_CTK="1",
        PATHFINDER_SOURCE="published",
        SANITIZER_CMD="",
        SKIP_CYTHON_TEST="1",
    )
    (tmp_path / "cuda_bindings").mkdir()
    (tmp_path / "cuda_bindings" / "pyproject.toml").write_text("[dependency-groups]\ntest = []\n", encoding="utf-8")
    (tmp_path / "bindings-artifacts").mkdir()
    (tmp_path / "bindings-artifacts" / "cuda_bindings-13.3.whl").touch()

    subprocess.run(  # noqa: S603 - invokes the repository script under test
        [str(RUN_TESTS), "bindings"],
        cwd=tmp_path,
        env=env,
        check=True,
        capture_output=True,
        text=True,
    )

    calls = command_log.read_text(encoding="utf-8").splitlines()
    assert calls[0].startswith("pip install cuda-pathfinder --group test")
    assert calls[1].startswith("python -m pip install ")


@pytest.mark.parametrize(
    ("bindings_source", "expected_artifact"),
    [("local", "cuda-python-wheel-cuda13.3.0"), ("published", None)],
)
@pytest.mark.agent_authored(model="gpt-5.6")
def test_cuda_python_artifact_name_exists_only_for_local_bindings(
    tmp_path: Path,
    bindings_source: str,
    expected_artifact: str | None,
) -> None:
    result = _run_env_vars(
        tmp_path,
        bindings_source=bindings_source,
        pathfinder_source="artifact",
    )

    assert result.returncode == 0, result.stderr
    github_env = (tmp_path / "github-env").read_text(encoding="utf-8").splitlines()
    artifact_lines = [line for line in github_env if line.startswith("CUDA_PYTHON_ARTIFACT_NAME=")]
    if expected_artifact is None:
        assert artifact_lines == []
    else:
        assert artifact_lines == [f"CUDA_PYTHON_ARTIFACT_NAME={expected_artifact}"]
