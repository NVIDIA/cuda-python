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

from ci.tools.bindings_config import load_config

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


def _run_env_vars(
    tmp_path: Path,
    *,
    bindings_source: str,
    pathfinder_source: str,
    cuda_version: str = "13.3.0",
    bindings_root: str = "cuda_bindings",
) -> subprocess.CompletedProcess[str]:
    for relative in (
        f"{bindings_root}/dist",
        f"{bindings_root}/tests/cython",
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
        "CUDA_VER": cuda_version,
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
        env["BINDINGS_SOURCE_DIR"] = bindings_root
    else:
        env["DEFAULT_BINDINGS_SOURCE_DIR"] = bindings_root
    return subprocess.run(  # noqa: S603 - invokes the repository script under test
        [str(ENV_VARS), "test"],
        cwd=tmp_path,
        env=env,
        check=False,
        capture_output=True,
        text=True,
    )


def _read_github_env(path: Path) -> dict[str, str]:
    return dict(line.split("=", 1) for line in path.read_text(encoding="utf-8").splitlines())


@pytest.mark.agent_authored(model="gpt-5.6-sol")
def test_env_vars_rejects_unknown_mode() -> None:
    result = subprocess.run(  # noqa: S603 - invokes the repository script under test
        [str(ENV_VARS), "unknown"],
        check=False,
        capture_output=True,
        text=True,
    )

    assert result.returncode == 1
    assert "build mode must be build or test" in result.stderr


@pytest.mark.agent_authored(model="gpt-5.6")
def test_build_env_uses_registry_packages_without_generic_bindings_aliases(tmp_path: Path) -> None:
    github_env = tmp_path / "github-env"
    env = {
        **os.environ,
        "GITHUB_ENV": str(github_env),
        "GITHUB_PATH": str(tmp_path / "github-path"),
        "HOST_PLATFORM": "linux-64",
        "PY_VER": "3.13",
        "SHA": "abcdef0",
    }

    result = subprocess.run(  # noqa: S603 - invokes the repository script under test
        [str(ENV_VARS), "build"],
        cwd=REPO_ROOT,
        env=env,
        check=False,
        capture_output=True,
        text=True,
    )

    assert result.returncode == 0, result.stderr
    values = _read_github_env(github_env)
    config = load_config()
    for status in ("current", "maintenance"):
        package = config.package_for_release_status(status)
        prefix = status.upper()
        assert values[f"{prefix}_BINDINGS_ROOT"] == package.package_root
        assert values[f"{prefix}_CUDA_VERSION"] == package.toolkit_version
        assert values[f"{prefix}_CUDA_MAJOR"] == package.cuda_major
        assert values[f"{prefix}_CUDA_VARIANT"] == package.cuda_variant
        assert values[f"{prefix}_BINDINGS_ARTIFACT_NAME"].endswith("-abcdef0")

    assert (
        not {
            "CUDA_BINDINGS_ARTIFACT_BASENAME",
            "CUDA_BINDINGS_ARTIFACT_NAME",
            "CUDA_BINDINGS_ARTIFACTS_DIR",
            "CUDA_BINDINGS_CYTHON_TESTS_DIR",
        }
        & values.keys()
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
    ("bindings_source", "cuda_version", "bindings_root", "expected_artifact"),
    [
        ("local", "13.4.2", "cuda_bindings", "cuda-python-wheel-cuda13.4.2"),
        ("local", "12.9.1", "cuda_bindings_12", "cuda-python-wheel-cuda12.9.1"),
        ("floor", "13.0.2", "cuda_bindings", None),
        ("floor", "12.6.3", "cuda_bindings", None),
    ],
)
@pytest.mark.agent_authored(model="gpt-6-astra")
def test_cuda_python_artifact_name_exists_only_for_local_bindings(
    tmp_path: Path,
    bindings_source: str,
    cuda_version: str,
    bindings_root: str,
    expected_artifact: str | None,
) -> None:
    result = _run_env_vars(
        tmp_path,
        bindings_source=bindings_source,
        pathfinder_source="artifact",
        cuda_version=cuda_version,
        bindings_root=bindings_root,
    )

    assert result.returncode == 0, result.stderr
    github_env = _read_github_env(tmp_path / "github-env")
    assert {
        "CUDA_BINDINGS_ARTIFACT_BASENAME",
        "CUDA_BINDINGS_ARTIFACT_NAME",
        "CUDA_BINDINGS_ARTIFACTS_DIR",
        "CUDA_BINDINGS_CYTHON_TESTS_DIR",
        "CUDA_CORE_CYTHON_TEST_ARTIFACT_NAME",
    } <= github_env.keys()
    assert github_env["BINDINGS_SOURCE"] == bindings_source
    assert github_env["CUDA_BINDINGS_ROOT"] == bindings_root
    assert github_env["SKIP_CUDA_BINDINGS_TEST"] == ("1" if bindings_source == "floor" else "0")
    assert github_env["SKIP_CYTHON_TEST"] == ("1" if bindings_source == "floor" else "0")
    assert github_env["CUDA_CORE_CYTHON_TEST_ARTIFACT_NAME"].endswith(f"-cu{cuda_version.split('.')[0]}-tests")
    if expected_artifact is None:
        assert "CUDA_PYTHON_ARTIFACT_NAME" not in github_env
    else:
        assert github_env["CUDA_PYTHON_ARTIFACT_NAME"] == expected_artifact


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
