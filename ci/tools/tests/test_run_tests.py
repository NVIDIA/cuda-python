# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import json
import os
import shutil
import subprocess
import sys
import tomllib
import zipfile
from pathlib import Path

import pytest

TOOLS = Path(__file__).resolve().parents[1]


@pytest.fixture
def run_tests(tmp_path, monkeypatch):
    checkout = tmp_path / "checkout"
    tools = checkout / "ci" / "tools"
    tools.mkdir(parents=True)
    for name in ("run-tests", "cuda_core_bindings_floor.py", "fold_pytest_sections.py"):
        shutil.copyfile(TOOLS / name, tools / name)

    pathfinder = checkout / "cuda_pathfinder"
    pathfinder.mkdir()
    (pathfinder / "cuda_pathfinder-1.0-py3-none-any.whl").touch()
    core_dir = checkout / "cuda_core"
    core_dir.mkdir()
    core_wheel = core_dir / "cuda_core-1.3.0-cp312-cp312-linux_x86_64.whl"
    bindings_dir = checkout / "bindings with spaces"
    bindings_dir.mkdir()

    # Run the real shell and metadata reader without installing packages or using a GPU.
    bin_dir = tmp_path / "bin"
    bin_dir.mkdir()
    python = bin_dir / "python"
    python.write_text('#!/usr/bin/env bash\nexec "$RUN_TESTS_PYTHON" "$@"\n', encoding="utf-8")
    python.chmod(0o755)
    for command in ("pip", "pytest"):
        executable = bin_dir / command
        executable.write_text(
            f"#!{sys.executable}\n"
            "import json, os, sys\n"
            "from pathlib import Path\n"
            'with open(os.environ["RUN_TESTS_COMMAND_LOG"], "a", encoding="utf-8") as log:\n'
            '    log.write(json.dumps([Path(sys.argv[0]).name, sys.argv[1:]]) + "\\n")\n',
            encoding="utf-8",
        )
        executable.chmod(0o755)

    command_log = tmp_path / "commands.jsonl"
    monkeypatch.setenv("PATH", f"{bin_dir}{os.pathsep}{os.environ['PATH']}")
    monkeypatch.setenv("RUN_TESTS_PYTHON", sys.executable)
    monkeypatch.setenv("RUN_TESTS_COMMAND_LOG", str(command_log))
    monkeypatch.setenv("CUDA_CORE_ARTIFACTS_DIR", str(core_dir))
    monkeypatch.setenv("CUDA_BINDINGS_ARTIFACTS_DIR", str(bindings_dir))
    monkeypatch.setenv("BUILD_CUDA_MAJOR", "13")
    monkeypatch.setenv("BUILD_CUDA_MINOR", "5")
    monkeypatch.setenv("SANITIZER_CMD", "")
    monkeypatch.setenv("SKIP_CYTHON_TEST", "1")
    monkeypatch.setenv("TORCH_VER", "2.9.1")
    monkeypatch.setenv("TORCH_CUDA", "cu130")
    monkeypatch.delenv("GITHUB_ENV", raising=False)

    def run(*, mode="core", source="floor", floor="13.4.1", cuda_version="13.0.2", local_ctk=True):
        metadata = (
            "Metadata-Version: 2.4\nName: cuda-core\nVersion: 1.3.0\n"
            'Requires-Dist: cuda-bindings[all]<13,>=12.9.8; extra == "cu12"\n'
            f'Requires-Dist: cuda-bindings[all]<14,>={floor}; extra == "cu13"\n'
        )
        with zipfile.ZipFile(core_wheel, "w") as wheel:
            wheel.writestr("cuda/core/__init__.py", "")
            wheel.writestr("cuda_core-1.3.0.dist-info/METADATA", metadata)
        bindings_version = "12.9.8" if source == "backport" else "13.4.2.dev1"
        bindings_wheel = bindings_dir / f"cuda_bindings-{bindings_version}-cp312-cp312-linux_x86_64.whl"
        bindings_wheel.touch()
        monkeypatch.setenv("BINDINGS_SOURCE", source)
        monkeypatch.setenv("CUDA_VER", cuda_version)
        monkeypatch.setenv("TEST_CUDA_MINOR", cuda_version.split(".")[1])
        monkeypatch.setenv("LOCAL_CTK", "1" if local_ctk else "0")
        subprocess.run(["bash", str(tools / "run-tests"), mode], cwd=checkout, check=True)  # noqa: S603, S607

        calls = [json.loads(line) for line in command_log.read_text(encoding="utf-8").splitlines()]
        if mode == "core":
            assert any(command == "pytest" and args[-1] == "tests/" for command, args in calls)
        installs = [args for command, args in calls if command == "pip" and args[0] == "install"]
        bindings_install = next(
            args
            for args in installs
            if any(arg.startswith("cuda-bindings==") or Path(arg).name.startswith("cuda_bindings-") for arg in args)
        )
        core_install = next(
            args
            for args in installs
            if any(arg.startswith("cuda-core[") or Path(arg).name.startswith("cuda_core-") for arg in args)
        )
        if mode == "core":
            major = cuda_version.split(".")[0]
            group = core_install[core_install.index("--group") + 1]
            assert group == f"test-cu{major}" + ("-local" if local_ctk else "")
            toolkit_args = [arg for args in installs for arg in args if arg.startswith("cuda-toolkit")]
            if local_ctk:
                assert toolkit_args == []
            else:
                minor = ".".join(cuda_version.split(".")[:2])
                assert f"cuda-toolkit=={minor}.*" in core_install
                assert f"cuda-toolkit[curand]=={minor}.*" in toolkit_args
        return bindings_install, core_install, str(bindings_wheel)

    return run


@pytest.mark.parametrize("local_ctk", [True, False], ids=["local", "wheels"])
@pytest.mark.parametrize("floor", ["13.4.1", "13.5.0"], ids=["bump-window", "floor-caught-up"])
@pytest.mark.agent_authored(model="gpt-6")
def test_floor_uses_current_bindings_only_during_toolkit_bump(run_tests, local_ctk, floor):
    bindings, core, wheel = run_tests(local_ctk=local_ctk, floor=floor)
    expected = wheel + ("" if local_ctk else "[all]") if floor == "13.4.1" else "cuda-bindings==13.5.0"
    assert bindings == ["install", expected]


@pytest.mark.parametrize("local_ctk", [True, False], ids=["local", "wheels"])
@pytest.mark.agent_authored(model="gpt-6")
def test_current_minor_uses_local_or_wheel_test_dependencies(run_tests, local_ctk):
    bindings, core, wheel = run_tests(source="main", cuda_version="13.5.0", local_ctk=local_ctk)
    assert bindings == ["install", wheel + ("" if local_ctk else "[all]")]


@pytest.mark.parametrize(("source", "cuda_version"), [("floor", "12.6.3"), ("backport", "12.9.1")])
@pytest.mark.parametrize("local_ctk", [True, False], ids=["local", "wheels"])
@pytest.mark.agent_authored(model="gpt-6")
def test_prior_major_keeps_its_floor_or_backport_bindings(run_tests, source, cuda_version, local_ctk):
    bindings, core, wheel = run_tests(source=source, cuda_version=cuda_version, local_ctk=local_ctk)
    expected = "cuda-bindings==12.9.8" if source == "floor" else wheel + ("" if local_ctk else "[all]")
    assert bindings == ["install", expected]


@pytest.mark.parametrize("major", [12, 13])
@pytest.mark.agent_authored(model="gpt-6")
def test_local_core_groups_keep_test_dependencies_without_toolkit_wheels(major):
    pyproject = TOOLS.parents[1] / "cuda_core" / "pyproject.toml"
    groups = tomllib.loads(pyproject.read_text(encoding="utf-8"))["dependency-groups"]

    def requirements(name):
        for item in groups[name]:
            if isinstance(item, str):
                yield item
            else:
                yield from requirements(item["include-group"])

    local = set(requirements(f"test-cu{major}-local"))
    assert set(requirements("test")) | set(requirements("ml-dtypes")) <= local
    assert any(requirement.partition(";")[0].strip() == f"cupy-cuda{major}x" for requirement in local)
    assert not any(requirement.startswith("cuda-toolkit") for requirement in local)
    assert set(requirements(f"test-cu{major}")) == local | {f"cuda-toolkit[cudart]=={major}.*"}


@pytest.mark.parametrize("mode", ["nightly-pytorch", "nightly-numba-cuda", "nightly-numba-cuda-mlir"])
@pytest.mark.agent_authored(model="gpt-6")
def test_nightly_local_core_uses_current_bindings_during_toolkit_bump(run_tests, mode):
    bindings, core, wheel = run_tests(mode=mode, local_ctk=False)
    assert wheel + "[all]" in bindings
    assert any(Path(arg).name.startswith("cuda_core-") for arg in core)


@pytest.mark.agent_authored(model="gpt-6")
def test_released_core_nightly_does_not_use_local_core_fallback(run_tests):
    bindings, core, wheel = run_tests(mode="nightly-cuda-core", local_ctk=False)
    assert "cuda-bindings==13.4.1" in bindings
    assert wheel + "[all]" not in bindings
    assert "cuda-core[cu13]" in core
    assert not any(Path(arg).name.startswith("cuda_core-") for arg in core)
