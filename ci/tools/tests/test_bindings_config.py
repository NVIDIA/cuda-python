# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import json
import runpy
import shutil
import subprocess
import tomllib
from pathlib import Path

import pytest
import setuptools
from packaging.version import Version
from setuptools_scm import get_version

from ci.tools import check_pixi_cuda_version
from ci.tools.bindings_config import BindingsPackage, load_config, main
from ci.tools.compute_ci_plan import compute_workplan

ROOT = Path(__file__).resolve().parents[3]


@pytest.fixture
def legacy_config(tmp_path):
    path = tmp_path / "versions.yml"
    path.write_text(
        'cuda:\n  build:\n    version: "13.4.2"\n  prev_build:\n    version: "12.9.1"\n'
        'python:\n  supported: ["3.11", "3.14t"]\n'
    )
    return path


@pytest.mark.agent_authored(model="gpt-6")
def test_normalizes_legacy_toolkit_pins(legacy_config):
    config = load_config(legacy_config)
    current = config.package_for_release_status("current")
    maintenance = config.get_package("cuda_bindings_12")
    assert current == BindingsPackage("cuda_bindings_13", (13, 4, 2), "current")
    assert maintenance == config.package_for_release_status("maintenance")
    assert maintenance.toolkit_version == "12.9.1"
    assert maintenance.ctk_target == "12.9"
    assert maintenance.cuda_major == "12"
    assert maintenance.cuda_variant == "cu12"
    assert config.python_supported == ("3.11", "3.14t")


@pytest.mark.agent_authored(model="gpt-6")
def test_cli_supplies_matrix_and_release_role_records(legacy_config, capsys):
    config_args = ["--config", str(legacy_config)]
    assert main(config_args) == 0
    full = json.loads(capsys.readouterr().out)
    assert full["python"]["supported"] == ["3.11", "3.14t"]
    assert main([*config_args, "--package-roots"]) == 0
    assert json.loads(capsys.readouterr().out) == full["package_roots"]
    assert main([*config_args, "--release-status", "maintenance"]) == 0
    assert json.loads(capsys.readouterr().out) == {
        "package_root": "cuda_bindings_12",
        "toolkit_version": "12.9.1",
        "release_status": "maintenance",
        "ctk_target": "12.9",
        "cuda_major": "12",
        "cuda_variant": "cu12",
    }


@pytest.fixture
def pixi_checkout(tmp_path, monkeypatch, legacy_config):
    for package_root, variant, pin in (
        ("cuda_bindings_12", "cu12", "12.*"),
        ("cuda_bindings_13", "cu13", "13.4.*"),
        ("cuda_core", "cu13", "13.4.*"),
    ):
        path = tmp_path / package_root / "pixi.toml"
        path.parent.mkdir()
        path.write_text(
            f'[workspace.build-variants]\ncuda-version = ["{pin}"]\n'
            f'[feature.{variant}.dependencies]\ncuda-version = "{pin}"\n'
        )
    monkeypatch.setattr(check_pixi_cuda_version, "ROOT", tmp_path)
    monkeypatch.setattr(check_pixi_cuda_version, "VERSIONS_FILE_PATH", legacy_config)
    return tmp_path


@pytest.mark.agent_authored(model="gpt-6")
def test_pixi_check_covers_both_bindings_and_current_core(pixi_checkout, capsys):
    assert check_pixi_cuda_version.main() == 0
    output = capsys.readouterr().out
    assert "cuda_bindings_13/pixi.toml" in output
    assert "cuda_bindings_12/pixi.toml" in output
    assert "cuda_core/pixi.toml" in output


@pytest.mark.agent_authored(model="gpt-6")
def test_pixi_check_reports_core_pin_drift(pixi_checkout, capsys):
    path = pixi_checkout / "cuda_core" / "pixi.toml"
    path.write_text(path.read_text().replace('cuda-version = "13.4.*"', 'cuda-version = "13.3.*"'))
    assert check_pixi_cuda_version.main() == 1
    assert "cuda_core/pixi.toml: feature.cu13.dependencies.cuda-version='13.3.*'" in capsys.readouterr().err


@pytest.mark.parametrize(
    ("package_root", "old_pin", "new_pin"),
    [("cuda_bindings_12", "12.*", "11.*"), ("cuda_bindings_13", "13.4.*", "13.*")],
)
@pytest.mark.agent_authored(model="gpt-6")
def test_pixi_check_rejects_wrong_maintenance_major_and_broad_current_pin(
    pixi_checkout, package_root, old_pin, new_pin, capsys
):
    path = pixi_checkout / package_root / "pixi.toml"
    path.write_text(path.read_text().replace(old_pin, new_pin))
    assert check_pixi_cuda_version.main() == 1
    assert f"{package_root}/pixi.toml" in capsys.readouterr().err


@pytest.mark.agent_authored(model="gpt-6")
def test_cuda12_source_changes_select_bindings_consumers():
    plan = compute_workplan(
        ["cuda_bindings_12/cuda/bindings/_internal/utils.pyx"], merge_base="base", baseline_run_id="123"
    )
    assert {name for name, decision in plan["modules"].items() if decision["needs_build"]} == {
        "bindings",
        "core",
        "python",
    }


@pytest.mark.agent_authored(model="gpt-6")
def test_cuda12_milestone_does_not_change_current_package_versions(tmp_path, monkeypatch):
    for package_root, filename in (("cuda_bindings_13", "pyproject.toml"), ("cuda_python", "setup.py")):
        destination = tmp_path / package_root / filename
        destination.parent.mkdir()
        shutil.copyfile(ROOT / package_root / filename, destination)

    def git(*args):
        return subprocess.run(  # noqa: S603
            ["git", "-c", "user.name=CI test", "-c", "user.email=ci-test@example.invalid", *args],  # noqa: S607
            cwd=tmp_path,
            check=True,
            capture_output=True,
            text=True,
        )

    git("init", "--quiet")
    git("add", ".")
    git("commit", "--quiet", "-m", "Current CUDA 13 release")
    git("tag", "v13.4.3")
    (tmp_path / "maintenance-change").touch()
    git("add", ".")
    git("commit", "--quiet", "-m", "CUDA 12 maintenance milestone")
    git("tag", "v12.9.11a0")

    with (tmp_path / "cuda_bindings_13" / "pyproject.toml").open("rb") as stream:
        scm = tomllib.load(stream)["tool"]["setuptools_scm"]
    bindings_version = get_version(root=tmp_path, git_describe_command=scm["git_describe_command"])
    metadata = {}
    monkeypatch.setattr(setuptools, "setup", lambda **kwargs: metadata.update(kwargs))
    runpy.run_path(str(tmp_path / "cuda_python" / "setup.py"))
    assert Version(bindings_version).release[:2] == (13, 4)
    assert metadata["version"] == bindings_version
