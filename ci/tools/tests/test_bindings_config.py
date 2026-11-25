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

from ci.tools import bindings_config, check_pixi_cuda_version
from ci.tools.bindings_config import load_config, main

ROOT = Path(__file__).resolve().parents[3]


@pytest.fixture
def registry_config(tmp_path):
    path = tmp_path / "versions.yml"
    path.write_text(
        json.dumps(
            {
                "schema_version": 2,
                "cuda": {
                    "bindings": {
                        "package_roots": {
                            "cuda_bindings_13": {"toolkit_version": "13.4.2", "release_status": "current"},
                            "cuda_bindings_12": {"toolkit_version": "12.9.1", "release_status": "maintenance"},
                        }
                    }
                },
                "python": {"supported": ["3.11", "3.14t"]},
            }
        )
    )
    return path


@pytest.mark.agent_authored(model="gpt-6")
def test_cli_supplies_matrix_and_release_role_records(registry_config, capsys):
    config_args = ["--config", str(registry_config)]
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


@pytest.mark.agent_authored(model="gpt-6")
def test_cli_uses_artifact_source_registry_instead_of_control_config(tmp_path, monkeypatch, registry_config, capsys):
    monkeypatch.setattr(bindings_config, "DEFAULT_CONFIG", registry_config)
    raw = {
        "schema_version": 2,
        "python": {"supported": ["3.12"]},
        "cuda": {
            "bindings": {
                "package_roots": {
                    "snapshot_bindings_12": {"toolkit_version": "12.9.1", "release_status": "maintenance"},
                    "snapshot_bindings_13": {"toolkit_version": "13.3.2", "release_status": "current"},
                }
            }
        },
    }
    source = tmp_path / "artifact-source" / "ci" / "versions.yml"
    source.parent.mkdir(parents=True)
    source.write_text(json.dumps(raw))

    assert main(["--config", str(source)]) == 0

    config = json.loads(capsys.readouterr().out)
    assert config["python"]["supported"] == ["3.12"]
    assert [(package["package_root"], package["toolkit_version"]) for package in config["package_roots"]] == [
        ("snapshot_bindings_12", "12.9.1"),
        ("snapshot_bindings_13", "13.3.2"),
    ]


@pytest.fixture
def pixi_checkout(tmp_path, monkeypatch, registry_config):
    for package_root, variant, pin in (
        ("cuda_bindings_12", "cu12", "12.*"),
        ("cuda_bindings_13", "cu13", "13.4.*"),
        ("cuda_core", "cu13", "13.4.*"),
    ):
        path = tmp_path / package_root / "pixi.toml"
        path.parent.mkdir()
        variants = ["12.*", pin] if package_root == "cuda_core" else [pin]
        path.write_text(
            f"[workspace.build-variants]\ncuda-version = {json.dumps(variants)}\n"
            f'[feature.{variant}.dependencies]\ncuda-version = "{pin}"\n'
        )
        if package_root == "cuda_core":
            with path.open("a") as stream:
                stream.write('[feature.cu12.dependencies]\ncuda-version = "12.*"\n')
    monkeypatch.setattr(check_pixi_cuda_version, "ROOT", tmp_path)
    monkeypatch.setattr(check_pixi_cuda_version.bindings_config, "load_config", lambda: load_config(registry_config))
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


@pytest.mark.agent_authored(model="gpt-6")
def test_pixi_check_reports_maintenance_core_pin_drift(pixi_checkout, capsys):
    path = pixi_checkout / "cuda_core" / "pixi.toml"
    path.write_text(path.read_text().replace('cuda-version = "12.*"', 'cuda-version = "11.*"'))
    assert check_pixi_cuda_version.main() == 1
    assert "cuda_core/pixi.toml: feature.cu12.dependencies.cuda-version='11.*'" in capsys.readouterr().err


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
def test_cuda12_milestone_does_not_change_current_package_versions(tmp_path, monkeypatch):
    current = load_config(ROOT / "ci" / "versions.yml").package_for_release_status("current")
    for package_root, filename in (
        ("cuda_bindings_13", "pyproject.toml"),
        ("cuda_python", "setup.py"),
        ("cuda_python", "pyproject.toml"),
    ):
        destination = tmp_path / package_root / filename
        destination.parent.mkdir(exist_ok=True)
        shutil.copyfile(ROOT / package_root / filename, destination)

    def git(*args):
        return subprocess.run(  # noqa: S603
            [  # noqa: S607
                "git",
                "-c",
                "commit.gpgsign=false",
                "-c",
                "tag.gpgsign=false",
                "-c",
                "core.hooksPath=/dev/null",
                "-c",
                "user.name=CI test",
                "-c",
                "user.email=ci-test@example.invalid",
                *args,
            ],
            cwd=tmp_path,
            check=True,
            capture_output=True,
            text=True,
        )

    git("init", "--quiet")
    git("add", ".")
    git("commit", "--quiet", "-m", "Current CUDA 13 release")
    git("tag", f"v{current.toolkit_version}")
    (tmp_path / "maintenance-change").touch()
    git("add", ".")
    git("commit", "--quiet", "-m", "CUDA 12 maintenance milestone")
    git("tag", "v12.9.11a0")

    with (tmp_path / "cuda_bindings_13" / "pyproject.toml").open("rb") as stream:
        scm = tomllib.load(stream)["tool"]["setuptools_scm"]
    assert scm["git_describe_command"][-1] == f"v{current.cuda_major}.*"
    bindings_version = get_version(root=tmp_path, git_describe_command=scm["git_describe_command"])
    metadata = {}
    monkeypatch.setattr(setuptools, "setup", lambda **kwargs: metadata.update(kwargs))
    monkeypatch.setenv("CUDA_PYTHON_BUILD_MAJOR", "13")
    runpy.run_path(str(tmp_path / "cuda_python" / "setup.py"))
    assert Version(bindings_version).release[:2] == current.toolkit[:2]
    assert metadata["version"] == bindings_version
