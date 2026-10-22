# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import json
import runpy
import shutil
import subprocess
import tomllib
from pathlib import Path

import pytest
import setuptools
import yaml
from packaging.version import Version
from setuptools_scm import get_version

from ci.tools.bindings_config import check_package_metadata, load_config, main, parse_config

ROOT = Path(__file__).resolve().parents[3]


def _registry(maintenance_root="cuda_bindings_12", current_root="cuda_bindings_13"):
    return {
        "schema_version": 2,
        "python": {"supported": ["3.11", "3.14t", "3.15"]},
        "cuda": {
            "bindings": {
                "package_roots": {
                    maintenance_root: {"toolkit_version": "12.9.1", "release_status": "maintenance"},
                    current_root: {"toolkit_version": "13.4.1", "release_status": "current"},
                }
            }
        },
    }


def _write_config(root, data):
    path = root / "ci" / "versions.yml"
    path.parent.mkdir(parents=True)
    path.write_text(yaml.safe_dump(data), encoding="utf-8")
    return path


def _write_package_metadata(root):
    config = parse_config(_registry())
    for package in config.package_roots:
        path = root / package.package_root / "pyproject.toml"
        path.parent.mkdir(parents=True)
        path.write_text(
            '[tool.setuptools_scm]\ngit_describe_command = ["git", "describe", "--match", '
            f'"v{package.cuda_major}.*"]\n',
            encoding="utf-8",
        )
    meta = root / "cuda_python" / "pyproject.toml"
    meta.parent.mkdir()
    meta.write_text(
        '[tool.cuda-python.scm-describe-match-by-major]\n"12" = "v12.*"\n"13" = "v13.*"\n',
        encoding="utf-8",
    )
    core = root / "cuda_core"
    core.mkdir()
    (core / "pyproject.toml").write_text(
        '[project.optional-dependencies]\ncu12 = ["cuda-bindings>=12.9.0,<13"]\ncu13 = ["cuda-bindings>=13.4.0,<14"]\n',
        encoding="utf-8",
    )
    floor_reader = Path("cuda_core/cuda/core/_bindings_floor.py")
    (root / floor_reader).parent.mkdir(parents=True)
    shutil.copyfile(Path(__file__).resolve().parents[3] / floor_reader, root / floor_reader)
    return config


@pytest.mark.parametrize(
    ("tag", "expected"),
    (("v12.9.8", "cuda_bindings_12"), ("v13.4.2rc1", "cuda_bindings_13"), ("v14.0.0", None)),
)
@pytest.mark.agent_authored(model="gpt-6.1-sol")
def test_release_tags_select_only_the_matching_bindings_line(tag, expected):
    package = parse_config(_registry()).match_tag(tag)
    assert (package.package_root if package else None) == expected


@pytest.mark.parametrize(
    "changed_paths",
    (
        ("cuda_bindings_13/pyproject.toml",),
        ("cuda_python/pyproject.toml",),
        ("cuda_bindings_13/pyproject.toml", "cuda_python/pyproject.toml"),
    ),
)
@pytest.mark.agent_authored(model="gpt-6.1-sol")
def test_metadata_check_catches_selectors_drifting_from_the_registered_major(tmp_path, changed_paths):
    config = _write_package_metadata(tmp_path)
    check_package_metadata(config, tmp_path)
    for relative in changed_paths:
        path = tmp_path / relative
        path.write_text(path.read_text().replace("v13.*", "v14.*"), encoding="utf-8")

    with pytest.raises(ValueError):
        check_package_metadata(config, tmp_path)


@pytest.mark.parametrize(
    ("tag", "root", "toolkit", "version"),
    (
        ("v12.9.8.post1", "tag-maintenance", "12.9.1", "12.9.8.post1"),
        ("v13.4.2rc1", "tag-current", "13.4.1", "13.4.2rc1"),
        ("cuda-core-v1.2.0", "tag-current", "13.4.1", None),
        ("cuda-pathfinder-v1.1.0", "tag-current", "13.4.1", None),
    ),
)
@pytest.mark.agent_authored(model="gpt-6.1-sol")
def test_release_uses_its_tagged_registry_instead_of_the_control_tree(tmp_path, capsys, tag, root, toolkit, version):
    control = _write_config(tmp_path / "control", _registry("control-maintenance", "control-current"))
    source = tmp_path / "release"
    registry = _registry("tag-maintenance", "tag-current")
    _write_config(source, registry)

    assert main(["--config", str(control), "--release-tag", tag, "--release-source-root", str(source)]) == 0

    expected = {"package_root": root, "toolkit_version": toolkit}
    if version is not None:
        expected["release_version"] = version
    assert json.loads(capsys.readouterr().out) == expected


@pytest.mark.agent_authored(model="gpt-6.1-sol")
def test_missing_tagged_registry_does_not_fall_back_to_the_control_tree(tmp_path):
    control = _write_config(tmp_path / "control", _registry())
    source = tmp_path / "release"
    source.mkdir()
    assert load_config(control).match_tag("v12.9.8") is not None

    with pytest.raises(SystemExit):
        main(["--config", str(control), "--release-tag", "v12.9.8", "--release-source-root", str(source)])


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
