# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import json
import shutil
from pathlib import Path

import pytest
import yaml

from ci.tools.bindings_config import check_package_metadata, load_config, main, parse_config


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
            f'"v{package.ctk_target}.*"]\n',
            encoding="utf-8",
        )
    meta = root / "cuda_python" / "pyproject.toml"
    meta.parent.mkdir()
    meta.write_text(
        '[tool.cuda-python.scm-describe-match-by-major]\n"12" = "v12.9.*"\n"13" = "v13.4.*"\n',
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
def test_metadata_check_catches_selectors_drifting_from_the_registered_line(tmp_path, changed_paths):
    config = _write_package_metadata(tmp_path)
    check_package_metadata(config, tmp_path)
    for relative in changed_paths:
        path = tmp_path / relative
        path.write_text(path.read_text().replace("v13.4.*", "v13.5.*"), encoding="utf-8")

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
    registry["backport_branch"] = "12.9.x"
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
