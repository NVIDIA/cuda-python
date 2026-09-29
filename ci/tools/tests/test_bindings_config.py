# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import io
import json
from pathlib import Path

import pytest
import yaml
from packaging.version import Version

from ci.tools.bindings_config import (
    BindingsConfigError,
    load_config,
    main,
    resolve_release_bindings_package,
    validate_config,
)


def _registry(
    maintenance_root: str = "cuda_bindings_12",
    current_root: str = "cuda_bindings",
) -> dict[str, object]:
    return {
        "schema_version": 2,
        "cuda": {
            "bindings": {
                "package_roots": {
                    maintenance_root: {"toolkit_version": "12.9.1", "release_status": "maintenance"},
                    current_root: {"toolkit_version": "13.4.1", "release_status": "current"},
                }
            }
        },
    }


def _write_config(root: Path, data: object, filename: str = "versions.yml") -> None:
    path = root / "ci" / filename
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.suffix == ".json":
        path.write_text(json.dumps(data), encoding="utf-8")
    else:
        path.write_text(yaml.safe_dump(data, sort_keys=False), encoding="utf-8")


def _write_legacy_package(root: Path, *, tag_regex: str | None = None, scm: bool = True) -> None:
    path = root / "cuda_bindings" / "pyproject.toml"
    path.parent.mkdir(parents=True, exist_ok=True)
    if scm:
        text = "[tool.setuptools_scm]\n"
        if tag_regex is not None:
            text += f"tag_regex = '{tag_regex}'\n"
    else:
        text = '[project]\nname = "cuda-bindings"\n'
    path.write_text(text, encoding="utf-8")


@pytest.mark.agent_authored(model="gpt-6-sol")
def test_live_registry_maps_release_statuses_to_package_roots():
    config = load_config()

    assert config.schema_version == 2
    assert [package.package_root for package in config.package_roots] == ["cuda_bindings_12", "cuda_bindings"]
    assert config.package_for_release_status("maintenance").ctk_target == "12.9"
    current = config.package_for_release_status("current")
    assert current.package_root == "cuda_bindings"
    assert current.cuda_major == "13"
    assert current.cuda_variant == "cu13"
    assert json.loads(config.to_json()) == config.to_dict()
    assert "tag_regex" not in current.to_dict()


@pytest.mark.parametrize(
    ("tag", "package_root", "version"),
    (
        ("v12.9.8", "cuda_bindings_12", "12.9.8"),
        ("v12.9.8a2", "cuda_bindings_12", "12.9.8a2"),
        ("v12.9.8rc2", "cuda_bindings_12", "12.9.8rc2"),
        ("v12.9.8.post1", "cuda_bindings_12", "12.9.8.post1"),
        ("v13.4.2b1", "cuda_bindings", "13.4.2b1"),
        ("v13.4.2rc1.dev2", "cuda_bindings", "13.4.2rc1.dev2"),
        ("v13.4.2.dev3", "cuda_bindings", "13.4.2.dev3"),
    ),
)
@pytest.mark.agent_authored(model="gpt-6-sol")
def test_canonical_bindings_tags_route_to_their_release_line(tag, package_root, version):
    config = validate_config(_registry())

    package = config.match_tag(tag)
    assert package is not None
    assert package.package_root == package_root
    assert package.version_from_tag(tag) == Version(version)


@pytest.mark.parametrize(
    "tag",
    (
        "12.9.8",
        "v12.9.08",
        "v12.9.8rc01",
        "v13.4.01",
        "v13.4.2.post01",
        "v13.4.2+local",
        "v13.5.0",
        "v14.0.0",
        "cuda-core-v1.2.0",
    ),
)
@pytest.mark.agent_authored(model="gpt-6-sol")
def test_registry_does_not_route_invalid_or_unconfigured_bindings_tags(tag):
    assert validate_config(_registry()).match_tag(tag) is None


@pytest.mark.parametrize(
    ("change", "message"),
    (
        (("schema_version", 3), "schema_version must be 2"),
        (("maintenance_root", "../outside"), "normalized repository-relative POSIX path"),
        (("maintenance_toolkit", "12.9"), "toolkit_version has invalid format"),
        (("current_status", "maintenance"), "exactly one current and one maintenance"),
        (("current_toolkit", "12.8.1"), "cuda_major values must be unique"),
    ),
)
@pytest.mark.agent_authored(model="gpt-6-sol")
def test_registry_validation_rejects_invalid_schema_and_release_layout(change, message):
    field, value = change
    data = _registry(maintenance_root=value) if field == "maintenance_root" else _registry()
    if field == "schema_version":
        data["schema_version"] = value
    elif field == "maintenance_toolkit":
        data["cuda"]["bindings"]["package_roots"]["cuda_bindings_12"]["toolkit_version"] = value
    elif field == "current_status":
        data["cuda"]["bindings"]["package_roots"]["cuda_bindings"]["release_status"] = value
    elif field == "current_toolkit":
        data["cuda"]["bindings"]["package_roots"]["cuda_bindings"]["toolkit_version"] = value

    with pytest.raises(BindingsConfigError, match=message):
        validate_config(data)


@pytest.mark.agent_authored(model="gpt-6-sol")
def test_registry_rejects_unknown_package_fields():
    data = _registry()
    data["cuda"]["bindings"]["package_roots"]["cuda_bindings"]["tag_regex"] = ".*"

    with pytest.raises(BindingsConfigError, match="must contain exactly"):
        validate_config(data)


@pytest.mark.agent_authored(model="gpt-6-sol")
def test_config_loader_reports_invalid_yaml(tmp_path):
    path = tmp_path / "versions.yml"
    path.write_text("cuda: [unterminated", encoding="utf-8")

    with pytest.raises(BindingsConfigError, match="could not read"):
        load_config(path)


@pytest.mark.agent_authored(model="gpt-6-sol")
def test_cli_emits_registry_and_selected_package_as_json(capsys):
    assert main([]) == 0
    registry = json.loads(capsys.readouterr().out)
    assert registry == load_config().to_dict()

    assert main(["--package-roots"]) == 0
    packages = json.loads(capsys.readouterr().out)
    assert packages == registry["package_roots"]

    assert main(["--release-status", "current"]) == 0
    current = json.loads(capsys.readouterr().out)
    assert current["package_root"] == "cuda_bindings"


@pytest.mark.agent_authored(model="gpt-6-sol")
def test_cli_writes_selected_package_to_github_env(tmp_path, capsys, monkeypatch):
    output = tmp_path / "github-env"
    package = load_config().package_for_release_status("current").to_dict()
    monkeypatch.setattr("sys.stdin", io.StringIO(json.dumps(package)))

    assert main(["write-github-env", str(output)]) == 0

    assert capsys.readouterr().out == ""
    assert output.read_text(encoding="utf-8").splitlines() == [
        f"BUILD_CTK_VER={package['toolkit_version']}",
        "BINDINGS_PACKAGE_ROOT=cuda_bindings",
        "BINDINGS_REGISTRY_ORIGIN=tag",
    ]


@pytest.mark.agent_authored(model="gpt-6-sol")
def test_cli_rejects_nonobject_package_json(tmp_path, capsys, monkeypatch):
    monkeypatch.setattr("sys.stdin", io.StringIO("[]"))

    with pytest.raises(SystemExit, match="2"):
        main(["write-github-env", str(tmp_path / "github-env")])

    assert "stdin for write-github-env must contain a JSON object" in capsys.readouterr().err


@pytest.mark.parametrize(
    ("tag", "package_root", "toolkit_version", "release_version"),
    (
        ("v12.9.8.post1", "tag-maintenance", "12.9.1", "12.9.8.post1"),
        ("v13.4.2rc1", "tag-current", "13.4.1", "13.4.2rc1"),
    ),
)
@pytest.mark.agent_authored(model="gpt-6-sol")
def test_bindings_release_uses_registry_from_its_tag_tree(
    tmp_path, tag, package_root, toolkit_version, release_version
):
    release_root = tmp_path / "release"
    _write_config(release_root, _registry("tag-maintenance", "tag-current"))
    control_root = tmp_path / "control"
    _write_config(control_root, _registry("control-maintenance", "control-current"))

    resolved = resolve_release_bindings_package(tag, release_root, control_root / "ci" / "versions.yml")

    assert resolved == {
        "package_root": package_root,
        "toolkit_version": toolkit_version,
        "release_version": release_version,
        "release_registry_origin": "tag",
    }


@pytest.mark.parametrize("tag", ("cuda-core-v1.2.0", "cuda-pathfinder-v1.1.0"))
@pytest.mark.agent_authored(model="gpt-6-sol")
def test_component_release_uses_current_bindings_from_its_tag_tree(tmp_path, tag):
    release_root = tmp_path / "release"
    _write_config(release_root, _registry("tag-maintenance", "tag-current"))

    resolved = resolve_release_bindings_package(tag, release_root, tmp_path / "unused-control.yml")

    assert resolved == {
        "package_root": "tag-current",
        "toolkit_version": "13.4.1",
        "release_registry_origin": "tag",
    }


@pytest.mark.parametrize("filename", ("versions.yml", "versions.json"))
@pytest.mark.agent_authored(model="gpt-6-sol")
def test_legacy_release_uses_its_tag_tree_toolkit_pin(tmp_path, filename):
    release_root = tmp_path / "release"
    _write_config(release_root, {"cuda": {"build": {"version": "12.9.1"}}}, filename)
    _write_legacy_package(release_root)

    resolved = resolve_release_bindings_package("v12.9.8", release_root, tmp_path / "unused-control.yml")

    assert resolved == {
        "package_root": "cuda_bindings",
        "toolkit_version": "12.9.1",
        "release_version": "12.9.8",
        "release_registry_origin": "control",
    }


@pytest.mark.agent_authored(model="gpt-6-sol")
def test_legacy_tree_without_setuptools_scm_parses_canonical_tag(tmp_path):
    release_root = tmp_path / "release"
    _write_config(release_root, {"cuda": {"build": {"version": "13.0.2"}}}, "versions.json")
    _write_legacy_package(release_root, scm=False)

    resolved = resolve_release_bindings_package("v13.0.3rc1", release_root, tmp_path / "unused-control.yml")

    assert resolved["release_version"] == "13.0.3rc1"
    assert resolved["toolkit_version"] == "13.0.2"


@pytest.mark.agent_authored(model="gpt-6-sol")
def test_historical_custom_regex_retains_its_tag_parsing(tmp_path):
    release_root = tmp_path / "release"
    _write_config(release_root, {"cuda": {"build": {"version": "13.1.0"}}})
    _write_legacy_package(release_root, tag_regex=r"^(?P<version>v\d+\.\d+\.\d+)")

    resolved = resolve_release_bindings_package("v13.2.0rc1", release_root, tmp_path / "unused-control.yml")

    assert resolved["release_version"] == "13.2.0"
    assert resolved["toolkit_version"] == "13.1.0"


@pytest.mark.agent_authored(model="gpt-6-sol")
def test_legacy_tree_without_toolkit_pin_uses_control_registry(tmp_path):
    release_root = tmp_path / "release"
    _write_legacy_package(release_root)
    control_root = tmp_path / "control"
    _write_config(control_root, _registry())

    resolved = resolve_release_bindings_package("v12.9.8", release_root, control_root / "ci" / "versions.yml")

    assert resolved["toolkit_version"] == "12.9.1"
    assert resolved["release_registry_origin"] == "control"


@pytest.mark.agent_authored(model="gpt-6-sol")
def test_legacy_component_release_uses_tagged_toolkit_dependency(tmp_path):
    release_root = tmp_path / "release"
    _write_config(release_root, {"cuda": {"build": {"version": "13.2.1"}}}, "versions.json")
    (release_root / "cuda_bindings").mkdir()

    resolved = resolve_release_bindings_package("cuda-core-v1.0.0", release_root, tmp_path / "unused-control.yml")

    assert resolved == {
        "package_root": "cuda_bindings",
        "toolkit_version": "13.2.1",
        "release_registry_origin": "control",
    }


@pytest.mark.agent_authored(model="gpt-6-sol")
def test_invalid_modern_tag_tree_does_not_fall_back_to_control_registry(tmp_path):
    release_root = tmp_path / "release"
    _write_config(release_root, {"schema_version": 2, "cuda": {}})
    control_root = tmp_path / "control"
    _write_config(control_root, _registry())

    with pytest.raises(BindingsConfigError, match="invalid schema-2 tagged config"):
        resolve_release_bindings_package("v13.4.2", release_root, control_root / "ci" / "versions.yml")


@pytest.mark.parametrize(
    ("tag", "message"),
    (
        ("v13.5.0", "no CUDA bindings package root"),
        ("v14.0.0", "no CUDA bindings package root"),
        ("v13.4.02", "unsupported release tag"),
    ),
)
@pytest.mark.agent_authored(model="gpt-6-sol")
def test_modern_tag_tree_rejects_unknown_or_malformed_bindings_release(tmp_path, tag, message):
    release_root = tmp_path / "release"
    _write_config(release_root, _registry())

    with pytest.raises(BindingsConfigError, match=message):
        resolve_release_bindings_package(tag, release_root, tmp_path / "unused-control.yml")


@pytest.mark.agent_authored(model="gpt-6-sol")
def test_legacy_release_requires_package_metadata(tmp_path):
    release_root = tmp_path / "release"
    _write_config(release_root, {"cuda": {"build": {"version": "12.9.1"}}})
    (release_root / "cuda_bindings").mkdir()

    with pytest.raises(BindingsConfigError, match=r"could not inspect .*cuda_bindings/pyproject\.toml"):
        resolve_release_bindings_package("v12.9.8", release_root, tmp_path / "unused-control.yml")


@pytest.mark.agent_authored(model="gpt-6-sol")
def test_legacy_release_without_toolkit_pin_fails_for_unknown_family(tmp_path):
    release_root = tmp_path / "release"
    _write_legacy_package(release_root)
    control_root = tmp_path / "control"
    _write_config(control_root, _registry())

    with pytest.raises(BindingsConfigError, match="exactly one toolkit pin for legacy CUDA 11.8; found 0"):
        resolve_release_bindings_package("v11.8.0", release_root, control_root / "ci" / "versions.yml")
