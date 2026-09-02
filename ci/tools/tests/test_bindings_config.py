# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import copy
import json
from pathlib import Path

import pytest
import yaml

from ci.tools.bindings_config import (
    BindingsConfigError,
    load_config,
    main,
    resolve_release_bindings_line,
    validate_config,
)


def line(source_dir, toolkit_version):
    return {
        "source_dir": source_dir,
        "toolkit_version": toolkit_version,
    }


def valid_config():
    return {
        "schema_version": 2,
        "cuda": {
            "bindings": {
                "lines": {
                    "released-12": line("cuda_bindings_12", "12.9.1"),
                    "released-13": line("cuda_bindings", "13.3.0"),
                },
                "roles": {"current": "released-13", "maintenance": "released-12"},
            }
        },
    }


def write_scm_config(root: Path, source_dir: str, tag_regex: str) -> None:
    path = root / source_dir / "pyproject.toml"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(f"[tool.setuptools_scm]\ntag_regex = '{tag_regex}'\n", encoding="utf-8")


def write_yaml(path: Path, data: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(yaml.safe_dump(data, sort_keys=False), encoding="utf-8")


@pytest.mark.agent_authored(model="gpt-5.6")
def test_live_registry_has_ordered_lines_and_roles():
    config = load_config()

    assert [line.line_id for line in config.lines] == ["released-12", "released-13"]
    assert config.line_for_role("current").line_id == "released-13"
    assert config.line_for_role("maintenance").line_id == "released-12"
    assert config.get_line("released-12").source_dir == "cuda_bindings_12"
    assert config.get_line("released-12").ctk_target == "12.9"
    assert config.get_line("released-12").tag_regex.startswith("^(?P<version>v12")
    assert config.get_line("released-12").cuda_major == "12"
    assert config.get_line("released-12").cuda_variant == "cu12"
    normalized = config.to_dict()
    assert normalized["roles"] == {"current": "released-13", "maintenance": "released-12"}
    assert json.loads(config.to_json()) == normalized


@pytest.mark.agent_authored(model="gpt-5.6")
def test_tag_matching_uses_each_source_lines_scm_regex():
    config = validate_config(valid_config())
    assert config.match_tag("v12.9.8").line_id == "released-12"
    assert config.match_tag("v12.9.8a1") is None
    assert config.match_tag("v13.3.0b1").line_id == "released-13"
    assert config.match_tag("v13.3.0rc1").line_id == "released-13"
    assert config.match_tag("v13.3.0.dev1").line_id == "released-13"
    assert config.match_tag("v13.3.2.post1").line_id == "released-13"


@pytest.mark.agent_authored(model="gpt-5.6")
def test_public_registry_requires_distinct_cuda_abi_majors(tmp_path):
    raw = valid_config()
    bindings = raw["cuda"]["bindings"]
    bindings["lines"] = {
        "released-11-7": line("cuda_bindings_11_7", "11.7.1"),
        "released-11-8": line("cuda_bindings_11_8", "11.8.0"),
    }
    bindings["roles"] = {"current": "released-11-8", "maintenance": "released-11-7"}
    write_scm_config(tmp_path, "cuda_bindings_11_7", r"^(?P<version>v11\.7\.\d+)$")
    write_scm_config(tmp_path, "cuda_bindings_11_8", r"^(?P<version>v11\.8\.\d+)$")

    with pytest.raises(BindingsConfigError, match="cuda_major values must be unique"):
        validate_config(raw, tmp_path)


@pytest.mark.parametrize(
    ("field", "value", "message"),
    [
        ("toolkit_version", "12.9", "toolkit_version has invalid format"),
        ("source_dir", "../cuda_bindings_12", "normalized repository-relative POSIX path"),
        ("source_dir", "cuda_bindings_12\nINJECTED=value", "source_dir has invalid format"),
    ],
)
@pytest.mark.agent_authored(model="gpt-5.6")
def test_invalid_registry_is_rejected(field, value, message):
    data = copy.deepcopy(valid_config())
    data["cuda"]["bindings"]["lines"]["released-12"][field] = value
    with pytest.raises(BindingsConfigError, match=message):
        validate_config(data)


@pytest.mark.agent_authored(model="gpt-5.6")
def test_load_wraps_yaml_errors(tmp_path):
    path = tmp_path / "versions.yml"
    path.write_text("cuda: [unterminated", encoding="utf-8")

    with pytest.raises(BindingsConfigError, match="could not read"):
        load_config(path)


@pytest.mark.agent_authored(model="gpt-5.6")
def test_cli_emits_full_registry_lines_and_one_role(capsys):
    assert main([]) == 0
    registry = json.loads(capsys.readouterr().out)
    assert registry["roles"]["current"] == "released-13"

    assert main(["--lines"]) == 0
    lines = json.loads(capsys.readouterr().out)
    assert [line["line_id"] for line in lines] == ["released-12", "released-13"]

    assert main(["--role", "current"]) == 0
    current = json.loads(capsys.readouterr().out)
    assert current["line_id"] == "released-13"
    assert current["cuda_variant"] == "cu13"


@pytest.mark.agent_authored(model="gpt-5.6")
def test_cli_writes_selected_line_directly_to_github_env(tmp_path, capsys):
    output = tmp_path / "github-env"

    assert main(["--role", "current", "--github-env", str(output)]) == 0

    assert capsys.readouterr().out == ""
    assert output.read_text(encoding="utf-8").splitlines() == [
        "BUILD_CTK_VER=13.3.0",
        "BINDINGS_COMPONENT_DIR=cuda_bindings",
        "BINDINGS_REGISTRY_ORIGIN=tag",
    ]


@pytest.mark.agent_authored(model="gpt-5.6")
def test_list_valued_role_is_rejected():
    data = valid_config()
    data["cuda"]["bindings"]["roles"]["maintenance"] = ["released-12"]

    with pytest.raises(BindingsConfigError, match="maintenance must be a non-empty, trimmed string"):
        validate_config(data)


@pytest.mark.parametrize(
    ("roles", "message"),
    (
        ({"current": "released-13"}, "exactly current and maintenance"),
        (
            {"current": "released-13", "maintenance": "released-13"},
            "select each configured line exactly once",
        ),
    ),
)
@pytest.mark.agent_authored(model="gpt-5.6")
def test_public_roles_must_cover_both_lines_once(roles, message):
    data = valid_config()
    data["cuda"]["bindings"]["roles"] = roles

    with pytest.raises(BindingsConfigError, match=message):
        validate_config(data)


@pytest.mark.agent_authored(model="gpt-5.6")
def test_public_roles_must_select_two_distinct_configured_lines():
    data = valid_config()
    data["cuda"]["bindings"]["lines"] = {
        "released-13": line("cuda_bindings", "13.3.0"),
    }
    data["cuda"]["bindings"]["roles"] = {
        "current": "released-13",
        "maintenance": "released-13",
    }

    with pytest.raises(BindingsConfigError, match="select each configured line exactly once"):
        validate_config(data)


@pytest.mark.agent_authored(model="gpt-5.6")
def test_scm_regex_must_match_the_lines_toolkit_release(tmp_path):
    data = valid_config()
    write_scm_config(tmp_path, "cuda_bindings_12", r"^(?P<version>v13\.\d+\.\d+)$")
    write_scm_config(tmp_path, "cuda_bindings", r"^(?P<version>v13\.\d+\.\d+)$")

    with pytest.raises(BindingsConfigError, match="must match its configured toolkit release tag"):
        validate_config(data, tmp_path)


def release_registry(*, current_dir: str = "cuda_bindings", maintenance_dir: str = "cuda_bindings_12"):
    data = valid_config()
    data["cuda"]["bindings"]["lines"]["released-12"]["source_dir"] = maintenance_dir
    data["cuda"]["bindings"]["lines"]["released-13"]["source_dir"] = current_dir
    return data


def write_release_scm_configs(root: Path, *, current_dir: str, maintenance_dir: str) -> None:
    write_scm_config(root, current_dir, r"^(?P<version>v13\.\d+\.\d+(?:rc\d+)?)$")
    write_scm_config(root, maintenance_dir, r"^(?P<version>v12\.9\.\d+(?:\.post\d+)?)$")


@pytest.mark.parametrize(
    ("release_tag", "expected_line", "expected_dir"),
    (
        ("v13.3.2", "released-13", "tag-current"),
        ("v12.9.8", "released-12", "tag-maintenance"),
    ),
)
@pytest.mark.agent_authored(model="gpt-5.6")
def test_release_cli_uses_authoritative_tag_tree(tmp_path, capsys, release_tag, expected_line, expected_dir):
    release_root = tmp_path / "release"
    tagged_config = release_root / "ci" / "versions.yml"
    control_config = tmp_path / "control" / "ci" / "versions.yml"
    write_yaml(tagged_config, release_registry(current_dir="tag-current", maintenance_dir="tag-maintenance"))
    write_release_scm_configs(release_root, current_dir="tag-current", maintenance_dir="tag-maintenance")
    write_yaml(control_config, release_registry(current_dir="control-current", maintenance_dir="control-maintenance"))

    assert (
        main(
            [
                "--release-tag",
                release_tag,
                "--release-source-root",
                str(release_root),
                "--control-config",
                str(control_config),
            ]
        )
        == 0
    )

    resolved = json.loads(capsys.readouterr().out)
    assert resolved["line_id"] == expected_line
    assert resolved["release_source_dir"] == expected_dir
    assert resolved["release_registry_origin"] == "tag"
    assert resolved["release_version"] == release_tag.removeprefix("v")


@pytest.mark.agent_authored(model="gpt-5.6")
def test_legacy_tag_tree_uses_control_registry_and_legacy_layout(tmp_path):
    release_root = tmp_path / "release"
    write_yaml(release_root / "ci" / "versions.yml", {"cuda": {"build": {"version": "12.9.1"}}})
    (release_root / "cuda_bindings").mkdir(parents=True)
    write_scm_config(release_root, "cuda_bindings", r"^(?P<version>v\d+\.\d+\.\d+)")
    control_root = tmp_path / "control"
    control_config = control_root / "ci" / "versions.yml"
    write_yaml(control_config, release_registry())
    write_release_scm_configs(control_root, current_dir="cuda_bindings", maintenance_dir="cuda_bindings_12")

    resolved = resolve_release_bindings_line("v12.9.8", release_root, control_config)

    assert resolved["line_id"] == "legacy"
    assert resolved["source_dir"] == "cuda_bindings"
    assert resolved["release_source_dir"] == "cuda_bindings"
    assert resolved["release_registry_origin"] == "control"
    assert resolved["release_version"] == "12.9.8"
    assert resolved["toolkit_version"] == "12.9.1"


@pytest.mark.agent_authored(model="gpt-5.6")
def test_legacy_prerelease_preserves_tagged_tree_toolkit_and_scm_semantics(tmp_path):
    release_root = tmp_path / "release"
    write_yaml(release_root / "ci" / "versions.yml", {"cuda": {"build": {"version": "13.1.0"}}})
    write_scm_config(release_root, "cuda_bindings", r"^(?P<version>v\d+\.\d+\.\d+)")

    resolved = resolve_release_bindings_line(
        "v13.2.0rc1",
        release_root,
        tmp_path / "unused-control.yml",
    )

    assert resolved["release_version"] == "13.2.0"
    assert resolved["toolkit_version"] == "13.1.0"
    assert resolved["release_source_dir"] == "cuda_bindings"


@pytest.mark.agent_authored(model="gpt-5.6")
def test_legacy_release_requires_source_scm_metadata(tmp_path):
    release_root = tmp_path / "release"
    write_yaml(release_root / "ci" / "versions.yml", {"cuda": {"build": {"version": "12.9.1"}}})
    (release_root / "cuda_bindings").mkdir(parents=True)

    with pytest.raises(BindingsConfigError, match=r"could not read .*cuda_bindings/pyproject\.toml"):
        resolve_release_bindings_line("v12.9.8", release_root, tmp_path / "unused-control.yml")


@pytest.mark.agent_authored(model="gpt-5.6")
def test_legacy_release_requires_an_authoritative_toolkit_pin(tmp_path):
    release_root = tmp_path / "release"
    write_scm_config(release_root, "cuda_bindings", r"^(?P<version>v\d+\.\d+\.\d+)")
    control_root = tmp_path / "control"
    control_config = control_root / "ci" / "versions.yml"
    write_yaml(control_config, release_registry())
    write_release_scm_configs(control_root, current_dir="cuda_bindings", maintenance_dir="cuda_bindings_12")

    with pytest.raises(BindingsConfigError, match="exactly one toolkit pin for legacy CUDA 11.8; found 0"):
        resolve_release_bindings_line("v11.8.0", release_root, control_config)


@pytest.mark.agent_authored(model="gpt-5.6")
def test_invalid_modern_tag_config_does_not_fall_back(tmp_path):
    release_root = tmp_path / "release"
    control_config = tmp_path / "control" / "ci" / "versions.yml"
    write_yaml(release_root / "ci" / "versions.yml", {"schema_version": 2, "cuda": {}})
    write_yaml(control_config, release_registry())

    with pytest.raises(BindingsConfigError, match="invalid schema-2 tagged config"):
        resolve_release_bindings_line("v13.3.0", release_root, control_config)


@pytest.mark.agent_authored(model="gpt-5.6")
def test_unknown_release_tag_fails_closed(tmp_path):
    release_root = tmp_path / "release"
    write_scm_config(release_root, "cuda_bindings", r"^(?P<version>v13\.\d+\.\d+)$")
    control_root = tmp_path / "control"
    control_config = control_root / "ci" / "versions.yml"
    write_yaml(control_config, release_registry())
    write_release_scm_configs(control_root, current_dir="cuda_bindings", maintenance_dir="cuda_bindings_12")

    with pytest.raises(BindingsConfigError, match="does not match release tag"):
        resolve_release_bindings_line("v14.0.0", release_root, control_config)


@pytest.mark.agent_authored(model="gpt-5.6")
def test_modern_release_tag_must_match_the_configured_toolkit_minor(tmp_path):
    release_root = tmp_path / "release"
    control_config = tmp_path / "control" / "ci" / "versions.yml"
    write_yaml(release_root / "ci" / "versions.yml", release_registry())
    write_release_scm_configs(release_root, current_dir="cuda_bindings", maintenance_dir="cuda_bindings_12")

    with pytest.raises(BindingsConfigError, match="no CUDA bindings line"):
        resolve_release_bindings_line("v13.4.0", release_root, control_config)
