# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import io
import json
import shutil
from pathlib import Path

import pytest
import yaml
from packaging.version import Version

from ci.tools.bindings_config import (
    BindingsConfigError,
    check_core_toolkit_floors,
    load_config,
    main,
    resolve_release_bindings_package,
    validate_config,
)


def _registry(
    maintenance_root: str = "cuda_bindings_12",
    current_root: str = "cuda_bindings_13",
) -> dict[str, object]:
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


def _write_config(root: Path, data: object, filename: str = "versions.yml") -> None:
    path = root / "ci" / filename
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.suffix == ".json":
        path.write_text(json.dumps(data), encoding="utf-8")
    else:
        path.write_text(yaml.safe_dump(data, sort_keys=False), encoding="utf-8")


def _copy_package_metadata(root: Path) -> None:
    repo_root = Path(__file__).resolve().parents[3]
    _write_config(root, _registry())
    for relative in (
        "cuda_bindings_13/pyproject.toml",
        "cuda_bindings_12/pyproject.toml",
        "cuda_bindings_12/pixi.toml",
        "cuda_python/setup.py",
        "cuda_core/pyproject.toml",
        "cuda_core/cuda/core/_bindings_floor.py",
    ):
        destination = root / relative
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(repo_root / relative, destination)


@pytest.mark.parametrize(
    ("tag", "package_root", "version"),
    (
        ("v12.9.8", "cuda_bindings_12", "12.9.8"),
        ("v12.9.8a2", "cuda_bindings_12", "12.9.8a2"),
        ("v12.9.8rc2", "cuda_bindings_12", "12.9.8rc2"),
        ("v12.9.8.post1", "cuda_bindings_12", "12.9.8.post1"),
        ("v13.4.2b1", "cuda_bindings_13", "13.4.2b1"),
        ("v13.4.2rc1.dev2", "cuda_bindings_13", "13.4.2rc1.dev2"),
        ("v13.4.2.dev3", "cuda_bindings_13", "13.4.2.dev3"),
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
        data["cuda"]["bindings"]["package_roots"]["cuda_bindings_13"]["release_status"] = value
    elif field == "current_toolkit":
        data["cuda"]["bindings"]["package_roots"]["cuda_bindings_13"]["toolkit_version"] = value

    with pytest.raises(BindingsConfigError, match=message):
        validate_config(data)


@pytest.mark.parametrize("toolkit_version", ("12.9.1", "12.8.4"))
@pytest.mark.agent_authored(model="gpt-6.1-sol")
def test_backport_branch_agrees_with_the_maintenance_toolkit_line(toolkit_version):
    data = _registry()
    data["cuda"]["bindings"]["package_roots"]["cuda_bindings_12"]["toolkit_version"] = toolkit_version
    data["backport_branch"] = ".".join(toolkit_version.split(".")[:2]) + ".x"

    config = validate_config(data)

    assert data["backport_branch"] == f"{config.package_for_release_status('maintenance').ctk_target}.x"


@pytest.mark.parametrize("branch", ("13.4.x", "12.8.x", "12.9", "", None, True))
@pytest.mark.agent_authored(model="gpt-6.1-sol")
def test_registry_rejects_backport_branches_outside_the_maintenance_line(branch):
    data = _registry()
    data["backport_branch"] = branch

    with pytest.raises(BindingsConfigError, match="backport_branch must match maintenance CUDA line '12.9.x'"):
        validate_config(data)


@pytest.mark.agent_authored(model="gpt-6.1-sol")
def test_registry_accepts_no_legacy_backport_branch():
    data = _registry()

    config = validate_config(data)

    assert config.package_for_release_status("maintenance").ctk_target == "12.9"


@pytest.mark.agent_authored(model="gpt-6.1-sol")
def test_registry_rejects_unknown_root_fields():
    data = _registry()
    data["backport_branch"] = "12.9.x"
    data["unexpected"] = True

    with pytest.raises(BindingsConfigError, match="must contain exactly"):
        validate_config(data)


@pytest.mark.agent_authored(model="gpt-6-sol")
def test_registry_rejects_unknown_package_fields():
    data = _registry()
    data["cuda"]["bindings"]["package_roots"]["cuda_bindings_13"]["tag_regex"] = ".*"

    with pytest.raises(BindingsConfigError, match="must contain exactly"):
        validate_config(data)


@pytest.mark.agent_authored(model="gpt-6-sol")
def test_config_loader_reports_invalid_yaml(tmp_path):
    path = tmp_path / "versions.yml"
    path.write_text("cuda: [unterminated", encoding="utf-8")

    with pytest.raises(BindingsConfigError, match="could not read"):
        load_config(path)


@pytest.mark.parametrize(
    ("path", "before", "after", "message"),
    (
        ("ci/versions.yml", "13.4.1", "13.5.0", "must select registered CUDA 13.5"),
        ("cuda_bindings_13/pyproject.toml", "v13.4.*", "v13.*", "must select registered CUDA 13.4"),
        (
            "cuda_bindings_13/pyproject.toml",
            "[tool.setuptools_scm]",
            '[tool.setuptools_scm]\ntag_regex = "custom"',
            "must use setuptools-scm's default tag parser",
        ),
        (
            "cuda_bindings_13/pyproject.toml",
            "[tool.setuptools_scm]",
            '[tool.setuptools_scm]\nfallback_version = "13.4.0.dev0"',
            "bindings root must not declare fallback_version",
        ),
        ("cuda_python/setup.py", "v13.4.*", "v13.5.*", "SCM_DESCRIBE_MATCH_BY_MAJOR must match"),
    ),
)
@pytest.mark.agent_authored(model="gpt-6")
def test_metadata_check_rejects_independent_package_drift(tmp_path, capsys, path, before, after, message):
    _copy_package_metadata(tmp_path)
    changed = tmp_path / path
    content = changed.read_text(encoding="utf-8")
    assert before in content
    changed.write_text(content.replace(before, after), encoding="utf-8")

    with pytest.raises(SystemExit, match="2"):
        main(["--config", str(tmp_path / "ci" / "versions.yml"), "--check-package-metadata"])

    assert message in capsys.readouterr().err


@pytest.mark.parametrize(
    ("package_root", "role", "before", "after"),
    (
        ("cuda_bindings_12", "maintenance", "v12.9.*", "v12.9.[1-9]*"),
        ("cuda_bindings_13", "current", "v13.4.*", "v13.4.[1-9]*"),
    ),
)
@pytest.mark.agent_authored(model="gpt-6")
def test_metadata_check_rejects_selectors_changed_together(tmp_path, capsys, package_root, role, before, after):
    _copy_package_metadata(tmp_path)
    args = ["--config", str(tmp_path / "ci" / "versions.yml"), "--check-package-metadata"]
    assert main(args) == 0

    for relative in (f"{package_root}/pyproject.toml", "cuda_python/setup.py"):
        changed = tmp_path / relative
        content = changed.read_text(encoding="utf-8")
        assert before in content
        changed.write_text(content.replace(before, after), encoding="utf-8")

    with pytest.raises(SystemExit, match="2"):
        main(args)

    assert f"using {before!r} for the {role} root" in capsys.readouterr().err


@pytest.mark.agent_authored(model="gpt-6-sol")
def test_cli_writes_selected_package_to_github_env(tmp_path, capsys, monkeypatch):
    output = tmp_path / "github-env"
    package = load_config().package_for_release_status("current").to_dict()
    monkeypatch.setattr("sys.stdin", io.StringIO(json.dumps(package)))

    assert main(["write-github-env", str(output)]) == 0

    assert capsys.readouterr().out == ""
    assert output.read_text(encoding="utf-8").splitlines() == [
        f"BUILD_CTK_VER={package['toolkit_version']}",
        "BINDINGS_PACKAGE_ROOT=cuda_bindings_13",
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

    resolved = resolve_release_bindings_package(tag, release_root)

    assert resolved == {
        "package_root": package_root,
        "toolkit_version": toolkit_version,
        "release_version": release_version,
    }


@pytest.mark.parametrize("tag", ("cuda-core-v1.2.0", "cuda-pathfinder-v1.1.0"))
@pytest.mark.agent_authored(model="gpt-6-sol")
def test_component_release_uses_current_bindings_from_its_tag_tree(tmp_path, tag):
    release_root = tmp_path / "release"
    _write_config(release_root, _registry("tag-maintenance", "tag-current"))

    resolved = resolve_release_bindings_package(tag, release_root)

    assert resolved == {
        "package_root": "tag-current",
        "toolkit_version": "13.4.1",
    }


@pytest.mark.parametrize("tag", ("v12.9.8", "cuda-core-v1.2.0", "cuda-pathfinder-v1.1.0"))
@pytest.mark.parametrize(
    "layout", ("missing", "legacy-yaml", "legacy-json", "invalid", "wrong-schema", "malformed-yaml")
)
@pytest.mark.agent_authored(model="gpt-6")
def test_release_rejects_missing_or_invalid_tagged_registry(tmp_path, capsys, tag, layout):
    release_root = tmp_path / "release"
    release_root.mkdir()
    if layout.startswith("legacy-"):
        filename = "versions.json" if layout == "legacy-json" else "versions.yml"
        _write_config(release_root, {"cuda": {"build": {"version": "12.9.1"}}}, filename)
    elif layout == "invalid":
        _write_config(release_root, {"schema_version": 2, "cuda": {}})
    elif layout == "malformed-yaml":
        _write_config(release_root, _registry())
        (release_root / "ci" / "versions.yml").write_text("cuda: [\n", encoding="utf-8")
    elif layout == "wrong-schema":
        registry = _registry()
        registry["schema_version"] = 3
        _write_config(release_root, registry)
    control_root = tmp_path / "control"
    _write_config(control_root, _registry())

    with pytest.raises(SystemExit, match="2"):
        main(
            [
                "--config",
                str(control_root / "ci" / "versions.yml"),
                "--release-tag",
                tag,
                "--release-source-root",
                str(release_root),
            ]
        )

    error = capsys.readouterr().err
    assert "release source requires a valid schema-2 registry" in error
    assert "use compatible historical release tooling" in error


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
        resolve_release_bindings_package(tag, release_root)


@pytest.mark.parametrize("versions", (None, [], ["3.11", "3.11"], ["3.15-dev"], [315], [" 3.11"]))
@pytest.mark.agent_authored(model="gpt-6")
def test_registry_rejects_invalid_python_matrices(versions):
    data = _registry()
    data["python"]["supported"] = versions
    with pytest.raises(BindingsConfigError, match="python.supported must contain unique Python 3 versions"):
        validate_config(data)


@pytest.mark.agent_authored(model="gpt-6")
def test_registry_preserves_the_supported_python_matrix():
    config = validate_config(_registry())
    assert config.python_supported == ("3.11", "3.14t", "3.15")
    assert config.to_dict()["python"] == {"supported": ["3.11", "3.14t", "3.15"]}


@pytest.mark.agent_authored(model="gpt-6")
def test_registered_toolkits_cover_core_bindings_floors(tmp_path):
    _copy_package_metadata(tmp_path)
    check_core_toolkit_floors(validate_config(_registry()), tmp_path)
    data = _registry()
    data["cuda"]["bindings"]["package_roots"]["cuda_bindings_13"]["toolkit_version"] = "13.5.0"
    check_core_toolkit_floors(validate_config(data), tmp_path)
    data["cuda"]["bindings"]["package_roots"]["cuda_bindings_13"]["toolkit_version"] = "13.3.0"
    with pytest.raises(BindingsConfigError, match="is below the CUDA 13 bindings floor 13.4.1"):
        check_core_toolkit_floors(validate_config(data), tmp_path)


@pytest.mark.agent_authored(model="gpt-6")
def test_registered_cuda_majors_must_match_core_extras(tmp_path):
    _copy_package_metadata(tmp_path)
    data = _registry()
    data["cuda"]["bindings"]["package_roots"]["cuda_bindings_13"]["toolkit_version"] = "14.1.0"
    with pytest.raises(BindingsConfigError, match="registered CUDA majors must match"):
        check_core_toolkit_floors(validate_config(data), tmp_path)
