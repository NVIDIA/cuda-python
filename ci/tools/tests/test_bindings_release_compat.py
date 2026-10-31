# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import json

import pytest
import yaml

from ci.tools.bindings_config import main, parse_config, resolve_release_bindings_package, write_github_env


def _source(root, *, old=False, schema2=False):
    current_root = "cuda_bindings" if old else "cuda_bindings_13"
    packages = {
        "cuda_bindings_12": {"toolkit_version": "12.9.1", "release_status": "maintenance"},
        current_root: {"toolkit_version": "13.4.2", "release_status": "current"},
    }
    raw = {"cuda": {"build": {"version": "13.4.2"}, "prev_build": {"version": "12.9.1"}}}
    if not old:
        raw["python"] = {"supported": ["3.11", "3.14t"]}
    if schema2:
        raw["schema_version"] = 2
        raw["cuda"] = {"bindings": {"package_roots": packages}}
    config = root / "ci" / "versions.yml"
    config.parent.mkdir(parents=True)
    config.write_text(yaml.safe_dump(raw))
    (root / current_root).mkdir()
    if not old:
        (root / "cuda_bindings_12").mkdir()
    return root, raw


@pytest.mark.parametrize(
    ("old", "schema2", "tag", "expected_root", "legacy_artifact"),
    [
        (True, False, "v13.4.4", "cuda_bindings", True),
        (False, False, "v12.9.11", "cuda_bindings_12", False),
        (False, True, "v12.9.11", "cuda_bindings_12", False),
        (False, True, "v13.4.9", "cuda_bindings_13", False),
    ],
)
@pytest.mark.agent_authored(model="gpt-6")
def test_tagged_source_compatibility(tmp_path, old, schema2, tag, expected_root, legacy_artifact):
    source, _ = _source(tmp_path, old=old, schema2=schema2)
    record = resolve_release_bindings_package(tag, source)
    assert record["package_root"] == expected_root
    assert record["release_version"] == tag.removeprefix("v")
    assert record.get("allow_legacy_python_artifact", False) is legacy_artifact


@pytest.mark.agent_authored(model="gpt-6")
def test_old_cuda13_source_cannot_resolve_cuda12(tmp_path):
    source, _ = _source(tmp_path, old=True)
    with pytest.raises(ValueError, match="no bindings root"):
        resolve_release_bindings_package("v12.9.11", source)


@pytest.mark.agent_authored(model="gpt-6")
def test_release_cli_uses_tagged_source_without_control_fallback(tmp_path, capsys):
    source, _ = _source(tmp_path / "tagged", old=True)
    assert (
        main(
            [
                "--config",
                str(tmp_path / "absent-control.yml"),
                "--release-tag",
                "v13.4.4",
                "--release-source-root",
                str(source),
            ]
        )
        == 0
    )
    assert json.loads(capsys.readouterr().out) == {
        "package_root": "cuda_bindings",
        "toolkit_version": "13.4.2",
        "release_version": "13.4.4",
        "allow_legacy_python_artifact": True,
    }
    empty = tmp_path / "absent-source"
    empty.mkdir()
    with pytest.raises(SystemExit):
        main(
            [
                "--config",
                str(source / "ci/versions.yml"),
                "--release-tag",
                "v13.4.4",
                "--release-source-root",
                str(empty),
            ]
        )


@pytest.mark.agent_authored(model="gpt-6")
def test_dependent_release_uses_current_tagged_bindings(tmp_path):
    source, _ = _source(tmp_path, schema2=True)
    assert resolve_release_bindings_package("cuda-core-v1.2.0", source) == {
        "package_root": "cuda_bindings_13",
        "toolkit_version": "13.4.2",
    }


@pytest.mark.parametrize("conflict", ["roles", "majors"])
@pytest.mark.agent_authored(model="gpt-6")
def test_schema2_rejects_ambiguous_release_routing(tmp_path, conflict):
    _, raw = _source(tmp_path, schema2=True)
    maintenance = raw["cuda"]["bindings"]["package_roots"]["cuda_bindings_12"]
    maintenance["release_status" if conflict == "roles" else "toolkit_version"] = (
        "current" if conflict == "roles" else "13.4.1"
    )
    message = "one current and one maintenance" if conflict == "roles" else "cuda_major values must be unique"
    with pytest.raises(ValueError, match=message):
        parse_config(raw)


@pytest.mark.agent_authored(model="gpt-6")
def test_github_env_exports_record_without_newline_injection(tmp_path):
    path = tmp_path / "github-env"
    record = {"package_root": "cuda_bindings_12", "toolkit_version": "12.9.1", "note": "one\ntwo"}
    write_github_env(record, path)
    before = path.read_text()
    lines = before.splitlines()
    assert lines[:2] == ["BUILD_CTK_VER=12.9.1", "BINDINGS_PACKAGE_ROOT=cuda_bindings_12"]
    assert len(lines) == 3
    assert json.loads(lines[2].split("=", 1)[1]) == record
    with pytest.raises(ValueError, match="resolved package_root"):
        write_github_env({**record, "package_root": "cuda_bindings_12\nINJECTED=1"}, path)
    assert path.read_text() == before
