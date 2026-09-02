# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest
import yaml

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from resolve_release_bindings_line import ReleaseBindingsLineError, main, resolve_release_bindings_line


def registry(*, current_dir: str = "cuda_bindings", maintenance_dir: str = "cuda_bindings_12") -> dict[str, object]:
    return {
        "schema_version": 2,
        "cuda": {
            "bindings": {
                "lines": {
                    "released-12": {
                        "source_dir": maintenance_dir,
                        "toolkit_version": "12.9.1",
                        "allow_alpha_beta_tags": False,
                    },
                    "released-13": {
                        "source_dir": current_dir,
                        "toolkit_version": "13.3.0",
                        "allow_alpha_beta_tags": True,
                    },
                },
                "roles": {
                    "current": "released-13",
                    "maintenance": ["released-12"],
                },
            }
        },
    }


def write_yaml(path: Path, data: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(yaml.safe_dump(data, sort_keys=False), encoding="utf-8")


@pytest.mark.agent_authored(model="gpt-5.6-sol")
@pytest.mark.parametrize(
    ("release_tag", "expected_line", "expected_dir"),
    (
        ("v13.3.2", "released-13", "tag-current"),
        ("v12.9.8", "released-12", "tag-maintenance"),
    ),
)
def test_modern_tag_tree_is_authoritative(tmp_path, capsys, release_tag, expected_line, expected_dir):
    release_root = tmp_path / "release"
    tagged_config = release_root / "ci" / "versions.yml"
    control_config = tmp_path / "control" / "versions.yml"
    write_yaml(tagged_config, registry(current_dir="tag-current", maintenance_dir="tag-maintenance"))
    write_yaml(control_config, registry(current_dir="control-current", maintenance_dir="control-maintenance"))

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

    line = json.loads(capsys.readouterr().out)
    assert line["line_id"] == expected_line
    assert line["release_source_dir"] == expected_dir
    assert line["release_registry_origin"] == "tag"


@pytest.mark.agent_authored(model="gpt-5.6-sol")
def test_legacy_tag_tree_uses_control_registry_and_legacy_layout(tmp_path):
    release_root = tmp_path / "release"
    write_yaml(release_root / "ci" / "versions.yml", {"cuda": {"build": {"version": "12.9.1"}}})
    (release_root / "cuda_bindings").mkdir(parents=True)
    control_config = tmp_path / "control" / "versions.yml"
    write_yaml(control_config, registry())

    line = resolve_release_bindings_line("v12.9.8", release_root, control_config)

    assert line["line_id"] == "released-12"
    assert line["source_dir"] == "cuda_bindings_12"
    assert line["release_source_dir"] == "cuda_bindings"
    assert line["release_registry_origin"] == "control"


@pytest.mark.agent_authored(model="gpt-5.6-sol")
def test_invalid_modern_tag_config_does_not_fall_back(tmp_path):
    release_root = tmp_path / "release"
    control_config = tmp_path / "control" / "versions.yml"
    write_yaml(release_root / "ci" / "versions.yml", {"schema_version": 2, "cuda": {}})
    write_yaml(control_config, registry())

    with pytest.raises(ReleaseBindingsLineError, match="invalid schema-2 tagged config"):
        resolve_release_bindings_line("v13.3.0", release_root, control_config)


@pytest.mark.agent_authored(model="gpt-5.6-sol")
def test_unknown_release_tag_fails_closed(tmp_path):
    release_root = tmp_path / "release"
    release_root.mkdir()
    control_config = tmp_path / "control" / "versions.yml"
    write_yaml(control_config, registry())

    with pytest.raises(ReleaseBindingsLineError, match="no CUDA bindings line"):
        resolve_release_bindings_line("v14.0.0", release_root, control_config)
