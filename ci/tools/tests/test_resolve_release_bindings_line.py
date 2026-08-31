# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest
import yaml

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from resolve_release_bindings_line import (
    ReleaseBindingsLineError,
    main,
    resolve_release_bindings_line,
)


def registry(
    *,
    line_id: str = "released-13",
    source_dir: str = "cuda_bindings",
    ctk_target: str = "13.3",
    allow_alpha_beta_tags: bool = True,
) -> dict[str, object]:
    return {
        "schema_version": 2,
        "cuda": {
            "bindings": {
                "lines": {
                    line_id: {
                        "source_dir": source_dir,
                        "ctk_target": ctk_target,
                        "toolkit_version": f"{ctk_target}.0",
                        "toolkit_channel": "stable",
                        "tag_series": f"v{ctk_target}.",
                        "allow_alpha_beta_tags": allow_alpha_beta_tags,
                    }
                },
                "roles": {
                    "current": line_id,
                    "maintenance": [],
                    "unreleased": [],
                },
            }
        },
    }


def write_yaml(path: Path, data: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(yaml.safe_dump(data, sort_keys=False), encoding="utf-8")


@pytest.mark.agent_authored(model="gpt-5.6-sol")
def test_modern_tag_tree_overrides_changed_control_registry(tmp_path, capsys):
    release_root = tmp_path / "release-source"
    tagged_config = release_root / "ci" / "versions.yml"
    control_config = tmp_path / "control" / "versions.yml"
    write_yaml(
        tagged_config,
        registry(line_id="released-13-at-tag", source_dir="cuda_bindings_13_at_tag"),
    )
    write_yaml(
        control_config,
        registry(line_id="released-13-now", source_dir="cuda_bindings_13_now"),
    )
    (release_root / "cuda_bindings").mkdir()

    assert (
        main(
            [
                "--release-tag",
                "v13.3.2",
                "--release-source-root",
                str(release_root),
                "--control-config",
                str(control_config),
            ]
        )
        == 0
    )

    output = capsys.readouterr().out.strip()
    assert " " not in output
    line = json.loads(output)
    assert line["line_id"] == "released-13-at-tag"
    assert line["source_dir"] == "cuda_bindings_13_at_tag"
    assert line["release_source_dir"] == "cuda_bindings_13_at_tag"
    assert line["release_registry_origin"] == "tag"


@pytest.mark.parametrize("has_legacy_config", [False, True])
@pytest.mark.agent_authored(model="gpt-5.6-sol")
def test_legacy_tag_tree_uses_control_registry(tmp_path, has_legacy_config):
    release_root = tmp_path / "release-source"
    control_config = tmp_path / "control" / "versions.yml"
    if has_legacy_config:
        write_yaml(
            release_root / "ci" / "versions.yml",
            {"cuda": {"build": {"version": "12.9.1"}}},
        )
    write_yaml(
        control_config,
        registry(
            line_id="released-12",
            source_dir="cuda_bindings_12",
            ctk_target="12.9",
            allow_alpha_beta_tags=False,
        ),
    )
    (release_root / "cuda_bindings").mkdir(parents=True)

    line = resolve_release_bindings_line("v12.9.8", release_root, control_config)

    assert line["line_id"] == "released-12"
    assert line["source_dir"] == "cuda_bindings_12"
    assert line["release_source_dir"] == "cuda_bindings"
    assert line["release_registry_origin"] == "control"


@pytest.mark.agent_authored(model="gpt-5.6-sol")
def test_legacy_layout_keeps_configured_root_when_present(tmp_path):
    release_root = tmp_path / "release-source"
    control_config = tmp_path / "control" / "versions.yml"
    write_yaml(
        release_root / "ci" / "versions.yml",
        {"cuda": {"build": {"version": "12.9.1"}}},
    )
    (release_root / "cuda_bindings").mkdir()
    (release_root / "cuda_bindings_12").mkdir()
    write_yaml(
        control_config,
        registry(
            line_id="released-12",
            source_dir="cuda_bindings_12",
            ctk_target="12.9",
            allow_alpha_beta_tags=False,
        ),
    )

    line = resolve_release_bindings_line("v12.9.8", release_root, control_config)

    assert line["release_source_dir"] == "cuda_bindings_12"
    assert line["release_registry_origin"] == "control"


@pytest.mark.parametrize("release_tag", ["v13.3.0a1", "v13.3.0b2", "v13.3.0.post1"])
@pytest.mark.agent_authored(model="gpt-5.6-sol")
def test_allowed_alpha_beta_and_post_tags_are_resolved(tmp_path, release_tag):
    release_root = tmp_path / "release-source"
    control_config = tmp_path / "control" / "versions.yml"
    write_yaml(release_root / "ci" / "versions.yml", registry())
    write_yaml(control_config, registry(line_id="unused-control"))

    line = resolve_release_bindings_line(release_tag, release_root, control_config)

    assert line["line_id"] == "released-13"


@pytest.mark.agent_authored(model="gpt-5.6-sol")
def test_invalid_modern_tag_config_fails_without_control_fallback(tmp_path):
    release_root = tmp_path / "release-source"
    control_config = tmp_path / "control" / "versions.yml"
    write_yaml(release_root / "ci" / "versions.yml", {"schema_version": 2, "cuda": {}})
    write_yaml(control_config, registry())

    with pytest.raises(ReleaseBindingsLineError, match="invalid schema-2 tagged config"):
        resolve_release_bindings_line("v13.3.0", release_root, control_config)


@pytest.mark.agent_authored(model="gpt-5.6-sol")
def test_unknown_release_tag_fails_closed(tmp_path):
    release_root = tmp_path / "release-source"
    release_root.mkdir()
    control_config = tmp_path / "control" / "versions.yml"
    write_yaml(control_config, registry())

    with pytest.raises(ReleaseBindingsLineError, match="no CUDA bindings line"):
        resolve_release_bindings_line("v14.0.0", release_root, control_config)
