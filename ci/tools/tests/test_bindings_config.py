# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import copy
import json
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from bindings_config import BindingsConfigError, load_config, main, validate_config


def valid_config() -> dict[str, object]:
    return {
        "schema_version": 2,
        "cuda": {
            "bindings": {
                "lines": {
                    "released-12": {
                        "source_dir": "cuda_bindings_12",
                        "ctk_target": "12.9",
                        "toolkit_version": "12.9.1",
                        "toolkit_channel": "stable",
                        "tag_series": "v12.9.",
                        "allow_alpha_beta_tags": False,
                    },
                    "released-13": {
                        "source_dir": "cuda_bindings",
                        "ctk_target": "13.3",
                        "toolkit_version": "13.3.0",
                        "toolkit_channel": "stable",
                        "tag_series": "v13.3.",
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


@pytest.mark.agent_authored(model="gpt-5.6-sol")
def test_live_registry_has_ordered_lines_and_roles():
    config = load_config()

    assert [line.line_id for line in config.lines] == ["released-12", "released-13"]
    assert config.line_for_role("current").line_id == "released-13"
    assert [line.line_id for line in config.lines_for_role("maintenance")] == ["released-12"]
    assert config.get_line("released-12").source_dir == "cuda_bindings_12"
    assert config.get_line("released-12").cuda_major == "12"
    assert config.get_line("released-12").cuda_variant == "cu12"


@pytest.mark.agent_authored(model="gpt-5.6-sol")
def test_tag_matching_honors_each_line_policy():
    config = validate_config(valid_config())

    assert config.match_tag("v12.9.8").line_id == "released-12"
    assert config.match_tag("v12.9.8a1") is None
    assert config.match_tag("v13.3.0").line_id == "released-13"
    assert config.match_tag("v13.3.0b1").line_id == "released-13"
    assert config.match_tag("v13.3.2.post1").line_id == "released-13"
    assert config.match_tag("v12.9.8.post1").line_id == "released-12"
    assert config.match_tag("v13.3.0rc1") is None
    assert config.match_tag("v13.2.0") is None
    assert config.match_tag("cuda-core-v1.0.0") is None


@pytest.mark.agent_authored(model="gpt-5.6-sol")
def test_same_major_lines_remain_distinct():
    raw = valid_config()
    bindings = raw["cuda"]["bindings"]
    bindings["lines"] = {
        "released-11-7": {
            "source_dir": "cuda_bindings_11_7",
            "ctk_target": "11.7",
            "toolkit_version": "11.7.1",
            "toolkit_channel": "stable",
            "tag_series": "v11.7.",
            "allow_alpha_beta_tags": False,
        },
        "released-11-8": {
            "source_dir": "cuda_bindings_11_8",
            "ctk_target": "11.8",
            "toolkit_version": "11.8.0",
            "toolkit_channel": "stable",
            "tag_series": "v11.8.",
            "allow_alpha_beta_tags": False,
        },
    }
    bindings["roles"] = {
        "current": "released-11-8",
        "maintenance": ["released-11-7"],
    }

    config = validate_config(raw)

    assert [line.line_id for line in config.lines] == ["released-11-7", "released-11-8"]
    assert [line.cuda_variant for line in config.lines] == ["cu11", "cu11"]
    assert config.line_for_role("current").line_id == "released-11-8"
    assert [line.line_id for line in config.lines_for_role("maintenance")] == ["released-11-7"]


@pytest.mark.agent_authored(model="gpt-5.6-sol")
def test_normalized_output_is_matrix_friendly_and_json_stable():
    config = validate_config(valid_config())
    normalized = config.to_dict()

    assert normalized["lines"] == [
        {
            "line_id": "released-12",
            "source_dir": "cuda_bindings_12",
            "ctk_target": "12.9",
            "toolkit_version": "12.9.1",
            "toolkit_channel": "stable",
            "tag_series": "v12.9.",
            "allow_alpha_beta_tags": False,
            "cuda_major": "12",
            "cuda_variant": "cu12",
            "roles": ["maintenance"],
        },
        {
            "line_id": "released-13",
            "source_dir": "cuda_bindings",
            "ctk_target": "13.3",
            "toolkit_version": "13.3.0",
            "toolkit_channel": "stable",
            "tag_series": "v13.3.",
            "allow_alpha_beta_tags": True,
            "cuda_major": "13",
            "cuda_variant": "cu13",
            "roles": ["current"],
        },
    ]
    assert normalized["roles"] == {
        "current": ["released-13"],
        "maintenance": ["released-12"],
    }
    assert json.loads(config.to_json()) == normalized
    assert " " not in config.to_json()


@pytest.mark.parametrize(
    ("mutate", "message"),
    [
        (lambda data: data.update(schema_version=1), "schema_version must be 2"),
        (
            lambda data: data["cuda"]["bindings"]["lines"]["released-12"].update(ctk_target="12"),
            "released-12.ctk_target has invalid format",
        ),
        (
            lambda data: data["cuda"]["bindings"]["lines"]["released-12"].update(toolkit_version="13.3.0"),
            "must belong to CTK target",
        ),
        (
            lambda data: data["cuda"]["bindings"]["lines"]["released-12"].update(tag_series="v13."),
            "tag_series must be",
        ),
        (
            lambda data: data["cuda"]["bindings"]["roles"].update(current="missing"),
            "reference unknown lines",
        ),
        (
            lambda data: data["cuda"]["bindings"]["roles"].update(maintenance=["released-12", "released-13"]),
            "must not overlap",
        ),
        (
            lambda data: data["cuda"]["bindings"]["roles"].update(unexpected=[]),
            "cuda.bindings.roles has unexpected keys",
        ),
        (
            lambda data: data["cuda"]["bindings"]["lines"]["released-12"].update(toolkit_channel="preview"),
            "toolkit_channel must be one of",
        ),
        (
            lambda data: data["cuda"]["bindings"]["lines"]["released-12"].update(allow_alpha_beta_tags="false"),
            "allow_alpha_beta_tags must be a boolean",
        ),
        (
            lambda data: data["cuda"]["bindings"]["lines"]["released-12"].update(
                ctk_target="13.3", toolkit_version="13.3.1", tag_series="v13.3."
            ),
            "ctk_target values must be unique",
        ),
        (
            lambda data: data["cuda"]["bindings"]["lines"]["released-12"].update(source_dir="../cuda_bindings_12"),
            "normalized repository-relative POSIX path",
        ),
        (
            lambda data: data["cuda"]["bindings"]["lines"]["released-12"].update(
                source_dir="cuda_bindings_12\nINJECTED=value"
            ),
            "source_dir has invalid format",
        ),
    ],
)
@pytest.mark.agent_authored(model="gpt-5.6-sol")
def test_invalid_registry_is_rejected(mutate, message):
    data = copy.deepcopy(valid_config())
    mutate(data)

    with pytest.raises(BindingsConfigError, match=message):
        validate_config(data)


@pytest.mark.agent_authored(model="gpt-5.6-sol")
def test_load_wraps_yaml_errors(tmp_path):
    path = tmp_path / "versions.yml"
    path.write_text("cuda: [unterminated", encoding="utf-8")

    with pytest.raises(BindingsConfigError, match="could not read"):
        load_config(path)


@pytest.mark.agent_authored(model="gpt-5.6-sol")
def test_cli_emits_normalized_records_and_rejects_unknown_tags(capsys):
    assert main(["list"]) == 0
    lines = json.loads(capsys.readouterr().out)
    assert [line["line_id"] for line in lines] == ["released-12", "released-13"]

    assert main(["get", "--role", "current"]) == 0
    current = json.loads(capsys.readouterr().out)
    assert current["line_id"] == "released-13"
    assert current["cuda_variant"] == "cu13"

    with pytest.raises(SystemExit, match="2"):
        main(["match-tag", "v14.0.0"])
    assert "no CUDA bindings line matches release tag" in capsys.readouterr().err
