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


def line(source_dir, toolkit_version, allow_alpha_beta_tags=False):
    return {
        "source_dir": source_dir,
        "toolkit_version": toolkit_version,
        "allow_alpha_beta_tags": allow_alpha_beta_tags,
    }


def valid_config():
    return {
        "schema_version": 2,
        "cuda": {
            "bindings": {
                "lines": {
                    "released-12": line("cuda_bindings_12", "12.9.1"),
                    "released-13": line("cuda_bindings", "13.3.0", True),
                },
                "roles": {"current": "released-13", "maintenance": ["released-12"]},
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
    assert config.get_line("released-12").ctk_target == "12.9"
    assert config.get_line("released-12").tag_series == "v12.9."
    assert config.get_line("released-12").cuda_major == "12"
    assert config.get_line("released-12").cuda_variant == "cu12"
    normalized = config.to_dict()
    assert normalized["roles"] == {"current": ["released-13"], "maintenance": ["released-12"]}
    assert json.loads(config.to_json()) == normalized


@pytest.mark.agent_authored(model="gpt-5.6-sol")
def test_tag_matching_honors_each_line_policy():
    config = validate_config(valid_config())
    assert config.match_tag("v12.9.8").line_id == "released-12"
    assert config.match_tag("v12.9.8a1") is None
    assert config.match_tag("v13.3.0b1").line_id == "released-13"
    assert config.match_tag("v13.3.2.post1").line_id == "released-13"
    assert config.match_tag("v13.3.0rc1") is None


@pytest.mark.agent_authored(model="gpt-5.6-sol")
def test_same_major_lines_remain_distinct():
    raw = valid_config()
    bindings = raw["cuda"]["bindings"]
    bindings["lines"] = {
        "released-11-7": line("cuda_bindings_11_7", "11.7.1"),
        "released-11-8": line("cuda_bindings_11_8", "11.8.0"),
    }
    bindings["roles"] = {"current": "released-11-8", "candidate": ["released-11-8"]}

    config = validate_config(raw)

    assert [line.line_id for line in config.lines] == ["released-11-7", "released-11-8"]
    assert [line.cuda_variant for line in config.lines] == ["cu11", "cu11"]
    assert config.line_for_role("current").line_id == "released-11-8"
    assert config.line_for_role("candidate").line_id == "released-11-8"
    assert config.line_to_dict(config.get_line("released-11-7"))["roles"] == []


@pytest.mark.parametrize(
    ("field", "value", "message"),
    [
        ("toolkit_version", "12.9", "toolkit_version has invalid format"),
        ("allow_alpha_beta_tags", "false", "allow_alpha_beta_tags must be a boolean"),
        ("source_dir", "../cuda_bindings_12", "normalized repository-relative POSIX path"),
        ("source_dir", "cuda_bindings_12\nINJECTED=value", "source_dir has invalid format"),
    ],
)
@pytest.mark.agent_authored(model="gpt-5.6-sol")
def test_invalid_registry_is_rejected(field, value, message):
    data = copy.deepcopy(valid_config())
    data["cuda"]["bindings"]["lines"]["released-12"][field] = value
    with pytest.raises(BindingsConfigError, match=message):
        validate_config(data)


@pytest.mark.agent_authored(model="gpt-5.6-sol")
def test_load_wraps_yaml_errors(tmp_path):
    path = tmp_path / "versions.yml"
    path.write_text("cuda: [unterminated", encoding="utf-8")

    with pytest.raises(BindingsConfigError, match="could not read"):
        load_config(path)


@pytest.mark.agent_authored(model="gpt-5.6-sol")
def test_cli_emits_full_registry_lines_and_one_role(capsys):
    assert main([]) == 0
    registry = json.loads(capsys.readouterr().out)
    assert registry["roles"]["current"] == ["released-13"]

    assert main(["--lines"]) == 0
    lines = json.loads(capsys.readouterr().out)
    assert [line["line_id"] for line in lines] == ["released-12", "released-13"]

    assert main(["--role", "current"]) == 0
    current = json.loads(capsys.readouterr().out)
    assert current["line_id"] == "released-13"
    assert current["cuda_variant"] == "cu13"
