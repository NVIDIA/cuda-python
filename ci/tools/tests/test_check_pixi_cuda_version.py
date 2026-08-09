# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import os
import sys
import textwrap

import pytest

# check_pixi_cuda_version imports PyYAML at module scope (the pre-commit hook
# declares it via additional_dependencies), so skip rather than fail collection
# when this module is exercised outside that environment.
pytest.importorskip("yaml")

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
import check_pixi_cuda_version as mod
from check_pixi_cuda_version import parse_build_version

PIXI_TOML = textwrap.dedent("""\
    [workspace.build-variants]
    cuda-version = ["12.*", "13.3.*"]

    [feature.cu13.dependencies]
    cuda-version = "13.3.*"
    """)


@pytest.mark.agent_authored(model="claude-opus-5")
@pytest.mark.parametrize(
    ("raw", "expected"),
    [
        pytest.param("13.3.0", ("13", "3"), id="three-part"),
        pytest.param("12.9.1", ("12", "9"), id="three-part-other"),
        pytest.param("13.3", ("13", "3"), id="two-part"),
        pytest.param("13.3.0.1", ("13", "3"), id="four-part"),
    ],
)
def test_parse_build_version_accepts_version_strings(raw, expected):
    assert parse_build_version(raw) == expected


@pytest.mark.agent_authored(model="claude-opus-5")
@pytest.mark.parametrize(
    "raw",
    [
        # YAML turns an unquoted `version: 13.3` into a float and an unquoted
        # `version: 13` into an int. Neither has .split(), so the tool used to
        # die with an AttributeError traceback.
        pytest.param(13.3, id="float-from-unquoted-yaml"),
        pytest.param(13, id="int-from-unquoted-yaml"),
        pytest.param(None, id="none-from-empty-yaml-value"),
        pytest.param(["13", "3"], id="list"),
        # Quoted, but not a <major>.<minor> version: the tuple unpacking used
        # to die with "not enough values to unpack".
        pytest.param("13", id="single-component"),
        pytest.param("", id="empty-string"),
        pytest.param("13.", id="trailing-dot"),
        pytest.param(".3", id="leading-dot"),
        pytest.param("cuda.13", id="non-numeric-major"),
    ],
)
def test_parse_build_version_rejects_everything_else(raw):
    assert parse_build_version(raw) is None


@pytest.mark.agent_authored(model="claude-opus-5")
@pytest.mark.parametrize(
    ("yaml_value", "note"),
    [
        pytest.param("13.3", "unquoted two-part version loads as a float", id="unquoted-float"),
        pytest.param("13", "unquoted single number loads as an int", id="unquoted-int"),
        pytest.param('"13"', "quoted but missing a minor component", id="quoted-single-component"),
    ],
)
def test_main_reports_a_malformed_build_version(tmp_path, monkeypatch, capsys, yaml_value, note):
    """A malformed ci/versions.yml must produce this tool's own diagnostic and
    exit 2, not an uncaught traceback out of a pre-commit hook."""
    (tmp_path / "ci").mkdir()
    (tmp_path / "ci" / "versions.yml").write_text(f"cuda:\n  build:\n    version: {yaml_value}\n", encoding="utf-8")
    pixi_files = []
    for package in ("cuda_bindings", "cuda_core"):
        (tmp_path / package).mkdir()
        path = tmp_path / package / "pixi.toml"
        path.write_text(PIXI_TOML, encoding="utf-8")
        pixi_files.append(path)

    monkeypatch.setattr(mod, "ROOT", tmp_path)
    monkeypatch.setattr(mod, "VERSIONS_FILE_PATH", tmp_path / "ci" / "versions.yml")
    monkeypatch.setattr(mod, "PIXI_FILES", pixi_files)

    assert mod.main() == 2, note
    assert "is not a '<major>.<minor>[.<patch>]' version string" in capsys.readouterr().err


@pytest.mark.agent_authored(model="claude-opus-5")
def test_main_accepts_a_well_formed_build_version(tmp_path, monkeypatch):
    (tmp_path / "ci").mkdir()
    (tmp_path / "ci" / "versions.yml").write_text('cuda:\n  build:\n    version: "13.3.0"\n', encoding="utf-8")
    pixi_files = []
    for package in ("cuda_bindings", "cuda_core"):
        (tmp_path / package).mkdir()
        path = tmp_path / package / "pixi.toml"
        path.write_text(PIXI_TOML, encoding="utf-8")
        pixi_files.append(path)

    monkeypatch.setattr(mod, "ROOT", tmp_path)
    monkeypatch.setattr(mod, "VERSIONS_FILE_PATH", tmp_path / "ci" / "versions.yml")
    monkeypatch.setattr(mod, "PIXI_FILES", pixi_files)

    assert mod.main() == 0
