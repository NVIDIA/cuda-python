# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import os
import sys
import xml.etree.ElementTree as ET

import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from allure_prepare import configuration_name, environment_id, main, prepare, render_config

JUNIT = """<?xml version="1.0" encoding="utf-8"?>
<testsuites><testsuite name="pytest" errors="0" failures="1" skipped="0" tests="2" time="0.1">
<testcase classname="tests.test_a" name="test_ok" time="0.01" />
<testcase classname="tests.test_a" name="test_bad" time="0.02"><failure message="boom">boom</failure></testcase>
</testsuite></testsuites>
"""


def write_artifacts(root):
    for artifact in ("test-results-standard-linux-64-py3.12", "test-results-nightly-cuda-core-win-64"):
        directory = root / artifact
        directory.mkdir(parents=True)
        (directory / "junit-core.xml").write_text(JUNIT, encoding="utf-8")
        (directory / "junit-bindings.xml").write_text(JUNIT, encoding="utf-8")


@pytest.mark.agent_authored(model="claude-fable-5-1")
def test_configuration_name_strips_the_artifact_and_standard_prefixes(tmp_path):
    assert configuration_name(tmp_path / "test-results-standard-linux-64-py3.12") == "linux-64-py3.12"
    assert configuration_name(tmp_path / "test-results-nightly-cuda-core-win-64") == "nightly-cuda-core-win-64"


@pytest.mark.agent_authored(model="claude-fable-5-1")
def test_prepare_flattens_the_files_and_tags_every_suite_with_its_configuration(tmp_path):
    write_artifacts(tmp_path / "in")
    counts = prepare(tmp_path / "in", tmp_path / "out")
    assert counts == {"linux-64-py3.12": 2, "nightly-cuda-core-win-64": 2}
    assert sorted(p.name for p in (tmp_path / "out").iterdir()) == [
        "linux-64-py3.12-junit-bindings.xml",
        "linux-64-py3.12-junit-core.xml",
        "nightly-cuda-core-win-64-junit-bindings.xml",
        "nightly-cuda-core-win-64-junit-core.xml",
    ]
    tree = ET.parse(tmp_path / "out" / "linux-64-py3.12-junit-core.xml")  # noqa: S314
    assert [suite.get("package") for suite in tree.iter("testsuite")] == ["linux-64-py3.12"]
    assert len(list(tree.iter("testcase"))) == 2


@pytest.mark.agent_authored(model="claude-fable-5-1")
def test_render_config_has_one_environment_per_configuration():
    text = render_config("demo", ["linux-64-py3.12", "win-64 (TCC)"])
    assert 'name: "demo"' in text
    assert '"linux-64-py3-12": {' in text
    assert '"win-64--TCC-": {' in text
    assert 'value === "win-64 (TCC)"' in text
    assert text.count("matcher:") == 2


@pytest.mark.agent_authored(model="claude-fable-5-1")
def test_environment_id_is_sanitized_and_unique():
    taken: set[str] = set()
    assert environment_id("a.b", taken) == "a-b"
    assert environment_id("a_b", taken) == "a_b"
    assert environment_id("a-b", taken) == "a-b-2"


@pytest.mark.agent_authored(model="claude-fable-5-1")
def test_main_fails_when_there_are_no_results(tmp_path, capsys):
    (tmp_path / "empty").mkdir()
    rc = main(
        ["--results", str(tmp_path / "empty"), "--output", str(tmp_path / "out"), "--config", str(tmp_path / "rc.mjs")]
    )
    assert rc == 1
    assert "no JUnit XML" in capsys.readouterr().err
