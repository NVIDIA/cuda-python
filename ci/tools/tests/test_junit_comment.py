# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import os
import subprocess
import sys

import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from junit_comment import MAX_LISTED, collect, render

SCRIPT = os.path.join(os.path.dirname(__file__), "..", "junit_comment.py")
AGENT = pytest.mark.agent_authored(model="claude-fable-5-1")

LINUX = "test-results-standard-linux-64-py3.12-cu13.4.2-wheels-l4-x1-latest"
WINDOWS = "test-results-standard-win-64-py3.12-cu13.4.2-local-a100-x1-latest-TCC"


def junit(cases: list[tuple[str, str]]) -> str:
    """A minimal pytest-style JUnit file; each case is (name, outcome)."""
    body = []
    for name, state in cases:
        child = {
            "passed": "",
            "skipped": '<skipped message="not here"/>',
            "failed": '<failure message="assert 1 == 2">trace</failure>',
            "error": '<error message="failed on setup">trace</error>',
        }[state]
        body.append(f'<testcase classname="tests.test_mod" name="{name}" time="0.1">{child}</testcase>')
    return f'<?xml version="1.0"?><testsuites><testsuite name="pytest" tests="{len(cases)}">{"".join(body)}</testsuite></testsuites>'


def write_results(root, files: dict[str, list[tuple[str, str]]]) -> None:
    for rel, cases in files.items():
        path = root / rel
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(junit(cases), encoding="utf-8")


@pytest.fixture
def two_configurations(tmp_path):
    results = tmp_path / "test-results"
    write_results(
        results,
        {
            f"{LINUX}/junit-core.xml": [
                ("t_ok", "passed"),
                ("t_skip", "skipped"),
                ("t_fail", "failed"),
                ("t_err", "error"),
            ],
            f"{LINUX}/junit-bindings.xml": [("t_bind", "passed")],
            f"{WINDOWS}/junit-core.xml": [
                ("t_ok", "passed"),
                ("t_skip", "skipped"),
                ("t_fail", "failed"),
                ("t_err", "passed"),
            ],
        },
    )
    return results


@AGENT
def test_collect_labels_configurations_and_counts_executions(two_configurations):
    results = collect(two_configurations)
    assert results.files == 3
    assert sorted(results.configurations) == [
        "linux-64-py3.12-cu13.4.2-wheels-l4-x1-latest",
        "win-64-py3.12-cu13.4.2-local-a100-x1-latest-TCC",
    ]
    linux = results.configurations["linux-64-py3.12-cu13.4.2-wheels-l4-x1-latest"]
    assert (linux.passed, linux.failed, linux.errors, linux.skipped) == (2, 1, 1, 1)
    assert results.not_passed == {
        "tests.test_mod::t_fail": {
            "linux-64-py3.12-cu13.4.2-wheels-l4-x1-latest": "failed",
            "win-64-py3.12-cu13.4.2-local-a100-x1-latest-TCC": "failed",
        },
        "tests.test_mod::t_err": {"linux-64-py3.12-cu13.4.2-wheels-l4-x1-latest": "error"},
    }


@AGENT
def test_render_explains_failures_in_plain_english(two_configurations):
    text = render(collect(two_configurations), "ce642a78deadbeef", "https://example.test/run/1")
    assert text.startswith("## Test results for commit ce642a7\n")
    assert "2 configurations ran the test suites: linux-64-py3.12-cu13.4.2-wheels-l4-x1-latest; win-64-" in text
    assert "2 tests did not pass in at least one configuration. 1 failed inside the test body" in text
    assert "1 hit an error outside the test body" in text
    assert "4 passed, 2 failed, 1 errored, 2 skipped." in text
    assert "- `tests.test_mod::t_fail`: failed in all 2 configurations" in text
    assert "- `tests.test_mod::t_err`: error in linux-64-py3.12-cu13.4.2-wheels-l4-x1-latest" in text
    assert "Summary page: https://example.test/run/1" in text
    assert "A test *fails* when" in text


@AGENT
def test_render_green_run_is_one_sentence_plus_link(tmp_path):
    results = tmp_path / "test-results"
    write_results(results, {f"{LINUX}/junit-core.xml": [("t_ok", "passed"), ("t_skip", "skipped")]})
    text = render(collect(results), "abc1234", "https://example.test/run/2")
    assert "1 configuration ran the test suites" in text
    assert "No test failed. Counting every test execution across all configurations: 1 passed, 1 skipped." in text
    assert "Tests that did not pass" not in text
    assert "https://example.test/run/2" in text


@AGENT
def test_render_without_result_files_says_so(tmp_path):
    empty = tmp_path / "test-results"
    empty.mkdir()
    text = render(collect(empty), "abc1234", "https://example.test/run/3")
    assert "No test result files were found for this run." in text
    assert "https://example.test/run/3" in text


@AGENT
def test_render_truncates_a_long_failure_list(tmp_path):
    results = tmp_path / "test-results"
    write_results(results, {f"{LINUX}/junit-core.xml": [(f"t_{i:03d}", "failed") for i in range(MAX_LISTED + 7)]})
    text = render(collect(results), "abc1234", "https://example.test/run/4")
    assert text.count("\n- `tests.test_mod::t_") == MAX_LISTED
    assert "- and 7 more" in text


@AGENT
def test_error_outranks_failure_for_the_same_test(tmp_path):
    results = tmp_path / "test-results"
    write_results(
        results,
        {
            f"{LINUX}/junit-core.xml": [("t_mixed", "failed")],
            f"{WINDOWS}/junit-core.xml": [("t_mixed", "error")],
        },
    )
    text = render(collect(results), "abc1234", "https://example.test/run/5")
    assert "1 test did not pass in at least one configuration. 0 failed inside the test body" in text
    assert "1 hit an error outside the test body" in text
    assert "- `tests.test_mod::t_mixed`: error in all 2 configurations" in text


@AGENT
def test_cli_writes_the_comment_file(two_configurations, tmp_path):
    out = tmp_path / "comment.md"
    result = subprocess.run(  # noqa: S603 - invokes the repository script under test
        [
            sys.executable,
            SCRIPT,
            "--results",
            str(two_configurations),
            "--commit",
            "ce642a78",
            "--run-url",
            "https://example.test/run/6",
            "--output",
            str(out),
        ],
        capture_output=True,
        text=True,
        check=True,
    )
    assert out.read_text(encoding="utf-8") == result.stdout
    assert "## Test results for commit ce642a7" in result.stdout
