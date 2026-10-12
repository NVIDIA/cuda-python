# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import os
import subprocess
import sys
import xml.etree.ElementTree as ET
from pathlib import Path

import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from pytest_batched import (
    _args_for_collection,
    chunk,
    main,
    merge_junit_xml,
    parse_collected_ids,
    split_junitxml,
    strip_path_positionals,
)

AGENT = pytest.mark.agent_authored(model="claude-sonnet-5-5")

SCRIPT = os.path.join(os.path.dirname(__file__), "..", "pytest_batched.py")


def _testsuite(path: Path) -> ET.Element:
    """Parse a JUnit XML file this same test suite wrote and return its ``<testsuite>``."""
    suite = ET.parse(path).getroot().find("testsuite")  # noqa: S314 - self-written fixture file, not untrusted input.
    assert suite is not None
    return suite


def _run_script(args: list[str], cwd: Path, env: dict[str, str] | None = None) -> subprocess.CompletedProcess:
    """Invoke pytest_batched.py as a subprocess with a fixed argv (no shell)."""
    return subprocess.run(  # noqa: S603 - argv is a fixed list, passed directly without a shell.
        [sys.executable, SCRIPT, *args], cwd=cwd, capture_output=True, text=True, env=env
    )


# --- parse_collected_ids ---------------------------------------------------

COLLECT_ONLY_OUTPUT = """\
tests/test_a.py::test_one
tests/test_a.py::test_two[0-5]
tests/test_a.py::test_two[1-5]
tests/test_b.py::TestThing::test_method

12 tests collected in 0.05s
"""


@AGENT
def test_parse_collected_ids_extracts_node_ids_only():
    assert parse_collected_ids(COLLECT_ONLY_OUTPUT) == [
        "tests/test_a.py::test_one",
        "tests/test_a.py::test_two[0-5]",
        "tests/test_a.py::test_two[1-5]",
        "tests/test_b.py::TestThing::test_method",
    ]


@AGENT
def test_parse_collected_ids_ignores_warnings_and_summary():
    output = "tests/test_a.py::test_one\n\n=== warnings summary ===\nsome::warning\n1 test collected\n"
    # "some::warning" contains "::" but is not a real node ID; this function
    # cannot distinguish it from a warning that happens to mention a path, so
    # it is not filtered here. Document the known limitation instead of
    # pretending otherwise.
    assert "tests/test_a.py::test_one" in parse_collected_ids(output)


@AGENT
def test_parse_collected_ids_empty_when_nothing_collected():
    assert parse_collected_ids("no tests ran in 0.00s\n") == []


# --- _args_for_collection ---------------------------------------------------


@AGENT
@pytest.mark.parametrize("flag", ["-v", "-vv", "-q", "-qq", "--verbose", "--quiet"])
def test_args_for_collection_strips_verbosity_flags(flag):
    # A caller-supplied -v would otherwise cancel out the -q that collect_ids
    # adds (pytest sums -v/-q counts), silently switching --collect-only from
    # the flat one-ID-per-line format to the indented tree format.
    assert flag not in _args_for_collection(["-rsxXfE", flag, "--randomly-dont-reorganize", "tests/"])


@AGENT
def test_args_for_collection_keeps_non_verbosity_flags():
    args = ["-rsxXfE", "-v", "--randomly-dont-reorganize", "tests/"]
    assert _args_for_collection(args) == ["-rsxXfE", "--randomly-dont-reorganize", "tests/"]


# --- chunk -------------------------------------------------------------


@AGENT
def test_chunk_splits_into_batches_of_the_requested_size():
    assert chunk(list(range(10)), 3) == [[0, 1, 2], [3, 4, 5], [6, 7, 8], [9]]


@AGENT
def test_chunk_single_batch_when_size_exceeds_item_count():
    assert chunk([1, 2], 1000) == [[1, 2]]


@AGENT
def test_chunk_empty_input_yields_no_batches():
    assert chunk([], 10) == []


@AGENT
def test_chunk_rejects_non_positive_batch_size():
    with pytest.raises(ValueError):
        chunk([1], 0)


# --- strip_path_positionals ----------------------------------------------


@AGENT
def test_strip_path_positionals_drops_existing_paths(tmp_path):
    (tmp_path / "tests").mkdir()
    args = ["-v", "tests", "tests/test_a.py::test_one", "--randomly-dont-reorganize"]
    assert strip_path_positionals(args, tmp_path) == [
        "-v",
        "tests/test_a.py::test_one",
        "--randomly-dont-reorganize",
    ]


@AGENT
def test_strip_path_positionals_keeps_node_ids_and_flags(tmp_path):
    args = ["-v", "tests/test_a.py::test_one[0-5]"]
    assert strip_path_positionals(args, tmp_path) == args


# --- split_junitxml --------------------------------------------------------


@AGENT
@pytest.mark.parametrize(
    "args",
    [
        ["--junitxml", "out.xml", "-v"],
        ["--junit-xml", "out.xml", "-v"],
        ["--junitxml=out.xml", "-v"],
        ["-v", "--junitxml=out.xml"],
    ],
)
def test_split_junitxml_extracts_path_in_any_accepted_form(args):
    remaining, path = split_junitxml(args)
    assert path == "out.xml"
    assert "-v" in remaining
    assert all("junit" not in a for a in remaining)


@AGENT
def test_split_junitxml_absent_returns_none():
    remaining, path = split_junitxml(["-v", "tests/"])
    assert path is None
    assert remaining == ["-v", "tests/"]


@AGENT
def test_split_junitxml_missing_value_raises():
    with pytest.raises(ValueError):
        split_junitxml(["--junitxml"])


# --- merge_junit_xml --------------------------------------------------------


def _write_batch_xml(path: Path, *, tests: int, failures: int, time: float, timestamp: str, testcase_names: list[str]):
    testsuites = ET.Element("testsuites", name="pytest tests")
    suite = ET.SubElement(
        testsuites,
        "testsuite",
        name="pytest",
        errors="0",
        failures=str(failures),
        skipped="0",
        tests=str(tests),
        time=f"{time:.3f}",
        timestamp=timestamp,
        hostname="runner-1",
    )
    for name in testcase_names:
        ET.SubElement(suite, "testcase", classname="tests.test_a", name=name, time="0.001")
    path.write_text('<?xml version="1.0" encoding="utf-8"?>' + ET.tostring(testsuites, encoding="unicode"))


@AGENT
def test_merge_junit_xml_sums_totals_and_concatenates_testcases(tmp_path):
    batch0 = tmp_path / "out.xml.batch0.xml"
    batch1 = tmp_path / "out.xml.batch1.xml"
    _write_batch_xml(batch0, tests=2, failures=1, time=0.5, timestamp="2026-01-01T00:00:00", testcase_names=["a", "b"])
    _write_batch_xml(
        batch1, tests=3, failures=0, time=1.25, timestamp="2026-01-01T00:00:05", testcase_names=["c", "d", "e"]
    )

    output = tmp_path / "merged.xml"
    merge_junit_xml([batch0, batch1], output)

    suite = _testsuite(output)
    assert suite.get("tests") == "5"
    assert suite.get("failures") == "1"
    assert suite.get("time") == "1.750"
    assert suite.get("timestamp") == "2026-01-01T00:00:00"
    assert [tc.get("name") for tc in suite.findall("testcase")] == ["a", "b", "c", "d", "e"]


@AGENT
def test_merge_junit_xml_no_batches_writes_empty_report(tmp_path):
    output = tmp_path / "merged.xml"
    merge_junit_xml([], output)

    suite = _testsuite(output)
    assert suite.get("tests") == "0"


@AGENT
def test_merge_junit_xml_keeps_only_first_batchs_global_properties(tmp_path):
    def write_with_properties(path, value):
        testsuites = ET.Element("testsuites", name="pytest tests")
        suite = ET.SubElement(
            testsuites,
            "testsuite",
            name="pytest",
            errors="0",
            failures="0",
            skipped="0",
            tests="1",
            time="0.001",
            timestamp="2026-01-01T00:00:00",
            hostname="runner-1",
        )
        props = ET.SubElement(suite, "properties")
        ET.SubElement(props, "property", name="ARCH", value=value)
        ET.SubElement(suite, "testcase", classname="tests.test_a", name="a", time="0.001")
        path.write_text(ET.tostring(testsuites, encoding="unicode"))

    batch0, batch1 = tmp_path / "b0.xml", tmp_path / "b1.xml"
    write_with_properties(batch0, "first")
    write_with_properties(batch1, "second")

    output = tmp_path / "merged.xml"
    merge_junit_xml([batch0, batch1], output)

    suite = _testsuite(output)
    properties = suite.findall("properties")
    assert len(properties) == 1
    assert properties[0].find("property").get("value") == "first"


# --- end-to-end via subprocess ----------------------------------------------


def _write_suite(tmp_path: Path, n_tests: int, n_failing: int = 0) -> None:
    lines = []
    for i in range(n_tests):
        if i < n_failing:
            lines.append(f"def test_item_{i}():\n    assert False\n")
        else:
            lines.append(f"def test_item_{i}():\n    assert True\n")
    (tmp_path / "test_suite.py").write_text("\n".join(lines))
    (tmp_path / "pytest.ini").write_text("[pytest]\n")


@AGENT
def test_end_to_end_runs_all_items_across_batches_and_merges_xml(tmp_path):
    _write_suite(tmp_path, n_tests=25)
    out = tmp_path / "junit.xml"

    result = _run_script(["--batch-size", "10", "-v", "--junitxml", str(out), "test_suite.py"], cwd=tmp_path)

    assert result.returncode == 0, result.stdout + result.stderr
    assert "3 batch(es)" in result.stderr

    suite = _testsuite(out)
    assert suite.get("tests") == "25"
    assert suite.get("failures") == "0"
    assert len(suite.findall("testcase")) == 25
    # No leftover per-batch files.
    assert list(tmp_path.glob("junit.xml.batch*.xml")) == []


@AGENT
def test_end_to_end_reports_failures_and_nonzero_exit(tmp_path):
    _write_suite(tmp_path, n_tests=15, n_failing=2)
    out = tmp_path / "junit.xml"

    result = _run_script(["--batch-size", "5", "--junitxml", str(out), "test_suite.py"], cwd=tmp_path)

    assert result.returncode == 1
    suite = _testsuite(out)
    assert suite.get("tests") == "15"
    assert suite.get("failures") == "2"


@AGENT
def test_end_to_end_no_tests_collected_exits_5_and_writes_empty_xml(tmp_path):
    (tmp_path / "pytest.ini").write_text("[pytest]\n")
    out = tmp_path / "junit.xml"

    result = _run_script(["--junitxml", str(out), "-k", "nonexistent"], cwd=tmp_path)

    assert result.returncode == 5
    suite = _testsuite(out)
    assert suite.get("tests") == "0"


@AGENT
def test_end_to_end_preserves_count_expanded_node_ids_across_batches(tmp_path):
    # Simulates PYTEST_ADDOPTS="--count=N". A repeated item's node ID (e.g.
    # "test_item_1[1-3]") only exists once pytest-repeat's collection hook
    # re-parametrizes the module, which needs --count active; dropping it
    # from the batch run env makes that ID a collection error ("not found"),
    # not a harmless no-op. Keeping --count active must not re-run items
    # beyond what each batch explicitly names.
    pytest.importorskip("pytest_repeat")
    _write_suite(tmp_path, n_tests=4)
    out = tmp_path / "junit.xml"

    env = dict(os.environ)
    env["PYTEST_ADDOPTS"] = "--count=3"
    result = _run_script(["--batch-size", "5", "--junitxml", str(out), "test_suite.py"], cwd=tmp_path, env=env)

    assert result.returncode == 0, result.stdout + result.stderr
    suite = _testsuite(out)
    assert suite.get("tests") == "12"  # 4 tests * 3 repeats, not 36


@AGENT
def test_end_to_end_v_flag_does_not_break_collection(tmp_path):
    # Regression test for the -v/-q cancellation described in
    # _args_for_collection: CI passes -v alongside this script's internal -q.
    _write_suite(tmp_path, n_tests=3)
    out = tmp_path / "junit.xml"

    result = _run_script(["-rsxXfE", "-v", "--junitxml", str(out), "test_suite.py"], cwd=tmp_path)

    assert result.returncode == 0, result.stdout + result.stderr
    suite = _testsuite(out)
    assert suite.get("tests") == "3"


@AGENT
def test_main_returns_int_exit_code_without_subprocess(tmp_path, monkeypatch):
    _write_suite(tmp_path, n_tests=2)
    monkeypatch.chdir(tmp_path)
    rc = main(["--batch-size", "1", "test_suite.py"])
    assert rc == 0
