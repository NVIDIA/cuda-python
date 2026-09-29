# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Write a plain-English pull-request comment from the JUnit files of a CI run.

Reads every ``*.xml`` under the test-results directory that the final CI job
downloads, one subdirectory per test job, and writes a short Markdown comment:
how many configurations ran, which tests did not pass and where, the execution
totals, and a link to the run's Summary page where the full reports live.

Usage::

    python ci/tools/junit_comment.py --results test-results \\
        --commit <sha> --run-url <url> --output comment.md
"""

from __future__ import annotations

import argparse
import sys
import xml.etree.ElementTree as ET
from dataclasses import dataclass, field
from pathlib import Path

ARTIFACT_PREFIX = "test-results-"
STANDARD_MODE_PREFIX = "standard-"
MAX_LISTED = 25


@dataclass
class Totals:
    passed: int = 0
    failed: int = 0
    errors: int = 0
    skipped: int = 0


@dataclass
class Results:
    files: int = 0
    configurations: dict[str, Totals] = field(default_factory=dict)
    # test id -> {configuration: "failed" | "error"} for tests that did not pass somewhere
    not_passed: dict[str, dict[str, str]] = field(default_factory=dict)


def configuration_label(results_dir: Path, xml_path: Path) -> str:
    """The test job's configuration, taken from the artifact directory name."""
    parts = xml_path.relative_to(results_dir).parts
    name = parts[0] if len(parts) > 1 else ""
    if name.startswith(ARTIFACT_PREFIX):
        name = name[len(ARTIFACT_PREFIX) :]
    if name.startswith(STANDARD_MODE_PREFIX):
        name = name[len(STANDARD_MODE_PREFIX) :]
    return name or "(unknown configuration)"


def outcome(case: ET.Element) -> str:
    """JUnit: <error> is an exception outside the test body, <failure> one inside it."""
    tags = {child.tag for child in case}
    if "error" in tags:
        return "error"
    if "failure" in tags:
        return "failed"
    if "skipped" in tags:
        return "skipped"
    return "passed"


def test_id(case: ET.Element) -> str:
    classname = case.get("classname") or ""
    name = case.get("name") or "(unnamed test)"
    return f"{classname}::{name}" if classname else name


def collect(results_dir: Path) -> Results:
    results = Results()
    for xml_path in sorted(results_dir.rglob("*.xml")):
        label = configuration_label(results_dir, xml_path)
        totals = results.configurations.setdefault(label, Totals())
        results.files += 1
        # The files are pytest's own output from this run, not untrusted input.
        for case in ET.parse(xml_path).getroot().iter("testcase"):  # noqa: S314
            state = outcome(case)
            if state == "passed":
                totals.passed += 1
            elif state == "failed":
                totals.failed += 1
            elif state == "error":
                totals.errors += 1
            else:
                totals.skipped += 1
            if state in ("failed", "error"):
                results.not_passed.setdefault(test_id(case), {})[label] = state
    return results


def plural(count: int, word: str) -> str:
    return f"{count:,} {word}" if count == 1 else f"{count:,} {word}s"


def render(results: Results, commit: str, run_url: str) -> str:
    lines = [f"## Test results for commit {commit[:7]}", ""]
    if results.files == 0:
        lines += ["No test result files were found for this run.", "", f"Run: {run_url}"]
        return "\n".join(lines) + "\n"

    n_conf = len(results.configurations)
    configurations = sorted(results.configurations)
    lines.append(f"{plural(n_conf, 'configuration')} ran the test suites: {'; '.join(configurations)}.")
    lines.append("")

    passed = sum(t.passed for t in results.configurations.values())
    failed = sum(t.failed for t in results.configurations.values())
    errors = sum(t.errors for t in results.configurations.values())
    skipped = sum(t.skipped for t in results.configurations.values())

    if not results.not_passed:
        lines.append(
            f"No test failed. Counting every test execution across all configurations: "
            f"{passed:,} passed, {skipped:,} skipped."
        )
    else:
        errored = {t for t, where in results.not_passed.items() if "error" in where.values()}
        only_failed = len(results.not_passed) - len(errored)
        lines.append(
            f"{plural(len(results.not_passed), 'test')} did not pass in at least one configuration. "
            f"{only_failed} failed inside the test body (an assertion or an exception). "
            f"{len(errored)} hit an error outside the test body (fixture setup or teardown, or collection)."
        )
        lines.append("")
        lines.append(
            f"Counting every test execution across all configurations: "
            f"{passed:,} passed, {failed:,} failed, {errors:,} errored, {skipped:,} skipped."
        )
        lines.append("")
        lines.append("Tests that did not pass:")
        lines.append("")
        listed = sorted(results.not_passed)
        for name in listed[:MAX_LISTED]:
            where = results.not_passed[name]
            state = "error" if "error" in where.values() else "failed"
            if n_conf > 1 and len(where) == n_conf:
                place = f"in all {n_conf} configurations"
            else:
                place = "in " + ", ".join(sorted(where))
            lines.append(f"- `{name}`: {state} {place}")
        if len(listed) > MAX_LISTED:
            lines.append(f"- and {len(listed) - MAX_LISTED:,} more")

    lines.append("")
    lines.append(f"The per-configuration tables and the failing-test details are on the run's Summary page: {run_url}")
    lines.append("")
    lines.append(
        "A test *fails* when an assertion or exception happens inside the test body. "
        "A test *errors* when the exception happens outside it: fixture setup or teardown, or collection."
    )
    return "\n".join(lines) + "\n"


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument(
        "--results", required=True, type=Path, help="directory holding the downloaded test-results-* artifacts"
    )
    parser.add_argument("--commit", required=True, help="commit SHA the results belong to")
    parser.add_argument("--run-url", required=True, help="URL of the workflow run whose Summary page holds the reports")
    parser.add_argument("--output", required=True, type=Path, help="file to write the Markdown comment to")
    args = parser.parse_args(argv)

    text = render(collect(args.results), args.commit, args.run_url)
    args.output.write_text(text, encoding="utf-8")
    sys.stdout.write(text)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
