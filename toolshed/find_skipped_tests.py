#!/usr/bin/env python3

# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

"""Report tests skipped across all test configurations.

Finds the latest ci.yml run triggered by a push to the main branch and the
latest ci-nightly.yml run, then reports tests that were skipped in every
wheel-based test configuration across both runs. sdist-based test jobs are
excluded. The report is written to stdout in markdown.

Requires `gh` on PATH and an authenticated GitHub CLI session.
"""

from __future__ import annotations

import argparse
import dataclasses
import json
import re
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path
from typing import Iterable

# Workflows to inspect, with the gh-api filter used to find the latest
# completed run for each. The order here determines report ordering.
WORKFLOWS: tuple[tuple[str, str], ...] = (
    ("ci.yml", "branch=main&event=push&status=completed&per_page=1"),
    ("ci-nightly.yml", "status=completed&per_page=1"),
)

ANSI_ESCAPE = re.compile(r"\x1B\[[0-9;]*[A-Za-z]")
PYTEST_NODE_ID = re.compile(r"tests/\S+\.py::\S+")
PYTEST_TEST_OUTCOME = re.compile(r"(tests/\S+\.py::\S+)\s+(PASSED|FAILED|ERROR|SKIPPED|XFAIL|XPASS)\b")

# GHA log format markers used to identify which test suite is active.
# `gh api` logs: ##[group]<step-name> opens a section, ##[endgroup] closes it.
GHA_GROUP = re.compile(r"##\[group\](.+)")
# `gh run view --log` logs: tab-separated  <job>\t<step>\t<timestamp>\t<content>
GHA_LOG_LINE = re.compile(r"^[^\t]+\t([^\t]+)\t[^\t]+\t(.*)", re.DOTALL)

# Map step-name substrings to canonical test suite names.
STEP_SUITE_PATTERNS: list[tuple[re.Pattern[str], str]] = [
    (re.compile(r"\brun-tests bindings\b"), "cuda_bindings"),
    (re.compile(r"\brun-tests core\b"), "cuda_core"),
    (re.compile(r"\brun-tests pathfinder\b"), "cuda_pathfinder"),
]


def step_name_to_suite(step_name: str) -> str:
    for pattern, suite in STEP_SUITE_PATTERNS:
        if pattern.search(step_name):
            return suite
    return ""


@dataclasses.dataclass(frozen=True)
class WorkflowRun:
    workflow_file: str
    run_id: int
    run_url: str
    head_sha: str
    created_at: str
    conclusion: str


@dataclasses.dataclass(frozen=True)
class ConfigResult:
    name: str
    workflow_file: str
    run_id: int
    job_ids: list[int]
    skipped: set[str]
    non_skipped: set[str]
    has_logs: bool
    has_pytest_activity: bool
    test_suites: dict[str, str] = dataclasses.field(default_factory=dict)


def run_gh(*args: str, check: bool = True) -> subprocess.CompletedProcess[str]:
    gh_exe = shutil.which("gh")
    if not gh_exe:
        raise RuntimeError("Could not find 'gh' executable in PATH")

    return subprocess.run(  # noqa: S603
        [gh_exe, *args],
        capture_output=True,
        text=True,
        check=check,
    )


def detect_repo() -> str:
    result = run_gh("repo", "view", "--json", "nameWithOwner", "--jq", ".nameWithOwner")
    repo = result.stdout.strip()
    if not repo:
        raise RuntimeError("Could not determine repo from `gh repo view`")
    return repo


def find_latest_run(repo: str, workflow_file: str, params: str) -> WorkflowRun | None:
    result = run_gh(
        "api",
        f"repos/{repo}/actions/workflows/{workflow_file}/runs?{params}",
        "--jq",
        ".workflow_runs[0] | {id, html_url, head_sha, created_at, conclusion}",
        check=False,
    )
    text = result.stdout.strip()
    if result.returncode != 0 or not text or text == "null":
        return None
    data = json.loads(text)
    return WorkflowRun(
        workflow_file=workflow_file,
        run_id=int(data["id"]),
        run_url=str(data.get("html_url") or ""),
        head_sha=str(data.get("head_sha") or ""),
        created_at=str(data.get("created_at") or ""),
        conclusion=str(data.get("conclusion") or ""),
    )


def load_run_jobs(repo: str, run_id: int) -> list[dict]:
    # --jq emits one compact JSON object per line across all pages.
    result = run_gh(
        "api",
        "--paginate",
        f"repos/{repo}/actions/runs/{run_id}/jobs?per_page=100",
        "--jq",
        ".jobs[] | @json",
    )
    jobs: list[dict] = []
    for line in result.stdout.splitlines():
        text = line.strip()
        if text:
            jobs.append(json.loads(text))
    return jobs


def is_test_config_job(job_name: str) -> bool:
    """True if the job appears to be a wheel-based test configuration."""
    lower = job_name.lower()
    if "sdist" in lower:
        return False
    if "check status" in lower or "check job status" in lower:
        return False
    return job_name.startswith(("Test ", "Nightly "))


def config_name_for_job(job_name: str) -> str:
    # Reusable-workflow jobs look like "<caller> / <called>"; group by caller.
    return job_name.split(" / ", 1)[0]


def group_jobs_into_configs(jobs: Iterable[dict]) -> dict[str, list[int]]:
    grouped: dict[str, list[int]] = {}
    for job in jobs:
        # Only consider jobs that actually executed and finished. Skipped or
        # cancelled jobs include the still-templated job name (e.g. when a
        # reusable workflow was skipped before matrix expansion), which would
        # otherwise show up as a bogus configuration.
        if job.get("conclusion") not in ("success", "failure"):
            continue
        name = str(job.get("name", ""))
        if not is_test_config_job(name):
            continue
        grouped.setdefault(config_name_for_job(name), []).append(int(job["id"]))
    return grouped


def download_job_log(repo: str, run_id: int, job_id: int, out_path: Path) -> bool:
    out_path.parent.mkdir(parents=True, exist_ok=True)

    api_result = run_gh("api", f"repos/{repo}/actions/jobs/{job_id}/logs", check=False)
    if api_result.returncode == 0:
        out_path.write_text(api_result.stdout, encoding="utf-8", errors="replace")
        return True

    view_result = run_gh("run", "view", str(run_id), "--job", str(job_id), "--log", check=False)
    if view_result.returncode == 0:
        out_path.write_text(view_result.stdout, encoding="utf-8", errors="replace")
        return True

    return False


def extract_test_status_sets(text: str) -> tuple[set[str], set[str], dict[str, str]]:
    """Parse pytest output and return (skipped, non_skipped, test_id->suite)."""
    skipped: set[str] = set()
    non_skipped: set[str] = set()
    test_suites: dict[str, str] = {}
    current_suite = ""

    for raw_line in text.splitlines():
        # Handle `gh run view --log` tab-separated format.
        # Each line: <job>\t<step>\t<timestamp>\t<content>
        if log_match := GHA_LOG_LINE.match(raw_line):
            suite = step_name_to_suite(log_match.group(1))
            if suite:
                current_suite = suite
            line = ANSI_ESCAPE.sub("", log_match.group(2)).replace("\\", "/")
        else:
            line = ANSI_ESCAPE.sub("", raw_line).replace("\\", "/")
            # Handle `gh api` log format: ##[group]<step-name> opens a section.
            if group_match := GHA_GROUP.search(line):
                suite = step_name_to_suite(group_match.group(1))
                if suite:
                    current_suite = suite
                continue

        # Parse per-test outcomes first so PASS/FAIL lines disqualify tests.
        for test_id, outcome in PYTEST_TEST_OUTCOME.findall(line):
            if outcome == "SKIPPED":
                skipped.add(test_id)
                if current_suite:
                    test_suites.setdefault(test_id, current_suite)
            else:
                non_skipped.add(test_id)

        if "SKIPPED" not in line:
            continue

        # Keep compatibility with summary-style SKIPPED lines that may still
        # include a node id but don't match the strict outcome pattern above.
        for test_id in PYTEST_NODE_ID.findall(line):
            skipped.add(test_id)
            if current_suite:
                test_suites.setdefault(test_id, current_suite)

    return skipped, non_skipped, test_suites


def analyze_config(
    config_name: str,
    run: WorkflowRun,
    job_ids: list[int],
    logs_root: Path,
    repo: str,
) -> ConfigResult:
    skipped_any: set[str] = set()
    non_skipped_any: set[str] = set()
    test_suites: dict[str, str] = {}
    have_any_log = False

    for job_id in job_ids:
        log_path = logs_root / f"{job_id}.log"
        if not download_job_log(repo, run.run_id, job_id, log_path):
            continue
        have_any_log = True
        text = log_path.read_text(encoding="utf-8", errors="replace")
        skipped_in_log, non_skipped_in_log, suites_in_log = extract_test_status_sets(text)
        skipped_any.update(skipped_in_log)
        non_skipped_any.update(non_skipped_in_log)
        for test_id, suite in suites_in_log.items():
            test_suites.setdefault(test_id, suite)
        log_path.unlink(missing_ok=True)

    # For sharded matrices a test may only appear in one log. Treat it as
    # config-skipped if it is skipped at least once and never non-skipped in
    # that config.
    skipped_for_config = skipped_any - non_skipped_any

    return ConfigResult(
        name=config_name,
        workflow_file=run.workflow_file,
        run_id=run.run_id,
        job_ids=job_ids,
        skipped=skipped_for_config,
        non_skipped=non_skipped_any,
        has_logs=have_any_log,
        has_pytest_activity=bool(skipped_any or non_skipped_any),
        test_suites=test_suites,
    )


def build_summary(runs: list[WorkflowRun], results: list[ConfigResult]) -> str:
    lines: list[str] = ["## Universally-skipped tests", ""]

    lines.append("### Analyzed runs")
    lines.append("")
    for run in runs:
        sha = run.head_sha[:7] if run.head_sha else "?"
        lines.append(
            f"- `{run.workflow_file}`: [run {run.run_id}]({run.run_url}) "
            f"(conclusion: {run.conclusion or 'unknown'}, {run.created_at}, sha `{sha}`)"
        )
    lines.append("")

    missing = [f"{r.workflow_file} / {r.name}" for r in results if not r.has_logs]
    # Configurations whose logs contained no pytest test outcomes (e.g.
    # nightly-numba-cuda, which runs numba's own test runner). These don't
    # have a meaningful "skipped" set and would otherwise empty the
    # intersection, so exclude them from it and call them out separately.
    inactive = [r for r in results if r.has_logs and not r.has_pytest_activity]
    active = [r for r in results if r.has_logs and r.has_pytest_activity]

    if not active:
        lines.append("_No pytest activity detected in any configuration._")
        return "\n".join(lines) + "\n"

    if missing:
        lines.append("_Warning: missing logs for configuration(s):_")
        for entry in missing:
            lines.append(f"- `{entry}`")
        lines.append("")

    if inactive:
        lines.append("_Configurations with no pytest activity (excluded from intersection):_")
        for r in inactive:
            lines.append(f"- `{r.workflow_file}` / `{r.name}`")
        lines.append("")

    # A test counts as universally-skipped if some configuration skipped it
    # AND no configuration ever recorded a non-skipped outcome for it. This
    # tolerates narrow configurations (e.g. nightly-pytorch) that only invoke
    # a subset of tests: such configs simply do not contribute either way for
    # tests they do not exercise.
    ever_skipped: set[str] = set()
    ever_non_skipped: set[str] = set()
    for result in active:
        ever_skipped |= result.skipped
        ever_non_skipped |= result.non_skipped
    universal_set = ever_skipped - ever_non_skipped

    if "tests/test_cuda.py::test_always_skip" not in universal_set:
        lines.append(
            "_Note: the test `tests/test_cuda.py::test_always_skip` is expected to be "
            "skipped in all configurations, but is missing._"
        )
        lines.append("")

    # Merge test->suite mappings across all configs (first to identify wins).
    test_suites: dict[str, str] = {}
    for result in results:
        for test_id, suite in result.test_suites.items():
            test_suites.setdefault(test_id, suite)

    def sort_key(test_id: str) -> tuple[bool, str, str]:
        suite = test_suites.get(test_id, "")
        return (not suite, suite, test_id)

    lines.append(f"### Configurations analyzed ({len(active)})")
    lines.append("")
    for result in active:
        lines.append(
            f"- `{result.workflow_file}` / `{result.name}` "
            f"({len(result.job_ids)} job(s), {len(result.skipped)} skipped)"
        )
    lines.append("")

    universal = sorted(universal_set, key=sort_key)
    lines.append(f"### Tests skipped in every configuration ({len(universal)})")
    lines.append("")
    if not universal:
        lines.append("_No tests were skipped in all configurations._")
    else:
        for test_id in universal:
            suite = test_suites.get(test_id, "")
            label = f"{suite}/{test_id}" if suite else test_id
            lines.append(f"- [ ] `{label}`")

    return "\n".join(lines) + "\n"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--repo",
        default=None,
        help="owner/repo (defaults to the repo of the current working directory)",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    repo = args.repo or detect_repo()

    runs: list[WorkflowRun] = []
    for workflow_file, params in WORKFLOWS:
        print(f"Finding latest run for {workflow_file}...", file=sys.stderr)
        run = find_latest_run(repo, workflow_file, params)
        if run is None:
            print(f"  No completed run found for {workflow_file}", file=sys.stderr)
            continue
        print(
            f"  Found run {run.run_id} ({run.created_at}, sha {run.head_sha[:7] or '?'})",
            file=sys.stderr,
        )
        runs.append(run)

    if not runs:
        print("No workflow runs found.", file=sys.stderr)
        return 1

    results: list[ConfigResult] = []
    with tempfile.TemporaryDirectory(prefix="universally-skipped-logs-") as tmp:
        logs_root = Path(tmp)
        for run in runs:
            print(f"Loading jobs for {run.workflow_file} run {run.run_id}...", file=sys.stderr)
            jobs = load_run_jobs(repo, run.run_id)
            grouped = group_jobs_into_configs(jobs)
            print(f"  Found {len(grouped)} test configuration(s)", file=sys.stderr)
            for config_name, job_ids in sorted(grouped.items()):
                print(
                    f"  Analyzing {config_name} ({len(job_ids)} job(s))...",
                    file=sys.stderr,
                )
                results.append(analyze_config(config_name, run, job_ids, logs_root, repo))

    summary = build_summary(runs, results)
    sys.stdout.write(summary)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
