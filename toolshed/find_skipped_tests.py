#!/usr/bin/env python3

# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

"""Report tests skipped across all test configurations.

Finds the latest ci.yml run triggered by a push to the main branch and the
latest scheduled ci-nightly.yml run, then reports tests that were skipped in
every wheel-based test configuration across both runs. sdist-based test jobs
are excluded. Test ids are qualified by the package directory pytest ran in
(e.g. ``cuda_bindings_12/tests/...``), so the two bindings majors are tracked
separately. The report is written to stdout in markdown.

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
from collections.abc import Iterable

# Workflows to inspect, with the gh-api filter used to find the latest
# completed run for each. The order here determines report ordering.
WORKFLOWS: tuple[tuple[str, str], ...] = (
    ("ci.yml", "branch=main&event=push&status=completed&per_page=1"),
    ("ci-nightly.yml", "branch=main&event=schedule&status=completed&per_page=1"),
)

ANSI_ESCAPE = re.compile(r"\x1B\[[0-9;]*[A-Za-z]")

PYTEST_OUTCOMES = (
    "PASSED",
    "FAILED",
    "ERROR",
    "SKIPPED",
    "XFAIL",
    "XPASS",
    "SUBPASSED",
    "SUBFAILED",
    "SUBERROR",
    "SUBSKIPPED",
    "SUBXFAIL",
    "SUBXPASS",
)
# Free-threaded jobs interleave output from parallel threads: a node id can
# start mid-line or be printed twice in a row, several progress markers can
# precede the outcome, and the outcome can run into the next thread's output
# ("PASSEDPARALLEL PASSED", "PASSEDtests/..."). An id never extends across the
# start of another one, and one line can hold several results.
PYTEST_RESULT = re.compile(
    r"(tests/\S*?\.py::(?:(?!\stests/\S*?\.py::).)+?)\s+(?:\[\s*\d+%\]\s*)*(?:PARALLEL\s+)?"
    r"(" + "|".join(PYTEST_OUTCOMES) + r")(?=[\s(\[]|PARALLEL|tests/|$)"
)
# pytest prints "rootdir: <path>" when a session starts. The package directory
# in that path names the suite (and bindings major) of the results that follow;
# the group is empty for a third-party suite, e.g. numba-cuda's in nightly CI.
PYTEST_ROOTDIR = re.compile(r"\brootdir: (?:\S*?/(cuda_bindings_\d+|cuda_core|cuda_pathfinder|cuda_python)(?=/|\s|$))?")
# Parameter id pytest uses for an empty parameter set; not a real test.
EMPTY_PARAMETER_SET = "[NOTSET]"
# Expected to be skipped everywhere; its absence means a suite did not run.
CANARY_TEST = "/tests/test_cuda.py::test_always_skip"

# GHA log format markers.
# `gh api` logs: "##[group]Run <command>" opens each step.
GHA_STEP_START = re.compile(r"##\[group\]Run ")
# `gh run view --log` logs: tab-separated  <job>\t<step>\t<timestamp>\t<content>
GHA_LOG_LINE = re.compile(r"^[^\t]+\t([^\t]+)\t[^\t]+\t(.*)", re.DOTALL)


@dataclasses.dataclass(frozen=True)
class WorkflowRun:
    workflow_file: str
    run_id: int
    run_url: str
    head_sha: str
    created_at: str
    conclusion: str


@dataclasses.dataclass(frozen=True)
class LogResults:
    """Pytest results parsed from one job log, keyed by suite-qualified node id."""

    skipped: set[str]
    non_skipped: set[str]
    # Number of pytest sessions started in the log, for any project.
    pytest_sessions: int


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
    # Jobs whose log shows no pytest session at all, e.g. a job that failed
    # while installing. Tests that only run in such a job look never-run.
    jobs_without_pytest: list[str] = dataclasses.field(default_factory=list)


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
        print("Error finding latest run:", result.stderr.strip(), file=sys.stderr)
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
    # The matrix-setup job of each reusable test workflow runs no tests.
    if job_name.endswith(" / compute-matrix"):
        return False
    return job_name.startswith(("Test ", "Nightly "))


def config_name_for_job(job_name: str) -> str:
    # Reusable-workflow jobs look like "<caller> / <called>"; group by caller.
    return job_name.split(" / ", 1)[0]


def group_jobs_into_configs(jobs: Iterable[dict]) -> dict[str, list[tuple[int, str]]]:
    """Group finished test jobs by configuration, as (job id, job name) pairs."""
    grouped: dict[str, list[tuple[int, str]]] = {}
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
        grouped.setdefault(config_name_for_job(name), []).append((int(job["id"]), name))
    return grouped


def fetch_job_log(repo: str, run_id: int, job_id: int) -> str | None:
    """Return a job's log, trying `gh api` first and `gh run view` second."""
    for args in (
        ("api", f"repos/{repo}/actions/jobs/{job_id}/logs"),
        ("run", "view", str(run_id), "--job", str(job_id), "--log"),
    ):
        result = run_gh(*args, check=False)
        if result.returncode == 0:
            return result.stdout
    return None


def parse_log(text: str) -> LogResults:
    """Parse the pytest results in one job log.

    Each node id is qualified by the package directory of the pytest session
    it ran in (e.g. ``cuda_bindings_12/tests/test_x.py::test_y``), so the same
    test in two bindings roots is tracked separately.
    """
    skipped: set[str] = set()
    non_skipped: set[str] = set()
    passed: set[str] = set()
    subtest_seen: set[str] = set()
    sessions = 0
    suite = ""
    step = None

    for line in text.splitlines():
        # `gh run view --log` lines are <job>\t<step>\t<timestamp>\t<content>.
        if log_match := GHA_LOG_LINE.match(line):
            if log_match.group(1) != step:
                step, suite = log_match.group(1), ""
            line = log_match.group(2)
        line = ANSI_ESCAPE.sub("", line).replace("\\", "/")

        if GHA_STEP_START.search(line):
            suite = ""
            continue
        if rootdir := PYTEST_ROOTDIR.search(line):
            sessions += 1
            suite = rootdir.group(1) or ""
            continue
        # Only results from a cuda-python package's own pytest session count.
        if not suite:
            continue

        for result in PYTEST_RESULT.finditer(line):
            node_id, outcome = result.groups()
            if node_id.endswith(EMPTY_PARAMETER_SET):
                continue
            test_id = f"{suite}/{node_id}"
            if outcome in ("SKIPPED", "SUBSKIPPED"):
                skipped.add(test_id)
            elif outcome == "PASSED":
                passed.add(test_id)
            else:
                non_skipped.add(test_id)
            if outcome.startswith("SUB"):
                subtest_seen.add(test_id)

    # Pytest reports a passing parent after its subtests even when every
    # subtest skipped. Only treat that parent pass as execution evidence when
    # the test did not emit subtest outcomes of its own.
    non_skipped.update(passed - subtest_seen)

    return LogResults(skipped=skipped, non_skipped=non_skipped, pytest_sessions=sessions)


def analyze_config(config_name: str, run: WorkflowRun, jobs: list[tuple[int, str]], repo: str) -> ConfigResult:
    skipped_any: set[str] = set()
    non_skipped_any: set[str] = set()
    jobs_without_pytest: list[str] = []
    have_any_log = False

    for job_id, job_name in jobs:
        text = fetch_job_log(repo, run.run_id, job_id)
        if text is None:
            continue
        have_any_log = True
        results = parse_log(text)
        skipped_any.update(results.skipped)
        non_skipped_any.update(results.non_skipped)
        if results.pytest_sessions == 0:
            jobs_without_pytest.append(job_name)

    # For sharded matrices a test may only appear in one log. Treat it as
    # config-skipped if it is skipped at least once and never non-skipped in
    # that config.
    skipped_for_config = skipped_any - non_skipped_any

    return ConfigResult(
        name=config_name,
        workflow_file=run.workflow_file,
        run_id=run.run_id,
        job_ids=[job_id for job_id, _ in jobs],
        skipped=skipped_for_config,
        non_skipped=non_skipped_any,
        has_logs=have_any_log,
        has_pytest_activity=bool(skipped_any or non_skipped_any),
        jobs_without_pytest=jobs_without_pytest,
    )


def universally_skipped(active: Iterable[ConfigResult]) -> set[str]:
    """Tests that some configuration skipped and no configuration ran.

    This tolerates narrow configurations (e.g. nightly-pytorch) that only
    invoke a subset of tests: such configs do not contribute either way for
    tests they do not exercise.
    """
    ever_skipped: set[str] = set()
    ever_non_skipped: set[str] = set()
    for result in active:
        ever_skipped |= result.skipped
        ever_non_skipped |= result.non_skipped
    return ever_skipped - ever_non_skipped


def bullet_list(header: str, entries: Iterable[str]) -> list[str]:
    """Markdown lines for a header and one code-formatted bullet per entry; none if empty."""
    bullets = [f"- `{entry}`" for entry in entries]
    if not bullets:
        return []
    return [header, *bullets, ""]


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

    # Configurations whose logs contained no cuda-python test outcomes (e.g.
    # nightly-numba-cuda, which runs only numba-cuda's own tests). These don't
    # have a meaningful "skipped" set and would otherwise empty the
    # intersection, so exclude them from it and call them out separately.
    inactive = [r for r in results if r.has_logs and not r.has_pytest_activity]
    active = [r for r in results if r.has_logs and r.has_pytest_activity]

    if not active:
        lines.append("_No pytest activity detected in any configuration._")
        return "\n".join(lines) + "\n"

    lines += bullet_list(
        "_Warning: missing logs for configuration(s):_",
        (f"{r.workflow_file} / {r.name}" for r in results if not r.has_logs),
    )
    lines += bullet_list(
        "_Configurations with no pytest activity (excluded from intersection):_",
        (f"{r.workflow_file} / {r.name}" for r in inactive),
    )
    lines += bullet_list(
        "_Warning: pytest never ran in these job(s) (for example, they failed during setup). "
        "Tests that run only in them may be listed below by mistake:_",
        (f"{r.workflow_file} / {job}" for r in results for job in r.jobs_without_pytest),
    )

    universal_set = universally_skipped(active)

    if not any(test_id.endswith(CANARY_TEST) for test_id in universal_set):
        lines.append(
            "_Note: `tests/test_cuda.py::test_always_skip` is expected to be "
            "skipped in all configurations, but is missing._"
        )
        lines.append("")

    lines.append(f"### Configurations analyzed ({len(active)})")
    lines.append("")
    for result in active:
        lines.append(
            f"- `{result.workflow_file}` / `{result.name}` "
            f"({len(result.job_ids)} job(s), {len(result.skipped)} skipped)"
        )
    lines.append("")

    universal = sorted(universal_set)
    lines.append(f"### Tests skipped in every configuration ({len(universal)})")
    lines.append("")
    if not universal:
        lines.append("_No tests were skipped in all configurations._")
    else:
        for test_id in universal:
            lines.append(f"- [ ] `{test_id}`")

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
    for run in runs:
        print(f"Loading jobs for {run.workflow_file} run {run.run_id}...", file=sys.stderr)
        grouped = group_jobs_into_configs(load_run_jobs(repo, run.run_id))
        print(f"  Found {len(grouped)} test configuration(s)", file=sys.stderr)
        for config_name, jobs in sorted(grouped.items()):
            print(f"  Analyzing {config_name} ({len(jobs)} job(s))...", file=sys.stderr)
            results.append(analyze_config(config_name, run, jobs, repo))

    summary = build_summary(runs, results)
    sys.stdout.write(summary)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
