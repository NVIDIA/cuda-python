#!/usr/bin/env python3

# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

"""Report tests skipped across all wheel test configurations.

The script can run in GitHub Actions (using GITHUB_REPOSITORY/GITHUB_RUN_ID
and GITHUB_STEP_SUMMARY) or locally by passing --repo and --run-id.
"""

from __future__ import annotations

import argparse
import contextlib
import dataclasses
import json
import os
import re
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Iterable

CONFIG_PATTERNS = {
    "test-linux-64": r"^Test linux-64 / ",
    "test-linux-aarch64": r"^Test linux-aarch64 / ",
    "test-windows": r"^Test (win-64|windows) / ",
}

INDEX_FILENAME = "job_index.json"

ANSI_ESCAPE = re.compile(r"\x1B\[[0-9;]*[A-Za-z]")
PYTEST_NODE_ID = re.compile(r"tests/\S+\.py::\S+")
PYTEST_TEST_OUTCOME = re.compile(r"(tests/\S+\.py::\S+)\s+(PASSED|FAILED|ERROR|SKIPPED|XFAIL|XPASS)\b")


@dataclasses.dataclass(frozen=True)
class ConfigResult:
    name: str
    job_ids: list[int]
    skipped: set[str]
    has_logs: bool
    # test_id -> suite name (e.g. "cuda_bindings"), empty string if unknown
    test_suites: dict[str, str] = dataclasses.field(default_factory=dict)


@dataclasses.dataclass(frozen=True)
class ConfigLogs:
    name: str
    job_ids: list[int]
    log_paths: list[Path]
    # job_id -> suite name extracted from the job name
    job_names: dict[int, str] = dataclasses.field(default_factory=dict)


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


def load_run_jobs(repo: str, run_id: str) -> list[dict]:
    # --jq emits one compact JSON object per line across all pages.
    result = run_gh(
        "api",
        "--paginate",
        f"repos/{repo}/actions/runs/{run_id}/jobs?per_page=100",
        "--jq",
        ".jobs[] | @json",
    )
    jobs = []
    for line in result.stdout.splitlines():
        text = line.strip()
        if not text:
            continue
        jobs.append(json.loads(text))
    return jobs


def download_job_log(repo: str, run_id: str, job_id: int, out_path: Path) -> bool:
    out_path.parent.mkdir(parents=True, exist_ok=True)

    api_result = run_gh("api", f"repos/{repo}/actions/jobs/{job_id}/logs", check=False)
    if api_result.returncode == 0:
        out_path.write_text(api_result.stdout, encoding="utf-8", errors="replace")
        return True

    view_result = run_gh("run", "view", run_id, "--job", str(job_id), "--log", check=False)
    if view_result.returncode == 0:
        out_path.write_text(view_result.stdout, encoding="utf-8", errors="replace")
        return True

    return False


def extract_test_status_sets(text: str) -> tuple[set[str], set[str]]:
    skipped: set[str] = set()
    non_skipped: set[str] = set()

    for raw_line in text.splitlines():
        line = ANSI_ESCAPE.sub("", raw_line).replace("\\", "/")

        # Parse per-test outcomes first so PASS/FAIL lines disqualify tests.
        for test_id, outcome in PYTEST_TEST_OUTCOME.findall(line):
            if outcome == "SKIPPED":
                skipped.add(test_id)
            else:
                non_skipped.add(test_id)

        if "SKIPPED" not in line:
            continue

        # Keep compatibility with summary-style SKIPPED lines that may still
        # include a node id but don't match the strict outcome pattern above.
        for test_id in PYTEST_NODE_ID.findall(line):
            skipped.add(test_id)

    return skipped, non_skipped


def extract_suite_name(job_name: str, config_name: str) -> str:
    """Return the test suite portion of a job name (first word after the config prefix)."""
    pattern = CONFIG_PATTERNS.get(config_name, "")
    if pattern:
        match = re.match(pattern, job_name)
        if match:
            remainder = job_name[match.end() :]
            parts = remainder.split()
            return parts[0] if parts else job_name
    return job_name


def save_job_index(logs_root: Path, index: dict[str, dict[str, str]]) -> None:
    (logs_root / INDEX_FILENAME).write_text(json.dumps(index, indent=2), encoding="utf-8")


def load_job_index(logs_root: Path) -> dict[str, dict[str, str]]:
    index_path = logs_root / INDEX_FILENAME
    if index_path.exists():
        return json.loads(index_path.read_text(encoding="utf-8"))
    return {}


def match_job_ids(jobs: Iterable[dict], pattern: str) -> list[int]:
    regex = re.compile(pattern)
    return [int(job["id"]) for job in jobs if regex.search(str(job.get("name", "")))]


def discover_config_logs(logs_root: Path) -> list[ConfigLogs]:
    configs: list[ConfigLogs] = []
    index = load_job_index(logs_root)

    for config in CONFIG_PATTERNS:
        config_dir = logs_root / config
        log_paths = sorted(config_dir.glob("*.log")) if config_dir.exists() else []
        job_ids: list[int] = []
        job_names: dict[int, str] = {}
        config_index = index.get(config, {})

        for log_path in log_paths:
            with contextlib.suppress(ValueError):
                job_id = int(log_path.stem)
                job_ids.append(job_id)
                suite = config_index.get(str(job_id), "")
                if suite:
                    job_names[job_id] = suite

        configs.append(ConfigLogs(name=config, job_ids=job_ids, log_paths=log_paths, job_names=job_names))

    return configs


def download_config_logs(jobs: list[dict], repo: str, run_id: str, logs_root: Path) -> list[ConfigLogs]:
    configs: list[ConfigLogs] = []
    index: dict[str, dict[str, str]] = {}

    for config, pattern in CONFIG_PATTERNS.items():
        config_dir = logs_root / config
        job_ids = match_job_ids(jobs, pattern)
        log_paths: list[Path] = []

        # Build job_id -> suite_name from job metadata before downloading logs.
        regex = re.compile(pattern)
        job_names: dict[int, str] = {}
        for job in jobs:
            job_name = str(job.get("name", ""))
            if not regex.search(job_name):
                continue
            job_id = int(job["id"])
            if job_id in job_ids:
                job_names[job_id] = extract_suite_name(job_name, config)

        for job_id in job_ids:
            log_path = config_dir / f"{job_id}.log"
            if not log_path.exists() and not download_job_log(repo, run_id, job_id, log_path):
                continue
            log_paths.append(log_path)

        configs.append(ConfigLogs(name=config, job_ids=job_ids, log_paths=log_paths, job_names=job_names))
        index[config] = {str(jid): name for jid, name in job_names.items()}

    save_job_index(logs_root, index)
    return configs


def analyze_config_logs(config_logs: list[ConfigLogs]) -> list[ConfigResult]:
    results: list[ConfigResult] = []

    for config in config_logs:
        skipped_any: set[str] = set()
        non_skipped_any: set[str] = set()
        test_suites: dict[str, str] = {}

        for log_path in config.log_paths:
            text = log_path.read_text(encoding="utf-8", errors="replace")

            skipped_in_log, non_skipped_in_log = extract_test_status_sets(text)
            skipped_any.update(skipped_in_log)
            non_skipped_any.update(non_skipped_in_log)

            # Associate skipped test IDs with the suite derived from the job name.
            with contextlib.suppress(ValueError):
                job_id = int(log_path.stem)
                suite = config.job_names.get(job_id, "")
                if suite:
                    for test_id in skipped_in_log:
                        test_suites.setdefault(test_id, suite)

        # For sharded matrices, a test may only appear in one log. Treat it as
        # config-skipped if it is skipped at least once and never non-skipped
        # (passed/failed/error/xpass/xfail) in that config.
        skipped_for_config = skipped_any - non_skipped_any

        results.append(
            ConfigResult(
                name=config.name,
                job_ids=config.job_ids,
                skipped=skipped_for_config,
                has_logs=bool(config.log_paths),
                test_suites=test_suites,
            )
        )

    return results


def build_summary(results: list[ConfigResult]) -> str:
    lines = ["## Universally-skipped tests", ""]

    available = [r for r in results if r.job_ids or r.has_logs]
    missing = [r.name for r in results if not (r.job_ids or r.has_logs)]

    if not available:
        lines.append("_No test job logs found in this run._")
        return "\n".join(lines) + "\n"

    if missing:
        lines.append(f"_Warning: missing logs for configuration(s): {' '.join(missing)}_")
        lines.append("")

    intersection: set[str] | None = None
    for result in results:
        if intersection is None:
            intersection = set(result.skipped)
            continue
        intersection &= result.skipped

    if intersection is None or "tests/test_cuda.py::test_always_skip" not in intersection:
        lines.append(
            "_Note: the test `tests/test_cuda.py::test_always_skip` is expected to be skipped in all configurations, but is missing._"
        )

    # Merge test->suite mappings across all configs (first one seen wins).
    test_suites: dict[str, str] = {}
    for result in results:
        for test_id, suite in result.test_suites.items():
            test_suites.setdefault(test_id, suite)

    universal = sorted(intersection or set())
    lines.append(f"Tests skipped across wheel test configurations ({len(results)}):")
    lines.append("")
    if not universal:
        lines.append("_No tests were skipped in all configurations._")
    else:
        for test in universal:
            suite = test_suites.get(test, "")
            label = f"{suite}/{test}" if suite else test
            lines.append(f"- [ ] `{label}`")

    return "\n".join(lines) + "\n"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo", default=os.environ.get("GITHUB_REPOSITORY"), help="owner/repo")
    parser.add_argument("--run-id", default=os.environ.get("GITHUB_RUN_ID"), help="GitHub Actions run id")
    parser.add_argument(
        "--summary-path",
        default=os.environ.get("GITHUB_STEP_SUMMARY"),
        help="Path to write markdown summary (stdout if omitted)",
    )
    parser.add_argument(
        "--logs-dir",
        default=None,
        help="Directory to store downloaded logs (defaults to temporary CI-style dir)",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    logs_root = Path(args.logs_dir) if args.logs_dir else Path(".tmp-universally-skipped-logs")

    if args.logs_dir and logs_root.exists():
        if not logs_root.is_dir():
            print(f"--logs-dir path exists but is not a directory: {logs_root}", file=sys.stderr)
            return 2
        print(f"Using existing logs in {logs_root}; skipping log downloads")
        config_logs = discover_config_logs(logs_root)
    else:
        if not args.repo or not args.run_id:
            print("--repo and --run-id are required (or set GITHUB_REPOSITORY/GITHUB_RUN_ID)", file=sys.stderr)
            return 2

        logs_root.mkdir(parents=True, exist_ok=True)
        jobs = load_run_jobs(args.repo, str(args.run_id))
        config_logs = download_config_logs(jobs=jobs, repo=args.repo, run_id=str(args.run_id), logs_root=logs_root)

    results = analyze_config_logs(config_logs)

    for result in results:
        print(f"{result.name}: {len(result.skipped)} skipped tests")

    summary = build_summary(results)
    if args.summary_path:
        Path(args.summary_path).write_text(summary, encoding="utf-8")
    else:
        print(summary)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
