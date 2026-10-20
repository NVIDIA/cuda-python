# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Check documentation links or warm caches through the configured pre-commit hooks."""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import uuid
from pathlib import Path

from retry_lychee import LINK_CHECK_FAILURE, read_report, retry_lychee

ROOT = Path(__file__).resolve().parents[2]


def collect_reports(directory: Path, output: Path, precommit_code: int) -> int:
    """Distinguish native link failures from pre-commit setup or hook failures."""
    statuses = sorted(directory.glob("*.status.json"))
    reports = set(directory.glob("*.json")) - set(statuses)
    expected = {path.with_name(path.name.removesuffix(".status.json") + ".json") for path in statuses}
    if not statuses:
        raise ValueError("The configured Lychee hook produced no native status; pre-commit setup or execution failed")
    merged = {"total": 0, "errors": 0, "timeouts": 0, "error_map": {}, "timeout_map": {}}
    native_codes = []
    for status in statuses:
        metadata = json.loads(status.read_text(encoding="utf-8"))
        if not isinstance(metadata, dict) or type(metadata.get("exit_code")) is not int:
            raise ValueError(f"Invalid native Lychee status: {status}")
        code = metadata["exit_code"]
        if code not in (0, LINK_CHECK_FAILURE):
            raise ValueError(f"Native Lychee execution failed with exit code {code}: {status}")
        native_codes.append(code)
        report = status.with_name(status.name.removesuffix(".status.json") + ".json")
        # A pre-commit chunk can contain no links; the complete scan must not be empty.
        read_report(report, code, allow_empty=True)
        data = json.loads(report.read_text(encoding="utf-8"))
        for name in ("total", "errors", "timeouts"):
            merged[name] += data[name]
        for name in ("error_map", "timeout_map"):
            for source, responses in data[name].items():
                merged[name].setdefault(source, []).extend(responses)
    if reports != expected:
        raise ValueError("Native Lychee reports and completed invocation statuses do not match")
    code = LINK_CHECK_FAILURE if LINK_CHECK_FAILURE in native_codes else 0
    if precommit_code != int(code != 0):
        raise ValueError(f"pre-commit exit {precommit_code} disagrees with native Lychee exit {code}")
    output.write_text(json.dumps(merged), encoding="utf-8")
    read_report(output, code)
    return code


def run_hook(kind: str, report: Path) -> int:
    """Run the real hook, retaining every native chunk report for diagnostics."""
    directory = report.parent / f"{report.stem}-parts" / uuid.uuid4().hex
    directory.mkdir(parents=True)
    hook = "lychee" if kind == "authored" else "lychee-rendered"
    environment = os.environ.copy()
    environment["LYCHEE_REPORT_DIR"] = str(directory)
    environment["PYTHONUTF8"] = "1"
    environment["SKIP"] = ",".join(
        name.strip() for name in environment.get("SKIP", "").split(",") if name.strip() and name.strip() != hook
    )
    command = [sys.executable, "-m", "pre_commit", "run", hook, "--all-files", "--show-diff-on-failure"]
    if kind == "rendered":
        command.extend(["--hook-stage", "manual"])
    result = subprocess.run(command, cwd=ROOT, env=environment, check=False)  # noqa: S603
    return collect_reports(directory, report, result.returncode)


def run_checks(kind: str, report: Path, max_attempts: int = 3, *, policy: str = "check") -> int:
    if max_attempts < 1:
        raise ValueError("max-attempts must be positive")
    report = report.resolve()
    report.parent.mkdir(parents=True, exist_ok=True)
    return retry_lychee(
        run_hook(kind, report),
        report,
        max_attempts,
        policy=policy,
        run_attempt=lambda path: run_hook(kind, path),
    )


def main(arguments: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("kind", choices=("authored", "rendered"))
    parser.add_argument("--policy", choices=("cache-warm", "check", "allow-429"), default="check")
    parser.add_argument("--max-attempts", type=int, default=3)
    parser.add_argument("--report", type=Path, required=True)
    args = parser.parse_args(arguments)
    try:
        return run_checks(args.kind, args.report, args.max_attempts, policy=args.policy)
    except (OSError, ValueError) as error:
        print(f"::error::{error}", flush=True)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
