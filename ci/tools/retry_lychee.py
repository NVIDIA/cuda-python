# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Retry complete lychee passes while retaining successful checks in its cache."""

from __future__ import annotations

import argparse
import os
import re
import shlex
import shutil
import subprocess
import time
from pathlib import Path

LINK_CHECK_FAILURE = 2


def read_report(path: Path) -> str:
    """Reject missing, malformed, or empty-check reports instead of accepting them."""
    text = path.read_text(encoding="utf-8")
    total = re.search(r"^\|[^|\n]*\bTotal\s*\|\s*(\d+)\s*\|", text, re.MULTILINE)
    if total is None:
        raise ValueError(f"Lychee report has no Total count: {path}")
    if int(total.group(1)) == 0:
        raise ValueError(f"Lychee checked no links: {path}")
    return text


def attempt_report(report: Path, attempt: int) -> Path:
    return report.with_name(f"{report.stem}-attempt-{attempt}{report.suffix}")


def finish(report_text: str | None, exit_codes: list[int], message: str) -> None:
    """Publish the final result without presenting recovered failures as current."""
    summary = f"## Documentation links\n\n{message}\n\n| Attempt | Exit code |\n| --- | --- |\n"
    summary += "".join(f"| {attempt} | {code} |\n" for attempt, code in enumerate(exit_codes, start=1))
    if report_text is not None:
        summary += f"\n### Final attempt\n\n{report_text}\n"
    else:
        summary += (
            "\nThe final attempt did not produce a link-check report. Earlier reports are retained as artifacts.\n"
        )
    print(message, flush=True)
    if summary_path := os.environ.get("GITHUB_STEP_SUMMARY"):
        with Path(summary_path).open("a", encoding="utf-8") as stream:
            stream.write(summary)


def retry_lychee(initial_exit_code: int, report: Path, max_attempts: int = 10) -> int:
    """Count the action's first pass toward the limit, and retry only link failures."""
    if initial_exit_code not in (0, 1, 2, 3):
        raise ValueError(f"Unexpected initial lychee exit code: {initial_exit_code}")
    if max_attempts < 1:
        raise ValueError("max_attempts must be positive")
    exit_codes = [initial_exit_code]
    if initial_exit_code not in (0, LINK_CHECK_FAILURE):
        finish(None, exit_codes, f"Lychee stopped on attempt 1 with exit code {initial_exit_code}; no retry.")
        return initial_exit_code

    report_text = read_report(report)
    shutil.copyfile(report, attempt_report(report, 1))
    if initial_exit_code == 0:
        finish(report_text, exit_codes, "Lychee passed on attempt 1.")
        return 0
    if max_attempts == 1:
        finish(report_text, exit_codes, "Lychee still has broken links after 1 attempt.")
        return LINK_CHECK_FAILURE

    raw_args = os.environ.get("LYCHEE_ARGS")
    if not raw_args or not (lychee_args := shlex.split(raw_args)):
        raise ValueError("LYCHEE_ARGS must contain the same arguments used by the first lychee pass")

    for attempt in range(2, max_attempts + 1):
        cooldown = min(60 * (attempt - 1), 120)
        print(
            f"Lychee attempt {attempt - 1} failed; waiting {cooldown}s before attempt {attempt}/{max_attempts}.",
            flush=True,
        )
        time.sleep(cooldown)
        current_report = attempt_report(report, attempt)
        result = subprocess.run(  # noqa: S603
            ["lychee", *lychee_args, "--mode", "task", "--format", "markdown", "--output", str(current_report)],  # noqa: S607
            check=False,
        )
        exit_codes.append(result.returncode)
        if result.returncode not in (0, LINK_CHECK_FAILURE):
            finish(
                None, exit_codes, f"Lychee stopped on attempt {attempt} with exit code {result.returncode}; no retry."
            )
            return result.returncode if result.returncode > 0 else 1
        report_text = read_report(current_report)
        shutil.copyfile(current_report, report)
        if result.returncode == 0:
            finish(
                report_text,
                exit_codes,
                f"Lychee passed on attempt {attempt}/{max_attempts}; successful checks were retained between passes.",
            )
            return 0

    finish(report_text, exit_codes, f"Lychee still has broken links after {max_attempts} attempts.")
    return LINK_CHECK_FAILURE


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--initial-exit-code", type=int, required=True)
    parser.add_argument("--report", type=Path, required=True)
    parser.add_argument("--max-attempts", type=int, default=10)
    args = parser.parse_args()
    try:
        return retry_lychee(args.initial_exit_code, args.report, args.max_attempts)
    except (OSError, ValueError) as error:
        print(f"::error::{error}", flush=True)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
