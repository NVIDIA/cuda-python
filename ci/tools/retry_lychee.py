# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Retry transient lychee failures while retaining successful checks in its cache."""

from __future__ import annotations

import html
import json
import os
import re
import shutil
import time
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from urllib.parse import urlsplit

LINK_CHECK_FAILURE = 2
MAX_RATE_LIMITED_URLS = 10


@dataclass(frozen=True)
class Failure:
    source: str
    url: str
    text: str
    details: str | None
    code: int | None
    timed_out: bool

    @property
    def is_http(self) -> bool:
        try:
            parsed = urlsplit(self.url)
            return parsed.scheme in ("http", "https") and bool(parsed.hostname)
        except ValueError:
            return False

    @property
    def rate_limited(self) -> bool:
        return self.is_http and not self.timed_out and self.code == 429

    @property
    def transient(self) -> bool:
        return self.is_http and (
            self.timed_out or self.code in (408, 429) or (self.code is not None and 500 <= self.code <= 599)
        )


@dataclass(frozen=True)
class Report:
    total: int
    errors: int
    timeouts: int
    failures: tuple[Failure, ...]

    @property
    def transient(self) -> bool:
        return bool(self.failures) and all(failure.transient for failure in self.failures)


def read_report(path: Path, exit_code: int, *, allow_empty: bool = False) -> Report:
    """Validate the pinned lychee JSON schema and its agreement with the exit code."""
    data = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(data, dict):
        raise ValueError(f"Lychee report must be a JSON object: {path}")
    counts = {}
    for name in ("total", "errors", "timeouts"):
        value = data.get(name)
        if type(value) is not int or value < 0:
            raise ValueError(f"Lychee report {name} must be a nonnegative integer: {path}")
        counts[name] = value
    if counts["total"] == 0 and not allow_empty:
        raise ValueError(f"Lychee checked no links: {path}")
    if counts["errors"] + counts["timeouts"] > counts["total"]:
        raise ValueError(f"Lychee failure counts exceed the total: {path}")

    failures = []
    for name, timed_out, count in (("error_map", False, counts["errors"]), ("timeout_map", True, counts["timeouts"])):
        mapping = data.get(name)
        if not isinstance(mapping, dict):
            raise ValueError(f"Lychee report {name} must be an object: {path}")
        entries = []
        for source, responses in mapping.items():
            if not isinstance(source, str) or not source or not isinstance(responses, list):
                raise ValueError(f"Lychee report {name} must map source names to response lists: {path}")
            for response in responses:
                if not isinstance(response, dict) or not isinstance(response.get("url"), str) or not response["url"]:
                    raise ValueError(f"Lychee report has a malformed failure URL: {path}")
                status = response.get("status")
                if not isinstance(status, dict) or not isinstance(status.get("text"), str) or not status["text"]:
                    raise ValueError(f"Lychee report has a malformed failure status: {path}")
                code = status.get("code")
                details = status.get("details")
                if "code" in status:
                    if type(code) is not int or not 100 <= code <= 999:
                        raise ValueError(f"Lychee report has a malformed HTTP status code: {path}")
                elif not isinstance(details, str):
                    raise ValueError(f"Lychee report non-HTTP status must contain string details: {path}")
                if "details" in status and not isinstance(details, str):
                    raise ValueError(f"Lychee report status details must be a string: {path}")
                entries.append(Failure(source, response["url"], status["text"], details, code, timed_out))
        # Counters count occurrences; response maps deduplicate equivalent checks.
        if bool(count) != bool(entries):
            raise ValueError(f"Lychee report {name} disagrees with its failure count: {path}")
        failures.extend(entries)

    if (exit_code == 0) != (not failures):
        raise ValueError(f"Lychee report failures disagree with exit code {exit_code}: {path}")
    return Report(counts["total"], counts["errors"], counts["timeouts"], tuple(failures))


def escape_markdown(text: str) -> str:
    text = html.escape(text).replace("\r", " ").replace("\n", " ")
    return re.sub(r"([\\`*_~\[\]|])", r"\\\1", text)


def render_report(report: Report) -> str:
    result = (
        "| Status | Count |\n| --- | --- |\n"
        f"| Total | {report.total} |\n| Errors | {report.errors} |\n| Timeouts | {report.timeouts} |\n"
    )
    if report.failures:
        result += "\n| Source | URL | Status | Details |\n| --- | --- | --- | --- |\n"
        for failure in report.failures:
            cells = (failure.source, failure.url, failure.text, failure.details or "")
            result += "| " + " | ".join(escape_markdown(cell) for cell in cells) + " |\n"
    return result


def attempt_report(report: Path, attempt: int) -> Path:
    return report.with_name(f"{report.stem}-attempt-{attempt}{report.suffix}")


def finish(report: Path, data: Report | None, exit_codes: list[int], message: str, *, warning: bool = False) -> None:
    """Publish the final result without presenting recovered failures as current."""
    summary = f"## Documentation links\n\n{message}\n\n| Attempt | Exit code |\n| --- | --- |\n"
    summary += "".join(f"| {attempt} | {code} |\n" for attempt, code in enumerate(exit_codes, start=1))
    if data is not None:
        summary += f"\n### Final attempt\n\n{render_report(data)}\n"
    else:
        summary += (
            "\nThe final attempt did not produce a link-check report. Earlier reports are retained as artifacts.\n"
        )
    report.with_suffix(".md").write_text(summary, encoding="utf-8")
    print(f"::warning::{message}" if warning else message, flush=True)
    if summary_path := os.environ.get("GITHUB_STEP_SUMMARY"):
        with Path(summary_path).open("a", encoding="utf-8") as stream:
            stream.write(summary)


def finish_link_failures(report: Path, data: Report, exit_codes: list[int], message: str, policy: str) -> int:
    """Apply the failure policy without changing lychee's reports or cache."""
    distinct_urls = len({failure.url for failure in data.failures})
    warning = False
    if policy == "cache-warm":
        message += (
            f" Warning: {distinct_urls} distinct URLs remain unverified. Cache warming is advisory;"
            " only successful checks are cached."
        )
        warning = True
    elif all(failure.rate_limited for failure in data.failures):
        message += f" {distinct_urls} distinct HTTP 429 URLs remain unverified"
        if distinct_urls <= MAX_RATE_LIMITED_URLS:
            message += f" (limit: {MAX_RATE_LIMITED_URLS}); accepting these rate limits with a warning."
            warning = True
        else:
            message += f", exceeding the limit of {MAX_RATE_LIMITED_URLS}."
    finish(report, data, exit_codes, message, warning=warning)
    return 0 if warning else LINK_CHECK_FAILURE


def retry_lychee(
    initial_exit_code: int,
    report: Path,
    max_attempts: int = 3,
    *,
    policy: str = "check",
    run_attempt: Callable[[Path], int],
) -> int:
    """Retry only if every remaining failure is a recognized transient HTTP error."""
    if initial_exit_code not in (0, 1, 2, 3):
        raise ValueError(f"Unexpected initial lychee exit code: {initial_exit_code}")
    if max_attempts < 1:
        raise ValueError("max_attempts must be positive")
    if policy not in ("check", "cache-warm"):
        raise ValueError(f"Unknown lychee failure policy: {policy}")
    exit_codes = [initial_exit_code]
    if initial_exit_code not in (0, LINK_CHECK_FAILURE):
        finish(report, None, exit_codes, f"Lychee stopped on attempt 1 with exit code {initial_exit_code}; no retry.")
        return initial_exit_code

    data = read_report(report, initial_exit_code)
    shutil.copyfile(report, attempt_report(report, 1))
    while True:
        attempt = len(exit_codes)
        if exit_codes[-1] == 0:
            finish(report, data, exit_codes, f"Lychee passed on attempt {attempt}/{max_attempts}.")
            return 0
        if not data.transient:
            return finish_link_failures(
                report,
                data,
                exit_codes,
                f"Lychee has permanent or unclassified link failures on attempt {attempt}; no retry.",
                policy,
            )
        if attempt >= max_attempts:
            return finish_link_failures(
                report,
                data,
                exit_codes,
                f"Lychee still has transient link failures after {max_attempts} attempts.",
                policy,
            )
        cooldown = min(60 * attempt, 120)
        print(
            f"Lychee attempt {attempt} had transient failures; waiting {cooldown}s before attempt {attempt + 1}/{max_attempts}.",
            flush=True,
        )
        time.sleep(cooldown)
        current_report = attempt_report(report, attempt + 1)
        code = run_attempt(current_report)
        exit_codes.append(code)
        if code not in (0, LINK_CHECK_FAILURE):
            finish(
                report,
                None,
                exit_codes,
                f"Lychee stopped on attempt {attempt + 1} with exit code {code}; no retry.",
            )
            return code if code > 0 else 1
        data = read_report(current_report, code)
        shutil.copyfile(current_report, report)
