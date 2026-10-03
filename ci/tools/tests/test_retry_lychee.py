# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).parent.parent))
import retry_lychee


def markdown_report(total=2, errors=0):
    return f"# Summary\n\n| Status | Count |\n| --- | --- |\n| \U0001f50d Total | {total} |\n| Error | {errors} |\n"


@pytest.mark.agent_authored(model="gpt-6")
def test_initial_success_never_runs_lychee_and_preserves_first_report(tmp_path, monkeypatch):
    report = tmp_path / "lychee-authored.md"
    report.write_text(markdown_report(), encoding="utf-8")
    summary = tmp_path / "summary.md"
    monkeypatch.setenv("GITHUB_STEP_SUMMARY", str(summary))
    monkeypatch.setattr(retry_lychee.subprocess, "run", lambda *_args, **_kwargs: pytest.fail("Unexpected retry"))
    monkeypatch.setattr(retry_lychee.time, "sleep", lambda _delay: pytest.fail("Unexpected sleep"))

    assert retry_lychee.retry_lychee(0, report) == 0
    assert (tmp_path / "lychee-authored-attempt-1.md").read_text(encoding="utf-8") == markdown_report()
    assert "passed on attempt 1" in summary.read_text(encoding="utf-8")


@pytest.mark.agent_authored(model="gpt-6")
def test_retries_reuse_cache_and_identical_arguments_until_success(tmp_path, monkeypatch):
    report = tmp_path / "lychee-rendered.md"
    report.write_text(markdown_report(errors=1), encoding="utf-8")
    cache = tmp_path / ".lycheecache"
    cache.write_text("first-success\n", encoding="utf-8")
    summary = tmp_path / "summary.md"
    summary.write_text("Earlier step\n", encoding="utf-8")
    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("GITHUB_STEP_SUMMARY", str(summary))
    monkeypatch.setenv("LYCHEE_ARGS", '--files-from "input files.txt" --cache --config "link config.toml"')
    delays = []
    monkeypatch.setattr(retry_lychee.time, "sleep", delays.append)
    commands = []

    def run(command, *, check):
        assert check is False
        commands.append(command)
        assert command[:6] == ["lychee", "--files-from", "input files.txt", "--cache", "--config", "link config.toml"]
        assert command[6:10] == ["--mode", "task", "--format", "markdown"]
        assert cache.read_text(encoding="utf-8") == "first-success\n" + (
            "second-success\n" if len(commands) == 2 else ""
        )
        code = 2 if len(commands) == 1 else 0
        Path(command[-1]).write_text(markdown_report(errors=int(code != 0)), encoding="utf-8")
        if code == 2:
            with cache.open("a", encoding="utf-8") as stream:
                stream.write("second-success\n")
        return subprocess.CompletedProcess(command, code)

    monkeypatch.setattr(retry_lychee.subprocess, "run", run)

    assert retry_lychee.retry_lychee(2, report) == 0
    assert delays == [60, 120]
    assert len(commands) == 2
    assert report.read_text(encoding="utf-8") == markdown_report()
    assert (tmp_path / "lychee-rendered-attempt-1.md").read_text(encoding="utf-8") == markdown_report(errors=1)
    assert (tmp_path / "lychee-rendered-attempt-2.md").read_text(encoding="utf-8") == markdown_report(errors=1)
    assert (tmp_path / "lychee-rendered-attempt-3.md").read_text(encoding="utf-8") == markdown_report()
    text = summary.read_text(encoding="utf-8")
    assert text.startswith("Earlier step\n")
    assert "passed on attempt 3/10" in text
    assert "| 1 | 2 |\n| 2 | 2 |\n| 3 | 0 |" in text
    assert text.endswith(markdown_report() + "\n")


@pytest.mark.agent_authored(model="gpt-6")
def test_ten_attempts_exhausted_remains_a_failure(tmp_path, monkeypatch):
    report = tmp_path / "lychee.md"
    report.write_text(markdown_report(errors=1), encoding="utf-8")
    monkeypatch.setenv("LYCHEE_ARGS", "--cache --files-from inputs.txt")
    delays = []
    monkeypatch.setattr(retry_lychee.time, "sleep", delays.append)
    calls = []

    def run(command, *, check):
        calls.append(command)
        Path(command[-1]).write_text(markdown_report(errors=1), encoding="utf-8")
        return subprocess.CompletedProcess(command, 2)

    monkeypatch.setattr(retry_lychee.subprocess, "run", run)

    assert retry_lychee.retry_lychee(2, report) == 2
    assert len(calls) == 9
    assert delays == [60, *([120] * 8)]
    assert len(list(tmp_path.glob("lychee-attempt-*.md"))) == 10


@pytest.mark.parametrize("code", [1, 3])
@pytest.mark.agent_authored(model="gpt-6")
def test_initial_non_link_failure_stops_without_retry(tmp_path, monkeypatch, code):
    monkeypatch.setattr(retry_lychee.subprocess, "run", lambda *_args, **_kwargs: pytest.fail("Unexpected retry"))
    monkeypatch.setattr(retry_lychee.time, "sleep", lambda _delay: pytest.fail("Unexpected sleep"))

    assert retry_lychee.retry_lychee(code, tmp_path / "missing.md") == code


@pytest.mark.parametrize("code", [1, 3, -15])
@pytest.mark.agent_authored(model="gpt-6")
def test_retry_non_link_failure_stops_and_summary_does_not_claim_stale_report(tmp_path, monkeypatch, code):
    report = tmp_path / "lychee.md"
    report.write_text(markdown_report(errors=1), encoding="utf-8")
    summary = tmp_path / "summary.md"
    monkeypatch.setenv("GITHUB_STEP_SUMMARY", str(summary))
    monkeypatch.setenv("LYCHEE_ARGS", "--cache --files-from inputs.txt")
    delays = []
    monkeypatch.setattr(retry_lychee.time, "sleep", delays.append)
    calls = []

    def run(command, *, check):
        calls.append(command)
        return subprocess.CompletedProcess(command, code)

    monkeypatch.setattr(retry_lychee.subprocess, "run", run)

    assert retry_lychee.retry_lychee(2, report) == (code if code > 0 else 1)
    assert len(calls) == 1
    assert delays == [60]
    text = summary.read_text(encoding="utf-8")
    assert "did not produce a link-check report" in text
    assert "### Final attempt" not in text
    assert "Total |" not in text


@pytest.mark.parametrize("contents", ["", "not a report", markdown_report(total=0)])
@pytest.mark.agent_authored(model="gpt-6")
def test_invalid_or_empty_initial_report_cannot_pass(tmp_path, monkeypatch, contents):
    report = tmp_path / "lychee.md"
    report.write_text(contents, encoding="utf-8")
    monkeypatch.setattr(retry_lychee.subprocess, "run", lambda *_args, **_kwargs: pytest.fail("Unexpected retry"))

    with pytest.raises(ValueError, match="no Total count|checked no links"):
        retry_lychee.retry_lychee(0, report)


@pytest.mark.agent_authored(model="gpt-6")
def test_missing_initial_report_cannot_pass(tmp_path):
    with pytest.raises(FileNotFoundError):
        retry_lychee.retry_lychee(0, tmp_path / "missing.md")


@pytest.mark.agent_authored(model="gpt-6")
def test_successful_retry_with_zero_links_still_fails(tmp_path, monkeypatch):
    report = tmp_path / "lychee.md"
    report.write_text(markdown_report(errors=1), encoding="utf-8")
    monkeypatch.setenv("LYCHEE_ARGS", "--cache --files-from inputs.txt")
    monkeypatch.setattr(retry_lychee.time, "sleep", lambda _delay: None)

    def run(command, *, check):
        Path(command[-1]).write_text(markdown_report(total=0), encoding="utf-8")
        return subprocess.CompletedProcess(command, 0)

    monkeypatch.setattr(retry_lychee.subprocess, "run", run)

    with pytest.raises(ValueError, match="checked no links"):
        retry_lychee.retry_lychee(2, report)


@pytest.mark.parametrize("code", [-1, 4, 255])
@pytest.mark.agent_authored(model="gpt-6")
def test_unknown_initial_exit_code_is_rejected(tmp_path, code):
    with pytest.raises(ValueError, match="Unexpected initial lychee exit code"):
        retry_lychee.retry_lychee(code, tmp_path / "missing.md")


@pytest.mark.agent_authored(model="gpt-6")
def test_retry_requires_shared_arguments_before_sleeping(tmp_path, monkeypatch):
    report = tmp_path / "lychee.md"
    report.write_text(markdown_report(errors=1), encoding="utf-8")
    monkeypatch.delenv("LYCHEE_ARGS", raising=False)
    monkeypatch.setattr(retry_lychee.time, "sleep", lambda _delay: pytest.fail("Unexpected sleep"))

    with pytest.raises(ValueError, match="LYCHEE_ARGS"):
        retry_lychee.retry_lychee(2, report)


@pytest.mark.agent_authored(model="gpt-6")
def test_one_attempt_limit_keeps_failure_without_requiring_retry_arguments(tmp_path, monkeypatch):
    report = tmp_path / "lychee.md"
    report.write_text(markdown_report(errors=1), encoding="utf-8")
    monkeypatch.delenv("LYCHEE_ARGS", raising=False)
    monkeypatch.setattr(retry_lychee.time, "sleep", lambda _delay: pytest.fail("Unexpected sleep"))

    assert retry_lychee.retry_lychee(2, report, max_attempts=1) == 2
