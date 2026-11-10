# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).parent.parent))
import retry_lychee
import run_lychee_precommit


def native_report(*, code=0, total=5, status=503):
    return {
        "total": total,
        "errors": int(code == 2),
        "timeouts": 0,
        "error_map": {
            "guide.md": [{"url": "https://example.test/guide", "status": {"text": "HTTP error", "code": status}}]
        }
        if code == 2
        else {},
        "timeout_map": {},
    }


def write_part(directory, name, *, code=0, total=5, status=503):
    (directory / f"{name}.status.json").write_text(json.dumps({"exit_code": code}))
    (directory / f"{name}.json").write_text(json.dumps(native_report(code=code, total=total, status=status)))


@pytest.mark.agent_authored(model="gpt-6.1-sol")
def test_collects_all_chunks_including_an_empty_chunk(tmp_path):
    directory = tmp_path / "parts"
    directory.mkdir()
    write_part(directory, "one", total=0)
    write_part(directory, "two", code=2)
    write_part(directory, "three")
    output = tmp_path / "report.json"

    assert run_lychee_precommit.collect_reports(directory, output, 1) == 2
    data = json.loads(output.read_text())
    assert data["total"] == 10
    assert data["errors"] == 1
    assert len(data["error_map"]["guide.md"]) == 1


@pytest.mark.parametrize(
    "problem",
    [
        "missing-status",
        "native-config-error",
        "boolean-exit",
        "missing-report",
        "bad-json",
        "empty-scan",
        "extra-incomplete-report",
        "outer-failure",
    ],
)
@pytest.mark.agent_authored(model="gpt-6.1-sol")
def test_machine_failures_never_become_advisory_link_failures(tmp_path, problem):
    write_part(tmp_path, "one")
    if problem == "missing-status":
        (tmp_path / "one.status.json").unlink()
    elif problem == "native-config-error":
        (tmp_path / "one.status.json").write_text('{"exit_code": 1}')
    elif problem == "boolean-exit":
        (tmp_path / "one.status.json").write_text('{"exit_code": false}')
    elif problem == "missing-report":
        (tmp_path / "one.json").unlink()
    elif problem == "bad-json":
        (tmp_path / "one.json").write_text("not JSON")
    elif problem == "empty-scan":
        write_part(tmp_path, "one", total=0)
    elif problem == "extra-incomplete-report":
        (tmp_path / "incomplete.json").write_text("{}")
    with pytest.raises((OSError, ValueError)):
        run_lychee_precommit.collect_reports(tmp_path, tmp_path / "report.json", int(problem == "outer-failure"))


@pytest.mark.parametrize("kind", ["authored", "rendered"])
@pytest.mark.agent_authored(model="gpt-6.1-sol")
def test_runs_actual_selected_hook_and_unskips_only_that_hook(tmp_path, monkeypatch, kind):
    monkeypatch.setattr(run_lychee_precommit, "ROOT", tmp_path)
    monkeypatch.setenv("SKIP", "lychee, check-precommit-installed, lychee-rendered")
    commands = []
    hook = "lychee" if kind == "authored" else "lychee-rendered"

    def run(command, *, cwd, env, check):
        assert check is False
        assert cwd == tmp_path
        assert hook not in env["SKIP"].split(",")
        assert "check-precommit-installed" in env["SKIP"]
        assert env["PYTHONUTF8"] == "1"
        write_part(Path(env["LYCHEE_REPORT_DIR"]), "native")
        commands.append(command)
        return subprocess.CompletedProcess(command, 0)

    monkeypatch.setattr(run_lychee_precommit.subprocess, "run", run)
    report = tmp_path / "report.json"
    assert run_lychee_precommit.run_hook(kind, report) == 0
    expected = [sys.executable, "-m", "pre_commit", "run", hook, "--all-files", "--show-diff-on-failure"]
    assert commands == [expected + (["--hook-stage", "manual"] if kind == "rendered" else [])]
    assert json.loads(report.read_text())["total"] == 5


@pytest.mark.agent_authored(model="gpt-6.1-sol")
def test_retries_real_hook_with_cache_intact_and_stops_after_recovery(tmp_path, monkeypatch):
    report = tmp_path / "lychee-authored.json"
    cache = tmp_path / ".lycheecache"
    cache.write_text("already verified")
    attempts = []
    delays = []

    def run_hook(kind, output):
        assert kind == "authored"
        assert cache.read_text() == "already verified"
        attempts.append(output)
        code = 2 if len(attempts) == 1 else 0
        output.write_text(json.dumps(native_report(code=code)))
        return code

    monkeypatch.setattr(run_lychee_precommit, "run_hook", run_hook)
    monkeypatch.setattr(retry_lychee.time, "sleep", delays.append)
    assert run_lychee_precommit.run_checks("authored", report, policy="cache-warm") == 0
    assert len(attempts) == 2
    assert delays == [60]
    assert "passed on attempt 2/3" in report.with_suffix(".md").read_text()
    assert "example.test" not in report.with_suffix(".md").read_text()


@pytest.mark.parametrize("policy", ["cache-warm", "allow-429"])
@pytest.mark.agent_authored(model="gpt-6.1-sol")
def test_missing_native_status_is_fatal_even_under_relaxed_link_policies(tmp_path, monkeypatch, capsys, policy):
    monkeypatch.setattr(
        run_lychee_precommit.subprocess, "run", lambda *_args, **_kwargs: subprocess.CompletedProcess([], 1)
    )
    assert run_lychee_precommit.main(["authored", "--policy", policy, "--report", str(tmp_path / "report.json")]) == 1
    assert "no native status" in capsys.readouterr().out


@pytest.mark.parametrize("kind", ["authored", "rendered"])
@pytest.mark.agent_authored(model="gpt-6")
def test_cli_maintainer_override_accepts_only_rate_limits_after_retrying(tmp_path, monkeypatch, kind):
    report = tmp_path / "report.json"
    calls = []

    def run_hook(selected_kind, output):
        calls.append(selected_kind)
        data = native_report(code=2, total=20, status=429)
        data["errors"] = 11
        data["error_map"]["guide.md"] = [
            {"url": f"https://example.test/{number}", "status": {"text": "Rate limited", "code": 429}}
            for number in range(11)
        ]
        output.write_text(json.dumps(data))
        return 2

    monkeypatch.setattr(run_lychee_precommit, "run_hook", run_hook)
    monkeypatch.setattr(retry_lychee.time, "sleep", lambda _: None)

    assert run_lychee_precommit.main([kind, "--policy", "allow-429", "--report", str(report)]) == 0
    assert calls == [kind] * 3
    assert len(list(tmp_path.glob("report-attempt-*.json"))) == 3
    assert "maintainer" in report.with_suffix(".md").read_text().lower()


@pytest.mark.agent_authored(model="gpt-6.1-sol")
def test_zero_attempts_rejected_before_running_precommit(tmp_path, monkeypatch):
    monkeypatch.setattr(run_lychee_precommit, "run_hook", lambda *_args: pytest.fail("Unexpected invocation"))
    with pytest.raises(ValueError, match="positive"):
        run_lychee_precommit.run_checks("authored", tmp_path / "report.json", 0)
