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


def response(url="https://example.test/guide", code=429, *, text="Request failed", details="Request timed out"):
    status = {"text": text}
    if code is None:
        status["details"] = details
    else:
        status["code"] = code
    return {"url": url, "status": status}


def report_data(*, errors=(), timeouts=(), total=5):
    return {
        "total": total,
        "errors": len(errors),
        "timeouts": len(timeouts),
        "error_map": {"docs/index.html": list(errors)} if errors else {},
        "timeout_map": {"docs/index.html": list(timeouts)} if timeouts else {},
    }


def write_report(path, data):
    path.write_text(json.dumps(data), encoding="utf-8")


@pytest.mark.agent_authored(model="gpt-6")
def test_initial_success_never_retries_and_preserves_json_and_final_markdown(tmp_path, monkeypatch):
    report = tmp_path / "lychee-authored.json"
    data = report_data()
    write_report(report, data)
    summary = tmp_path / "summary.md"
    monkeypatch.setenv("GITHUB_STEP_SUMMARY", str(summary))
    monkeypatch.setattr(retry_lychee.subprocess, "run", lambda *_args, **_kwargs: pytest.fail("Unexpected retry"))
    monkeypatch.setattr(retry_lychee.time, "sleep", lambda _delay: pytest.fail("Unexpected sleep"))

    assert retry_lychee.retry_lychee(0, report) == 0
    assert json.loads((tmp_path / "lychee-authored-attempt-1.json").read_text(encoding="utf-8")) == data
    text = summary.read_text(encoding="utf-8")
    assert "passed on attempt 1/3" in text
    assert "| Total | 5 |" in text
    assert report.with_suffix(".md").read_text(encoding="utf-8") == text


@pytest.mark.agent_authored(model="gpt-6")
def test_retries_reuse_cache_and_identical_arguments_until_success(tmp_path, monkeypatch):
    report = tmp_path / "lychee-rendered.json"
    initial = report_data(errors=[response()])
    write_report(report, initial)
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
        assert command[6:10] == ["--mode", "task", "--format", "json"]
        assert cache.read_text(encoding="utf-8") == "first-success\n" + (
            "second-success\n" if len(commands) == 2 else ""
        )
        code = 2 if len(commands) == 1 else 0
        write_report(Path(command[-1]), report_data(timeouts=[response(code=None)]) if code == 2 else report_data())
        if code == 2:
            with cache.open("a", encoding="utf-8") as stream:
                stream.write("second-success\n")
        return subprocess.CompletedProcess(command, code)

    monkeypatch.setattr(retry_lychee.subprocess, "run", run)

    assert retry_lychee.retry_lychee(2, report) == 0
    assert delays == [60, 120]
    assert len(commands) == 2
    assert json.loads(report.read_text(encoding="utf-8")) == report_data()
    assert json.loads((tmp_path / "lychee-rendered-attempt-1.json").read_text(encoding="utf-8")) == initial
    assert json.loads((tmp_path / "lychee-rendered-attempt-2.json").read_text(encoding="utf-8"))["timeouts"] == 1
    assert json.loads((tmp_path / "lychee-rendered-attempt-3.json").read_text(encoding="utf-8")) == report_data()
    text = summary.read_text(encoding="utf-8")
    assert text.startswith("Earlier step\n")
    assert "passed on attempt 3/3" in text
    assert "| 1 | 2 |\n| 2 | 2 |\n| 3 | 0 |" in text
    assert "example.test/guide" not in text
    assert "| Errors | 0 |\n| Timeouts | 0 |" in text


@pytest.mark.parametrize("code", [408, 429, 500, 503, 599])
@pytest.mark.agent_authored(model="gpt-6")
def test_known_transient_http_errors_are_retryable(tmp_path, code):
    report = tmp_path / "lychee.json"
    write_report(report, report_data(errors=[response(code=code)]))

    assert retry_lychee.read_report(report, 2).transient is True


@pytest.mark.parametrize("code", [None, 408, 200])
@pytest.mark.agent_authored(model="gpt-6")
def test_http_timeout_map_entries_are_retryable(tmp_path, code):
    report = tmp_path / "lychee.json"
    write_report(report, report_data(timeouts=[response(code=code)]))

    assert retry_lychee.read_report(report, 2).transient is True


@pytest.mark.parametrize(
    "failure",
    [
        response(code=404),
        response(code=403),
        response(code=400),
        response(code=600),
        response(code=None, text="Network error", details="Connection refused"),
        response("file:///docs/guide.html#missing", code=None, text="Missing fragment", details="Anchor not found"),
        response("mailto:someone@example.test", code=429),
        response("file:///docs/guide.html", code=503),
        response("https:missing-host", code=503),
        response("https://[malformed", code=503),
    ],
)
@pytest.mark.agent_authored(model="gpt-6")
def test_permanent_or_unknown_failures_stop_without_sleep_or_shared_arguments(tmp_path, monkeypatch, failure):
    report = tmp_path / "lychee.json"
    write_report(report, report_data(errors=[failure]))
    monkeypatch.delenv("LYCHEE_ARGS", raising=False)
    monkeypatch.setattr(retry_lychee.subprocess, "run", lambda *_args, **_kwargs: pytest.fail("Unexpected retry"))
    monkeypatch.setattr(retry_lychee.time, "sleep", lambda _delay: pytest.fail("Unexpected sleep"))

    assert retry_lychee.retry_lychee(2, report) == 2
    assert "permanent or unclassified" in report.with_suffix(".md").read_text(encoding="utf-8")


@pytest.mark.agent_authored(model="gpt-6")
def test_local_timeouts_are_not_retryable(tmp_path, monkeypatch):
    report = tmp_path / "lychee.json"
    write_report(report, report_data(timeouts=[response("file:///docs/guide.html", code=None)]))
    monkeypatch.delenv("LYCHEE_ARGS", raising=False)
    monkeypatch.setattr(retry_lychee.time, "sleep", lambda _delay: pytest.fail("Unexpected sleep"))

    assert retry_lychee.retry_lychee(2, report) == 2


@pytest.mark.agent_authored(model="gpt-6")
def test_mixed_permanent_and_transient_failures_stop_immediately(tmp_path, monkeypatch):
    report = tmp_path / "lychee.json"
    write_report(report, report_data(errors=[response(), response(code=404)], timeouts=[response(code=None)]))
    monkeypatch.delenv("LYCHEE_ARGS", raising=False)
    monkeypatch.setattr(retry_lychee.time, "sleep", lambda _delay: pytest.fail("Unexpected sleep"))

    assert retry_lychee.retry_lychee(2, report) == 2


@pytest.mark.agent_authored(model="gpt-6")
def test_transient_failure_turning_into_mixed_failures_stops_on_that_attempt(tmp_path, monkeypatch):
    report = tmp_path / "lychee.json"
    write_report(report, report_data(errors=[response()]))
    mixed = report_data(errors=[response(code=503), response(code=404)])
    monkeypatch.setenv("LYCHEE_ARGS", "--cache --files-from inputs.txt")
    delays = []
    monkeypatch.setattr(retry_lychee.time, "sleep", delays.append)
    calls = []

    def run(command, *, check):
        calls.append(command)
        write_report(Path(command[-1]), mixed)
        return subprocess.CompletedProcess(command, 2)

    monkeypatch.setattr(retry_lychee.subprocess, "run", run)

    assert retry_lychee.retry_lychee(2, report) == 2
    assert len(calls) == 1
    assert delays == [60]
    assert json.loads(report.read_text(encoding="utf-8")) == mixed
    text = report.with_suffix(".md").read_text(encoding="utf-8")
    assert "permanent or unclassified link failures on attempt 2" in text
    assert "| 1 | 2 |\n| 2 | 2 |" in text


@pytest.mark.agent_authored(model="gpt-6")
def test_deduplicated_response_maps_do_not_require_counts_to_equal_entries(tmp_path):
    report = tmp_path / "lychee.json"
    data = report_data(errors=[response()], timeouts=[response(code=None)], total=100)
    data.update(errors=27, timeouts=10)
    write_report(report, data)

    result = retry_lychee.read_report(report, 2)

    assert result.transient is True
    assert result.errors == 27
    assert result.timeouts == 10
    assert len(result.failures) == 2


@pytest.mark.parametrize("policy,expected", [("check", 2), ("cache-warm", 0)])
@pytest.mark.agent_authored(model="gpt-6")
def test_three_transient_attempts_exhausted_follow_failure_policy(tmp_path, monkeypatch, capsys, policy, expected):
    report = tmp_path / "lychee.json"
    data = report_data(errors=[response(code=503)])
    write_report(report, data)
    monkeypatch.setenv("LYCHEE_ARGS", "--cache --files-from inputs.txt")
    delays = []
    monkeypatch.setattr(retry_lychee.time, "sleep", delays.append)
    calls = []

    def run(command, *, check):
        calls.append(command)
        write_report(Path(command[-1]), data)
        return subprocess.CompletedProcess(command, 2)

    monkeypatch.setattr(retry_lychee.subprocess, "run", run)

    assert retry_lychee.retry_lychee(2, report, policy=policy) == expected
    assert len(calls) == 2
    assert delays == [60, 120]
    assert len(list(tmp_path.glob("lychee-attempt-*.json"))) == 3
    text = report.with_suffix(".md").read_text(encoding="utf-8")
    assert "after 3 attempts" in text
    assert "| 1 | 2 |\n| 2 | 2 |\n| 3 | 2 |" in text
    assert json.loads(report.read_text(encoding="utf-8")) == data
    assert ("::warning::" in capsys.readouterr().out) == (expected == 0)
    if expected == 0:
        assert "remain unverified" in text


@pytest.mark.parametrize("code", [1, 3])
@pytest.mark.parametrize("policy", ["check", "cache-warm"])
@pytest.mark.agent_authored(model="gpt-6")
def test_initial_non_link_failure_stops_without_retry(tmp_path, monkeypatch, code, policy):
    monkeypatch.setattr(retry_lychee.subprocess, "run", lambda *_args, **_kwargs: pytest.fail("Unexpected retry"))
    monkeypatch.setattr(retry_lychee.time, "sleep", lambda _delay: pytest.fail("Unexpected sleep"))

    assert retry_lychee.retry_lychee(code, tmp_path / "missing.json", policy=policy) == code


@pytest.mark.parametrize("code", [1, 3, -15])
@pytest.mark.parametrize("policy", ["check", "cache-warm"])
@pytest.mark.agent_authored(model="gpt-6")
def test_retry_non_link_failure_does_not_claim_stale_report(tmp_path, monkeypatch, code, policy):
    report = tmp_path / "lychee.json"
    write_report(report, report_data(errors=[response()]))
    monkeypatch.setenv("LYCHEE_ARGS", "--cache --files-from inputs.txt")
    delays = []
    monkeypatch.setattr(retry_lychee.time, "sleep", delays.append)
    calls = []

    def run(command, *, check):
        calls.append(command)
        return subprocess.CompletedProcess(command, code)

    monkeypatch.setattr(retry_lychee.subprocess, "run", run)

    assert retry_lychee.retry_lychee(2, report, policy=policy) == (code if code > 0 else 1)
    assert len(calls) == 1
    assert delays == [60]
    text = report.with_suffix(".md").read_text(encoding="utf-8")
    assert "did not produce a link-check report" in text
    assert "### Final attempt" not in text
    assert "| Total |" not in text


@pytest.mark.parametrize(
    "name,value", [("total", True), ("errors", False), ("timeouts", -1), ("errors", 1.0), ("total", "5")]
)
@pytest.mark.agent_authored(model="gpt-6")
def test_invalid_counts_are_rejected(tmp_path, name, value):
    report = tmp_path / "lychee.json"
    data = report_data()
    data[name] = value
    write_report(report, data)

    with pytest.raises(ValueError, match="nonnegative integer"):
        retry_lychee.read_report(report, 0)


@pytest.mark.parametrize(
    "data",
    [
        {},
        [],
        report_data(total=0),
        {**report_data(), "errors": 1},
        {**report_data(), "timeouts": 1},
        {**report_data(errors=[response()]), "errors": 0},
        {**report_data(timeouts=[response(code=None)]), "timeouts": 0},
        report_data(errors=[response()], timeouts=[response(code=None)], total=1),
        {**report_data(errors=[response()]), "error_map": []},
        {**report_data(errors=[response()]), "error_map": {"source": response()}},
        {**report_data(errors=[response()]), "error_map": {"source": ["response"]}},
        {
            **report_data(errors=[response()]),
            "error_map": {"source": [{"url": 42, "status": {"text": "Error", "code": 429}}]},
        },
        report_data(errors=[{"url": "https://example.test/", "status": {"code": 429}}]),
        report_data(errors=[response(code=True)]),
        report_data(errors=[response(code="429")]),
        report_data(errors=[response(code=None, details=None)]),
        report_data(errors=[{"url": "https://example.test/", "status": {"text": "Unknown error"}}]),
    ],
)
@pytest.mark.parametrize("policy", ["check", "cache-warm"])
@pytest.mark.agent_authored(model="gpt-6")
def test_malformed_or_contradictory_json_cannot_pass_or_trigger_retries(tmp_path, monkeypatch, data, policy):
    report = tmp_path / "lychee.json"
    write_report(report, data)
    monkeypatch.setattr(retry_lychee.time, "sleep", lambda _delay: pytest.fail("Unexpected sleep"))

    with pytest.raises(ValueError, match="Lychee"):
        retry_lychee.retry_lychee(2, report, policy=policy)


@pytest.mark.parametrize(
    "data,code",
    [(report_data(), 2), (report_data(errors=[response()]), 0), (report_data(timeouts=[response(code=None)]), 0)],
)
@pytest.mark.agent_authored(model="gpt-6")
def test_report_must_agree_with_exit_code(tmp_path, data, code):
    report = tmp_path / "lychee.json"
    write_report(report, data)

    with pytest.raises(ValueError, match="disagree with exit code"):
        retry_lychee.retry_lychee(code, report)


@pytest.mark.parametrize("contents", ["", "not json", "{broken}"])
@pytest.mark.parametrize("policy", ["check", "cache-warm"])
@pytest.mark.agent_authored(model="gpt-6")
def test_invalid_json_cannot_pass(tmp_path, contents, policy):
    report = tmp_path / "lychee.json"
    report.write_text(contents, encoding="utf-8")

    with pytest.raises(json.JSONDecodeError):
        retry_lychee.retry_lychee(0, report, policy=policy)


@pytest.mark.parametrize("policy", ["check", "cache-warm"])
@pytest.mark.agent_authored(model="gpt-6")
def test_missing_initial_report_cannot_pass(tmp_path, policy):
    with pytest.raises(FileNotFoundError):
        retry_lychee.retry_lychee(0, tmp_path / "missing.json", policy=policy)


@pytest.mark.agent_authored(model="gpt-6")
def test_final_markdown_escapes_untrusted_report_fields_and_retains_raw_json(tmp_path):
    report = tmp_path / "lychee.json"
    failure = response(
        "file:///docs/<b>x</b>|[link].html",
        code=None,
        text="Missing <script>fragment</script>",
        details="`a` *b* | c\n# injected",
    )
    data = report_data(errors=[failure])
    data["error_map"] = {"docs/[source]|file\n# heading": [failure]}
    write_report(report, data)

    assert retry_lychee.retry_lychee(2, report) == 2

    text = report.with_suffix(".md").read_text(encoding="utf-8")
    assert "<script>" not in text
    assert "&lt;script&gt;" in text
    assert "\\[source\\]\\|file # heading" in text
    assert "\\`a\\` \\*b\\* \\| c # injected" in text
    assert "\n# injected" not in text
    assert json.loads(report.read_text(encoding="utf-8")) == data


@pytest.mark.parametrize("code", [-1, 4, 255])
@pytest.mark.agent_authored(model="gpt-6")
def test_unknown_initial_exit_code_is_rejected(tmp_path, code):
    with pytest.raises(ValueError, match="Unexpected initial lychee exit code"):
        retry_lychee.retry_lychee(code, tmp_path / "missing.json")


@pytest.mark.agent_authored(model="gpt-6")
def test_retry_requires_shared_arguments_before_sleeping(tmp_path, monkeypatch):
    report = tmp_path / "lychee.json"
    write_report(report, report_data(errors=[response()]))
    monkeypatch.delenv("LYCHEE_ARGS", raising=False)
    monkeypatch.setattr(retry_lychee.time, "sleep", lambda _delay: pytest.fail("Unexpected sleep"))

    with pytest.raises(ValueError, match="LYCHEE_ARGS"):
        retry_lychee.retry_lychee(2, report)


@pytest.mark.agent_authored(model="gpt-6")
def test_one_attempt_limit_keeps_transient_failure_without_retry_arguments(tmp_path, monkeypatch):
    report = tmp_path / "lychee.json"
    write_report(report, report_data(errors=[response(code=503)]))
    monkeypatch.delenv("LYCHEE_ARGS", raising=False)
    monkeypatch.setattr(retry_lychee.time, "sleep", lambda _delay: pytest.fail("Unexpected sleep"))

    assert retry_lychee.retry_lychee(2, report, max_attempts=1) == 2


@pytest.mark.parametrize("url_count,expected", [(10, 0), (11, 2)])
@pytest.mark.agent_authored(model="gpt-6")
def test_rate_limit_allowance_counts_distinct_urls_after_all_attempts(
    tmp_path, monkeypatch, capsys, url_count, expected
):
    report = tmp_path / "lychee.json"
    data = report_data(errors=[response(f"https://example.test/{number}") for number in range(url_count)], total=100)
    write_report(report, data)
    monkeypatch.setenv("LYCHEE_ARGS", "--cache --files-from inputs.txt")
    delays = []
    monkeypatch.setattr(retry_lychee.time, "sleep", delays.append)

    def run(command, *, check):
        write_report(Path(command[-1]), data)
        return subprocess.CompletedProcess(command, 2)

    monkeypatch.setattr(retry_lychee.subprocess, "run", run)

    assert retry_lychee.retry_lychee(2, report) == expected
    assert delays == [60, 120]
    assert json.loads(report.read_text(encoding="utf-8")) == data
    text = report.with_suffix(".md").read_text(encoding="utf-8")
    assert f"{url_count} distinct HTTP 429 URLs remain unverified" in text
    assert "| 1 | 2 |\n| 2 | 2 |\n| 3 | 2 |" in text
    assert ("::warning::" in capsys.readouterr().out) == (expected == 0)
    for number in range(url_count):
        assert f"https://example.test/{number}" in text


@pytest.mark.agent_authored(model="gpt-6")
def test_rate_limit_allowance_deduplicates_exact_urls_across_sources(tmp_path, capsys):
    report = tmp_path / "lychee.json"
    data = report_data(total=100)
    data.update(errors=100, error_map={f"docs/{number}.html": [response()] for number in range(50)})
    write_report(report, data)

    assert retry_lychee.retry_lychee(2, report, max_attempts=1) == 0
    assert "1 distinct HTTP 429 URLs remain unverified" in capsys.readouterr().out
    assert json.loads(report.read_text(encoding="utf-8")) == data
    assert "| Errors | 100 |" in report.with_suffix(".md").read_text(encoding="utf-8")


@pytest.mark.parametrize(
    "other_failure,timed_out",
    [
        (response(code=404), False),
        (response(code=503), False),
        (response(code=None), True),
        (response(code=429), True),
        (response("mailto:someone@example.test"), False),
        (response("https:missing-host"), False),
        (response("https://[malformed"), False),
    ],
)
@pytest.mark.agent_authored(model="gpt-6")
def test_429_allowance_never_accepts_other_failures(tmp_path, capsys, other_failure, timed_out):
    report = tmp_path / "lychee.json"
    data = report_data(
        errors=[response(), *([] if timed_out else [other_failure])], timeouts=[other_failure] if timed_out else []
    )
    write_report(report, data)

    assert retry_lychee.retry_lychee(2, report, max_attempts=1) == 2
    assert "::warning::" not in capsys.readouterr().out


@pytest.mark.parametrize(
    "failure",
    [
        response(code=404),
        response(code=None, text="Network error", details="Connection refused"),
        response("file:///docs/guide.html#missing", code=None, text="Missing fragment", details="Anchor not found"),
    ],
)
@pytest.mark.agent_authored(model="gpt-6")
def test_cache_warming_warns_about_permanent_failures_without_retrying(tmp_path, monkeypatch, capsys, failure):
    report = tmp_path / "lychee.json"
    data = report_data(errors=[failure, response()])
    write_report(report, data)
    monkeypatch.delenv("LYCHEE_ARGS", raising=False)
    monkeypatch.setattr(retry_lychee.subprocess, "run", lambda *_args, **_kwargs: pytest.fail("Unexpected retry"))
    monkeypatch.setattr(retry_lychee.time, "sleep", lambda _delay: pytest.fail("Unexpected sleep"))

    assert retry_lychee.retry_lychee(2, report, policy="cache-warm") == 0
    assert "::warning::" in capsys.readouterr().out
    assert json.loads(report.read_text(encoding="utf-8")) == data
    text = report.with_suffix(".md").read_text(encoding="utf-8")
    assert "permanent or unclassified link failures on attempt 1; no retry" in text
    assert "remain unverified" in text
    assert "only successful checks are cached" in text
    assert "| 1 | 2 |" in text


@pytest.mark.parametrize(
    "extra_args,attempts,policy",
    [([], 3, "check"), (["--max-attempts", "1", "--policy", "cache-warm"], 1, "cache-warm")],
)
@pytest.mark.agent_authored(model="gpt-6")
def test_cli_passes_defaults_and_explicit_policy(monkeypatch, extra_args, attempts, policy):
    monkeypatch.setattr(
        sys, "argv", ["retry_lychee.py", "--initial-exit-code", "2", "--report", "report.json", *extra_args]
    )

    def retry(initial_exit_code, report, max_attempts, *, policy):
        assert initial_exit_code == 2
        assert report == Path("report.json")
        assert max_attempts == attempts
        assert policy == expected_policy
        return 2

    expected_policy = policy
    monkeypatch.setattr(retry_lychee, "retry_lychee", retry)
    assert retry_lychee.main() == 2


@pytest.mark.agent_authored(model="gpt-6")
def test_cli_rejects_unknown_failure_policy(monkeypatch):
    monkeypatch.setattr(
        sys, "argv", ["retry_lychee.py", "--initial-exit-code", "2", "--report", "report.json", "--policy", "unknown"]
    )

    with pytest.raises(SystemExit) as error:
        retry_lychee.main()
    assert error.value.code == 2


@pytest.mark.parametrize("policy", ["check", "cache-warm"])
@pytest.mark.parametrize("invalid_input", ["missing", "malformed", "empty", "invalid-attempts"])
@pytest.mark.agent_authored(model="gpt-6")
def test_cli_keeps_invalid_inputs_as_hard_failures(tmp_path, monkeypatch, capsys, policy, invalid_input):
    report = tmp_path / "lychee.json"
    if invalid_input == "malformed":
        report.write_text("not json", encoding="utf-8")
    elif invalid_input == "empty":
        write_report(report, report_data(total=0))
    elif invalid_input == "invalid-attempts":
        write_report(report, report_data())
    extra_args = ["--max-attempts", "0"] if invalid_input == "invalid-attempts" else []
    monkeypatch.setattr(
        sys,
        "argv",
        ["retry_lychee.py", "--initial-exit-code", "0", "--report", str(report), "--policy", policy, *extra_args],
    )

    assert retry_lychee.main() == 1
    text = capsys.readouterr().out
    assert "::error::" in text
    assert "::warning::" not in text
