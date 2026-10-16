# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

# ruff: noqa: S603

import json
import os
import subprocess
import sys
from pathlib import Path

import pytest

LOOKUP_RUN_ID = Path(__file__).resolve().parents[1] / "lookup-run-id"
HEAD_SHA = "a" * 40
OLDER_SHA = "b" * 40


@pytest.fixture
def lookup(tmp_path):
    fixture_path = tmp_path / "fixture.json"
    mock = tmp_path / "gh"
    mock.write_text(
        f"#!{sys.executable}\n"
        "import json, os, sys\n"
        "from pathlib import Path\n"
        "fixture = json.loads(Path(os.environ['LOOKUP_FIXTURE']).read_text())\n"
        "arguments = sys.argv[1:]\n"
        "endpoint = next(value for value in arguments if value.startswith('repos/'))\n"
        "if endpoint.endswith('/actions/workflows/ci.yml'):\n"
        "    print(155304118)\n"
        "elif '/branches/' in endpoint:\n"
        "    print(fixture['commits'][0])\n"
        "elif endpoint.endswith('/commits'):\n"
        "    print('\\n'.join(fixture['commits']))\n"
        "elif endpoint.endswith('/runs'):\n"
        "    sha = next(value.split('=', 1)[1] for value in arguments if value.startswith('head_sha='))\n"
        "    print(json.dumps({'workflow_runs': fixture['runs'].get(sha, [])}))\n"
        "else:\n"
        "    raise RuntimeError(endpoint)\n"
    )
    mock.chmod(0o755)

    def invoke(runs, *, event=None, commits=None, mode=None):
        fixture_path.write_text(json.dumps({"commits": commits or [HEAD_SHA], "runs": runs}))
        arguments = ["bash", str(LOOKUP_RUN_ID), *(mode or ["--branch", "main"]), "--head-sha"]
        if event is not None:
            arguments.extend(["--event", event])
        arguments.extend(["NVIDIA/cuda-python", "ci.yml"])
        return subprocess.run(
            arguments,
            env=dict(
                os.environ,
                PATH=str(tmp_path) + os.pathsep + os.environ["PATH"],
                GH_TOKEN=tmp_path.name,
                LOOKUP_FIXTURE=str(fixture_path),
                LOOKUP_RUN_ID_COMMIT_SCAN_LIMIT="2",
            ),
            capture_output=True,
            text=True,
            timeout=15,
        )

    return invoke


def make_run(run_id, event, *, sha=HEAD_SHA, conclusion="success"):
    return {
        "id": run_id,
        "head_sha": sha,
        "created_at": f"2026-10-14T00:00:{run_id:02d}Z",
        "event": event,
        "conclusion": conclusion,
    }


@pytest.mark.agent_authored(model="gpt-6")
@pytest.mark.parametrize("event,expected", [(None, 30), ("push", 20), ("workflow_dispatch", 10)])
def test_optional_event_filter_selects_latest_matching_successful_run(lookup, event, expected):
    result = lookup(
        {HEAD_SHA: [make_run(30, "schedule"), make_run(20, "push"), make_run(10, "workflow_dispatch")]},
        event=event,
    )
    assert result.returncode == 0, result.stderr
    assert result.stdout.splitlines() == [str(expected), HEAD_SHA]


@pytest.mark.agent_authored(model="gpt-6")
def test_push_lookup_continues_to_older_commit_after_scheduled_head_run(lookup):
    result = lookup(
        {
            HEAD_SHA: [make_run(30, "schedule")],
            OLDER_SHA: [make_run(20, "push", sha=OLDER_SHA)],
        },
        event="push",
        commits=[HEAD_SHA, OLDER_SHA],
    )
    assert result.returncode == 0, result.stderr
    assert result.stdout.splitlines() == ["20", OLDER_SHA]


@pytest.mark.agent_authored(model="gpt-6")
def test_push_lookup_rejects_failed_push_even_when_scheduled_run_succeeds(lookup):
    result = lookup(
        {HEAD_SHA: [make_run(30, "schedule"), make_run(20, "push", conclusion="failure")]},
        event="push",
    )
    assert result.returncode != 0
    assert result.stdout == ""
    assert "No successful" in result.stderr


@pytest.mark.agent_authored(model="gpt-6")
def test_event_filter_is_rejected_in_tag_mode(lookup):
    result = lookup({}, event="push", mode=["--tag", "v1.0.0"])
    assert result.returncode != 0
    assert "--event is only supported with --branch" in result.stderr


@pytest.mark.agent_authored(model="gpt-6")
def test_empty_event_is_rejected(lookup):
    result = lookup({}, event="")
    assert result.returncode != 0
    assert "Usage:" in result.stderr
