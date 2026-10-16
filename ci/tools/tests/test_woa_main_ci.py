# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

# ruff: noqa: S603

import json
import os
import shutil
import subprocess
import sys
from pathlib import Path

import pytest
import yaml

ROOT = Path(__file__).resolve().parents[3]
WORKFLOW = yaml.safe_load((ROOT / ".github/workflows/woa-main-ci.yml").read_text())
STEPS = WORKFLOW["jobs"]["resolve-public-build"]["steps"]
SHA = "a" * 40


@pytest.fixture
def run_step(tmp_path):
    mock = tmp_path / "gh"
    mock.write_text(
        f"#!{sys.executable}\n"
        "import json, os, sys\n"
        "from pathlib import Path\n"
        "args = sys.argv[1:]\n"
        "endpoint = next(arg for arg in args if arg.startswith('repos/'))\n"
        "run = json.loads(os.environ['RUN'])\n"
        "if endpoint.endswith('/ci.yml/runs'):\n"
        "    print(json.dumps({'workflow_runs': [dict(run, id=999, event='schedule'), "
        "dict(run, id=998, head_sha='b' * 40), run]}))\n"
        "elif endpoint.endswith('/artifacts'):\n"
        "    name = next(arg[5:] for arg in args if arg.startswith('name='))\n"
        "    with Path(os.environ['REQUESTS']).open('a') as output:\n"
        "        output.write(name + '\\n')\n"
        "    ready = os.environ['MODE'] != 'missing' and "
        "(os.environ['MODE'] != 'delayed' or len(Path(os.environ['REQUESTS']).read_text().splitlines()) > 1)\n"
        "    print(json.dumps({'total_count': int(ready), 'artifacts': "
        "[{'name': name, 'expired': False}] if ready else []}))\n"
        "else:\n"
        "    print(json.dumps(run))\n"
    )
    mock.chmod(0o755)
    for name, body in {
        "sleep": "exit 0",
        "yq": "case $* in *supported*) echo 3.11; echo 3.14t;; *) echo 13.4.2;; esac",
    }.items():
        executable = tmp_path / name
        executable.write_text(f"#!/bin/bash\n{body}\n")
        executable.chmod(0o755)

    def invoke(name, *, mode="ready", status="in_progress", conclusion=None):
        run = {
            "id": 123,
            "repository": {"full_name": "NVIDIA/cuda-python"},
            "workflow_id": 155304118,
            "path": ".github/workflows/ci.yml",
            "event": "push",
            "head_branch": "main",
            "head_sha": SHA,
            "status": status,
            "conclusion": conclusion,
        }
        result = subprocess.run(
            [shutil.which("bash"), "-c", next(step["run"] for step in STEPS if step["name"] == name)],
            env=dict(
                os.environ,
                PATH=str(tmp_path) + os.pathsep + os.environ["PATH"],
                PUBLIC_RUN_ID="123",
                PUBLIC_SHA=SHA,
                RUN=json.dumps(run),
                MODE=mode,
                REQUESTS=str(tmp_path / "requests"),
                GITHUB_OUTPUT=str(tmp_path / "outputs"),
                GITHUB_STEP_SUMMARY=str(tmp_path / "summary"),
            ),
            capture_output=True,
            text=True,
            timeout=10,
        )
        return result, tmp_path

    return invoke


@pytest.mark.agent_authored(model="gpt-6")
def test_entry_is_main_push_only():
    assert WORKFLOW[True] == {"push": {"branches": ["main"]}}
    ci = yaml.safe_load((ROOT / ".github/workflows/ci.yml").read_text())
    assert "woa-main-dispatch" not in ci["jobs"]
    assert WORKFLOW["concurrency"]["cancel-in-progress"] is False


@pytest.mark.agent_authored(model="gpt-6")
def test_resolver_selects_exact_push_sha_not_nightly(run_step):
    result, directory = run_step("Resolve exact public build", status="queued")
    assert result.returncode == 0, result.stderr
    assert "public_run_id=123\n" in (directory / "outputs").read_text()
    assert f"public_sha={SHA}\n" in (directory / "outputs").read_text()


@pytest.mark.agent_authored(model="gpt-6")
@pytest.mark.parametrize("mode", ["ready", "delayed"])
def test_waiter_accepts_ready_wheels_before_ci_finishes(run_step, mode):
    result, directory = run_step("Wait for public WoA wheel artifacts", mode=mode)
    assert result.returncode == 0, result.stderr
    expected = [
        "cuda-pathfinder-wheel",
        f"cuda-bindings-python311-cuda13.4.2-win-arm64-{SHA}",
        f"cuda-core-python311-win-arm64-{SHA}",
        f"cuda-bindings-python314t-cuda13.4.2-win-arm64-{SHA}",
        f"cuda-core-python314t-win-arm64-{SHA}",
    ]
    if mode == "delayed":
        expected.insert(0, "cuda-pathfinder-wheel")
    assert (directory / "requests").read_text().splitlines() == expected


@pytest.mark.agent_authored(model="gpt-6")
@pytest.mark.parametrize("conclusion", ["failure", "cancelled", "success"])
def test_waiter_rejects_unsuccessful_ci_or_missing_wheels(run_step, conclusion):
    result, _ = run_step(
        "Wait for public WoA wheel artifacts", mode="missing", status="completed", conclusion=conclusion
    )
    assert result.returncode != 0
    assert "::error::" in result.stdout
