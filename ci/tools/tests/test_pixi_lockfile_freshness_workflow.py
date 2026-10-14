# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Validate the temporary lockfile maintenance pause and its check contexts."""

from __future__ import annotations

import shlex
import subprocess
from pathlib import Path

import pytest
import yaml

ROOT = Path(__file__).resolve().parents[3]
WORKFLOWS = ROOT / ".github/workflows"
PAUSED_WORKFLOWS = (
    ("ci-pixi-lockfile-freshness-check.yml", "CI: pixi lockfile freshness check", "pixi lock --check (all workspaces)"),
    ("ci-pixi-lockfile-refresh.yml", "CI: pixi lockfile refresh", "pixi update (all workspaces)"),
)


def _workflow(filename):
    workflow = yaml.safe_load((WORKFLOWS / filename).read_text(encoding="utf-8"))
    # PyYAML interprets Actions' unquoted `on` key as a YAML 1.1 boolean.
    if True in workflow:
        workflow["on"] = workflow.pop(True)
    return workflow


@pytest.mark.agent_authored(model="gpt-6")
def test_freshness_notice_runs_on_every_pr_without_path_filters():
    workflow = _workflow("ci-pixi-lockfile-freshness-check.yml")

    assert workflow["on"]["pull_request"] == {}
    assert "pull_request_target" not in workflow["on"]
    assert workflow["on"]["workflow_dispatch"] == {}


@pytest.mark.agent_authored(model="gpt-6")
def test_refresh_is_manual_only_during_pause():
    workflow = _workflow("ci-pixi-lockfile-refresh.yml")

    assert workflow["on"] == {"workflow_dispatch": {}}


@pytest.mark.parametrize(("filename", "workflow_name", "job_name"), PAUSED_WORKFLOWS)
@pytest.mark.agent_authored(model="gpt-6")
def test_paused_workflows_preserve_names_and_run_only_the_notice(filename, workflow_name, job_name):
    workflow = _workflow(filename)

    assert workflow["name"] == workflow_name
    assert workflow["permissions"] == {}
    assert len(workflow["jobs"]) == 1
    job = next(iter(workflow["jobs"].values()))
    assert job["name"] == job_name
    assert not {"if", "needs", "uses", "strategy", "continue-on-error", "permissions"}.intersection(job)
    assert len(job["steps"]) == 1
    assert set(job["steps"][0]) == {"name", "run"}


@pytest.mark.parametrize(("filename", "workflow_name", "job_name"), PAUSED_WORKFLOWS)
@pytest.mark.agent_authored(model="gpt-6")
def test_suspension_notice_succeeds_without_external_tools_or_repository_writes(
    tmp_path, filename, workflow_name, job_name
):
    workflow = _workflow(filename)
    job = next(iter(workflow["jobs"].values()))
    summary = tmp_path / "summary.md"
    result = subprocess.run(  # noqa: S603 - execute the trusted notice with no external tools in PATH.
        ["/bin/bash", "--noprofile", "--norc", "-euo", "pipefail", "-c", job["steps"][0]["run"]],
        cwd=tmp_path,
        env={"PATH": str(tmp_path / "no-tools"), "GITHUB_STEP_SUMMARY": str(summary)},
        capture_output=True,
        text=True,
        timeout=10,
    )

    assert result.returncode == 0, result.stdout + result.stderr
    assert "::warning title=Pixi lockfile maintenance suspended::" in result.stdout
    assert "temporarily suspended" in result.stdout
    assert "Freshness was not checked." in result.stdout
    text = summary.read_text(encoding="utf-8")
    assert "Freshness enforcement and automated refreshes are temporarily suspended." in text
    assert "does not verify manifest/lock consistency" in text
    assert "manual review and validation" in text
    assert "PIXI_FROZEN=true" in text
    assert set(tmp_path.iterdir()) == {summary}


@pytest.mark.agent_authored(model="gpt-6")
def test_both_workflows_emit_the_same_suspension_notice():
    notices = [
        next(iter(_workflow(filename)["jobs"].values()))["steps"][0]["run"] for filename, _, _ in PAUSED_WORKFLOWS
    ]

    assert notices[0] == notices[1]


@pytest.mark.agent_authored(model="gpt-6")
def test_source_builds_and_tests_still_use_frozen_pixi():
    workflow = _workflow("ci-pixi-source-test.yml")

    assert workflow["env"]["PIXI_FROZEN"] == "true"
    assert "PIXI_LOCKED" not in workflow["env"]
    assert set(workflow["jobs"]) == {"build-smoke", "build-identity-roundtrip", "full-test"}
    assert {"pull_request", "schedule", "workflow_dispatch"} <= set(workflow["on"])
    for job in workflow["jobs"].values():
        assert any(step.get("uses") == "./.github/actions/setup-pixi" for step in job["steps"])
        assert any("pixi run" in step.get("run", "") for step in job["steps"])


@pytest.mark.agent_authored(model="gpt-6")
def test_nightly_ci_declares_workflow_test_dependencies():
    workflow = _workflow("ci-nightly.yml")
    step = next(step for step in workflow["jobs"]["test-ci-tools-for-release"]["steps"] if "run" in step)
    install = next(line for line in step["run"].splitlines() if "pip install" in line)
    dependencies = {argument.lower().split("==")[0] for argument in shlex.split(install)}

    assert {"pytest", "pyyaml"} <= dependencies
