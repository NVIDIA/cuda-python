# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Run the freshness workflow against local Git history and a mock Pixi."""

from __future__ import annotations

import os
import shutil
import subprocess
from pathlib import Path

import pytest
import yaml

ROOT = Path(__file__).resolve().parents[3]
WORKFLOW = ROOT / ".github/workflows/ci-pixi-lockfile-freshness-check.yml"


def _git(repo, *args):
    return subprocess.run(  # noqa: S603 - arguments are passed without a shell.
        ["git", "-C", str(repo), *args],  # noqa: S607
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()


def _commit(repo):
    _git(repo, "add", ".")
    _git(repo, "-c", "user.name=Test", "-c", "user.email=test@example.com", "commit", "-qm", "test")
    return _git(repo, "rev-parse", "HEAD")


def _run_check(tmp_path, manifest, changed_manifest, *, stale=True, known_defect=False):
    repo = tmp_path / "repo"
    repo.mkdir()
    _git(repo, "init", "-q")
    _git(repo, "config", "commit.gpgsign", "false")
    tools = repo / "ci/tools"
    tools.mkdir(parents=True)
    for name in ("list_pixi_workspaces.py", "classify_pixi_lockfile_freshness.py"):
        shutil.copyfile(ROOT / "ci/tools" / name, tools / name)
    # The exact defect detector is covered separately; exercise its workflow gate.
    (tools / "check_pixi_samples_source_pruning.py").write_text(
        f"raise SystemExit({0 if known_defect else 1})\n", encoding="utf-8"
    )
    for workspace in {".", manifest, changed_manifest}:
        directory = repo / workspace
        directory.mkdir(parents=True, exist_ok=True)
        (directory / "pixi.toml").write_text('[workspace]\nchannels = ["conda-forge"]\n', encoding="utf-8")
        content = "stale\n" if stale and workspace == manifest else "fresh\n"
        (directory / "pixi.lock").write_text(content, encoding="utf-8")
    base_sha = _commit(repo)
    with (repo / changed_manifest / "pixi.toml").open("a", encoding="utf-8") as stream:
        stream.write('\n[tasks]\nexample = "echo example"\n')
    _commit(repo)

    bin_dir = tmp_path / "bin"
    bin_dir.mkdir()
    pixi = bin_dir / "pixi"
    pixi.write_text(
        "#!/usr/bin/env python3\n"
        "import sys\n"
        "from pathlib import Path\n"
        "lockfile = Path(sys.argv[-1]) / 'pixi.lock'\n"
        "if lockfile.read_text() == 'stale\\n':\n"
        "    lockfile.write_text('repaired\\n')\n"
        "    sys.exit(1)\n",
        encoding="utf-8",
    )
    pixi.chmod(0o755)
    summary = tmp_path / "summary.md"
    output = tmp_path / "output.txt"
    workflow = yaml.safe_load(WORKFLOW.read_text(encoding="utf-8"))
    step = next(step for step in workflow["jobs"]["lockfile-fresh"]["steps"] if step.get("id") == "check")
    result = subprocess.run(  # noqa: S603 - run trusted workflow code with local fixtures and no network.
        ["bash", "--noprofile", "--norc", "-euo", "pipefail", "-c", step["run"]],  # noqa: S607
        cwd=repo,
        env={
            **os.environ,
            "PATH": f"{bin_dir}{os.pathsep}{os.environ['PATH']}",
            "BASE_SHA": base_sha,
            "EVENT_NAME": "pull_request",
            "MAINTENANCE_KIND": "workflow",
            "MAINTENANCE_URL": "https://example.com/refresh",
            "PIXI_VERSION": "v0.73.0",
            "WORKSPACE_TIMEOUT": "10s",
            "RUNNER_TEMP": str(tmp_path),
            "GITHUB_STEP_SUMMARY": str(summary),
            "GITHUB_OUTPUT": str(output),
        },
        capture_output=True,
        text=True,
        timeout=30,
    )
    return result, summary.read_text(encoding="utf-8")


@pytest.mark.parametrize("manifest", [".", "nested/workspace", "cuda_core"])
@pytest.mark.agent_authored(model="gpt-6")
def test_manifest_edit_without_lock_update_reports_pr_remediation(tmp_path, manifest):
    result, summary = _run_check(tmp_path, manifest, manifest, known_defect=manifest == "cuda_core")

    assert result.returncode == 1, result.stdout + result.stderr
    assert "manifest changed without a lockfile update" in summary
    assert "regenerate and commit the lockfile in this PR" in summary
    assert "Base maintenance:" not in summary
    assert "Do not add this refresh" not in result.stdout
    assert "Known inherited Pixi samples lockfile bug" not in result.stdout


@pytest.mark.agent_authored(model="gpt-6")
def test_other_workspace_manifest_edit_keeps_base_maintenance_attribution(tmp_path):
    result, summary = _run_check(tmp_path, "nested/workspace", ".")

    assert result.returncode == 1, result.stdout + result.stderr
    assert "Base maintenance:" in summary
    assert "PR-induced:" not in summary


@pytest.mark.agent_authored(model="gpt-6")
def test_fresh_lockfile_allows_task_only_manifest_edit(tmp_path):
    result, summary = _run_check(tmp_path, "nested/workspace", "nested/workspace", stale=False)

    assert result.returncode == 0, result.stdout + result.stderr
    assert "| `nested/workspace` | Fresh |" in summary
