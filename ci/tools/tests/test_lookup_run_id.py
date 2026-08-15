# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import json
import os
import subprocess
from pathlib import Path

import pytest

LOOKUP_RUN_ID = Path(__file__).parent.parent / "lookup-run-id"

FAKE_GH = r"""#!/usr/bin/env python3
import json
import os
import re
import sys

args = sys.argv[1:]
if args[:2] == ["run", "list"]:
    print(os.environ["FAKE_RUNS"])
    raise SystemExit(0)

if args[:1] == ["api"]:
    if "--paginate" not in args or "--jq" not in args:
        print("artifact lookup must be paginated and filtered", file=sys.stderr)
        raise SystemExit(2)
    match = re.search(r"/runs/(\d+)/artifacts", " ".join(args))
    if match is None:
        print("could not determine run ID", file=sys.stderr)
        raise SystemExit(2)
    artifacts_by_run = json.loads(os.environ["FAKE_ARTIFACTS"])
    artifacts = artifacts_by_run.get(match.group(1))
    if artifacts is None:
        print("simulated artifact API failure", file=sys.stderr)
        raise SystemExit(3)
    for artifact in artifacts:
        if not artifact.get("expired", False):
            print(artifact["name"])
    raise SystemExit(0)

print(f"unexpected gh arguments: {args!r}", file=sys.stderr)
raise SystemExit(2)
"""


def _run(run_id, created_at, *, branch="12.9.x", workflow="CI", conclusion="success"):
    return {
        "databaseId": run_id,
        "workflowName": workflow,
        "status": "completed",
        "conclusion": conclusion,
        "headSha": f"sha-{run_id}",
        "headBranch": branch,
        "createdAt": created_at,
        "url": f"https://example.invalid/runs/{run_id}",
    }


@pytest.fixture
def fake_gh(tmp_path):
    fake_bin = tmp_path / "bin"
    fake_bin.mkdir()
    gh = fake_bin / "gh"
    gh.write_text(FAKE_GH, encoding="utf-8")
    gh.chmod(0o755)
    return fake_bin


def _lookup(fake_gh, runs, artifacts, *args, workflow="CI"):
    env = os.environ.copy()
    env.update(
        {
            "FAKE_ARTIFACTS": json.dumps(artifacts),
            "FAKE_RUNS": json.dumps(runs),
            "GH_TOKEN": "test-token",
            "PATH": f"{fake_gh}{os.pathsep}{env['PATH']}",
        }
    )
    return subprocess.run(  # noqa: S603 - invokes the repository script under test
        [str(LOOKUP_RUN_ID), *args, "NVIDIA/cuda-python", workflow],
        check=False,
        capture_output=True,
        env=env,
        text=True,
    )


@pytest.mark.agent_authored(model="gpt-5.6")
class TestBranchLookup:
    def test_selects_newest_run_with_filename_workflow_selector(self, fake_gh):
        runs = [
            _run(100, "2026-08-10T12:00:00Z"),
            _run(400, "2026-08-13T12:00:00Z", conclusion="failure"),
            _run(300, "2026-08-12T12:00:00Z", branch="other"),
            _run(200, "2026-08-11T12:00:00Z"),
        ]

        result = _lookup(
            fake_gh,
            runs,
            {},
            "--branch",
            "12.9.x",
            "--head-sha",
            workflow="ci.yml",
        )

        assert result.returncode == 0, result.stderr
        assert result.stdout.splitlines() == ["200", "sha-200"]

    def test_falls_back_until_all_required_artifacts_are_unexpired(self, fake_gh):
        bindings_pattern = "cuda-bindings-python315-cuda*-linux-64*[0-9a-f]"
        runs = [
            _run(300, "2026-08-13T12:00:00Z"),
            _run(200, "2026-08-12T12:00:00Z"),
            _run(100, "2026-08-11T12:00:00Z"),
        ]
        artifacts = {
            "300": [
                {
                    "name": "cuda-bindings-python315-cuda12.9.1-linux-64-abc123",
                    "expired": True,
                },
                {
                    "name": "cuda-bindings-python315-cuda12.9.1-linux-64-abc123-tests",
                    "expired": False,
                },
                {"name": "cuda-python-wheel", "expired": False},
            ],
            "200": [
                {
                    "name": "cuda-bindings-python315-cuda12.9.1-linux-64-def456",
                    "expired": False,
                }
            ],
            "100": [
                {
                    "name": "cuda-bindings-python315-cuda12.9.1-linux-64-fedcba",
                    "expired": False,
                },
                {"name": "cuda-python-wheel", "expired": False},
            ],
        }

        result = _lookup(
            fake_gh,
            runs,
            artifacts,
            "--branch",
            "12.9.x",
            "--artifact",
            bindings_pattern,
            "--artifact",
            "cuda-python-wheel",
        )

        assert result.returncode == 0, result.stderr
        assert result.stdout.strip() == "100"
        assert "Skipping run 300" in result.stderr
        assert "Skipping run 200" in result.stderr

    def test_reports_when_no_successful_run_has_required_artifacts(self, fake_gh):
        runs = [_run(100, "2026-08-11T12:00:00Z")]
        artifacts = {
            "100": [
                {
                    "name": "cuda-bindings-python315-cuda12.9.1-linux-64-fedcba",
                    "expired": True,
                }
            ]
        }

        result = _lookup(
            fake_gh,
            runs,
            artifacts,
            "--branch",
            "12.9.x",
            "--artifact",
            "cuda-bindings-python315-cuda*-linux-64*[0-9a-f]",
        )

        assert result.returncode == 1
        assert "has all required artifacts" in result.stderr

    def test_propagates_artifact_api_failures(self, fake_gh):
        runs = [_run(100, "2026-08-11T12:00:00Z")]

        result = _lookup(
            fake_gh,
            runs,
            {},
            "--branch",
            "12.9.x",
            "--artifact",
            "cuda-bindings-*",
        )

        assert result.returncode == 1
        assert "Failed to list artifacts for run 100" in result.stderr
