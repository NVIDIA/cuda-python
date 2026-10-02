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


def _option_value(option):
    try:
        return args[args.index(option) + 1]
    except (ValueError, IndexError):
        return None


def _field_values():
    values = []
    for index, arg in enumerate(args):
        if arg in ("-f", "--raw-field", "-F", "--field"):
            try:
                values.append(args[index + 1])
            except IndexError:
                print(f"{arg} requires a value", file=sys.stderr)
                raise SystemExit(2)
    return values


def _endpoint():
    skip_next = False
    for arg in args[1:]:
        if skip_next:
            skip_next = False
            continue
        if arg in (
            "-f",
            "--raw-field",
            "-F",
            "--field",
            "--jq",
            "--method",
            "-X",
        ):
            skip_next = True
            continue
        if arg.startswith("-"):
            continue
        return arg
    return None


if args[:2] == ["run", "list"]:
    runs = json.loads(os.environ["FAKE_RUNS"])
    try:
        limit = int(args[args.index("--limit") + 1])
    except (ValueError, IndexError):
        print("run list requires --limit", file=sys.stderr)
        raise SystemExit(2)
    if "--status" in args:
        print("status filters are intentionally unsupported", file=sys.stderr)
        raise SystemExit(2)
    if "--jq" in args:
        try:
            jq_filter = args[args.index("--jq") + 1]
        except IndexError:
            print("run list --jq requires a value", file=sys.stderr)
            raise SystemExit(2)
        if jq_filter == 'map(select(.status == "completed"))':
            runs = [run for run in runs if run["status"] == "completed"]
        else:
            print(f"unsupported jq filter: {jq_filter}", file=sys.stderr)
            raise SystemExit(2)
    print(json.dumps(runs[:limit]))
    raise SystemExit(0)

if args[:2] == ["run", "view"]:
    print(json.dumps({"url": "https://example.invalid/runs/view"}))
    raise SystemExit(0)

if args[:1] == ["api"]:
    endpoint = _endpoint()
    if endpoint and endpoint.endswith("/actions/workflows?per_page=100"):
        if "--paginate" not in args or "--jq" not in args:
            print("workflow lookup must be paginated and filtered", file=sys.stderr)
            raise SystemExit(2)
        for workflow in json.loads(os.environ["FAKE_WORKFLOWS"]):
            print(json.dumps({key: workflow[key] for key in ("id", "name", "path")}))
        raise SystemExit(0)

    if endpoint and "/actions/workflows/" in endpoint and endpoint.endswith("/runs"):
        workflow_match = re.search(r"/actions/workflows/([^/]+)/runs$", endpoint)
        if workflow_match is None:
            print(f"could not determine workflow ID: {endpoint}", file=sys.stderr)
            raise SystemExit(2)
        workflow_id = workflow_match.group(1)
        workflows = json.loads(os.environ["FAKE_WORKFLOWS"])
        workflow_ids = {str(workflow["id"]) for workflow in workflows}
        if workflow_id not in workflow_ids:
            print(f"unexpected workflow run endpoint: {endpoint}", file=sys.stderr)
            raise SystemExit(2)
        if _option_value("--method") != "GET":
            print("workflow run lookup must use GET", file=sys.stderr)
            raise SystemExit(2)
        fields = _field_values()
        if "branch=12.9.x" not in fields or "status=success" not in fields or "per_page=100" not in fields:
            print(f"unexpected workflow run fields: {fields!r}", file=sys.stderr)
            raise SystemExit(2)
        if "--jq" not in args:
            print("workflow run lookup must normalize with --jq", file=sys.stderr)
            raise SystemExit(2)
        runs = json.loads(os.environ["FAKE_REST_RUNS"])
        print(json.dumps(runs))
        raise SystemExit(0)

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


def _run(
    run_id,
    created_at,
    *,
    branch="12.9.x",
    workflow="CI",
    conclusion="success",
    event="push",
    status="completed",
):
    return {
        "databaseId": run_id,
        "workflowName": workflow,
        "status": status,
        "conclusion": conclusion,
        "headSha": f"sha-{run_id}",
        "headBranch": branch,
        "event": event,
        "createdAt": created_at,
        "url": f"https://example.invalid/runs/{run_id}",
    }


DEFAULT_WORKFLOWS = [
    {"id": 1001, "name": "CI", "path": ".github/workflows/ci.yml"},
    {
        "id": 1002,
        "name": "CI: Coverage",
        "path": ".github/workflows/coverage.yml",
    },
]


@pytest.fixture
def fake_gh(tmp_path):
    fake_bin = tmp_path / "bin"
    fake_bin.mkdir()
    gh = fake_bin / "gh"
    gh.write_text(FAKE_GH, encoding="utf-8")
    gh.chmod(0o755)
    return fake_bin


def _lookup(
    fake_gh,
    runs,
    artifacts,
    *args,
    workflow="CI",
    rest_runs=None,
    workflows=None,
):
    env = os.environ.copy()
    env.update(
        {
            "FAKE_ARTIFACTS": json.dumps(artifacts),
            "FAKE_REST_RUNS": json.dumps(runs if rest_runs is None else rest_runs),
            "FAKE_RUNS": json.dumps(runs),
            "FAKE_WORKFLOWS": json.dumps(
                DEFAULT_WORKFLOWS if workflows is None else workflows
            ),
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
    def test_filters_successful_runs_without_status_filter(self, fake_gh):
        runs = [
            _run(
                run_id,
                "2026-08-13T12:00:00Z",
                conclusion="failure",
            )
            for run_id in range(200, 190, -1)
        ]
        runs.append(_run(50, "2026-08-12T12:00:00Z"))

        result = _lookup(
            fake_gh,
            runs,
            {},
            "--branch",
            "12.9.x",
        )

        assert result.returncode == 0, result.stderr
        assert result.stdout.strip() == "50"

    def test_unions_direct_rest_runs_with_run_list_results(self, fake_gh):
        runs = [_run(100, "2026-03-10T12:00:00Z")]
        rest_runs = [
            _run(300, "2026-09-22T12:00:00Z"),
            _run(100, "2026-03-10T12:00:00Z"),
        ]
        artifacts = {
            "300": [{"name": "cuda-python-wheel", "expired": False}],
            "100": [{"name": "old-wheel", "expired": False}],
        }

        result = _lookup(
            fake_gh,
            runs,
            artifacts,
            "--branch",
            "12.9.x",
            "--artifact",
            "cuda-python-wheel",
            rest_runs=rest_runs,
        )

        assert result.returncode == 0, result.stderr
        assert result.stdout.strip() == "300"

    def test_prefers_successful_rest_duplicate_over_stale_run_list_record(self, fake_gh):
        runs = [
            _run(
                300,
                "2026-09-22T12:00:00Z",
                conclusion="",
                status="in_progress",
            )
        ]
        rest_runs = [_run(300, "2026-09-22T12:00:00Z")]

        result = _lookup(
            fake_gh,
            runs,
            {},
            "--branch",
            "12.9.x",
            rest_runs=rest_runs,
        )

        assert result.returncode == 0, result.stderr
        assert result.stdout.strip() == "300"

    def test_resolves_non_ci_workflow_display_name_for_rest_cross_check(self, fake_gh):
        runs = [_run(200, "2026-08-11T12:00:00Z", workflow="CI: Coverage")]

        result = _lookup(
            fake_gh,
            runs,
            {},
            "--branch",
            "12.9.x",
            workflow="CI: Coverage",
        )

        assert result.returncode == 0, result.stderr
        assert result.stdout.strip() == "200"

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


@pytest.mark.agent_authored(model="gpt-5.6")
class TestTagLookup:
    def test_filters_completed_runs_without_status_filter(self, fake_gh):
        runs = [
            _run(300, "2026-08-13T12:00:00Z", branch="HEAD", status="in_progress"),
            _run(200, "2026-08-12T12:00:00Z", branch="HEAD"),
            _run(100, "2026-08-11T12:00:00Z", branch="HEAD"),
        ]

        result = _lookup(fake_gh, runs, {}, "HEAD")

        assert result.returncode == 0, result.stderr
        assert result.stdout.strip() == "200"
