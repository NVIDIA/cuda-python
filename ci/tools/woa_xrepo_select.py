#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

import argparse
import datetime
import json
import os
import re
import sys
import urllib.error
import urllib.parse
import urllib.request


PUBLIC_REPOSITORY = "NVIDIA/cuda-python"
PUBLIC_REPOSITORY_ID = 381173759
PUBLIC_WORKFLOW_ID = 155304118
PUBLIC_WORKFLOW_PATH = ".github/workflows/ci.yml"
REPORTING_APP_ID = 4954254
CHECK_NAME = "cuda-python WoA integration"
MAX_LEDGER_COMMITS = 100


class GitHubAPI:
    def __init__(self, token):
        self._token = token

    def get(self, path, params=None):
        url = f"https://api.github.com/{path.lstrip('/')}"
        if params:
            url = f"{url}?{urllib.parse.urlencode(params)}"
        request = urllib.request.Request(
            url,
            headers={
                "Accept": "application/vnd.github+json",
                "Authorization": f"Bearer {self._token}",
                "X-GitHub-Api-Version": "2026-03-10",
            },
        )
        try:
            with urllib.request.urlopen(request) as response:
                return json.load(response)
        except urllib.error.HTTPError as error:
            body = error.read().decode("utf-8", errors="replace")
            raise RuntimeError(f"GitHub API GET {path} failed: {error.code}: {body}") from error

    def paginate(self, path, item_key, params=None, max_pages=10):
        items = []
        query = dict(params or {})
        query["per_page"] = 100
        for page in range(1, max_pages + 1):
            query["page"] = page
            response = self.get(path, query)
            page_items = response[item_key]
            items.extend(page_items)
            if len(page_items) < 100:
                break
        return items


def parse_time(value):
    return datetime.datetime.fromisoformat(value.replace("Z", "+00:00"))


def select_artifacts(artifacts, sha):
    binding_name = f"cuda-bindings-python313-cuda13.4.2-win-arm64-{sha}"
    core_name = f"cuda-core-python313-win-arm64-{sha}"
    selected = [
        artifact
        for artifact in artifacts
        if not artifact["expired"]
        and (
            artifact["name"] == "cuda-pathfinder-wheel"
                  or artifact["name"] == binding_name
            or artifact["name"] == core_name
        )
    ]
    if len(selected) != 3:
        return None
    names = {artifact["name"] for artifact in selected}
    if len(names) != 3:
        return None
    result = []
    for artifact in sorted(selected, key=lambda item: item["name"]):
        digest = artifact.get("digest")
        if not isinstance(digest, str) or not re.fullmatch(r"sha256:[0-9a-f]{64}", digest):
            return None
        result.append(
            {"id": artifact["id"], "name": artifact["name"], "digest": digest}
        )
    return result


def validate_run(api, run):
    if not (
        run["repository"]["id"] == PUBLIC_REPOSITORY_ID
        and run["workflow_id"] == PUBLIC_WORKFLOW_ID
        and run["path"] == PUBLIC_WORKFLOW_PATH
        and run["event"] == "push"
        and run["head_branch"] == "main"
        and run["status"] == "completed"
        and run["conclusion"] == "success"
        and re.fullmatch(r"[0-9a-f]{40}", run["head_sha"])
    ):
        return None

    run_id = run["id"]
    attempt = run["run_attempt"]
    jobs = api.paginate(
        f"repos/{PUBLIC_REPOSITORY}/actions/runs/{run_id}/attempts/{attempt}/jobs",
        "jobs",
    )
    woa_jobs = [job for job in jobs if job["name"].startswith("Build win-arm64, CUDA ")]
    if not woa_jobs or any(
        job["status"] != "completed" or job["conclusion"] != "success" for job in woa_jobs
    ):
        return None

    artifacts = api.paginate(
        f"repos/{PUBLIC_REPOSITORY}/actions/runs/{run_id}/artifacts", "artifacts"
    )
    selected_artifacts = select_artifacts(artifacts, run["head_sha"])
    if selected_artifacts is None:
        return None

    comparison = api.get(
        f"repos/{PUBLIC_REPOSITORY}/compare/{run['head_sha']}...main"
    )
    if comparison["status"] not in {"ahead", "identical"}:
        return None

    return {
        "run_id": str(run_id),
        "run_attempt": str(attempt),
        "sha": run["head_sha"],
        "artifacts": selected_artifacts,
    }


def resolve_candidate(api, run_id):
    if run_id:
        run = api.get(f"repos/{PUBLIC_REPOSITORY}/actions/runs/{run_id}")
        return validate_run(api, run)

    runs = api.get(
        f"repos/{PUBLIC_REPOSITORY}/actions/workflows/ci.yml/runs",
        {
            "branch": "main",
            "event": "push",
            "status": "completed",
            "per_page": 30,
        },
    )["workflow_runs"]
    for run in runs:
        candidate = validate_run(api, run)
        if candidate is not None:
            return candidate
    return None


def app_checks(api, sha):
    response = api.get(
        f"repos/{PUBLIC_REPOSITORY}/commits/{sha}/check-runs",
        {"check_name": CHECK_NAME, "per_page": 100},
    )
    return [
        check
        for check in response["check_runs"]
        if check["name"] == CHECK_NAME and check["app"]["id"] == REPORTING_APP_ID
    ]


def list_commits(api):
    return api.get(
        f"repos/{PUBLIC_REPOSITORY}/commits",
        {"sha": "main", "per_page": MAX_LEDGER_COMMITS},
    )


def select_batch(api, candidate, batch_size, max_wait_seconds, now):
    candidate_checks = app_checks(api, candidate["sha"])
    external_id = (
        f"woa:v1:{PUBLIC_REPOSITORY_ID}:{candidate['run_id']}:"
        f"{candidate['run_attempt']}:{candidate['sha']}"
    )
    if any(
        check.get("external_id") == external_id
        for check in candidate_checks
    ):
        return {"dispatch": False, "reason": "candidate already has an App-owned attempt"}

    commits = list_commits(api)
    try:
        candidate_index = next(
            index for index, commit in enumerate(commits) if commit["sha"] == candidate["sha"]
        )
    except StopIteration as error:
        raise RuntimeError(
            f"candidate SHA was not found on the latest {MAX_LEDGER_COMMITS} main commits"
        ) from error

    baseline_index = None
    baseline_sha = ""
    for index in range(candidate_index + 1, len(commits)):
        checks = app_checks(api, commits[index]["sha"])
        successful = [
            check
            for check in checks
            if check["status"] == "completed" and check["conclusion"] == "success"
        ]
        if successful:
            baseline_index = index
            baseline_sha = commits[index]["sha"]
            break

    if baseline_index is None:
        return {
            "dispatch": True,
            "reason": (
                "bootstrap: no successful App-owned baseline exists in the bounded ledger"
            ),
            "baseline_sha": "",
            "commit_count": 1,
        }

    commit_count = baseline_index - candidate_index
    oldest_pending = commits[baseline_index - 1]
    oldest_time = parse_time(oldest_pending["commit"]["committer"]["date"])
    age_seconds = max(0, int((now - oldest_time).total_seconds()))
    if commit_count >= batch_size:
        reason = f"batch threshold reached: {commit_count} commits"
        dispatch = True
    elif age_seconds >= max_wait_seconds:
        reason = f"maximum wait reached: oldest pending commit is {age_seconds} seconds old"
        dispatch = True
    else:
        reason = (
            f"deferred: {commit_count}/{batch_size} commits and oldest pending age "
            f"{age_seconds}/{max_wait_seconds} seconds"
        )
        dispatch = False
    return {
        "dispatch": dispatch,
        "reason": reason,
        "baseline_sha": baseline_sha,
        "commit_count": commit_count,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-id", default="")
    parser.add_argument("--batch-size", type=int, default=3)
    parser.add_argument("--max-wait-seconds", type=int, default=7200)
    args = parser.parse_args()
    if args.run_id and not args.run_id.isdigit():
        parser.error("--run-id must be numeric")
    if args.batch_size < 1 or args.max_wait_seconds < 1:
        parser.error("batch size and maximum wait must be positive")

    token = os.environ.get("GITHUB_TOKEN")
    if not token:
        raise SystemExit("GITHUB_TOKEN is required")
    api = GitHubAPI(token)
    candidate = resolve_candidate(api, args.run_id)
    if candidate is None:
        print(json.dumps({"dispatch": False, "reason": "no eligible public WoA build"}))
        return

    selection = select_batch(
        api,
        candidate,
        args.batch_size,
        args.max_wait_seconds,
        datetime.datetime.now(datetime.timezone.utc),
    )
    selection.update(candidate)
    selection["correlation_id"] = (
        f"v1:{PUBLIC_REPOSITORY_ID}:{candidate['run_id']}:"
        f"{candidate['run_attempt']}:{candidate['sha']}"
    )
    print(json.dumps(selection, separators=(",", ":"), sort_keys=True))


if __name__ == "__main__":
    try:
        main()
    except Exception as error:
        print(f"error: {error}", file=sys.stderr)
        raise
