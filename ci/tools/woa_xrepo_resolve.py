#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

import argparse
import http.client
import json
import os
import re
import sys
import urllib.parse

PUBLIC_REPOSITORY = "NVIDIA/cuda-python"
PUBLIC_REPOSITORY_ID = 381173759
PUBLIC_WORKFLOW_ID = 155304118
PUBLIC_WORKFLOW_PATH = ".github/workflows/ci.yml"
REPORTING_APP_ID = 4954254
CHECK_NAME = "cuda-python WoA integration"
CUDA_BUILD_VERSION = "13.4.2"
PYTHON_VERSIONS = ("3.11", "3.12", "3.13", "3.14", "3.14t", "3.15", "3.15t")


class GitHubAPI:
    def __init__(self, token):
        self._token = token

    def get(self, path, params=None):
        request_path = f"/{path.lstrip('/')}"
        if params:
            request_path = f"{request_path}?{urllib.parse.urlencode(params)}"
        connection = http.client.HTTPSConnection("api.github.com")
        try:
            connection.request(
                "GET",
                request_path,
                headers={
                    "Accept": "application/vnd.github+json",
                    "Authorization": f"Bearer {self._token}",
                    "X-GitHub-Api-Version": "2026-03-10",
                    "User-Agent": "cuda-python-woa-resolver",
                },
            )
            response = connection.getresponse()
            body = response.read().decode("utf-8", errors="replace")
            if response.status >= 400:
                raise RuntimeError(f"GitHub API GET {path} failed: {response.status}: {body}")
            return json.loads(body)
        finally:
            connection.close()

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


def expected_producer_names():
    names = [f"Build linux-64, CUDA {CUDA_BUILD_VERSION} / py3.10"]
    names.extend(f"Build win-arm64, CUDA {CUDA_BUILD_VERSION} / py{version}" for version in PYTHON_VERSIONS)
    return names


def expected_artifact_names(sha):
    names = ["cuda-pathfinder-wheel"]
    for version in PYTHON_VERSIONS:
        python_tag = version.replace(".", "")
        names.extend(
            (
                f"cuda-bindings-python{python_tag}-cuda{CUDA_BUILD_VERSION}-win-arm64-{sha}",
                f"cuda-core-python{python_tag}-win-arm64-{sha}",
            )
        )
    return names


def select_artifacts(api, run_id, sha):
    result = []
    for name in expected_artifact_names(sha):
        response = api.get(
            f"repos/{PUBLIC_REPOSITORY}/actions/runs/{run_id}/artifacts",
            {"name": name, "per_page": 100},
        )
        artifacts = response["artifacts"]
        if response["total_count"] != 1 or len(artifacts) != 1:
            return None
        artifact = artifacts[0]
        digest = artifact.get("digest")
        if not (
            artifact["name"] == name
            and artifact["expired"] is False
            and isinstance(artifact["id"], int)
            and artifact["id"] > 0
            and isinstance(digest, str)
            and re.fullmatch(r"sha256:[0-9a-f]{64}", digest)
        ):
            return None
        result.append({"id": artifact["id"], "name": artifact["name"], "digest": digest})
    return sorted(result, key=lambda item: item["name"])


def run_is_admissible(run):
    state_is_admissible = (run["status"] == "in_progress" and run["conclusion"] is None) or (
        run["status"] == "completed" and run["conclusion"] == "success"
    )
    return (
        run["repository"]["id"] == PUBLIC_REPOSITORY_ID
        and run["workflow_id"] == PUBLIC_WORKFLOW_ID
        and run["path"] == PUBLIC_WORKFLOW_PATH
        and run["event"] == "push"
        and run["head_branch"] == "main"
        and state_is_admissible
        and re.fullmatch(r"[0-9a-f]{40}", run["head_sha"])
    )


def validate_run(api, run):
    if not run_is_admissible(run):
        return None

    run_id = run["id"]
    attempt = run["run_attempt"]
    jobs = api.paginate(
        f"repos/{PUBLIC_REPOSITORY}/actions/runs/{run_id}/attempts/{attempt}/jobs",
        "jobs",
    )
    producer_jobs = []
    for name in expected_producer_names():
        matches = [job for job in jobs if job["name"] == name]
        if not (len(matches) == 1 and matches[0]["status"] == "completed" and matches[0]["conclusion"] == "success"):
            return None
        producer_jobs.append({"id": matches[0]["id"], "name": name})

    selected_artifacts = select_artifacts(api, run_id, run["head_sha"])
    if selected_artifacts is None:
        return None

    comparison = api.get(f"repos/{PUBLIC_REPOSITORY}/compare/{run['head_sha']}...main")
    if comparison["status"] not in {"ahead", "identical"}:
        return None

    return {
        "run_id": str(run_id),
        "run_attempt": str(attempt),
        "producer_jobs": sorted(producer_jobs, key=lambda item: item["name"]),
        "sha": run["head_sha"],
        "artifacts": selected_artifacts,
        "correlation_id": f"v1:{PUBLIC_REPOSITORY_ID}:{run_id}:{attempt}:{run['head_sha']}",
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


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-id", required=True)
    args = parser.parse_args()
    if not args.run_id.isdigit():
        parser.error("--run-id must be numeric")

    token = os.environ.get("GITHUB_TOKEN")
    if not token:
        raise SystemExit("GITHUB_TOKEN is required")
    candidate = resolve_candidate(GitHubAPI(token), args.run_id)
    if candidate is None:
        raise RuntimeError("the requested public run is not ready for WoA validation")
    print(json.dumps(candidate, separators=(",", ":"), sort_keys=True))


if __name__ == "__main__":
    try:
        main()
    except Exception as error:
        print(f"error: {error}", file=sys.stderr)
        raise
