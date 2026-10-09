#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

import argparse
import datetime
import json
import os
import sys

from woa_xrepo_resolve import (
    CHECK_NAME,
    PUBLIC_REPOSITORY,
    REPORTING_APP_ID,
    GitHubAPI,
    resolve_candidate,
)

MAX_LEDGER_COMMITS = 100


def parse_time(value):
    return datetime.datetime.fromisoformat(value.replace("Z", "+00:00"))


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
    external_id = f"woa:{candidate['correlation_id']}"
    if any(check.get("external_id") == external_id for check in candidate_checks):
        return {"dispatch": False, "reason": "candidate already has an App-owned attempt"}

    commits = list_commits(api)
    try:
        candidate_index = next(index for index, commit in enumerate(commits) if commit["sha"] == candidate["sha"])
    except StopIteration as error:
        raise RuntimeError(f"candidate SHA was not found on the latest {MAX_LEDGER_COMMITS} main commits") from error

    baseline_index = None
    baseline_sha = ""
    for index in range(candidate_index + 1, len(commits)):
        checks = app_checks(api, commits[index]["sha"])
        successful = [check for check in checks if check["status"] == "completed" and check["conclusion"] == "success"]
        if successful:
            baseline_index = index
            baseline_sha = commits[index]["sha"]
            break

    if baseline_index is None:
        return {
            "dispatch": True,
            "reason": "bootstrap: no successful App-owned baseline exists in the bounded ledger",
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
    print(json.dumps(selection, separators=(",", ":"), sort_keys=True))


if __name__ == "__main__":
    try:
        main()
    except Exception as error:
        print(f"error: {error}", file=sys.stderr)
        raise
