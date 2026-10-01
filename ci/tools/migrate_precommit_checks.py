# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Preview the main ruleset cutover from pre-commit.ci to GitHub Actions.

Writes require --apply, merged workflows, and successful representative checks.
This tool does not uninstall the pre-commit.ci GitHub App.
"""

from __future__ import annotations

import argparse
import copy
import json
import re
import subprocess
import sys
from urllib.parse import urlparse

OLD_CONTEXT = "pre-commit.ci - pr"
OLD_INTEGRATION = 68672
ACTIONS_INTEGRATION = 15368
CHECK_WORKFLOWS = {
    "Pre-commit (Linux)": ".github/workflows/pre-commit.yml",
    "Pre-commit (Windows)": ".github/workflows/pre-commit.yml",
    "Documentation links": ".github/workflows/lychee.yml",
}


def github_api(endpoint: str, *, method: str = "GET", payload: dict | None = None, paginate: bool = False):
    """Call gh with structured JSON input, without invoking a shell."""
    command = ["gh", "api", "--method", method, endpoint, "-H", "Accept: application/vnd.github+json"]
    if paginate:
        command.extend(["--paginate", "--slurp"])
    if payload is not None:
        command.extend(["--input", "-"])
    result = subprocess.run(  # noqa: S603 - arguments are passed directly, without a shell.
        command,
        input=json.dumps(payload) if payload is not None else None,
        capture_output=True,
        text=True,
        check=True,
    )
    return json.loads(result.stdout)


def migrate_rules(rules: list[dict]) -> list[dict]:
    """Replace only the known pre-commit.ci requirement, preserving all other rules."""
    migrated = copy.deepcopy(rules)
    for rule in migrated:
        if rule["type"] != "required_status_checks":
            continue
        checks = rule["parameters"]["required_status_checks"]
        old_checks = [check for check in checks if check["context"] == OLD_CONTEXT]
        if not old_checks:
            continue
        if any(check.get("integration_id") != OLD_INTEGRATION for check in old_checks):
            raise RuntimeError(f"{OLD_CONTEXT!r} is not bound to expected integration {OLD_INTEGRATION}")
        existing = set()
        for check in checks:
            context = check["context"]
            if context in CHECK_WORKFLOWS:
                if check.get("integration_id") != ACTIONS_INTEGRATION:
                    raise RuntimeError(f"{context!r} already exists with a different or unspecified integration")
                existing.add(context)
        replacement = [
            {"context": context, "integration_id": ACTIONS_INTEGRATION}
            for context in CHECK_WORKFLOWS
            if context not in existing
        ]
        updated = []
        for check in checks:
            if check["context"] == OLD_CONTEXT:
                updated.extend(replacement)
                replacement = []
            else:
                updated.append(check)
        rule["parameters"]["required_status_checks"] = updated
    return migrated


def discover_changes(repository: str) -> list[dict]:
    """Find active requirements on main and fetch their complete repository rulesets."""
    pages = github_api(f"repos/{repository}/rules/branches/main?per_page=100", paginate=True)
    ruleset_ids = set()
    for rule in (rule for page in pages for rule in page):
        if rule["type"] != "required_status_checks":
            continue
        checks = rule["parameters"]["required_status_checks"]
        if not any(check["context"] == OLD_CONTEXT for check in checks):
            continue
        if rule["ruleset_source_type"] != "Repository" or rule["ruleset_source"].lower() != repository.lower():
            raise RuntimeError(
                f"Requirement is inherited from {rule['ruleset_source_type']} {rule['ruleset_source']} "
                f"ruleset {rule['ruleset_id']}; its owner must migrate it. No repository changes were made."
            )
        ruleset_ids.add(rule["ruleset_id"])

    changes = []
    for ruleset_id in sorted(ruleset_ids):
        endpoint = f"repos/{repository}/rulesets/{ruleset_id}"
        original = github_api(endpoint)
        if original["source_type"] != "Repository" or original["source"].lower() != repository.lower():
            raise RuntimeError(f"Ruleset {ruleset_id} is not owned by {repository}")
        if original["target"] != "branch" or original["enforcement"] != "active":
            raise RuntimeError(f"Ruleset {ruleset_id} changed scope or enforcement; rerun the preview")
        migrated = migrate_rules(original["rules"])
        if migrated != original["rules"]:
            changes.append({"endpoint": endpoint, "original": original, "payload": {"rules": migrated}})
    return changes


def verify_readiness(repository: str, verified_sha: str | None) -> str:
    """Require merged workflow files and green checks on main or an identified main PR."""
    metadata = github_api(f"repos/{repository}")
    if metadata["default_branch"] != "main":
        raise RuntimeError("This migration is scoped to repositories whose default branch is main")
    main_sha = github_api(f"repos/{repository}/branches/main")["commit"]["sha"]
    for path in sorted(set(CHECK_WORKFLOWS.values())):
        content = github_api(f"repos/{repository}/contents/{path}?ref={main_sha}")
        if not isinstance(content, dict) or content.get("type") != "file":
            raise RuntimeError(f"{path} is not a workflow file on current main")

    sha = verified_sha or main_sha
    if sha != main_sha:
        pages = github_api(f"repos/{repository}/commits/{sha}/pulls?per_page=100", paginate=True)
        matching_prs = [
            pr
            for page in pages
            for pr in page
            if pr["state"] == "open"
            and pr["base"]["ref"] == "main"
            and pr["base"]["repo"]["full_name"].lower() == repository.lower()
            and sha in (pr["head"]["sha"], pr.get("merge_commit_sha"))
        ]
        if not matching_prs:
            raise RuntimeError("--verified-sha must be current main or the head/merge SHA of an open PR targeting main")

    pages = github_api(f"repos/{repository}/commits/{sha}/check-runs?filter=latest&per_page=100", paginate=True)
    checks = [check for page in pages for check in page["check_runs"]]
    runs = {}
    for context, workflow in CHECK_WORKFLOWS.items():
        matching = [check for check in checks if check["name"] == context and check["app"]["id"] == ACTIONS_INTEGRATION]
        if not matching:
            raise RuntimeError(f"Missing GitHub Actions check {context!r} on {sha}")
        check = max(matching, key=lambda check: check["id"])
        if check["status"] != "completed" or check["conclusion"] != "success":
            raise RuntimeError(f"Check {context!r} on {sha} is not completed and successful")
        url = urlparse(check["details_url"])
        match = re.fullmatch(rf"/{re.escape(repository)}/actions/runs/([0-9]+)/job/[0-9]+", url.path, re.IGNORECASE)
        if url.hostname != "github.com" or match is None:
            raise RuntimeError(f"Cannot identify the workflow run for {context!r}")
        run_id = match.group(1)
        if run_id not in runs:
            runs[run_id] = github_api(f"repos/{repository}/actions/runs/{run_id}")
        run = runs[run_id]
        if (
            run["path"].split("@", 1)[0] != workflow
            or run["check_suite_id"] != check["check_suite"]["id"]
            or run["status"] != "completed"
            or run["conclusion"] != "success"
        ):
            raise RuntimeError(f"Check {context!r} does not belong to a successful {workflow} run")
    return sha


def apply_changes(repository: str, changes: list[dict], verified_sha: str | None) -> str:
    """Preflight all changes before issuing narrowly scoped rules updates."""
    sha = verify_readiness(repository, verified_sha)
    for change in changes:
        if github_api(change["endpoint"]) != change["original"]:
            raise RuntimeError(f"{change['endpoint']} changed after discovery; rerun the preview")
    for change in changes:
        updated = github_api(change["endpoint"], method="PUT", payload=change["payload"])
        original = change["original"]
        settings = ("name", "target", "enforcement", "conditions", "bypass_actors")
        if updated["rules"] != change["payload"]["rules"] or any(
            updated.get(key) != original.get(key) for key in settings
        ):
            raise RuntimeError(f"Unexpected response after updating {change['endpoint']}; inspect that ruleset")
    return sha


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo", required=True, choices=("NVIDIA/cuda-python", "NVIDIA-dev/cuda-python-private"))
    parser.add_argument("--verified-sha", help="Full SHA of current main or a representative open PR head/merge commit")
    parser.add_argument(
        "--apply", action="store_true", help="Update required checks after explicit cutover authorization"
    )
    args = parser.parse_args(argv)
    if args.verified_sha and re.fullmatch(r"[0-9a-f]{40}", args.verified_sha) is None:
        parser.error("--verified-sha must be a full 40-character lowercase commit SHA")

    try:
        changes = discover_changes(args.repo)
        preview = {
            "repository": args.repo,
            "branch": "main",
            "mode": "apply" if args.apply else "dry-run",
            "changes": [
                {
                    "endpoint": change["endpoint"],
                    "before": change["original"],
                    "put_payload": change["payload"],
                }
                for change in changes
            ],
        }
        print(json.dumps(preview, indent=2))
        if args.apply and changes:
            sha = apply_changes(args.repo, changes, args.verified_sha)
            print(f"Updated {len(changes)} ruleset(s); representative checks verified on {sha}", file=sys.stderr)
        elif not changes:
            print("No matching active pre-commit.ci requirement on main; no updates made.", file=sys.stderr)
        else:
            print(
                "Dry run: no writes made. --apply checks merged workflows and successful checks before writing.",
                file=sys.stderr,
            )
    except (RuntimeError, subprocess.CalledProcessError, json.JSONDecodeError) as exc:
        if isinstance(exc, subprocess.CalledProcessError) and exc.stderr:
            print(exc.stderr.strip(), file=sys.stderr)
        print(f"error: {exc}", file=sys.stderr)
        return 2
    return 0


if __name__ == "__main__":
    sys.exit(main())
