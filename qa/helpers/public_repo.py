#!/usr/bin/env python3

# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: LicenseRef-NVIDIA-SOFTWARE-LICENSE

"""
Helper script for syncing branches and tags from public cuda-python repo to private repo.

Usage:
    ./qa/helpers/public_repo.py sync          # Sync public-main branch and all tags
    ./qa/helpers/public_repo.py sync-tags     # Sync only tags
    ./qa/helpers/public_repo.py sync-branch   # Sync only public-main branch
    ./qa/helpers/public_repo.py show-tags     # Show tag comparison
    ./qa/helpers/public_repo.py ensure-remote # Ensure public_repo remote exists
"""

import os
import subprocess
import sys
from pathlib import Path

PUBLIC_REPO_URL = "https://github.com/NVIDIA/cuda-python.git"
PUBLIC_BRANCH = "main"
PRIVATE_BRANCH = "public-main"


def run_cmd(cmd, check=True, capture_output=False):
    """Run a git command and return the result."""
    if isinstance(cmd, str):
        cmd = cmd.split()
    result = subprocess.run(  # noqa: S603
        ["git"] + cmd,
        check=check,
        capture_output=capture_output,
        text=True,
    )
    if capture_output:
        return result.stdout.strip()
    return result


def ensure_public_repo_remote():
    """Ensure public_repo remote exists, add it if not."""
    try:
        run_cmd(["remote", "get-url", "public_repo"], check=False, capture_output=True)
        print("✓ public_repo remote already exists")
        return True
    except subprocess.CalledProcessError:
        print(f"Adding public_repo remote: {PUBLIC_REPO_URL}")
        run_cmd(["remote", "add", "-f", "public_repo", PUBLIC_REPO_URL])
        print("✓ public_repo remote added")
        return False


def get_tags_from_remote(remote):
    """Get list of tags from a remote."""
    try:
        output = run_cmd(["ls-remote", "--tags", remote], capture_output=True, check=False)
        if not output:
            return set()
        tags = set()
        for line in output.splitlines():
            if line.strip():
                tag_ref = line.split()[1]
                # Remove refs/tags/ prefix and ^{} suffix
                tag = tag_ref.replace("refs/tags/", "").replace("^{}", "")
                tags.add(tag)
        return tags
    except subprocess.CalledProcessError:
        return set()


def sync_tags(dry_run=False):
    """Sync tags from public_repo to upstream."""
    print("\n=== Syncing Tags ===")

    ensure_public_repo_remote()

    print("Fetching tags from public_repo...")
    run_cmd(["fetch", "--tags", "public_repo"])

    # Get tag counts
    public_tags = get_tags_from_remote("public_repo")
    upstream_tags = get_tags_from_remote("upstream")

    print(f"Tags on public_repo: {len(public_tags)}")
    print(f"Tags on upstream: {len(upstream_tags)}")

    tags_to_push = public_tags - upstream_tags
    if tags_to_push:
        print(f"\nTags to push: {len(tags_to_push)}")
        if len(tags_to_push) <= 20:
            for tag in sorted(tags_to_push):
                print(f"  - {tag}")
        else:
            print("  (showing first 20)")
            for tag in sorted(list(tags_to_push))[:20]:
                print(f"  - {tag}")
            print(f"  ... and {len(tags_to_push) - 20} more")

        if dry_run:
            print("\n[DRY RUN] Would push tags to upstream")
        else:
            print("\nPushing tags to upstream...")
            run_cmd(["push", "upstream", "--tags"])
            print("✓ Tags synced successfully")
    else:
        print("\n✓ All tags already synced")


def sync_branch(dry_run=False):
    """Sync public-main branch from public_repo to upstream."""
    print("\n=== Syncing Branch ===")

    ensure_public_repo_remote()

    print(f"Fetching {PUBLIC_BRANCH} from public_repo...")
    run_cmd(["fetch", "public_repo", PUBLIC_BRANCH])

    if dry_run:
        print(f"[DRY RUN] Would push refs/remotes/public_repo/{PUBLIC_BRANCH} to upstream {PRIVATE_BRANCH}")
    else:
        print(f"Pushing {PUBLIC_BRANCH} to upstream as {PRIVATE_BRANCH}...")
        run_cmd(["push", "upstream", f"refs/remotes/public_repo/{PUBLIC_BRANCH}:refs/heads/{PRIVATE_BRANCH}"])
        print("✓ Branch synced successfully")


def show_tags():
    """Show tag comparison between public_repo and upstream."""
    print("\n=== Tag Comparison ===")

    ensure_public_repo_remote()

    print("Fetching tags from remotes...")
    run_cmd(["fetch", "--tags", "public_repo"], check=False)
    run_cmd(["fetch", "--tags", "upstream"], check=False)

    public_tags = get_tags_from_remote("public_repo")
    upstream_tags = get_tags_from_remote("upstream")

    common_tags = public_tags & upstream_tags
    only_public = public_tags - upstream_tags
    only_upstream = upstream_tags - public_tags

    print(f"\nCommon tags ({len(common_tags)}):")
    if common_tags:
        for tag in sorted(list(common_tags))[:20]:
            print(f"  ✓ {tag}")
        if len(common_tags) > 20:
            print(f"  ... and {len(common_tags) - 20} more")
    else:
        print("  (none)")

    print(f"\nTags only on public_repo ({len(only_public)}):")
    if only_public:
        for tag in sorted(list(only_public))[:20]:
            print(f"  → {tag}")
        if len(only_public) > 20:
            print(f"  ... and {len(only_public) - 20} more")
        print("\n  Run './qa/helpers/public_repo sync-tags' to sync these")
    else:
        print("  (none)")

    print(f"\nTags only on upstream ({len(only_upstream)}):")
    if only_upstream:
        for tag in sorted(list(only_upstream))[:20]:
            print(f"  ⚠ {tag}")
        if len(only_upstream) > 20:
            print(f"  ... and {len(only_upstream) - 20} more")
        print("\n  These are private-only tags (consider cleanup if not needed)")
    else:
        print("  (none)")


def main():
    """Main entry point."""
    # Change to repo root (qa/helpers -> qa -> repo root)
    script_dir = Path(__file__).resolve().parent
    repo_root = script_dir.parent.parent
    os.chdir(repo_root)

    # Verify we're in a git repo
    if not Path(".git").exists():
        print(f"Error: Not in a git repository. Expected to be in {repo_root}")
        sys.exit(1)

    if len(sys.argv) < 2:
        print(__doc__)
        sys.exit(1)

    command = sys.argv[1]

    try:
        if command == "sync":
            sync_branch()
            sync_tags()
        elif command == "sync-tags":
            sync_tags()
        elif command == "sync-branch":
            sync_branch()
        elif command == "show-tags":
            show_tags()
        elif command == "ensure-remote":
            ensure_public_repo_remote()
        else:
            print(f"Unknown command: {command}")
            print(__doc__)
            sys.exit(1)
    except subprocess.CalledProcessError as e:
        print("\n✗ Error: Command failed")
        if e.stderr:
            print(e.stderr)
        sys.exit(1)
    except KeyboardInterrupt:
        print("\n\nCancelled by user")
        sys.exit(1)


if __name__ == "__main__":
    main()
