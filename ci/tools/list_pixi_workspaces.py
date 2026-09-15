# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""List the checked-in pixi workspaces as JSON.

Single source of truth for the pixi lockfile CI workflows, so their job
matrices cannot drift from the manifests actually committed to the repository
(see #2298). Each workspace is reported as an object with:

  id          stable, human-readable name used for selection and job names
              ("root" for the top-level workspace, otherwise the directory)
  branch_key  deterministic, collision-resistant, ref-safe key used for Git
              branches and workflow concurrency
  manifest    --manifest-path argument for pixi
  lockfile    path of the committed lockfile

Every manifest must have a sibling lockfile; a missing one is an error rather
than a silently skipped workspace.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
ROOT_WORKSPACE_ID = "root"
ALL = "all"
RESERVED_WORKSPACE_IDS = frozenset({ROOT_WORKSPACE_ID, ALL})
_REF_KEY_UNSAFE = re.compile(r"[^a-z0-9]+")


def _branch_key(workspace_id: str) -> str:
    """Return a readable, collision-resistant Git ref component for an ID."""
    slug = _REF_KEY_UNSAFE.sub("-", workspace_id.lower()).strip("-") or "workspace"
    slug = slug[:40].rstrip("-")
    digest = hashlib.sha256(workspace_id.encode("utf-8")).hexdigest()[:16]
    return f"{slug}-{digest}"


def discover() -> list[dict[str, str]]:
    """Return every tracked pixi workspace, ordered by manifest path."""
    tracked = subprocess.run(  # noqa: S603 - argv is passed directly without a shell.
        ["git", "-C", str(ROOT), "ls-files", "-z", "pixi.toml", "*/pixi.toml"],  # noqa: S607
        capture_output=True,
        check=True,
        text=True,
    ).stdout
    # Root workspace first, then nested ones by path, so job names and matrix
    # order stay stable as workspaces are added.
    manifests = sorted((path for path in tracked.split("\0") if path), key=lambda p: (p != "pixi.toml", p))
    if not manifests:
        raise RuntimeError(f"no tracked pixi.toml found under {ROOT}")

    workspaces: list[dict[str, str]] = []
    missing: list[str] = []
    invalid: list[str] = []
    for manifest in manifests:
        directory = str(Path(manifest).parent.as_posix())
        workspace_id = ROOT_WORKSPACE_ID if directory == "." else directory
        if directory != "." and workspace_id in RESERVED_WORKSPACE_IDS:
            invalid.append(
                f"{manifest} uses reserved workspace id {workspace_id!r}; "
                f"{sorted(RESERVED_WORKSPACE_IDS)!r} are reserved for selection"
            )
            continue
        lockfile = "pixi.lock" if directory == "." else f"{directory}/pixi.lock"
        if not (ROOT / lockfile).is_file():
            missing.append(f"{manifest} has no committed {lockfile}")
            continue
        workspaces.append(
            {
                "id": workspace_id,
                "branch_key": _branch_key(workspace_id),
                "manifest": directory,
                "lockfile": lockfile,
            }
        )

    if missing:
        raise RuntimeError("incomplete pixi workspaces:\n  - " + "\n  - ".join(missing))
    if invalid:
        raise RuntimeError("invalid pixi workspaces:\n  - " + "\n  - ".join(invalid))

    ids: dict[str, str] = {}
    branch_keys: dict[str, str] = {}
    for workspace in workspaces:
        workspace_id = workspace["id"]
        manifest = workspace["manifest"]
        branch_key = workspace["branch_key"]
        if previous := ids.get(workspace_id):
            invalid.append(f"{manifest} and {previous} have duplicate workspace id {workspace_id!r}")
        else:
            ids[workspace_id] = manifest
        if previous := branch_keys.get(branch_key):
            invalid.append(f"{manifest} and {previous} map to duplicate refresh branch key {branch_key!r}")
        else:
            branch_keys[branch_key] = manifest
    if invalid:
        raise RuntimeError("pixi workspace inventory cannot be represented safely:\n  - " + "\n  - ".join(invalid))
    return workspaces


def main() -> int:
    """Print the workspace inventory, optionally narrowed to one workspace."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--select",
        default=ALL,
        help=f"workspace id to keep, or {ALL!r} (the default) for every workspace",
    )
    args = parser.parse_args()

    try:
        workspaces = discover()
    except (RuntimeError, subprocess.CalledProcessError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2

    if args.select != ALL:
        selected = [w for w in workspaces if w["id"] == args.select]
        if not selected:
            valid = ", ".join([ALL, *(w["id"] for w in workspaces)])
            print(
                f"error: unknown pixi workspace {args.select!r}; expected one of: {valid}",
                file=sys.stderr,
            )
            return 1
        workspaces = selected

    print(json.dumps(workspaces))
    return 0


if __name__ == "__main__":
    sys.exit(main())
