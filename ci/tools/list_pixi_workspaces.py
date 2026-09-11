# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""List the checked-in pixi workspaces as JSON.

Single source of truth for the pixi lockfile CI workflows, so their job
matrices cannot drift from the manifests actually committed to the repository
(see #2298). Each workspace is reported as an object with:

  id        stable name used for job names and refresh branches ("root" for
            the top-level workspace, otherwise the directory path)
  manifest  --manifest-path argument for pixi
  lockfile  path of the committed lockfile

Every manifest must have a sibling lockfile; a missing one is an error rather
than a silently skipped workspace.
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
ROOT_WORKSPACE_ID = "root"
ALL = "all"


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
    for manifest in manifests:
        directory = str(Path(manifest).parent.as_posix())
        lockfile = "pixi.lock" if directory == "." else f"{directory}/pixi.lock"
        if not (ROOT / lockfile).is_file():
            missing.append(f"{manifest} has no committed {lockfile}")
            continue
        workspaces.append(
            {
                "id": ROOT_WORKSPACE_ID if directory == "." else directory,
                "manifest": directory,
                "lockfile": lockfile,
            }
        )

    if missing:
        raise RuntimeError("incomplete pixi workspaces:\n  - " + "\n  - ".join(missing))
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
