# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Check that versioned release-notes files exist before releasing.

Usage:
    python check_release_notes.py --git-tag <tag> --component <component>

Exit codes:
    0 — release notes present and non-empty (or .post version, skipped)
    1 — release notes missing or empty
    2 — invalid arguments (including unparsable tag, or component/tag-prefix mismatch)
"""

from __future__ import annotations

import argparse
import os
import re
import sys

COMPONENT_TO_PACKAGE: dict[str, str] = {
    "cuda-core": "cuda_core",
    "cuda-bindings": "cuda_bindings",
    "cuda-pathfinder": "cuda_pathfinder",
    "cuda-python": "cuda_python",
}

# Version characters are restricted to digit-prefixed word chars and dots, so
# malformed inputs like "v../evil" or "v1/2/3" cannot flow into the notes path.
_VERSION_PATTERN = r"\d[\w.]*"

# Each component has exactly one valid tag-prefix form. cuda-bindings and
# cuda-python share the bare "v<version>" namespace (setuptools-scm lookup).
COMPONENT_TO_TAG_RE: dict[str, re.Pattern[str]] = {
    "cuda-bindings": re.compile(rf"^v(?P<version>{_VERSION_PATTERN})$"),
    "cuda-python": re.compile(rf"^v(?P<version>{_VERSION_PATTERN})$"),
    "cuda-core": re.compile(rf"^cuda-core-v(?P<version>{_VERSION_PATTERN})$"),
    "cuda-pathfinder": re.compile(rf"^cuda-pathfinder-v(?P<version>{_VERSION_PATTERN})$"),
}

BACKPORT_PLANNING_COMPONENTS = frozenset({"cuda-bindings", "cuda-python"})
BACKPORT_NOT_PLANNED = "not planned"
BACKPORT_BRANCH_RE = re.compile(r"""^backport_branch:\s*["']?(?P<branch>[^"'\s#]+)""")


def parse_version_from_tag(git_tag: str, component: str) -> str | None:
    """Extract the version string from a tag, given the target component.

    Returns None if the tag does not match the component's expected prefix
    or contains characters outside the allowed version set.
    """
    pattern = COMPONENT_TO_TAG_RE.get(component)
    if pattern is None:
        return None
    m = pattern.match(git_tag)
    return m.group("version") if m else None


def is_post_release(version: str) -> bool:
    return ".post" in version


def load_backport_branch(repo_root: str = ".") -> str | None:
    path = os.path.join(repo_root, "ci", "versions.yml")
    try:
        with open(path, encoding="utf-8") as f:
            for line in f:
                m = BACKPORT_BRANCH_RE.match(line.strip())
                if m:
                    return m.group("branch")
    except FileNotFoundError:
        return None
    return None


def is_backport_version(version: str, backport_branch: str) -> bool:
    if backport_branch.endswith(".x"):
        return version.startswith(backport_branch[:-1])
    return version == backport_branch


def notes_path(package: str, version: str) -> str:
    return os.path.join(package, "docs", "source", "release", f"{version}-notes.rst")


def check_release_notes(git_tag: str, component: str, repo_root: str = ".") -> list[tuple[str, str]]:
    """Return a list of (path, reason) for missing or empty release notes.

    Returns an empty list when notes are present and non-empty, or when the
    tag is a .post release (no new notes required).
    """
    if component not in COMPONENT_TO_PACKAGE:
        return [("<component>", f"unknown component '{component}'")]

    version = parse_version_from_tag(git_tag, component)
    if version is None:
        return [("<tag>", f"cannot parse version from tag '{git_tag}' for component '{component}'")]

    if is_post_release(version):
        return []

    path = notes_path(COMPONENT_TO_PACKAGE[component], version)
    full = os.path.join(repo_root, path)
    if not os.path.isfile(full):
        return [(path, "missing")]
    if os.path.getsize(full) == 0:
        return [(path, "empty")]
    return []


def write_step_summary(message: str) -> None:
    summary_path = os.environ.get("GITHUB_STEP_SUMMARY")
    if not summary_path:
        return
    with open(summary_path, "a", encoding="utf-8") as f:
        f.write(message)
        if not message.endswith("\n"):
            f.write("\n")


def warn_missing_backport_notes(git_tag: str, component: str, problems: list[tuple[str, str]]) -> None:
    print(f"WARNING: missing or empty release notes for backport tag {git_tag}:")
    summary_lines = [
        "## Release Notes Reminder",
        "",
        f"Backport release `{git_tag}` for `{component}` is allowed to continue,",
        "but the following release-note files are missing or empty in the workflow source:",
        "",
    ]
    for path, reason in problems:
        print(f"::warning file={path}::Release notes for backport tag {git_tag} are {reason}.")
        print(f"  - {path} ({reason})")
        summary_lines.append(f"- `{path}` ({reason})")
    summary_lines.extend(["", "Please add the backport release notes on `main` if they are not already present."])
    write_step_summary("\n".join(summary_lines))


def validate_backport_decision(
    *,
    git_tag: str,
    component: str,
    version: str,
    backport_git_tag: str,
    backport_branch: str | None,
    repo_root: str,
) -> tuple[int | None, list[tuple[str, str]]]:
    if component not in BACKPORT_PLANNING_COMPONENTS or is_post_release(version):
        return None, []

    if backport_branch is None:
        print("ERROR: cannot determine backport branch from ci/versions.yml.", file=sys.stderr)
        return 2, []

    if is_backport_version(version, backport_branch):
        problems = check_release_notes(git_tag, component, repo_root)
        if problems:
            warn_missing_backport_notes(git_tag, component, problems)
        else:
            print(f"Release notes present for backport tag {git_tag}, component {component}.")
        return 0, []

    decision = backport_git_tag.strip()
    if not decision:
        return (
            1,
            [
                (
                    "<backport-git-tag>",
                    f"required for {component} mainline releases; use a backport tag or '{BACKPORT_NOT_PLANNED}'",
                )
            ],
        )

    if decision == BACKPORT_NOT_PLANNED:
        print(f"Backport release not planned for {git_tag}, skipping backport release-notes check.")
        return None, []

    backport_version = parse_version_from_tag(decision, component)
    if backport_version is None:
        print(
            f"ERROR: backport tag {decision!r} does not match the expected format for component {component!r}.",
            file=sys.stderr,
        )
        return 2, []

    if not is_backport_version(backport_version, backport_branch):
        print(
            f"ERROR: backport tag {decision!r} does not match configured backport branch {backport_branch!r}.",
            file=sys.stderr,
        )
        return 2, []

    problems = check_release_notes(decision, component, repo_root)
    if problems:
        return 1, problems
    return None, []


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--git-tag", required=True)
    parser.add_argument("--component", required=True, choices=list(COMPONENT_TO_PACKAGE))
    parser.add_argument("--repo-root", default=".")
    parser.add_argument("--backport-git-tag", default="")
    parser.add_argument("--backport-branch", default="")
    args = parser.parse_args(argv)

    version = parse_version_from_tag(args.git_tag, args.component)
    if version is None:
        print(
            f"ERROR: tag {args.git_tag!r} does not match the expected format for component {args.component!r}.",
            file=sys.stderr,
        )
        return 2

    if is_post_release(version):
        print(f"Post-release tag ({args.git_tag}), skipping release-notes check.")
        return 0

    backport_branch = args.backport_branch or load_backport_branch(args.repo_root)
    rc, problems = validate_backport_decision(
        git_tag=args.git_tag,
        component=args.component,
        version=version,
        backport_git_tag=args.backport_git_tag,
        backport_branch=backport_branch,
        repo_root=args.repo_root,
    )
    if rc is not None:
        if problems:
            print(f"ERROR: release notes policy failed for tag {args.git_tag}:", file=sys.stderr)
            for path, reason in problems:
                print(f"  - {path} ({reason})", file=sys.stderr)
        return rc

    if not problems:
        problems = check_release_notes(args.git_tag, args.component, args.repo_root)

    if not problems:
        print(f"Release notes present for tag {args.git_tag}, component {args.component}.")
        return 0

    print(f"ERROR: missing or empty release notes for tag {args.git_tag}:", file=sys.stderr)
    for path, reason in problems:
        print(f"  - {path} ({reason})", file=sys.stderr)
    print("Add versioned release notes before releasing.", file=sys.stderr)
    return 1


if __name__ == "__main__":
    sys.exit(main())
