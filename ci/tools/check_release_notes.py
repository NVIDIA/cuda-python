# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Check that versioned release-notes files exist before releasing.

Releases of components listed in ``release_ranges.PACKAGES`` (at or after their
first version with note files) are checked differently: at least one note must have
been added in the release's range (see ``release_notes`` in
``cuda_python/docs/exts``); the release tag must be in a full clone.

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
import subprocess
import sys
from pathlib import Path

# The release-notes rules are shared with the docs build.
sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "cuda_python" / "docs" / "exts"))
import release_ranges

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
BACKPORT_BRANCH_NAME_RE = re.compile(r"^\d+\.\d+\.x$")


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


def load_backport_branch(repo_root: Path = Path(".")) -> str | None:
    path = repo_root / "ci" / "versions.yml"
    try:
        with open(path, encoding="utf-8") as f:
            for line in f:
                m = BACKPORT_BRANCH_RE.match(line.strip())
                if m:
                    return m.group("branch")
    except FileNotFoundError:
        pass
    github_ref_name = os.environ.get("GITHUB_REF_NAME", "")
    if BACKPORT_BRANCH_NAME_RE.match(github_ref_name):
        return github_ref_name
    return None


def is_backport_version(version: str, backport_branch: str) -> bool:
    if backport_branch.endswith(".x"):
        return version.startswith(backport_branch[:-1])
    return version == backport_branch


_BASE_VERSION_RE = re.compile(r"^(\d+)\.(\d+)\.(\d+)")


def notes_package(component: str, version: str) -> release_ranges.NotesPackage | None:
    """Return the notes settings if *version* of *component* has note files."""
    package = release_ranges.PACKAGES.get(component)
    if package is None:
        return None
    m = _BASE_VERSION_RE.match(version)
    if m is None:
        return None
    if tuple(int(part) for part in m.groups()) < package.first_version:
        return None
    return package


def _git(repo_root: Path, *args: str) -> str:
    result = subprocess.run(  # noqa: S603
        ["git", *args],  # noqa: S607
        cwd=repo_root,
        capture_output=True,
        text=True,
        check=False,
    )
    if result.returncode != 0:
        raise RuntimeError(f"`git {' '.join(args)}` failed: {result.stderr.strip()}")
    return result.stdout


def check_notes_in_range(
    git_tag: str, package: release_ranges.NotesPackage, repo_root: Path = Path(".")
) -> list[tuple[str | Path, str]]:
    """Return problems if no note was added in *git_tag*'s release range.

    The range is the one the docs use (see ``release_ranges.previous_release``).
    Pre-release tags need no notes; theirs roll into the final release.
    """
    notes_path = package.notes_dir
    if _git(repo_root, "rev-parse", "--is-shallow-repository").strip() == "true":
        return [("<clone>", "shallow clone: fetch the full history and tags (fetch-depth: 0)")]

    parsed = release_ranges.parse_release_tags([git_tag], package.tag_prefix)
    if not parsed:
        return []  # a pre-release tag
    (version,) = parsed
    releases = release_ranges.parse_release_tags(_git(repo_root, "tag", "--list").split(), package.tag_prefix)
    if version not in releases:
        return [("<tag>", f"tag {git_tag} not found in the clone; fetch tags")]
    notes_releases = sorted(v for v in releases if v >= package.first_version)
    previous = release_ranges.previous_release(version, notes_releases)

    if previous is None:
        added = _git(repo_root, "ls-tree", "-r", "--name-only", git_tag, "--", notes_path)
        range_text = "in the tree"
    else:
        previous_tag = releases[previous]
        added = _git(repo_root, "diff", "--name-only", "--diff-filter=A", previous_tag, git_tag, "--", notes_path)
        range_text = f"since {previous_tag}"
    if not added.strip():
        return [(Path(notes_path), f"no release notes added {range_text}")]
    return []


def notes_path(package: str, version: str) -> Path:
    return Path(package, "docs", "source", "release", f"{version}-notes.rst")


def check_release_notes(git_tag: str, component: str, repo_root: Path = Path(".")) -> list[tuple[str | Path, str]]:
    """Return a list of (path, reason) for missing or empty release notes.

    ``path`` is the repo-relative notes path, or a ``<placeholder>`` naming the
    offending argument when the tag or component itself is the problem.

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

    package = notes_package(component, version)
    if package is not None:
        return check_notes_in_range(git_tag, package, repo_root)

    path = notes_path(COMPONENT_TO_PACKAGE[component], version)
    full = repo_root / path
    if not full.is_file():
        return [(path, "missing")]
    if full.stat().st_size == 0:
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


def warn_missing_backport_notes(git_tag: str, component: str, problems: list[tuple[str | Path, str]]) -> None:
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
    repo_root: Path,
) -> tuple[int | None, list[tuple[str | Path, str]]]:
    if component not in BACKPORT_PLANNING_COMPONENTS or is_post_release(version):
        return None, []

    if backport_branch is None:
        print("ERROR: cannot determine backport branch from ci/versions.yml or GITHUB_REF_NAME.", file=sys.stderr)
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

    if notes_package(component, backport_version) is not None:
        # Patch notes are note files on the maintenance branch, collected at the
        # backport tag; there is nothing to prepare on this branch.
        print(f"Backport release notes for {decision} come from note files at that tag, skipping check.")
        return None, []

    problems = check_release_notes(decision, component, repo_root)
    if problems:
        return 1, problems
    return None, []


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--git-tag", required=True)
    parser.add_argument("--component", required=True, choices=list(COMPONENT_TO_PACKAGE))
    parser.add_argument("--repo-root", default=Path("."), type=Path)
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
