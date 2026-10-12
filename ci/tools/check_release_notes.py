# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Check that versioned release-notes files exist before releasing.

Usage:
    python check_release_notes.py --git-tag <tag> --component <component>
    python check_release_notes.py --git-tag <tag> --component <component> --git-ref refs/tags/<tag>

By default, notes and release configuration come from the working tree.
--git-ref reads those files from the given ref without changing the checkout,
so the workflow's current checker can validate older release tags. Successful
checks append the parsed version to GITHUB_OUTPUT when it is set.

Exit codes:
    0 — release-notes policy satisfied
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


def read_repo_file(path: Path, repo_root: Path, git_ref: str | None = None) -> bytes | None:
    """Read a file from the working tree or an already-resolved Git commit."""
    if git_ref is None:
        full = repo_root / path
        return full.read_bytes() if full.is_file() else None

    object_name = f"{git_ref}:{path.as_posix()}"
    result = subprocess.run(
        ["git", "cat-file", "--batch-check=%(objecttype)"],  # noqa: S607
        input=f"{object_name}\n",
        cwd=repo_root,
        check=True,
        capture_output=True,
        text=True,
    )
    object_type = result.stdout.strip()
    if object_type == f"{object_name} missing" or object_type == "tree":
        return None
    if object_type != "blob":
        raise ValueError(f"Cannot read release-notes source {object_name!r}: unexpected object type {object_type!r}")
    return subprocess.run(  # noqa: S603 - the resolved commit and path are passed without a shell.
        ["git", "show", object_name],  # noqa: S607
        cwd=repo_root,
        check=True,
        capture_output=True,
    ).stdout


def load_backport_branch(repo_root: Path = Path("."), git_ref: str | None = None) -> str | None:
    contents = read_repo_file(Path("ci", "versions.yml"), repo_root, git_ref)
    if contents is not None:
        for line in contents.decode("utf-8").splitlines():
            m = BACKPORT_BRANCH_RE.match(line.strip())
            if m:
                return m.group("branch")
    github_ref_name = os.environ.get("GITHUB_REF_NAME", "")
    if BACKPORT_BRANCH_NAME_RE.match(github_ref_name):
        return github_ref_name
    return None


def is_backport_version(version: str, backport_branch: str) -> bool:
    if backport_branch.endswith(".x"):
        return version.startswith(backport_branch[:-1])
    return version == backport_branch


def notes_path(package: str, version: str) -> Path:
    return Path(package, "docs", "source", "release", f"{version}-notes.rst")


def check_release_notes(
    git_tag: str, component: str, repo_root: Path = Path("."), git_ref: str | None = None
) -> list[tuple[str | Path, str]]:
    """Return a list of (path, reason) for missing or empty release notes.

    ``path`` is the repo-relative notes path, or a ``<placeholder>`` naming the
    offending argument when the tag or component itself is the problem.

    Pathfinder requires a notes file, including for .post releases, but allows
    an empty file. Other components require non-empty notes and skip .post.
    """
    if component not in COMPONENT_TO_PACKAGE:
        return [("<component>", f"unknown component '{component}'")]

    version = parse_version_from_tag(git_tag, component)
    if version is None:
        return [("<tag>", f"cannot parse version from tag '{git_tag}' for component '{component}'")]

    if is_post_release(version) and component != "cuda-pathfinder":
        return []

    path = notes_path(COMPONENT_TO_PACKAGE[component], version)
    contents = read_repo_file(path, repo_root, git_ref)
    if contents is None:
        return [(path, "missing")]
    if not contents and component != "cuda-pathfinder":
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


def write_version_output(version: str) -> None:
    output_path = os.environ.get("GITHUB_OUTPUT")
    if output_path:
        with open(output_path, "a", encoding="utf-8") as f:
            f.write(f"version={version}\n")


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
    git_ref: str | None = None,
) -> tuple[int | None, list[tuple[str | Path, str]]]:
    if component not in BACKPORT_PLANNING_COMPONENTS or is_post_release(version):
        return None, []

    if backport_branch is None:
        print("ERROR: cannot determine backport branch from ci/versions.yml or GITHUB_REF_NAME.", file=sys.stderr)
        return 2, []

    if is_backport_version(version, backport_branch):
        problems = check_release_notes(git_tag, component, repo_root, git_ref)
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

    problems = check_release_notes(decision, component, repo_root, git_ref)
    if problems:
        return 1, problems
    return None, []


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--git-tag", required=True)
    parser.add_argument("--component", required=True, choices=list(COMPONENT_TO_PACKAGE))
    parser.add_argument("--repo-root", default=Path("."), type=Path)
    parser.add_argument(
        "--git-ref", help="Read notes and release configuration from this Git ref without checking it out"
    )
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

    source_ref = args.git_ref
    try:
        if args.git_ref is not None:
            args.git_ref = subprocess.run(  # noqa: S603 - --end-of-options prevents the ref from becoming an option.
                ["git", "rev-parse", "--verify", "--end-of-options", f"{args.git_ref}^{{commit}}"],  # noqa: S607
                cwd=args.repo_root,
                check=True,
                capture_output=True,
                text=True,
            ).stdout.strip()
        return check_policy(args, version)
    except subprocess.CalledProcessError as exc:
        detail = exc.stderr.decode("utf-8") if isinstance(exc.stderr, bytes) else exc.stderr
        print(f"ERROR: cannot read Git release-notes source {source_ref!r}: {detail.strip()}", file=sys.stderr)
        return 2
    except ValueError as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 2


def check_policy(args: argparse.Namespace, version: str) -> int:
    if is_post_release(version) and args.component != "cuda-pathfinder":
        print(f"Post-release tag ({args.git_tag}), skipping release-notes check.")
        write_version_output(version)
        return 0

    backport_branch = args.backport_branch or load_backport_branch(args.repo_root, args.git_ref)
    rc, problems = validate_backport_decision(
        git_tag=args.git_tag,
        component=args.component,
        version=version,
        backport_git_tag=args.backport_git_tag,
        backport_branch=backport_branch,
        repo_root=args.repo_root,
        git_ref=args.git_ref,
    )
    if rc is not None:
        if problems:
            print(f"ERROR: release notes policy failed for tag {args.git_tag}:", file=sys.stderr)
            for path, reason in problems:
                print(f"  - {path} ({reason})", file=sys.stderr)
        if rc == 0:
            write_version_output(version)
        return rc

    if not problems:
        problems = check_release_notes(args.git_tag, args.component, args.repo_root, args.git_ref)

    if not problems:
        print(f"Release notes present for tag {args.git_tag}, component {args.component}.")
        write_version_output(version)
        return 0

    print(f"ERROR: missing or empty release notes for tag {args.git_tag}:", file=sys.stderr)
    for path, reason in problems:
        print(f"  - {path} ({reason})", file=sys.stderr)
    print("Add versioned release notes before releasing.", file=sys.stderr)
    return 1


if __name__ == "__main__":
    sys.exit(main())
