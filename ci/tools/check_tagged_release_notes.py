# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Check that versioned release-notes files exist before releasing.

Usage:
    python -m ci.tools.check_tagged_release_notes --git-tag <tag> --component <component>
    python -m ci.tools.check_tagged_release_notes --tag-ci --git-tag <tag>
    python -m ci.tools.check_tagged_release_notes --git-tag <tag> --component <component> --rendered-docs <staging-dir>

Each normal release requires nonempty notes for its exact version and component
in the tagged source checkout. Post releases do not require new notes, except
Pathfinder requires a notes file for every release and permits empty files.

The dependency-free --tag-ci mode checks only that the exact component notes file
exists. It skips internal alpha/development tags and, except for Pathfinder,
post releases. It runs before CI sets up Python dependencies.

The --rendered-docs mode also checks that the versioned staging tree contains
the notes HTML and that its release index links to it. It skips internal tags
and empty Pathfinder placeholders, which need no rendered notes page.

Successful publication checks append the parsed version to GITHUB_OUTPUT when it is set.

Exit codes:
    0 — release-notes policy satisfied
    1 — release notes missing or empty
    2 — invalid arguments (including unparsable tag, or component/tag-prefix mismatch)
"""

from __future__ import annotations

import argparse
import json
import os
import re
import sys
from collections.abc import Mapping
from html.parser import HTMLParser
from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from packaging.version import Version

COMPONENTS: dict[str, tuple[str, str]] = {
    "cuda-core": ("cuda_core", "cuda-core-v"),
    "cuda-bindings": ("cuda_bindings_13", "v"),
    "cuda-pathfinder": ("cuda_pathfinder", "cuda-pathfinder-v"),
    "cuda-python": ("cuda_python", "v"),
}


def _resolved_bindings_target(data: Mapping[str, object], git_tag: str) -> tuple[str, Version]:
    """Read the release resolver fields consumed by this script."""
    from packaging.version import Version

    from . import bindings_config

    package_root = bindings_config.parse_package_root(data.get("package_root"), "resolved package_root")
    raw_version = data.get("release_version")
    if not isinstance(raw_version, str):
        raise ValueError("resolved CUDA bindings package has no release_version")
    version = Version(raw_version)
    tag_version = bindings_config.parse_prefixed_version(git_tag, "v")
    if tag_version != version:
        raise ValueError(f"resolved CUDA bindings package does not match release tag {git_tag!r}")
    return package_root, version


def _release_target_from_tag(
    git_tag: str,
    component: str,
    bindings_package: Mapping[str, object] | None = None,
) -> tuple[str, str] | None:
    """Return the release version and source tree selected by a component tag."""
    from . import bindings_config

    metadata = COMPONENTS.get(component)
    if metadata is None:
        return None
    package_dir, prefix = metadata
    if component == "cuda-bindings":
        if bindings_package is None:
            package = bindings_config.load_config().match_tag(git_tag)
            if package is None:
                return None
            package_root = package.package_root
            version = package.version_from_tag(git_tag)
        else:
            package_root, version = _resolved_bindings_target(bindings_package, git_tag)
    else:
        version = bindings_config.parse_prefixed_version(git_tag, prefix)
    if version is None:
        return None
    return str(version), package_root if component == "cuda-bindings" else package_dir


def parse_version_from_tag(
    git_tag: str,
    component: str,
    bindings_package: Mapping[str, object] | None = None,
) -> str | None:
    """Extract the version string from a tag, given the target component.

    Returns None if the tag does not match the component's expected prefix,
    contains characters outside the allowed version set, or (for bindings)
    does not select a configured package root.
    """
    target = _release_target_from_tag(git_tag, component, bindings_package)
    return target[0] if target is not None else None


def is_post_release(version: str) -> bool:
    from packaging.version import Version

    return Version(version).post is not None


def notes_path(package: str, version: str) -> Path:
    return Path(package, "docs", "source", "release", f"{version}-notes.rst")


def _component_for_tag_ci(git_tag: str) -> str:
    for component in ("cuda-core", "cuda-pathfinder"):
        _, prefix = COMPONENTS[component]
        if git_tag.startswith(prefix):
            return component
    return "cuda-bindings"


def _is_internal_version(version: str) -> bool:
    return re.search(r"(?:a|\.dev)\d", version) is not None


def _check_tag_ci(git_tag: str, repo_root: Path) -> int:
    """Check one notes file for the tag namespaces selected by ci.yml."""
    component = _component_for_tag_ci(git_tag)
    package, prefix = COMPONENTS[component]
    version = git_tag.removeprefix(prefix)
    if component == "cuda-bindings":
        package = f"cuda_bindings_{version.partition('.')[0]}"

    if _is_internal_version(version):
        print(f"Internal alpha/development tag {git_tag}; no release note required.")
        return 0
    if package != "cuda_pathfinder" and re.search(r"\.post\d", version):
        print(f"Post-release rebuild {git_tag}; no new release note required.")
        return 0

    path = notes_path(package, version)
    if not (repo_root / path).is_file():
        print(
            f"::error::Missing release notes for tag {git_tag}: {path}. Add this file before creating the tag.",
            file=sys.stderr,
        )
        return 1
    print(f"Found release notes for {git_tag}: {path}")
    return 0


def _check_release_target(version: str, package: str, component: str, repo_root: Path) -> list[tuple[str | Path, str]]:
    if is_post_release(version) and component != "cuda-pathfinder":
        return []

    path = notes_path(package, version)
    full = repo_root / path
    if not full.is_file():
        return [(path, "missing")]
    if full.stat().st_size == 0 and component != "cuda-pathfinder":
        return [(path, "empty")]
    return []


class _ReleaseIndexLinks(HTMLParser):
    def __init__(self) -> None:
        super().__init__()
        self.paths: set[str] = set()

    def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        if tag == "a":
            href = dict(attrs).get("href")
            if href is not None:
                self.paths.add(href.split("#", 1)[0].split("?", 1)[0].removeprefix("./"))


def _check_rendered_target(docs_root: Path, component: str, version: str) -> int:
    # Pathfinder post releases put their exact notes under the base docs directory.
    version_dir = version.split(".post", 1)[0]
    component_root = docs_root if component == "cuda-python" else docs_root / component
    version_root = component_root / version_dir
    notes_href = f"release/{version}-notes.html"
    rendered_notes = version_root / notes_href
    if not rendered_notes.is_file():
        print(f"ERROR: missing rendered release notes: {rendered_notes}", file=sys.stderr)
        return 1
    if rendered_notes.stat().st_size == 0:
        print(f"ERROR: empty rendered release notes: {rendered_notes}", file=sys.stderr)
        return 1

    index = version_root / "release.html"
    if not index.is_file():
        print(f"ERROR: missing rendered release index: {index}", file=sys.stderr)
        return 1
    links = _ReleaseIndexLinks()
    links.feed(index.read_text(encoding="utf-8"))
    if notes_href not in links.paths:
        print(f"ERROR: missing relative link to {notes_href} in {index}", file=sys.stderr)
        return 1
    print(f"Found rendered release notes and release index link: {rendered_notes}")
    return 0


def write_version_output(version: str) -> None:
    output_path = os.environ.get("GITHUB_OUTPUT")
    if output_path:
        with open(output_path, "a", encoding="utf-8") as f:
            f.write(f"version={version}\n")


def check_release_notes(
    git_tag: str,
    component: str,
    repo_root: Path = Path("."),
    bindings_package: Mapping[str, object] | None = None,
) -> list[tuple[str | Path, str]]:
    """Return a list of (path, reason) for missing or empty release notes.

    ``path`` is the repo-relative notes path, or a ``<placeholder>`` naming the
    offending argument when the tag or component itself is the problem.

    Pathfinder requires a notes file, including for .post releases, but allows
    an empty file. Other components require non-empty notes and skip .post.
    """
    if component not in COMPONENTS:
        return [("<component>", f"unknown component '{component}'")]

    target = _release_target_from_tag(git_tag, component, bindings_package)
    if target is None:
        return [("<tag>", f"cannot parse version from tag '{git_tag}' for component '{component}'")]
    version, package = target
    return _check_release_target(version, package, component, repo_root)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--git-tag", required=True)
    parser.add_argument("--component", choices=[*COMPONENTS, "all"], help="required unless --tag-ci is used")
    parser.add_argument("--tag-ci", action="store_true", help="check only file existence for a CI tag push")
    parser.add_argument("--rendered-docs", type=Path, help="staged docs payload to check after the source")
    parser.add_argument(
        "--repo-root",
        default=Path("."),
        type=Path,
        help="tagged source checkout containing the component's versioned release notes",
    )
    parser.add_argument(
        "--bindings-package",
        default="",
        help="normalized CUDA bindings package JSON from the release resolver",
    )
    args = parser.parse_args(argv)

    if args.tag_ci:
        return _check_tag_ci(args.git_tag, args.repo_root)
    if args.component is None:
        parser.error("--component is required unless --tag-ci is used")
    component = args.component
    if component == "all" and args.rendered_docs is not None:
        component = _component_for_tag_ci(args.git_tag)

    try:
        bindings_package = json.loads(args.bindings_package) if args.bindings_package else None
        if bindings_package is not None and not isinstance(bindings_package, dict):
            raise ValueError("resolved CUDA bindings package must be a JSON object")
        target = _release_target_from_tag(args.git_tag, component, bindings_package)
    except ValueError as error:
        print(f"ERROR: invalid CUDA bindings configuration: {error}", file=sys.stderr)
        return 2
    if target is None:
        print(
            f"ERROR: tag {args.git_tag!r} does not match the expected format for component {component!r}.",
            file=sys.stderr,
        )
        return 2
    version, package = target

    if args.rendered_docs is not None and _is_internal_version(version):
        print(f"Internal alpha/development tag {args.git_tag}; no rendered release note required.")
        return 0

    if is_post_release(version) and component != "cuda-pathfinder":
        print(f"Post-release tag ({args.git_tag}), skipping release-notes check.")
        write_version_output(version)
        return 0

    problems = _check_release_target(version, package, component, args.repo_root)

    if not problems:
        if args.rendered_docs is not None:
            source = args.repo_root / notes_path(package, version)
            if component == "cuda-pathfinder" and source.stat().st_size == 0:
                print(f"Empty Pathfinder release note placeholder {source}; no rendered release note required.")
                return 0
            return _check_rendered_target(args.rendered_docs, component, version)
        print(f"Release notes present for tag {args.git_tag}, component {component}.")
        write_version_output(version)
        return 0

    print(f"ERROR: missing or empty release notes for tag {args.git_tag}:", file=sys.stderr)
    for path, reason in problems:
        print(f"  - {path} ({reason})", file=sys.stderr)
    print("Add versioned release notes to the tagged source before releasing.", file=sys.stderr)
    return 1


if __name__ == "__main__":
    sys.exit(main())
