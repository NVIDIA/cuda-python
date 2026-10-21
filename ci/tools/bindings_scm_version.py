# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Bootstrap standard SCM versioning in a disposable pre-milestone build checkout.

Synthetic tags stay local to this checkout. Never push them or run this helper
in a shared development repository.
"""

from __future__ import annotations

import argparse
import re
import subprocess
import sys
import tomllib
from pathlib import Path

from packaging.version import Version
from setuptools_scm import Configuration, get_version
from setuptools_scm.version import tag_to_version

from . import bindings_config

REPO_ROOT = Path(__file__).resolve().parents[2]
COMMIT_PATTERN = re.compile(r"[0-9a-fA-F]{7,64}")


def read_version_config(path: Path, ctk_target: str) -> tuple[Version, list[str]]:
    with path.open("rb") as stream:
        metadata = tomllib.load(stream)
    value = metadata.get("tool", {}).get("cuda-python", {}).get("scm-bootstrap-release")
    if not isinstance(value, str):
        raise ValueError(f"expected tool.cuda-python.scm-bootstrap-release in {path}")
    version = Version(value)
    if (
        str(version) != version.base_version
        or len(version.release) != 3
        or version.release[:2] != Version(ctk_target).release
    ):
        raise ValueError(f"expected an exact CUDA {ctk_target} bootstrap release in {path}, got {value!r}")
    scm = metadata["tool"]["setuptools_scm"]
    if "tag_regex" in scm or "fallback_version" in scm:
        raise ValueError(f"{path} must use standard SCM parsing without a fallback version")
    command = scm.get("git_describe_command")
    if (
        not isinstance(command, list)
        or not all(isinstance(part, str) for part in command)
        or command[:2] != ["git", "describe"]
        or command.count("--match") != 1
        or command[-1] == "--match"
    ):
        raise ValueError(f"expected one git_describe_command --match selector in {path}")
    return version, command


def _git(repo_root: Path, *args: str) -> str:
    return subprocess.run(  # noqa: S603 - Git receives separate arguments, never shell text.
        ["git", *args],  # noqa: S607
        cwd=repo_root,
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()


def bootstrap_version(
    repo_root: Path,
    commit_sha: str,
    package: bindings_config.BindingsPackage,
    base_ref: str = "HEAD^",
) -> str:
    """Use existing SCM history, or add one local preview tag at an ancestor."""
    if COMMIT_PATTERN.fullmatch(commit_sha) is None:
        raise ValueError(f"expected a 7-64 digit hexadecimal commit SHA, got {commit_sha!r}")
    head = _git(repo_root, "rev-parse", "--verify", "HEAD^{commit}")
    if not head.startswith(commit_sha.lower()):
        raise ValueError(f"requested source SHA {commit_sha} does not match checkout HEAD {head}")
    target, command = read_version_config(repo_root / package.package_root / "pyproject.toml", package.ctk_target)
    selector = command[command.index("--match") + 1]
    tags = _git(repo_root, "tag", "--merged", "HEAD", "--list", selector).splitlines()
    scm_config = Configuration(root=repo_root)
    versions = [version for tag in tags if (version := tag_to_version(tag, scm_config)) is not None]
    # A previous final release already yields the next development release.
    # A tag for this release (including an internal alpha) or a newer release
    # always governs its own descendants through standard setuptools-scm.
    previous = Version(".".join(map(str, (*target.release[:2], target.release[2] - 1)))) if target.release[2] else None
    existing = any(
        version.release >= target.release or (previous is not None and version >= previous) for version in versions
    )
    if not existing:
        base = _git(repo_root, "rev-parse", "--verify", "--end-of-options", f"{base_ref}^{{commit}}")
        if base == head:
            raise ValueError("the synthetic preview tag requires an ancestor before the build commit")
        _git(repo_root, "merge-base", "--is-ancestor", base, head)
        # No force flag: even an unreachable existing tag must never be moved.
        tag = f"v{target}a0.dev0"
        _git(repo_root, "-c", "tag.gpgSign=false", "tag", "--no-sign", tag, base)
        print(f"Created local preview tag {tag} at validated ancestor {base}", file=sys.stderr)

    version = get_version(root=repo_root, git_describe_command=command)
    if not existing and Version(version).release != target.release:
        raise ValueError(f"SCM bootstrap produced release {Version(version).release}, expected {target.release}")
    describe = _git(repo_root, *command[1:])
    print(f"{package.package_root}: git describe: {describe}; SCM version: {version}", file=sys.stderr)
    return version


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo-root", type=Path, default=REPO_ROOT)
    parser.add_argument("--config", type=Path, default=bindings_config.DEFAULT_CONFIG)
    parser.add_argument("--package-root", required=True)
    parser.add_argument("--sha", required=True)
    parser.add_argument("--base-ref", default="HEAD^")
    args = parser.parse_args(argv)

    package = bindings_config.load_config(args.config).get_package(args.package_root)
    bootstrap_version(args.repo_root, args.sha, package, args.base_ref)
    return 0


if __name__ == "__main__":
    sys.exit(main())
