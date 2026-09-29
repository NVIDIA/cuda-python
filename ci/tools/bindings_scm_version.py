# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Select a maintenance bindings package's SCM version before its release tag."""

from __future__ import annotations

import argparse
import re
import subprocess
import sys
from pathlib import Path

import tomllib
from packaging.version import Version

from . import bindings_config

REPO_ROOT = Path(__file__).resolve().parents[2]
COMMIT_PATTERN = re.compile(r"[0-9a-fA-F]{7,64}")


def read_version_config(path: Path, ctk_target: str) -> tuple[Version, str]:
    try:
        with path.open("rb") as stream:
            scm = tomllib.load(stream)["tool"]["setuptools_scm"]
            value = scm["fallback_version"]
    except (OSError, KeyError, TypeError, tomllib.TOMLDecodeError) as error:
        raise ValueError(f"could not read [tool.setuptools_scm].fallback_version from {path}: {error}") from error
    if not isinstance(value, str):
        raise ValueError(f"expected fallback_version in {path} to be a string")
    version = bindings_config.parse_pep440_version(value, f"fallback_version in {path}")
    target = bindings_config.parse_pep440_version(ctk_target, "CTK target")
    if version.dev is None or version.release[:2] != target.release[:2]:
        raise ValueError(f"expected a CUDA {ctk_target} development fallback in {path}, got {value!r}")
    command = scm.get("git_describe_command")
    if (
        not isinstance(command, list)
        or not all(isinstance(part, str) for part in command)
        or command.count("--match") != 1
        or command[-1] == "--match"
    ):
        raise ValueError(f"expected one git_describe_command --match selector in {path}")
    return version, command[command.index("--match") + 1]


def has_reachable_tag(repo_root: Path, tag_selector: str) -> bool:
    process = subprocess.run(  # noqa: S603 - selector is a validated argument; no shell is invoked.
        ["git", "tag", "--merged", "HEAD", "--list", tag_selector],  # noqa: S607
        cwd=repo_root,
        capture_output=True,
        text=True,
    )
    if process.returncode != 0:
        detail = process.stderr.strip() or f"git tag exited with status {process.returncode}"
        raise RuntimeError(f"could not inspect reachable tags matching {tag_selector!r}: {detail}")
    return bool(process.stdout.strip())


def pretend_version(
    repo_root: Path,
    commit_sha: str,
    package: bindings_config.BindingsPackage,
) -> str | None:
    """Return the pre-tag override, or None once the package has its release tag."""
    if COMMIT_PATTERN.fullmatch(commit_sha) is None:
        raise ValueError(f"expected a 7-64 digit hexadecimal commit SHA, got {commit_sha!r}")
    config_path = repo_root / package.package_root / "pyproject.toml"
    fallback_version, tag_selector = read_version_config(config_path, package.ctk_target)
    # Once a matching tag is reachable, standard SCM progression is authoritative,
    # including descendants of prerelease and post-release tags below the fallback.
    if has_reachable_tag(repo_root, tag_selector):
        return None

    return f"{fallback_version}+g{commit_sha[:7].lower()}"


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo-root", type=Path, default=REPO_ROOT)
    parser.add_argument("--config", type=Path, default=bindings_config.DEFAULT_CONFIG)
    parser.add_argument("--package-root", required=True)
    parser.add_argument("--sha", required=True)
    args = parser.parse_args(argv)

    package = bindings_config.load_config(args.config).get_package(args.package_root)
    version = pretend_version(args.repo_root, args.sha, package)
    if version is not None:
        print(version)
    return 0


if __name__ == "__main__":
    sys.exit(main())
