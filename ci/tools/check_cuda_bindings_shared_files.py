# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Check that explicitly shared files match across public bindings roots."""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path, PurePosixPath, PureWindowsPath
from typing import Any

import bindings_config

REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_POLICY = REPO_ROOT / "ci" / "cuda-bindings-shared-files.json"
DEFAULT_CONFIG = bindings_config.DEFAULT_CONFIG
POLICY_VERSION = 1


class PolicyError(ValueError):
    """The shared-file policy cannot be interpreted safely."""


def _relative_posix_path(value: Any, label: str) -> str:
    if not isinstance(value, str) or not value:
        raise PolicyError(f"{label} must be a non-empty string")
    if "\\" in value:
        raise PolicyError(f"{label} must use forward slashes: {value!r}")
    if PureWindowsPath(value).drive:
        raise PolicyError(f"{label} must not be drive-qualified: {value!r}")

    path = PurePosixPath(value)
    if path.is_absolute() or path.as_posix() != value or any(part in (".", "..") for part in path.parts):
        raise PolicyError(f"{label} must be a normalized relative path: {value!r}")
    return value


def _sorted_unique_paths(value: Any, label: str, *, minimum: int) -> tuple[str, ...]:
    if not isinstance(value, list):
        raise PolicyError(f"{label} must be a list")
    paths = tuple(_relative_posix_path(item, f"{label} entry") for item in value)
    if len(paths) < minimum:
        raise PolicyError(f"{label} must contain at least {minimum} entries")
    if list(paths) != sorted(set(paths)):
        raise PolicyError(f"{label} must be sorted and contain no duplicates")
    return paths


def load_policy(path: Path) -> tuple[str, ...]:
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise PolicyError(f"could not read {path}: {error}") from error

    if not isinstance(data, dict):
        raise PolicyError("policy must be a JSON object")
    allowed_keys = {"schema_version", "shared_paths"}
    unexpected = sorted(set(data) - allowed_keys)
    if unexpected:
        raise PolicyError(f"unexpected policy keys: {', '.join(unexpected)}")
    schema_version = data.get("schema_version")
    if type(schema_version) is not int or schema_version != POLICY_VERSION:
        raise PolicyError(f"schema_version must be {POLICY_VERSION}")

    shared_paths = _sorted_unique_paths(data.get("shared_paths"), "shared_paths", minimum=1)
    return shared_paths


def _digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()[:12]


def _contains_symlink(base: Path, relative: str) -> bool:
    path = base
    for part in PurePosixPath(relative).parts:
        path /= part
        if path.is_symlink():
            return True
    return False


def find_drift(repo_root: Path, roots: tuple[str, ...], shared_paths: tuple[str, ...]) -> list[str]:
    violations = []
    for root in roots:
        root_path = repo_root / root
        if _contains_symlink(repo_root, root):
            violations.append(f"bindings root must not be a symlink: {root}")
        elif not root_path.is_dir():
            violations.append(f"bindings root is missing or not a directory: {root}")

    if violations:
        return violations

    for relative in shared_paths:
        candidates = [(root, repo_root / root / relative) for root in roots]
        symlinks = [root for root, _ in candidates if _contains_symlink(repo_root / root, relative)]
        if symlinks:
            violations.append(f"{relative}: symlink in {', '.join(symlinks)}")
            continue
        missing = [root for root, path in candidates if not path.is_file()]
        if missing:
            violations.append(f"{relative}: missing from {', '.join(missing)}")
            continue

        contents = [path.read_bytes() for _, path in candidates]
        if any(content != contents[0] for content in contents[1:]):
            digests = ", ".join(f"{root}={_digest(path)}" for root, path in candidates)
            violations.append(f"{relative}: byte mismatch ({digests})")

    return violations


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--policy", type=Path, default=DEFAULT_POLICY)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--repo-root", type=Path, default=REPO_ROOT)
    args = parser.parse_args(argv)

    try:
        shared_paths = load_policy(args.policy)
    except PolicyError as error:
        print(f"error: invalid CUDA bindings shared-file policy: {error}", file=sys.stderr)
        return 2
    try:
        config = bindings_config.load_config(args.config)
    except bindings_config.BindingsConfigError as error:
        print(f"error: invalid CUDA bindings release-line registry: {error}", file=sys.stderr)
        return 2

    roots = tuple(line.source_dir for line in config.lines)
    violations = find_drift(args.repo_root, roots, shared_paths)
    if not violations:
        return 0

    print("error: CUDA bindings shared-file drift detected:", file=sys.stderr)
    for violation in violations:
        print(f"  - {violation}", file=sys.stderr)
    print(
        f"Update every applicable bindings root, or remove intentionally divergent paths from {args.policy}.",
        file=sys.stderr,
    )
    return 1


if __name__ == "__main__":
    sys.exit(main())
