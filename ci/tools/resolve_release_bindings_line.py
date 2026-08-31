# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Resolve the CUDA bindings line for a release source tree and tag."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import bindings_config
import yaml


class ReleaseBindingsLineError(ValueError):
    """The release tag cannot be resolved to a trusted bindings line."""


def _tag_tree_config(config_path: Path) -> bindings_config.BindingsConfig | None:
    """Load a schema-2 tag-tree registry, or return ``None`` for a legacy tree."""
    try:
        raw: Any = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    except FileNotFoundError:
        return None
    except (OSError, yaml.YAMLError) as error:
        raise ReleaseBindingsLineError(f"could not inspect tagged config {config_path}: {error}") from error

    if not isinstance(raw, dict):
        raise ReleaseBindingsLineError(f"tagged config {config_path} must contain a YAML mapping")
    if "schema_version" not in raw:
        return None
    if type(raw["schema_version"]) is not int or raw["schema_version"] != 2:
        raise ReleaseBindingsLineError(
            f"tagged config {config_path} has unsupported schema_version: {raw['schema_version']!r}"
        )
    try:
        return bindings_config.validate_config(raw)
    except bindings_config.BindingsConfigError as error:
        raise ReleaseBindingsLineError(f"invalid schema-2 tagged config {config_path}: {error}") from error


def resolve_release_bindings_line(
    release_tag: str,
    release_source_root: Path,
    control_config_path: Path,
) -> dict[str, object]:
    """Return the normalized bindings line selected for *release_tag*.

    Modern release trees are authoritative for their own layout. Trees from
    before the release-line registry use the control checkout's registry.
    """
    if not release_source_root.is_dir():
        raise ReleaseBindingsLineError(f"release source root is not a directory: {release_source_root}")

    tagged_config_path = release_source_root / "ci" / "versions.yml"
    config = _tag_tree_config(tagged_config_path)
    config_source = f"tagged config {tagged_config_path}"
    registry_origin = "tag"
    if config is None:
        registry_origin = "control"
        config_source = f"control config {control_config_path}"
        try:
            config = bindings_config.load_config(control_config_path)
        except bindings_config.BindingsConfigError as error:
            raise ReleaseBindingsLineError(f"invalid {config_source}: {error}") from error

    line = config.match_tag(release_tag)
    if line is None:
        raise ReleaseBindingsLineError(f"no CUDA bindings line in {config_source} matches release tag: {release_tag!r}")

    release_source_dir = line.source_dir
    if (
        registry_origin == "control"
        and not (release_source_root / release_source_dir).exists()
        and (release_source_root / "cuda_bindings").is_dir()
    ):
        # Before schema 2, the maintenance line was still rooted at the generic
        # directory name in its tagged source tree.
        release_source_dir = "cuda_bindings"

    normalized = config.line_to_dict(line)
    normalized["release_source_dir"] = release_source_dir
    normalized["release_registry_origin"] = registry_origin
    return normalized


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--release-tag", required=True)
    parser.add_argument("--release-source-root", required=True, type=Path)
    parser.add_argument("--control-config", required=True, type=Path)
    args = parser.parse_args(argv)

    try:
        line = resolve_release_bindings_line(
            args.release_tag,
            args.release_source_root,
            args.control_config,
        )
    except ReleaseBindingsLineError as error:
        parser.error(str(error))
    print(json.dumps(line, separators=(",", ":"), sort_keys=True))
    return 0


if __name__ == "__main__":
    sys.exit(main())
