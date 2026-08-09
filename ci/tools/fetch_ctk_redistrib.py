#!/usr/bin/env python3
#
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

"""Resolve mini-CTK components from NVIDIA redistrib metadata."""

from __future__ import annotations

import argparse
import json
import sys
import urllib.error
import urllib.parse
import urllib.request
from pathlib import Path
from typing import Any

HOST_PLATFORM_TO_SUBDIR: dict[str, str] = {
    "linux-64": "linux-x86_64",
    "linux-aarch64": "linux-sbsa",
    "win-64": "windows-x86_64",
}

# CTK 13.3.0 renamed the redistrib key from cuda_cccl to cccl.
COMPONENT_ALIASES: dict[str, tuple[str, ...]] = {
    "cuda_cccl": ("cccl",),
}


def host_platform_to_subdir(host_platform: str) -> str:
    try:
        return HOST_PLATFORM_TO_SUBDIR[host_platform]
    except KeyError as exc:
        raise ValueError(f"unsupported host-platform: {host_platform!r}") from exc


def split_components(components: str) -> list[str]:
    return [component for component in components.split(",") if component]


def filter_static_components(components: list[str], host_platform: str, cuda_version: str) -> list[str]:
    try:
        cuda_major = int(cuda_version.split(".", 1)[0])
    except ValueError as exc:
        raise ValueError(f"invalid cuda-version: {cuda_version!r}") from exc

    filtered = []
    for component in components:
        if component == "libnvjitlink" and cuda_major < 12:
            continue
        if component in {"cuda_crt", "libnvvm"} and cuda_major < 13:
            continue
        if component == "libcufile" and host_platform.startswith("win-"):
            continue
        filtered.append(component)
    return filtered


def validate_metadata_url(metadata_url: str) -> str:
    parsed = urllib.parse.urlsplit(metadata_url)
    if parsed.scheme != "https" or not parsed.netloc:
        raise ValueError(f"metadata URL must be an https URL: {metadata_url!r}")
    return metadata_url


def load_metadata(*, metadata_path: str | None, metadata_url: str | None) -> dict[str, Any]:
    if (metadata_path is None) == (metadata_url is None):
        raise ValueError("exactly one of --metadata-path or --metadata-url is required")

    if metadata_path is not None:
        return _as_metadata_object(json.loads(Path(metadata_path).read_text(encoding="utf-8")), metadata_path)

    assert metadata_url is not None
    metadata_url = validate_metadata_url(metadata_url)
    with urllib.request.urlopen(metadata_url) as response:  # noqa: S310 - scheme is restricted to https above
        return _as_metadata_object(json.load(response), metadata_url)


def _as_metadata_object(metadata: Any, source: str) -> dict[str, Any]:
    """Reject JSON that parsed fine but is not a redistrib manifest.

    The manifest is downloaded with ``curl -LSs`` (no ``--fail``), so an error
    page or a redirect body lands in the file and may still be valid JSON --
    just not an object. Without this the failure surfaces several frames later
    as ``TypeError: argument of type 'NoneType' is not iterable``.
    """
    if not isinstance(metadata, dict):
        raise ValueError(f"CTK redistrib metadata from {source} must be a JSON object, got {type(metadata).__name__}")
    return metadata


def resolve_component_name(metadata: dict[str, Any], component: str) -> str:
    if component in metadata:
        return component

    for alias in COMPONENT_ALIASES.get(component, ()):
        if alias in metadata:
            return alias

    return component


def filter_components(
    metadata: dict[str, Any],
    *,
    host_platform: str,
    cuda_version: str,
    components: str,
) -> tuple[list[str], list[str]]:
    ctk_subdir = host_platform_to_subdir(host_platform)
    filtered = []
    skipped = []
    for component in filter_static_components(split_components(components), host_platform, cuda_version):
        resolved_component = resolve_component_name(metadata, component)
        # Guard the type: a top-level key such as "release_label" holds a
        # string, and ``ctk_subdir in "13.0.0"`` is a substring test rather
        # than the intended key lookup.
        component_info = metadata.get(resolved_component)
        if isinstance(component_info, dict) and ctk_subdir in component_info:
            filtered.append(resolved_component)
        else:
            skipped.append(component)
    return filtered, skipped


def get_component_relative_path(metadata: dict[str, Any], *, host_platform: str, component: str) -> str:
    ctk_subdir = host_platform_to_subdir(host_platform)
    component = resolve_component_name(metadata, component)
    component_info = metadata.get(component)
    if component_info is None:
        raise KeyError(f"unknown CTK component {component!r}")
    if not isinstance(component_info, dict):
        # Real manifests carry string-valued top-level keys ("release_date",
        # "release_label", "release_product") alongside the component objects,
        # so "present" is not the same as "is a component".
        raise KeyError(
            f"CTK metadata entry {component!r} is not a component object (got {type(component_info).__name__})"
        )

    subdir_info = component_info.get(ctk_subdir)
    if subdir_info is None:
        raise KeyError(f"CTK component {component!r} is not available for redistrib subdir {ctk_subdir!r}")
    if not isinstance(subdir_info, dict):
        raise KeyError(
            f"CTK component {component!r} entry for redistrib subdir {ctk_subdir!r} "
            f"is not an object (got {type(subdir_info).__name__})"
        )

    relative_path = subdir_info.get("relative_path")
    if relative_path is None:
        raise KeyError(f"CTK component {component!r} for redistrib subdir {ctk_subdir!r} is missing 'relative_path'")
    return relative_path


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)

    filter_parser = subparsers.add_parser("filter-components")
    filter_parser.add_argument("--host-platform", required=True)
    filter_parser.add_argument("--cuda-version", required=True)
    filter_parser.add_argument("--components", required=True)
    filter_parser.add_argument("--metadata-path")
    filter_parser.add_argument("--metadata-url")

    relpath_parser = subparsers.add_parser("component-relative-path")
    relpath_parser.add_argument("--host-platform", required=True)
    relpath_parser.add_argument("--component", required=True)
    relpath_parser.add_argument("--metadata-path")
    relpath_parser.add_argument("--metadata-url")

    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)

    try:
        metadata = load_metadata(metadata_path=args.metadata_path, metadata_url=args.metadata_url)

        if args.command == "filter-components":
            filtered, skipped = filter_components(
                metadata,
                host_platform=args.host_platform,
                cuda_version=args.cuda_version,
                components=args.components,
            )
            for component in skipped:
                print(
                    f"Skipping unsupported CTK component {component!r} for host-platform {args.host_platform!r}",
                    file=sys.stderr,
                )
            print(",".join(filtered))
            return 0

        if args.command == "component-relative-path":
            print(
                get_component_relative_path(
                    metadata,
                    host_platform=args.host_platform,
                    component=args.component,
                )
            )
            return 0

        raise AssertionError(f"unexpected command: {args.command!r}")
    except (ValueError, KeyError, OSError, urllib.error.URLError, json.JSONDecodeError) as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
