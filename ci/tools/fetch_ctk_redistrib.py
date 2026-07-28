#!/usr/bin/env python3
#
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

"""Resolve mini-CTK components and prerelease installers."""

from __future__ import annotations

import argparse
import json
import shutil
import sys
import urllib.error
import urllib.parse
import urllib.request
from dataclasses import dataclass
from pathlib import Path
from typing import Any

HOST_PLATFORM_TO_SUBDIR: dict[str, str] = {
    "linux-64": "linux-x86_64",
    "linux-aarch64": "linux-sbsa",
    "win-64": "windows-x86_64",
    "win-arm64": "windows-arm64",
}

# CTK 13.3.0 renamed the redistrib key from cuda_cccl to cccl.
COMPONENT_ALIASES: dict[str, tuple[str, ...]] = {
    "cuda_cccl": ("cccl",),
}

PREVIEW_COMPONENT_PACKAGES: dict[str, str] = {
    "cuda_cccl": "cccl",
    "cuda_crt": "cuda-crt",
    "cuda_cudart": "cuda-cudart-dev",
    "cuda_cupti": "cuda-cupti-dev",
    "cuda_nvcc": "cuda-nvcc",
    "cuda_nvrtc": "cuda-nvrtc-dev",
    "cuda_profiler_api": "cuda-profiler-api",
    "libcudla": "libcudla-dev",
    "libcufile": "libcufile-dev",
    "libnvfatbin": "libnvfatbin-dev",
    "libnvjitlink": "libnvjitlink-dev",
    "libnvvm": "libnvvm",
}

# Top-level directories inside the Windows local installer archive.
PREVIEW_WINDOWS_ARCHIVE_DIRS: dict[str, str] = {
    "cuda_cccl": "cccl",
    "cuda_crt": "cuda_crt",
    "cuda_cudart": "cuda_cudart",
    "cuda_cupti": "cuda_cupti",
    "cuda_nvcc": "cuda_nvcc",
    "cuda_nvrtc": "cuda_nvrtc",
    "cuda_profiler_api": "cuda_profiler_api",
    "libnvfatbin": "libnvfatbin",
    "libnvjitlink": "libnvjitlink",
    "libnvvm": "libnvvm",
}

# Paths inside the extracted installer tree for each component.
PREVIEW_WINDOWS_COMPONENT_ROOTS: dict[str, str] = {
    "cuda_cccl": "cccl/cccl",
    "cuda_crt": "cuda_crt/crt",
    "cuda_cudart": "cuda_cudart/cudart",
    "cuda_cupti": "cuda_cupti/cupti",
    "cuda_nvcc": "cuda_nvcc/nvcc",
    "cuda_nvrtc": "cuda_nvrtc",
    "cuda_profiler_api": "cuda_profiler_api/cuda_profiler_api",
    "libnvfatbin": "libnvfatbin/nvfatbin",
    "libnvjitlink": "libnvjitlink/nvjitlink",
    "libnvvm": "libnvvm/nvvm/nvvm",
}


@dataclass(frozen=True)
class PreviewInstaller:
    url: str
    sha256: str


# Source: https://packages.nvidia.com/prerelease/cuda/13.4.0/local_installers/sha256sum.txt
PREVIEW_WINDOWS_INSTALLERS: dict[tuple[str, str], PreviewInstaller] = {
    ("13.4.0", "win-64"): PreviewInstaller(
        url=("https://packages.nvidia.com/prerelease/cuda/13.4.0/local_installers/cuda_13.4.0_windows_x86_64.exe"),
        sha256="b743a3323116bf33404953ef58a9b9a3319368241f6352e933e9461409e9a759",
    ),
    ("13.4.0", "win-arm64"): PreviewInstaller(
        url=("https://packages.nvidia.com/prerelease/cuda/13.4.0/local_installers/cuda_13.4.0_windows_arm64.exe"),
        sha256="a1f68c81160b16d519c4087788b9c07de41306c3f1b872471ceee0996621374d",
    ),
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
        return json.loads(Path(metadata_path).read_text(encoding="utf-8"))

    assert metadata_url is not None
    metadata_url = validate_metadata_url(metadata_url)
    with urllib.request.urlopen(metadata_url) as response:  # noqa: S310 - scheme is restricted to https above
        return json.load(response)


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
        if ctk_subdir in metadata.get(resolved_component, {}):
            filtered.append(resolved_component)
        else:
            skipped.append(component)
    return filtered, skipped


def get_preview_packages(*, host_platform: str, cuda_version: str, components: str) -> tuple[list[str], list[str]]:
    if not host_platform.startswith("linux-"):
        raise ValueError(f"CUDA prerelease packages are not supported for host-platform {host_platform!r}")

    version_parts = cuda_version.split(".")
    if len(version_parts) != 3 or not all(part.isdigit() for part in version_parts):
        raise ValueError(f"invalid cuda-version: {cuda_version!r}")
    package_suffix = "-".join(version_parts[:2])

    packages = []
    skipped = []
    for component in filter_static_components(split_components(components), host_platform, cuda_version):
        if component == "libcudla" and host_platform != "linux-aarch64":
            skipped.append(component)
            continue
        try:
            package_base = PREVIEW_COMPONENT_PACKAGES[component]
        except KeyError as exc:
            raise ValueError(f"unsupported CUDA prerelease component: {component!r}") from exc
        package = f"{package_base}-{package_suffix}"
        if package not in packages:
            packages.append(package)
    return packages, skipped


def windows_arch_for_host_platform(host_platform: str) -> str:
    if host_platform == "win-arm64":
        return "arm64"
    if host_platform == "win-64":
        return "x64"
    raise ValueError(f"unsupported Windows host-platform: {host_platform!r}")


def get_preview_windows_archive_dirs(
    *, host_platform: str, cuda_version: str, components: str
) -> tuple[list[str], list[str]]:
    if not host_platform.startswith("win-"):
        raise ValueError(f"CUDA prerelease Windows installer is not supported for host-platform {host_platform!r}")

    archive_dirs: list[str] = []
    skipped: list[str] = []
    for component in filter_static_components(split_components(components), host_platform, cuda_version):
        if component == "libcudla":
            skipped.append(component)
            continue
        try:
            archive_dir = PREVIEW_WINDOWS_ARCHIVE_DIRS[component]
        except KeyError as exc:
            raise ValueError(f"unsupported CUDA prerelease component: {component!r}") from exc
        if archive_dir not in archive_dirs:
            archive_dirs.append(archive_dir)
    return archive_dirs, skipped


def _merge_tree(source: Path, destination: Path) -> None:
    if not source.exists():
        return
    destination.mkdir(parents=True, exist_ok=True)
    for item in source.iterdir():
        target = destination / item.name
        if item.is_dir():
            _merge_tree(item, target)
        elif target.exists():
            target.unlink()
            shutil.copy2(item, target)
        else:
            shutil.copy2(item, target)


def merge_windows_preview_ctk(
    *,
    extract_root: Path,
    destination: Path,
    host_platform: str,
    cuda_version: str,
    components: str,
) -> None:
    arch = windows_arch_for_host_platform(host_platform)
    if destination.exists():
        shutil.rmtree(destination)
    destination.mkdir(parents=True)

    for component in filter_static_components(split_components(components), host_platform, cuda_version):
        if component not in PREVIEW_WINDOWS_COMPONENT_ROOTS:
            continue
        component_root = extract_root / PREVIEW_WINDOWS_COMPONENT_ROOTS[component]
        if not component_root.exists():
            raise ValueError(f"CUDA prerelease installer did not provide {component_root}")

        lib_dir = destination / "lib" / arch

        if component == "cuda_nvrtc":
            _merge_tree(component_root / "nvrtc_dev/include", destination / "include")
            _merge_tree(component_root / "nvrtc_dev/lib" / arch, lib_dir)
            _merge_tree(component_root / "nvrtc/bin" / arch, destination / "bin")
            continue

        if component == "cuda_cupti":
            _merge_tree(component_root / "extras/CUPTI", destination / "extras/CUPTI")
            continue

        if component == "libnvvm":
            _merge_tree(component_root, destination / "nvvm")
            continue

        _merge_tree(component_root / "include", destination / "include")
        arch_bin = component_root / "bin" / arch
        if arch_bin.exists():
            _merge_tree(arch_bin, destination / "bin")
        elif (component_root / "bin").exists():
            _merge_tree(component_root / "bin", destination / "bin")
        arch_lib = component_root / "lib" / arch
        if arch_lib.exists():
            _merge_tree(arch_lib, lib_dir)

    if not (destination / "include").is_dir() or not (destination / "bin/nvcc.exe").is_file():
        raise ValueError("CUDA prerelease installer did not provide the expected toolkit layout")


def get_preview_installer(*, host_platform: str, cuda_version: str) -> PreviewInstaller:
    try:
        return PREVIEW_WINDOWS_INSTALLERS[(cuda_version, host_platform)]
    except KeyError as exc:
        raise ValueError(
            f"CUDA prerelease installer is not supported for "
            f"cuda-version {cuda_version!r}, host-platform {host_platform!r}"
        ) from exc


def get_component_relative_path(metadata: dict[str, Any], *, host_platform: str, component: str) -> str:
    ctk_subdir = host_platform_to_subdir(host_platform)
    component = resolve_component_name(metadata, component)
    component_info = metadata.get(component)
    if component_info is None:
        raise KeyError(f"unknown CTK component {component!r}")

    subdir_info = component_info.get(ctk_subdir)
    if subdir_info is None:
        raise KeyError(f"CTK component {component!r} is not available for redistrib subdir {ctk_subdir!r}")

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

    preview_parser = subparsers.add_parser("preview-packages")
    preview_parser.add_argument("--host-platform", required=True)
    preview_parser.add_argument("--cuda-version", required=True)
    preview_parser.add_argument("--components", required=True)

    preview_installer_parser = subparsers.add_parser("preview-installer")
    preview_installer_parser.add_argument("--host-platform", required=True)
    preview_installer_parser.add_argument("--cuda-version", required=True)

    preview_windows_archives_parser = subparsers.add_parser("preview-windows-archives")
    preview_windows_archives_parser.add_argument("--host-platform", required=True)
    preview_windows_archives_parser.add_argument("--cuda-version", required=True)
    preview_windows_archives_parser.add_argument("--components", required=True)

    merge_windows_preview_parser = subparsers.add_parser("merge-windows-preview")
    merge_windows_preview_parser.add_argument("--host-platform", required=True)
    merge_windows_preview_parser.add_argument("--cuda-version", required=True)
    merge_windows_preview_parser.add_argument("--components", required=True)
    merge_windows_preview_parser.add_argument("--extract-root", required=True)
    merge_windows_preview_parser.add_argument("--destination", required=True)

    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)

    try:
        if args.command == "preview-packages":
            packages, skipped = get_preview_packages(
                host_platform=args.host_platform,
                cuda_version=args.cuda_version,
                components=args.components,
            )
            for component in skipped:
                print(
                    f"Skipping unsupported CUDA prerelease component {component!r} "
                    f"for host-platform {args.host_platform!r}",
                    file=sys.stderr,
                )
            print(",".join(packages))
            return 0

        if args.command == "preview-installer":
            installer = get_preview_installer(
                host_platform=args.host_platform,
                cuda_version=args.cuda_version,
            )
            print(f"{installer.url}\t{installer.sha256}")
            return 0

        if args.command == "preview-windows-archives":
            archive_dirs, skipped = get_preview_windows_archive_dirs(
                host_platform=args.host_platform,
                cuda_version=args.cuda_version,
                components=args.components,
            )
            for component in skipped:
                print(
                    f"Skipping unsupported CUDA prerelease component {component!r} "
                    f"for host-platform {args.host_platform!r}",
                    file=sys.stderr,
                )
            print(",".join(archive_dirs))
            return 0

        if args.command == "merge-windows-preview":
            merge_windows_preview_ctk(
                extract_root=Path(args.extract_root),
                destination=Path(args.destination),
                host_platform=args.host_platform,
                cuda_version=args.cuda_version,
                components=args.components,
            )
            return 0

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
