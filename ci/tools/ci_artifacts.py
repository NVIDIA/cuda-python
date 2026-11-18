# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Artifact names shared by wheel producers, consumers and baseline checks."""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from .bindings_config import BindingsConfig


def bindings_artifact(python_version: str, toolkit_version: str, host_platform: str, sha: str = "") -> str:
    name = f"cuda-bindings-python{python_version.replace('.', '')}-cuda{toolkit_version}-{host_platform}"
    return f"{name}-{sha}" if sha else name


def core_artifact(python_version: str, host_platform: str, sha: str = "") -> str:
    name = f"cuda-core-python{python_version.replace('.', '')}-{host_platform}"
    return f"{name}-{sha}" if sha else name


def python_artifact(toolkit_version: str) -> str:
    return f"cuda-python-wheel-cuda{toolkit_version}"


def expected_baseline_artifacts(config: BindingsConfig, platforms: list[str], sha: str) -> list[str]:
    """Return the complete wheel set required before enabling artifact reuse."""
    from .bindings_config import wheel_python_versions

    current = config.package_for_release_status("current")
    windows_arm64_versions = wheel_python_versions(current.toolkit_version, "win-arm64", config.python_supported)
    platforms = [*platforms, *(["win-arm64"] if windows_arm64_versions else [])]
    names = ["cuda-pathfinder-wheel", *(python_artifact(package.toolkit_version) for package in config.package_roots)]
    for python_version in config.python_supported:
        for platform in platforms:
            if platform == "win-arm64" and python_version not in windows_arm64_versions:
                continue
            for package in config.package_roots:
                if platform == "win-arm64" and config.windows_arm64_single_cuda_major and package != current:
                    continue
                names.append(bindings_artifact(python_version, package.toolkit_version, platform, sha))
            names.append(core_artifact(python_version, platform, sha))
    return names


def main() -> None:
    import argparse
    from pathlib import Path

    import yaml

    from .bindings_config import load_config

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--baseline-sha", required=True)
    args = parser.parse_args()
    matrix = Path(__file__).resolve().parents[2] / "ci" / "test-matrix.yml"
    platforms = yaml.safe_load(matrix.read_text(encoding="utf-8"))["platforms"]
    print("\n".join(expected_baseline_artifacts(load_config(), platforms, args.baseline_sha)))


if __name__ == "__main__":
    main()
