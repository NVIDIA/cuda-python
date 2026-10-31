# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Select wheel-test rows and their local or published bindings sources."""

from __future__ import annotations

import argparse
import json
import os
import sys
from typing import Any


def select_test_matrix(
    rows: list[dict[str, Any]],
    packages: list[dict[str, Any]],
    *,
    workplan: dict[str, Any] | None = None,
    build_type: str = "pull-request",
    test_mode: str = "standard",
    platform: str = "linux",
) -> dict[str, list[dict[str, Any]]]:
    """Apply the workplan after the caller's optional jq matrix filter."""
    current_root = next(package["package_root"] for package in packages if package["release_status"] == "current")
    enabled = (
        {package["cuda_variant"]: True for package in packages}
        if workplan is None
        else workplan["jobs"]["test_cuda_majors"]
    )
    modules = None if workplan is None else workplan["modules"]
    selected = []
    for source_row in rows:
        major = source_row["CUDA_VER"].split(".")[0]
        variant = f"cu{major}"
        if not enabled.get(variant, False):
            continue
        package = next((package for package in packages if package["toolkit_version"] == source_row["CUDA_VER"]), None)
        root = "" if package is None else package["package_root"]
        suites = []
        if test_mode != "standard":
            suites.append(test_mode)
        else:
            if modules is None or modules["pathfinder"]["needs_test"]:
                suites.append("pathfinder")
            if package is not None and (modules is None or modules["bindings"]["package_roots"][root]["needs_test"]):
                suites.append(f"bindings{major}")
            if modules is None or modules["core"]["cuda_majors"][variant]["needs_test"]:
                suites.append(f"core{major}")
            if package is not None and (modules is None or modules["python"]["package_roots"][root]["needs_test"]):
                suites.append(f"meta{major}")
        if not suites:
            continue
        driver = source_row["DRIVER"]
        # Custom versions are installed after scheduling on a latest-driver
        # runner. WSL cannot perform that driver swap.
        runner_driver = driver if driver in ("latest", "earliest") else "latest"
        if platform == "linux" and runner_driver != driver and source_row.get("FLAVOR") == "wsl":
            raise ValueError("custom DRIVER is not supported with FLAVOR=wsl")
        selected.append(
            {
                **source_row,
                "CUDA_VARIANT": variant,
                "BINDINGS_SOURCE": "floor" if package is None else "local",
                "BINDINGS_SOURCE_DIR": root,
                "DEFAULT_BINDINGS_SOURCE_DIR": current_root,
                "PATHFINDER_SOURCE": "artifact" if workplan is None else workplan["sources"]["pathfinder"],
                "SUITES": "+".join(suites),
                "RUNNER_DRIVER": runner_driver,
            }
        )
    if not selected:
        raise ValueError("Empty matrix after CUDA-major workplan filtering")
    # Nightly integrations may deliberately use published bindings with older
    # toolkits. Ordinary PRs must not silently lose a required local-wheel row.
    if test_mode == "standard" and build_type == "pull-request":
        covered = {row["BINDINGS_SOURCE_DIR"] for row in selected if row["BINDINGS_SOURCE"] == "local"}
        missing = [
            package["package_root"]
            for package in packages
            if enabled.get(package["cuda_variant"], False)
            and (
                modules is None
                or modules["bindings"]["package_roots"][package["package_root"]]["needs_test"]
                or modules["python"]["package_roots"][package["package_root"]]["needs_test"]
            )
            and package["package_root"] not in covered
        ]
        if missing:
            raise ValueError(f"filtered test matrix omits required local-wheel coverage: {', '.join(missing)}")
    return {"include": selected}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--platform", choices=("linux", "windows"), required=True)
    args = parser.parse_args()
    try:
        result = select_test_matrix(
            json.load(sys.stdin),
            json.loads(os.environ["BINDINGS_CONFIG"])["package_roots"],
            workplan=json.loads(os.environ.get("WORKPLAN") or "null"),
            build_type=os.environ["BUILD_TYPE"],
            test_mode=os.environ["TEST_MODE"],
            platform=args.platform,
        )
    except ValueError as error:
        parser.error(str(error))
    print(json.dumps(result, separators=(",", ":")))


if __name__ == "__main__":
    main()
