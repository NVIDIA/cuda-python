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
    """Apply the workplan to rows already filtered by the workflow's jq filter."""
    current_package = next(package for package in packages if package["release_status"] == "current")
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
        ctk_target = ".".join(source_row["CUDA_VER"].split(".")[:2])
        package = next((package for package in packages if package["ctk_target"] == ctk_target), None)
        # Bindings and metapackage suites need the row's own local wheels;
        # published-floor rows test Core or a nightly integration only.
        local = package is not None
        bindings = package or current_package
        root = bindings["package_root"]
        tests = {
            "TEST_PATHFINDER": modules is None or modules["pathfinder"]["needs_test"],
            "TEST_BINDINGS": local and (modules is None or modules["bindings"]["package_roots"][root]["needs_test"]),
            "TEST_CORE": modules is None or modules["core"]["cuda_majors"][variant]["needs_test"],
            "TEST_PYTHON": local and (modules is None or modules["python"]["package_roots"][root]["needs_test"]),
        }
        suites = []
        if test_mode != "standard":
            suites.append(test_mode)
        else:
            if tests["TEST_PATHFINDER"]:
                suites.append("pathfinder")
            if tests["TEST_BINDINGS"]:
                suites.append(f"bindings{major}")
            if tests["TEST_CORE"]:
                suites.append(f"core{major}")
            if tests["TEST_PYTHON"]:
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
                **{name: str(value).lower() for name, value in tests.items()},
                "RUNNER_DRIVER": runner_driver,
                "BINDINGS_SOURCE": "local" if local else "floor",
                "BINDINGS_SOURCE_DIR": root,
                "BINDINGS_BUILD_VERSION": bindings["toolkit_version"],
                "CURRENT_BUILD_CUDA_VER": current_package["toolkit_version"],
                "PATHFINDER_SOURCE": "artifact" if workplan is None else workplan["sources"]["pathfinder"],
                "SUITES": "+".join(suites),
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
        config = json.loads(os.environ["BINDINGS_CONFIG"])
        result = select_test_matrix(
            json.load(sys.stdin),
            config["package_roots"],
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
