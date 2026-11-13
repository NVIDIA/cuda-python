# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Materialize build and test environments from CI decisions."""

from __future__ import annotations

import argparse
import json
import os
import subprocess
from collections.abc import Mapping
from pathlib import Path

if __package__:
    from .ci_artifacts import bindings_artifact, core_artifact, python_artifact
else:
    from ci_artifacts import bindings_artifact, core_artifact, python_artifact


def common_environment(env: Mapping[str, str], root: Path) -> dict[str, str]:
    python, host, sha = env["PY_VER"], env["HOST_PLATFORM"], env["SHA"]
    return {
        "CUDA_CORE_ARTIFACT_BASENAME": core_artifact(python, host),
        "CUDA_CORE_ARTIFACT_NAME": core_artifact(python, host, sha),
        "CUDA_CORE_ARTIFACTS_DIR": (root / "cuda_core/dist").as_posix(),
        "CUDA_CORE_CYTHON_TESTS_DIR": (root / "cuda_core/tests/cython").as_posix(),
        "CUDA_CORE_TEST_BINARIES_DIR": (root / "cuda_core/tests/test_binaries").as_posix(),
    }


def build_environment(env: Mapping[str, str], root: Path) -> dict[str, object]:
    if __package__:
        from .bindings_config import load_config
    else:
        from bindings_config import load_config

    config = load_config(root / "ci/versions.yml")
    plan = json.loads(env["WORKPLAN"])
    modules = plan["modules"]
    python, host, sha = env["PY_VER"], env["HOST_PLATFORM"], env["SHA"]
    platform_tag = {
        "linux-64": "manylinux_x86_64",
        "linux-aarch64": "manylinux_aarch64",
        "win-64": "win_amd64",
        "win-arm64": "win_arm64",
    }[host]
    values: dict[str, object] = {
        "CIBW_BUILD": f"cp{python.replace('.', '')}-{platform_tag}",
        "PATHFINDER_SOURCE": plan["sources"]["pathfinder"],
        "BUILD_PATHFINDER": modules["pathfinder"]["needs_build"],
        "BUILD_BINDINGS": modules["bindings"]["needs_build"],
        "BUILD_CORE": modules["core"]["needs_build"],
        "BUILD_CUDA_PYTHON": modules["python"]["needs_build"],
        "TEST_BINDINGS": modules["bindings"]["needs_test"],
        "TEST_CORE": modules["core"]["needs_test"],
        "BASELINE_RUN_ID": plan["baseline"]["run_id"],
        "BASELINE_SHA": plan["baseline"]["sha"],
    }
    for package in config.package_roots:
        prefix = package.release_status.upper()
        source = root / package.package_root
        bindings = modules["bindings"]["package_roots"][package.package_root]
        metapackage = modules["python"]["package_roots"][package.package_root]
        core = modules["core"]["cuda_majors"][package.cuda_variant]
        values.update(
            {
                f"{prefix}_BINDINGS_ROOT": package.package_root,
                f"{prefix}_CUDA_VERSION": package.toolkit_version,
                f"{prefix}_CUDA_MAJOR": package.cuda_major,
                f"{prefix}_CUDA_VARIANT": package.cuda_variant,
                f"{prefix}_BINDINGS_ARTIFACT_BASENAME": bindings_artifact(python, package.toolkit_version, host),
                f"{prefix}_BINDINGS_ARTIFACT_NAME": bindings_artifact(python, package.toolkit_version, host, sha),
                f"{prefix}_BINDINGS_ARTIFACTS_DIR": (source / "dist").as_posix(),
                f"{prefix}_BINDINGS_CYTHON_TESTS_DIR": (source / "tests/cython").as_posix(),
                f"{prefix}_PYTHON_ARTIFACT_NAME": python_artifact(package.toolkit_version),
                f"BUILD_BINDINGS_{prefix}": bindings["needs_build"],
                f"BUILD_CUDA_PYTHON_{prefix}": metapackage["needs_build"],
                f"TEST_BINDINGS_{prefix}": bindings["needs_test"],
                f"TEST_CORE_{prefix}": core["needs_test"],
            }
        )
    return values


def test_environment(env: Mapping[str, str], root: Path) -> dict[str, str]:
    python, host, sha = env["PY_VER"], env["HOST_PLATFORM"], env["SHA"]
    test_major, test_minor = env["CUDA_VER"].split(".")[:2]
    current = env["CURRENT_BUILD_CUDA_VER"]
    source = env["BINDINGS_SOURCE"]
    bindings_root, build = env["BINDINGS_SOURCE_DIR"], env["BUILD_CUDA_VER"]
    sanitizer = (
        python == "3.12" and env["CUDA_VER"] != "11.8.0" and env["LOCAL_CTK"] == "1" and host.startswith("linux")
    )
    values = {
        "SETUP_SANITIZER": "1" if sanitizer else "0",
        "BINDINGS_SOURCE": source,
        "CUDA_BINDINGS_ROOT": bindings_root,
        "PATHFINDER_SOURCE": env["PATHFINDER_SOURCE"],
        "CUDA_PYTHON_ARTIFACT_NAME": python_artifact(build),
        # Compiled Cython tests require the local bindings they were built with.
        "SKIP_CYTHON_TEST": "0" if source == "local" else "1",
        "TEST_CUDA_MAJOR": test_major,
        "TEST_CUDA_MINOR": test_minor,
        "CUDA_BINDINGS_ARTIFACT_NAME": bindings_artifact(python, build, host, sha),
        "CUDA_BINDINGS_ARTIFACTS_DIR": (root / bindings_root / "dist").as_posix(),
        "CUDA_BINDINGS_CYTHON_TESTS_DIR": (root / bindings_root / "tests/cython").as_posix(),
        "CUDA_CORE_CYTHON_TEST_ARTIFACT_NAME": core_artifact(python, host, sha) + f"-cu{test_major}-tests",
        "CURRENT_BUILD_CUDA_VER": current,
        "BUILD_CUDA_MAJOR": current.split(".")[0],
        "BUILD_CUDA_MINOR": current.split(".")[1],
    }
    if not sanitizer:
        values["SANITIZER_CMD"] = ""
    return values


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("mode", choices=("build", "test"))
    args = parser.parse_args()
    root = Path.cwd()
    values: dict[str, object] = common_environment(os.environ, root)
    values.update((build_environment if args.mode == "build" else test_environment)(os.environ, root))
    values["CUDA_PYTHON_PARALLEL_LEVEL"] = subprocess.check_output(["nproc"], text=True).strip()  # noqa: S607
    with Path(os.environ["GITHUB_ENV"]).open("a", encoding="utf-8") as output:
        for name, value in values.items():
            output.write(f"{name}={str(value).lower() if isinstance(value, bool) else value}\n")
    with Path(os.environ["GITHUB_PATH"]).open("a", encoding="utf-8") as output:
        output.write(f"{Path(__file__).resolve().parent}\n")


if __name__ == "__main__":
    main()
