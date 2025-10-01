#!/usr/bin/env python

# ruff: noqa: S603, S607

# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

import argparse
import importlib
import subprocess
import sys
from pathlib import Path


def get_package_path(package_name):
    package = importlib.import_module(package_name)
    return Path(package.__file__).parent


def regenerate(build_dir, abi_dir):
    for so_path in Path(build_dir).glob("**/*.so"):
        print(f"Generating ABI from {so_path.relative_to(build_dir)}")
        abi_name = so_path.stem[: so_path.stem.find(".")] + ".abi"
        abi_path = abi_dir / so_path.parent.relative_to(build_dir) / abi_name
        abi_path.parent.mkdir(parents=True, exist_ok=True)
        subprocess.run(["abidw", so_path, "--no-architecture", "--out-file", abi_path], check=True)


def check(build_dir, abi_dir):
    found_failures = []
    missing_so = []
    for abi_path in Path(abi_dir).glob("**/*.abi"):
        so_candidates = list((Path(build_dir) / abi_path.parent.relative_to(abi_dir)).glob(f"{abi_path.stem}.*.so"))
        if len(so_candidates) == 1:
            so_path = so_candidates[0]
            proc = subprocess.run(
                [
                    "abidiff",
                    abi_path,
                    so_path,
                    "--drop-private-types",
                    "--no-architecture",
                    "--no-added-syms",
                ],
            )
            if proc.returncode != 0:
                found_failures.append(so_path)
        elif len(so_candidates) == 0:
            missing_so.append(abi_path)
        else:
            print(f"Multiple .so candidates found for {abi_path}:")
            for p in so_candidates:
                print(f"  {p}")
            missing_so.append(abi_path)

    if len(found_failures):
        print("ABI differences found in the following files:")
        for p in found_failures:
            print(f"  {p.relative_to(build_dir)}")

    if len(missing_so):
        print("Expected .so file(s) have been removed:")
        for p in missing_so:
            print(f"  {p.relative_to(abi_dir)}")

    return len(found_failures) or len(missing_so)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Check or regenerate ABI files for a package")
    parser.add_argument("action", choices=["regenerate", "check"])
    parser.add_argument("package", help="Python package path containing .so files to check")
    parser.add_argument("abi_dir", help="Directory containing .abi files")
    args = parser.parse_args()

    build_dir = get_package_path(args.package)

    if args.action == "regenerate":
        sys.exit(regenerate(build_dir, Path(args.abi_dir)))
    elif args.action == "check":
        sys.exit(check(build_dir, Path(args.abi_dir)))
