# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0


"""
Tool to check for Cython ABI changes in a given package.

The workflow is basically:

1) Build and install a "clean" upstream version of the package.

2) Generate ABI files from the package by running (in the same venv in which the
   package is installed), where `package_name` is the import path to the package,
   e.g. `cuda.bindings`:

     python check_cython_abi.py generate <package_name> <dir>

3) Checkout a version with the changes to be tested, and build and install.

4) Check the ABI against the previously generated files by running:

     python check_cython_abi.py check <package_name> <dir>
"""

import importlib
import json
import re
import sys
import sysconfig
from pathlib import Path

EXT_SUFFIX = sysconfig.get_config_var("EXT_SUFFIX")
ABI_SUFFIX = ".abi.json"


def short_stem(name: str) -> str:
    return name.split(".", 1)[0]


def get_package_path(package_name: str) -> Path:
    package = importlib.import_module(package_name)
    return Path(package.__file__).parent


def import_from_path(root_package: str, root_dir: Path, path: Path) -> object:
    path = path.relative_to(root_dir)
    parts = [root_package] + list(path.parts[:-1]) + [short_stem(path.name)]
    return importlib.import_module(".".join(parts))


def so_path_to_abi_path(so_path: Path, build_dir: Path, abi_dir: Path) -> Path:
    abi_name = short_stem(so_path.name) + ABI_SUFFIX
    return abi_dir / so_path.parent.relative_to(build_dir) / abi_name


def abi_path_to_so_path(abi_path: Path, build_dir: Path, abi_dir: Path) -> Path:
    so_name = short_stem(abi_path.name) + EXT_SUFFIX
    return build_dir / abi_path.parent.relative_to(abi_dir) / so_name


def pyx_capi_to_json(d: dict[str, object]) -> dict[str, str]:
    """
    Converts the __pyx_capi__ dictionary to a JSON-serializable dictionary,
    removing any memory addresses that are irrelevant for comparison.
    """

    def extract_name(v: object) -> str:
        v = str(v)
        match = re.match(r'<capsule object "([^\"]+)" at 0x[0-9a-fA-F]+>', v)
        if match is None:
            raise ValueError(f"Could not parse __pyx_capi__ entry: {v}")
        return match.group(1)

    # Sort the dictionary by keys to make diffs in the JSON files smaller
    return {k: extract_name(d[k]) for k in sorted(d.keys())}


def check_abi(expected: dict[str, str], found: dict[str, str]) -> tuple[bool, bool]:
    has_errors = False
    has_allowed_changes = False
    for k, v in expected.items():
        if k not in found:
            print(f"  Missing symbol: {k}")
            has_errors = True
        elif found[k] != v:
            print(f"  Changed symbol: {k}: expected {v}, got {found[k]}")
            has_errors = True
    for k, v in found.items():
        if k not in expected:
            print(f"  Added symbol: {k}")
            has_allowed_changes = True
    return has_errors, has_allowed_changes


def check(package: str, abi_dir: Path) -> bool:
    build_dir = get_package_path(package)

    has_errors = False
    has_allowed_changes = False
    for abi_path in Path(abi_dir).glob(f"**/*{ABI_SUFFIX}"):
        so_path = abi_path_to_so_path(abi_path, build_dir, abi_dir)
        if so_path.is_file():
            module = import_from_path(package, build_dir, so_path)
            if hasattr(module, "__pyx_capi__"):
                found_json = pyx_capi_to_json(module.__pyx_capi__)
                with open(abi_path, encoding="utf-8") as f:
                    expected_json = json.load(f)
                print(f"Checking module: {so_path.relative_to(build_dir)}")
                check_errors, check_allowed_changes = check_abi(expected_json, found_json)
                has_errors |= check_errors
                has_allowed_changes |= check_allowed_changes
            else:
                print(f"Module no longer has an exposed ABI: {so_path.relative_to(build_dir)}")
                has_errors = True
        else:
            print(f"No module found for {abi_path.relative_to(abi_dir)}")
            has_errors = True

    for so_path in Path(build_dir).glob(f"**/*{EXT_SUFFIX}"):
        module = import_from_path(package, build_dir, so_path)
        if hasattr(module, "__pyx_capi__"):
            abi_path = so_path_to_abi_path(so_path, build_dir, abi_dir)
            if not abi_path.is_file():
                print(f"New module added {so_path.relative_to(build_dir)}")
                has_allowed_changes = True

    if has_errors:
        print("ERRORS FOUND")
        return True
    elif has_allowed_changes:
        print("Allowed changes found.")
    return False


def regenerate(package: str, abi_dir: Path) -> bool:
    if not abi_dir.is_dir():
        abi_dir.mkdir(parents=True, exist_ok=True)

    build_dir = get_package_path(package)
    for so_path in Path(build_dir).glob(f"**/*{EXT_SUFFIX}"):
        print(f"Generating ABI from {so_path.relative_to(build_dir)}")
        module = import_from_path(package, build_dir, so_path)
        if hasattr(module, "__pyx_capi__"):
            abi_path = so_path_to_abi_path(so_path, build_dir, abi_dir)
            abi_path.parent.mkdir(parents=True, exist_ok=True)
            with open(abi_path, "w", encoding="utf-8") as f:
                json.dump(pyx_capi_to_json(module.__pyx_capi__), f, indent=2)

    return False


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        prog="check_cython_abi", description="Checks for changes in the Cython ABI of a given package"
    )

    subparsers = parser.add_subparsers()

    regen_parser = subparsers.add_parser("generate", help="Regenerate the ABI files")
    regen_parser.set_defaults(func=regenerate)
    regen_parser.add_argument("package", help="Python package to collect data from")
    regen_parser.add_argument("dir", help="Output directory to save data to")

    check_parser = subparsers.add_parser("check", help="Check the API against existing ABI files")
    check_parser.set_defaults(func=check)
    check_parser.add_argument("package", help="Python package to collect data from")
    check_parser.add_argument("dir", help="Input directory to read data from")

    args = parser.parse_args()
    if hasattr(args, "func"):
        if args.func(args.package, Path(args.dir)):
            sys.exit(1)
    else:
        parser.print_help()
        sys.exit(1)
