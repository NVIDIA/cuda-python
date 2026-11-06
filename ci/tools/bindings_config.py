# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Read bindings package roots and release roles from the existing toolkit pins."""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import dataclass
from pathlib import Path

import yaml

DEFAULT_CONFIG = Path(__file__).resolve().parents[2] / "ci" / "versions.yml"


@dataclass(frozen=True)
class BindingsPackage:
    package_root: str
    toolkit: tuple[int, int, int]
    release_status: str

    @property
    def toolkit_version(self) -> str:
        return ".".join(map(str, self.toolkit))

    @property
    def ctk_target(self) -> str:
        return f"{self.toolkit[0]}.{self.toolkit[1]}"

    @property
    def cuda_major(self) -> str:
        return str(self.toolkit[0])

    @property
    def cuda_variant(self) -> str:
        return f"cu{self.cuda_major}"

    def to_dict(self) -> dict[str, str]:
        return {
            "package_root": self.package_root,
            "toolkit_version": self.toolkit_version,
            "release_status": self.release_status,
            "ctk_target": self.ctk_target,
            "cuda_major": self.cuda_major,
            "cuda_variant": self.cuda_variant,
        }


@dataclass(frozen=True)
class BindingsConfig:
    package_roots: tuple[BindingsPackage, ...]
    python_supported: tuple[str, ...]

    def get_package(self, package_root: str) -> BindingsPackage:
        for package in self.package_roots:
            if package.package_root == package_root:
                return package
        raise ValueError(f"unknown CUDA bindings package root: {package_root!r}")

    def package_for_release_status(self, release_status: str) -> BindingsPackage:
        return next(package for package in self.package_roots if package.release_status == release_status)


def load_config(path: Path = DEFAULT_CONFIG) -> BindingsConfig:
    """Normalize legacy build/prev_build fields without changing their consumers."""
    raw = yaml.safe_load(path.read_text(encoding="utf-8"))
    packages = []
    for field, release_status in (("build", "current"), ("prev_build", "maintenance")):
        major, minor, patch = map(int, raw["cuda"][field]["version"].split("."))
        packages.append(BindingsPackage(f"cuda_bindings_{major}", (major, minor, patch), release_status))
    return BindingsConfig(tuple(packages), tuple(raw["python"]["supported"]))


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    output = parser.add_mutually_exclusive_group()
    output.add_argument("--package-roots", action="store_true")
    output.add_argument("--release-status", choices=("current", "maintenance"))
    args = parser.parse_args(argv)

    config = load_config(args.config)
    if args.release_status:
        result = config.package_for_release_status(args.release_status).to_dict()
    else:
        packages = [package.to_dict() for package in config.package_roots]
        result = (
            packages
            if args.package_roots
            else {
                "python": {"supported": list(config.python_supported)},
                "package_roots": packages,
            }
        )
    print(json.dumps(result, separators=(",", ":"), sort_keys=True))
    return 0


if __name__ == "__main__":
    sys.exit(main())
