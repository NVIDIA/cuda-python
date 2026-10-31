# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Read bindings package roots and release roles from the existing toolkit pins."""

from __future__ import annotations

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


@dataclass(frozen=True)
class BindingsConfig:
    package_roots: tuple[BindingsPackage, ...]

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
    return BindingsConfig(tuple(packages))
