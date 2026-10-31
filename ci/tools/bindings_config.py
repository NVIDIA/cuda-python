# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Read bindings configuration and resolve releases against their tagged source."""

from __future__ import annotations

import argparse
import importlib.util
import json
import re
import sys
import tomllib
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping

import yaml
from packaging.version import InvalidVersion, Version

REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CONFIG = REPO_ROOT / "ci" / "versions.yml"
SCHEMA_VERSION = 2
RELEASE_STATUSES = frozenset({"current", "maintenance"})
_DEPENDENT_RELEASE_TAG_PREFIXES = ("cuda-core-v", "cuda-pathfinder-v")
_PACKAGE_ROOT_PATTERN = re.compile(r"[A-Za-z0-9._-]+(?:/[A-Za-z0-9._-]+)*")
_TOOLKIT_VERSION_PATTERN = re.compile(r"[1-9][0-9]*\.[0-9]+\.[0-9]+")


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

    def version_from_tag(self, tag: str) -> Version | None:
        """Return a matching release version for this configured package root."""
        version = parse_prefixed_version(tag, "v")
        if version is None or version.release[:2] != self.toolkit[:2]:
            return None
        return version

    def matches_tag(self, tag: str) -> bool:
        return self.version_from_tag(tag) is not None

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

    def to_dict(self) -> dict[str, object]:
        return {
            "python": {"supported": list(self.python_supported)},
            "package_roots": [package.to_dict() for package in self.package_roots],
        }

    def to_json(self) -> str:
        return json.dumps(self.to_dict(), separators=(",", ":"), sort_keys=True)

    def match_tag(self, tag: str) -> BindingsPackage | None:
        return next((package for package in self.package_roots if package.matches_tag(tag)), None)


def parse_config(raw: Mapping[str, Any]) -> BindingsConfig:
    """Normalize legacy toolkit pins or the explicit package-root registry."""
    packages = []
    if "schema_version" in raw:
        if raw["schema_version"] != SCHEMA_VERSION:
            raise ValueError(f"schema_version must be {SCHEMA_VERSION}")
        for package_root, data in raw["cuda"]["bindings"]["package_roots"].items():
            major, minor, patch = map(int, data["toolkit_version"].split("."))
            packages.append(BindingsPackage(package_root, (major, minor, patch), data["release_status"]))
    else:
        for field, release_status in (("build", "current"), ("prev_build", "maintenance")):
            major, minor, patch = map(int, raw["cuda"][field]["version"].split("."))
            packages.append(BindingsPackage(f"cuda_bindings_{major}", (major, minor, patch), release_status))
    if len(packages) != 2 or {package.release_status for package in packages} != RELEASE_STATUSES:
        raise ValueError("bindings configuration must contain one current and one maintenance package")
    if len({package.toolkit[0] for package in packages}) != len(packages):
        raise ValueError("CUDA bindings cuda_major values must be unique")
    return BindingsConfig(tuple(packages), tuple(raw["python"]["supported"]))


def load_config(path: Path = DEFAULT_CONFIG) -> BindingsConfig:
    return parse_config(yaml.safe_load(path.read_text(encoding="utf-8")))


def check_core_toolkit_floors(config: BindingsConfig, repo_root: Path) -> None:
    """Check toolkit pins against core's existing, standard-library floor reader."""
    core = repo_root / "cuda_core"
    floor_path = core / "cuda" / "core" / "_bindings_floor.py"
    spec = importlib.util.spec_from_file_location("_cuda_core_bindings_floor", floor_path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    with (core / "pyproject.toml").open("rb") as stream:
        extras = tomllib.load(stream)["project"]["optional-dependencies"]
    floors = module.floors_from_extras(extras)
    pins = {int(package.cuda_major): package for package in config.package_roots}
    if pins.keys() != floors.keys():
        raise ValueError("registered CUDA majors must match cuda_core's cu<N> extras")
    for major, floor in floors.items():
        package = pins[major]
        if package.toolkit[:2] < floor[:2]:
            raise ValueError(
                f"{package.package_root} toolkit {package.toolkit_version} is below "
                f"the CUDA {major} bindings floor {'.'.join(map(str, floor))}"
            )


def check_package_metadata(config: BindingsConfig, repo_root: Path) -> None:
    """Check consistency of package metadata in the selected source tree."""
    check_core_toolkit_floors(config, repo_root)
    selectors: dict[str, str] = {}
    for package in config.package_roots:
        path = repo_root / package.package_root / "pyproject.toml"
        with path.open("rb") as stream:
            scm = tomllib.load(stream)["tool"]["setuptools_scm"]
        if "tag_regex" in scm:
            raise ValueError(f"{path} must use setuptools-scm's default tag parser")
        command = scm["git_describe_command"]
        if command.count("--match") != 1:
            raise ValueError(f"{path} must specify one git_describe_command --match selector")
        selector = command[command.index("--match") + 1]
        expected_selector = f"v{package.ctk_target}.*"
        if selector != expected_selector:
            raise ValueError(
                f"{path} --match must select registered CUDA {package.ctk_target} "
                f"using {expected_selector!r} for the {package.release_status} root, got {selector!r}"
            )
        selectors[package.cuda_major] = selector
        if "fallback_version" in scm:
            raise ValueError(f"{path} bindings root must not declare fallback_version")

    path = repo_root / "cuda_python" / "pyproject.toml"
    with path.open("rb") as stream:
        meta_selectors = tomllib.load(stream)["tool"]["cuda-python"]["scm-describe-match-by-major"]
    if meta_selectors != selectors:
        raise ValueError(f"{path} scm-describe-match-by-major must match the bindings package selectors")


def parse_prefixed_version(tag: str, prefix: str) -> Version | None:
    """Parse the PEP 440 version following an exact component tag prefix."""
    if not tag.startswith(prefix):
        return None
    value = tag.removeprefix(prefix)
    try:
        version = Version(value)
    except InvalidVersion:
        return None
    if str(version) != value or len(version.release) != 3 or version.epoch != 0 or version.local is not None:
        return None
    return version


def parse_package_root(value: Any, label: str = "package_root") -> str:
    """Validate a package path received through the release JSON interface."""
    if (
        not isinstance(value, str)
        or _PACKAGE_ROOT_PATTERN.fullmatch(value) is None
        or any(part in (".", "..") for part in value.split("/"))
    ):
        raise ValueError(f"{label} must be a normalized repository-relative POSIX path: {value!r}")
    return value


def resolve_release_bindings_package(
    release_tag: str,
    release_source_root: Path,
) -> dict[str, object]:
    """Resolve a release from its tagged source, including retained CUDA 13 tags."""
    if not release_source_root.is_dir():
        raise ValueError(f"release source root is not a directory: {release_source_root}")

    config_path = release_source_root / "ci" / "versions.yml"
    bindings_version = parse_prefixed_version(release_tag, "v")
    is_dependent_release = any(
        parse_prefixed_version(release_tag, prefix) is not None for prefix in _DEPENDENT_RELEASE_TAG_PREFIXES
    )
    if bindings_version is None and not is_dependent_release:
        raise ValueError(f"unsupported release tag: {release_tag!r}")

    try:
        raw = yaml.safe_load(config_path.read_text(encoding="utf-8"))
        legacy_config = "schema_version" not in raw
        if legacy_config:
            current_toolkit = raw["cuda"]["build"]["version"]
            if _TOOLKIT_VERSION_PATTERN.fullmatch(current_toolkit) is None or not current_toolkit.startswith("13."):
                raise ValueError("legacy release compatibility requires a CUDA 13 build toolkit")
            current_root = (
                "cuda_bindings_13" if (release_source_root / "cuda_bindings_13").is_dir() else "cuda_bindings"
            )
            package = BindingsPackage(current_root, tuple(map(int, current_toolkit.split("."))), "current")
            if bindings_version is not None and bindings_version.release[0] == 12:
                maintenance_toolkit = raw["cuda"]["prev_build"]["version"]
                if _TOOLKIT_VERSION_PATTERN.fullmatch(maintenance_toolkit) is None:
                    raise ValueError("legacy prev_build toolkit must have major.minor.patch format")
                package = BindingsPackage(
                    "cuda_bindings_12", tuple(map(int, maintenance_toolkit.split("."))), "maintenance"
                )
        else:
            config = parse_config(raw)
            package = (
                config.package_for_release_status("current") if is_dependent_release else config.match_tag(release_tag)
            )
    except (OSError, yaml.YAMLError, KeyError, TypeError, ValueError) as error:
        raise ValueError(f"could not resolve release source config {config_path}: {error}") from error
    if package is None:
        raise ValueError(
            f"no CUDA bindings package root in tagged config {config_path} matches release tag: {release_tag!r}"
        )
    if bindings_version is not None and not package.matches_tag(release_tag):
        raise ValueError(f"release tag {release_tag!r} does not match tagged toolkit {package.toolkit_version}")
    if legacy_config and not (release_source_root / package.package_root).is_dir():
        raise ValueError(f"tagged source has no bindings root {package.package_root!r}")

    record: dict[str, object] = {
        "package_root": package.package_root,
        "toolkit_version": package.toolkit_version,
    }
    if bindings_version is not None:
        record["release_version"] = str(bindings_version)
    if legacy_config and package.release_status == "current":
        record["allow_legacy_python_artifact"] = True
    return record


def write_github_env(data: Mapping[str, object], path: Path) -> None:
    """Append the bindings build environment consumed by documentation jobs."""
    package_root = parse_package_root(data.get("package_root"), "resolved package_root")
    toolkit_version = data.get("toolkit_version")
    if not isinstance(toolkit_version, str) or _TOOLKIT_VERSION_PATTERN.fullmatch(toolkit_version) is None:
        raise ValueError(f"resolved toolkit_version must have major.minor.patch format: {toolkit_version!r}")
    with path.open("a", encoding="utf-8") as stream:
        stream.write(f"BUILD_CTK_VER={toolkit_version}\n")
        stream.write(f"BINDINGS_PACKAGE_ROOT={package_root}\n")
        stream.write(f"RESOLVED_BINDINGS_PACKAGE={json.dumps(data, separators=(',', ':'))}\n")


def write_build_github_env(
    config: BindingsConfig,
    path: Path,
    *,
    repo_root: Path,
    python_version: str,
    host_platform: str,
    sha: str,
) -> None:
    """Append both bindings lines' build paths and artifact names for Actions."""
    python_tag = python_version.replace(".", "")
    values = {}
    for role in ("current", "maintenance"):
        package = config.package_for_release_status(role)
        prefix = role.upper()
        source = repo_root / package.package_root
        artifact = f"cuda-bindings-python{python_tag}-cuda{package.toolkit_version}-{host_platform}"
        values.update(
            {
                f"{prefix}_BINDINGS_ROOT": package.package_root,
                f"{prefix}_CUDA_VERSION": package.toolkit_version,
                f"{prefix}_CUDA_MAJOR": package.cuda_major,
                f"{prefix}_CUDA_VARIANT": package.cuda_variant,
                f"{prefix}_BINDINGS_ARTIFACT_BASENAME": artifact,
                f"{prefix}_BINDINGS_ARTIFACT_NAME": f"{artifact}-{sha}",
                # Forward slashes work in native Python and Git Bash on Windows.
                f"{prefix}_BINDINGS_ARTIFACTS_DIR": (source / "dist").resolve().as_posix(),
                f"{prefix}_BINDINGS_CYTHON_TESTS_DIR": (source / "tests" / "cython").resolve().as_posix(),
                f"{prefix}_PYTHON_ARTIFACT_NAME": f"cuda-python-wheel-cuda{package.toolkit_version}",
            }
        )
    with path.open("a", encoding="utf-8") as stream:
        for name, value in values.items():
            stream.write(f"{name}={value}\n")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    output = parser.add_mutually_exclusive_group()
    output.add_argument("--package-roots", action="store_true")
    output.add_argument("--check-package-metadata", action="store_true")
    output.add_argument("--release-status", choices=sorted(RELEASE_STATUSES))
    output.add_argument("--release-tag", help="resolve a release tag against its source tree")
    parser.add_argument("--release-source-root", type=Path)
    commands = parser.add_subparsers(dest="command")
    write_env = commands.add_parser("write-github-env", help="append package JSON from stdin to GITHUB_ENV")
    write_env.add_argument("github_env", type=Path, metavar="GITHUB_ENV")
    write_build_env = commands.add_parser("write-build-env", help="append both bindings lines' build variables")
    write_build_env.add_argument("github_env", type=Path, metavar="GITHUB_ENV")
    write_build_env.add_argument("--repo-root", type=Path, default=REPO_ROOT)
    write_build_env.add_argument("--python-version", required=True)
    write_build_env.add_argument("--host-platform", required=True)
    write_build_env.add_argument("--sha", required=True)
    args = parser.parse_args(argv)

    try:
        if args.command == "write-build-env":
            write_build_github_env(
                load_config(args.config),
                args.github_env,
                repo_root=args.repo_root,
                python_version=args.python_version,
                host_platform=args.host_platform,
                sha=args.sha,
            )
            return 0
        if args.command == "write-github-env":
            if (
                args.package_roots
                or args.check_package_metadata
                or args.release_status
                or args.release_tag
                or args.release_source_root is not None
            ):
                parser.error("write-github-env does not accept registry selectors")
            value = json.load(sys.stdin)
            if not isinstance(value, dict):
                raise ValueError("stdin for write-github-env must contain a JSON object")
            write_github_env(value, args.github_env)
            return 0
        if args.release_tag:
            if args.release_source_root is None:
                parser.error("--release-tag requires --release-source-root")
            result: object = resolve_release_bindings_package(args.release_tag, args.release_source_root)
        else:
            if args.release_source_root is not None:
                parser.error("--release-source-root requires --release-tag")
            config = load_config(args.config)
            if args.check_package_metadata:
                check_package_metadata(config, REPO_ROOT)
                return 0
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
    except ValueError as error:
        parser.error(str(error))


if __name__ == "__main__":
    sys.exit(main())
