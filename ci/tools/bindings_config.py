# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Load and validate the CUDA bindings package-root registry."""

from __future__ import annotations

import ast
import json
import re
import sys
from argparse import ArgumentParser
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping

import tomllib
import yaml
from packaging.version import InvalidVersion, Version

REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CONFIG = REPO_ROOT / "ci" / "versions.yml"
SCHEMA_VERSION = 2
RELEASE_STATUSES = frozenset({"current", "maintenance"})
_DEPENDENT_RELEASE_TAG_PREFIXES = ("cuda-core-v", "cuda-pathfinder-v")

_NAME_PATTERN = re.compile(r"[a-z][a-z0-9]*(?:-[a-z0-9]+)*")
_PACKAGE_ROOT_PATTERN = re.compile(r"[A-Za-z0-9._-]+(?:/[A-Za-z0-9._-]+)*")
_TOOLKIT_VERSION_PATTERN = re.compile(r"[1-9][0-9]*\.[0-9]+\.[0-9]+")
_RELEASE_VERSION_PATTERN = re.compile(
    r"(?:0|[1-9][0-9]*)\.(?:0|[1-9][0-9]*)\.(?:0|[1-9][0-9]*)"
    r"(?:(?:a|b|rc)(?:0|[1-9][0-9]*))?"
    r"(?:\.post(?:0|[1-9][0-9]*))?"
    r"(?:\.dev(?:0|[1-9][0-9]*))?"
)


class BindingsConfigError(ValueError):
    """The CUDA bindings package-root registry is invalid."""


def parse_pep440_version(value: str, label: str = "version") -> Version:
    """Parse a PEP 440 version and give configuration errors useful context."""
    try:
        return Version(value)
    except InvalidVersion as error:
        raise BindingsConfigError(f"{label} is not a valid PEP 440 version: {value!r}") from error


def parse_prefixed_version(tag: str, prefix: str) -> Version | None:
    """Parse the PEP 440 version following an exact component tag prefix."""
    if not tag.startswith(prefix):
        return None
    value = tag.removeprefix(prefix)
    # Keep one canonical spelling for every accepted release version while
    # delegating PEP 440 interpretation and comparison to packaging.
    if _RELEASE_VERSION_PATTERN.fullmatch(value) is None:
        return None
    try:
        version = parse_pep440_version(value, "release tag version")
    except BindingsConfigError:
        return None
    return None if version.local is not None else version


@dataclass(frozen=True)
class BindingsPackage:
    package_root: str
    toolkit_version: str
    release_status: str

    @property
    def ctk_target(self) -> str:
        major, minor, _ = self.toolkit_version.split(".", maxsplit=2)
        return f"{major}.{minor}"

    @property
    def cuda_major(self) -> str:
        return self.ctk_target.partition(".")[0]

    @property
    def cuda_variant(self) -> str:
        return f"cu{self.cuda_major}"

    def version_from_tag(self, tag: str) -> Version | None:
        """Return a matching release version for this configured package root."""
        version = parse_prefixed_version(tag, "v")
        if version is None or f"{version.release[0]}.{version.release[1]}" != self.ctk_target:
            return None
        return version

    def matches_tag(self, tag: str) -> bool:
        return self.version_from_tag(tag) is not None

    def to_dict(self) -> dict[str, object]:
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
    schema_version: int
    package_roots: tuple[BindingsPackage, ...]

    def get_package(self, package_root: str) -> BindingsPackage:
        package = next((package for package in self.package_roots if package.package_root == package_root), None)
        if package is None:
            raise BindingsConfigError(f"unknown CUDA bindings package root: {package_root!r}")
        return package

    def package_for_release_status(self, release_status: str) -> BindingsPackage:
        return next(package for package in self.package_roots if package.release_status == release_status)

    def match_tag(self, tag: str) -> BindingsPackage | None:
        return next((package for package in self.package_roots if package.matches_tag(tag)), None)

    def to_dict(self) -> dict[str, object]:
        return {
            "schema_version": self.schema_version,
            "package_roots": [package.to_dict() for package in self.package_roots],
        }

    def to_json(self) -> str:
        return json.dumps(self.to_dict(), separators=(",", ":"), sort_keys=True)


def _mapping(value: Any, label: str, keys: set[str] | None = None) -> Mapping[str, Any]:
    if not isinstance(value, dict) or not all(isinstance(key, str) for key in value):
        raise BindingsConfigError(f"{label} must be a mapping with string keys")
    if keys is not None and set(value) != keys:
        raise BindingsConfigError(f"{label} must contain exactly: {', '.join(sorted(keys))}")
    return value


def _text(value: Any, label: str, pattern: re.Pattern[str]) -> str:
    if not isinstance(value, str) or not value or value != value.strip():
        raise BindingsConfigError(f"{label} must be a non-empty, trimmed string")
    if pattern.fullmatch(value) is None:
        raise BindingsConfigError(f"{label} has invalid format: {value!r}")
    return value


def parse_package_root(value: Any, label: str = "package_root") -> str:
    """Validate a repository-relative package root."""
    package_root = _text(value, label, _PACKAGE_ROOT_PATTERN)
    if any(part in (".", "..") for part in package_root.split("/")):
        raise BindingsConfigError(f"{label} must be a normalized repository-relative POSIX path: {package_root!r}")
    return package_root


def _package(package_root: str, raw: Any) -> BindingsPackage:
    package_root = parse_package_root(package_root, "CUDA bindings package root")
    data = _mapping(
        raw,
        f"CUDA bindings package root {package_root!r}",
        {"toolkit_version", "release_status"},
    )
    return BindingsPackage(
        package_root=package_root,
        toolkit_version=_text(
            data["toolkit_version"],
            f"{package_root}.toolkit_version",
            _TOOLKIT_VERSION_PATTERN,
        ),
        release_status=_text(
            data["release_status"],
            f"{package_root}.release_status",
            _NAME_PATTERN,
        ),
    )


def _validate_release_statuses(packages: tuple[BindingsPackage, ...]) -> None:
    release_statuses = [package.release_status for package in packages]
    if len(packages) != 2 or set(release_statuses) != RELEASE_STATUSES:
        raise BindingsConfigError(
            "cuda.bindings.package_roots must contain exactly one current and one maintenance release status"
        )


def validate_config(raw: Any) -> BindingsConfig:
    root = _mapping(raw, "versions configuration", {"schema_version", "cuda"})
    if type(root["schema_version"]) is not int or root["schema_version"] != SCHEMA_VERSION:
        raise BindingsConfigError(f"schema_version must be {SCHEMA_VERSION}")
    cuda = _mapping(root["cuda"], "cuda", {"bindings"})
    bindings = _mapping(cuda["bindings"], "cuda.bindings", {"package_roots"})
    raw_package_roots = _mapping(bindings["package_roots"], "cuda.bindings.package_roots")
    packages = tuple(_package(package_root, value) for package_root, value in raw_package_roots.items())
    _validate_release_statuses(packages)
    cuda_majors = [package.cuda_major for package in packages]
    if len(set(cuda_majors)) != len(cuda_majors):
        raise BindingsConfigError("CUDA bindings cuda_major values must be unique")
    return BindingsConfig(
        schema_version=SCHEMA_VERSION,
        package_roots=packages,
    )


def load_config(path: Path = DEFAULT_CONFIG) -> BindingsConfig:
    try:
        raw = yaml.safe_load(path.read_text(encoding="utf-8"))
    except (OSError, yaml.YAMLError) as error:
        raise BindingsConfigError(f"could not read {path}: {error}") from error
    return validate_config(raw)


def check_package_metadata(config: BindingsConfig, repo_root: Path) -> None:
    """Check consistency of package metadata in the selected source tree."""
    selectors: dict[str, str] = {}
    maintenance_fallback = None
    for package in config.package_roots:
        path = repo_root / package.package_root / "pyproject.toml"
        try:
            with path.open("rb") as stream:
                scm = tomllib.load(stream)["tool"]["setuptools_scm"]
        except (OSError, tomllib.TOMLDecodeError, KeyError, TypeError) as error:
            raise BindingsConfigError(f"could not read SCM metadata from {path}: {error}") from error
        if not isinstance(scm, dict) or "tag_regex" in scm:
            raise BindingsConfigError(f"{path} must use setuptools-scm's default tag parser")
        command = scm.get("git_describe_command")
        if (
            not isinstance(command, list)
            or not all(isinstance(part, str) for part in command)
            or command.count("--match") != 1
            or command[-1] == "--match"
        ):
            raise BindingsConfigError(f"{path} must specify one git_describe_command --match selector")
        selector = command[command.index("--match") + 1]
        # Maintenance excludes the old .0 baseline; agreement between copies alone
        # must not allow that baseline to reenter standard SCM version selection.
        patch_selector = "[1-9]*" if package.release_status == "maintenance" else "*"
        expected_selector = f"v{package.ctk_target}.{patch_selector}"
        if selector != expected_selector:
            raise BindingsConfigError(
                f"{path} --match must select registered CUDA {package.ctk_target} "
                f"using {expected_selector!r} for the {package.release_status} root, got {selector!r}"
            )
        selectors[package.cuda_major] = selector
        if package.release_status == "maintenance":
            maintenance_fallback = scm.get("fallback_version")
            if not isinstance(maintenance_fallback, str):
                raise BindingsConfigError(f"{path} must declare a maintenance fallback_version")
            fallback = parse_pep440_version(maintenance_fallback, f"{path} fallback_version")
            if fallback.dev is None or fallback.release[:2] != tuple(map(int, package.ctk_target.split("."))):
                raise BindingsConfigError(
                    f"{path} fallback_version must be a CUDA {package.ctk_target} development version"
                )
        elif "fallback_version" in scm:
            raise BindingsConfigError(f"{path} current bindings root must not declare fallback_version")

    setup_path = repo_root / "cuda_python" / "setup.py"
    names = {"SCM_DESCRIBE_MATCH_BY_MAJOR", "MAINTENANCE_FALLBACK_VERSION"}
    try:
        tree = ast.parse(setup_path.read_text(encoding="utf-8"))
        metadata = {
            target.id: ast.literal_eval(node.value)
            for node in tree.body
            if isinstance(node, ast.Assign)
            for target in node.targets
            if isinstance(target, ast.Name) and target.id in names
        }
    except (OSError, SyntaxError, ValueError) as error:
        raise BindingsConfigError(f"could not read literal build metadata from {setup_path}: {error}") from error
    if metadata.get("SCM_DESCRIBE_MATCH_BY_MAJOR") != selectors:
        raise BindingsConfigError(f"{setup_path} SCM_DESCRIBE_MATCH_BY_MAJOR must match the bindings package selectors")
    if metadata.get("MAINTENANCE_FALLBACK_VERSION") != maintenance_fallback:
        raise BindingsConfigError(
            f"{setup_path} MAINTENANCE_FALLBACK_VERSION must match the maintenance bindings package"
        )

    maintenance = config.package_for_release_status("maintenance")
    pixi_path = repo_root / maintenance.package_root / "pixi.toml"
    try:
        with pixi_path.open("rb") as stream:
            pixi_version = tomllib.load(stream)["package"]["version"]
    except (OSError, tomllib.TOMLDecodeError, KeyError, TypeError) as error:
        raise BindingsConfigError(f"could not read package version from {pixi_path}: {error}") from error
    if pixi_version != maintenance_fallback:
        raise BindingsConfigError(f"{pixi_path} package.version must match the maintenance bindings fallback_version")


def resolve_release_bindings_package(
    release_tag: str,
    release_source_root: Path,
) -> dict[str, object]:
    """Resolve a release using only the schema-2 registry in its tagged source."""
    if not release_source_root.is_dir():
        raise BindingsConfigError(f"release source root is not a directory: {release_source_root}")

    config_path = release_source_root / "ci" / "versions.yml"
    try:
        config = load_config(config_path)
    except BindingsConfigError as error:
        raise BindingsConfigError(
            f"release source requires a valid schema-{SCHEMA_VERSION} registry at {config_path}: {error}. "
            "For pre-registry tags, use compatible historical release tooling."
        ) from error
    bindings_version = parse_prefixed_version(release_tag, "v")
    is_dependent_release = any(
        parse_prefixed_version(release_tag, prefix) is not None for prefix in _DEPENDENT_RELEASE_TAG_PREFIXES
    )
    if bindings_version is None and not is_dependent_release:
        raise BindingsConfigError(f"unsupported release tag: {release_tag!r}")

    package = config.package_for_release_status("current") if is_dependent_release else config.match_tag(release_tag)
    if package is None:
        raise BindingsConfigError(
            f"no CUDA bindings package root in tagged config {config_path} matches release tag: {release_tag!r}"
        )

    record: dict[str, object] = {
        "package_root": package.package_root,
        "toolkit_version": package.toolkit_version,
    }
    if bindings_version is not None:
        record["release_version"] = str(bindings_version)
    return record


def write_github_env(data: Mapping[str, object], path: Path) -> None:
    """Append the bindings build environment consumed by documentation jobs."""
    package_root = parse_package_root(data.get("package_root"), "resolved package_root")
    toolkit_version = _text(
        data.get("toolkit_version"),
        "resolved toolkit_version",
        _TOOLKIT_VERSION_PATTERN,
    )
    with path.open("a", encoding="utf-8") as stream:
        stream.write(f"BUILD_CTK_VER={toolkit_version}\n")
        stream.write(f"BINDINGS_PACKAGE_ROOT={package_root}\n")


def main(argv: list[str] | None = None) -> int:
    """Emit normalized registry JSON or export one package for GitHub Actions."""
    parser = ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    output = parser.add_mutually_exclusive_group()
    output.add_argument("--package-roots", action="store_true", help="print normalized package-root records")
    output.add_argument(
        "--release-status",
        choices=sorted(RELEASE_STATUSES),
        help="print the package root with this release status",
    )
    output.add_argument("--release-tag", help="resolve a release tag against its source tree")
    output.add_argument(
        "--check-package-metadata",
        action="store_true",
        help="check package metadata in the source tree containing --config (ci/versions.yml)",
    )
    parser.add_argument("--release-source-root", type=Path)
    commands = parser.add_subparsers(dest="command")
    write_env = commands.add_parser(
        "write-github-env",
        help="append bindings build variables from package JSON on stdin",
    )
    write_env.add_argument("github_env", type=Path, metavar="GITHUB_ENV")
    args = parser.parse_args(argv)

    try:
        if args.command == "write-github-env":
            if (
                args.package_roots
                or args.release_status
                or args.release_tag
                or args.check_package_metadata
                or args.release_source_root is not None
            ):
                parser.error("write-github-env does not accept registry selectors")
            value = json.load(sys.stdin)
            if not isinstance(value, dict):
                raise BindingsConfigError("stdin for write-github-env must contain a JSON object")
            write_github_env(value, args.github_env)
            return 0
        if args.release_tag:
            if args.release_source_root is None:
                parser.error("--release-tag requires --release-source-root")
            value: object = resolve_release_bindings_package(
                args.release_tag,
                args.release_source_root,
            )
        else:
            if args.release_source_root is not None:
                parser.error("--release-source-root requires --release-tag")
            config = load_config(args.config)
            if args.check_package_metadata:
                check_package_metadata(config, args.config.resolve().parent.parent)
                return 0
            if args.package_roots:
                value = [package.to_dict() for package in config.package_roots]
            elif args.release_status:
                value = config.package_for_release_status(args.release_status).to_dict()
            else:
                value = config.to_dict()
        print(json.dumps(value, separators=(",", ":"), sort_keys=True))
        return 0
    except (BindingsConfigError, json.JSONDecodeError) as error:
        parser.error(str(error))


if __name__ == "__main__":
    sys.exit(main())
