# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Load and validate the CUDA bindings package-root registry."""

from __future__ import annotations

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


def _compile_tag_regex(pattern: str, label: str) -> re.Pattern[str]:
    try:
        compiled = re.compile(pattern)
    except re.error as error:
        raise BindingsConfigError(f"{label} is not a valid regular expression: {error}") from error
    if "version" not in compiled.groupindex:
        raise BindingsConfigError(f"{label} must define a named 'version' group")
    return compiled


@dataclass(frozen=True)
class BindingsPackage:
    package_root: str
    toolkit_version: str
    release_status: str | None
    tag_regex: str

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

    def scm_version_from_tag(self, tag: str, *, fullmatch: bool = True) -> Version | None:
        """Return the PEP 440 version captured by this package root's SCM regex."""
        regex = _compile_tag_regex(self.tag_regex, f"{self.package_root} setuptools-scm tag_regex")
        match = regex.fullmatch(tag) if fullmatch else regex.match(tag)
        if match is None:
            return None
        try:
            version = parse_pep440_version(match.group("version"), f"release tag {tag!r}")
        except BindingsConfigError:
            return None
        return None if version.local is not None else version

    def version_from_tag(self, tag: str) -> Version | None:
        """Return a matching release version for this configured package root."""
        version = self.scm_version_from_tag(tag)
        toolkit = parse_pep440_version(self.toolkit_version, f"{self.package_root}.toolkit_version")
        if version is None or version.release[:2] != toolkit.release[:2]:
            return None
        return version

    def matches_tag(self, tag: str) -> bool:
        return self.version_from_tag(tag) is not None

    def to_dict(self) -> dict[str, object]:
        return {
            "package_root": self.package_root,
            "toolkit_version": self.toolkit_version,
            "release_status": self.release_status,
            "tag_regex": self.tag_regex,
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


def _package_root(value: Any, label: str) -> str:
    package_root = _text(value, label, _PACKAGE_ROOT_PATTERN)
    if any(part in (".", "..") for part in package_root.split("/")):
        raise BindingsConfigError(f"{label} must be a normalized repository-relative POSIX path: {package_root!r}")
    return package_root


def _read_tag_regex(repo_root: Path, package_root: str) -> str:
    path = repo_root / package_root / "pyproject.toml"
    try:
        with path.open("rb") as stream:
            pyproject = tomllib.load(stream)
        pattern = pyproject["tool"]["setuptools_scm"]["tag_regex"]
    except (OSError, KeyError, TypeError, tomllib.TOMLDecodeError) as error:
        raise BindingsConfigError(f"could not read [tool.setuptools_scm].tag_regex from {path}: {error}") from error
    if not isinstance(pattern, str) or not pattern:
        raise BindingsConfigError(f"[tool.setuptools_scm].tag_regex in {path} must be a non-empty string")
    _compile_tag_regex(pattern, f"[tool.setuptools_scm].tag_regex in {path}")
    return pattern


def _package(package_root: str, raw: Any, repo_root: Path) -> BindingsPackage:
    package_root = _package_root(package_root, "CUDA bindings package root")
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
        tag_regex=_read_tag_regex(repo_root, package_root),
    )


def package_from_dict(data: Mapping[str, object]) -> BindingsPackage:
    """Validate a normalized package record passed between release jobs."""
    package_root = _package_root(data.get("package_root"), "resolved package_root")
    toolkit_version = _text(
        data.get("toolkit_version"),
        "resolved toolkit_version",
        _TOOLKIT_VERSION_PATTERN,
    )
    release_status_value = data.get("release_status")
    release_status = (
        None if release_status_value is None else _text(release_status_value, "resolved release_status", _NAME_PATTERN)
    )
    if release_status is not None and release_status not in RELEASE_STATUSES:
        raise BindingsConfigError(
            f"resolved release_status must be one of {', '.join(sorted(RELEASE_STATUSES))}: {release_status!r}"
        )
    tag_regex = data.get("tag_regex")
    if not isinstance(tag_regex, str) or not tag_regex:
        raise BindingsConfigError("resolved tag_regex must be a non-empty string")
    _compile_tag_regex(tag_regex, "resolved tag_regex")
    package = BindingsPackage(package_root, toolkit_version, release_status, tag_regex)
    expected = package.to_dict()
    for key in ("ctk_target", "cuda_major", "cuda_variant"):
        if key in data and data[key] != expected[key]:
            raise BindingsConfigError(f"resolved {key} is inconsistent with toolkit_version")
    return package


def _validate_release_statuses(packages: tuple[BindingsPackage, ...]) -> None:
    release_statuses = [package.release_status for package in packages]
    if len(packages) != 2 or set(release_statuses) != RELEASE_STATUSES:
        raise BindingsConfigError(
            "cuda.bindings.package_roots must contain exactly one current and one maintenance release status"
        )


def _validate_scm_conformance(packages: tuple[BindingsPackage, ...]) -> None:
    for package in packages:
        tag = f"v{package.toolkit_version}"
        version = package.version_from_tag(tag)
        expected = parse_pep440_version(package.toolkit_version, f"{package.package_root}.toolkit_version")
        if version != expected:
            raise BindingsConfigError(
                f"{package.package_root} setuptools-scm tag_regex must match its configured toolkit release tag {tag!r}"
            )


def validate_config(raw: Any, repo_root: Path = REPO_ROOT) -> BindingsConfig:
    root = _mapping(raw, "versions configuration", {"schema_version", "cuda"})
    if type(root["schema_version"]) is not int or root["schema_version"] != SCHEMA_VERSION:
        raise BindingsConfigError(f"schema_version must be {SCHEMA_VERSION}")
    cuda = _mapping(root["cuda"], "cuda", {"bindings"})
    bindings = _mapping(cuda["bindings"], "cuda.bindings", {"package_roots"})
    raw_package_roots = _mapping(bindings["package_roots"], "cuda.bindings.package_roots")
    packages = tuple(_package(package_root, value, repo_root) for package_root, value in raw_package_roots.items())
    _validate_release_statuses(packages)
    cuda_majors = [package.cuda_major for package in packages]
    if len(set(cuda_majors)) != len(cuda_majors):
        raise BindingsConfigError("CUDA bindings cuda_major values must be unique")
    _validate_scm_conformance(packages)
    return BindingsConfig(
        schema_version=SCHEMA_VERSION,
        package_roots=packages,
    )


def load_config(path: Path = DEFAULT_CONFIG, repo_root: Path = REPO_ROOT) -> BindingsConfig:
    try:
        raw = yaml.safe_load(path.read_text(encoding="utf-8"))
    except (OSError, yaml.YAMLError) as error:
        raise BindingsConfigError(f"could not read {path}: {error}") from error
    return validate_config(raw, repo_root)


def _tag_tree_config(config_path: Path, release_source_root: Path) -> tuple[BindingsConfig | None, Any]:
    """Load a schema-2 tag-tree registry and retain legacy metadata."""
    try:
        raw: Any = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    except FileNotFoundError:
        return None, None
    except (OSError, yaml.YAMLError) as error:
        raise BindingsConfigError(f"could not inspect tagged config {config_path}: {error}") from error

    if not isinstance(raw, dict):
        raise BindingsConfigError(f"tagged config {config_path} must contain a YAML mapping")
    if "schema_version" not in raw:
        return None, raw
    try:
        return validate_config(raw, release_source_root), raw
    except BindingsConfigError as error:
        raise BindingsConfigError(f"invalid schema-2 tagged config {config_path}: {error}") from error


def _legacy_toolkit_version(raw: Any, release_version: Version, control_config_path: Path) -> str:
    """Recover the CTK build pin used by a pre-registry release tree."""
    try:
        value = raw["cuda"]["build"]["version"]
    except (KeyError, TypeError):
        value = None
    if value is not None:
        return _text(value, "legacy cuda.build.version", _TOOLKIT_VERSION_PATTERN)

    control = load_config(control_config_path, control_config_path.parent.parent)
    target = release_version.release[:2]
    if len(target) != 2:
        raise BindingsConfigError(f"legacy release version has no CUDA minor: {release_version}")
    matches = [
        package.toolkit_version
        for package in control.package_roots
        if parse_pep440_version(package.toolkit_version).release[:2] == target
    ]
    if len(matches) != 1:
        raise BindingsConfigError(
            f"control registry must contain exactly one toolkit pin for legacy CUDA {target[0]}.{target[1]}; "
            f"found {len(matches)}"
        )
    return matches[0]


def _legacy_release_package(
    release_tag: str,
    release_source_root: Path,
    control_config_path: Path,
    raw: Any,
) -> dict[str, object]:
    """Resolve a release tree from before the schema-2 registry existed."""
    package_root = "cuda_bindings"
    if not (release_source_root / package_root).is_dir():
        raise BindingsConfigError(f"legacy release package root is missing: {package_root}")

    tag_regex = _read_tag_regex(release_source_root, package_root)

    probe = BindingsPackage(package_root, "1.0.0", None, tag_regex)
    release_version = probe.scm_version_from_tag(release_tag, fullmatch=False)
    if release_version is None:
        raise BindingsConfigError(f"legacy source SCM metadata does not match release tag: {release_tag!r}")
    package = BindingsPackage(
        package_root=package_root,
        toolkit_version=_legacy_toolkit_version(raw, release_version, control_config_path),
        release_status=None,
        tag_regex=tag_regex,
    )
    normalized = package.to_dict()
    normalized.update(
        release_version=str(release_version),
        release_package_root=package_root,
        release_registry_origin="control",
    )
    return normalized


def resolve_release_bindings_package(
    release_tag: str,
    release_source_root: Path,
    control_config_path: Path,
) -> dict[str, object]:
    """Resolve a release tag using its tag tree, with legacy-layout fallback."""
    if not release_source_root.is_dir():
        raise BindingsConfigError(f"release source root is not a directory: {release_source_root}")

    tagged_config_path = release_source_root / "ci" / "versions.yml"
    config, raw = _tag_tree_config(tagged_config_path, release_source_root)
    config_source = f"tagged config {tagged_config_path}"
    if config is None:
        return _legacy_release_package(release_tag, release_source_root, control_config_path, raw)

    package = config.match_tag(release_tag)
    if package is None:
        raise BindingsConfigError(
            f"no CUDA bindings package root in {config_source} matches release tag: {release_tag!r}"
        )

    normalized = package.to_dict()
    version = package.version_from_tag(release_tag)
    assert version is not None
    normalized["release_version"] = str(version)
    normalized["release_package_root"] = package.package_root
    normalized["release_registry_origin"] = "tag"
    return normalized


def write_github_env(data: Mapping[str, object], path: Path) -> None:
    """Append the bindings build environment consumed by documentation jobs."""
    package = package_from_dict(data)
    package_root = _package_root(
        data.get("release_package_root", package.package_root),
        "release package_root",
    )
    origin = data.get("release_registry_origin", "tag")
    if origin not in {"tag", "control"}:
        raise BindingsConfigError("release_registry_origin must be tag or control")
    with path.open("a", encoding="utf-8") as stream:
        stream.write(f"BUILD_CTK_VER={package.toolkit_version}\n")
        stream.write(f"BINDINGS_PACKAGE_ROOT={package_root}\n")
        stream.write(f"BINDINGS_REGISTRY_ORIGIN={origin}\n")


def main(argv: list[str] | None = None) -> int:
    """Emit normalized registry JSON or export one package for GitHub Actions."""
    parser = ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--repo-root", type=Path, default=REPO_ROOT)
    output = parser.add_mutually_exclusive_group()
    output.add_argument("--package-roots", action="store_true", help="print normalized package-root records")
    output.add_argument(
        "--release-status",
        choices=sorted(RELEASE_STATUSES),
        help="print the package root with this release status",
    )
    output.add_argument("--release-tag", help="resolve a release tag against its source tree")
    parser.add_argument("--release-source-root", type=Path)
    parser.add_argument("--control-config", type=Path)
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
                or args.release_source_root is not None
                or args.control_config is not None
            ):
                parser.error("write-github-env does not accept registry selectors")
            value = json.load(sys.stdin)
            if not isinstance(value, dict):
                raise BindingsConfigError("stdin for write-github-env must contain a JSON object")
            write_github_env(value, args.github_env)
            return 0
        if args.release_tag:
            if args.release_source_root is None or args.control_config is None:
                parser.error("--release-tag requires --release-source-root and --control-config")
            value: object = resolve_release_bindings_package(
                args.release_tag,
                args.release_source_root,
                args.control_config,
            )
        else:
            if args.release_source_root is not None or args.control_config is not None:
                parser.error("--release-source-root and --control-config require --release-tag")
            config = load_config(args.config, args.repo_root)
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
