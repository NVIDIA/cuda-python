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
_TAGGED_CONFIG_FILENAMES = ("versions.yml", "versions.json")

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


def _legacy_tag_pattern(repo_root: Path, package_root: str) -> re.Pattern[str] | None:
    """Return a tagged tree's custom SCM parser, if one was configured."""
    path = repo_root / package_root / "pyproject.toml"
    try:
        with path.open("rb") as stream:
            pyproject = tomllib.load(stream)
    except (OSError, tomllib.TOMLDecodeError) as error:
        raise BindingsConfigError(f"could not inspect legacy package metadata {path}: {error}") from error

    pattern = pyproject.get("tool", {}).get("setuptools_scm", {}).get("tag_regex")
    if pattern is not None and (not isinstance(pattern, str) or not pattern):
        raise BindingsConfigError(f"[tool.setuptools_scm].tag_regex in {path} must be a non-empty string")
    return _compile_tag_regex(pattern, f"[tool.setuptools_scm].tag_regex in {path}") if pattern else None


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
    """Check current source-build metadata without constraining historical trees."""
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
        # The second form excludes the pre-maintenance .0 tag on main.
        if selector not in {f"v{package.ctk_target}.*", f"v{package.ctk_target}.[1-9]*"}:
            raise BindingsConfigError(
                f"{path} --match must select registered CUDA {package.ctk_target}, got {selector!r}"
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


def _tag_tree_config(release_source_root: Path) -> tuple[BindingsConfig | None, Any, Path | None]:
    """Load a schema-2 tag-tree registry and retain legacy metadata."""
    config_path = next(
        (
            release_source_root / "ci" / filename
            for filename in _TAGGED_CONFIG_FILENAMES
            if (release_source_root / "ci" / filename).is_file()
        ),
        None,
    )
    if config_path is None:
        return None, None, None

    try:
        text = config_path.read_text(encoding="utf-8")
        raw: Any = json.loads(text) if config_path.suffix == ".json" else yaml.safe_load(text)
    except (OSError, json.JSONDecodeError, yaml.YAMLError) as error:
        raise BindingsConfigError(f"could not inspect tagged config {config_path}: {error}") from error

    if not isinstance(raw, dict):
        raise BindingsConfigError(f"tagged config {config_path} must contain a mapping")
    if "schema_version" not in raw:
        return None, raw, config_path
    try:
        return validate_config(raw), raw, config_path
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

    control = load_config(control_config_path)
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

    tag_pattern = _legacy_tag_pattern(release_source_root, package_root)
    if tag_pattern is not None:
        # Historical release trees can deliberately omit a prerelease suffix
        # from the SCM version captured by their custom regex.
        match = tag_pattern.match(release_tag)
        parsed_tag = match.group("version") if match else ""
    else:
        parsed_tag = release_tag
    release_version = parse_prefixed_version(parsed_tag, "v")
    if release_version is None:
        raise BindingsConfigError(f"legacy source metadata does not match release tag: {release_tag!r}")
    return {
        "package_root": package_root,
        "toolkit_version": _legacy_toolkit_version(raw, release_version, control_config_path),
        "release_version": str(release_version),
        "release_registry_origin": "control",
    }


def _legacy_dependency_package(release_source_root: Path, raw: Any) -> dict[str, object]:
    """Resolve the current bindings dependency from a legacy release tree."""
    package_root = "cuda_bindings"
    if not (release_source_root / package_root).is_dir():
        raise BindingsConfigError(f"legacy release package root is missing: {package_root}")
    try:
        toolkit_version = raw["cuda"]["build"]["version"]
    except (KeyError, TypeError):
        toolkit_version = None
    if toolkit_version is None:
        raise BindingsConfigError("legacy tagged config has no cuda.build.version for the bindings dependency")
    return {
        "package_root": package_root,
        "toolkit_version": _text(toolkit_version, "legacy cuda.build.version", _TOOLKIT_VERSION_PATTERN),
        # Legacy CI used the unqualified cuda-python-wheel artifact name.
        "release_registry_origin": "control",
    }


def _release_record(package: BindingsPackage, version: Version, origin: str) -> dict[str, object]:
    """Return the package fields consumed by release jobs."""
    return {
        "package_root": package.package_root,
        "toolkit_version": package.toolkit_version,
        "release_version": str(version),
        "release_registry_origin": origin,
    }


def resolve_release_bindings_package(
    release_tag: str,
    release_source_root: Path,
    control_config_path: Path,
) -> dict[str, object]:
    """Resolve the bindings package needed by a release from its tag tree."""
    if not release_source_root.is_dir():
        raise BindingsConfigError(f"release source root is not a directory: {release_source_root}")

    config, raw, tagged_config_path = _tag_tree_config(release_source_root)
    config_source = f"tagged config {tagged_config_path}" if tagged_config_path is not None else "tagged config"
    bindings_version = parse_prefixed_version(release_tag, "v")
    is_dependent_release = any(
        parse_prefixed_version(release_tag, prefix) is not None for prefix in _DEPENDENT_RELEASE_TAG_PREFIXES
    )
    if bindings_version is None and not is_dependent_release:
        raise BindingsConfigError(f"unsupported release tag: {release_tag!r}")

    if config is None:
        if is_dependent_release:
            return _legacy_dependency_package(release_source_root, raw)
        return _legacy_release_package(release_tag, release_source_root, control_config_path, raw)

    package = config.package_for_release_status("current") if is_dependent_release else config.match_tag(release_tag)
    if package is None:
        raise BindingsConfigError(
            f"no CUDA bindings package root in {config_source} matches release tag: {release_tag!r}"
        )

    if is_dependent_release:
        return {
            "package_root": package.package_root,
            "toolkit_version": package.toolkit_version,
            "release_registry_origin": "tag",
        }

    version = package.version_from_tag(release_tag)
    assert version is not None
    return _release_record(package, version, "tag")


def write_github_env(data: Mapping[str, object], path: Path) -> None:
    """Append the bindings build environment consumed by documentation jobs."""
    package_root = parse_package_root(data.get("package_root"), "resolved package_root")
    toolkit_version = _text(
        data.get("toolkit_version"),
        "resolved toolkit_version",
        _TOOLKIT_VERSION_PATTERN,
    )
    origin = data.get("release_registry_origin", "tag")
    if origin not in {"tag", "control"}:
        raise BindingsConfigError("release_registry_origin must be tag or control")
    with path.open("a", encoding="utf-8") as stream:
        stream.write(f"BUILD_CTK_VER={toolkit_version}\n")
        stream.write(f"BINDINGS_PACKAGE_ROOT={package_root}\n")
        stream.write(f"BINDINGS_REGISTRY_ORIGIN={origin}\n")


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
                or args.check_package_metadata
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
