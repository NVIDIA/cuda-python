# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Load and validate the CUDA bindings release-line registry."""

from __future__ import annotations

import json
import re
import sys
from argparse import ArgumentParser
from dataclasses import dataclass
from pathlib import Path
from types import MappingProxyType
from typing import Any, Mapping

import tomllib
import yaml
from packaging.version import InvalidVersion, Version

REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CONFIG = REPO_ROOT / "ci" / "versions.yml"
SCHEMA_VERSION = 2

_NAME_PATTERN = re.compile(r"[a-z][a-z0-9]*(?:-[a-z0-9]+)*")
_SOURCE_DIR_PATTERN = re.compile(r"[A-Za-z0-9._-]+(?:/[A-Za-z0-9._-]+)*")
_TOOLKIT_VERSION_PATTERN = re.compile(r"[1-9][0-9]*\.[0-9]+\.[0-9]+")
_RELEASE_VERSION_PATTERN = re.compile(
    r"(?:0|[1-9][0-9]*)\.(?:0|[1-9][0-9]*)\.(?:0|[1-9][0-9]*)"
    r"(?:(?:a|b|rc)(?:0|[1-9][0-9]*))?"
    r"(?:\.post(?:0|[1-9][0-9]*))?"
    r"(?:\.dev(?:0|[1-9][0-9]*))?"
)


class BindingsConfigError(ValueError):
    """The CUDA bindings release-line registry is invalid."""


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
class BindingsLine:
    line_id: str
    source_dir: str
    toolkit_version: str
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
        """Return the PEP 440 version captured by this source line's SCM regex."""
        regex = _compile_tag_regex(self.tag_regex, f"{self.source_dir} setuptools-scm tag_regex")
        match = regex.fullmatch(tag) if fullmatch else regex.match(tag)
        if match is None:
            return None
        try:
            version = parse_pep440_version(match.group("version"), f"release tag {tag!r}")
        except BindingsConfigError:
            return None
        return None if version.local is not None else version

    def version_from_tag(self, tag: str) -> Version | None:
        """Return a matching release version for this configured CTK line."""
        version = self.scm_version_from_tag(tag)
        toolkit = parse_pep440_version(self.toolkit_version, f"{self.line_id}.toolkit_version")
        if version is None or version.release[:2] != toolkit.release[:2]:
            return None
        return version

    def matches_tag(self, tag: str) -> bool:
        return self.version_from_tag(tag) is not None

    def to_dict(self) -> dict[str, object]:
        return {
            "line_id": self.line_id,
            "source_dir": self.source_dir,
            "toolkit_version": self.toolkit_version,
            "tag_regex": self.tag_regex,
            "ctk_target": self.ctk_target,
            "cuda_major": self.cuda_major,
            "cuda_variant": self.cuda_variant,
        }


@dataclass(frozen=True)
class BindingsConfig:
    schema_version: int
    lines: tuple[BindingsLine, ...]
    roles: Mapping[str, str]

    def get_line(self, line_id: str) -> BindingsLine:
        line = next((line for line in self.lines if line.line_id == line_id), None)
        if line is None:
            raise BindingsConfigError(f"unknown CUDA bindings line: {line_id!r}")
        return line

    def line_for_role(self, role: str) -> BindingsLine:
        try:
            return self.get_line(self.roles[role])
        except KeyError as error:
            raise BindingsConfigError(f"unknown CUDA bindings role: {role!r}") from error

    def match_tag(self, tag: str) -> BindingsLine | None:
        matches = [line for line in self.lines if line.matches_tag(tag)]
        if len(matches) > 1:
            line_ids = ", ".join(line.line_id for line in matches)
            raise BindingsConfigError(f"release tag {tag!r} matches multiple CUDA bindings lines: {line_ids}")
        return matches[0] if matches else None

    def line_to_dict(self, line: BindingsLine) -> dict[str, object]:
        normalized: dict[str, object] = line.to_dict()
        normalized["role"] = next(role for role, line_id in self.roles.items() if line.line_id == line_id)
        return normalized

    def to_dict(self) -> dict[str, object]:
        return {
            "schema_version": self.schema_version,
            "lines": [self.line_to_dict(line) for line in self.lines],
            "roles": dict(self.roles),
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


def _source_dir(value: Any, label: str) -> str:
    source_dir = _text(value, label, _SOURCE_DIR_PATTERN)
    if any(part in (".", "..") for part in source_dir.split("/")):
        raise BindingsConfigError(f"{label} must be a normalized repository-relative POSIX path: {source_dir!r}")
    return source_dir


def _read_tag_regex(repo_root: Path, source_dir: str) -> str:
    path = repo_root / source_dir / "pyproject.toml"
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


def _line(line_id: str, raw: Any, repo_root: Path) -> BindingsLine:
    _text(line_id, "CUDA bindings line ID", _NAME_PATTERN)
    data = _mapping(
        raw,
        f"CUDA bindings line {line_id!r}",
        {"source_dir", "toolkit_version"},
    )
    source_dir = _source_dir(data["source_dir"], f"{line_id}.source_dir")
    return BindingsLine(
        line_id,
        source_dir,
        _text(data["toolkit_version"], f"{line_id}.toolkit_version", _TOOLKIT_VERSION_PATTERN),
        _read_tag_regex(repo_root, source_dir),
    )


def line_from_dict(data: Mapping[str, object]) -> BindingsLine:
    """Validate a normalized line record passed between release jobs."""
    line_id = _text(data.get("line_id"), "resolved line_id", _NAME_PATTERN)
    source_dir = _source_dir(data.get("source_dir"), "resolved source_dir")
    toolkit_version = _text(
        data.get("toolkit_version"),
        "resolved toolkit_version",
        _TOOLKIT_VERSION_PATTERN,
    )
    tag_regex = data.get("tag_regex")
    if not isinstance(tag_regex, str) or not tag_regex:
        raise BindingsConfigError("resolved tag_regex must be a non-empty string")
    _compile_tag_regex(tag_regex, "resolved tag_regex")
    line = BindingsLine(line_id, source_dir, toolkit_version, tag_regex)
    expected = line.to_dict()
    for key in ("ctk_target", "cuda_major", "cuda_variant"):
        if key in data and data[key] != expected[key]:
            raise BindingsConfigError(f"resolved {key} is inconsistent with toolkit_version")
    return line


def _roles(raw: Any, line_ids: set[str]) -> Mapping[str, str]:
    roles: dict[str, str] = {}
    for role, value in _mapping(raw, "cuda.bindings.roles").items():
        _text(role, "CUDA bindings role", _NAME_PATTERN)
        line_id = _text(value, f"cuda.bindings.roles.{role}", _NAME_PATTERN)
        if line_id not in line_ids:
            raise BindingsConfigError(f"cuda.bindings.roles.{role} references unknown line: {line_id}")
        roles[role] = line_id
    if set(roles) != {"current", "maintenance"}:
        raise BindingsConfigError("cuda.bindings.roles must contain exactly current and maintenance")
    selected_line_ids = tuple(roles.values())
    if len(line_ids) != 2 or len(set(selected_line_ids)) != 2 or set(selected_line_ids) != line_ids:
        raise BindingsConfigError("current and maintenance must select each configured line exactly once")
    return MappingProxyType(roles)


def _validate_scm_conformance(lines: tuple[BindingsLine, ...]) -> None:
    for line in lines:
        tag = f"v{line.toolkit_version}"
        version = line.version_from_tag(tag)
        expected = parse_pep440_version(line.toolkit_version, f"{line.line_id}.toolkit_version")
        if version != expected:
            raise BindingsConfigError(
                f"{line.source_dir} setuptools-scm tag_regex must match its configured toolkit release tag {tag!r}"
            )


def validate_config(raw: Any, repo_root: Path = REPO_ROOT) -> BindingsConfig:
    root = _mapping(raw, "versions configuration", {"schema_version", "cuda"})
    if type(root["schema_version"]) is not int or root["schema_version"] != SCHEMA_VERSION:
        raise BindingsConfigError(f"schema_version must be {SCHEMA_VERSION}")
    cuda = _mapping(root["cuda"], "cuda", {"bindings"})
    bindings = _mapping(cuda["bindings"], "cuda.bindings", {"lines", "roles"})
    raw_lines = _mapping(bindings["lines"], "cuda.bindings.lines")
    if not raw_lines:
        raise BindingsConfigError("cuda.bindings.lines must not be empty")
    lines = tuple(_line(line_id, value, repo_root) for line_id, value in raw_lines.items())
    for attribute in ("source_dir", "toolkit_version", "ctk_target", "cuda_major"):
        values = [getattr(line, attribute) for line in lines]
        if len(set(values)) != len(values):
            raise BindingsConfigError(f"CUDA bindings {attribute} values must be unique")
    _validate_scm_conformance(lines)
    return BindingsConfig(
        schema_version=SCHEMA_VERSION,
        lines=lines,
        roles=_roles(bindings["roles"], {line.line_id for line in lines}),
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
        line.toolkit_version
        for line in control.lines
        if parse_pep440_version(line.toolkit_version).release[:2] == target
    ]
    if len(matches) != 1:
        raise BindingsConfigError(
            f"control registry must contain exactly one toolkit pin for legacy CUDA {target[0]}.{target[1]}; "
            f"found {len(matches)}"
        )
    return matches[0]


def _legacy_release_line(
    release_tag: str,
    release_source_root: Path,
    control_config_path: Path,
    raw: Any,
) -> dict[str, object]:
    """Resolve a release tree from before the schema-2 registry existed."""
    source_dir = "cuda_bindings"
    if not (release_source_root / source_dir).is_dir():
        raise BindingsConfigError(f"legacy release source directory is missing: {source_dir}")

    tag_regex = _read_tag_regex(release_source_root, source_dir)

    probe = BindingsLine("legacy", source_dir, "1.0.0", tag_regex)
    release_version = probe.scm_version_from_tag(release_tag, fullmatch=False)
    if release_version is None:
        raise BindingsConfigError(f"legacy source SCM metadata does not match release tag: {release_tag!r}")
    line = BindingsLine(
        "legacy",
        source_dir,
        _legacy_toolkit_version(raw, release_version, control_config_path),
        tag_regex,
    )
    normalized = line.to_dict()
    normalized.update(
        role=None,
        release_version=str(release_version),
        release_source_dir=source_dir,
        release_registry_origin="control",
    )
    return normalized


def resolve_release_bindings_line(
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
        return _legacy_release_line(release_tag, release_source_root, control_config_path, raw)

    line = config.match_tag(release_tag)
    if line is None:
        raise BindingsConfigError(f"no CUDA bindings line in {config_source} matches release tag: {release_tag!r}")

    normalized = config.line_to_dict(line)
    version = line.version_from_tag(release_tag)
    assert version is not None
    normalized["release_version"] = str(version)
    normalized["release_source_dir"] = line.source_dir
    normalized["release_registry_origin"] = "tag"
    return normalized


def write_github_env(data: Mapping[str, object], path: Path) -> None:
    """Append the bindings build environment consumed by documentation jobs."""
    line = line_from_dict(data)
    source_dir = _source_dir(data.get("release_source_dir", line.source_dir), "release source_dir")
    origin = data.get("release_registry_origin", "tag")
    if origin not in {"tag", "control"}:
        raise BindingsConfigError("release_registry_origin must be tag or control")
    with path.open("a", encoding="utf-8") as stream:
        stream.write(f"BUILD_CTK_VER={line.toolkit_version}\n")
        stream.write(f"BINDINGS_COMPONENT_DIR={source_dir}\n")
        stream.write(f"BINDINGS_REGISTRY_ORIGIN={origin}\n")


def main(argv: list[str] | None = None) -> int:
    """Emit normalized registry or release-line JSON for CI consumers."""
    parser = ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--repo-root", type=Path, default=REPO_ROOT)
    output = parser.add_mutually_exclusive_group()
    output.add_argument("--lines", action="store_true", help="print normalized line records")
    output.add_argument("--role", help="print the normalized line for this role")
    output.add_argument("--release-tag", help="resolve a release tag against its source tree")
    output.add_argument("--line-json", help="consume an already normalized line record")
    parser.add_argument("--release-source-root", type=Path)
    parser.add_argument("--control-config", type=Path)
    parser.add_argument("--github-env", type=Path, help="append one selected line as GitHub environment variables")
    args = parser.parse_args(argv)

    try:
        if args.release_tag:
            if args.release_source_root is None or args.control_config is None:
                parser.error("--release-tag requires --release-source-root and --control-config")
            value: object = resolve_release_bindings_line(
                args.release_tag,
                args.release_source_root,
                args.control_config,
            )
        elif args.line_json:
            value = json.loads(args.line_json)
            if not isinstance(value, dict):
                raise BindingsConfigError("--line-json must contain a JSON object")
        else:
            if args.release_source_root is not None or args.control_config is not None:
                parser.error("--release-source-root and --control-config require --release-tag")
            config = load_config(args.config, args.repo_root)
            if args.lines:
                value = [config.line_to_dict(line) for line in config.lines]
            elif args.role:
                value = config.line_to_dict(config.line_for_role(args.role))
            else:
                value = config.to_dict()
        if args.github_env is not None:
            if not isinstance(value, dict) or (not args.role and not args.release_tag and not args.line_json):
                parser.error("--github-env requires --role, --release-tag, or --line-json")
            write_github_env(value, args.github_env)
        else:
            print(json.dumps(value, separators=(",", ":"), sort_keys=True))
        return 0
    except (BindingsConfigError, json.JSONDecodeError) as error:
        parser.error(str(error))


if __name__ == "__main__":
    sys.exit(main())
