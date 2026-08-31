# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Load and validate the CUDA bindings release-line registry."""

from __future__ import annotations

import json
import re
import sys
from argparse import ArgumentParser
from dataclasses import dataclass
from pathlib import Path, PurePosixPath, PureWindowsPath
from types import MappingProxyType
from typing import Any, Mapping

import yaml

REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CONFIG = REPO_ROOT / "ci" / "versions.yml"
SCHEMA_VERSION = 2

_LINE_ID_PATTERN = re.compile(r"[a-z][a-z0-9]*(?:-[a-z0-9]+)*")
_SOURCE_DIR_PATTERN = re.compile(r"[A-Za-z0-9._-]+(?:/[A-Za-z0-9._-]+)*")
_CTK_TARGET_PATTERN = re.compile(r"[1-9][0-9]*\.[0-9]+")
_TOOLKIT_VERSION_PATTERN = re.compile(r"[1-9][0-9]*\.[0-9]+\.[0-9]+(?:[.-][A-Za-z0-9]+)*")
_TAG_SERIES_PATTERN = re.compile(r"v[1-9][0-9]*(?:\.[0-9]+)*\.")
_FINAL_TAG_SUFFIX_PATTERN = re.compile(r"[0-9]+(?:\.post[0-9]+)?")
_ALPHA_BETA_TAG_SUFFIX_PATTERN = re.compile(r"[0-9]+(?:[ab][0-9]+)?(?:\.post[0-9]+)?")
_TOOLKIT_CHANNELS = frozenset({"prerelease", "stable"})


class BindingsConfigError(ValueError):
    """The CUDA bindings release-line registry is invalid."""


@dataclass(frozen=True)
class BindingsLine:
    """One versioned CUDA bindings source line."""

    line_id: str
    source_dir: str
    ctk_target: str
    toolkit_version: str
    toolkit_channel: str
    tag_series: str
    allow_alpha_beta_tags: bool

    @property
    def cuda_major(self) -> str:
        """Return the CUDA ABI/package major shared by compatible lines."""
        return self.ctk_target.partition(".")[0]

    @property
    def cuda_variant(self) -> str:
        """Return the conventional CUDA ABI variant name."""
        return f"cu{self.cuda_major}"

    def matches_tag(self, tag: str) -> bool:
        """Return whether *tag* is a valid release in this line's tag family."""
        if not tag.startswith(self.tag_series):
            return False
        pattern = _ALPHA_BETA_TAG_SUFFIX_PATTERN if self.allow_alpha_beta_tags else _FINAL_TAG_SUFFIX_PATTERN
        return pattern.fullmatch(tag.removeprefix(self.tag_series)) is not None

    def to_dict(self) -> dict[str, object]:
        """Return the line in the normalized form consumed by CI."""
        return {
            "line_id": self.line_id,
            "source_dir": self.source_dir,
            "ctk_target": self.ctk_target,
            "toolkit_version": self.toolkit_version,
            "toolkit_channel": self.toolkit_channel,
            "tag_series": self.tag_series,
            "allow_alpha_beta_tags": self.allow_alpha_beta_tags,
            "cuda_major": self.cuda_major,
            "cuda_variant": self.cuda_variant,
        }


@dataclass(frozen=True)
class BindingsConfig:
    """Validated CUDA bindings lines and their orchestration roles."""

    schema_version: int
    lines: tuple[BindingsLine, ...]
    roles: Mapping[str, tuple[str, ...]]

    def get_line(self, line_id: str) -> BindingsLine:
        """Return a line by its stable ID."""
        for line in self.lines:
            if line.line_id == line_id:
                return line
        raise BindingsConfigError(f"unknown CUDA bindings line: {line_id!r}")

    @property
    def public_lines(self) -> tuple[BindingsLine, ...]:
        """Return public lines in their configured order."""
        public_ids = set(self.roles["current"]) | set(self.roles["maintenance"])
        return tuple(line for line in self.lines if line.line_id in public_ids)

    def lines_for_role(self, role: str) -> tuple[BindingsLine, ...]:
        """Return the ordered lines assigned to a role."""
        try:
            line_ids = self.roles[role]
        except KeyError as error:
            raise BindingsConfigError(f"unknown CUDA bindings role: {role!r}") from error
        return tuple(self.get_line(line_id) for line_id in line_ids)

    def line_for_role(self, role: str) -> BindingsLine:
        """Return the sole line assigned to a singular role."""
        lines = self.lines_for_role(role)
        if len(lines) != 1:
            raise BindingsConfigError(f"CUDA bindings role {role!r} must resolve to exactly one line")
        return lines[0]

    def match_tag(self, tag: str) -> BindingsLine | None:
        """Return the release line matching an exact release tag."""
        return next((line for line in self.lines if line.matches_tag(tag)), None)

    def line_to_dict(self, line: BindingsLine) -> dict[str, object]:
        """Return one line with its normalized role membership."""
        normalized: dict[str, object] = line.to_dict()
        normalized["roles"] = [role for role, line_ids in self.roles.items() if line.line_id in line_ids]
        return normalized

    def to_dict(self) -> dict[str, object]:
        """Return a stable, matrix-friendly representation for CI consumers."""
        return {
            "schema_version": self.schema_version,
            "lines": [self.line_to_dict(line) for line in self.lines],
            "roles": {role: list(line_ids) for role, line_ids in self.roles.items()},
        }

    def to_json(self) -> str:
        """Serialize the normalized representation as compact deterministic JSON."""
        return json.dumps(self.to_dict(), separators=(",", ":"), sort_keys=True)


def _mapping(value: Any, label: str) -> Mapping[str, Any]:
    if not isinstance(value, dict) or not all(isinstance(key, str) for key in value):
        raise BindingsConfigError(f"{label} must be a mapping with string keys")
    return value


def _check_keys(value: Mapping[str, Any], label: str, required: set[str]) -> None:
    missing = sorted(required - set(value))
    if missing:
        raise BindingsConfigError(f"{label} is missing required keys: {', '.join(missing)}")
    unexpected = sorted(set(value) - required)
    if unexpected:
        raise BindingsConfigError(f"{label} has unexpected keys: {', '.join(unexpected)}")


def _string(value: Any, label: str, pattern: re.Pattern[str] | None = None) -> str:
    if not isinstance(value, str) or not value or value != value.strip():
        raise BindingsConfigError(f"{label} must be a non-empty, trimmed string")
    if pattern is not None and pattern.fullmatch(value) is None:
        raise BindingsConfigError(f"{label} has invalid format: {value!r}")
    return value


def _boolean(value: Any, label: str) -> bool:
    if type(value) is not bool:
        raise BindingsConfigError(f"{label} must be a boolean")
    return value


def _source_dir(value: Any, label: str) -> str:
    source_dir = _string(value, label, _SOURCE_DIR_PATTERN)
    if "\\" in source_dir or PureWindowsPath(source_dir).drive:
        raise BindingsConfigError(f"{label} must be a normalized repository-relative POSIX path: {source_dir!r}")
    path = PurePosixPath(source_dir)
    if path.is_absolute() or path.as_posix() != source_dir or any(part in (".", "..") for part in path.parts):
        raise BindingsConfigError(f"{label} must be a normalized repository-relative POSIX path: {source_dir!r}")
    return source_dir


def _line(line_id: str, raw: Any) -> BindingsLine:
    _string(line_id, "CUDA bindings line ID", _LINE_ID_PATTERN)
    data = _mapping(raw, f"CUDA bindings line {line_id!r}")
    _check_keys(
        data,
        f"CUDA bindings line {line_id!r}",
        {
            "source_dir",
            "ctk_target",
            "toolkit_version",
            "toolkit_channel",
            "tag_series",
            "allow_alpha_beta_tags",
        },
    )
    ctk_target = _string(data["ctk_target"], f"{line_id}.ctk_target", _CTK_TARGET_PATTERN)
    toolkit_version = _string(data["toolkit_version"], f"{line_id}.toolkit_version", _TOOLKIT_VERSION_PATTERN)
    if not toolkit_version.startswith(f"{ctk_target}."):
        raise BindingsConfigError(
            f"{line_id}.toolkit_version must belong to CTK target {ctk_target!r}: {toolkit_version!r}"
        )
    tag_series = _string(data["tag_series"], f"{line_id}.tag_series", _TAG_SERIES_PATTERN)
    expected_tag_series = f"v{ctk_target}."
    if tag_series != expected_tag_series:
        raise BindingsConfigError(f"{line_id}.tag_series must be {expected_tag_series!r} for CTK target {ctk_target!r}")
    toolkit_channel = _string(data["toolkit_channel"], f"{line_id}.toolkit_channel")
    if toolkit_channel not in _TOOLKIT_CHANNELS:
        raise BindingsConfigError(f"{line_id}.toolkit_channel must be one of: {', '.join(sorted(_TOOLKIT_CHANNELS))}")
    return BindingsLine(
        line_id=line_id,
        source_dir=_source_dir(data["source_dir"], f"{line_id}.source_dir"),
        ctk_target=ctk_target,
        toolkit_version=toolkit_version,
        toolkit_channel=toolkit_channel,
        tag_series=tag_series,
        allow_alpha_beta_tags=_boolean(data["allow_alpha_beta_tags"], f"{line_id}.allow_alpha_beta_tags"),
    )


def _roles(raw: Any, line_ids: set[str]) -> Mapping[str, tuple[str, ...]]:
    data = _mapping(raw, "cuda.bindings.roles")
    _check_keys(data, "cuda.bindings.roles", {"current", "maintenance", "unreleased"})

    current = _string(data["current"], "cuda.bindings.roles.current", _LINE_ID_PATTERN)
    lists: dict[str, tuple[str, ...]] = {}
    for role in ("maintenance", "unreleased"):
        values = data[role]
        if not isinstance(values, list):
            raise BindingsConfigError(f"cuda.bindings.roles.{role} must be a list")
        line_list = tuple(_string(value, f"cuda.bindings.roles.{role} entry", _LINE_ID_PATTERN) for value in values)
        if len(set(line_list)) != len(line_list):
            raise BindingsConfigError(f"cuda.bindings.roles.{role} must not contain duplicates")
        lists[role] = line_list

    role_sets = {"current": {current}, **{role: set(values) for role, values in lists.items()}}
    overlaps = {
        line_id
        for role, members in role_sets.items()
        for other_role, other_members in role_sets.items()
        if role < other_role
        for line_id in members & other_members
    }
    if overlaps:
        raise BindingsConfigError(f"CUDA bindings roles must not overlap: {', '.join(sorted(overlaps))}")

    referenced = set().union(*role_sets.values())
    unknown = sorted(referenced - line_ids)
    if unknown:
        raise BindingsConfigError(f"CUDA bindings roles reference unknown lines: {', '.join(unknown)}")
    unassigned = sorted(line_ids - referenced)
    if unassigned:
        raise BindingsConfigError(f"public CUDA bindings lines are missing a role: {', '.join(unassigned)}")
    return MappingProxyType(
        {"current": (current,), "maintenance": lists["maintenance"], "unreleased": lists["unreleased"]}
    )


def validate_config(raw: Any) -> BindingsConfig:
    """Validate parsed ``ci/versions.yml`` data and return the bindings registry."""
    root = _mapping(raw, "versions configuration")
    _check_keys(root, "versions configuration", {"schema_version", "cuda"})
    schema_version = root["schema_version"]
    if type(schema_version) is not int or schema_version != SCHEMA_VERSION:
        raise BindingsConfigError(f"schema_version must be {SCHEMA_VERSION}")

    cuda = _mapping(root["cuda"], "cuda")
    _check_keys(cuda, "cuda", {"bindings"})
    bindings = _mapping(cuda["bindings"], "cuda.bindings")
    _check_keys(bindings, "cuda.bindings", {"lines", "roles"})
    raw_lines = _mapping(bindings["lines"], "cuda.bindings.lines")
    if not raw_lines:
        raise BindingsConfigError("cuda.bindings.lines must not be empty")

    lines = tuple(_line(line_id, value) for line_id, value in raw_lines.items())
    source_dirs = [line.source_dir for line in lines]
    if len(set(source_dirs)) != len(source_dirs):
        raise BindingsConfigError("CUDA bindings source_dir values must be unique")
    ctk_targets = [line.ctk_target for line in lines]
    if len(set(ctk_targets)) != len(ctk_targets):
        raise BindingsConfigError("CUDA bindings ctk_target values must be unique")
    tag_series = [line.tag_series for line in lines]
    if len(set(tag_series)) != len(tag_series):
        raise BindingsConfigError("CUDA bindings tag_series values must be unique")

    roles = _roles(bindings["roles"], {line.line_id for line in lines})
    return BindingsConfig(schema_version=schema_version, lines=lines, roles=roles)


def load_config(path: Path = DEFAULT_CONFIG) -> BindingsConfig:
    """Read and validate a CUDA bindings release-line registry."""
    try:
        raw = yaml.safe_load(path.read_text(encoding="utf-8"))
    except (OSError, yaml.YAMLError) as error:
        raise BindingsConfigError(f"could not read {path}: {error}") from error
    return validate_config(raw)


def _json(value: object) -> str:
    return json.dumps(value, separators=(",", ":"), sort_keys=True)


def main(argv: list[str] | None = None) -> int:
    """Resolve the registry for shell and GitHub Actions consumers."""
    parser = ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    commands = parser.add_subparsers(dest="command", required=True)
    commands.add_parser("validate", help="validate the registry")
    commands.add_parser("json", help="print the full normalized registry")

    list_parser = commands.add_parser("list", help="print normalized line records")
    list_parser.add_argument("--scope", choices=("all", "public"), default="public")

    get_parser = commands.add_parser("get", help="print one normalized line record")
    selector = get_parser.add_mutually_exclusive_group(required=True)
    selector.add_argument("--line")
    selector.add_argument("--role")

    match_parser = commands.add_parser("match-tag", help="print the line selected by a release tag")
    match_parser.add_argument("tag")

    args = parser.parse_args(argv)
    try:
        config = load_config(args.config)
        if args.command == "validate":
            return 0
        if args.command == "json":
            print(config.to_json())
            return 0
        if args.command == "list":
            lines = config.lines if args.scope == "all" else config.public_lines
            print(_json([config.line_to_dict(line) for line in lines]))
            return 0
        if args.command == "get":
            line = config.get_line(args.line) if args.line else config.line_for_role(args.role)
            print(_json(config.line_to_dict(line)))
            return 0
        if args.command == "match-tag":
            line = config.match_tag(args.tag)
            if line is None:
                raise BindingsConfigError(f"no CUDA bindings line matches release tag: {args.tag!r}")
            print(_json(config.line_to_dict(line)))
            return 0
        raise AssertionError(f"unhandled command: {args.command}")
    except BindingsConfigError as error:
        parser.error(str(error))


if __name__ == "__main__":
    sys.exit(main())
