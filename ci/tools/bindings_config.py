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

import yaml

REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CONFIG = REPO_ROOT / "ci" / "versions.yml"
SCHEMA_VERSION = 2

_NAME_PATTERN = re.compile(r"[a-z][a-z0-9]*(?:-[a-z0-9]+)*")
_SOURCE_DIR_PATTERN = re.compile(r"[A-Za-z0-9._-]+(?:/[A-Za-z0-9._-]+)*")
_TOOLKIT_VERSION_PATTERN = re.compile(r"[1-9][0-9]*\.[0-9]+\.[0-9]+(?:[.-][A-Za-z0-9]+)*")
_FINAL_TAG_SUFFIX_PATTERN = re.compile(r"[0-9]+(?:\.post[0-9]+)?")
_ALPHA_BETA_TAG_SUFFIX_PATTERN = re.compile(r"[0-9]+(?:[ab][0-9]+)?(?:\.post[0-9]+)?")


class BindingsConfigError(ValueError):
    """The CUDA bindings release-line registry is invalid."""


@dataclass(frozen=True)
class BindingsLine:
    line_id: str
    source_dir: str
    toolkit_version: str
    allow_alpha_beta_tags: bool

    @property
    def ctk_target(self) -> str:
        major, minor, _ = self.toolkit_version.split(".", maxsplit=2)
        return f"{major}.{minor}"

    @property
    def tag_series(self) -> str:
        return f"v{self.ctk_target}."

    @property
    def cuda_major(self) -> str:
        return self.ctk_target.partition(".")[0]

    @property
    def cuda_variant(self) -> str:
        return f"cu{self.cuda_major}"

    def matches_tag(self, tag: str) -> bool:
        if not tag.startswith(self.tag_series):
            return False
        pattern = _ALPHA_BETA_TAG_SUFFIX_PATTERN if self.allow_alpha_beta_tags else _FINAL_TAG_SUFFIX_PATTERN
        return pattern.fullmatch(tag.removeprefix(self.tag_series)) is not None

    def to_dict(self) -> dict[str, object]:
        return {
            "line_id": self.line_id,
            "source_dir": self.source_dir,
            "toolkit_version": self.toolkit_version,
            "allow_alpha_beta_tags": self.allow_alpha_beta_tags,
            "ctk_target": self.ctk_target,
            "tag_series": self.tag_series,
            "cuda_major": self.cuda_major,
            "cuda_variant": self.cuda_variant,
        }


@dataclass(frozen=True)
class BindingsConfig:
    schema_version: int
    lines: tuple[BindingsLine, ...]
    roles: Mapping[str, tuple[str, ...]]

    def get_line(self, line_id: str) -> BindingsLine:
        line = next((line for line in self.lines if line.line_id == line_id), None)
        if line is None:
            raise BindingsConfigError(f"unknown CUDA bindings line: {line_id!r}")
        return line

    def lines_for_role(self, role: str) -> tuple[BindingsLine, ...]:
        try:
            return tuple(self.get_line(line_id) for line_id in self.roles[role])
        except KeyError as error:
            raise BindingsConfigError(f"unknown CUDA bindings role: {role!r}") from error

    def line_for_role(self, role: str) -> BindingsLine:
        lines = self.lines_for_role(role)
        if len(lines) != 1:
            raise BindingsConfigError(f"CUDA bindings role {role!r} must resolve to exactly one line")
        return lines[0]

    def match_tag(self, tag: str) -> BindingsLine | None:
        return next((line for line in self.lines if line.matches_tag(tag)), None)

    def line_to_dict(self, line: BindingsLine) -> dict[str, object]:
        normalized: dict[str, object] = line.to_dict()
        normalized["roles"] = [role for role, line_ids in self.roles.items() if line.line_id in line_ids]
        return normalized

    def to_dict(self) -> dict[str, object]:
        return {
            "schema_version": self.schema_version,
            "lines": [self.line_to_dict(line) for line in self.lines],
            "roles": {role: list(line_ids) for role, line_ids in self.roles.items()},
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


def _line(line_id: str, raw: Any) -> BindingsLine:
    _text(line_id, "CUDA bindings line ID", _NAME_PATTERN)
    data = _mapping(
        raw,
        f"CUDA bindings line {line_id!r}",
        {"source_dir", "toolkit_version", "allow_alpha_beta_tags"},
    )
    alpha_beta = data["allow_alpha_beta_tags"]
    if type(alpha_beta) is not bool:
        raise BindingsConfigError(f"{line_id}.allow_alpha_beta_tags must be a boolean")
    return BindingsLine(
        line_id,
        _source_dir(data["source_dir"], f"{line_id}.source_dir"),
        _text(data["toolkit_version"], f"{line_id}.toolkit_version", _TOOLKIT_VERSION_PATTERN),
        alpha_beta,
    )


def _roles(raw: Any, line_ids: set[str]) -> Mapping[str, tuple[str, ...]]:
    roles: dict[str, tuple[str, ...]] = {}
    for role, value in _mapping(raw, "cuda.bindings.roles").items():
        _text(role, "CUDA bindings role", _NAME_PATTERN)
        raw_members = [value] if isinstance(value, str) else value
        if not isinstance(raw_members, list):
            raise BindingsConfigError(f"cuda.bindings.roles.{role} must be a line ID or list of line IDs")
        members = tuple(_text(member, f"cuda.bindings.roles.{role} entry", _NAME_PATTERN) for member in raw_members)
        if len(set(members)) != len(members):
            raise BindingsConfigError(f"cuda.bindings.roles.{role} must not contain duplicates")
        if unknown := sorted(set(members) - line_ids):
            raise BindingsConfigError(f"cuda.bindings.roles.{role} references unknown lines: {', '.join(unknown)}")
        roles[role] = members
    return MappingProxyType(roles)


def validate_config(raw: Any) -> BindingsConfig:
    root = _mapping(raw, "versions configuration", {"schema_version", "cuda"})
    if type(root["schema_version"]) is not int or root["schema_version"] != SCHEMA_VERSION:
        raise BindingsConfigError(f"schema_version must be {SCHEMA_VERSION}")
    cuda = _mapping(root["cuda"], "cuda", {"bindings"})
    bindings = _mapping(cuda["bindings"], "cuda.bindings", {"lines", "roles"})
    raw_lines = _mapping(bindings["lines"], "cuda.bindings.lines")
    if not raw_lines:
        raise BindingsConfigError("cuda.bindings.lines must not be empty")
    lines = tuple(_line(line_id, value) for line_id, value in raw_lines.items())
    for attribute in ("source_dir", "toolkit_version", "ctk_target"):
        values = [getattr(line, attribute) for line in lines]
        if len(set(values)) != len(values):
            raise BindingsConfigError(f"CUDA bindings {attribute} values must be unique")
    return BindingsConfig(
        schema_version=SCHEMA_VERSION,
        lines=lines,
        roles=_roles(bindings["roles"], {line.line_id for line in lines}),
    )


def load_config(path: Path = DEFAULT_CONFIG) -> BindingsConfig:
    try:
        return validate_config(yaml.safe_load(path.read_text(encoding="utf-8")))
    except (OSError, yaml.YAMLError) as error:
        raise BindingsConfigError(f"could not read {path}: {error}") from error


def main(argv: list[str] | None = None) -> int:
    """Emit normalized registry JSON for shell and GitHub Actions consumers."""
    parser = ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    output = parser.add_mutually_exclusive_group()
    output.add_argument("--lines", action="store_true", help="print normalized line records")
    output.add_argument("--role", help="print the sole normalized line for this role")
    args = parser.parse_args(argv)

    try:
        config = load_config(args.config)
        if args.lines:
            value: object = [config.line_to_dict(line) for line in config.lines]
        elif args.role:
            value = config.line_to_dict(config.line_for_role(args.role))
        else:
            value = config.to_dict()
        print(json.dumps(value, separators=(",", ":"), sort_keys=True))
        return 0
    except BindingsConfigError as error:
        parser.error(str(error))


if __name__ == "__main__":
    sys.exit(main())
