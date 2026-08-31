# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Check that selected CUDA bindings lines have exact-pin test-matrix rows."""

from __future__ import annotations

import argparse
import json
import re
import sys
from typing import Any

_CUDA_VARIANT_PATTERN = re.compile(r"cu[1-9][0-9]*")
_TOOLKIT_VERSION_PATTERN = re.compile(r"[1-9][0-9]*\.[0-9]+\.[0-9]+(?:[.-][A-Za-z0-9]+)*")
_PUBLIC_ROLES = frozenset({"current", "maintenance"})


class MatrixBindingsError(ValueError):
    """The bindings registry, enabled variants, or test matrix is invalid."""


def _json_value(text: str, label: str) -> Any:
    try:
        return json.loads(text)
    except json.JSONDecodeError as error:
        raise MatrixBindingsError(f"{label} is not valid JSON: {error.msg}") from error


def _public_lines(bindings_config: Any) -> list[tuple[str, str, str]]:
    if not isinstance(bindings_config, dict):
        raise MatrixBindingsError("bindings config must be a JSON object")
    if bindings_config.get("schema_version") != 2:
        raise MatrixBindingsError("bindings config schema_version must be 2")

    lines = bindings_config.get("lines")
    if not isinstance(lines, list):
        raise MatrixBindingsError("bindings config lines must be a JSON list")

    public_lines: list[tuple[str, str, str]] = []
    seen_line_ids: set[str] = set()
    seen_toolkit_versions: set[str] = set()
    for index, line in enumerate(lines):
        label = f"bindings config line {index}"
        if not isinstance(line, dict):
            raise MatrixBindingsError(f"{label} must be a JSON object")

        roles = line.get("roles")
        if not isinstance(roles, list) or not all(isinstance(role, str) for role in roles):
            raise MatrixBindingsError(f"{label} roles must be a JSON list of strings")
        if not _PUBLIC_ROLES.intersection(roles):
            continue

        line_id = line.get("line_id")
        toolkit_version = line.get("toolkit_version")
        cuda_variant = line.get("cuda_variant")
        if not isinstance(line_id, str) or not line_id or line_id != line_id.strip():
            raise MatrixBindingsError(f"{label} line_id must be a non-empty, trimmed string")
        if not isinstance(toolkit_version, str) or _TOOLKIT_VERSION_PATTERN.fullmatch(toolkit_version) is None:
            raise MatrixBindingsError(f"bindings line {line_id!r} has invalid toolkit_version")
        if not isinstance(cuda_variant, str) or _CUDA_VARIANT_PATTERN.fullmatch(cuda_variant) is None:
            raise MatrixBindingsError(f"bindings line {line_id!r} has invalid cuda_variant")
        toolkit_major = toolkit_version.partition(".")[0]
        if cuda_variant != f"cu{toolkit_major}":
            raise MatrixBindingsError(
                f"bindings line {line_id!r} cuda_variant {cuda_variant!r} does not match "
                f"toolkit_version {toolkit_version!r}"
            )
        if line_id in seen_line_ids:
            raise MatrixBindingsError(f"duplicate public bindings line_id: {line_id!r}")
        if toolkit_version in seen_toolkit_versions:
            raise MatrixBindingsError(f"duplicate public bindings toolkit_version: {toolkit_version!r}")
        seen_line_ids.add(line_id)
        seen_toolkit_versions.add(toolkit_version)
        public_lines.append((line_id, cuda_variant, toolkit_version))

    if not public_lines:
        raise MatrixBindingsError("bindings config has no public current or maintenance lines")
    return public_lines


def check_test_matrix_bindings(
    bindings_config: Any,
    enabled_cuda_variants: Any,
    test_matrix: Any,
) -> None:
    """Raise if an enabled public bindings line has no exact toolkit-pin row."""
    public_lines = _public_lines(bindings_config)
    public_variants = {cuda_variant for _, cuda_variant, _ in public_lines}

    if not isinstance(enabled_cuda_variants, dict) or not all(
        isinstance(cuda_variant, str) and type(enabled) is bool
        for cuda_variant, enabled in enabled_cuda_variants.items()
    ):
        raise MatrixBindingsError("enabled CUDA variants must be a JSON object of boolean values")
    unknown_enabled = sorted(
        cuda_variant
        for cuda_variant, enabled in enabled_cuda_variants.items()
        if enabled and cuda_variant not in public_variants
    )
    if unknown_enabled:
        raise MatrixBindingsError(
            "enabled CUDA variants are absent from the public bindings registry: " + ", ".join(unknown_enabled)
        )

    if not isinstance(test_matrix, list):
        raise MatrixBindingsError("test matrix must be a JSON list")
    matrix_versions: set[str] = set()
    for index, row in enumerate(test_matrix):
        if not isinstance(row, dict):
            raise MatrixBindingsError(f"test matrix row {index} must be a JSON object")
        cuda_version = row.get("CUDA_VER")
        if not isinstance(cuda_version, str) or _TOOLKIT_VERSION_PATTERN.fullmatch(cuda_version) is None:
            raise MatrixBindingsError(f"test matrix row {index} has invalid CUDA_VER")
        matrix_versions.add(cuda_version)

    missing = [
        (line_id, cuda_variant, toolkit_version)
        for line_id, cuda_variant, toolkit_version in public_lines
        if enabled_cuda_variants.get(cuda_variant) is True and toolkit_version not in matrix_versions
    ]
    if missing:
        detail = ", ".join(
            f"{line_id} ({cuda_variant}, CUDA {toolkit_version})" for line_id, cuda_variant, toolkit_version in missing
        )
        raise MatrixBindingsError(f"test matrix has no exact toolkit-pin row for: {detail}")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bindings-config", required=True, help="normalized bindings registry JSON")
    parser.add_argument("--enabled-cuda-variants", required=True, help="JSON object of cuN boolean gates")
    parser.add_argument("--test-matrix", required=True, help="pre-filter test-matrix JSON list")
    args = parser.parse_args(argv)

    try:
        check_test_matrix_bindings(
            _json_value(args.bindings_config, "bindings config"),
            _json_value(args.enabled_cuda_variants, "enabled CUDA variants"),
            _json_value(args.test_matrix, "test matrix"),
        )
    except MatrixBindingsError as error:
        print(f"error: {error}", file=sys.stderr)
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
