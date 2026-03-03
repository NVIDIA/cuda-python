#!/usr/bin/env python3

# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Shared helper for reading, updating, and rewriting descriptor_catalog.py.

Each toolshed script that extracts data from CTK distributions or wheel
layouts uses this module to merge its findings into the authored catalog
without touching fields it doesn't own.
"""

from __future__ import annotations

import dataclasses
import json
import sys
from pathlib import Path

# Ensure the cuda_pathfinder package is importable.
_REPO_ROOT = Path(__file__).resolve().parents[1]
_PATHFINDER_ROOT = _REPO_ROOT / "cuda_pathfinder"
if str(_PATHFINDER_ROOT) not in sys.path:
    sys.path.insert(0, str(_PATHFINDER_ROOT))

from cuda.pathfinder._dynamic_libs.descriptor_catalog import (  # noqa: E402
    DESCRIPTOR_CATALOG,
    DescriptorSpec,
)

CATALOG_PATH = _PATHFINDER_ROOT / "cuda" / "pathfinder" / "_dynamic_libs" / "descriptor_catalog.py"

_DEFAULTS = DescriptorSpec(name="", packaged_with="ctk")

_SECTION_COMMENTS = {
    "ctk": (
        "    # -----------------------------------------------------------------------\n"
        "    # CTK (CUDA Toolkit) libraries\n"
        "    # -----------------------------------------------------------------------"
    ),
    "other": (
        "    # -----------------------------------------------------------------------\n"
        "    # Third-party / separately packaged libraries\n"
        "    # -----------------------------------------------------------------------"
    ),
    "driver": (
        "    # -----------------------------------------------------------------------\n"
        "    # Driver libraries (system-search only, no CTK cascade)\n"
        "    # -----------------------------------------------------------------------"
    ),
}


def _quote(s: str) -> str:
    return json.dumps(s)


def _render_tuple(values: tuple[str, ...]) -> str:
    if not values:
        return "()"
    if len(values) == 1:
        return f"({_quote(values[0])},)"
    return "(" + ", ".join(_quote(v) for v in values) + ")"


def _render_spec(spec: DescriptorSpec) -> str:
    """Render a single DescriptorSpec constructor call, omitting default-valued fields."""
    lines = [
        "    DescriptorSpec(",
        f"        name={_quote(spec.name)},",
        f'        packaged_with="{spec.packaged_with}",',
    ]

    tuple_fields = [
        "linux_sonames",
        "windows_dlls",
        "site_packages_linux",
        "site_packages_windows",
        "dependencies",
        "anchor_rel_dirs_linux",
        "anchor_rel_dirs_windows",
        "ctk_root_canary_anchor_libnames",
    ]
    bool_fields = [
        "requires_add_dll_directory",
        "requires_rtld_deepbind",
    ]

    for field in tuple_fields:
        value = getattr(spec, field)
        default = getattr(_DEFAULTS, field)
        if value != default:
            lines.append(f"        {field}={_render_tuple(value)},")

    for field in bool_fields:
        value = getattr(spec, field)
        default = getattr(_DEFAULTS, field)
        if value != default:
            lines.append(f"        {field}={value},")

    lines.append("    ),")
    return "\n".join(lines)


def render_catalog(specs: tuple[DescriptorSpec, ...]) -> str:
    """Render the full descriptor_catalog.py file content."""
    header = '''\
# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Canonical authored descriptor catalog for dynamic libraries."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

PackagedWith = Literal["ctk", "other", "driver"]
# Backward-compatible alias for downstream imports.
Strategy = PackagedWith


@dataclass(frozen=True, slots=True)
class DescriptorSpec:
    name: str
    packaged_with: PackagedWith
    linux_sonames: tuple[str, ...] = ()
    windows_dlls: tuple[str, ...] = ()
    site_packages_linux: tuple[str, ...] = ()
    site_packages_windows: tuple[str, ...] = ()
    dependencies: tuple[str, ...] = ()
    anchor_rel_dirs_linux: tuple[str, ...] = ("lib64", "lib")
    anchor_rel_dirs_windows: tuple[str, ...] = ("bin/x64", "bin")
    ctk_root_canary_anchor_libnames: tuple[str, ...] = ()
    requires_add_dll_directory: bool = False
    requires_rtld_deepbind: bool = False


DESCRIPTOR_CATALOG: tuple[DescriptorSpec, ...] = (
'''

    body_parts: list[str] = []
    prev_packaged_with = None
    for spec in specs:
        if spec.packaged_with != prev_packaged_with:
            comment = _SECTION_COMMENTS.get(spec.packaged_with)
            if comment is not None:
                body_parts.append(comment)
            prev_packaged_with = spec.packaged_with
        body_parts.append(_render_spec(spec))

    footer = ")\n"
    return header + "\n".join(body_parts) + "\n" + footer


def load_catalog() -> tuple[DescriptorSpec, ...]:
    """Return the current DESCRIPTOR_CATALOG from disk."""
    return DESCRIPTOR_CATALOG


def load_catalog_as_dict() -> dict[str, DescriptorSpec]:
    """Return the current catalog keyed by name."""
    return {spec.name: spec for spec in DESCRIPTOR_CATALOG}


def update_specs(
    catalog: tuple[DescriptorSpec, ...],
    updates: dict[str, dict[str, object]],
) -> tuple[DescriptorSpec, ...]:
    """Apply field updates to matching specs by name, preserving order."""
    result = []
    for spec in catalog:
        if spec.name in updates:
            result.append(dataclasses.replace(spec, **updates[spec.name]))
        else:
            result.append(spec)
    return tuple(result)


def write_catalog(specs: tuple[DescriptorSpec, ...], path: Path | None = None) -> None:
    """Render and write the catalog to disk."""
    if path is None:
        path = CATALOG_PATH
    path.write_text(render_catalog(specs), encoding="utf-8")
