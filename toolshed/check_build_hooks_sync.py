# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Check that the shared toolchain helpers are byte-identical in both build_hooks.py files.

The block delimited by '# --- begin shared toolchain helpers' and
'# --- end shared toolchain helpers ---' is duplicated verbatim between
cuda_bindings/build_hooks.py and cuda_core/build_hooks.py (PEP 517 build
isolation forbids a shared import). Run as a pre-commit hook so drift is
caught at commit time.
"""

from __future__ import annotations

import sys
from pathlib import Path

_MARKER_START = "# --- begin shared toolchain helpers"
_MARKER_END = "# --- end shared toolchain helpers ---"

ROOT = Path(__file__).resolve().parents[1]
_BINDINGS = ROOT / "cuda_bindings" / "build_hooks.py"
_CORE = ROOT / "cuda_core" / "build_hooks.py"


def _shared_block(path: Path) -> str:
    text = path.read_text(encoding="utf-8")
    try:
        start = text.index(_MARKER_START)
        end = text.index(_MARKER_END) + len(_MARKER_END)
    except ValueError as exc:
        sys.exit(f"ERROR: sync marker not found in {path}: {exc}")
    return text[start:end]


def main() -> None:
    bindings_block = _shared_block(_BINDINGS)
    core_block = _shared_block(_CORE)
    if bindings_block != core_block:
        sys.exit(
            "ERROR: shared toolchain helpers are out of sync between\n"
            f"  {_BINDINGS}\n"
            f"  {_CORE}\n"
            "Edit both files to match and commit again."
        )


if __name__ == "__main__":
    main()
