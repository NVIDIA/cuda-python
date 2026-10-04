# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Check that the shared build-helpers block is byte-identical in all build_hooks.py files.

The block delimited by '# --- begin shared build helpers' and
'# --- end shared build helpers ---' is duplicated verbatim between
cuda_bindings/build_hooks.py, cuda_bindings_12/build_hooks.py, and
cuda_core/build_hooks.py (PEP 517 build isolation forbids a shared import).
It contains the toolchain helpers and
the Cython cache helpers. Run as a pre-commit hook so drift is caught at
commit time.
"""

from __future__ import annotations

import sys
from pathlib import Path

_MARKER_START = "# --- begin shared build helpers"
_MARKER_END = "# --- end shared build helpers ---"

ROOT = Path(__file__).resolve().parents[1]
_BINDINGS = ROOT / "cuda_bindings" / "build_hooks.py"
_MAINTENANCE_BINDINGS = ROOT / "cuda_bindings_12" / "build_hooks.py"
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
    for path in (_MAINTENANCE_BINDINGS, _CORE):
        if bindings_block != _shared_block(path):
            sys.exit(
                "ERROR: shared build helpers are out of sync between\n"
                f"  {_BINDINGS}\n"
                f"  {path}\n"
                "Edit all copies to match and commit again."
            )


if __name__ == "__main__":
    main()
