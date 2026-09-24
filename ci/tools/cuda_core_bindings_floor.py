#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

"""Print the cuda-bindings floor of a cuda-core wheel for one CUDA major.

    cuda_core_bindings_floor.py --wheel dist/cuda_core-*.whl --major 13
    -> 13.4.1

CI installs `cuda-bindings==<floor>` next to a freshly built cuda-core wheel to
test the oldest cuda-bindings that wheel supports (BINDINGS_SOURCE=floor in
ci/tools/env-vars). This script reads the floor from the wheel under test, not
from the checkout, so a nightly job that tests a wheel built from another
commit reads that wheel's floor.

Each build records its floor in the generated cuda/core/_build_info.py. A
single-major build places the file at top level. The merged wheel places it
under cuda/core/cu<major>/. This script parses the CUDA_BINDINGS_FLOOR literal
out of it and never runs it.
"""

from __future__ import annotations

import argparse
import ast
import sys
import zipfile
from pathlib import Path

MODULE = "_build_info.py"


def _literal(source: str, name: str):
    """The literal that `source` assigns to `name` at module level. Parses the source and never runs it."""
    for node in ast.parse(source, MODULE).body:
        if isinstance(node, ast.AnnAssign):
            targets = [node.target]
        elif isinstance(node, ast.Assign):
            targets = node.targets
        else:
            continue
        if node.value is not None and any(isinstance(t, ast.Name) and t.id == name for t in targets):
            return ast.literal_eval(node.value)
    raise SystemExit(f"{MODULE} does not assign {name}")


def floor_from_source(source: str, major: int) -> str:
    if _literal(source, "CUDA_MAJOR") != major:
        raise SystemExit(f"{MODULE} records a CUDA {_literal(source, 'CUDA_MAJOR')} build, not CUDA {major}")
    return ".".join(str(part) for part in _literal(source, "CUDA_BINDINGS_FLOOR"))


def floor_from_wheel(wheel: Path, major: int) -> str:
    with zipfile.ZipFile(wheel) as zf:
        names = set(zf.namelist())
        for candidate in (f"cuda/core/cu{major}/{MODULE}", f"cuda/core/{MODULE}"):
            if candidate in names:
                return floor_from_source(zf.read(candidate).decode("utf-8"), major)
    raise SystemExit(f"{wheel.name} contains no build for CUDA {major}: it has no {MODULE}. Is it a cuda-core wheel?")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--wheel", type=Path, required=True, help="the cuda-core wheel under test")
    parser.add_argument("--major", type=int, required=True, help="CUDA major series (12 or 13)")
    args = parser.parse_args(argv)
    print(floor_from_wheel(args.wheel, args.major))
    return 0


if __name__ == "__main__":
    sys.exit(main())
