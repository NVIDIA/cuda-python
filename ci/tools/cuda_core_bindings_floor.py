#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

"""Print the cuda-bindings floor of a cuda-core wheel for one CUDA major.

    cuda_core_bindings_floor.py --wheel dist/cuda_core-*.whl --major 13
    -> 13.4.1

CI installs `cuda-bindings==<floor>` next to a freshly built cuda-core wheel to
test the oldest cuda-bindings that wheel supports (BINDINGS_SOURCE=floor in
ci/tools/env-vars). The floor is read from the wheel under test rather than
from the checkout, so a nightly job that tests a wheel built from another
commit reads that wheel's floor.

The wheel carries the import-free module cuda/core/_bindings_floor.py, at top
level in a single-major build and under cuda/core/cu<major>/ in the merged
wheel; this script evaluates it and prints CUDA_BINDINGS_FLOOR[major].
"""

from __future__ import annotations

import argparse
import sys
import zipfile
from pathlib import Path

MODULE = "_bindings_floor.py"


def floor_from_source(source: str, major: int) -> str:
    namespace: dict = {}
    exec(compile(source, MODULE, "exec"), namespace)
    floors = namespace["CUDA_BINDINGS_FLOOR"]
    if major not in floors:
        raise SystemExit(f"CUDA {major} is not a supported major (floors: {sorted(floors)})")
    return namespace["format_version"](floors[major])


def floor_from_wheel(wheel: Path, major: int) -> str:
    with zipfile.ZipFile(wheel) as zf:
        names = set(zf.namelist())
        for candidate in (f"cuda/core/cu{major}/{MODULE}", f"cuda/core/{MODULE}"):
            if candidate in names:
                return floor_from_source(zf.read(candidate).decode("utf-8"), major)
    raise SystemExit(f"{wheel.name} contains no {MODULE}; is it a cuda-core wheel?")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--wheel", type=Path, required=True, help="the cuda-core wheel under test")
    parser.add_argument("--major", type=int, required=True, help="CUDA major series (12 or 13)")
    args = parser.parse_args(argv)
    print(floor_from_wheel(args.wheel, args.major))
    return 0


if __name__ == "__main__":
    sys.exit(main())
