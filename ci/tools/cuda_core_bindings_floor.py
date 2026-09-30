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

The floors are declared once, by the `cu12`/`cu13` extras in
cuda_core/pyproject.toml, and the wheel's METADATA carries them as
`Requires-Dist: cuda-bindings[all]<14,>=13.4.1; extra == "cu13"`. This script
reads that line.
"""

from __future__ import annotations

import argparse
import re
import sys
import zipfile
from pathlib import Path

# The cuda-bindings requirement of one `cu<major>` extra in METADATA. The name
# is followed by `[`, an operator or whitespace so that other names do not match.
_REQUIRES_DIST_RE = re.compile(
    r"^Requires-Dist:\s*cuda-bindings(?=[\[<>=!~\s])(?P<spec>[^;]*);\s*extra\s*==\s*['\"]cu(?P<major>\d+)['\"]"
)
_FLOOR_RE = re.compile(r">=\s*(\d+)\.(\d+)\.(\d+)")


def floor_from_metadata(metadata: str, major: int) -> str:
    """The floor that the `cu<major>` extra of a wheel's METADATA pins cuda-bindings to."""
    for line in metadata.splitlines():
        m = _REQUIRES_DIST_RE.match(line)
        if m is None or int(m.group("major")) != major:
            continue
        floor = _FLOOR_RE.search(m.group("spec"))
        if floor is None:
            raise SystemExit(f"METADATA pins cuda-bindings without a floor for the cu{major} extra: {line.strip()!r}")
        if int(floor.group(1)) != major:
            raise SystemExit(
                f"METADATA pins a cuda-bindings floor of another major for the cu{major} extra: {line.strip()!r}"
            )
        return ".".join(floor.groups())
    raise SystemExit(f"METADATA declares no cuda-bindings requirement for the cu{major} extra")


def floor_from_wheel(wheel: Path, major: int) -> str:
    with zipfile.ZipFile(wheel) as zf:
        metadata = [name for name in zf.namelist() if name.endswith(".dist-info/METADATA")]
        if len(metadata) != 1:
            raise SystemExit(f"{wheel.name} contains {len(metadata)} METADATA files, expected 1. Is it a wheel?")
        return floor_from_metadata(zf.read(metadata[0]).decode("utf-8"), major)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--wheel", type=Path, required=True, help="the cuda-core wheel under test")
    parser.add_argument("--major", type=int, required=True, help="CUDA major series (12 or 13)")
    args = parser.parse_args(argv)
    print(floor_from_wheel(args.wheel, args.major))
    return 0


if __name__ == "__main__":
    sys.exit(main())
