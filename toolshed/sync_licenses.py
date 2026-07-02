# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import argparse
import sys
from pathlib import Path

PACKAGE_LICENSES = (
    Path("cuda_bindings/LICENSE"),
    Path("cuda_core/LICENSE"),
    Path("cuda_pathfinder/LICENSE"),
    Path("cuda_python/LICENSE"),
)


def sync_licenses(root: Path, fix: bool) -> int:
    canonical_path = root / "LICENSE"
    canonical = canonical_path.read_bytes()
    mismatches = [relative for relative in PACKAGE_LICENSES if (root / relative).read_bytes() != canonical]

    if not mismatches:
        return 0

    if fix:
        for relative in mismatches:
            (root / relative).write_bytes(canonical)
            print(f"Updated {relative} from LICENSE")
        return 0

    for relative in mismatches:
        print(f"Package license differs from LICENSE: {relative}", file=sys.stderr)
    print("Run toolshed/sync_licenses.py --fix to update package copies.", file=sys.stderr)
    return 1


def main() -> int:
    parser = argparse.ArgumentParser(description="Check package license files against the canonical root LICENSE.")
    parser.add_argument("--fix", action="store_true", help="replace mismatched package copies")
    args = parser.parse_args()

    root = Path(__file__).resolve().parents[1]
    return sync_licenses(root, args.fix)


if __name__ == "__main__":
    raise SystemExit(main())
