#!/usr/bin/env python3

# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

"""Update descriptor_catalog.py from CTK installations.

On Linux, scans directories for .so files and extracts SONAMEs via readelf.
On Windows, parses 7z listing files generated from CTK .exe installers.

Usage:
    # Linux — pass one or more CTK lib directories:
    python toolshed/update_catalog.py /path/to/ctk12/lib64 /path/to/ctk13/lib64

    # Windows — pass 7z listing .txt files:
    #   for exe in *.exe; do 7z l "$exe" > "${exe%.exe}.txt"; done
    python toolshed/update_catalog.py listing12.txt listing13.txt
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))


def main() -> None:
    if len(sys.argv) < 2:
        print(__doc__, file=sys.stderr)
        sys.exit(1)

    args = sys.argv[1:]

    if sys.platform == "win32":
        from build_pathfinder_dlls import run as run_dlls

        run_dlls(listing_files=args)
    else:
        from build_pathfinder_sonames import run as run_sonames

        run_sonames(roots=args)


if __name__ == "__main__":
    main()
