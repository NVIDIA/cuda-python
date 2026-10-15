#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Check this repository's main-branch source links against the local checkout."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]


def preprocess_links(content: str, root: Path = ROOT) -> str:
    """Keep published URLs in source files while checking paths before merge."""
    local_prefix = root.resolve().as_uri() + "/"
    for view in ("blob", "tree"):
        content = content.replace(f"https://github.com/NVIDIA/cuda-python/{view}/main/", local_prefix)
    return content


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("input", type=Path)
    args = parser.parse_args()
    # Lychee consumes stdout without changing the input file or its relative-link base.
    sys.stdout.write(preprocess_links(args.input.read_text(encoding="utf-8")))


if __name__ == "__main__":
    main()
