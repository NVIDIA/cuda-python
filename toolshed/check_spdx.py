#!/usr/bin/env python3

# Copyright (c) 2025, NVIDIA CORPORATION & AFFILIATES. ALL RIGHTS RESERVED.
# SPDX-License-Identifier: Apache-2.0

import os
import sys

# Intentionally puzzling together EXPECTED_SPDX_STR so that we don't overlook
# if the identifier is missing in this file.
EXPECTED_SPDX_STR = "-".join(("SPDX", "License", "Identifier: "))
EXPECTED_SPDX_BYTES = EXPECTED_SPDX_STR.encode()

SPDX_IGNORE_FILENAME = ".spdx-ignore"


def load_spdx_ignore():
    if not os.path.exists(SPDX_IGNORE_FILENAME):
        return set()
    lines = []
    with open(SPDX_IGNORE_FILENAME, "r", encoding="utf-8") as f:
        for line in f.read().splitlines():
            if not line.startswith("#"):
                lines.append(line)
    ignore_set = set(lines)
    ignore_set.add(SPDX_IGNORE_FILENAME)
    return ignore_set


def has_spdx(filepath):
    with open(filepath, "rb") as f:
        blob = f.read()
    return EXPECTED_SPDX_BYTES in blob


def main(args):
    assert args, "filepaths expected to be passed from pre-commit"

    ignore_set = load_spdx_ignore()

    returncode = 0
    for filepath in args:
        if not has_spdx(filepath) and filepath not in ignore_set:
            print(f"MISSING {EXPECTED_SPDX_STR} {filepath!r}")
            returncode = 1
    return returncode


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
