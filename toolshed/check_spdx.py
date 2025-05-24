# Copyright (c) 2025, NVIDIA CORPORATION & AFFILIATES. ALL RIGHTS RESERVED.
# SPDX-License-Identifier: Apache-2.0

import os
import sys

import pathspec

# Intentionally puzzling together EXPECTED_SPDX_STR so that we don't overlook
# if the identifier is missing in this file.
EXPECTED_SPDX_STR = "-".join(("SPDX", "License", "Identifier: "))
EXPECTED_SPDX_BYTES = EXPECTED_SPDX_STR.encode()

SPDX_IGNORE_FILENAME = ".spdx-ignore"


def load_spdx_ignore():
    if os.path.exists(SPDX_IGNORE_FILENAME):
        with open(SPDX_IGNORE_FILENAME, "r", encoding="utf-8") as f:
            lines = f.readlines()
    else:
        lines = []
    lines.append(SPDX_IGNORE_FILENAME + "\n")
    return pathspec.PathSpec.from_lines("gitwildmatch", lines)


def has_spdx_or_is_empty(filepath):
    with open(filepath, "rb") as f:
        blob = f.read()
    return len(blob.strip()) == 0 or EXPECTED_SPDX_BYTES in blob


def main(args):
    assert args, "filepaths expected to be passed from pre-commit"

    ignore_spec = load_spdx_ignore()

    returncode = 0
    for filepath in args:
        if not ignore_spec.match_file(filepath) and not has_spdx_or_is_empty(filepath):
            print(f"MISSING {EXPECTED_SPDX_STR}{filepath!r}")
            returncode = 1
    return returncode


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
