# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import datetime
import os
import re
import subprocess
import sys

import pathspec

# Intentionally puzzling together EXPECTED_SPDX_BYTES so that we don't overlook
# if the identifiers are missing in this file.
EXPECTED_SPDX_BYTES = (
    b"-".join((b"SPDX", b"License", b"Identifier: ")),
    b"-".join((b"SPDX", b"FileCopyrightText: ")),
)

SPDX_IGNORE_FILENAME = ".spdx-ignore"


def load_spdx_ignore():
    if os.path.exists(SPDX_IGNORE_FILENAME):
        with open(SPDX_IGNORE_FILENAME, encoding="utf-8") as f:
            lines = f.readlines()
    else:
        lines = []
    lines.append(SPDX_IGNORE_FILENAME + "\n")
    return pathspec.PathSpec.from_lines("gitwildmatch", lines)


COPYRIGHT_REGEX = (
    rb"Copyright \(c\) (?P<years>[0-9]{4}(-[0-9]{4})?) "
    rb"(?P<affiliation>NVIDIA CORPORATION( & AFFILIATES\. All rights reserved\.)?)"
)
COPYRIGHT_SUB = r"Copyright (c) {} \g<affiliation>"
CURRENT_YEAR = str(datetime.date.today().year)


def is_staged(filepath):
    # If the file is staged, we need to update it to the current year
    process = subprocess.run(  # noqa: S603
        ["git", "diff", "--staged", "--", filepath],  # noqa: S607
        capture_output=True,
        text=True,
    )
    return process.stdout.strip() != ""


def find_or_fix_spdx(filepath, fix):
    with open(filepath, "rb") as f:
        blob = f.read()
    if len(blob.strip()) == 0:
        return True

    good = True
    for expected_bytes in EXPECTED_SPDX_BYTES:
        if expected_bytes not in blob:
            print(f"MISSING {expected_bytes.decode()}{filepath!r}")
            good = False
            continue

        match = re.search(COPYRIGHT_REGEX, blob)
        if match is None:
            print(f"MISSING valid copyright line in {filepath!r}")
            good = False
            continue

        years = match.group("years").decode()
        if "-" in years:
            start_year, end_year = years.split("-", 1)
            if int(start_year) > int(end_year):
                print(f"INVALID copyright years {years!r} in {filepath!r}")
                good = False
                continue
        else:
            start_year = end_year = years

        staged = is_staged(filepath)

        if staged and int(end_year) < int(CURRENT_YEAR):
            print(f"OUTDATED copyright {years!r} (expected {CURRENT_YEAR!r}) in {filepath!r}")
            good = False

            if fix:
                new_years = f"{start_year}-{CURRENT_YEAR}"
                blob = re.sub(
                    COPYRIGHT_REGEX,
                    COPYRIGHT_SUB.format(new_years).encode("ascii"),
                    blob,
                )
                with open(filepath, "wb") as f:
                    f.write(blob)

    return good


def main(args):
    assert args, "filepaths expected to be passed from pre-commit"

    if "--fix" in args:
        fix = True
        del args[args.index("--fix")]
    else:
        fix = False

    ignore_spec = load_spdx_ignore()

    returncode = 0
    for filepath in args:
        if ignore_spec.match_file(filepath):
            continue
        if not find_or_fix_spdx(filepath, fix):
            returncode = 1
    return returncode


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
