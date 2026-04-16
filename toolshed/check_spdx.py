# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import datetime
import os
import re
import subprocess
import sys
from pathlib import PureWindowsPath

import pathspec

# Intentionally puzzling together SPDX prefixes so that we don't overlook if the
# identifiers are missing in this file.
SPDX_LICENSE_IDENTIFIER_PREFIX = b"-".join((b"SPDX", b"License", b"Identifier: "))
SPDX_FILE_COPYRIGHT_TEXT_PREFIX = b"-".join((b"SPDX", b"FileCopyrightText: "))

LICENSE_IDENTIFIER_REGEX = re.compile(re.escape(SPDX_LICENSE_IDENTIFIER_PREFIX) + rb"(?P<license_identifier>[^\r\n]+)")

EXPECTED_LICENSE_IDENTIFIERS = (
    ("cuda_bindings/", "LicenseRef-NVIDIA-SOFTWARE-LICENSE"),
    ("cuda_core/", "Apache-2.0"),
    ("cuda_pathfinder/", "Apache-2.0"),
    ("cuda_python/", "LicenseRef-NVIDIA-SOFTWARE-LICENSE"),
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
CURRENT_YEAR = str(datetime.datetime.now(tz=datetime.timezone.utc).year)


def is_staged(filepath):
    # If the file is staged, we need to update it to the current year
    process = subprocess.run(  # noqa: S603
        ["git", "diff", "--staged", "--", filepath],  # noqa: S607
        capture_output=True,
        text=True,
    )
    return process.stdout.strip() != ""


def normalize_repo_path(filepath):
    # We compare against repo prefixes like "cuda_core/" regardless of host OS.
    # os.path.normpath is host-dependent: on POSIX it leaves "\" untouched, and
    # on Windows it normalizes to "\" separators, so neither gives a stable
    # forward-slash form for this prefix check.
    return PureWindowsPath(filepath).as_posix()


def get_expected_license_identifier(filepath):
    normalized_path = normalize_repo_path(filepath)
    for prefix, license_identifier in EXPECTED_LICENSE_IDENTIFIERS:
        if normalized_path.startswith(prefix):
            return license_identifier
    return None


def validate_required_spdx_field(filepath, blob, expected_bytes):
    if expected_bytes in blob:
        return True
    print(f"MISSING {expected_bytes.decode()}{filepath!r}")
    return False


def extract_license_identifier(blob):
    match = LICENSE_IDENTIFIER_REGEX.search(blob)
    if match is None:
        return None
    try:
        return match.group("license_identifier").decode("ascii")
    except UnicodeDecodeError:
        return None


def validate_license_identifier(filepath, blob):
    license_identifier = extract_license_identifier(blob)
    if license_identifier is None:
        print(f"MISSING valid SPDX license identifier in {filepath!r}")
        return False

    expected_license_identifier = get_expected_license_identifier(filepath)
    if expected_license_identifier is None:
        return True

    if license_identifier != expected_license_identifier:
        print(
            f"INVALID SPDX license identifier {license_identifier!r} "
            f"(expected {expected_license_identifier!r}) in {filepath!r}"
        )
        return False

    return True


def validate_or_fix_copyright(filepath, blob, fix):
    match = re.search(COPYRIGHT_REGEX, blob)
    if match is None:
        print(f"MISSING valid copyright line in {filepath!r}")
        return False, blob

    years = match.group("years").decode()
    if "-" in years:
        start_year, end_year = years.split("-", 1)
        if int(start_year) > int(end_year):
            print(f"INVALID copyright years {years!r} in {filepath!r}")
            return False, blob
    else:
        start_year = end_year = years

    if not is_staged(filepath) or int(end_year) >= int(CURRENT_YEAR):
        return True, blob

    print(f"OUTDATED copyright {years!r} (expected {CURRENT_YEAR!r}) in {filepath!r}")
    if not fix:
        return False, blob

    new_years = f"{start_year}-{CURRENT_YEAR}"
    return (
        False,
        re.sub(
            COPYRIGHT_REGEX,
            COPYRIGHT_SUB.format(new_years).encode("ascii"),
            blob,
        ),
    )


def find_or_fix_spdx(filepath, fix):
    with open(filepath, "rb") as f:
        blob = f.read()
    if len(blob.strip()) == 0:
        return True

    good = True
    has_license_identifier = validate_required_spdx_field(filepath, blob, SPDX_LICENSE_IDENTIFIER_PREFIX)
    has_copyright = validate_required_spdx_field(filepath, blob, SPDX_FILE_COPYRIGHT_TEXT_PREFIX)

    if not has_license_identifier or not validate_license_identifier(filepath, blob):
        good = False

    if not has_copyright:
        good = False
    else:
        copyright_ok, updated_blob = validate_or_fix_copyright(filepath, blob, fix)
        if updated_blob != blob:
            with open(filepath, "wb") as f:
                f.write(updated_blob)
        if not copyright_ok:
            good = False

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
