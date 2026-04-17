# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import datetime
import fnmatch
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

TOP_LEVEL_FILE_LICENSE_IDENTIFIER = "Apache-2.0"

# Every top-level directory needs to have an entry here, so new paths
# can't slip in without a reviewed license decision.
TOP_LEVEL_DIRS_LICENSE_IDENTIFIERS = {
    ".github": "Apache-2.0",
    "ci": "Apache-2.0",
    "cuda_bindings": "LicenseRef-NVIDIA-SOFTWARE-LICENSE",
    "cuda_core": "Apache-2.0",
    "cuda_pathfinder": "Apache-2.0",
    "cuda_python": "LicenseRef-NVIDIA-SOFTWARE-LICENSE",
    "cuda_python_test_helpers": "Apache-2.0",
    "scripts": "Apache-2.0",
    "toolshed": "Apache-2.0",
}

SPECIAL_CASE_LICENSE_IDENTIFIERS = {
    # key: repo-relative path or glob, value: expected SPDX license identifier
    "cuda_bindings/benchmarks/*": "Apache-2.0",
    "cuda_bindings/benchmarks/pytest-legacy/*": "LicenseRef-NVIDIA-SOFTWARE-LICENSE",
}

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


def get_top_level_directory(normalized_path):
    if "/" not in normalized_path:
        return None
    return normalized_path.split("/", 1)[0]


def get_expected_license_identifier(filepath):
    normalized_path = normalize_repo_path(filepath)
    matching_special_cases = [
        (prefix, license_identifier)
        for prefix, license_identifier in SPECIAL_CASE_LICENSE_IDENTIFIERS.items()
        if fnmatch.fnmatchcase(normalized_path, prefix)
    ]
    if matching_special_cases:
        return max(matching_special_cases, key=lambda item: len(item[0]))[1], None

    top_level_directory = get_top_level_directory(normalized_path)
    if top_level_directory is None:
        return TOP_LEVEL_FILE_LICENSE_IDENTIFIER, None

    if top_level_directory not in TOP_LEVEL_DIRS_LICENSE_IDENTIFIERS:
        return (
            None,
            f"MISSING TOP_LEVEL_DIRS_LICENSE_IDENTIFIERS entry for top-level directory "
            f"{top_level_directory!r} required by {filepath!r}",
        )

    return TOP_LEVEL_DIRS_LICENSE_IDENTIFIERS[top_level_directory], None


def validate_required_spdx_field(filepath, blob, expected_bytes):
    if expected_bytes in blob:
        return True
    print(f"MISSING {expected_bytes.decode()}{filepath!r}")
    return False


def extract_license_identifier(blob):
    match = LICENSE_IDENTIFIER_REGEX.search(blob)
    if match is None:
        return None
    license_identifier = match.group("license_identifier").decode("ascii", errors="replace").strip()
    for comment_suffix in ("-->", "*/"):
        if license_identifier.endswith(comment_suffix):
            license_identifier = license_identifier.removesuffix(comment_suffix).rstrip()
    return license_identifier or None


def validate_license_identifier(filepath, blob):
    license_identifier = extract_license_identifier(blob)
    if license_identifier is None:
        print(f"MISSING valid SPDX license identifier in {filepath!r}")
        return False

    expected_license_identifier, configuration_error = get_expected_license_identifier(filepath)
    if configuration_error is not None:
        print(configuration_error)
        return False

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
