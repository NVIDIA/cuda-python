# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import hashlib
import os
import re
import subprocess
import sys
from pathlib import Path, PureWindowsPath

GENERATED_FILE_MARKER_FRAGMENT = "-GENERATED-DO-NOT-MODIFY-THIS-FILE"
GENERATED_FILE_SEAL_TOKEN = "CYTHON-BINDINGS-GENERATED-DO-NOT-MODIFY-THIS-FILE:"  # noqa: S105
SUPPORTED_GENERATED_FILE_SEAL_FORMATS = frozenset({1})

assert GENERATED_FILE_MARKER_FRAGMENT in GENERATED_FILE_SEAL_TOKEN
_TOKEN_BYTES = GENERATED_FILE_SEAL_TOKEN.encode("ascii")
_MARKER_REGEX = re.compile(
    rb"^(?P<prefix>#|\.\.) "
    + re.escape(_TOKEN_BYTES)
    + rb" format=(?P<format>[0-9]+); content-sha256=(?P<digest>[0-9a-f]{64})\n$"
)
_SOURCE_SUFFIXES = frozenset({".py", ".pxd", ".pxi", ".pyx"})


def normalize_repo_path(filepath):
    return PureWindowsPath(filepath).as_posix()


def expected_comment_prefix(filepath):
    suffixes = Path(normalize_repo_path(filepath)).suffixes
    if suffixes and suffixes[-1] == ".in":
        suffixes = suffixes[:-1]
    suffix = suffixes[-1] if suffixes else ""
    if suffix in _SOURCE_SUFFIXES:
        return b"#"
    if suffix == ".rst":
        return b".."
    return None


def load_previously_sealed_paths():
    process = subprocess.run(  # noqa: S603
        ["git", "grep", "-l", "-I", "-F", "-e", GENERATED_FILE_MARKER_FRAGMENT, "HEAD", "--"],  # noqa: S607
        capture_output=True,
        text=True,
    )
    if process.returncode not in (0, 1):
        detail = process.stderr.strip() or f"git grep exited with status {process.returncode}"
        raise RuntimeError(f"could not inspect previously sealed files: {detail}")

    head_prefix = "HEAD:"
    return {
        normalize_repo_path(line.removeprefix(head_prefix))
        for line in process.stdout.splitlines()
        if line.startswith(head_prefix)
    }


def validate_generated_file_seal(filepath, previously_sealed_paths):
    normalized_path = normalize_repo_path(filepath)
    try:
        blob = Path(filepath).read_bytes()
    except OSError as error:
        print(f"ERROR reading {filepath!r}: {error}")
        return False

    lines = blob.splitlines(keepends=True)
    marker_indexes = [index for index, line in enumerate(lines) if _TOKEN_BYTES in line]

    if not marker_indexes:
        if normalized_path in previously_sealed_paths:
            print(f"MISSING generated-file seal in {filepath!r}")
            return False
        return True

    if len(marker_indexes) != 1:
        print(f"INVALID generated-file seal count in {filepath!r}: found {len(marker_indexes)}, expected 1")
        return False

    marker_index = marker_indexes[0]
    match = _MARKER_REGEX.fullmatch(lines[marker_index])
    if match is None:
        print(f"MALFORMED generated-file seal in {filepath!r}")
        return False

    seal_format = int(match.group("format"))
    if seal_format not in SUPPORTED_GENERATED_FILE_SEAL_FORMATS:
        print(f"UNSUPPORTED generated-file seal format {seal_format} in {filepath!r}")
        return False

    expected_prefix = expected_comment_prefix(filepath)
    if expected_prefix is None:
        print(f"UNSUPPORTED sealed generated-file extension in {filepath!r}")
        return False
    if match.group("prefix") != expected_prefix:
        print(f"INVALID generated-file seal comment prefix in {filepath!r}")
        return False

    unsealed_blob = b"".join(lines[:marker_index] + lines[marker_index + 1 :])
    computed_digest = hashlib.sha256(unsealed_blob).hexdigest().encode("ascii")
    recorded_digest = match.group("digest")
    if recorded_digest != computed_digest:
        print(
            f"MISMATCHED generated-file seal in {filepath!r}: "
            f"recorded {recorded_digest.decode()}, computed {computed_digest.decode()}"
        )
        return False

    return True


def main(args):
    assert args, "filepaths expected to be passed from pre-commit"

    try:
        previously_sealed_paths = load_previously_sealed_paths()
    except RuntimeError as error:
        print(f"ERROR: {error}")
        return 2

    returncode = 0
    for filepath in args:
        if not os.path.isfile(filepath):
            continue
        if not validate_generated_file_seal(filepath, previously_sealed_paths):
            returncode = 1
    return returncode


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
