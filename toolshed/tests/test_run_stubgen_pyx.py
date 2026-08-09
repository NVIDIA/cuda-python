# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import os
import sys

import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from run_stubgen_pyx import _HEADER_PREFIX, _normalize_stub_headers

WINDOWS_HEADER = _HEADER_PREFIX + b" from cuda_core\\cuda\\core\\_device.pyx"
POSIX_HEADER = _HEADER_PREFIX + b" from cuda_core/cuda/core/_device.pyx"


def write_stub(tmp_path, content):
    stub = tmp_path / "_device.pyi"
    stub.write_bytes(content)
    return stub


@pytest.mark.agent_authored(model="claude-opus-5")
@pytest.mark.parametrize(
    ("body", "note"),
    [
        pytest.param(b"\n\nclass Device: ...\n", "header, blank line, body", id="body-with-trailing-newline"),
        pytest.param(b"\nclass Device: ...", "body without a trailing newline", id="body-no-trailing-newline"),
        pytest.param(b"\n", "header line only", id="header-only-with-newline"),
        # `.pyi` is excluded from the end-of-file-fixer hook, so a stub with no
        # trailing newline at all is a shape this repo tolerates -- and it is
        # the one the -1 sentinel from `bytes.find` corrupted: `data[-1:]`
        # appended a duplicate of the file's last byte to the header.
        pytest.param(b"", "no trailing newline anywhere", id="header-only-no-newline"),
    ],
)
def test_separator_is_rewritten_without_touching_anything_else(tmp_path, body, note):
    stub = write_stub(tmp_path, WINDOWS_HEADER + body)

    _normalize_stub_headers(tmp_path)

    assert stub.read_bytes() == POSIX_HEADER + body, note


@pytest.mark.agent_authored(model="claude-opus-5")
@pytest.mark.parametrize(
    "content",
    [
        pytest.param(POSIX_HEADER, id="already-posix-no-newline"),
        pytest.param(POSIX_HEADER + b"\nclass Device: ...\n", id="already-posix"),
        pytest.param(b"# hand-written stub\\with\\backslashes", id="not-a-generated-header"),
        pytest.param(b"", id="empty-file"),
    ],
)
def test_files_that_need_no_rewrite_are_left_byte_identical(tmp_path, content):
    stub = write_stub(tmp_path, content)

    _normalize_stub_headers(tmp_path)

    assert stub.read_bytes() == content
