# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Fold pytest's trailing report sections into GitHub Actions log groups.

Reads a pytest console log on stdin and writes it to stdout unchanged, except
that a ``::group::`` / ``::endgroup::`` pair surrounds each of these sections:

* ``warnings summary`` (also ``warnings summary (final)``)
* ``slowest N durations``
* ``pytest-run-parallel report``
* ``short test summary info``, up to its first FAILED or ERROR line

The GitHub log viewer collapses each group by default. ``ci/tools/run-tests``
passes ``-rsxXfE``, which orders the short test summary as skips, xfails,
xpasses, failures, errors. After folding, the end of a failed job's log reads:
the tracebacks, three collapsed sections, the collapsed skip list, the
FAILED/ERROR one-liners, the final count.

Usage::

    pytest ... | python ci/tools/fold_pytest_sections.py

Pipe only stdout. pytest-github-actions-annotate-failures writes its
``::error`` commands to stderr, and the runner honors a workflow command only
at the start of a line of its own stream. Merging the streams with ``2>&1``
splices the commands into pytest's progress lines, where the runner ignores
them.
"""

from __future__ import annotations

import re
import sys
from collections.abc import Iterable
from typing import IO

_ANSI = re.compile(rb"\x1b\[[0-9;]*[A-Za-z]")

# A pytest section banner: "===== title =====". The final count line
# ("=== 3 failed, 2 passed in 1.23s ===") has the same shape and ends any open group.
_BANNER = re.compile(rb"^=+ (?P<title>.+?) =+$")

_FOLDED_TITLES = (
    re.compile(rb"warnings summary( \(final\))?"),
    re.compile(rb"slowest( \d+)? durations"),
    re.compile(rb"pytest-run-parallel report"),
    re.compile(rb"short test summary info"),
)

# The failure one-liners of the short test summary stay outside the fold.
# pytest-run-parallel words a failed parallel test as "PARALLEL FAILED" and a
# failed thread-unsafe test as "FAILED ([thread-unsafe]: reason)".
_FAILURE_LINE = re.compile(rb"^(PARALLEL FAILED|FAILED|ERROR)\b")


def _is_folded(title: bytes) -> bool:
    return any(pattern.fullmatch(title) for pattern in _FOLDED_TITLES)


def fold(lines: Iterable[bytes], out: IO[bytes]) -> None:
    """Copy ``lines`` to ``out`` and wrap the foldable sections in log groups."""
    group_open = False

    def end_group() -> None:
        nonlocal group_open
        if group_open:
            out.write(b"::endgroup::\n")
            group_open = False

    for line in lines:
        plain = _ANSI.sub(b"", line.rstrip(b"\r\n"))
        banner = _BANNER.match(plain)
        if banner:
            end_group()
            title = banner.group("title")
            if _is_folded(title):
                out.write(b"::group::" + title + b"\n")
                group_open = True
        elif group_open and _FAILURE_LINE.match(plain):
            end_group()
        out.write(line)
        out.flush()
    end_group()


def main() -> int:
    fold(sys.stdin.buffer, sys.stdout.buffer)
    return 0


if __name__ == "__main__":
    sys.exit(main())
