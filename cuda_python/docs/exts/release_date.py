# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: LicenseRef-NVIDIA-SOFTWARE-LICENSE

"""Sphinx extension to auto-inject release dates from git tags.

For every release-notes page (``release/<version>-notes``), this
extension looks up the corresponding git tag and injects a
``Released on <date>`` line after the RST title.  Pages that already
contain such a line, or whose version has no tag yet, are left
untouched.
"""

from __future__ import annotations

import re
import subprocess
from datetime import UTC, datetime

from sphinx.application import Sphinx

_RELEASED_ON_RE = re.compile(r"Released on ", re.IGNORECASE)
_RELEASE_NOTE_RE = re.compile(r"^release/(.+)-notes$")
_UNDERLINE_RE = re.compile(r"^={3,}[ \t]*$", re.MULTILINE)

# project name (from conf.py) -> git tag prefix
_TAG_PREFIXES: dict[str, str] = {
    "cuda.core": "cuda-core-v",
    "cuda.pathfinder": "cuda-pathfinder-v",
    "cuda.bindings": "v",
    "CUDA Python": "v",
}


def _format_date(iso_date: str) -> str:
    """``2026-03-06`` -> ``Mar 6, 2026``."""
    dt = datetime.strptime(iso_date, "%Y-%m-%d").replace(tzinfo=UTC)
    return f"{dt.strftime('%b')} {dt.day}, {dt.year}"


def _git_tag_date(tag: str) -> str | None:
    """Return the creator date (YYYY-MM-DD) for *tag*, or None."""
    try:
        result = subprocess.run(  # noqa: S603
            [  # noqa: S607
                "git",
                "for-each-ref",
                "--format=%(creatordate:short)",
                f"refs/tags/{tag}",
            ],
            capture_output=True,
            text=True,
            timeout=5,
        )
        date_str = result.stdout.strip()
    except (subprocess.TimeoutExpired, FileNotFoundError):
        date_str = ""
    return date_str or None


def _on_source_read(app: Sphinx, docname: str, source: list[str]) -> None:
    m = _RELEASE_NOTE_RE.match(docname)
    if not m:
        return

    text = source[0]
    if _RELEASED_ON_RE.search(text):
        return

    version = m.group(1)
    prefix = _TAG_PREFIXES.get(app.config.project)
    if prefix is None:
        return

    tag = prefix + version
    iso_date = _git_tag_date(tag)
    if not iso_date:
        return

    underline = _UNDERLINE_RE.search(text)
    if not underline:
        return

    date_line = f"Released on {_format_date(iso_date)}"

    # Insert after the title underline: skip any blank lines, then place
    # the date line surrounded by single blank lines before the content.
    after = text[underline.end() :]
    stripped = after.lstrip("\n")
    source[0] = text[: underline.end()] + f"\n\n{date_line}\n\n" + stripped


def setup(app: Sphinx) -> dict:
    app.connect("source-read", _on_source_read)
    return {"version": "1.0", "parallel_read_safe": True}
