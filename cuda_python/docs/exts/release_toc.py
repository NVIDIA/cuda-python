# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from pathlib import Path

from packaging.version import InvalidVersion, Version
from sphinx.directives.other import TocTree


def _version_sort_key(version_text):
    normalized = version_text.replace(".x", ".999999")
    try:
        return (1, Version(normalized))
    except InvalidVersion:
        return (0, version_text)


def _is_prerelease(version_text):
    try:
        version = Version(version_text)
        return version.is_prerelease
    except InvalidVersion:
        return False


class TocTreeSorted(TocTree):
    """A toctree directive that sorts entries by version."""

    def parse_content(self, toctree):
        super().parse_content(toctree)

        if not toctree["glob"]:
            return

        entries = toctree.get("entries", [])
        if not entries:
            return

        entries = [(Path(x[1]).name.removesuffix("-notes"), x[1]) for x in entries]
        # Don't include any prereleases in the toctree
        entries = [entry for entry in entries if not _is_prerelease(entry[0])]
        entries.sort(key=lambda x: _version_sort_key(x[0]), reverse=True)
        toctree["entries"] = entries


def setup(app):
    app.add_directive("toctree", TocTreeSorted, override=True)
    return {"parallel_read_safe": True, "parallel_write_safe": True}
