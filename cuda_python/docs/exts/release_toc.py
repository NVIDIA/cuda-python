# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: LicenseRef-NVIDIA-SOFTWARE-LICENSE

from pathlib import Path

from packaging.version import Version
from sphinx.directives.other import TocTree


class TocTreeSorted(TocTree):
    """A toctree directive that sorts entries by version."""

    def parse_content(self, toctree):
        super().parse_content(toctree)

        if not toctree["glob"]:
            return

        toctree["entries"] = [
            (Version(Path(x[1]).name.removesuffix("-notes")), x[1]) for x in toctree.get("entries", [])
        ]
        toctree["entries"].sort(key=lambda x: x[0], reverse=True)
        toctree["entries"] = [(str(x[0]), x[1]) for x in toctree["entries"]]


def setup(app):
    app.add_directive("toctree", TocTreeSorted, override=True)
