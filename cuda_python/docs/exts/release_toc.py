# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: LicenseRef-NVIDIA-SOFTWARE-LICENSE

from pathlib import Path

from packaging.version import Version


def reverse_toctree(app, doctree, docname):
    """Sort the entries in a release toctree in by version."""
    if docname == "release":
        for node in doctree.traverse():
            if node.tagname == "toctree":
                node["entries"].sort(key=lambda x: Version(Path(x[1]).name.removesuffix("-notes")), reverse=True)
                break


def setup(app):
    app.connect("doctree-resolved", reverse_toctree)
