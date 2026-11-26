# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import zlib

import pytest

from ci.tools.rebase_sphinx_inventory import rebase_root_inventories


@pytest.mark.agent_authored(model="gpt-6")
def test_only_root_inventory_targets_are_rebased(tmp_path):
    header = (
        b"# Sphinx inventory version 2\n"
        b"# Project: cuda.bindings\n"
        b"# Version: 13.5.0\n"
        b"# The remainder of this file is compressed using zlib.\n"
    )
    entries = (
        b"cuda.bindings.driver.cuInit py:function 1 module/driver.html#$ -\n"
        b"figure 1 std:label -1 overview.html#figure-1 Example with spaces\n"
        b"external std:label -1 https://example.org/overview.html#target -\n"
    )
    original = header + zlib.compress(entries)
    roots = [tmp_path, *(tmp_path / f"cuda-{part}" for part in ("bindings", "core", "pathfinder"))]
    for root in roots:
        for path in (root, root / "latest", root / "13.5.0"):
            path.mkdir(parents=True, exist_ok=True)
            (path / "objects.inv").write_bytes(original)

    rebase_root_inventories(tmp_path)

    for root in roots:
        rebased = (root / "objects.inv").read_bytes()
        assert rebased.startswith(header)
        assert zlib.decompress(rebased[len(header) :]) == (
            b"cuda.bindings.driver.cuInit py:function 1 latest/module/driver.html#$ -\n"
            b"figure 1 std:label -1 latest/overview.html#figure-1 Example with spaces\n"
            b"external std:label -1 https://example.org/overview.html#target -\n"
        )
        assert (root / "latest" / "objects.inv").read_bytes() == original
        assert (root / "13.5.0" / "objects.inv").read_bytes() == original
