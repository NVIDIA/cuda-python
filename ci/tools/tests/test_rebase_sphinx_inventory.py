# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import zlib

import pytest

from cuda_python.docs.copy_rebased_sphinx_inventory import copy_rebased_inventory


@pytest.mark.agent_authored(model="gpt-6")
def test_copy_rebases_relative_inventory_targets(tmp_path):
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
    source = tmp_path / "latest" / "objects.inv"
    source.parent.mkdir()
    source.write_bytes(original)
    destination = tmp_path / "objects.inv"

    copy_rebased_inventory(source, destination)

    rebased = destination.read_bytes()
    assert rebased.startswith(header)
    assert zlib.decompress(rebased[len(header) :]) == (
        b"cuda.bindings.driver.cuInit py:function 1 latest/module/driver.html#$ -\n"
        b"figure 1 std:label -1 latest/overview.html#figure-1 Example with spaces\n"
        b"external std:label -1 https://example.org/overview.html#target -\n"
    )
    assert source.read_bytes() == original
