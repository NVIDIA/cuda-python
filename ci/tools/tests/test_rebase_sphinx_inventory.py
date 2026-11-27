# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import zlib
from pathlib import Path

import pytest

from cuda_python.docs.copy_rebased_sphinx_inventory_one_level_up import copy_rebased_inventory_one_level_up

_INVENTORY_HEADER = (
    b"# Sphinx inventory version 2\n"
    b"# Project: cuda.bindings\n"
    b"# Version: 13.5.0\n"
    b"# The remainder of this file is compressed using zlib.\n"
)


@pytest.mark.agent_authored(model="gpt-6")
def test_copy_rebases_relative_inventory_targets(tmp_path, monkeypatch):
    entries = (
        b"cuda.bindings.driver.cuInit py:function 1 module/driver.html#$ -\n"
        b"figure 1 std:label -1 overview.html#figure-1 Example with spaces\n"
        b"external std:label -1 https://example.org/overview.html#target -\n"
    )
    original = _INVENTORY_HEADER + zlib.compress(entries)
    source = tmp_path / "latest" / "objects.inv"
    source.parent.mkdir()
    source.write_bytes(original)
    destination = tmp_path / "objects.inv"

    monkeypatch.chdir(source.parent)
    copy_rebased_inventory_one_level_up(Path("objects.inv"))

    rebased = destination.read_bytes()
    assert rebased.startswith(_INVENTORY_HEADER)
    assert zlib.decompress(rebased[len(_INVENTORY_HEADER) :]) == (
        b"cuda.bindings.driver.cuInit py:function 1 latest/module/driver.html#$ -\n"
        b"figure 1 std:label -1 latest/overview.html#figure-1 Example with spaces\n"
        b"external std:label -1 https://example.org/overview.html#target -\n"
    )
    assert source.read_bytes() == original


@pytest.mark.agent_authored(model="gpt-6")
def test_copy_rejects_malformed_inventory_entries(tmp_path):
    entries = b"cuda.bindings.driver.cuInit py:function 1 module/driver.html#$ -\nunexpected entry\n"
    source = tmp_path / "latest" / "objects.inv"
    source.parent.mkdir()
    source.write_bytes(_INVENTORY_HEADER + zlib.compress(entries))

    with pytest.raises(ValueError, match="1 of 2 inventory entries did not match the expected format"):
        copy_rebased_inventory_one_level_up(source)

    assert not (tmp_path / "objects.inv").exists()
