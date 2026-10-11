# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import os
import sys

import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from check_mempool_hygiene import DEFAULT_TREE, main, violations_in


def write(tmp_path, source):
    path = tmp_path / "test_sample.py"
    path.write_text(source, encoding="utf-8")
    return path


UNCAPPED = [
    pytest.param("DeviceMemoryResource(dev, DeviceMemoryResourceOptions(ipc_enabled=True))", id="options-kwarg"),
    pytest.param("PinnedMemoryResource(PinnedMemoryResourceOptions())", id="options-empty"),
    pytest.param('DeviceMemoryResource(dev, {"ipc_enabled": True})', id="options-dict"),
]

CAPPED = [
    pytest.param("DeviceMemoryResource(dev, DeviceMemoryResourceOptions(max_size=POOL_SIZE))", id="capped-kwarg"),
    pytest.param('DeviceMemoryResource(dev, {"max_size": POOL_SIZE})', id="capped-dict"),
    pytest.param("DeviceMemoryResource(dev, DeviceMemoryResourceOptions(**opts))", id="opaque-kwargs"),
    # No options at all wraps the device's default pool and reserves nothing, so
    # capping it would convert a free wrapper into a new pool.
    pytest.param("DeviceMemoryResource(dev)", id="default-pool-wrapper"),
    # cuMemPoolCreate requires maxSize == 0 for managed pools, so these have no
    # max_size to set.
    pytest.param("ManagedMemoryResource(ManagedMemoryResourceOptions(preferred_location=0))", id="managed-exempt"),
]


@pytest.mark.agent_authored(model="claude-opus-5")
@pytest.mark.parametrize("source", UNCAPPED)
def test_uncapped_pool_is_reported(tmp_path, source):
    assert violations_in(write(tmp_path, source))


@pytest.mark.agent_authored(model="claude-opus-5")
@pytest.mark.parametrize("source", CAPPED)
def test_acceptable_construction_is_not_reported(tmp_path, source):
    assert violations_in(write(tmp_path, source)) == []


@pytest.mark.agent_authored(model="claude-opus-5")
@pytest.mark.parametrize("comment_line", [0, 1], ids=["marker-above", "marker-inline"])
def test_marker_opts_a_call_out(tmp_path, comment_line):
    # The escape hatch exists mainly for pytest.raises cases, where validation
    # rejects the arguments before any pool is created.
    call = "PinnedMemoryResource(PinnedMemoryResourceOptions())"
    marker = "# uncapped-pool-ok: raises before the pool is created"
    source = f"{marker}\n{call}" if comment_line == 0 else f"{call}  {marker}"

    assert violations_in(write(tmp_path, source)) == []


@pytest.mark.agent_authored(model="claude-opus-5")
def test_reported_message_names_file_line_and_symbol(tmp_path):
    path = write(tmp_path, "x = 1\nDeviceMemoryResource(dev, DeviceMemoryResourceOptions())\n")

    (violation,) = violations_in(path)

    assert violation.startswith(path.as_posix())
    assert ":2:" in violation
    assert "DeviceMemoryResourceOptions without max_size" in violation


@pytest.mark.agent_authored(model="claude-opus-5")
def test_main_reports_failure_for_the_files_it_is_given(tmp_path, capsys):
    path = write(tmp_path, "DeviceMemoryResource(dev, DeviceMemoryResourceOptions())")

    assert main([str(path)]) == 1
    assert "must set max_size" in capsys.readouterr().err


@pytest.mark.agent_authored(model="claude-opus-5")
def test_main_ignores_non_python_files(tmp_path):
    unrelated = tmp_path / "notes.txt"
    unrelated.write_text("DeviceMemoryResourceOptions()", encoding="utf-8")

    assert main([str(unrelated)]) == 0


@pytest.mark.agent_authored(model="claude-opus-5")
def test_the_live_test_suite_is_clean():
    # Without a default the hook would only ever see changed files, so a
    # violation could ride in on a rename or a merge.
    assert DEFAULT_TREE.is_dir()
    assert main([]) == 0
