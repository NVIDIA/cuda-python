# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import os
import sys

import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from check_mempool_hygiene import DEFAULT_TREES, main, violations_in


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
    assert all(tree.is_dir() for tree in DEFAULT_TREES)
    assert main([]) == 0


POOL_PROPS_CAPPED = [
    pytest.param("props = driver.MemPoolProps_v1()\nprops.max_size = POOL_SIZE\n", id="v2-max_size"),
    pytest.param("props = cuda.CUmemPoolProps()\nprops.maxSize = POOL_SIZE\n", id="driver-maxSize"),
    pytest.param("props = cudart.cudaMemPoolProps()\nprops.maxSize = POOL_SIZE\n", id="runtime-maxSize"),
    pytest.param("def f():\n    props = cuda.CUmemPoolProps_v1()\n    props.maxSize = POOL_SIZE\n", id="in-function"),
    # A field of an alloc-node params object is not a constructor call, so it is not checked.
    pytest.param("params.poolProps.allocType = TYPE\nparams.pool_props = other\n", id="graph-params-fields"),
    # Type-annotated assignment (ast.AnnAssign) must also be detected and treated as capped.
    pytest.param(
        "props: cuda.CUmemPoolProps = cuda.CUmemPoolProps()\nprops.maxSize = POOL_SIZE\n",
        id="annotated-capped",
    ),
    # Second construction caps correctly; the first was already capped.
    pytest.param(
        "props = cuda.CUmemPoolProps()\nprops.maxSize = POOL_SIZE\nprops = cuda.CUmemPoolProps()\nprops.maxSize = POOL_SIZE\n",
        id="two-constructions-both-capped",
    ),
]

POOL_PROPS_UNCAPPED = [
    pytest.param("props = driver.MemPoolProps_v1()\nprops.location = loc\n", id="v2"),
    pytest.param("props = cuda.CUmemPoolProps()\n", id="driver"),
    pytest.param("props = cudart.cudaMemPoolProps()\n", id="runtime"),
    pytest.param(
        "def f():\n    props = cuda.CUmemPoolProps()\n    other.maxSize = POOL_SIZE\n",
        id="other-object",
    ),
    # The cap must be set in the function that constructed the props.
    pytest.param(
        "def make():\n    return cuda.CUmemPoolProps()\n\n\ndef f():\n    props = cuda.CUmemPoolProps()\n    g(props)\n\n\ndef g(props):\n    props.maxSize = POOL_SIZE\n",
        id="capped-in-another-function",
    ),
    # Type-annotated assignment without a cap must also be reported (ast.AnnAssign, Finding 1).
    pytest.param("props: cuda.CUmemPoolProps = cuda.CUmemPoolProps()\n", id="annotated-uncapped"),
    # A cap on the first object does not cover a second construction of the same name (Finding 2).
    pytest.param(
        "props = cuda.CUmemPoolProps()\nprops.maxSize = POOL_SIZE\nprops = cuda.CUmemPoolProps()\n",
        id="reassignment-second-uncapped",
    ),
]


@pytest.mark.agent_authored(model="claude-sonnet-5.5")
@pytest.mark.parametrize("source", POOL_PROPS_CAPPED)
def test_capped_pool_props_are_not_reported(tmp_path, source):
    assert violations_in(write(tmp_path, source)) == []


@pytest.mark.agent_authored(model="claude-sonnet-5.5")
@pytest.mark.parametrize("source", POOL_PROPS_UNCAPPED)
def test_uncapped_pool_props_are_reported(tmp_path, source):
    assert violations_in(write(tmp_path, source))


@pytest.mark.agent_authored(model="claude-sonnet-5.5")
def test_pool_props_marker_opts_out(tmp_path):
    source = "# uncapped-pool-ok: pool is never created\nprops = cuda.CUmemPoolProps()\n"

    assert violations_in(write(tmp_path, source)) == []


@pytest.mark.agent_authored(model="claude-sonnet-5.5")
def test_pool_props_message_names_file_line_and_variable(tmp_path):
    path = write(tmp_path, "x = 1\nprops = driver.MemPoolProps_v1()\n")

    (violation,) = violations_in(path)

    assert violation.startswith(path.as_posix())
    assert ":2:" in violation
    assert "MemPoolProps_v1 assigned to 'props' without props.max_size / props.maxSize" in violation


@pytest.mark.agent_authored(model="claude-sonnet-5.5")
def test_main_reports_pool_props_with_their_own_message(tmp_path, capsys):
    path = write(tmp_path, "props = cuda.CUmemPoolProps()\n")

    assert main([str(path)]) == 1

    err = capsys.readouterr().err
    assert "pool properties created by tests must set max_size / maxSize" in err
    assert "must set max_size:" not in err


@pytest.mark.agent_authored(model="claude-sonnet-5.5")
def test_main_reports_both_rules_separately(tmp_path, capsys):
    path = write(
        tmp_path,
        "DeviceMemoryResource(dev, DeviceMemoryResourceOptions())\nprops = cuda.CUmemPoolProps()\n",
    )

    assert main([str(path)]) == 1

    err = capsys.readouterr().err
    assert "memory pools created by tests must set max_size:" in err
    assert "pool properties created by tests must set max_size / maxSize" in err


@pytest.mark.agent_authored(model="claude-sonnet-4.6")
def test_annotated_pool_props_assignment_reports_correct_line(tmp_path):
    # Finding 1: props: T = CUmemPoolProps() is ast.AnnAssign; it must be detected.
    path = write(tmp_path, "x = 1\nprops: cuda.CUmemPoolProps = cuda.CUmemPoolProps()\n")

    (violation,) = violations_in(path)

    assert ":2:" in violation


@pytest.mark.agent_authored(model="claude-sonnet-4.6")
def test_reassignment_reports_only_the_uncapped_construction(tmp_path):
    # Finding 2: a cap on the first object must not excuse a second construction
    # of the same variable.  Only the second (uncapped) object should be reported.
    source = (
        "props = cuda.CUmemPoolProps()\n"  # line 1
        "props.maxSize = POOL_SIZE\n"  # line 2 — caps line 1
        "props = cuda.CUmemPoolProps()\n"  # line 3 — new object, no cap
    )
    path = write(tmp_path, source)

    (violation,) = violations_in(path)

    assert ":3:" in violation
