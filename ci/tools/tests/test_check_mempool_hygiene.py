# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import os
import sys
import textwrap

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


POOL = "DeviceMemoryResource(dev, DeviceMemoryResourceOptions(max_size=POOL_SIZE))"


def in_test(body, decorators="@pytest.mark.owns_pool\n"):
    """A test function around ``body``, marked by default so that a case exercises one rule at a time."""
    return f"{decorators}def test_sample(dev):\n" + textwrap.indent(body, "    ")


UNCLOSED = [
    pytest.param(f"mr = {POOL}\nassert mr\n", id="never-closed"),
    pytest.param(
        f"mr = {POOL}\npools.append(mr)\nfor pool in pools:\n    pool.close()\n", id="closed-under-another-name"
    ),
    pytest.param(
        "mr = create_pinned_memory_resource_or_xfail(PinnedMemoryResourceOptions(max_size=POOL_SIZE))\n",
        id="pinned-helper",
    ),
]

CLOSED = [
    pytest.param(f"mr = {POOL}\ntry:\n    assert mr\nfinally:\n    mr.close()\n", id="close-in-finally"),
    pytest.param(f"mrs = [{POOL} for _ in range(2)]\nfor mr in mrs:\n    mr.close()\n", id="list-closed-by-loop"),
    pytest.param(f"# unclosed-pool-ok: closed by the harness\nmr = {POOL}\n", id="opt-out"),
    pytest.param("mr = DeviceMemoryResource(dev)\n", id="default-pool-wrapper"),
    pytest.param("mr = ManagedMemoryResource(ManagedMemoryResourceOptions())\n", id="managed-outside-the-close-rule"),
]


@pytest.mark.agent_authored(model="claude-fable-5-1")
@pytest.mark.parametrize("body", UNCLOSED)
def test_unclosed_pool_is_reported(tmp_path, body):
    (violation,) = violations_in(write(tmp_path, in_test(body)))

    assert violation.endswith("in test_sample is not closed")


@pytest.mark.agent_authored(model="claude-fable-5-1")
@pytest.mark.parametrize("body", CLOSED)
def test_closed_pool_is_not_reported(tmp_path, body):
    assert violations_in(write(tmp_path, in_test(body))) == []


@pytest.mark.agent_authored(model="claude-fable-5-1")
def test_uncapped_opt_out_also_satisfies_the_close_rule(tmp_path):
    # The annotation marks a call that raises before any pool exists, so there is nothing to close.
    body = (
        "with pytest.raises(ValueError):\n"
        "    # uncapped-pool-ok: numa_id is validated before the pool is created\n"
        "    PinnedMemoryResource(PinnedMemoryResourceOptions(numa_id=-1))\n"
    )

    assert violations_in(write(tmp_path, in_test(body))) == []


CREATES_POOL = [
    pytest.param(f"mr = {POOL}\nmr.close()\n", id="device-options"),
    pytest.param('mr = DeviceMemoryResource(dev, {"max_size": POOL_SIZE})\nmr.close()\n', id="device-dict-options"),
    pytest.param("mr = create_managed_memory_resource_or_skip(ManagedMemoryResourceOptions())\n", id="managed-helper"),
    pytest.param(
        'mr = create_pinned_memory_resource_or_xfail(options={"max_size": POOL_SIZE}, xfail_device=dev)\nmr.close()\n',
        id="pinned-helper",
    ),
    pytest.param(
        'mr = VirtualMemoryResource(dev, config=VirtualMemoryResourceOptions(handle_type="posix_fd"))\n',
        id="virtual-config",
    ),
]

NO_POOL = [
    pytest.param("mr = DeviceMemoryResource(dev)\n", id="default-pool-wrapper"),
    pytest.param("mr = PinnedMemoryResource()\n", id="pinned-current-pool"),
    pytest.param("mr = create_managed_memory_resource_or_skip()\n", id="managed-current-pool"),
    pytest.param("mr = create_pinned_memory_resource_or_xfail(xfail_device=dev)\n", id="pinned-helper-current-pool"),
]


@pytest.mark.agent_authored(model="claude-fable-5-1")
@pytest.mark.parametrize("body", CREATES_POOL)
def test_test_that_creates_a_pool_needs_the_marker(tmp_path, body):
    (violation,) = violations_in(write(tmp_path, in_test(body, decorators="")))

    assert violation.endswith("test_sample creates a memory pool but is not marked owns_pool")
    assert violations_in(write(tmp_path, in_test(body))) == []


@pytest.mark.agent_authored(model="claude-fable-5-1")
@pytest.mark.parametrize("body", NO_POOL)
def test_test_without_an_owned_pool_needs_no_marker(tmp_path, body):
    assert violations_in(write(tmp_path, in_test(body, decorators=""))) == []


MARK_PLACEMENTS = [
    pytest.param(
        "@pytest.mark.owns_pool\nclass TestSample:\n"
        f"    def test_sample(self, dev):\n        mr = {POOL}\n        mr.close()\n",
        id="class",
    ),
    pytest.param(
        "pytestmark = [pytest.mark.thread_unsafe, pytest.mark.owns_pool]\n\n\n"
        f"def test_sample(dev):\n    mr = {POOL}\n    mr.close()\n",
        id="module",
    ),
]


@pytest.mark.agent_authored(model="claude-fable-5-1")
@pytest.mark.parametrize("source", MARK_PLACEMENTS)
def test_marker_on_the_class_or_module_counts(tmp_path, source):
    assert violations_in(write(tmp_path, source)) == []


HELPER = "create_managed_memory_resource_or_skip(ManagedMemoryResourceOptions())"
HELPER_CALLS = [
    pytest.param(
        f"def _pool(dev):\n    return {HELPER}\n\n\n{{mark}}def test_sample(dev):\n    mr = _pool(dev)\n",
        "@pytest.mark.owns_pool\n",
        id="module-function",
    ),
    pytest.param(
        f"class TestSample:\n    def _pool(self, dev):\n        return {HELPER}\n\n"
        "    {mark}def test_sample(self, dev):\n        mr = self._pool(dev)\n",
        "@pytest.mark.owns_pool\n    ",
        id="method-through-self",
    ),
]


@pytest.mark.agent_authored(model="claude-fable-5-1")
@pytest.mark.parametrize(("source", "mark"), HELPER_CALLS)
def test_pool_created_through_a_module_helper_needs_the_marker(tmp_path, source, mark):
    (violation,) = violations_in(write(tmp_path, source.format(mark="")))

    assert violation.endswith("test_sample creates a memory pool but is not marked owns_pool")
    assert violations_in(write(tmp_path, source.format(mark=mark))) == []


@pytest.mark.agent_authored(model="claude-fable-5-1")
@pytest.mark.parametrize("adds_marker", [False, True], ids=["without-add_marker", "with-add_marker"])
def test_fixture_that_creates_a_pool_must_add_the_marker(tmp_path, adds_marker):
    marker_line = '    request.node.add_marker("owns_pool")\n' if adds_marker else ""
    source = f"@pytest.fixture\ndef pool(request, dev):\n{marker_line}    mr = {POOL}\n    yield mr\n    mr.close()\n"

    violations = violations_in(write(tmp_path, source))

    if adds_marker:
        assert violations == []
    else:
        (violation,) = violations
        assert violation.endswith(
            'fixture pool creates a memory pool but does not call request.node.add_marker("owns_pool")'
        )
