# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Package/cython markers must survive the way CI actually invokes pytest.

``ci/tools/run-tests`` runs ``pushd ./cuda_core && pytest tests/``. Each
subpackage ships its own pytest.ini, so rootdir becomes that subpackage and
every nodeid starts at ``tests/`` -- never ``cuda_core/tests/``. The node ids
below are taken verbatim from NVIDIA CI job logs.
"""

from __future__ import annotations

import os
import pathlib
import sys

import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from cuda_python_test_helpers import _pytest_plugin

REPO = pathlib.Path("/home/runner/work/cuda-python/cuda-python")


class FakeItem:
    """Minimal stand-in for a collected pytest item."""

    def __init__(self, relpath: str, nodeid: str):
        self.path = REPO / relpath
        self.nodeid = nodeid
        self.own_markers = []
        self.keywords = set()

    def add_marker(self, marker):
        self.own_markers.append(marker)
        self.keywords.add(marker.name)

    @property
    def marker_names(self):
        return {m.name for m in self.own_markers}


def collect(items, *, have_headers=True, monkeypatch=None):
    monkeypatch.setattr(_pytest_plugin, "_cuda_headers_available", lambda: have_headers)
    _pytest_plugin.pytest_collection_modifyitems(None, items)
    return items


# (relative path, nodeid as pytest -v prints it in CI, expected marker)
CI_ITEMS = [
    ("cuda_core/tests/test_memory.py", "tests/test_memory.py::test_buffer", "core"),
    ("cuda_core/tests/graph/test_graph_builder.py", "tests/graph/test_graph_builder.py::test_build", "core"),
    ("cuda_bindings/tests/test_cuda.py", "tests/test_cuda.py::test_x", "bindings"),
    ("cuda_pathfinder/tests/test_search_steps.py", "tests/test_search_steps.py::test_y", "pathfinder"),
]


@pytest.mark.agent_authored(model="claude-opus-5")
@pytest.mark.parametrize(("relpath", "nodeid", "expected"), CI_ITEMS)
def test_package_marker_applied_for_ci_node_ids(monkeypatch, relpath, nodeid, expected):
    (item,) = collect([FakeItem(relpath, nodeid)], monkeypatch=monkeypatch)

    assert expected in item.marker_names


@pytest.mark.agent_authored(model="claude-opus-5")
def test_package_marker_applied_for_repo_root_node_ids(monkeypatch):
    """The repo-root invocation must keep working too."""
    item = FakeItem("cuda_core/tests/test_memory.py", "cuda_core/tests/test_memory.py::test_buffer")

    (item,) = collect([item], monkeypatch=monkeypatch)

    assert "core" in item.marker_names


@pytest.mark.agent_authored(model="claude-opus-5")
def test_cython_marker_applied_for_ci_node_ids(monkeypatch):
    item = FakeItem("cuda_core/tests/cython/test_cython.py", "tests/cython/test_cython.py::test_ccuda_memcpy")

    (item,) = collect([item], monkeypatch=monkeypatch)

    assert {"core", "cython"} <= item.marker_names


@pytest.mark.agent_authored(model="claude-opus-5")
def test_core_cython_tests_are_skipped_without_cuda_headers(monkeypatch):
    """The gate this plugin exists for: no CUDA headers means no core cython."""
    item = FakeItem("cuda_core/tests/cython/test_cython.py", "tests/cython/test_cython.py::test_ccuda_memcpy")

    (item,) = collect([item], have_headers=False, monkeypatch=monkeypatch)

    assert "skip" in item.marker_names


@pytest.mark.agent_authored(model="claude-opus-5")
def test_core_cython_tests_run_when_headers_are_present(monkeypatch):
    item = FakeItem("cuda_core/tests/cython/test_cython.py", "tests/cython/test_cython.py::test_ccuda_memcpy")

    (item,) = collect([item], have_headers=True, monkeypatch=monkeypatch)

    assert "skip" not in item.marker_names


@pytest.mark.agent_authored(model="claude-opus-5")
def test_integration_tests_are_marked_smoke(monkeypatch):
    item = FakeItem("tests/integration/test_smoke.py", "tests/integration/test_smoke.py::test_z")

    (item,) = collect([item], monkeypatch=monkeypatch)

    assert "smoke" in item.marker_names


@pytest.mark.agent_authored(model="claude-opus-5")
def test_unrelated_paths_get_no_package_marker(monkeypatch):
    item = FakeItem("toolshed/tests/test_thing.py", "toolshed/tests/test_thing.py::test_w")

    (item,) = collect([item], monkeypatch=monkeypatch)

    assert item.marker_names & {"core", "bindings", "pathfinder"} == set()
