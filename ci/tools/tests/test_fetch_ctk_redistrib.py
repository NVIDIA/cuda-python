# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import json
import os
import sys

import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from fetch_ctk_redistrib import main

# Shaped like a real redistrib_*.json: string-valued release keys sit at the
# top level alongside the component objects.
METADATA = {
    "release_date": "2026-01-01",
    "release_label": "13.0.0",
    "release_product": "cuda",
    "cuda_nvcc": {
        "linux-x86_64": {"relative_path": "cuda_nvcc/linux-x86_64/cuda_nvcc-linux-x86_64.tar.xz"},
    },
}


def write_metadata(tmp_path, payload):
    path = tmp_path / "redistrib.json"
    path.write_text(json.dumps(payload), encoding="utf-8")
    return str(path)


def relpath_argv(metadata_path, component):
    return [
        "component-relative-path",
        "--host-platform",
        "linux-64",
        "--component",
        component,
        "--metadata-path",
        metadata_path,
    ]


def filter_argv(metadata_path, components="cuda_nvcc"):
    return [
        "filter-components",
        "--host-platform",
        "linux-64",
        "--cuda-version",
        "13.0.0",
        "--components",
        components,
        "--metadata-path",
        metadata_path,
    ]


@pytest.mark.agent_authored(model="claude-opus-5")
def test_valid_component_is_resolved(tmp_path, capsys):
    assert main(relpath_argv(write_metadata(tmp_path, METADATA), "cuda_nvcc")) == 0
    assert capsys.readouterr().out.strip() == "cuda_nvcc/linux-x86_64/cuda_nvcc-linux-x86_64.tar.xz"


@pytest.mark.agent_authored(model="claude-opus-5")
@pytest.mark.parametrize("component", ["release_label", "release_date", "release_product"])
def test_string_valued_top_level_key_is_not_a_component(tmp_path, capsys, component):
    """`is None` only rejects an absent key, not a wrongly-typed one.

    Every real manifest carries these string-valued keys next to the component
    objects, so asking for one used to reach `component_info.get(...)` and die
    with `AttributeError: 'str' object has no attribute 'get'` instead of the
    tool's own diagnostic.
    """
    assert main(relpath_argv(write_metadata(tmp_path, METADATA), component)) == 1
    assert "ERROR:" in capsys.readouterr().err


@pytest.mark.agent_authored(model="claude-opus-5")
def test_absent_component_still_reports_cleanly(tmp_path, capsys):
    assert main(relpath_argv(write_metadata(tmp_path, METADATA), "not_a_component")) == 1
    assert "unknown CTK component" in capsys.readouterr().err


@pytest.mark.agent_authored(model="claude-opus-5")
def test_non_object_subdir_entry_is_reported(tmp_path, capsys):
    metadata = {"cuda_nvcc": {"linux-x86_64": "cuda_nvcc/linux-x86_64/x.tar.xz"}}
    assert main(relpath_argv(write_metadata(tmp_path, metadata), "cuda_nvcc")) == 1
    assert "ERROR:" in capsys.readouterr().err


@pytest.mark.agent_authored(model="claude-opus-5")
@pytest.mark.parametrize(
    "payload",
    [
        pytest.param(None, id="null"),
        pytest.param([1, 2], id="array"),
        pytest.param("13.0.0", id="string"),
    ],
)
@pytest.mark.parametrize("argv_builder", [relpath_argv, filter_argv], ids=["relative-path", "filter"])
def test_metadata_that_is_not_an_object_is_reported(tmp_path, capsys, payload, argv_builder):
    """The manifest is downloaded with `curl -LSs` (no --fail), so an error
    page or redirect body can parse as valid JSON that is not an object."""
    path = write_metadata(tmp_path, payload)
    argv = argv_builder(path, "cuda_nvcc") if argv_builder is relpath_argv else argv_builder(path)

    assert main(argv) == 1
    assert "must be a JSON object" in capsys.readouterr().err


@pytest.mark.agent_authored(model="claude-opus-5")
def test_filter_skips_a_string_valued_top_level_key(tmp_path, capsys):
    assert main(filter_argv(write_metadata(tmp_path, METADATA), "release_label")) == 0
    captured = capsys.readouterr()
    assert captured.out.strip() == ""
    assert "Skipping unsupported CTK component 'release_label'" in captured.err
