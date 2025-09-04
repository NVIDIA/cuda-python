# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import os

import pytest

from cuda.pathfinder._utils.env_vars_for_include import (
    VNAMES,
    iter_env_vars_for_include_dirs,
)

# --- Fixtures ----------------------------------------------------------------


@pytest.fixture(autouse=True)
def _clean_env(monkeypatch):
    # Ensure a clean slate for all variables the helper may read
    for k in set(VNAMES) | {"CPATH", "C_INCLUDE_PATH", "CPLUS_INCLUDE_PATH", "INCLUDE"}:
        monkeypatch.delenv(k, raising=False)
    return


def _join(*parts: str) -> str:
    return os.pathsep.join(parts)


# --- Tests -------------------------------------------------------------------


def test_no_relevant_env_vars_yields_nothing():
    assert list(iter_env_vars_for_include_dirs()) == []


def test_cpath_splits_on_pathsep(monkeypatch):
    # Includes an empty element which should be ignored
    monkeypatch.setenv("CPATH", _join("/a/include", "", "/b/include"))
    out = list(iter_env_vars_for_include_dirs())
    assert out == ["/a/include", "/b/include"]


def test_c_include_path_splits_on_pathsep(monkeypatch):
    monkeypatch.setenv("C_INCLUDE_PATH", _join("/x", "/y"))
    out = list(iter_env_vars_for_include_dirs())
    # Order depends on VNAMES; on all platforms C_INCLUDE_PATH is included
    # but may come after CPATH (and INCLUDE on Windows).
    # Since only C_INCLUDE_PATH is set here, we get exactly those two.
    assert out == ["/x", "/y"]


def test_duplicates_are_not_deduplicated(monkeypatch):
    # Same directory appears across different vars; should be yielded twice
    monkeypatch.setenv("CPATH", _join("/shared", "/only-in-cpath"))
    monkeypatch.setenv("C_INCLUDE_PATH", _join("/shared", "/only-in-c-include"))
    out = list(iter_env_vars_for_include_dirs())
    expected = []
    # Build the expected list in VNAMES order so the test is platform-agnostic
    env_values = {
        "CPATH": ["/shared", "/only-in-cpath"],
        "C_INCLUDE_PATH": ["/shared", "/only-in-c-include"],
        "INCLUDE": [],  # may or may not be consulted, but it's unset here
    }
    for var in VNAMES:
        expected.extend(env_values.get(var, []))
    assert out == expected


def test_order_follows_vnames(monkeypatch):
    # Put distinctive values in each variable to verify overall ordering
    mapping = {
        "INCLUDE": ["W1", "W2"],  # only used on Windows
        "CPATH": ["P1", "P2"],
        "C_INCLUDE_PATH": ["C1", "C2"],
        "CPLUS_INCLUDE_PATH": ["CP1", "CP2"],
    }
    for var, vals in mapping.items():
        # Only set those that are actually referenced on this platform
        if var in VNAMES:
            monkeypatch.setenv(var, _join(*vals))

    out = list(iter_env_vars_for_include_dirs())

    expected = []
    for var in VNAMES:
        expected.extend(mapping.get(var, []))
    assert out == expected


def test_ignore_wholly_empty_values(monkeypatch):
    # Variable is set but contains only separators / empties
    monkeypatch.setenv("CPATH", _join("", ""))  # effectively empty
    assert list(iter_env_vars_for_include_dirs()) == []


def test_windows_include_behavior(monkeypatch):
    # This test is platform-agnostic by keying off VNAMES:
    # - On Windows, INCLUDE is honored and should appear in output first.
    # - On non-Windows, INCLUDE is ignored entirely.
    monkeypatch.setenv("INCLUDE", _join("W:/inc1", "W:/inc2"))
    out = list(iter_env_vars_for_include_dirs())

    if "INCLUDE" in VNAMES:
        assert out[:2] == ["W:/inc1", "W:/inc2"]
    else:
        # Non-Windows platforms should ignore INCLUDE
        assert "W:/inc1" not in out
        assert "W:/inc2" not in out
