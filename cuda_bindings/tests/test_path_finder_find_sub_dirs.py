# Copyright 2025 NVIDIA Corporation.  All rights reserved.
# SPDX-License-Identifier: LicenseRef-NVIDIA-SOFTWARE-LICENSE

import os

import pytest

from cuda.bindings._path_finder.find_sub_dirs import (
    find_sub_dirs,
    find_sub_dirs_all_sitepackages,
    find_sub_dirs_sys_path,
)

NONEXISTENT = "NonExistentE12DBF1Fbe948337576B5F1E88f60bb2"


@pytest.fixture
def test_tree(tmp_path):
    # Build:
    # tmp_path/
    #   sys1/nvidia/foo/lib
    #   sys1/nvidia/bar/lib
    #   sys2/nvidia/baz/nvvm/lib64
    base = tmp_path
    (base / "sys1" / "nvidia" / "foo" / "lib").mkdir(parents=True)
    (base / "sys1" / "nvidia" / "bar" / "lib").mkdir(parents=True)
    (base / "sys2" / "nvidia" / "baz" / "nvvm" / "lib64").mkdir(parents=True)

    return {
        "parent_paths": (
            str(base / "sys1"),
            str(base / "sys2"),
            str(base / NONEXISTENT),
        ),
        "base": base,
    }


def test_exact_match(test_tree):
    parent_paths = test_tree["parent_paths"]
    base = test_tree["base"]
    result = find_sub_dirs(parent_paths, ("nvidia", "foo", "lib"))
    expected = [str(base / "sys1" / "nvidia" / "foo" / "lib")]
    assert result == expected


def test_single_wildcard(test_tree):
    parent_paths = test_tree["parent_paths"]
    base = test_tree["base"]
    result = find_sub_dirs(parent_paths, ("nvidia", "*", "lib"))
    expected = [
        str(base / "sys1" / "nvidia" / "bar" / "lib"),
        str(base / "sys1" / "nvidia" / "foo" / "lib"),
    ]
    assert sorted(result) == sorted(expected)


def test_double_wildcard(test_tree):
    parent_paths = test_tree["parent_paths"]
    base = test_tree["base"]
    result = find_sub_dirs(parent_paths, ("nvidia", "*", "nvvm", "lib64"))
    expected = [str(base / "sys2" / "nvidia" / "baz" / "nvvm" / "lib64")]
    assert result == expected


def test_no_match(test_tree):
    parent_paths = test_tree["parent_paths"]
    result = find_sub_dirs(parent_paths, (NONEXISTENT,))
    assert result == []


def test_empty_parent_paths():
    result = find_sub_dirs((), ("nvidia", "*", "lib"))
    assert result == []


def test_empty_sub_dirs(test_tree):
    parent_paths = test_tree["parent_paths"]
    result = find_sub_dirs(parent_paths, ())
    expected = [p for p in parent_paths if os.path.isdir(p)]
    assert sorted(result) == sorted(expected)


def test_find_sub_dirs_sys_path_no_math():
    result = find_sub_dirs_sys_path((NONEXISTENT,))
    assert result == []


def test_find_sub_dirs_all_sitepackages_no_match():
    result = find_sub_dirs_all_sitepackages((NONEXISTENT,))
    assert result == []
