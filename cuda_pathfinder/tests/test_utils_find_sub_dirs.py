# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import os

import pytest

from cuda.pathfinder._utils.find_sub_dirs import (
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


def test_find_sub_dirs_all_sitepackages_venv_order(mocker, tmp_path):
    """Test that in a venv with --system-site-packages, search order is: venv, user, system.

    This test verifies fix for issue #1716: user-site-packages should come after
    venv site-packages but before system site-packages.
    """
    # Create test directories
    venv_site = tmp_path / "venv" / "lib" / "python3.12" / "site-packages"
    user_site = tmp_path / "user" / ".local" / "lib" / "python3.12" / "site-packages"
    system_site = tmp_path / "system" / "lib" / "python3.12" / "dist-packages"

    venv_site.mkdir(parents=True)
    user_site.mkdir(parents=True)
    system_site.mkdir(parents=True)

    # Create a test subdirectory in each
    test_subdir = ("nvidia", "cuda_runtime", "lib")
    (venv_site / "nvidia" / "cuda_runtime" / "lib").mkdir(parents=True)
    (user_site / "nvidia" / "cuda_runtime" / "lib").mkdir(parents=True)
    (system_site / "nvidia" / "cuda_runtime" / "lib").mkdir(parents=True)

    # Mock site.getsitepackages() to return venv first, then system
    mocker.patch(
        "cuda.pathfinder._utils.find_sub_dirs.site.getsitepackages",
        return_value=[str(venv_site), str(system_site)],
    )
    # Mock user site-packages
    mocker.patch(
        "cuda.pathfinder._utils.find_sub_dirs.site.getusersitepackages",
        return_value=str(user_site),
    )
    mocker.patch(
        "cuda.pathfinder._utils.find_sub_dirs.site.ENABLE_USER_SITE",
        True,
    )
    # Mock sys.prefix != sys.base_prefix to simulate venv
    mocker.patch(
        "cuda.pathfinder._utils.find_sub_dirs.sys.prefix",
        str(tmp_path / "venv"),
    )
    mocker.patch(
        "cuda.pathfinder._utils.find_sub_dirs.sys.base_prefix",
        str(tmp_path / "system"),
    )

    # Clear cache to ensure mocks take effect
    from cuda.pathfinder._utils.find_sub_dirs import find_sub_dirs_cached

    find_sub_dirs_cached.cache_clear()

    result = find_sub_dirs_all_sitepackages(test_subdir)

    # Verify order: venv should come first, then user, then system
    assert len(result) == 3
    assert result[0] == str(venv_site / "nvidia" / "cuda_runtime" / "lib")
    assert result[1] == str(user_site / "nvidia" / "cuda_runtime" / "lib")
    assert result[2] == str(system_site / "nvidia" / "cuda_runtime" / "lib")


def test_find_sub_dirs_all_sitepackages_non_venv_order(mocker, tmp_path):
    """Test that outside a venv, search order is: user, system.

    This verifies PEP 370 behavior: user-site-packages should come before
    system site-packages when not in a venv.
    """
    # Create test directories
    user_site = tmp_path / "user" / ".local" / "lib" / "python3.12" / "site-packages"
    system_site = tmp_path / "system" / "lib" / "python3.12" / "dist-packages"

    user_site.mkdir(parents=True)
    system_site.mkdir(parents=True)

    # Create a test subdirectory in each
    test_subdir = ("nvidia", "cuda_runtime", "lib")
    (user_site / "nvidia" / "cuda_runtime" / "lib").mkdir(parents=True)
    (system_site / "nvidia" / "cuda_runtime" / "lib").mkdir(parents=True)

    # Mock site.getsitepackages() to return only system site-packages
    mocker.patch(
        "cuda.pathfinder._utils.find_sub_dirs.site.getsitepackages",
        return_value=[str(system_site)],
    )
    # Mock user site-packages
    mocker.patch(
        "cuda.pathfinder._utils.find_sub_dirs.site.getusersitepackages",
        return_value=str(user_site),
    )
    mocker.patch(
        "cuda.pathfinder._utils.find_sub_dirs.site.ENABLE_USER_SITE",
        True,
    )
    # Mock sys.prefix == sys.base_prefix to simulate non-venv
    mocker.patch(
        "cuda.pathfinder._utils.find_sub_dirs.sys.prefix",
        str(tmp_path / "system"),
    )
    mocker.patch(
        "cuda.pathfinder._utils.find_sub_dirs.sys.base_prefix",
        str(tmp_path / "system"),
    )

    # Clear cache to ensure mocks take effect
    from cuda.pathfinder._utils.find_sub_dirs import find_sub_dirs_cached

    find_sub_dirs_cached.cache_clear()

    result = find_sub_dirs_all_sitepackages(test_subdir)

    # Verify order: user should come first, then system
    assert len(result) == 2
    assert result[0] == str(user_site / "nvidia" / "cuda_runtime" / "lib")
    assert result[1] == str(system_site / "nvidia" / "cuda_runtime" / "lib")


def test_find_sub_dirs_all_sitepackages_venv_first_match(mocker, tmp_path):
    """Test that in a venv, venv site-packages is searched first (fixes #1716).

    This test ensures that when a file exists in both venv and user site-packages,
    the venv version is found first, matching Python's import behavior.
    """
    # Create test directories
    venv_site = tmp_path / "venv" / "lib" / "python3.12" / "site-packages"
    user_site = tmp_path / "user" / ".local" / "lib" / "python3.12" / "site-packages"

    venv_site.mkdir(parents=True)
    user_site.mkdir(parents=True)

    # Create the same subdirectory in both
    test_subdir = ("nvidia", "cuda_runtime", "lib")
    (venv_site / "nvidia" / "cuda_runtime" / "lib").mkdir(parents=True)
    (user_site / "nvidia" / "cuda_runtime" / "lib").mkdir(parents=True)

    # Mock site.getsitepackages() to return venv first
    mocker.patch(
        "cuda.pathfinder._utils.find_sub_dirs.site.getsitepackages",
        return_value=[str(venv_site)],
    )
    # Mock user site-packages
    mocker.patch(
        "cuda.pathfinder._utils.find_sub_dirs.site.getusersitepackages",
        return_value=str(user_site),
    )
    mocker.patch(
        "cuda.pathfinder._utils.find_sub_dirs.site.ENABLE_USER_SITE",
        True,
    )
    # Mock sys.prefix != sys.base_prefix to simulate venv
    mocker.patch(
        "cuda.pathfinder._utils.find_sub_dirs.sys.prefix",
        str(tmp_path / "venv"),
    )
    mocker.patch(
        "cuda.pathfinder._utils.find_sub_dirs.sys.base_prefix",
        str(tmp_path / "system"),
    )

    # Clear cache to ensure mocks take effect
    from cuda.pathfinder._utils.find_sub_dirs import find_sub_dirs_cached

    find_sub_dirs_cached.cache_clear()

    result = find_sub_dirs_all_sitepackages(test_subdir)

    # Verify venv comes first (this would fail with old code that puts user first)
    assert len(result) >= 2
    assert result[0] == str(venv_site / "nvidia" / "cuda_runtime" / "lib")
    assert result[1] == str(user_site / "nvidia" / "cuda_runtime" / "lib")
