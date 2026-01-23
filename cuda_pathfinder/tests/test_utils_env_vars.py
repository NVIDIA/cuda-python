# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import os
import pathlib
import sys
import warnings

import pytest

from cuda.pathfinder._utils.env_vars import (
    CUDA_ENV_VARS_ORDERED,
    _paths_differ,
    get_cuda_home_or_path,
)

skip_symlink_tests = pytest.mark.skipif(
    sys.platform == "win32",
    reason="Exercising symlinks intentionally omitted for simplicity",
)


def unset_env(monkeypatch):
    """Helper to clear both env vars for each test."""
    monkeypatch.delenv("CUDA_HOME", raising=False)
    monkeypatch.delenv("CUDA_PATH", raising=False)
    # Clear the cache so each test gets fresh behavior
    get_cuda_home_or_path.cache_clear()


def test_returns_none_when_unset(monkeypatch):
    unset_env(monkeypatch)
    assert get_cuda_home_or_path() is None


def test_empty_cuda_path_preserved(monkeypatch):
    # empty string is returned as-is if set.
    unset_env(monkeypatch)
    monkeypatch.setenv("CUDA_PATH", "")
    monkeypatch.setenv("CUDA_HOME", "/does/not/matter")
    assert get_cuda_home_or_path() == ""


def test_prefers_cuda_path_over_cuda_home(monkeypatch, tmp_path):
    unset_env(monkeypatch)
    home = tmp_path / "home"
    path = tmp_path / "path"
    home.mkdir()
    path.mkdir()

    monkeypatch.setenv("CUDA_HOME", str(home))
    monkeypatch.setenv("CUDA_PATH", str(path))

    # Different directories -> warning + prefer CUDA_PATH
    with pytest.warns(UserWarning, match="Multiple CUDA environment variables are set but differ"):
        result = get_cuda_home_or_path()
    assert pathlib.Path(result) == path


def test_uses_cuda_home_if_path_missing(monkeypatch, tmp_path):
    unset_env(monkeypatch)
    only_home = tmp_path / "home"
    only_home.mkdir()
    monkeypatch.setenv("CUDA_HOME", str(only_home))
    assert pathlib.Path(get_cuda_home_or_path()) == only_home


def test_no_warning_when_textually_equal_after_normalization(monkeypatch, tmp_path):
    """
    Trailing slashes should not trigger a warning, thanks to normpath.
    This works cross-platform.
    """
    unset_env(monkeypatch)
    d = tmp_path / "cuda"
    d.mkdir()

    with_slash = str(d) + ("/" if os.sep == "/" else "\\")
    monkeypatch.setenv("CUDA_PATH", str(d))
    monkeypatch.setenv("CUDA_HOME", with_slash)

    # No warning; same logical directory
    with warnings.catch_warnings(record=True) as record:
        result = get_cuda_home_or_path()
    assert pathlib.Path(result) == d
    assert len(record) == 0


@pytest.mark.skipif(sys.platform != "win32", reason="Windows-specific case-folding check")
def test_no_warning_on_windows_case_only_difference(monkeypatch, tmp_path):
    """
    On Windows, paths differing only by case should not warn because normcase collapses case.
    """
    unset_env(monkeypatch)
    d = tmp_path / "Cuda"
    d.mkdir()

    upper = str(d).upper()
    lower = str(d).lower()
    monkeypatch.setenv("CUDA_PATH", upper)
    monkeypatch.setenv("CUDA_HOME", lower)

    with warnings.catch_warnings(record=True) as record:
        warnings.simplefilter("always")
        result = get_cuda_home_or_path()
    assert pathlib.Path(result).samefile(d)
    assert len(record) == 0


def test_warning_when_both_exist_and_are_different(monkeypatch, tmp_path):
    unset_env(monkeypatch)
    a = tmp_path / "a"
    b = tmp_path / "b"
    a.mkdir()
    b.mkdir()

    monkeypatch.setenv("CUDA_PATH", str(a))
    monkeypatch.setenv("CUDA_HOME", str(b))

    # Different actual dirs -> warning
    with pytest.warns(UserWarning, match="Multiple CUDA environment variables are set but differ"):
        result = get_cuda_home_or_path()
    assert pathlib.Path(result) == a


def test_nonexistent_paths_fall_back_to_text_comparison(monkeypatch, tmp_path):
    """
    If one or both paths don't exist, we compare normalized strings.
    Different strings should warn.
    """
    unset_env(monkeypatch)
    a = tmp_path / "does_not_exist_a"
    b = tmp_path / "does_not_exist_b"

    monkeypatch.setenv("CUDA_PATH", str(a))
    monkeypatch.setenv("CUDA_HOME", str(b))

    with pytest.warns(UserWarning, match="Multiple CUDA environment variables are set but differ"):
        result = get_cuda_home_or_path()
    assert pathlib.Path(result) == a


@skip_symlink_tests
def test_samefile_equivalence_via_symlink_when_possible(monkeypatch, tmp_path):
    """
    If both paths exist and one is a symlink/junction to the other, we should NOT warn.
    """
    unset_env(monkeypatch)
    real_dir = tmp_path / "real"
    real_dir.mkdir()

    link_dir = tmp_path / "alias"

    os.symlink(str(real_dir), str(link_dir), target_is_directory=True)

    # Set env vars to real and alias
    monkeypatch.setenv("CUDA_PATH", str(real_dir))
    monkeypatch.setenv("CUDA_HOME", str(link_dir))

    # Because they resolve to the same entry, no warning should be raised
    with warnings.catch_warnings(record=True) as record:
        warnings.simplefilter("always")
        result = get_cuda_home_or_path()
    assert pathlib.Path(result) == real_dir
    assert len(record) == 0


def test_cuda_env_vars_ordered_constant():
    """
    Verify the canonical search order constant is defined correctly.
    CUDA_PATH must have higher priority than CUDA_HOME.
    """
    assert CUDA_ENV_VARS_ORDERED == ("CUDA_PATH", "CUDA_HOME")
    assert CUDA_ENV_VARS_ORDERED[0] == "CUDA_PATH"  # highest priority
    assert CUDA_ENV_VARS_ORDERED[1] == "CUDA_HOME"  # lower priority


def test_search_order_matches_implementation(monkeypatch, tmp_path):
    """
    Verify that get_cuda_home_or_path() follows the documented search order.
    """
    unset_env(monkeypatch)
    path_dir = tmp_path / "path_dir"
    home_dir = tmp_path / "home_dir"
    path_dir.mkdir()
    home_dir.mkdir()

    # Set both env vars to different values
    monkeypatch.setenv("CUDA_PATH", str(path_dir))
    monkeypatch.setenv("CUDA_HOME", str(home_dir))

    # The result should match the first (highest priority) variable in CUDA_ENV_VARS_ORDERED
    with warnings.catch_warnings(record=True):
        warnings.simplefilter("always")
        result = get_cuda_home_or_path()

    highest_priority_var = CUDA_ENV_VARS_ORDERED[0]
    expected = os.environ.get(highest_priority_var)
    assert result == expected
    assert pathlib.Path(result) == path_dir  # CUDA_PATH should win


# --- unit tests for the helper itself (optional but nice to have) ---


def test_paths_differ_text_only(tmp_path):
    a = tmp_path / "x"
    b = tmp_path / "x" / ".." / "x"  # normalizes to same
    assert _paths_differ(str(a), str(b)) is False

    a = tmp_path / "x"
    b = tmp_path / "y"
    assert _paths_differ(str(a), str(b)) is True


@skip_symlink_tests
def test_paths_differ_samefile(tmp_path):
    real_dir = tmp_path / "r"
    real_dir.mkdir()
    alias = tmp_path / "alias"
    os.symlink(str(real_dir), str(alias), target_is_directory=True)

    # Should detect equivalence via samefile
    assert _paths_differ(str(real_dir), str(alias)) is False


def test_caching_behavior(monkeypatch, tmp_path):
    """
    Verify that get_cuda_home_or_path() caches the result and returns the same
    value even if environment variables change after the first call.
    """
    unset_env(monkeypatch)

    first_dir = tmp_path / "first"
    second_dir = tmp_path / "second"
    first_dir.mkdir()
    second_dir.mkdir()

    # Set initial value
    monkeypatch.setenv("CUDA_PATH", str(first_dir))

    # First call should return first_dir
    result1 = get_cuda_home_or_path()
    assert pathlib.Path(result1) == first_dir

    # Change the environment variable
    monkeypatch.setenv("CUDA_PATH", str(second_dir))

    # Second call should still return first_dir (cached value)
    result2 = get_cuda_home_or_path()
    assert pathlib.Path(result2) == first_dir
    assert result1 == result2

    # After clearing cache, should get new value
    get_cuda_home_or_path.cache_clear()
    result3 = get_cuda_home_or_path()
    assert pathlib.Path(result3) == second_dir
