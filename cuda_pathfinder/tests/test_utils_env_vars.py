# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import os
import pathlib
import sys
import warnings

import pytest

from cuda.pathfinder._utils.env_vars import _paths_differ, get_cuda_home_or_path

skip_symlink_tests = pytest.mark.skipif(
    sys.platform == "win32",
    reason="Exercising symlinks intentionally omitted for simplicity",
)


def unset_env(monkeypatch):
    """Helper to clear both env vars for each test."""
    monkeypatch.delenv("CUDA_HOME", raising=False)
    monkeypatch.delenv("CUDA_PATH", raising=False)


def test_returns_none_when_unset(monkeypatch):
    unset_env(monkeypatch)
    assert get_cuda_home_or_path() is None


def test_empty_cuda_home_preserved(monkeypatch):
    # empty string is returned as-is if set.
    monkeypatch.setenv("CUDA_HOME", "")
    monkeypatch.setenv("CUDA_PATH", "/does/not/matter")
    assert get_cuda_home_or_path() == ""


def test_prefers_cuda_home_over_cuda_path(monkeypatch, tmp_path):
    unset_env(monkeypatch)
    home = tmp_path / "home"
    path = tmp_path / "path"
    home.mkdir()
    path.mkdir()

    monkeypatch.setenv("CUDA_HOME", str(home))
    monkeypatch.setenv("CUDA_PATH", str(path))

    # Different directories -> warning + prefer CUDA_HOME
    with pytest.warns(UserWarning, match="Both CUDA_HOME and CUDA_PATH are set but differ"):
        result = get_cuda_home_or_path()
    assert pathlib.Path(result) == home


def test_uses_cuda_path_if_home_missing(monkeypatch, tmp_path):
    unset_env(monkeypatch)
    only_path = tmp_path / "path"
    only_path.mkdir()
    monkeypatch.setenv("CUDA_PATH", str(only_path))
    assert pathlib.Path(get_cuda_home_or_path()) == only_path


def test_no_warning_when_textually_equal_after_normalization(monkeypatch, tmp_path):
    """
    Trailing slashes should not trigger a warning, thanks to normpath.
    This works cross-platform.
    """
    unset_env(monkeypatch)
    d = tmp_path / "cuda"
    d.mkdir()

    with_slash = str(d) + ("/" if os.sep == "/" else "\\")
    monkeypatch.setenv("CUDA_HOME", str(d))
    monkeypatch.setenv("CUDA_PATH", with_slash)

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
    monkeypatch.setenv("CUDA_HOME", upper)
    monkeypatch.setenv("CUDA_PATH", lower)

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

    monkeypatch.setenv("CUDA_HOME", str(a))
    monkeypatch.setenv("CUDA_PATH", str(b))

    # Different actual dirs -> warning
    with pytest.warns(UserWarning, match="Both CUDA_HOME and CUDA_PATH are set but differ"):
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

    monkeypatch.setenv("CUDA_HOME", str(a))
    monkeypatch.setenv("CUDA_PATH", str(b))

    with pytest.warns(UserWarning, match="Both CUDA_HOME and CUDA_PATH are set but differ"):
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
    monkeypatch.setenv("CUDA_HOME", str(real_dir))
    monkeypatch.setenv("CUDA_PATH", str(link_dir))

    # Because they resolve to the same entry, no warning should be raised
    with warnings.catch_warnings(record=True) as record:
        warnings.simplefilter("always")
        result = get_cuda_home_or_path()
    assert pathlib.Path(result) == real_dir
    assert len(record) == 0


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
