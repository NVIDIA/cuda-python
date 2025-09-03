# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import pytest

from cuda.pathfinder._utils.conda_env import (
    BUILD_STATES,
    CondaPrefix,
    get_conda_prefix,
)

# Auto-clean environment & cache before every test -----------------------------


@pytest.fixture(autouse=True)
def _clean_env_and_cache(monkeypatch):
    # Remove any possibly inherited variables from the test runner environment
    for k in ("CONDA_BUILD_STATE", "PREFIX", "CONDA_PREFIX"):
        monkeypatch.delenv(k, raising=False)
    # Clear the cached result between tests
    get_conda_prefix.cache_clear()
    return
    # (No teardown needed; monkeypatch auto-reverts)


# Tests -----------------------------------------------------------------------


def test_returns_none_when_no_relevant_env_vars():
    assert get_conda_prefix() is None


@pytest.mark.parametrize("state", BUILD_STATES)
def test_build_state_returns_prefix_when_present(state, monkeypatch, tmp_path):
    monkeypatch.setenv("CONDA_BUILD_STATE", state)
    monkeypatch.setenv("PREFIX", str(tmp_path))
    res = get_conda_prefix()
    assert isinstance(res, CondaPrefix)
    assert res.env_state == state
    assert res.path == tmp_path


@pytest.mark.parametrize("state", BUILD_STATES)
def test_build_state_requires_prefix_otherwise_none(state, monkeypatch):
    monkeypatch.setenv("CONDA_BUILD_STATE", state)
    # No PREFIX set
    assert get_conda_prefix() is None


@pytest.mark.parametrize("state", BUILD_STATES)
def test_build_state_with_empty_prefix_returns_none(state, monkeypatch):
    monkeypatch.setenv("CONDA_BUILD_STATE", state)
    monkeypatch.setenv("PREFIX", "")
    assert get_conda_prefix() is None


def test_activated_env_returns_conda_prefix(monkeypatch, tmp_path):
    monkeypatch.setenv("CONDA_PREFIX", str(tmp_path))
    res = get_conda_prefix()
    assert isinstance(res, CondaPrefix)
    assert res.env_state == "activated"
    assert res.path == tmp_path


def test_activated_env_ignores_empty_conda_prefix(monkeypatch):
    monkeypatch.setenv("CONDA_PREFIX", "")
    assert get_conda_prefix() is None


def test_build_state_wins_over_activated_when_valid(monkeypatch, tmp_path):
    build_p = tmp_path / "host"
    user_p = tmp_path / "user"
    monkeypatch.setenv("CONDA_BUILD_STATE", "TEST")
    monkeypatch.setenv("PREFIX", str(build_p))
    monkeypatch.setenv("CONDA_PREFIX", str(user_p))
    res = get_conda_prefix()
    assert res
    assert res.env_state == "TEST"
    assert res.path == build_p


def test_unknown_build_state_returns_none_even_if_conda_prefix_set(monkeypatch, tmp_path):
    # Any non-empty CONDA_BUILD_STATE that is not recognized -> None
    monkeypatch.setenv("CONDA_BUILD_STATE", "SOMETHING_ELSE")
    monkeypatch.setenv("CONDA_PREFIX", str(tmp_path))
    assert get_conda_prefix() is None


def test_empty_build_state_treated_as_absent_and_falls_back_to_activated(monkeypatch, tmp_path):
    # Empty string is falsy -> treated like "not set" -> activated path
    monkeypatch.setenv("CONDA_BUILD_STATE", "")
    monkeypatch.setenv("CONDA_PREFIX", str(tmp_path))
    res = get_conda_prefix()
    assert res
    assert res.env_state == "activated"
    assert res.path == tmp_path


def test_have_cache(monkeypatch, tmp_path):
    monkeypatch.setenv("CONDA_PREFIX", str(tmp_path))
    res = get_conda_prefix()
    assert res
    assert res.path == tmp_path
    res2 = get_conda_prefix()
    assert res2 is res
