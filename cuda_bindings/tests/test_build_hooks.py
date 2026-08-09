# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Tests for the integer build knobs read by build_hooks.py.

These tests do NOT require cuda.bindings to be built/installed since they test
build-time infrastructure. Run with --noconftest to avoid loading conftest.py,
which imports cuda.bindings.driver:

    pytest tests/test_build_hooks.py -v --noconftest

Mirrors cuda_core/tests/test_build_hooks.py.
"""

import importlib.util
import warnings
from pathlib import Path

import pytest

# build_hooks.py imports setuptools at the top level, so skip if not available.
pytest.importorskip("setuptools")


def _load_build_hooks():
    """Load build_hooks module from source without permanently modifying sys.path.

    build_hooks.py is a PEP 517 build backend, not an installed module. Load it
    directly from source so sys.path never gains the cuda_bindings/ directory
    (which contains cuda/bindings/ sources that could shadow the installed
    package).
    """
    build_hooks_path = Path(__file__).parent.parent / "build_hooks.py"
    spec = importlib.util.spec_from_file_location("build_hooks", build_hooks_path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


build_hooks = _load_build_hooks()


class TestEnvInt:
    """Integer build knobs read from the environment."""

    @pytest.mark.agent_authored(model="claude-opus-5")
    @pytest.mark.parametrize(
        "raw",
        [
            pytest.param("", id="empty"),
            pytest.param(" ", id="space"),
            pytest.param("\t", id="tab"),
        ],
    )
    def test_empty_value_means_unset(self, monkeypatch, raw):
        """``CUDA_PYTHON_COVERAGE= pip install .`` neutralises the variable.

        A bare ``int("")`` raises while the PEP 517 backend is still importing,
        so the build dies before it starts with an anonymous
        ``invalid literal for int() with base 10: ''``.
        """
        monkeypatch.setenv("CUDA_PYTHON_COVERAGE", raw)
        assert build_hooks.env_int("CUDA_PYTHON_COVERAGE", 7) == 7

    @pytest.mark.agent_authored(model="claude-opus-5")
    def test_unset_value_uses_the_default(self, monkeypatch):
        monkeypatch.delenv("CUDA_PYTHON_COVERAGE", raising=False)
        assert build_hooks.env_int("CUDA_PYTHON_COVERAGE", 3) == 3

    @pytest.mark.agent_authored(model="claude-opus-5")
    @pytest.mark.parametrize(
        ("raw", "expected"),
        [("0", 0), ("1", 1), ("12", 12), (" 4 ", 4), ("-1", -1)],
    )
    def test_integer_value_is_parsed(self, monkeypatch, raw, expected):
        monkeypatch.setenv("CUDA_PYTHON_PARALLEL_LEVEL", raw)
        assert build_hooks.env_int("CUDA_PYTHON_PARALLEL_LEVEL", 99) == expected

    @pytest.mark.agent_authored(model="claude-opus-5")
    def test_non_integer_value_names_the_variable(self, monkeypatch):
        monkeypatch.setenv("CUDA_PYTHON_COVERAGE", "yes")
        with pytest.raises(ValueError, match="CUDA_PYTHON_COVERAGE='yes' must be an integer"):
            build_hooks.env_int("CUDA_PYTHON_COVERAGE", 0)


class TestParallelLevel:
    """PARALLEL_LEVEL is the deprecated spelling of CUDA_PYTHON_PARALLEL_LEVEL."""

    @pytest.mark.agent_authored(model="claude-opus-5")
    def test_neither_set(self, monkeypatch):
        monkeypatch.delenv("PARALLEL_LEVEL", raising=False)
        monkeypatch.delenv("CUDA_PYTHON_PARALLEL_LEVEL", raising=False)
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            assert build_hooks.parallel_level() == 0
        assert caught == []

    @pytest.mark.agent_authored(model="claude-opus-5")
    def test_current_variable_is_used(self, monkeypatch):
        monkeypatch.delenv("PARALLEL_LEVEL", raising=False)
        monkeypatch.setenv("CUDA_PYTHON_PARALLEL_LEVEL", "6")
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            assert build_hooks.parallel_level() == 6
        assert caught == []

    @pytest.mark.agent_authored(model="claude-opus-5")
    def test_deprecated_variable_still_works_and_warns(self, monkeypatch):
        monkeypatch.setenv("PARALLEL_LEVEL", "5")
        monkeypatch.delenv("CUDA_PYTHON_PARALLEL_LEVEL", raising=False)
        with pytest.warns(DeprecationWarning, match="PARALLEL_LEVEL is deprecated"):
            assert build_hooks.parallel_level() == 5

    @pytest.mark.agent_authored(model="claude-opus-5")
    def test_empty_deprecated_variable_does_not_abort_the_build(self, monkeypatch):
        """``PARALLEL_LEVEL=`` used to take the deprecated branch on ``is not
        None`` and then raise from ``int("")``, so neutralising the old
        variable aborted the build instead of falling through.
        """
        monkeypatch.setenv("PARALLEL_LEVEL", "")
        monkeypatch.setenv("CUDA_PYTHON_PARALLEL_LEVEL", "4")
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            assert build_hooks.parallel_level() == 4
        # An empty value is how a variable gets neutralised: warning about a
        # variable the caller is not using is noise.
        assert [w for w in caught if issubclass(w.category, DeprecationWarning)] == []

    @pytest.mark.agent_authored(model="claude-opus-5")
    def test_non_integer_deprecated_variable_names_the_variable(self, monkeypatch):
        monkeypatch.setenv("PARALLEL_LEVEL", "auto")
        with (
            pytest.warns(DeprecationWarning, match="PARALLEL_LEVEL is deprecated"),
            pytest.raises(ValueError, match="PARALLEL_LEVEL='auto' must be an integer"),
        ):
            build_hooks.parallel_level()
