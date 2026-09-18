# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

"""Tests for cuda_bindings/build_hooks.py build infrastructure.

Mirrors the toolchain tests in cuda_core/tests/test_build_hooks.py. These
tests do NOT require cuda.bindings to be built/installed since they test
build-time infrastructure. Run with --noconftest to avoid loading conftest.py
which imports cuda.bindings modules:

    pytest tests/test_build_hooks.py -v --noconftest

These tests require Cython to be installed (build_hooks.py imports it).
"""

import importlib.util
import os
import sys
from pathlib import Path

# build_hooks.py imports Cython and setuptools at the top level; both are
# declared test dependencies, so a missing install must surface as an
# ImportError at collection time rather than being hidden by importorskip.
import Cython  # noqa: F401
import pytest
import setuptools  # noqa: F401


def _load_build_hooks():
    """Load build_hooks module from source without polluting sys.path."""
    build_hooks_path = Path(__file__).parent.parent / "build_hooks.py"
    spec = importlib.util.spec_from_file_location("build_hooks", build_hooks_path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


build_hooks = _load_build_hooks()


@pytest.fixture(autouse=True)
def _clean_cc_env(monkeypatch):
    # _apply_toolchain_env sets CC/CXX/LDSHARED directly in os.environ (so
    # distutils picks them up), which monkeypatch does not revert because it
    # did not set them. Clean them per test so toolchain state never leaks
    # across tests.
    for k in ("CUDA_PYTHON_TOOLCHAIN", "CC", "CXX", "LDSHARED"):
        monkeypatch.delenv(k, raising=False)


class TestResolveToolchain:
    """_resolve_toolchain: pick compiler/linker/flags from CUDA_PYTHON_TOOLCHAIN.

    The default toolchain (gnu on Linux, msvc on Windows) must reproduce the
    previous build behavior exactly and must not touch CC/CXX/LDSHARED, so an
    externally-set compiler (e.g. the sccache wrapper in CI) survives.
    """

    @pytest.mark.agent_authored(model="glm-5.2")
    def test_default_does_not_touch_env(self, monkeypatch):
        monkeypatch.delenv("CUDA_PYTHON_TOOLCHAIN", raising=False)
        monkeypatch.delenv("CC", raising=False)
        monkeypatch.delenv("CXX", raising=False)
        monkeypatch.delenv("LDSHARED", raising=False)
        name, cc, cxx, _cargs, _largs = build_hooks._resolve_toolchain()
        if sys.platform == "win32":
            assert name == "msvc"
            assert cc is None and cxx is None
        else:
            assert name == "gnu"
            assert (cc, cxx) == ("cc", "c++")
        assert "CC" not in os.environ and "CXX" not in os.environ

    @pytest.mark.agent_authored(model="glm-5.2")
    def test_default_preserves_existing_cc(self, monkeypatch):
        # An externally-set CC (e.g. sccache) must survive the default toolchain.
        monkeypatch.delenv("CUDA_PYTHON_TOOLCHAIN", raising=False)
        monkeypatch.setenv("CC", "sccache cc")
        monkeypatch.setenv("CXX", "sccache c++")
        _name, _cc, _cxx, _cargs, _largs = build_hooks._resolve_toolchain()
        assert os.environ["CC"] == "sccache cc"
        assert os.environ["CXX"] == "sccache c++"

    @pytest.mark.agent_authored(model="glm-5.2")
    def test_case_insensitive(self, monkeypatch):
        if sys.platform == "win32":
            pytest.skip("llvm only valid on Linux")
        monkeypatch.setenv("CUDA_PYTHON_TOOLCHAIN", "LLVM")
        name, _cc, _cxx, _cargs, _largs = build_hooks._resolve_toolchain()
        assert name == "llvm"

    @pytest.mark.agent_authored(model="glm-5.2")
    def test_invalid_value_raises(self, monkeypatch):
        monkeypatch.setenv("CUDA_PYTHON_TOOLCHAIN", "icc")
        with pytest.raises(RuntimeError, match="not supported"):
            build_hooks._resolve_toolchain()

    @pytest.mark.agent_authored(model="glm-5.2")
    def test_llvm_sets_env_and_flags(self, monkeypatch):
        if sys.platform == "win32":
            pytest.skip("llvm only valid on Linux")
        monkeypatch.setenv("CUDA_PYTHON_TOOLCHAIN", "llvm")
        monkeypatch.delenv("CC", raising=False)
        monkeypatch.delenv("CXX", raising=False)
        monkeypatch.delenv("LDSHARED", raising=False)
        name, cc, cxx, cargs, largs = build_hooks._resolve_toolchain()
        assert name == "llvm"
        assert (cc, cxx) == ("clang", "clang++")
        assert os.environ["CC"] == "clang"
        assert os.environ["CXX"] == "clang++"
        assert "-fuse-ld=lld" in largs
        # clang rejects the gcc-only flags that gnu uses; they must be absent.
        assert "-fpermissive" not in cargs
        assert "-fno-var-tracking-assignments" not in cargs

    @pytest.mark.agent_authored(model="glm-5.2")
    def test_gnu_keeps_gcc_only_flags(self, monkeypatch):
        if sys.platform == "win32":
            pytest.skip("gnu only valid on Linux")
        monkeypatch.setenv("CUDA_PYTHON_TOOLCHAIN", "gnu")
        _name, _cc, _cxx, cargs, _largs = build_hooks._resolve_toolchain()
        assert "-fpermissive" in cargs
        assert "-fno-var-tracking-assignments" in cargs


class TestCheckToolchainAvailable:
    """_check_toolchain_available: fast, helpful failure when a tool is missing."""

    @pytest.mark.agent_authored(model="glm-5.2")
    def test_default_is_noop(self):
        build_hooks._check_toolchain_available("gnu")
        build_hooks._check_toolchain_available("msvc")

    @pytest.mark.agent_authored(model="glm-5.2")
    def test_llvm_missing_tool_lists_install_hint(self, monkeypatch):
        def fake_which(name):
            return None if name in ("clang", "clang++", "ld.lld") else "/bin/" + name

        monkeypatch.setattr(build_hooks.shutil, "which", fake_which)
        with pytest.raises(RuntimeError, match="clang and lld"):
            build_hooks._check_toolchain_available("llvm")

    @pytest.mark.agent_authored(model="glm-5.2")
    def test_llvm_present_passes(self, monkeypatch):
        monkeypatch.setattr(build_hooks.shutil, "which", lambda name: "/bin/" + name)
        build_hooks._check_toolchain_available("llvm")


class TestInferToolchain:
    """When CUDA_PYTHON_TOOLCHAIN is unset, infer the toolchain from CC/CXX.

    Regression for the externally-supplied-clang path (cuda.bindings): previously
    the default 'gnu' flag set (incl. -fno-var-tracking-assignments) reached clang because the default path did not override CC/CXX and the old _is_clang strip was removed. Now clang is inferred and the llvm flag set (no gcc-only flags, -fuse-ld=lld) is used, and the external compiler is left in place.
    """

    @pytest.mark.agent_authored(model="glm-5.2")
    def test_external_cc_clang_infers_llvm(self, monkeypatch):
        if sys.platform == "win32":
            pytest.skip("inference is Linux-only")
        monkeypatch.setenv("CC", "clang")
        name, _cc, _cxx, cargs, largs = build_hooks._resolve_toolchain()
        assert name == "llvm"
        # llvm flags: no gcc-only -fno-var-tracking-assignments, uses lld
        assert "-fno-var-tracking-assignments" not in cargs
        assert "-fpermissive" not in cargs  # cuda.bindings gnu keeps this; llvm drops it
        assert "-fuse-ld=lld" in largs
        # inferred path does not override the external compiler
        assert os.environ["CC"] == "clang"

    @pytest.mark.agent_authored(model="glm-5.2")
    def test_external_cxx_clang_infers_llvm(self, monkeypatch):
        if sys.platform == "win32":
            pytest.skip("inference is Linux-only")
        monkeypatch.setenv("CXX", "clang++")
        name, _cc, _cxx, _cargs, _largs = build_hooks._resolve_toolchain()
        assert name == "llvm"

    @pytest.mark.agent_authored(model="glm-5.2")
    def test_external_cc_gcc_infers_gnu(self, monkeypatch):
        if sys.platform == "win32":
            pytest.skip("inference is Linux-only")
        monkeypatch.setenv("CC", "gcc")
        name, _cc, _cxx, cargs, _largs = build_hooks._resolve_toolchain()
        assert name == "gnu"
        # gnu keeps the gcc-only flag (the regression was that clang got it)
        assert "-fno-var-tracking-assignments" in cargs
        assert os.environ["CC"] == "gcc"  # not overridden

    @pytest.mark.agent_authored(model="glm-5.2")
    def test_explicit_gnu_with_clang_cc_warns_and_overrides(self, monkeypatch):
        if sys.platform == "win32":
            pytest.skip("Linux-only")
        monkeypatch.setenv("CUDA_PYTHON_TOOLCHAIN", "gnu")
        monkeypatch.setenv("CC", "clang")
        with pytest.warns(UserWarning, match="takes precedence"):
            name, _cc, _cxx, _cargs, _largs = build_hooks._resolve_toolchain()
        assert name == "gnu"
        assert os.environ["CC"] == "cc"  # explicit toolchain governs

    @pytest.mark.agent_authored(model="glm-5.2")
    def test_explicit_llvm_with_clang_cc_no_warning(self, monkeypatch):
        if sys.platform == "win32":
            pytest.skip("Linux-only")
        monkeypatch.setenv("CUDA_PYTHON_TOOLCHAIN", "llvm")
        monkeypatch.setenv("CC", "clang")
        import warnings

        with warnings.catch_warnings():
            warnings.simplefilter("error")  # any warning here would fail the test
            name, _cc, _cxx, _cargs, _largs = build_hooks._resolve_toolchain()
        assert name == "llvm"
        assert os.environ["CC"] == "clang"  # overridden to clang (same family)
