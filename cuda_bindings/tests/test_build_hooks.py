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
            assert (cc, cxx) == ("gcc", "g++")
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
    def test_gnu_sets_env_and_flags(self, monkeypatch):
        if sys.platform == "win32":
            pytest.skip("gnu only valid on Linux")
        monkeypatch.setenv("CUDA_PYTHON_TOOLCHAIN", "gnu")
        monkeypatch.delenv("CC", raising=False)
        monkeypatch.delenv("CXX", raising=False)
        monkeypatch.delenv("LDSHARED", raising=False)
        name, cc, cxx, cargs, largs = build_hooks._resolve_toolchain()
        assert name == "gnu"
        assert (cc, cxx) == ("gcc", "g++")
        assert os.environ["CC"] == "gcc"
        assert os.environ["CXX"] == "g++"
        # gcc-only flags are present (this is the point of P2: explicit gnu must use gcc, not generic cc)
        assert "-fpermissive" in cargs
        assert "-fno-var-tracking-assignments" in cargs

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


@pytest.fixture
def stamp(tmp_path, monkeypatch):
    """Redirect the toolchain stamp to a scratch path."""
    scratch = tmp_path / "build" / ".build-toolchain"
    monkeypatch.setattr(build_hooks, "_BUILD_TOOLCHAIN_STAMP", scratch)
    monkeypatch.setattr(build_hooks, "force_build_ext", False)
    monkeypatch.delenv("CUDA_PYTHON_TOOLCHAIN", raising=False)
    return scratch


def _write_stamp(stamp, toolchain):
    stamp.parent.mkdir(parents=True, exist_ok=True)
    stamp.write_text(toolchain + "\n")


class TestBuildToolchainStamp:
    """Tests for _check_build_toolchain() and record_build_toolchain()."""

    @pytest.mark.agent_authored(model="grok-4.6")
    def test_stamp_path_is_scoped_to_extension_abi(self, monkeypatch):
        monkeypatch.setattr(build_hooks.sysconfig, "get_config_var", lambda _name: ".cpython-310-x86_64-linux-gnu.so")
        python_310 = build_hooks._abi_stamp_path(".build-toolchain")
        monkeypatch.setattr(build_hooks.sysconfig, "get_config_var", lambda _name: ".cpython-311-x86_64-linux-gnu.so")
        python_311 = build_hooks._abi_stamp_path(".build-toolchain")

        assert python_310 != python_311
        assert python_310.name == ".build-toolchain.cpython-310-x86_64-linux-gnu.so"
        assert python_311.name == ".build-toolchain.cpython-311-x86_64-linux-gnu.so"

    @pytest.mark.agent_authored(model="glm-5.2")
    def test_missing_stamp_forces_rebuild(self, stamp):
        build_hooks._check_build_toolchain("gnu")
        assert build_hooks.force_build_ext is True

    @pytest.mark.agent_authored(model="glm-5.2")
    def test_same_toolchain_does_not_force(self, stamp):
        _write_stamp(stamp, "gnu")
        build_hooks._check_build_toolchain("gnu")
        assert build_hooks.force_build_ext is False

    @pytest.mark.agent_authored(model="glm-5.2")
    def test_changed_toolchain_forces_rebuild(self, stamp):
        _write_stamp(stamp, "gnu")
        build_hooks._check_build_toolchain("llvm")
        assert build_hooks.force_build_ext is True

    @pytest.mark.agent_authored(model="glm-5.2")
    def test_record_writes_stamp(self, stamp):
        # record_build_toolchain re-derives from env (CUDA_PYTHON_TOOLCHAIN unset →
        # platform default: gnu on Linux, msvc on Windows).
        build_hooks.record_build_toolchain()
        expected = "msvc" if sys.platform == "win32" else "gnu"
        assert stamp.read_text().strip() == expected
