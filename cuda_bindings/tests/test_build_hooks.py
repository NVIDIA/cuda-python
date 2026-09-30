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
def _isolate_toolchain_env():
    names = ("CUDA_PYTHON_TOOLCHAIN", "CC", "CXX", "LDSHARED", "LDCXXSHARED", "CUDA_PYTHON_CYTHON_CACHE_DIR")
    original = {name: os.environ[name] for name in names if name in os.environ}
    for name in names:
        os.environ.pop(name, None)
    try:
        yield
    finally:
        for name in names:
            os.environ.pop(name, None)
        os.environ.update(original)


class TestResolveToolchain:
    """_resolve_toolchain: pick compiler/linker/flags from CUDA_PYTHON_TOOLCHAIN.

    The default toolchain (gnu on Linux, msvc on Windows) must reproduce the
    previous build behavior exactly and must not touch CC/CXX/LDCXXSHARED, so an
    externally-set compiler (e.g. the sccache wrapper in CI) survives.
    """

    @pytest.mark.agent_authored(model="glm-5.2")
    def test_default_does_not_touch_env(self, monkeypatch):
        monkeypatch.delenv("CUDA_PYTHON_TOOLCHAIN", raising=False)
        monkeypatch.delenv("CC", raising=False)
        monkeypatch.delenv("CXX", raising=False)
        monkeypatch.delenv("LDSHARED", raising=False)
        monkeypatch.delenv("LDCXXSHARED", raising=False)
        name, cc, cxx, _cargs, _largs = build_hooks._resolve_toolchain()
        if sys.platform == "win32":
            assert name == "msvc"
            assert cc is None and cxx is None
        else:
            assert name == "gnu"
            assert (cc, cxx) == ("gcc", "g++")
        assert "CC" not in os.environ and "CXX" not in os.environ
        assert "LDCXXSHARED" not in os.environ

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
        assert os.environ["LDCXXSHARED"].startswith("clang++")
        assert "LDSHARED" not in os.environ
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
        assert os.environ["LDCXXSHARED"].startswith("g++")
        assert "LDSHARED" not in os.environ
        # gcc-only flags are present (this is the point of P2: explicit gnu must use gcc, not generic cc)
        assert "-fpermissive" in cargs
        assert "-fno-var-tracking-assignments" in cargs

    @pytest.mark.agent_authored(model="grok-4.6")
    def test_llvm_keeps_sccache_prefix(self, monkeypatch):
        if sys.platform == "win32":
            pytest.skip("llvm only valid on Linux")
        monkeypatch.setenv("CUDA_PYTHON_TOOLCHAIN", "llvm")
        monkeypatch.setenv("CC", "sccache cc")
        monkeypatch.setenv("CXX", "sccache c++")
        _name, _cc, _cxx, _cargs, _largs = build_hooks._resolve_toolchain()
        assert os.environ["CC"] == "sccache clang"
        assert os.environ["CXX"] == "sccache clang++"
        assert os.environ["LDSHARED"] == "sccache clang++ -shared"

    @pytest.mark.agent_authored(model="glm-5.2")
    def test_gnu_keeps_gcc_only_flags(self, monkeypatch):
        if sys.platform == "win32":
            pytest.skip("gnu only valid on Linux")
        monkeypatch.setenv("CUDA_PYTHON_TOOLCHAIN", "gnu")
        _name, _cc, _cxx, cargs, _largs = build_hooks._resolve_toolchain()
        assert "-fpermissive" in cargs
        assert "-fno-var-tracking-assignments" in cargs

    @pytest.mark.agent_authored(model="grok-4.6")
    def test_llvm_keeps_sccache_prefix(self, monkeypatch):
        if sys.platform == "win32":
            pytest.skip("llvm only valid on Linux")
        monkeypatch.setenv("CUDA_PYTHON_TOOLCHAIN", "llvm")
        monkeypatch.setenv("CC", "sccache cc")
        monkeypatch.setenv("CXX", "sccache c++")
        _name, _cc, _cxx, _cargs, _largs = build_hooks._resolve_toolchain()
        assert os.environ["CC"] == "sccache clang"
        assert os.environ["CXX"] == "sccache clang++"
        assert "LDSHARED" not in os.environ
        assert os.environ["LDCXXSHARED"].startswith("clang++")

    @pytest.mark.agent_authored(model="grok-4.6")
    def test_llvm_keeps_host_sccache_path(self, monkeypatch):
        if sys.platform == "win32":
            pytest.skip("llvm only valid on Linux")
        launcher = "/host/usr/local/bin/sccache"
        monkeypatch.setenv("CUDA_PYTHON_TOOLCHAIN", "llvm")
        monkeypatch.setenv("CC", f"{launcher} cc")
        monkeypatch.setenv("CXX", f"{launcher} c++")
        _name, _cc, _cxx, _cargs, _largs = build_hooks._resolve_toolchain()
        assert os.environ["CC"] == f"{launcher} clang"
        assert os.environ["CXX"] == f"{launcher} clang++"

    @pytest.mark.agent_authored(model="grok-4.6")
    def test_explicit_toolchain_preserves_sysconfig_ldcxxshared_extras(self, monkeypatch):
        if sys.platform == "win32":
            pytest.skip("gnu/llvm only valid on Linux")
        extras = "-shared -Wl,-O1 -Wl,-Bsymbolic-functions"
        values = {
            "CC": "gcc",
            "CXX": "g++",
            "LDSHARED": f"gcc {extras}",
            "LDCXXSHARED": f"g++ {extras}",
        }
        monkeypatch.setattr(build_hooks.sysconfig, "get_config_var", lambda name: values.get(name))
        monkeypatch.setenv("CUDA_PYTHON_TOOLCHAIN", "llvm")
        build_hooks._resolve_toolchain()
        assert os.environ["CC"] == "clang"
        assert os.environ["CXX"] == "clang++"
        assert os.environ["LDCXXSHARED"] == f"clang++ {extras}"
        assert "LDSHARED" not in os.environ

    @pytest.mark.agent_authored(model="grok-4.6")
    def test_explicit_gnu_keeps_conda_sysconfig_flags(self, monkeypatch):
        if sys.platform == "win32":
            pytest.skip("gnu/llvm only valid on Linux")
        prefix = "-pthread -B /compat"
        extras = f"{prefix} -shared -Wl,-rpath,/lib"
        values = {
            "CC": f"gcc {prefix}",
            "CXX": f"g++ {prefix}",
            "LDSHARED": f"gcc {extras}",
            "LDCXXSHARED": f"g++ {extras}",
        }
        monkeypatch.setattr(build_hooks.sysconfig, "get_config_var", lambda name: values.get(name))
        monkeypatch.setenv("CUDA_PYTHON_TOOLCHAIN", "gnu")
        build_hooks._resolve_toolchain()
        assert os.environ["CC"] == "gcc"
        assert os.environ["CXX"] == "g++"
        assert os.environ["LDCXXSHARED"] == values["LDCXXSHARED"]
        assert "LDSHARED" not in os.environ

    @pytest.mark.agent_authored(model="grok-4.6")
    def test_explicit_llvm_keeps_conda_sysconfig_flags(self, monkeypatch):
        if sys.platform == "win32":
            pytest.skip("gnu/llvm only valid on Linux")
        prefix = "-pthread -B /compat"
        extras = f"{prefix} -shared -Wl,-rpath,/lib"
        values = {
            "CC": f"gcc {prefix}",
            "CXX": f"g++ {prefix}",
            "LDSHARED": f"gcc {extras}",
            "LDCXXSHARED": f"g++ {extras}",
        }
        monkeypatch.setattr(build_hooks.sysconfig, "get_config_var", lambda name: values.get(name))
        monkeypatch.setenv("CUDA_PYTHON_TOOLCHAIN", "llvm")
        build_hooks._resolve_toolchain()
        assert os.environ["CC"] == "clang"
        assert os.environ["CXX"] == "clang++"
        assert os.environ["LDCXXSHARED"] == f"clang++ {extras}"
        assert "LDSHARED" not in os.environ


class TestWithSccache:
    """_with_sccache: keep a leading sccache token, swap the compiler."""

    @pytest.mark.agent_authored(model="grok-4.6")
    def test_keeps_sccache_and_swaps_compiler(self):
        assert build_hooks._with_sccache("sccache cc", "clang") == "sccache clang"

    @pytest.mark.agent_authored(model="grok-4.6")
    def test_keeps_absolute_sccache_path(self):
        assert (
            build_hooks._with_sccache("/host/usr/local/bin/sccache cc", "clang") == "/host/usr/local/bin/sccache clang"
        )

    @pytest.mark.agent_authored(model="grok-4.6")
    def test_bare_or_unrelated_cc_returns_compiler(self):
        assert build_hooks._with_sccache("", "clang") == "clang"
        assert build_hooks._with_sccache("gcc", "clang") == "clang"
        assert build_hooks._with_sccache("ccache gcc", "clang") == "clang"


class TestWithCompiler:
    """_with_compiler: replace the compiler executable, keep following flags."""

    @pytest.mark.agent_authored(model="grok-4.6")
    def test_replaces_leading_compiler_keeps_flags(self):
        assert build_hooks._with_compiler("g++ -shared -Wl,-O1", "clang++") == "clang++ -shared -Wl,-O1"

    @pytest.mark.agent_authored(model="grok-4.6")
    def test_keeps_flags_that_were_part_of_sysconfig_cxx(self):
        assert (
            build_hooks._with_compiler("g++ -pthread -B /compat -shared -Wl,-rpath,/lib", "clang++")
            == "clang++ -pthread -B /compat -shared -Wl,-rpath,/lib"
        )

    @pytest.mark.agent_authored(model="grok-4.6")
    def test_replaces_prefixed_compiler_path(self):
        assert (
            build_hooks._with_compiler("x86_64-linux-gnu-gcc -shared -Wl,-z,relro", "clang")
            == "clang -shared -Wl,-z,relro"
        )

    @pytest.mark.agent_authored(model="grok-4.6")
    def test_compiler_only_command_returns_compiler(self):
        assert build_hooks._with_compiler("g++", "clang++") == "clang++"
        assert build_hooks._with_compiler("", "clang++") == "clang++"
        assert build_hooks._with_compiler(None, "clang++") == "clang++"


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


# ---------------------------------------------------------------------------
# Cython cache path helper (workaround for cython/cython#7532)
#
# These tests cover the configuration-digest workaround in build_hooks.py.
# They can be deleted together with the `_cython_cache_path` helper once
# cython/cython#7532 is resolved in a released Cython version and
# cuda-python's minimum Cython version includes the fix.
# See https://github.com/cython/cython/issues/7532


_test_helpers_root = Path(__file__).parents[2] / "cuda_python_test_helpers"
if _test_helpers_root.is_dir() and str(_test_helpers_root) not in sys.path:
    sys.path.insert(0, str(_test_helpers_root))

from cuda_python_test_helpers.cython_cache import POSIX_ONLY_CACHE, CythonAliasMixin, CythonCachePathMixin


class TestCythonCachePath(CythonCachePathMixin):
    """`_cython_cache_path` tests specific to cuda.bindings.

    Inherits the common tests from CythonCachePathMixin; the mixin
    covers the package-agnostic behavior. cuda.bindings does not pass
    ``compile_time_env`` or ``cuda_major``, so the debug-only partition is
    tested here.
    """

    build_hooks = build_hooks
    package = "cuda-bindings"

    @POSIX_ONLY_CACHE
    @pytest.mark.agent_authored(model="grok-4.6")
    def test_changed_debug_changes_namespace(self, monkeypatch, tmp_path):
        """``debug`` partitions the namespace."""
        self._set_env(monkeypatch, str(tmp_path))
        p1 = build_hooks._cython_cache_path("cuda-bindings", debug=False)
        p2 = build_hooks._cython_cache_path("cuda-bindings", debug=True)
        assert p1 != p2


class TestCythonCacheSmokeTest:
    """Real Cython cache miss/hit through `_cython_cache_path`.

    The actual cythonize exercise lives in
    ``cuda_python_test_helpers.cython_cache`` so it is shared with
    ``cuda_core/tests/test_build_hooks.py``.
    """

    @POSIX_ONLY_CACHE
    @pytest.mark.agent_authored(model="grok-4.6")
    def test_cache_miss_then_hit(self, monkeypatch, tmp_path, capsys):
        from cuda_python_test_helpers.cython_cache import (
            cython_cache_miss_then_hit,
        )

        cache_root = tmp_path / "cython-cache"
        cache_root.mkdir()
        monkeypatch.setenv("CUDA_PYTHON_CYTHON_CACHE_DIR", str(cache_root))

        cache_path = build_hooks._cython_cache_path(
            "cuda-bindings",
            compiler_directives={"language_level": 3},
            language_level=3,
            cplus=True,
        )
        assert cache_path is not None
        cython_cache_miss_then_hit(cache_path, tmp_path, capsys)


class TestCythonAlias(CythonAliasMixin):
    """`_stable_cython_alias` tests for cuda.bindings."""

    build_hooks = build_hooks
