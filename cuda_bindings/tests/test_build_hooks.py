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
    """Load build_hooks module from source without polluting sys.path.

    build_hooks.py does `from _build_shared import ...` at module top,
    so pre-load the shared module into sys.modules first. Modifying sys.path
    to make the import work naturally would let cuda_bindings/ shadow the
    installed cuda.bindings package for the rest of the test session.
    """
    build_hooks_dir = Path(__file__).parent.parent
    shared_spec = importlib.util.spec_from_file_location("_build_shared", build_hooks_dir / "_build_shared.py")
    shared_module = importlib.util.module_from_spec(shared_spec)
    sys.modules["_build_shared"] = shared_module
    shared_spec.loader.exec_module(shared_module)

    spec = importlib.util.spec_from_file_location("build_hooks", build_hooks_dir / "build_hooks.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


build_hooks = _load_build_hooks()


@pytest.fixture(autouse=True)
def _isolate_toolchain_env():
    names = ("CUDA_PYTHON_TOOLCHAIN", "CC", "CXX", "LDSHARED", "CUDA_PYTHON_CYTHON_CACHE_DIR")
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
    """cuda.bindings-specific ``_resolve_toolchain`` assertions.

    Shared behavior (default no-touch, sccache preservation, case-insensitive
    parsing, invalid-value error, llvm-overrides-external-CC) is covered by
    ``TestResolveToolchainShared`` further down via the shared mixin. The
    tests here assert the bindings-specific flag set: gnu adds
    ``-fpermissive`` and ``-fno-var-tracking-assignments``; core does not.
    """

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


@pytest.fixture
def stamp(tmp_path, monkeypatch):
    """Redirect the toolchain stamp to a scratch path and reset the shared force flag."""
    scratch = tmp_path / "build" / ".build-toolchain"
    monkeypatch.setattr(build_hooks, "_BUILD_TOOLCHAIN_STAMP", scratch)
    monkeypatch.setattr(sys.modules["_build_shared"], "force_build_ext", False)
    monkeypatch.delenv("CUDA_PYTHON_TOOLCHAIN", raising=False)
    return scratch


def _write_stamp(stamp, toolchain):
    stamp.parent.mkdir(parents=True, exist_ok=True)
    stamp.write_text(toolchain + "\n")


class TestBuildToolchainStamp:
    """cuda.bindings-specific stamp bookkeeping tests.

    The check/record mechanics themselves live in ``_build_shared`` and are
    called through ``check_build_key`` / ``record_build_key``. The tests here
    assert bindings-specific behavior: the stamp value is just the toolchain
    name, and ``record`` re-derives that name from the environment.

    The ``_abi_stamp_path`` scoping mechanism is covered by ``TestAbiStampPath``
    via the shared mixin.
    """

    @pytest.mark.agent_authored(model="glm-5.2")
    def test_same_toolchain_does_not_force(self, stamp):
        _write_stamp(stamp, "gnu")
        build_hooks.check_build_key(stamp, lambda: "gnu")
        assert build_hooks.force_build_ext is False

    @pytest.mark.agent_authored(model="glm-5.2")
    def test_changed_toolchain_forces_rebuild(self, stamp):
        _write_stamp(stamp, "gnu")
        build_hooks.check_build_key(stamp, lambda: "llvm")
        assert build_hooks.force_build_ext is True

    @pytest.mark.agent_authored(model="glm-5.2")
    def test_record_writes_stamp(self, stamp):
        # _current_toolchain_key re-derives from env (CUDA_PYTHON_TOOLCHAIN unset →
        # platform default: gnu on Linux, msvc on Windows).
        build_hooks.record_build_key(stamp, build_hooks._current_toolchain_key)
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

from cuda_python_test_helpers.build_shared import (
    AbiStampPathMixin,
    CheckToolchainAvailableSharedMixin,
    ResolveToolchainSharedMixin,
)
from cuda_python_test_helpers.cython_cache import POSIX_ONLY_CACHE, CythonAliasMixin, CythonCachePathMixin


class TestResolveToolchainShared(ResolveToolchainSharedMixin):
    build_hooks = build_hooks


class TestCheckToolchainAvailable(CheckToolchainAvailableSharedMixin):
    build_hooks = build_hooks


class TestAbiStampPath(AbiStampPathMixin):
    build_hooks = build_hooks


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
