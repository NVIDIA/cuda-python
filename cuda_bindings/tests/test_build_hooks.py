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


def _load_module(name, path, *, register=False):
    """Load a module from source without permanently modifying sys.path.

    build_hooks.py and _build_shared.py are PEP 517 backend files, not
    installed modules. We use importlib to load them directly from source to
    avoid polluting sys.path with the package directory (which contains
    cuda/ source that could shadow the installed package). With ``register``
    the module is also entered into sys.modules, which is how build_hooks.py's
    ``from _build_shared import ...`` finds this copy.
    """
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    if register:
        sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


# Load the modules once at import time; _build_shared must come first.
_PACKAGE_ROOT = Path(__file__).parent.parent
_build_shared = _load_module("_build_shared", _PACKAGE_ROOT / "_build_shared.py", register=True)
build_hooks = _load_module("build_hooks", _PACKAGE_ROOT / "build_hooks.py")


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


def _fake_sysconfig(monkeypatch, **values):
    """Pin sysconfig.get_config_var so linker-command assertions are exact."""
    monkeypatch.setattr(_build_shared.sysconfig, "get_config_var", lambda name: values.get(name))


@pytest.fixture
def stamp(tmp_path, monkeypatch):
    """Redirect the toolchain stamp to a scratch path and reset the shared force flag."""
    scratch = tmp_path / "build" / ".build-toolchain"
    monkeypatch.setattr(build_hooks, "_BUILD_TOOLCHAIN_STAMP", scratch)
    monkeypatch.setattr(_build_shared, "force_build_ext", False)
    monkeypatch.delenv("CUDA_PYTHON_TOOLCHAIN", raising=False)
    return scratch


def _write_stamp(stamp, toolchain):
    stamp.parent.mkdir(parents=True, exist_ok=True)
    stamp.write_text(toolchain + "\n")


class TestBuildToolchainStamp:
    """cuda.bindings stamps the toolchain name, through _check_build_toolchain() and record_build_toolchain().

    The stamp-and-force protocol itself is tested by TestBuildKeyStamp.
    """

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

from cuda_python_test_helpers.build_shared import (
    AbiStampPathMixin,
    BuildKeyStampMixin,
    CheckToolchainAvailableMixin,
    DistutilsLinkerIntegrationMixin,
    ForceBuildExtReexportMixin,
    ResolveToolchainMixin,
    WithCompilerMixin,
    WithSccacheMixin,
)
from cuda_python_test_helpers.cython_cache import POSIX_ONLY_CACHE, CythonAliasMixin, CythonCachePathMixin


class TestResolveToolchainShared(ResolveToolchainMixin):
    build_shared = _build_shared


class TestWithSccache(WithSccacheMixin):
    build_shared = _build_shared


class TestWithCompiler(WithCompilerMixin):
    build_shared = _build_shared


class TestDistutilsLinkerIntegration(DistutilsLinkerIntegrationMixin):
    build_shared = _build_shared


class TestCheckToolchainAvailable(CheckToolchainAvailableMixin):
    build_shared = _build_shared


class TestAbiStampPath(AbiStampPathMixin):
    build_shared = _build_shared


class TestBuildKeyStamp(BuildKeyStampMixin):
    build_shared = _build_shared


class TestForceBuildExtReexport(ForceBuildExtReexportMixin):
    build_hooks = build_hooks
    build_shared = _build_shared


class TestResolveToolchain:
    """What cuda.bindings chooses in its ``_resolve_toolchain`` wrapper."""

    @pytest.mark.agent_authored(model="claude-sonnet-4-6")
    def test_linux_flag_set(self, monkeypatch):
        """c++14 (c++17 costs ~15% on launch benchmarks), plus -Wno-deprecated-declarations; no -Werror."""
        if sys.platform == "win32":
            pytest.skip("Linux flags only")
        monkeypatch.delenv("CUDA_PYTHON_TOOLCHAIN", raising=False)
        _name, _cc, _cxx, cargs, _largs = build_hooks._resolve_toolchain(debug=False)
        assert "-std=c++14" in cargs
        assert "-Wno-deprecated-declarations" in cargs
        assert "-Werror" not in cargs

    @pytest.mark.agent_authored(model="claude-sonnet-4-6")
    def test_msvc_flag_set(self, monkeypatch):
        if sys.platform != "win32":
            pytest.skip("MSVC flags only on Windows")
        monkeypatch.delenv("CUDA_PYTHON_TOOLCHAIN", raising=False)
        _name, _cc, _cxx, cargs, _largs = build_hooks._resolve_toolchain(debug=False)
        assert "/std:c++14" in cargs
        assert "/WX" not in cargs


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


# ---------------------------------------------------------------------------
# CUDA header check
#
# A cuda-bindings source tree is generated from one CUDA header set. It
# compiles only against a toolkit of that major.minor. Against another minor,
# the C++ compile fails with redefinition errors that do not name the cause.
# _check_cuda_headers() reads both versions before cythonize and fails early
# with a message that does. No GPU needed.


def _write_cuda_h(tmp_path, cuda_version):
    include = tmp_path / "include"
    include.mkdir(exist_ok=True)
    (include / "cuda.h").write_text(f"#define CUDA_VERSION {cuda_version}\n", encoding="utf-8")
    return str(tmp_path)


class TestCudaHeaderCheck:
    @pytest.mark.agent_authored(model="claude-fable-5-1")
    def test_generated_header_version_is_read_from_cydriver_pxd(self):
        generated = build_hooks._generated_cuda_version()
        assert generated // 1000 in (12, 13)
        assert build_hooks._major_minor(13040) == "13.4"
        assert build_hooks._major_minor(12090) == "12.9"

    @pytest.mark.agent_authored(model="claude-fable-5-1")
    def test_a_header_of_the_generated_major_minor_passes(self, tmp_path):
        generated = build_hooks._generated_cuda_version()
        build_hooks._check_cuda_headers(_write_cuda_h(tmp_path, generated))
        # Only major.minor matters. The last digit, as in 13041, is a toolkit patch.
        build_hooks._check_cuda_headers(_write_cuda_h(tmp_path, generated + 1))

    @pytest.mark.agent_authored(model="claude-fable-5-1")
    @pytest.mark.parametrize("delta", [-10, 10, -1000, 1000])
    def test_another_header_fails_and_names_both_versions(self, tmp_path, delta):
        generated = build_hooks._generated_cuda_version()
        cuda_path = _write_cuda_h(tmp_path, generated + delta)
        with pytest.raises(RuntimeError) as excinfo:
            build_hooks._check_cuda_headers(cuda_path)
        message = str(excinfo.value)
        needed, found = build_hooks._major_minor(generated), build_hooks._major_minor(generated + delta)
        assert message.startswith(f"This cuda-bindings source tree needs CUDA {needed} headers, but ")
        assert os.path.realpath(os.path.join(cuda_path, "include", "cuda.h")) in message  # the resolved path
        assert f" is CUDA {found}. This is a build-time requirement only" in message
        assert build_hooks._INSTALL_URL in message
        assert message.endswith(
            f"Point CUDA_PATH or CUDA_HOME at a CUDA {needed} toolkit, or build from cuda-bindings {found}.x sources."
        )

    @pytest.mark.agent_authored(model="claude-fable-5-1")
    def test_an_unreadable_cuda_h_is_a_clear_error(self, tmp_path):
        with pytest.raises(RuntimeError, match=r"Cannot read CUDA_VERSION from .*cuda\.h"):
            build_hooks._check_cuda_headers(str(tmp_path))  # no include/cuda.h
        (tmp_path / "include").mkdir()
        (tmp_path / "include" / "cuda.h").write_text("/* no CUDA_VERSION macro */\n", encoding="utf-8")
        with pytest.raises(RuntimeError, match=r"Cannot read CUDA_VERSION from .*cuda\.h"):
            build_hooks._check_cuda_headers(str(tmp_path))

    @pytest.mark.agent_authored(model="claude-fable-5-1")
    def test_the_build_checks_the_header_before_it_touches_the_tree(self, tmp_path, monkeypatch):
        cuda_path = _write_cuda_h(tmp_path, build_hooks._generated_cuda_version() + 10)
        monkeypatch.setattr(build_hooks, "_get_cuda_path", lambda: cuda_path)

        def not_reached():
            raise AssertionError("the header check must run before the build modifies the source tree")

        monkeypatch.setattr(build_hooks, "_rename_architecture_specific_files", not_reached)
        with pytest.raises(RuntimeError, match="source tree needs CUDA .* headers"):
            build_hooks._build_cuda_bindings()
