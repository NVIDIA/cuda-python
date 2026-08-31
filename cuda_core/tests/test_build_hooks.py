# SPDX-FileCopyrightText: Copyright (c) 2024-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Tests for build_hooks.py build infrastructure.

These tests verify the CUDA version detection logic used during builds,
particularly the _determine_cuda_major_version() function which derives the
CUDA major version from headers.

Note: These tests do NOT require cuda.core to be built/installed since they
test build-time infrastructure. Run with --noconftest to avoid loading
conftest.py which imports cuda.core modules:

    pytest tests/test_build_hooks.py -v --noconftest

These tests require Cython to be installed (build_hooks.py imports it).
"""

import builtins
import importlib.util
import os
import sys
import tempfile
from pathlib import Path
from unittest import mock

# build_hooks.py imports Cython and setuptools at the top level; both are
# declared test dependencies, so a missing install must surface as an
# ImportError at collection time rather than being hidden by importorskip.
import Cython  # noqa: F401
import pytest
import setuptools  # noqa: F401

from cuda.pathfinder import get_cuda_path_or_home


def _load_build_hooks():
    """Load build_hooks module from source without permanently modifying sys.path.

    build_hooks.py is a PEP 517 build backend, not an installed module.
    We use importlib to load it directly from source to avoid polluting
    sys.path with the cuda_core/ directory (which contains cuda/core/ source
    that could shadow the installed package).
    """
    build_hooks_path = Path(__file__).parent.parent / "build_hooks.py"
    spec = importlib.util.spec_from_file_location("build_hooks", build_hooks_path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


# Load the module once at import time
build_hooks = _load_build_hooks()


@pytest.mark.agent_authored(model="gpt-5.6")
def test_cuda_path_is_resolved_before_importing_bindings(monkeypatch):
    """PEP 517 namespace repair runs before cuda.bindings is imported."""
    events = []

    class StopBuildError(Exception):
        pass

    def get_cuda_path():
        events.append("cuda-path")
        return "/cuda"

    original_import = builtins.__import__

    def stop_at_bindings_import(name, *args, **kwargs):
        if name == "cuda.bindings":
            events.append("cuda-bindings")
            raise StopBuildError
        return original_import(name, *args, **kwargs)

    monkeypatch.setattr(build_hooks, "_get_cuda_path", get_cuda_path)
    monkeypatch.setattr(builtins, "__import__", stop_at_bindings_import)

    with pytest.raises(StopBuildError):
        build_hooks._build_cuda_core()

    assert events == ["cuda-path", "cuda-bindings"]


def _check_version_detection(
    cuda_version, expected_major, *, use_cuda_path=True, use_cuda_home=False, cuda_core_build_major=None
):
    """Test version detection with a mock cuda.h.

    Args:
        cuda_version: CUDA_VERSION to write in mock cuda.h (e.g., 12080)
        expected_major: Expected return value (e.g., "12")
        use_cuda_path: If True, set CUDA_PATH to the mock headers directory
        use_cuda_home: If True, set CUDA_HOME to the mock headers directory
        cuda_core_build_major: If set, override with this CUDA_CORE_BUILD_MAJOR env var
    """
    with tempfile.TemporaryDirectory() as tmpdir:
        include_dir = Path(tmpdir) / "include"
        include_dir.mkdir()
        cuda_h = include_dir / "cuda.h"
        cuda_h.write_text(f"#define CUDA_VERSION {cuda_version}\n")

        build_hooks._get_cuda_path.cache_clear()
        build_hooks._determine_cuda_major_version.cache_clear()
        get_cuda_path_or_home.cache_clear()

        mock_env = {
            k: v
            for k, v in {
                "CUDA_CORE_BUILD_MAJOR": cuda_core_build_major,
                "CUDA_PATH": tmpdir if use_cuda_path else None,
                "CUDA_HOME": tmpdir if use_cuda_home else None,
            }.items()
            if v is not None
        }

        with mock.patch.dict(os.environ, mock_env, clear=True):
            result = build_hooks._determine_cuda_major_version()
            assert result == expected_major


class TestGetCudaMajorVersion:
    """Tests for _determine_cuda_major_version()."""

    @pytest.mark.parametrize("version", ["11", "12", "13", "14"])
    def test_env_var_override(self, version):
        """CUDA_CORE_BUILD_MAJOR env var override works with various versions."""
        build_hooks._get_cuda_path.cache_clear()
        build_hooks._determine_cuda_major_version.cache_clear()
        get_cuda_path_or_home.cache_clear()
        with mock.patch.dict(os.environ, {"CUDA_CORE_BUILD_MAJOR": version}, clear=False):
            result = build_hooks._determine_cuda_major_version()
            assert result == version

    @pytest.mark.parametrize(
        ("cuda_version", "expected_major"),
        [
            (11000, "11"),  # CUDA 11.0
            (11080, "11"),  # CUDA 11.8
            (12000, "12"),  # CUDA 12.0
            (12020, "12"),  # CUDA 12.2
            (12080, "12"),  # CUDA 12.8
            (13000, "13"),  # CUDA 13.0
            (13010, "13"),  # CUDA 13.1
        ],
        ids=["11.0", "11.8", "12.0", "12.2", "12.8", "13.0", "13.1"],
    )
    def test_cuda_headers_parsing(self, cuda_version, expected_major):
        """CUDA_VERSION is correctly parsed from cuda.h headers."""
        _check_version_detection(cuda_version, expected_major)

    def test_cuda_home_fallback(self):
        """CUDA_HOME is used if CUDA_PATH is not set."""
        _check_version_detection(12050, "12", use_cuda_path=False, use_cuda_home=True)

    def test_env_var_takes_priority_over_headers(self):
        """Env var override takes priority even when headers exist."""
        _check_version_detection(12080, "11", cuda_core_build_major="11")

    def test_missing_cuda_path_raises_error(self):
        """RuntimeError is raised when CUDA_PATH/CUDA_HOME not set and no env var override."""
        build_hooks._get_cuda_path.cache_clear()
        build_hooks._determine_cuda_major_version.cache_clear()
        get_cuda_path_or_home.cache_clear()
        with (
            mock.patch.dict(os.environ, {}, clear=True),
            pytest.raises(RuntimeError, match="CUDA_PATH or CUDA_HOME"),
        ):
            build_hooks._determine_cuda_major_version()


@pytest.fixture
def stamp(tmp_path, monkeypatch):
    """Redirect the build stamp to a scratch path.

    _BUILD_MAJOR_STAMP is anchored to build_hooks.py rather than the working
    directory, so it has to be replaced outright; chdir would not move it, and
    record_build_major() would write into the real source tree.
    """
    scratch = tmp_path / "build" / ".build-cuda-major"
    monkeypatch.setattr(build_hooks, "_BUILD_MAJOR_STAMP", scratch)
    monkeypatch.setattr(build_hooks, "force_build_ext", False)
    build_hooks._get_cuda_path.cache_clear()
    build_hooks._determine_cuda_major_version.cache_clear()
    get_cuda_path_or_home.cache_clear()
    monkeypatch.setenv("CUDA_CORE_BUILD_MAJOR", "13")
    return scratch


def _write_stamp(stamp, cuda_major):
    stamp.parent.mkdir(parents=True, exist_ok=True)
    stamp.write_text(cuda_major + "\n")


class TestBuildMajorStamp:
    """Tests for _check_build_major() and record_build_major()."""

    def test_missing_stamp_forces_rebuild(self, stamp):
        # No stamp means the last build's major is unknown, so rebuild.
        assert build_hooks._check_build_major() == "13"
        assert build_hooks.force_build_ext is True

    def test_same_major_does_not_force(self, stamp):
        _write_stamp(stamp, "13")
        assert build_hooks._check_build_major() == "13"
        assert build_hooks.force_build_ext is False

    def test_changed_major_forces_rebuild(self, stamp):
        _write_stamp(stamp, "12")
        assert build_hooks._check_build_major() == "13"
        assert build_hooks.force_build_ext is True

    def test_record_writes_stamp(self, stamp):
        build_hooks.record_build_major()
        assert stamp.read_text().strip() == "13"


def _capture_cythonize_build_dir(monkeypatch, cuda_major):
    """Run the cythonize setup for one CUDA major and report its build_dir.

    cythonize() is replaced, so nothing is generated or compiled: this only
    observes which directory the build was about to write into.
    """
    captured = {}

    def fake_cythonize(ext_modules, **kwargs):
        captured.update(kwargs)
        return []

    # Builds resolve the CTK for include dirs; stub it so the test runs
    # where no toolkit is installed (e.g. the wheels CI jobs).
    monkeypatch.setattr(build_hooks, "_get_cuda_path", lambda: "/nonexistent-cuda")
    monkeypatch.setattr(build_hooks, "cythonize", fake_cythonize)
    monkeypatch.setenv("CUDA_CORE_BUILD_MAJOR", cuda_major)
    build_hooks._determine_cuda_major_version.cache_clear()
    # _build_cuda_core() globs cuda/core/**/*.pyx relative to the cwd.
    monkeypatch.chdir(Path(__file__).parent.parent)
    # It also prepends cuda_bindings/ to sys.path; swap in a copy so the
    # mutation lands there and the real list is restored on teardown.
    monkeypatch.setattr(sys, "path", list(sys.path))

    build_hooks._build_cuda_core()
    return Path(captured["build_dir"])


class TestGeneratedSourceDirIsKeyed:
    """Generated C++ must not be shared between CUDA majors.

    Cython's up-to-date check does not hash compile_time_env, so without a
    per-major directory a cu13 build's generated sources are handed to a cu12
    compiler (and vice versa).
    """

    def test_majors_use_different_dirs(self, monkeypatch):
        dir_12 = _capture_cythonize_build_dir(monkeypatch, "12")
        dir_13 = _capture_cythonize_build_dir(monkeypatch, "13")

        assert dir_12 != dir_13
        assert dir_12.name == "cu12"
        assert dir_13.name == "cu13"

    def test_dir_is_anchored_not_relative_to_cwd(self, monkeypatch):
        # Anchored to build_hooks.py, so it must agree with the stamp
        # regardless of where the build was invoked from.
        build_dir = _capture_cythonize_build_dir(monkeypatch, "13")

        assert build_dir.is_absolute()
        assert build_dir.parent.parent == build_hooks._BUILD_MAJOR_STAMP.parent


def _load_setup_py(monkeypatch):
    """Import setup.py for its command classes.

    Importing rather than running is only possible because setup() is guarded
    by __name__ == "__main__"; setuptools invokes the file as a script, so the
    guard does not affect real builds.

    setup.py does a bare ``import build_hooks``, which resolves to
    cuda_bindings' copy if that directory is on sys.path. Pin cuda_core's, so
    the flag the test sets is the one setup.py reads.
    """
    monkeypatch.setitem(sys.modules, "build_hooks", build_hooks)
    setup_path = Path(__file__).parent.parent / "setup.py"
    spec = importlib.util.spec_from_file_location("cuda_core_setup", setup_path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class TestForceReachesBuildExt:
    """The rebuild decision must actually be handed to setuptools.

    _check_build_major() only sets a flag; if build_ext does not read it, a
    stale extension is silently kept because its mtime looks newer than the
    regenerated sources.
    """

    @staticmethod
    def _finalized_build_ext(force_flag, monkeypatch):
        from setuptools.dist import Distribution

        setup_py = _load_setup_py(monkeypatch)
        assert setup_py.build_hooks is build_hooks
        monkeypatch.setattr(build_hooks, "force_build_ext", force_flag)

        cmd = setup_py.build_ext(Distribution({"name": "cuda-core", "version": "0"}))
        cmd.finalize_options()
        return cmd

    def test_flag_set_forces_rebuild(self, monkeypatch):
        assert self._finalized_build_ext(True, monkeypatch).force

    def test_flag_clear_leaves_default(self, monkeypatch):
        assert not self._finalized_build_ext(False, monkeypatch).force
