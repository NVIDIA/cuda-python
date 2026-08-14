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

These tests require scikit-build-core to be installed (build_hooks.py wraps it).
"""

import importlib.util
import json
import os
import sys
import tempfile
from pathlib import Path
from unittest import mock

import pytest

from cuda.pathfinder import get_cuda_path_or_home

# build_hooks.py imports scikit-build-core at the top level, so skip if not available.
pytest.importorskip("scikit_build_core")


def _load_build_hooks():
    """Load build_hooks module from source without permanently modifying sys.path.

    build_hooks.py is a PEP 517 build backend, not an installed module.
    We use importlib to load it directly from source to avoid polluting
    sys.path with the cuda_core/ directory (which contains cuda/core/ source
    that could shadow the installed package).
    """
    build_hooks_path = Path(__file__).parent.parent / "build_hooks.py"
    helper_name = "_cuda_core_cython_path"
    helper_path = build_hooks_path.with_name(f"{helper_name}.py")
    helper_spec = importlib.util.spec_from_file_location(helper_name, helper_path)
    helper_module = importlib.util.module_from_spec(helper_spec)
    previous_helper = sys.modules.get(helper_name)
    sys.modules[helper_name] = helper_module
    helper_spec.loader.exec_module(helper_module)

    spec = importlib.util.spec_from_file_location("build_hooks", build_hooks_path)
    module = importlib.util.module_from_spec(spec)
    try:
        spec.loader.exec_module(module)
    finally:
        if previous_helper is None:
            del sys.modules[helper_name]
        else:
            sys.modules[helper_name] = previous_helper
    return module


# Load the module once at import time
build_hooks = _load_build_hooks()


@pytest.mark.agent_authored(model="gpt-5.6")
@pytest.mark.parametrize(
    "hook_name",
    ["get_requires_for_build_wheel", "get_requires_for_build_editable"],
)
def test_binary_build_requirements_include_matching_bindings(monkeypatch, hook_name):
    """Binary hooks preserve backend requirements and append CUDA-matched bindings."""
    backend_hook = mock.Mock(return_value=["backend-requirement"])
    monkeypatch.setattr(build_hooks._build_backend, hook_name, backend_hook)
    monkeypatch.setattr(build_hooks, "_configured_cuda_major", lambda _settings: "13")
    config_settings = {"cmake.define.SENTINEL": "value"}

    requirements = getattr(build_hooks, hook_name)(config_settings)

    assert requirements == ["backend-requirement", "cuda-bindings==13.*"]
    expected_settings = dict(config_settings)
    if hook_name == "get_requires_for_build_editable":
        expected_settings["cmake.build-type"] = "Release" if sys.platform == "win32" else "Debug"
    backend_hook.assert_called_once_with(expected_settings)


@pytest.mark.agent_authored(model="gpt-5.6")
def test_sdist_requirements_are_delegated_directly():
    """The sdist hook is delegated directly and remains toolkit-independent."""
    assert build_hooks.get_requires_for_build_sdist is build_hooks._build_backend.get_requires_for_build_sdist


@pytest.mark.agent_authored(model="gpt-5.6")
@pytest.mark.parametrize(
    ("debug", "expected"),
    [("true", "Debug"), ("false", "Release"), (True, "Debug"), (False, "Release")],
)
def test_legacy_debug_setting_maps_to_cmake_build_type(monkeypatch, debug, expected):
    monkeypatch.setattr(sys, "platform", "linux")
    settings = build_hooks._translate_config_settings({"debug": debug}, editable=False)

    assert settings == {"cmake.build-type": expected}


@pytest.mark.agent_authored(model="gpt-5.6")
@pytest.mark.parametrize(("platform", "expected"), [("linux", "Debug"), ("win32", "Release")])
def test_editable_build_uses_platform_default(monkeypatch, platform, expected):
    monkeypatch.setattr(sys, "platform", platform)

    settings = build_hooks._translate_config_settings({}, editable=True)

    assert settings == {"cmake.build-type": expected}


@pytest.mark.agent_authored(model="gpt-5.6")
def test_debug_build_is_rejected_on_windows(monkeypatch):
    monkeypatch.setattr(sys, "platform", "win32")

    with pytest.raises(RuntimeError, match="not supported on Windows"):
        build_hooks._translate_config_settings({"debug": "true"}, editable=False)


@pytest.mark.agent_authored(model="gpt-5.6")
def test_build_config_forwards_cuda_and_coverage(monkeypatch):
    monkeypatch.setattr(build_hooks, "_configured_cuda_root", lambda _settings: "/cuda")
    monkeypatch.setattr(build_hooks, "_configured_cuda_major", lambda _settings: "13")
    bindings_root = mock.Mock(return_value=None)
    monkeypatch.setattr(build_hooks._cuda_core_cython_path, "find_cuda_bindings_cython_root", bindings_root)
    monkeypatch.setenv("CUDA_PYTHON_COVERAGE", "1")

    settings = build_hooks._build_config_settings(
        {
            "debug": "false",
            "cmake.define.CUDA_BINDINGS_CYTHON_ROOT": "/bindings",
            "cmake.define.SENTINEL": "value",
        },
        editable=False,
    )

    assert settings == {
        "cmake.build-type": "Release",
        "cmake.define.CUDA_BINDINGS_CYTHON_ROOT": "/bindings",
        "cmake.define.SENTINEL": "value",
        "cmake.define.CUDA_CORE_CUDA_ROOT": "/cuda",
        "cmake.define.CUDA_CORE_BUILD_MAJOR": "13",
        "cmake.define.CUDA_PYTHON_COVERAGE": "1",
    }
    bindings_root.assert_not_called()


class _DirectUrlDistribution:
    def __init__(self, direct_url):
        self._direct_url = direct_url

    def read_text(self, filename):
        assert filename == "direct_url.json"
        return self._direct_url if isinstance(self._direct_url, str) else json.dumps(self._direct_url)


def _write_bindings_declaration(root):
    declaration = root / "cuda" / "bindings" / "cydriver.pxd"
    declaration.parent.mkdir(parents=True)
    declaration.touch()


def _mock_cuda_build_settings(monkeypatch):
    monkeypatch.setattr(build_hooks, "_configured_cuda_root", lambda _settings: "/cuda")
    monkeypatch.setattr(build_hooks, "_configured_cuda_major", lambda _settings: "13")
    monkeypatch.setenv("CUDA_PYTHON_COVERAGE", "0")


@pytest.mark.agent_authored(model="gpt-5.6")
def test_physical_bindings_wheel_clears_cached_cmake_root(tmp_path, monkeypatch):
    site_packages = tmp_path / "site-packages"
    _write_bindings_declaration(site_packages)
    monkeypatch.setattr(sys, "path", [str(site_packages)])
    _mock_cuda_build_settings(monkeypatch)

    settings = build_hooks._build_config_settings({}, editable=False)

    assert settings["cmake.define.CUDA_BINDINGS_CYTHON_ROOT"] == ""


@pytest.mark.agent_authored(model="gpt-5.6")
def test_editable_meta_finder_passes_verified_physical_roots(tmp_path, monkeypatch):
    project_root = tmp_path / "cuda-bindings"
    source_root = project_root / "src"
    _write_bindings_declaration(source_root)
    direct_url = {
        "url": project_root.as_uri(),
        "dir_info": {"editable": True},
    }
    find_spec = mock.Mock(
        return_value=mock.Mock(
            origin=str(source_root / "cuda" / "bindings" / "__init__.py"),
            submodule_search_locations=[str(source_root / "cuda" / "bindings")],
        )
    )
    monkeypatch.setattr(sys, "path", [str(tmp_path / "build-site-packages")])
    monkeypatch.setattr(
        build_hooks._cuda_core_cython_path.importlib.metadata,
        "distribution",
        lambda _name: _DirectUrlDistribution(direct_url),
    )
    monkeypatch.setattr(build_hooks._cuda_core_cython_path.importlib.util, "find_spec", find_spec)
    _mock_cuda_build_settings(monkeypatch)

    expected = str(source_root.resolve())
    settings = build_hooks._build_config_settings({}, editable=True)

    assert settings["cmake.define.CUDA_BINDINGS_CYTHON_ROOT"] == expected
    find_spec.assert_called_once_with("cuda.bindings")


@pytest.mark.agent_authored(model="gpt-5.6")
def test_editable_runtime_exposes_bindings_source_root(tmp_path, monkeypatch):
    source_root = tmp_path / "cuda-bindings"
    _write_bindings_declaration(source_root)
    direct_url = {
        "url": source_root.as_uri(),
        "dir_info": {"editable": True},
    }
    runtime_sys_path = [str(tmp_path / "site-packages")]
    monkeypatch.setattr(sys, "path", runtime_sys_path)
    monkeypatch.setattr(
        build_hooks._cuda_core_cython_path.importlib.metadata,
        "distribution",
        lambda _name: _DirectUrlDistribution(direct_url),
    )

    build_hooks._cuda_core_cython_path.add_editable_cuda_bindings_path()

    assert sys.path == [runtime_sys_path[0], str(source_root.resolve())]


@pytest.mark.agent_authored(model="gpt-5.6")
def test_editable_runtime_uses_spec_without_direct_url(tmp_path, monkeypatch):
    source_root = tmp_path / "cuda-bindings"
    _write_bindings_declaration(source_root)
    runtime_sys_path = [str(tmp_path / "site-packages")]
    find_spec = mock.Mock(
        return_value=mock.Mock(
            origin=str(source_root / "cuda" / "bindings" / "__init__.py"),
            submodule_search_locations=[str(source_root / "cuda" / "bindings")],
        )
    )
    distribution = mock.Mock()
    distribution.read_text.return_value = None
    monkeypatch.setattr(sys, "path", runtime_sys_path)
    monkeypatch.setattr(
        build_hooks._cuda_core_cython_path.importlib.metadata,
        "distribution",
        lambda _name: distribution,
    )
    monkeypatch.setattr(build_hooks._cuda_core_cython_path.importlib.util, "find_spec", find_spec)

    build_hooks._cuda_core_cython_path.add_editable_cuda_bindings_path()

    assert sys.path == [runtime_sys_path[0], str(source_root.resolve())]
    find_spec.assert_called_once_with("cuda.bindings")


@pytest.mark.agent_authored(model="gpt-5.6")
def test_editable_build_does_not_persist_regular_wheel_root(tmp_path, monkeypatch):
    site_packages = tmp_path / "site-packages"
    _write_bindings_declaration(site_packages)
    direct_url = {
        "url": site_packages.as_uri(),
        "dir_info": {"editable": False},
    }
    monkeypatch.setattr(sys, "path", [str(site_packages)])
    monkeypatch.setattr(
        build_hooks._cuda_core_cython_path.importlib.metadata,
        "distribution",
        lambda _name: _DirectUrlDistribution(direct_url),
    )
    _mock_cuda_build_settings(monkeypatch)

    settings = build_hooks._build_config_settings({}, editable=True)
    build_hooks._cuda_core_cython_path.add_editable_cuda_bindings_path()

    assert settings["cmake.define.CUDA_BINDINGS_CYTHON_ROOT"] == ""
    assert sys.path == [str(site_packages)]


@pytest.mark.agent_authored(model="gpt-5.6")
def test_editable_runtime_warns_on_invalid_bindings_metadata(tmp_path, monkeypatch):
    runtime_sys_path = [str(tmp_path / "site-packages")]
    monkeypatch.setattr(sys, "path", runtime_sys_path)
    monkeypatch.setattr(
        build_hooks._cuda_core_cython_path.importlib.metadata,
        "distribution",
        lambda _name: _DirectUrlDistribution("{"),
    )

    with pytest.warns(RuntimeWarning, match="invalid direct_url.json"):
        build_hooks._cuda_core_cython_path.add_editable_cuda_bindings_path()

    assert sys.path == runtime_sys_path


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
