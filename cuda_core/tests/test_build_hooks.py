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
import os
import sys
import tempfile
from pathlib import Path
from unittest import mock

import pytest

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
    spec = importlib.util.spec_from_file_location("build_hooks", build_hooks_path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
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
    backend_hook.assert_called_once_with(config_settings)


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
    settings = build_hooks._translate_config_settings({"debug": debug})

    assert settings == {"cmake.build-type": expected}


@pytest.mark.agent_authored(model="gpt-5.6")
def test_debug_build_is_rejected_on_windows(monkeypatch):
    monkeypatch.setattr(sys, "platform", "win32")

    with pytest.raises(RuntimeError, match="not supported on Windows"):
        build_hooks._translate_config_settings({"debug": "true"})


@pytest.mark.agent_authored(model="gpt-5.6")
@pytest.mark.parametrize("prefix", ["cmake.define.", "skbuild.cmake.define."])
def test_build_hooks_preserve_explicit_cuda_settings(monkeypatch, prefix):
    config_settings = {
        f"{prefix}CUDA_CORE_CUDA_ROOT": "/configured-cuda",
        f"{prefix}CUDA_CORE_BUILD_MAJOR": "12",
    }
    backend_hook = mock.Mock(return_value="wheel.whl")
    monkeypatch.setattr(build_hooks._build_backend, "build_wheel", backend_hook)

    assert build_hooks.build_wheel("dist", config_settings, "metadata") == "wheel.whl"
    backend_hook.assert_called_once_with("dist", config_settings, "metadata")


@pytest.mark.agent_authored(model="gpt-5.6")
def test_prefixed_cuda_settings_follow_backend_precedence(tmp_path, monkeypatch):
    monkeypatch.delenv("CUDA_CORE_BUILD_MAJOR", raising=False)
    unprefixed_root = tmp_path / "unprefixed"
    prefixed_root = tmp_path / "prefixed"
    for root, version in ((unprefixed_root, 12000), (prefixed_root, 13000)):
        include = root / "include"
        include.mkdir(parents=True)
        (include / "cuda.h").write_text(f"#define CUDA_VERSION {version}\n")

    config_settings = {
        "cmake.define.CUDA_CORE_CUDA_ROOT": str(unprefixed_root),
        "skbuild.cmake.define.CUDA_CORE_CUDA_ROOT": str(prefixed_root),
    }

    assert build_hooks._configured_cuda_major(config_settings) == "13"


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
        with (
            mock.patch.dict(os.environ, {}, clear=True),
            pytest.raises(RuntimeError, match="CUDA_PATH or CUDA_HOME"),
        ):
            build_hooks._determine_cuda_major_version()
