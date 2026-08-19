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
import tempfile
import warnings
from pathlib import Path
from unittest import mock

import pytest


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
    monkeypatch.setattr(build_hooks, "_determine_cuda_major_version", lambda: "13")
    config_settings = {"cmake.define.SENTINEL": "value"}

    requirements = getattr(build_hooks, hook_name)(config_settings)

    assert requirements == ["backend-requirement", "cuda-bindings==13.*"]
    backend_hook.assert_called_once_with(config_settings)
    assert backend_hook.call_args.args[0] is config_settings


@pytest.mark.agent_authored(model="gpt-5.6")
@pytest.mark.parametrize(
    ("hook_name", "args", "config_index"),
    [
        pytest.param("get_requires_for_build_wheel", (), 0, id="wheel-requirements"),
        pytest.param("get_requires_for_build_editable", (), 0, id="editable-requirements"),
        pytest.param("get_requires_for_build_sdist", (), 0, id="sdist-requirements"),
        pytest.param("prepare_metadata_for_build_wheel", ("metadata",), 1, id="wheel-metadata"),
        pytest.param("prepare_metadata_for_build_editable", ("metadata",), 1, id="editable-metadata"),
        pytest.param("build_wheel", ("dist", "metadata"), 1, id="wheel-build"),
        pytest.param("build_editable", ("dist", "metadata"), 1, id="editable-build"),
        pytest.param("build_sdist", ("dist",), 1, id="sdist-build"),
    ],
)
def test_legacy_debug_is_translated_for_pep517_hooks(monkeypatch, hook_name, args, config_index):
    is_requirements_hook = hook_name.startswith("get_requires")
    backend_result = ["backend-requirement"] if is_requirements_hook else "artifact"
    backend_hook = mock.Mock(return_value=backend_result)
    monkeypatch.setattr(build_hooks._build_backend, hook_name, backend_hook)
    determine_cuda_major = mock.Mock(return_value="13")
    monkeypatch.setattr(build_hooks, "_determine_cuda_major_version", determine_cuda_major)
    config_settings = {"debug": "true", "cmake.define.SENTINEL": "value"}
    call_args = list(args)
    call_args.insert(config_index, config_settings)

    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        result = getattr(build_hooks, hook_name)(*call_args)

    expected_args = list(args)
    expected_args.insert(
        config_index,
        {"cmake.build-type": "Debug", "cmake.define.SENTINEL": "value"},
    )
    backend_hook.assert_called_once_with(*expected_args)
    adds_cuda_requirement = hook_name in {
        "get_requires_for_build_wheel",
        "get_requires_for_build_editable",
    }
    expected_result = ["backend-requirement", "cuda-bindings==13.*"] if adds_cuda_requirement else backend_result
    assert result == expected_result
    assert config_settings == {"debug": "true", "cmake.define.SENTINEL": "value"}
    assert determine_cuda_major.call_count == int(adds_cuda_requirement)

    if hook_name in {"build_wheel", "build_editable"}:
        assert len(caught) == 1
        assert caught[0].category is FutureWarning
        assert "removed in cuda.core 2.0" in str(caught[0].message)
    else:
        assert caught == []


@pytest.mark.agent_authored(model="gpt-5.6")
@pytest.mark.parametrize(
    ("debug", "expected"),
    [
        pytest.param(True, "Debug", id="bool-true"),
        pytest.param(False, "Release", id="bool-false"),
        pytest.param("TRUE", "Debug", id="string-true"),
        pytest.param("off", "Release", id="string-false"),
        pytest.param(["false", "yes"], "Debug", id="repeated-last-value"),
    ],
)
def test_legacy_debug_values(debug, expected):
    assert build_hooks._translate_legacy_debug({"debug": debug}) == {"cmake.build-type": expected}


@pytest.mark.agent_authored(model="gpt-5.6")
@pytest.mark.parametrize("native_setting", ["cmake.build-type", "skbuild.cmake.build-type"])
def test_native_build_type_precedes_legacy_debug(native_setting):
    config_settings = {"debug": "not-a-boolean", native_setting: "RelWithDebInfo"}

    assert build_hooks._translate_legacy_debug(config_settings) == {native_setting: "RelWithDebInfo"}
    assert config_settings == {"debug": "not-a-boolean", native_setting: "RelWithDebInfo"}


@pytest.mark.agent_authored(model="gpt-5.6")
@pytest.mark.parametrize("debug", ["maybe", []])
def test_invalid_legacy_debug_value_is_rejected(debug):
    with pytest.raises(ValueError, match="debug must"):
        build_hooks._translate_legacy_debug({"debug": debug})


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
        with mock.patch.dict(os.environ, {"CUDA_CORE_BUILD_MAJOR": version}, clear=True):
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

    @pytest.mark.agent_authored(model="gpt-5.6")
    def test_cuda_path_precedes_cuda_home(self, tmp_path):
        cuda_path = tmp_path / "cuda-path"
        cuda_home = tmp_path / "cuda-home"
        for root, version in ((cuda_path, 12080), (cuda_home, 13010)):
            include = root / "include"
            include.mkdir(parents=True)
            (include / "cuda.h").write_text(f"#define CUDA_VERSION {version}\n")

        with mock.patch.dict(
            os.environ,
            {"CUDA_PATH": str(cuda_path), "CUDA_HOME": str(cuda_home)},
            clear=True,
        ):
            assert build_hooks._determine_cuda_major_version() == "12"

    def test_env_var_takes_priority_over_headers(self):
        """Env var override takes priority even when headers exist."""
        _check_version_detection(12080, "11", cuda_core_build_major="11")

    @pytest.mark.agent_authored(model="gpt-5.6")
    def test_empty_env_var_falls_back_to_headers(self):
        _check_version_detection(13010, "13", cuda_core_build_major="")

    @pytest.mark.agent_authored(model="gpt-5.6")
    def test_invalid_env_var_raises_error(self):
        with (
            mock.patch.dict(os.environ, {"CUDA_CORE_BUILD_MAJOR": "thirteen"}, clear=True),
            pytest.raises(RuntimeError, match="must be an integer"),
        ):
            build_hooks._determine_cuda_major_version()

    @pytest.mark.agent_authored(model="gpt-5.6")
    def test_missing_cuda_header_raises_error(self, tmp_path):
        with (
            mock.patch.dict(os.environ, {"CUDA_PATH": str(tmp_path)}, clear=True),
            pytest.raises(RuntimeError, match="valid CUDA installation with include/cuda.h"),
        ):
            build_hooks._determine_cuda_major_version()

    def test_missing_cuda_path_raises_error(self):
        """RuntimeError is raised when CUDA_PATH/CUDA_HOME not set and no env var override."""
        with (
            mock.patch.dict(os.environ, {}, clear=True),
            pytest.raises(RuntimeError, match="CUDA_PATH or CUDA_HOME"),
        ):
            build_hooks._determine_cuda_major_version()
