# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import sys

import pytest
from compatibility_guard_rails_test_utils import (
    COMPATIBILITY_GUARD_RAILS_ENV_VAR,
    DRIVER_COMPATIBILITY_ENV_VAR,
    _default_process_wide_guard_rails_mode,  # noqa: F401
    _DelegatingProcessWideGuardRails,
    _driver_cuda_version,
    _loaded_dl,
    _located_bitcode_lib,
    _located_static_lib,
    _touch,
    _touch_ctk_file,
    compatibility_module,
    process_wide_module,
)

from cuda import pathfinder
from cuda.pathfinder import (
    CompatibilityCheckError,
    CompatibilityGuardRails,
    CompatibilityInsufficientMetadataError,
    LoadedDL,
    LocatedHeaderDir,
    process_wide_compatibility_guard_rails,
)


def test_process_wide_compatibility_guard_rails_is_public_singleton():
    assert process_wide_compatibility_guard_rails is pathfinder.process_wide_compatibility_guard_rails
    assert isinstance(process_wide_compatibility_guard_rails, CompatibilityGuardRails)


@pytest.mark.parametrize(
    ("public_api_name", "guard_rails_method_name", "args", "return_value"),
    [
        (
            "load_nvidia_dynamic_lib",
            "load_nvidia_dynamic_lib",
            ("nvrtc",),
            _loaded_dl("/opt/mock/libnvrtc.so.12"),
        ),
        (
            "locate_nvidia_header_directory",
            "locate_nvidia_header_directory",
            ("nvrtc",),
            LocatedHeaderDir(abs_path="/opt/mock/include", found_via="CUDA_PATH"),
        ),
        ("find_nvidia_header_directory", "find_nvidia_header_directory", ("nvrtc",), "/opt/mock/include"),
        (
            "locate_static_lib",
            "locate_static_lib",
            ("cudadevrt",),
            _located_static_lib("cudadevrt", "/opt/mock/libcudadevrt.a"),
        ),
        ("find_static_lib", "find_static_lib", ("cudadevrt",), "/opt/mock/libcudadevrt.a"),
        (
            "locate_bitcode_lib",
            "locate_bitcode_lib",
            ("device",),
            _located_bitcode_lib("device", "/opt/mock/libdevice.10.bc"),
        ),
        ("find_bitcode_lib", "find_bitcode_lib", ("device",), "/opt/mock/libdevice.10.bc"),
        ("find_nvidia_binary_utility", "find_nvidia_binary_utility", ("nvcc",), "/opt/mock/nvcc"),
    ],
)
def test_public_apis_route_through_process_wide_guard_rails(
    monkeypatch, public_api_name, guard_rails_method_name, args, return_value
):
    fake_guard_rails = _DelegatingProcessWideGuardRails(guard_rails_method_name, return_value)
    monkeypatch.setattr(pathfinder, "process_wide_compatibility_guard_rails", fake_guard_rails)

    result = getattr(pathfinder, public_api_name)(*args)

    assert result == return_value
    assert fake_guard_rails.calls == [(guard_rails_method_name, args)]


def test_public_driver_libs_are_allowed_in_strict_mode(monkeypatch, tmp_path):
    driver_lib_path = _touch(tmp_path / "driver-root" / "libnvidia-ml.so.1")

    monkeypatch.setattr(
        compatibility_module,
        "_load_nvidia_dynamic_lib",
        lambda _libname: _loaded_dl(driver_lib_path, found_via="system-search"),
    )
    monkeypatch.setattr(
        pathfinder,
        "process_wide_compatibility_guard_rails",
        CompatibilityGuardRails(driver_cuda_version=_driver_cuda_version(13000)),
    )

    def fail_raw_fallback(_libname: str) -> LoadedDL:
        pytest.fail("strict mode must not fall back to raw loading")

    monkeypatch.setattr(process_wide_module, "_load_nvidia_dynamic_lib", fail_raw_fallback)

    loaded = pathfinder.load_nvidia_dynamic_lib("nvml")

    assert loaded.abs_path == driver_lib_path


@pytest.mark.parametrize("env_value", [None, ""])
def test_public_apis_default_mode_applies_when_env_var_is_unset_or_empty(monkeypatch, tmp_path, env_value):
    guarded_lib_path = _touch(tmp_path / "no-cuda-h" / "targets" / "x86_64-linux" / "lib" / "libnvrtc.so.12")
    raw_loaded = _loaded_dl("/opt/mock/libnvrtc.so.12", found_via="system-search")

    monkeypatch.setattr(compatibility_module, "_load_nvidia_dynamic_lib", lambda _libname: _loaded_dl(guarded_lib_path))
    monkeypatch.setattr(process_wide_module, "_load_nvidia_dynamic_lib", lambda _libname: raw_loaded)
    monkeypatch.setattr(
        pathfinder,
        "process_wide_compatibility_guard_rails",
        CompatibilityGuardRails(driver_cuda_version=_driver_cuda_version(13000)),
    )

    if env_value is None:
        monkeypatch.delenv(COMPATIBILITY_GUARD_RAILS_ENV_VAR, raising=False)
    else:
        monkeypatch.setenv(COMPATIBILITY_GUARD_RAILS_ENV_VAR, env_value)

    default_mode = process_wide_module._COMPATIBILITY_GUARD_RAILS_DEFAULT_MODE
    if default_mode == "strict":
        with pytest.raises(CompatibilityInsufficientMetadataError, match="cuda.h"):
            pathfinder.load_nvidia_dynamic_lib("nvrtc")
        return

    loaded = pathfinder.load_nvidia_dynamic_lib("nvrtc")
    assert loaded is raw_loaded


def test_public_apis_best_effort_fall_back_on_insufficient_metadata(monkeypatch, tmp_path):
    guarded_lib_path = _touch(tmp_path / "no-cuda-h" / "targets" / "x86_64-linux" / "lib" / "libnvrtc.so.12")
    raw_loaded = _loaded_dl("/opt/mock/libnvrtc.so.12", found_via="system-search")

    monkeypatch.setenv(COMPATIBILITY_GUARD_RAILS_ENV_VAR, "best_effort")
    monkeypatch.setattr(compatibility_module, "_load_nvidia_dynamic_lib", lambda _libname: _loaded_dl(guarded_lib_path))
    monkeypatch.setattr(process_wide_module, "_load_nvidia_dynamic_lib", lambda _libname: raw_loaded)
    monkeypatch.setattr(
        pathfinder,
        "process_wide_compatibility_guard_rails",
        CompatibilityGuardRails(driver_cuda_version=_driver_cuda_version(13000)),
    )

    loaded = pathfinder.load_nvidia_dynamic_lib("nvrtc")

    assert loaded is raw_loaded


def test_public_apis_off_bypass_process_wide_guard_rails(monkeypatch):
    raw_loaded = _loaded_dl("/opt/mock/libnvrtc.so.12", found_via="system-search")
    fake_guard_rails = _DelegatingProcessWideGuardRails(
        "load_nvidia_dynamic_lib",
        _loaded_dl("/opt/mock/guard-rails/libnvrtc.so.12"),
    )

    monkeypatch.setenv(COMPATIBILITY_GUARD_RAILS_ENV_VAR, "off")
    monkeypatch.setattr(pathfinder, "process_wide_compatibility_guard_rails", fake_guard_rails)
    monkeypatch.setattr(process_wide_module, "_load_nvidia_dynamic_lib", lambda _libname: raw_loaded)

    loaded = pathfinder.load_nvidia_dynamic_lib("nvrtc")

    assert loaded is raw_loaded
    assert fake_guard_rails.calls == []


def test_public_apis_reject_invalid_guard_rails_mode(monkeypatch):
    monkeypatch.setenv(COMPATIBILITY_GUARD_RAILS_ENV_VAR, "unexpected")

    with pytest.raises(RuntimeError, match=COMPATIBILITY_GUARD_RAILS_ENV_VAR) as exc_info:
        pathfinder.find_nvidia_binary_utility("nvcc")

    message = str(exc_info.value)
    assert "'off'" in message
    assert "'best_effort'" in message
    assert "'strict'" in message
    assert f"defaults to {process_wide_module._COMPATIBILITY_GUARD_RAILS_DEFAULT_MODE!r}" in message


def test_public_apis_reject_invalid_driver_compatibility_mode(monkeypatch):
    monkeypatch.setenv(DRIVER_COMPATIBILITY_ENV_VAR, "unexpected")

    with pytest.raises(RuntimeError, match=DRIVER_COMPATIBILITY_ENV_VAR) as exc_info:
        pathfinder.find_nvidia_binary_utility("nvcc")

    message = str(exc_info.value)
    assert "'default'" in message
    assert "'assume_forward_compatibility'" in message
    assert f"defaults to {process_wide_module._DRIVER_COMPATIBILITY_DEFAULT_MODE!r}" in message


def test_driver_compatibility_override_is_linux_only(monkeypatch):
    monkeypatch.setenv(DRIVER_COMPATIBILITY_ENV_VAR, "assume_forward_compatibility")
    monkeypatch.setattr(process_wide_module.sys, "platform", "win32")

    with pytest.raises(RuntimeError, match="only supported on Linux"):
        pathfinder.find_nvidia_binary_utility("nvcc")


@pytest.mark.skipif(
    not sys.platform.startswith("linux"),
    reason="driver forward-compatibility override is Linux-only",
)
def test_public_driver_mismatch_advertises_forward_compatibility_override(monkeypatch, tmp_path):
    ctk_root = tmp_path / "cuda-13.0"
    lib_path = _touch_ctk_file(ctk_root, "13.0.20251003", "targets/x86_64-linux/lib/libnvrtc.so.13")

    monkeypatch.setattr(compatibility_module, "_load_nvidia_dynamic_lib", lambda _libname: _loaded_dl(lib_path))
    monkeypatch.setattr(
        pathfinder,
        "process_wide_compatibility_guard_rails",
        CompatibilityGuardRails(driver_cuda_version=_driver_cuda_version(12080)),
    )

    def fail_raw_fallback(_libname: str) -> LoadedDL:
        pytest.fail("driver mismatch should not fall back without explicit override")

    monkeypatch.setattr(process_wide_module, "_load_nvidia_dynamic_lib", fail_raw_fallback)

    with pytest.raises(CompatibilityCheckError, match="driver_major >= ctk_major") as exc_info:
        pathfinder.load_nvidia_dynamic_lib("nvrtc")

    message = str(exc_info.value)
    assert DRIVER_COMPATIBILITY_ENV_VAR in message
    assert "assume_forward_compatibility" in message
    assert "does not relax CTK-coherence checks" in message


@pytest.mark.skipif(
    not sys.platform.startswith("linux"),
    reason="driver forward-compatibility override is Linux-only",
)
def test_public_driver_mismatch_falls_back_when_assuming_forward_compatibility(monkeypatch, tmp_path):
    ctk_root = tmp_path / "cuda-13.0"
    guarded_lib_path = _touch_ctk_file(ctk_root, "13.0.20251003", "targets/x86_64-linux/lib/libnvrtc.so.13")
    raw_loaded = _loaded_dl("/opt/mock/libnvrtc.so.13", found_via="system-search")

    monkeypatch.setenv(DRIVER_COMPATIBILITY_ENV_VAR, "assume_forward_compatibility")
    monkeypatch.setattr(
        compatibility_module,
        "_load_nvidia_dynamic_lib",
        lambda _libname: _loaded_dl(guarded_lib_path),
    )
    monkeypatch.setattr(process_wide_module, "_load_nvidia_dynamic_lib", lambda _libname: raw_loaded)
    monkeypatch.setattr(
        pathfinder,
        "process_wide_compatibility_guard_rails",
        CompatibilityGuardRails(driver_cuda_version=_driver_cuda_version(12080)),
    )

    loaded = pathfinder.load_nvidia_dynamic_lib("nvrtc")

    assert loaded is raw_loaded


@pytest.mark.skipif(
    not sys.platform.startswith("linux"),
    reason="driver forward-compatibility override is Linux-only",
)
def test_forward_compatibility_override_does_not_relax_ctk_coherence_checks(monkeypatch, tmp_path):
    lib_root = tmp_path / "cuda-12.8"
    hdr_root = tmp_path / "cuda-12.9"
    lib_path = _touch_ctk_file(lib_root, "12.8.20250303", "targets/x86_64-linux/lib/libnvrtc.so.12")
    hdr_dir = hdr_root / "targets" / "x86_64-linux" / "include"
    _touch_ctk_file(hdr_root, "12.9.20250531", "targets/x86_64-linux/include/nvrtc.h")

    monkeypatch.setenv(DRIVER_COMPATIBILITY_ENV_VAR, "assume_forward_compatibility")
    monkeypatch.setattr(compatibility_module, "_load_nvidia_dynamic_lib", lambda _libname: _loaded_dl(lib_path))
    monkeypatch.setattr(
        compatibility_module,
        "_locate_nvidia_header_directory",
        lambda _libname: LocatedHeaderDir(abs_path=str(hdr_dir), found_via="CUDA_PATH"),
    )
    monkeypatch.setattr(
        pathfinder,
        "process_wide_compatibility_guard_rails",
        CompatibilityGuardRails(driver_cuda_version=_driver_cuda_version(13000)),
    )

    loaded = pathfinder.load_nvidia_dynamic_lib("nvrtc")

    assert loaded.abs_path == lib_path
    with pytest.raises(CompatibilityCheckError, match=r"companion tag 'api_nvrtc'"):
        pathfinder.find_nvidia_header_directory("nvrtc")


def test_public_apis_share_process_wide_guard_rails_state(monkeypatch, tmp_path):
    lib_root = tmp_path / "cuda-12.8"
    hdr_root = tmp_path / "cuda-12.9"
    lib_path = _touch_ctk_file(lib_root, "12.8.20250303", "targets/x86_64-linux/lib/libnvrtc.so.12")
    hdr_dir = hdr_root / "targets" / "x86_64-linux" / "include"
    _touch_ctk_file(hdr_root, "12.9.20250531", "targets/x86_64-linux/include/nvrtc.h")

    monkeypatch.setattr(compatibility_module, "_load_nvidia_dynamic_lib", lambda _libname: _loaded_dl(lib_path))
    monkeypatch.setattr(
        compatibility_module,
        "_locate_nvidia_header_directory",
        lambda _libname: LocatedHeaderDir(abs_path=str(hdr_dir), found_via="CUDA_PATH"),
    )
    monkeypatch.setattr(
        pathfinder,
        "process_wide_compatibility_guard_rails",
        CompatibilityGuardRails(driver_cuda_version=_driver_cuda_version(13000)),
    )

    loaded = pathfinder.load_nvidia_dynamic_lib("nvrtc")

    assert loaded.abs_path == lib_path
    with pytest.raises(CompatibilityCheckError, match=r"companion tag 'api_nvrtc'"):
        pathfinder.find_nvidia_header_directory("nvrtc")
