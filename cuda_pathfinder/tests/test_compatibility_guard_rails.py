# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import json
import os
from pathlib import Path

import pytest

import cuda.pathfinder._compatibility_guard_rails as compatibility_module
from cuda import pathfinder
from cuda.pathfinder import (
    BitcodeLibNotFoundError,
    CompatibilityCheckError,
    CompatibilityGuardRails,
    CompatibilityInsufficientMetadataError,
    DynamicLibNotFoundError,
    LoadedDL,
    LocatedBitcodeLib,
    LocatedHeaderDir,
    LocatedStaticLib,
    StaticLibNotFoundError,
    process_wide_compatibility_guard_rails,
)
from cuda.pathfinder._utils.driver_info import DriverCudaVersion, QueryDriverCudaVersionError

STRICTNESS = os.environ.get("CUDA_PATHFINDER_TEST_COMPATIBILITY_GUARD_RAILS_STRICTNESS", "see_what_works")
assert STRICTNESS in ("see_what_works", "all_must_work")


def _write_version_json(ctk_root: Path, toolkit_version: str) -> None:
    ctk_root.mkdir(parents=True, exist_ok=True)
    payload = {"cuda": {"version": toolkit_version}}
    (ctk_root / "version.json").write_text(json.dumps(payload), encoding="utf-8")


def _touch(path: Path) -> str:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.touch()
    return str(path)


def _loaded_dl(abs_path: str, *, found_via: str = "CUDA_PATH") -> LoadedDL:
    return LoadedDL(
        abs_path=abs_path,
        was_already_loaded_from_elsewhere=False,
        _handle_uint=1,
        found_via=found_via,
    )


def _located_static_lib(name: str, abs_path: str) -> LocatedStaticLib:
    return LocatedStaticLib(
        name=name,
        abs_path=abs_path,
        filename=os.path.basename(abs_path),
        found_via="CUDA_PATH",
    )


def _located_bitcode_lib(name: str, abs_path: str) -> LocatedBitcodeLib:
    return LocatedBitcodeLib(
        name=name,
        abs_path=abs_path,
        filename=os.path.basename(abs_path),
        found_via="CUDA_PATH",
    )


def _driver_cuda_version(encoded: int) -> DriverCudaVersion:
    return DriverCudaVersion(
        encoded=encoded,
        major=encoded // 1000,
        minor=(encoded % 1000) // 10,
    )


def _assert_real_ctk_backed_path(path: str) -> None:
    norm_path = os.path.normpath(os.path.abspath(path))
    if "site-packages" in Path(norm_path).parts:
        return
    current = Path(norm_path)
    if current.is_file():
        current = current.parent
    for candidate in (current, *current.parents):
        version_json_path = candidate / "version.json"
        if version_json_path.is_file():
            return
    for env_var in ("CUDA_PATH", "CUDA_HOME"):
        ctk_root = os.environ.get(env_var)
        if not ctk_root:
            continue
        norm_ctk_root = os.path.normpath(os.path.abspath(ctk_root))
        if os.path.commonpath((norm_path, norm_ctk_root)) == norm_ctk_root:
            return
    raise AssertionError(
        "Expected a site-packages path, a path under a CTK root with version.json, "
        f"or a path under CUDA_PATH/CUDA_HOME, got {path!r}"
    )


class _DelegatingProcessWideGuardRails:
    def __init__(self, method_name: str, return_value: object) -> None:
        self._method_name = method_name
        self._return_value = return_value
        self.calls: list[tuple[str, tuple[object, ...]]] = []

    def __getattr__(self, name: str):
        if name != self._method_name:
            raise AttributeError(name)

        def delegated(*args: object) -> object:
            self.calls.append((name, args))
            return self._return_value

        return delegated


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


def test_public_apis_share_process_wide_guard_rails_state(monkeypatch, tmp_path):
    lib_root = tmp_path / "cuda-12.8"
    hdr_root = tmp_path / "cuda-12.9"
    _write_version_json(lib_root, "12.8.20250303")
    _write_version_json(hdr_root, "12.9.20250531")

    lib_path = _touch(lib_root / "targets" / "x86_64-linux" / "lib" / "libnvrtc.so.12")
    hdr_dir = hdr_root / "targets" / "x86_64-linux" / "include"
    _touch(hdr_dir / "nvrtc.h")

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
    with pytest.raises(CompatibilityCheckError, match="exact CTK major.minor match"):
        pathfinder.find_nvidia_header_directory("nvrtc")


def test_load_dynamic_lib_then_find_headers_same_ctk_version(monkeypatch, tmp_path):
    ctk_root = tmp_path / "cuda-12.9"
    _write_version_json(ctk_root, "12.9.20250531")
    lib_path = _touch(ctk_root / "targets" / "x86_64-linux" / "lib" / "libnvrtc.so.12")
    hdr_dir = ctk_root / "targets" / "x86_64-linux" / "include"
    _touch(hdr_dir / "nvrtc.h")

    monkeypatch.setattr(compatibility_module, "_load_nvidia_dynamic_lib", lambda _libname: _loaded_dl(lib_path))
    monkeypatch.setattr(
        compatibility_module,
        "_locate_nvidia_header_directory",
        lambda _libname: LocatedHeaderDir(abs_path=str(hdr_dir), found_via="CUDA_PATH"),
    )

    guard_rails = CompatibilityGuardRails(driver_cuda_version=_driver_cuda_version(13000))

    loaded = guard_rails.load_nvidia_dynamic_lib("nvrtc")
    hdr_path = guard_rails.find_nvidia_header_directory("nvrtc")

    assert loaded.abs_path == lib_path
    assert hdr_path == str(hdr_dir)


def test_exact_ctk_major_minor_match_is_required(monkeypatch, tmp_path):
    lib_root = tmp_path / "cuda-12.8"
    hdr_root = tmp_path / "cuda-12.9"
    _write_version_json(lib_root, "12.8.20250303")
    _write_version_json(hdr_root, "12.9.20250531")

    lib_path = _touch(lib_root / "targets" / "x86_64-linux" / "lib" / "libnvrtc.so.12")
    hdr_dir = hdr_root / "targets" / "x86_64-linux" / "include"
    _touch(hdr_dir / "nvrtc.h")

    monkeypatch.setattr(compatibility_module, "_load_nvidia_dynamic_lib", lambda _libname: _loaded_dl(lib_path))
    monkeypatch.setattr(
        compatibility_module,
        "_locate_nvidia_header_directory",
        lambda _libname: LocatedHeaderDir(abs_path=str(hdr_dir), found_via="CUDA_PATH"),
    )

    guard_rails = CompatibilityGuardRails(driver_cuda_version=_driver_cuda_version(13000))
    guard_rails.load_nvidia_dynamic_lib("nvrtc")

    with pytest.raises(CompatibilityCheckError, match="exact CTK major.minor match"):
        guard_rails.find_nvidia_header_directory("nvrtc")


def test_driver_major_must_not_be_older_than_ctk_major(monkeypatch, tmp_path):
    ctk_root = tmp_path / "cuda-13.0"
    _write_version_json(ctk_root, "13.0.20251003")
    lib_path = _touch(ctk_root / "targets" / "x86_64-linux" / "lib" / "libnvrtc.so.13")

    monkeypatch.setattr(compatibility_module, "_load_nvidia_dynamic_lib", lambda _libname: _loaded_dl(lib_path))

    guard_rails = CompatibilityGuardRails(driver_cuda_version=_driver_cuda_version(12080))

    with pytest.raises(CompatibilityCheckError, match="driver_major >= ctk_major"):
        guard_rails.load_nvidia_dynamic_lib("nvrtc")


def test_missing_version_json_raises_insufficient_metadata(monkeypatch, tmp_path):
    lib_path = _touch(tmp_path / "no-version-json" / "targets" / "x86_64-linux" / "lib" / "libnvrtc.so.12")

    monkeypatch.setattr(compatibility_module, "_load_nvidia_dynamic_lib", lambda _libname: _loaded_dl(lib_path))

    guard_rails = CompatibilityGuardRails(driver_cuda_version=_driver_cuda_version(13000))

    with pytest.raises(CompatibilityInsufficientMetadataError, match="version.json"):
        guard_rails.load_nvidia_dynamic_lib("nvrtc")


def test_other_packaging_raises_insufficient_metadata(monkeypatch, tmp_path):
    abs_path = _touch(tmp_path / "site-packages" / "nvidia" / "nvshmem" / "lib" / "libnvshmem_device.bc")

    monkeypatch.setattr(
        compatibility_module,
        "_locate_bitcode_lib",
        lambda _name: _located_bitcode_lib("nvshmem_device", abs_path),
    )

    guard_rails = CompatibilityGuardRails(driver_cuda_version=_driver_cuda_version(13000))

    with pytest.raises(CompatibilityInsufficientMetadataError, match="packaged_with='ctk'"):
        guard_rails.find_bitcode_lib("nvshmem_device")


def test_constraints_accept_string_and_tuple_forms(monkeypatch, tmp_path):
    ctk_root = tmp_path / "cuda-12.9"
    _write_version_json(ctk_root, "12.9.20250531")
    lib_path = _touch(ctk_root / "targets" / "x86_64-linux" / "lib" / "libnvrtc.so.12")

    monkeypatch.setattr(compatibility_module, "_load_nvidia_dynamic_lib", lambda _libname: _loaded_dl(lib_path))

    guard_rails = CompatibilityGuardRails(
        ctk_major=(">=", 12),
        ctk_minor=">=9",
        driver_cuda_version=_driver_cuda_version(13000),
    )

    loaded = guard_rails.load_nvidia_dynamic_lib("nvrtc")

    assert loaded.abs_path == lib_path


def test_constraint_failure_raises(monkeypatch, tmp_path):
    ctk_root = tmp_path / "cuda-12.9"
    _write_version_json(ctk_root, "12.9.20250531")
    lib_path = _touch(ctk_root / "targets" / "x86_64-linux" / "lib" / "libnvrtc.so.12")

    monkeypatch.setattr(compatibility_module, "_load_nvidia_dynamic_lib", lambda _libname: _loaded_dl(lib_path))

    guard_rails = CompatibilityGuardRails(
        ctk_major=12,
        ctk_minor="<9",
        driver_cuda_version=_driver_cuda_version(13000),
    )

    with pytest.raises(CompatibilityCheckError, match="ctk_minor<9"):
        guard_rails.load_nvidia_dynamic_lib("nvrtc")


def test_static_bitcode_and_binary_methods_participate_in_checks(monkeypatch, tmp_path):
    ctk_root = tmp_path / "cuda-12.9"
    _write_version_json(ctk_root, "12.9.20250531")

    lib_path = _touch(ctk_root / "targets" / "x86_64-linux" / "lib" / "libnvrtc.so.12")
    static_path = _touch(ctk_root / "targets" / "x86_64-linux" / "lib" / "libcudadevrt.a")
    bitcode_path = _touch(ctk_root / "nvvm" / "libdevice" / "libdevice.10.bc")
    binary_path = _touch(ctk_root / "bin" / "nvcc")

    monkeypatch.setattr(compatibility_module, "_load_nvidia_dynamic_lib", lambda _libname: _loaded_dl(lib_path))
    monkeypatch.setattr(
        compatibility_module,
        "_locate_static_lib",
        lambda _name: _located_static_lib("cudadevrt", static_path),
    )
    monkeypatch.setattr(
        compatibility_module,
        "_locate_bitcode_lib",
        lambda _name: _located_bitcode_lib("device", bitcode_path),
    )
    monkeypatch.setattr(
        compatibility_module,
        "_find_nvidia_binary_utility",
        lambda _utility_name: binary_path,
    )

    guard_rails = CompatibilityGuardRails(driver_cuda_version=_driver_cuda_version(13000))

    guard_rails.load_nvidia_dynamic_lib("nvrtc")
    assert guard_rails.find_static_lib("cudadevrt") == static_path
    assert guard_rails.find_bitcode_lib("device") == bitcode_path
    assert guard_rails.find_nvidia_binary_utility("nvcc") == binary_path


def test_guard_rails_query_driver_cuda_version_by_default(monkeypatch, tmp_path):
    ctk_root = tmp_path / "cuda-12.9"
    _write_version_json(ctk_root, "12.9.20250531")
    lib_path = _touch(ctk_root / "targets" / "x86_64-linux" / "lib" / "libnvrtc.so.12")

    query_calls: list[int] = []

    monkeypatch.setattr(compatibility_module, "_load_nvidia_dynamic_lib", lambda _libname: _loaded_dl(lib_path))

    def fake_query_driver_cuda_version() -> DriverCudaVersion:
        query_calls.append(1)
        return _driver_cuda_version(13000)

    monkeypatch.setattr(compatibility_module, "query_driver_cuda_version", fake_query_driver_cuda_version)

    guard_rails = CompatibilityGuardRails()

    guard_rails.load_nvidia_dynamic_lib("nvrtc")
    guard_rails.load_nvidia_dynamic_lib("nvrtc")

    assert len(query_calls) == 1


def test_guard_rails_wrap_driver_query_failures(monkeypatch, tmp_path):
    ctk_root = tmp_path / "cuda-12.9"
    _write_version_json(ctk_root, "12.9.20250531")
    lib_path = _touch(ctk_root / "targets" / "x86_64-linux" / "lib" / "libnvrtc.so.12")

    monkeypatch.setattr(compatibility_module, "_load_nvidia_dynamic_lib", lambda _libname: _loaded_dl(lib_path))

    def fail_query_driver_cuda_version() -> DriverCudaVersion:
        raise QueryDriverCudaVersionError("driver query failed")

    monkeypatch.setattr(compatibility_module, "query_driver_cuda_version", fail_query_driver_cuda_version)

    guard_rails = CompatibilityGuardRails()

    with pytest.raises(
        CompatibilityCheckError,
        match="Failed to query the CUDA driver version needed for compatibility checks",
    ) as exc_info:
        guard_rails.load_nvidia_dynamic_lib("nvrtc")

    assert isinstance(exc_info.value.__cause__, QueryDriverCudaVersionError)


def test_find_nvidia_header_directory_returns_none_when_unresolved(monkeypatch):
    monkeypatch.setattr(
        compatibility_module,
        "_locate_nvidia_header_directory",
        lambda _libname: None,
    )

    guard_rails = CompatibilityGuardRails(driver_cuda_version=_driver_cuda_version(13000))

    assert guard_rails.find_nvidia_header_directory("nvrtc") is None


def test_real_wheel_ctk_items_are_compatible(info_summary_append):
    guard_rails = CompatibilityGuardRails(
        ctk_major=13,
        ctk_minor=2,
        driver_cuda_version=_driver_cuda_version(13000),
    )

    try:
        loaded = guard_rails.load_nvidia_dynamic_lib("nvrtc")
        header_dir = guard_rails.find_nvidia_header_directory("nvrtc")
        static_lib = guard_rails.find_static_lib("cudadevrt")
        bitcode_lib = guard_rails.find_bitcode_lib("device")
        nvcc = guard_rails.find_nvidia_binary_utility("nvcc")
    except (
        CompatibilityCheckError,
        CompatibilityInsufficientMetadataError,
        DynamicLibNotFoundError,
        StaticLibNotFoundError,
        BitcodeLibNotFoundError,
    ) as exc:
        if STRICTNESS == "all_must_work":
            raise
        info_summary_append(f"real CTK check unavailable: {exc.__class__.__name__}: {exc}")
        return

    info_summary_append(f"nvrtc={loaded.abs_path!r}")
    info_summary_append(f"nvrtc_headers={header_dir!r}")
    info_summary_append(f"cudadevrt={static_lib!r}")
    info_summary_append(f"libdevice={bitcode_lib!r}")
    info_summary_append(f"nvcc={nvcc!r}")

    assert isinstance(loaded.abs_path, str)
    assert header_dir is not None
    assert nvcc is not None
    for path in (loaded.abs_path, header_dir, static_lib, bitcode_lib, nvcc):
        _assert_real_ctk_backed_path(path)


def test_real_wheel_component_version_does_not_override_ctk_line(info_summary_append):
    guard_rails = CompatibilityGuardRails(
        ctk_major=13,
        ctk_minor=2,
        driver_cuda_version=_driver_cuda_version(13000),
    )

    try:
        header_dir = guard_rails.find_nvidia_header_directory("cufft")
    except (CompatibilityCheckError, CompatibilityInsufficientMetadataError) as exc:
        if STRICTNESS == "all_must_work":
            raise
        info_summary_append(f"real cufft CTK check unavailable: {exc.__class__.__name__}: {exc}")
        return

    if header_dir is None:
        if STRICTNESS == "all_must_work":
            raise AssertionError("Expected CTK-backed cufft headers to be discoverable.")
        info_summary_append("real cufft CTK check unavailable: cufft headers not found")
        return

    info_summary_append(f"cufft_headers={header_dir!r}")
    _assert_real_ctk_backed_path(header_dir)
