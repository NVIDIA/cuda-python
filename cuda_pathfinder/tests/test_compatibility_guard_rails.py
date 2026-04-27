# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import importlib
import os
from pathlib import Path

import pytest
from local_helpers import (
    have_distribution,
    locate_real_cuda_toolkit_version_from_cuda_h,
    require_real_cuda_toolkit_version_from_cuda_h,
    require_real_driver_cuda_version,
)

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
from cuda.pathfinder._dynamic_libs.load_nvidia_dynamic_lib import _resolve_system_loaded_abs_path_in_subprocess
from cuda.pathfinder._headers.find_nvidia_headers import (
    locate_nvidia_header_directory as locate_nvidia_header_directory_raw,
)
from cuda.pathfinder._utils import driver_info
from cuda.pathfinder._utils.driver_info import DriverCudaVersion, QueryDriverCudaVersionError
from cuda.pathfinder._utils.env_vars import get_cuda_path_or_home
from cuda.pathfinder._utils.toolkit_info import read_cuda_header_version

STRICTNESS = os.environ.get("CUDA_PATHFINDER_TEST_COMPATIBILITY_GUARD_RAILS_STRICTNESS", "see_what_works")
assert STRICTNESS in ("see_what_works", "all_must_work")
COMPATIBILITY_GUARD_RAILS_ENV_VAR = "CUDA_PATHFINDER_COMPATIBILITY_GUARD_RAILS"
process_wide_module = importlib.import_module("cuda.pathfinder._process_wide_compatibility_guard_rails")


@pytest.fixture(autouse=True)
def _default_process_wide_guard_rails_mode(monkeypatch):
    monkeypatch.delenv(COMPATIBILITY_GUARD_RAILS_ENV_VAR, raising=False)


@pytest.fixture
def clear_real_host_probe_caches():
    have_distribution.cache_clear()
    locate_real_cuda_toolkit_version_from_cuda_h.cache_clear()
    locate_nvidia_header_directory_raw.cache_clear()
    _resolve_system_loaded_abs_path_in_subprocess.cache_clear()
    get_cuda_path_or_home.cache_clear()
    read_cuda_header_version.cache_clear()
    driver_info._load_nvidia_dynamic_lib.cache_clear()
    driver_info.query_driver_cuda_version.cache_clear()
    yield
    have_distribution.cache_clear()
    locate_real_cuda_toolkit_version_from_cuda_h.cache_clear()
    locate_nvidia_header_directory_raw.cache_clear()
    _resolve_system_loaded_abs_path_in_subprocess.cache_clear()
    get_cuda_path_or_home.cache_clear()
    read_cuda_header_version.cache_clear()
    driver_info._load_nvidia_dynamic_lib.cache_clear()
    driver_info.query_driver_cuda_version.cache_clear()


def _write_cuda_h(
    ctk_root: Path,
    toolkit_version: str,
    *,
    include_dir_parts: tuple[str, ...] = ("targets", "x86_64-linux", "include"),
) -> None:
    parts = toolkit_version.split(".")
    if len(parts) < 2:
        raise AssertionError(f"Expected at least major.minor in toolkit version, got {toolkit_version!r}")
    encoded = int(parts[0]) * 1000 + int(parts[1]) * 10
    cuda_h_path = ctk_root.joinpath(*include_dir_parts, "cuda.h")
    cuda_h_path.parent.mkdir(parents=True, exist_ok=True)
    cuda_h_path.write_text(
        f"#ifndef CUDA_H\n#define CUDA_H\n#define CUDA_VERSION {encoded}\n#endif\n",
        encoding="utf-8",
    )


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
    return DriverCudaVersion.from_encoded(encoded)


class _FakeDistribution:
    def __init__(
        self,
        *,
        name: str,
        version: str,
        root: Path,
        files: tuple[str, ...] = (),
        requires: tuple[str, ...] = (),
    ) -> None:
        self.metadata = {"Name": name}
        self.version = version
        self.files = tuple(Path(file) for file in files)
        self.requires = list(requires)
        self._root = root

    def locate_file(self, file: Path) -> Path:
        return self._root / file


def _assert_real_ctk_backed_path(path: str) -> None:
    norm_path = os.path.normpath(os.path.abspath(path))
    if "site-packages" in Path(norm_path).parts:
        return
    current = Path(norm_path)
    if current.is_file():
        current = current.parent
    for candidate in (current, *current.parents):
        if (candidate / "include" / "cuda.h").is_file():
            return
        if any(path.is_file() for path in (candidate / "targets").glob("*/include/cuda.h")):
            return
    for env_var in ("CUDA_PATH", "CUDA_HOME"):
        ctk_root = os.environ.get(env_var)
        if not ctk_root:
            continue
        norm_ctk_root = os.path.normpath(os.path.abspath(ctk_root))
        if os.path.commonpath((norm_path, norm_ctk_root)) == norm_ctk_root:
            return
    raise AssertionError(
        "Expected a site-packages path, a path under a CTK root with cuda.h, "
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
def test_public_apis_default_to_strict_when_env_var_is_unset_or_empty(monkeypatch, tmp_path, env_value):
    lib_path = _touch(tmp_path / "no-cuda-h" / "targets" / "x86_64-linux" / "lib" / "libnvrtc.so.12")

    monkeypatch.setattr(compatibility_module, "_load_nvidia_dynamic_lib", lambda _libname: _loaded_dl(lib_path))
    monkeypatch.setattr(
        pathfinder,
        "process_wide_compatibility_guard_rails",
        CompatibilityGuardRails(driver_cuda_version=_driver_cuda_version(13000)),
    )

    def fail_raw_fallback(_libname: str) -> LoadedDL:
        pytest.fail("strict mode must not fall back to raw loading")

    monkeypatch.setattr(process_wide_module, "_load_nvidia_dynamic_lib", fail_raw_fallback)
    if env_value is None:
        monkeypatch.delenv(COMPATIBILITY_GUARD_RAILS_ENV_VAR, raising=False)
    else:
        monkeypatch.setenv(COMPATIBILITY_GUARD_RAILS_ENV_VAR, env_value)

    with pytest.raises(CompatibilityInsufficientMetadataError, match="cuda.h"):
        pathfinder.load_nvidia_dynamic_lib("nvrtc")


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


def test_public_apis_share_process_wide_guard_rails_state(monkeypatch, tmp_path):
    lib_root = tmp_path / "cuda-12.8"
    hdr_root = tmp_path / "cuda-12.9"
    _write_cuda_h(lib_root, "12.8.20250303")
    _write_cuda_h(hdr_root, "12.9.20250531")

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
    _write_cuda_h(ctk_root, "12.9.20250531")
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
    _write_cuda_h(lib_root, "12.8.20250303")
    _write_cuda_h(hdr_root, "12.9.20250531")

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
    _write_cuda_h(ctk_root, "13.0.20251003")
    lib_path = _touch(ctk_root / "targets" / "x86_64-linux" / "lib" / "libnvrtc.so.13")

    monkeypatch.setattr(compatibility_module, "_load_nvidia_dynamic_lib", lambda _libname: _loaded_dl(lib_path))

    guard_rails = CompatibilityGuardRails(driver_cuda_version=_driver_cuda_version(12080))

    with pytest.raises(CompatibilityCheckError, match="driver_major >= ctk_major"):
        guard_rails.load_nvidia_dynamic_lib("nvrtc")


def test_missing_cuda_h_raises_insufficient_metadata(monkeypatch, tmp_path):
    lib_path = _touch(tmp_path / "no-cuda-h" / "targets" / "x86_64-linux" / "lib" / "libnvrtc.so.12")

    monkeypatch.setattr(compatibility_module, "_load_nvidia_dynamic_lib", lambda _libname: _loaded_dl(lib_path))

    guard_rails = CompatibilityGuardRails(driver_cuda_version=_driver_cuda_version(13000))

    with pytest.raises(CompatibilityInsufficientMetadataError, match="cuda.h"):
        guard_rails.load_nvidia_dynamic_lib("nvrtc")


def test_windows_style_ctk_root_uses_root_include_cuda_h(monkeypatch, tmp_path):
    ctk_root = tmp_path / "cuda-13.2"
    _write_cuda_h(ctk_root, "13.2.20251003", include_dir_parts=("include",))
    lib_path = _touch(ctk_root / "bin" / "x64" / "nvrtc64_130_0.dll")

    monkeypatch.setattr(compatibility_module, "_load_nvidia_dynamic_lib", lambda _libname: _loaded_dl(lib_path))

    guard_rails = CompatibilityGuardRails(driver_cuda_version=_driver_cuda_version(13000))

    loaded = guard_rails.load_nvidia_dynamic_lib("nvrtc")

    assert loaded.abs_path == lib_path


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


def test_driver_libs_do_not_lock_ctk_anchor(monkeypatch, tmp_path):
    driver_lib_path = _touch(tmp_path / "driver-root" / "libnvidia-ml.so.1")
    ctk_root = tmp_path / "cuda-12.9"
    _write_cuda_h(ctk_root, "12.9.20250531")
    ctk_lib_path = _touch(ctk_root / "targets" / "x86_64-linux" / "lib" / "libnvrtc.so.12")

    def fake_load_nvidia_dynamic_lib(libname: str) -> LoadedDL:
        if libname == "nvml":
            return _loaded_dl(driver_lib_path, found_via="system-search")
        if libname == "nvrtc":
            return _loaded_dl(ctk_lib_path)
        raise AssertionError(f"Unexpected libname: {libname!r}")

    monkeypatch.setattr(compatibility_module, "_load_nvidia_dynamic_lib", fake_load_nvidia_dynamic_lib)

    guard_rails = CompatibilityGuardRails(driver_cuda_version=_driver_cuda_version(13000))

    driver_loaded = guard_rails.load_nvidia_dynamic_lib("nvml")
    ctk_loaded = guard_rails.load_nvidia_dynamic_lib("nvrtc")

    assert driver_loaded.abs_path == driver_lib_path
    assert ctk_loaded.abs_path == ctk_lib_path


def test_driver_libs_do_not_mask_later_ctk_mismatch(monkeypatch, tmp_path):
    driver_lib_path = _touch(tmp_path / "driver-root" / "libnvidia-ml.so.1")
    lib_root = tmp_path / "cuda-12.8"
    hdr_root = tmp_path / "cuda-12.9"
    _write_cuda_h(lib_root, "12.8.20250303")
    _write_cuda_h(hdr_root, "12.9.20250531")

    lib_path = _touch(lib_root / "targets" / "x86_64-linux" / "lib" / "libnvrtc.so.12")
    hdr_dir = hdr_root / "targets" / "x86_64-linux" / "include"
    _touch(hdr_dir / "nvrtc.h")

    def fake_load_nvidia_dynamic_lib(libname: str) -> LoadedDL:
        if libname == "nvml":
            return _loaded_dl(driver_lib_path, found_via="system-search")
        if libname == "nvrtc":
            return _loaded_dl(lib_path)
        raise AssertionError(f"Unexpected libname: {libname!r}")

    monkeypatch.setattr(compatibility_module, "_load_nvidia_dynamic_lib", fake_load_nvidia_dynamic_lib)
    monkeypatch.setattr(
        compatibility_module,
        "_locate_nvidia_header_directory",
        lambda _libname: LocatedHeaderDir(abs_path=str(hdr_dir), found_via="CUDA_PATH"),
    )

    guard_rails = CompatibilityGuardRails(driver_cuda_version=_driver_cuda_version(13000))
    guard_rails.load_nvidia_dynamic_lib("nvml")
    guard_rails.load_nvidia_dynamic_lib("nvrtc")

    with pytest.raises(CompatibilityCheckError, match="exact CTK major.minor match"):
        guard_rails.find_nvidia_header_directory("nvrtc")


@pytest.mark.parametrize(
    "requirement",
    (
        "nvidia-nvjitlink == 13.2.78.*; extra == 'nvjitlink'",
        "nvidia-nvjitlink<14,>=13.2.78; extra == 'nvjitlink'",
    ),
)
def test_wheel_metadata_accepts_exact_and_range_requirements(monkeypatch, tmp_path, requirement):
    site_packages = tmp_path / "site-packages"
    lib_path = _touch(site_packages / "nvidia" / "cu13" / "lib" / "libnvJitLink.so.13")
    owner_dist = _FakeDistribution(
        name="nvidia-nvjitlink",
        version="13.2.78",
        root=site_packages,
        files=("nvidia/cu13/lib/libnvJitLink.so.13",),
    )
    cuda_toolkit_dist = _FakeDistribution(
        name="cuda-toolkit",
        version="13.2.1",
        root=site_packages,
        requires=(requirement,),
    )

    compatibility_module._owned_distribution_candidates.cache_clear()
    compatibility_module._cuda_toolkit_requirement_maps.cache_clear()
    try:
        monkeypatch.setattr(
            compatibility_module.importlib.metadata,
            "distributions",
            lambda: (owner_dist, cuda_toolkit_dist),
        )

        metadata = compatibility_module._wheel_metadata_for_abs_path(lib_path)
    finally:
        compatibility_module._owned_distribution_candidates.cache_clear()
        compatibility_module._cuda_toolkit_requirement_maps.cache_clear()

    assert metadata is not None
    assert metadata.ctk_version.major == 13
    assert metadata.ctk_version.minor == 2
    assert metadata.source == "wheel metadata via nvidia-nvjitlink==13.2.78 pinned by cuda-toolkit==13.2.1"


def test_constraints_accept_string_and_tuple_forms(monkeypatch, tmp_path):
    ctk_root = tmp_path / "cuda-12.9"
    _write_cuda_h(ctk_root, "12.9.20250531")
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
    _write_cuda_h(ctk_root, "12.9.20250531")
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
    _write_cuda_h(ctk_root, "12.9.20250531")

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
    _write_cuda_h(ctk_root, "12.9.20250531")
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
    _write_cuda_h(ctk_root, "12.9.20250531")
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


@pytest.mark.usefixtures("clear_real_host_probe_caches")
def test_real_driver(info_summary_append):
    real_driver = require_real_driver_cuda_version()
    info_summary_append(
        f"real driver CUDA version={real_driver.major}.{real_driver.minor} (encoded={real_driver.encoded})"
    )


@pytest.mark.usefixtures("clear_real_host_probe_caches")
def test_real_ctk(info_summary_append):
    real_ctk = require_real_cuda_toolkit_version_from_cuda_h()
    info_summary_append(
        f"real cuda.h CTK version={real_ctk.version.major}.{real_ctk.version.minor} "
        f"via {real_ctk.found_via} at {real_ctk.cuda_h_path!r}"
    )


@pytest.mark.usefixtures("clear_real_host_probe_caches")
def test_real_wheel_ctk_items_are_compatible(info_summary_append):
    real_ctk = require_real_cuda_toolkit_version_from_cuda_h()
    real_driver = require_real_driver_cuda_version()
    guard_rails = CompatibilityGuardRails(
        ctk_major=real_ctk.version.major,
        ctk_minor=real_ctk.version.minor,
        driver_cuda_version=real_driver,
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
        pytest.skip(f"real CTK check unavailable: {exc.__class__.__name__}: {exc}")

    assert isinstance(loaded.abs_path, str)
    assert header_dir is not None
    for path in (loaded.abs_path, header_dir, static_lib, bitcode_lib):
        _assert_real_ctk_backed_path(path)
    if have_distribution(r"^nvidia-cuda-nvcc-cu12$"):
        # For CUDA 12, NVIDIA publishes a PyPI package named nvidia-cuda-nvcc-cu12,
        # but the wheels only contain nvcc-adjacent compiler components such as
        # ptxas, CRT headers, libnvvm, and libdevice; the nvcc executable itself
        # is not included.
        if nvcc is not None:
            # nvcc found elsewhere, e.g. /usr/local or Conda.
            _assert_real_ctk_backed_path(nvcc)
    else:
        assert nvcc is not None
        _assert_real_ctk_backed_path(nvcc)


@pytest.mark.usefixtures("clear_real_host_probe_caches")
def test_real_wheel_component_version_does_not_override_ctk_line(info_summary_append):
    real_ctk = require_real_cuda_toolkit_version_from_cuda_h()
    real_driver = require_real_driver_cuda_version()
    guard_rails = CompatibilityGuardRails(
        ctk_major=real_ctk.version.major,
        ctk_minor=real_ctk.version.minor,
        driver_cuda_version=real_driver,
    )

    try:
        header_dir = guard_rails.find_nvidia_header_directory("cufft")
    except (CompatibilityCheckError, CompatibilityInsufficientMetadataError) as exc:
        if STRICTNESS == "all_must_work":
            raise
        pytest.skip(f"real cufft CTK check unavailable: {exc.__class__.__name__}: {exc}")

    if header_dir is None:
        if STRICTNESS == "all_must_work":
            raise AssertionError("Expected CTK-backed cufft headers to be discoverable.")
        pytest.skip("real cufft CTK check unavailable: cufft headers not found")

    _assert_real_ctk_backed_path(header_dir)
