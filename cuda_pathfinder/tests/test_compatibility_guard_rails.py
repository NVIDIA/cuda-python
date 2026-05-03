# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import pytest
from compatibility_guard_rails_test_utils import (
    _default_process_wide_guard_rails_mode,  # noqa: F401
    _driver_cuda_version,
    _driver_release_version,
    _FakeDistribution,
    _loaded_dl,
    _located_bitcode_lib,
    _located_static_lib,
    _make_ctk_root,
    _patch_dynamic_lib_loader,
    _touch,
    _touch_ctk_file,
    _write_cuda_h,
)
from packaging.specifiers import SpecifierSet

import cuda.pathfinder._compatibility_guard_rails as compatibility_module
from cuda.pathfinder import (
    CompatibilityCheckError,
    CompatibilityGuardRails,
    CompatibilityInsufficientMetadataError,
    LocatedHeaderDir,
)
from cuda.pathfinder._binaries.supported_nvidia_binaries import SUPPORTED_BINARIES_ALL
from cuda.pathfinder._static_libs.find_bitcode_lib import SUPPORTED_BITCODE_LIBS
from cuda.pathfinder._static_libs.find_static_lib import SUPPORTED_STATIC_LIBS
from cuda.pathfinder._utils.driver_info import (
    DriverCudaVersion,
    DriverReleaseVersion,
    QueryDriverCudaVersionError,
    QueryDriverReleaseVersionError,
)


def test_same_dynamic_link_component_requires_exact_ctk_major_minor_match(monkeypatch, tmp_path):
    cublas_path = _touch_ctk_file(tmp_path / "cuda-12.8", "12.8.20250303", "targets/x86_64-linux/lib/libcublas.so.12")
    cusolver_path = _touch_ctk_file(
        tmp_path / "cuda-12.9",
        "12.9.20250531",
        "targets/x86_64-linux/lib/libcusolver.so.12",
    )

    _patch_dynamic_lib_loader(
        monkeypatch,
        cublas=_loaded_dl(cublas_path),
        cusolver=_loaded_dl(cusolver_path),
    )

    guard_rails = CompatibilityGuardRails(driver_cuda_version=_driver_cuda_version(13000))
    guard_rails.load_nvidia_dynamic_lib("cublas")

    with pytest.raises(
        CompatibilityCheckError,
        match=r"dynamic-link component 'cuda_blas_solver_runtime'",
    ):
        guard_rails.load_nvidia_dynamic_lib("cusolver")


def test_independent_dynamic_libs_may_resolve_to_different_ctk_minors(monkeypatch, tmp_path):
    nvrtc_path = _touch_ctk_file(tmp_path / "cuda-12.8", "12.8.20250303", "targets/x86_64-linux/lib/libnvrtc.so.12")
    nvjitlink_path = _touch_ctk_file(
        tmp_path / "cuda-12.9",
        "12.9.20250531",
        "targets/x86_64-linux/lib/libnvJitLink.so.12",
    )

    _patch_dynamic_lib_loader(
        monkeypatch,
        nvrtc=_loaded_dl(nvrtc_path),
        nvJitLink=_loaded_dl(nvjitlink_path),
    )

    guard_rails = CompatibilityGuardRails(driver_cuda_version=_driver_cuda_version(13000))

    loaded_nvrtc = guard_rails.load_nvidia_dynamic_lib("nvrtc")
    loaded_nvjitlink = guard_rails.load_nvidia_dynamic_lib("nvJitLink")

    assert loaded_nvrtc.abs_path == nvrtc_path
    assert loaded_nvjitlink.abs_path == nvjitlink_path


def test_toolchain_companions_require_exact_ctk_major_minor_match(monkeypatch, tmp_path):
    static_path = _touch_ctk_file(tmp_path / "cuda-12.8", "12.8.20250303", "targets/x86_64-linux/lib/libcudadevrt.a")
    binary_path = _touch_ctk_file(tmp_path / "cuda-12.9", "12.9.20250531", "bin/nvcc")

    monkeypatch.setattr(
        compatibility_module,
        "_locate_static_lib",
        lambda _name: _located_static_lib("cudadevrt", static_path),
    )
    monkeypatch.setattr(
        compatibility_module,
        "_find_nvidia_binary_utility",
        lambda _utility_name: binary_path,
    )

    guard_rails = CompatibilityGuardRails(driver_cuda_version=_driver_cuda_version(13000))
    assert guard_rails.find_static_lib("cudadevrt") == static_path

    with pytest.raises(
        CompatibilityCheckError,
        match=r"companion tag 'toolchain_cuda_nvcc'",
    ):
        guard_rails.find_nvidia_binary_utility("nvcc")


def test_declared_ltoir_pipeline_requires_nvjitlink_not_older_than_nvrtc(monkeypatch, tmp_path):
    nvrtc_path = _touch_ctk_file(tmp_path / "cuda-12.9", "12.9.20250531", "targets/x86_64-linux/lib/libnvrtc.so.12")
    nvjitlink_path = _touch_ctk_file(
        tmp_path / "cuda-12.8",
        "12.8.20250303",
        "targets/x86_64-linux/lib/libnvJitLink.so.12",
    )

    _patch_dynamic_lib_loader(
        monkeypatch,
        nvrtc=_loaded_dl(nvrtc_path),
        nvJitLink=_loaded_dl(nvjitlink_path),
    )

    guard_rails = CompatibilityGuardRails(driver_cuda_version=_driver_cuda_version(13000))
    guard_rails.load_nvidia_dynamic_lib("nvrtc")
    guard_rails._declare_dynamic_lib_pipeline(
        producer_libname="nvrtc",
        consumer_libname="nvJitLink",
        artifact_kind="ltoir",
    )

    with pytest.raises(CompatibilityCheckError, match=r"nvJitLink must be >= the producer version"):
        guard_rails.load_nvidia_dynamic_lib("nvJitLink")


def test_declared_ltoir_pipeline_allows_same_major_newer_nvjitlink(monkeypatch, tmp_path):
    nvrtc_path = _touch_ctk_file(tmp_path / "cuda-12.8", "12.8.20250303", "targets/x86_64-linux/lib/libnvrtc.so.12")
    nvjitlink_path = _touch_ctk_file(
        tmp_path / "cuda-12.9",
        "12.9.20250531",
        "targets/x86_64-linux/lib/libnvJitLink.so.12",
    )

    _patch_dynamic_lib_loader(
        monkeypatch,
        nvrtc=_loaded_dl(nvrtc_path),
        nvJitLink=_loaded_dl(nvjitlink_path),
    )

    guard_rails = CompatibilityGuardRails(driver_cuda_version=_driver_cuda_version(13000))
    loaded_nvrtc = guard_rails.load_nvidia_dynamic_lib("nvrtc")
    loaded_nvjitlink = guard_rails.load_nvidia_dynamic_lib("nvJitLink")
    guard_rails._declare_dynamic_lib_pipeline(
        producer_libname="nvrtc",
        consumer_libname="nvJitLink",
        artifact_kind="ltoir",
    )

    assert loaded_nvrtc.abs_path == nvrtc_path
    assert loaded_nvjitlink.abs_path == nvjitlink_path


@pytest.mark.parametrize("artifact_kind", ("ptx", "elf", "cubin"))
def test_declared_non_lto_pipeline_allows_cross_major_nvrtc_to_nvjitlink(monkeypatch, tmp_path, artifact_kind):
    nvrtc_path = _touch_ctk_file(tmp_path / "cuda-12.8", "12.8.20250303", "targets/x86_64-linux/lib/libnvrtc.so.12")
    nvjitlink_path = _touch_ctk_file(
        tmp_path / "cuda-13.0",
        "13.0.20251003",
        "targets/x86_64-linux/lib/libnvJitLink.so.13",
    )

    _patch_dynamic_lib_loader(
        monkeypatch,
        nvrtc=_loaded_dl(nvrtc_path),
        nvJitLink=_loaded_dl(nvjitlink_path),
    )

    guard_rails = CompatibilityGuardRails(driver_cuda_version=_driver_cuda_version(13000))
    loaded_nvrtc = guard_rails.load_nvidia_dynamic_lib("nvrtc")
    loaded_nvjitlink = guard_rails.load_nvidia_dynamic_lib("nvJitLink")
    guard_rails._declare_dynamic_lib_pipeline(
        producer_libname="nvrtc",
        consumer_libname="nvJitLink",
        artifact_kind=artifact_kind,
    )

    assert loaded_nvrtc.abs_path == nvrtc_path
    assert loaded_nvjitlink.abs_path == nvjitlink_path


def test_declared_nvvm_pipeline_remains_conservative(monkeypatch, tmp_path):
    nvvm_path = _touch_ctk_file(tmp_path / "cuda-12.8", "12.8.20250303", "nvvm/lib64/libnvvm.so.4")
    nvjitlink_path = _touch_ctk_file(
        tmp_path / "cuda-12.9",
        "12.9.20250531",
        "targets/x86_64-linux/lib/libnvJitLink.so.12",
    )

    _patch_dynamic_lib_loader(
        monkeypatch,
        nvvm=_loaded_dl(nvvm_path),
        nvJitLink=_loaded_dl(nvjitlink_path),
    )

    guard_rails = CompatibilityGuardRails(driver_cuda_version=_driver_cuda_version(13000))
    guard_rails.load_nvidia_dynamic_lib("nvvm")
    guard_rails.load_nvidia_dynamic_lib("nvJitLink")

    with pytest.raises(
        CompatibilityCheckError,
        match=r"remains conservative for explicit nvvm pipeline contexts",
    ):
        guard_rails._declare_dynamic_lib_pipeline(
            producer_libname="nvvm",
            consumer_libname="nvJitLink",
            artifact_kind="ptx",
        )


def test_declared_dynamic_lib_pipeline_rejects_invalid_artifact_kind():
    guard_rails = CompatibilityGuardRails(driver_cuda_version=_driver_cuda_version(13000))

    with pytest.raises(ValueError, match="Invalid pipeline artifact kind"):
        guard_rails._declare_dynamic_lib_pipeline(
            producer_libname="nvrtc",
            consumer_libname="nvJitLink",
            artifact_kind="fatbin",
        )


def test_driver_major_must_not_be_older_than_ctk_major(monkeypatch, tmp_path):
    lib_path = _touch_ctk_file(tmp_path / "cuda-13.0", "13.0.20251003", "targets/x86_64-linux/lib/libnvrtc.so.13")

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
    ctk_lib_path = _touch_ctk_file(tmp_path / "cuda-12.9", "12.9.20250531", "targets/x86_64-linux/lib/libnvrtc.so.12")

    _patch_dynamic_lib_loader(
        monkeypatch,
        nvml=_loaded_dl(driver_lib_path, found_via="system-search"),
        nvrtc=_loaded_dl(ctk_lib_path),
    )

    guard_rails = CompatibilityGuardRails(driver_cuda_version=_driver_cuda_version(13000))

    driver_loaded = guard_rails.load_nvidia_dynamic_lib("nvml")
    ctk_loaded = guard_rails.load_nvidia_dynamic_lib("nvrtc")

    assert driver_loaded.abs_path == driver_lib_path
    assert ctk_loaded.abs_path == ctk_lib_path


def test_driver_libs_do_not_mask_later_ctk_mismatch(monkeypatch, tmp_path):
    driver_lib_path = _touch(tmp_path / "driver-root" / "libnvidia-ml.so.1")
    lib_root = tmp_path / "cuda-12.8"
    hdr_root = tmp_path / "cuda-12.9"
    lib_path = _touch_ctk_file(lib_root, "12.8.20250303", "targets/x86_64-linux/lib/libnvrtc.so.12")
    hdr_dir = hdr_root / "targets" / "x86_64-linux" / "include"
    _touch_ctk_file(hdr_root, "12.9.20250531", "targets/x86_64-linux/include/nvrtc.h")

    _patch_dynamic_lib_loader(
        monkeypatch,
        nvml=_loaded_dl(driver_lib_path, found_via="system-search"),
        nvrtc=_loaded_dl(lib_path),
    )
    monkeypatch.setattr(
        compatibility_module,
        "_locate_nvidia_header_directory",
        lambda _libname: LocatedHeaderDir(abs_path=str(hdr_dir), found_via="CUDA_PATH"),
    )

    guard_rails = CompatibilityGuardRails(driver_cuda_version=_driver_cuda_version(13000))
    guard_rails.load_nvidia_dynamic_lib("nvml")
    guard_rails.load_nvidia_dynamic_lib("nvrtc")

    with pytest.raises(CompatibilityCheckError, match=r"companion tag 'api_nvrtc'"):
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


def test_ctk_version_constraint_accepts_pep440_string(monkeypatch, tmp_path):
    lib_path = _touch_ctk_file(tmp_path / "cuda-12.9", "12.9.20250531", "targets/x86_64-linux/lib/libnvrtc.so.12")

    monkeypatch.setattr(compatibility_module, "_load_nvidia_dynamic_lib", lambda _libname: _loaded_dl(lib_path))

    guard_rails = CompatibilityGuardRails(
        ctk_version=">=12.9,<13",
        driver_cuda_version=_driver_cuda_version(13000),
    )

    loaded = guard_rails.load_nvidia_dynamic_lib("nvrtc")

    assert loaded.abs_path == lib_path


def test_ctk_version_constraint_accepts_specifier_set_instance(monkeypatch, tmp_path):
    lib_path = _touch_ctk_file(tmp_path / "cuda-12.9", "12.9.20250531", "targets/x86_64-linux/lib/libnvrtc.so.12")

    monkeypatch.setattr(compatibility_module, "_load_nvidia_dynamic_lib", lambda _libname: _loaded_dl(lib_path))

    guard_rails = CompatibilityGuardRails(
        ctk_version=SpecifierSet(">=12.9,<13"),
        driver_cuda_version=_driver_cuda_version(13000),
    )

    loaded = guard_rails.load_nvidia_dynamic_lib("nvrtc")

    assert loaded.abs_path == lib_path


def test_ctk_version_constraint_failure_raises(monkeypatch, tmp_path):
    lib_path = _touch_ctk_file(tmp_path / "cuda-12.9", "12.9.20250531", "targets/x86_64-linux/lib/libnvrtc.so.12")

    monkeypatch.setattr(compatibility_module, "_load_nvidia_dynamic_lib", lambda _libname: _loaded_dl(lib_path))

    guard_rails = CompatibilityGuardRails(
        ctk_version="<12.9",
        driver_cuda_version=_driver_cuda_version(13000),
    )

    with pytest.raises(CompatibilityCheckError, match="ctk_version<12.9"):
        guard_rails.load_nvidia_dynamic_lib("nvrtc")


def test_ctk_version_constraint_rejects_invalid_specifier():
    with pytest.raises(ValueError, match="PEP 440 specifier"):
        CompatibilityGuardRails(ctk_version="13.2")


def test_resolved_items_capture_relation_metadata(tmp_path):
    ctk_root = _make_ctk_root(tmp_path / "cuda-12.9", "12.9.20250531")

    lib_path = _touch(ctk_root / "targets" / "x86_64-linux" / "lib" / "libnvrtc.so.12")
    header_dir = ctk_root / "targets" / "x86_64-linux" / "include"
    _touch(header_dir / "fatbinary_section.h")
    static_path = _touch(ctk_root / "targets" / "x86_64-linux" / "lib" / "libcudadevrt.a")
    bitcode_path = _touch(ctk_root / "nvvm" / "libdevice" / "libdevice.10.bc")
    binary_path = _touch(ctk_root / "bin" / "nvcc")

    dynamic_item = compatibility_module._resolve_dynamic_lib_item("nvrtc", _loaded_dl(lib_path))
    header_item = compatibility_module._resolve_header_item(
        "nvcc",
        LocatedHeaderDir(abs_path=str(header_dir), found_via="CUDA_PATH"),
    )
    static_item = compatibility_module._resolve_static_lib_item(_located_static_lib("cudadevrt", static_path))
    bitcode_item = compatibility_module._resolve_bitcode_lib_item(_located_bitcode_lib("device", bitcode_path))
    binary_item = compatibility_module._resolve_binary_item("nvcc", binary_path)

    assert dynamic_item.dynamic_link_component == "nvrtc_mathdx"
    assert dynamic_item.ctk_companion_tags == ("api_nvrtc",)
    assert header_item.dynamic_link_component is None
    assert header_item.ctk_companion_tags == ("toolchain_cuda_nvcc",)
    assert static_item.ctk_companion_tags == ("toolchain_cuda_nvcc",)
    assert bitcode_item.ctk_companion_tags == ("toolchain_cuda_nvcc",)
    assert binary_item.ctk_companion_tags == ("toolchain_cuda_nvcc",)
    assert dynamic_item.ctk_version == header_item.ctk_version == static_item.ctk_version == bitcode_item.ctk_version
    assert binary_item.ctk_version == dynamic_item.ctk_version


@pytest.mark.parametrize("name", SUPPORTED_BITCODE_LIBS)
def test_resolve_bitcode_lib_item_covers_every_supported_name(tmp_path, name):
    abs_path = _touch(tmp_path / "site-packages" / f"{name}.bc")
    item = compatibility_module._resolve_bitcode_lib_item(_located_bitcode_lib(name, abs_path))
    assert item.name == name
    assert item.kind == "bitcode-lib"
    assert item.packaged_with in ("ctk", "other")


@pytest.mark.parametrize("name", SUPPORTED_STATIC_LIBS)
def test_resolve_static_lib_item_covers_every_supported_name(tmp_path, name):
    abs_path = _touch(tmp_path / "site-packages" / f"{name}.a")
    item = compatibility_module._resolve_static_lib_item(_located_static_lib(name, abs_path))
    assert item.name == name
    assert item.kind == "static-lib"
    assert item.packaged_with in ("ctk", "other")


@pytest.mark.parametrize("utility_name", SUPPORTED_BINARIES_ALL)
def test_resolve_binary_item_covers_every_supported_name(tmp_path, utility_name):
    abs_path = _touch(tmp_path / "bin" / utility_name)
    item = compatibility_module._resolve_binary_item(utility_name, abs_path)
    assert item.name == utility_name
    assert item.kind == "binary"
    expected_packaged_with = "other" if utility_name in {"nsys", "nsight-sys", "ncu", "nsight-compute"} else "ctk"
    assert item.packaged_with == expected_packaged_with


def test_static_bitcode_and_binary_methods_participate_in_checks(monkeypatch, tmp_path):
    ctk_root = _make_ctk_root(tmp_path / "cuda-12.9", "12.9.20250531")

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
    lib_path = _touch_ctk_file(tmp_path / "cuda-12.9", "12.9.20250531", "targets/x86_64-linux/lib/libnvrtc.so.12")

    query_calls: list[int] = []

    monkeypatch.setattr(compatibility_module, "_load_nvidia_dynamic_lib", lambda _libname: _loaded_dl(lib_path))

    def fake_query_driver_cuda_version() -> DriverCudaVersion:
        query_calls.append(1)
        return _driver_cuda_version(13000)

    monkeypatch.setattr(compatibility_module, "query_driver_cuda_version", fake_query_driver_cuda_version)
    monkeypatch.setattr(
        compatibility_module,
        "query_driver_release_version",
        lambda: pytest.fail("backward-compatible driver should not need display-driver release metadata"),
    )

    guard_rails = CompatibilityGuardRails()

    guard_rails.load_nvidia_dynamic_lib("nvrtc")
    guard_rails.load_nvidia_dynamic_lib("nvrtc")

    assert len(query_calls) == 1


def test_guard_rails_wrap_driver_query_failures(monkeypatch, tmp_path):
    lib_path = _touch_ctk_file(tmp_path / "cuda-12.9", "12.9.20250531", "targets/x86_64-linux/lib/libnvrtc.so.12")

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


def test_guard_rails_accept_minor_version_compatibility_with_driver_release_branch(monkeypatch, tmp_path):
    lib_path = _touch_ctk_file(tmp_path / "cuda-12.9", "12.9.20250531", "targets/x86_64-linux/lib/libnvrtc.so.12")

    monkeypatch.setattr(compatibility_module, "_load_nvidia_dynamic_lib", lambda _libname: _loaded_dl(lib_path))

    guard_rails = CompatibilityGuardRails(
        driver_cuda_version=_driver_cuda_version(12000),
        driver_release_version=_driver_release_version("525.60.13"),
    )

    loaded = guard_rails.load_nvidia_dynamic_lib("nvrtc")

    assert loaded.abs_path == lib_path


def test_guard_rails_reject_same_major_older_driver_when_release_branch_too_old(monkeypatch, tmp_path):
    lib_path = _touch_ctk_file(tmp_path / "cuda-12.9", "12.9.20250531", "targets/x86_64-linux/lib/libnvrtc.so.12")

    monkeypatch.setattr(compatibility_module, "_load_nvidia_dynamic_lib", lambda _libname: _loaded_dl(lib_path))

    guard_rails = CompatibilityGuardRails(
        driver_cuda_version=_driver_cuda_version(12000),
        driver_release_version=_driver_release_version("520.30.01"),
    )

    with pytest.raises(
        CompatibilityCheckError,
        match=r"branch 520\) is below NVIDIA's published CUDA 12\.x minimum branch >= 525",
    ):
        guard_rails.load_nvidia_dynamic_lib("nvrtc")


def test_guard_rails_require_driver_release_metadata_for_same_major_older_driver(monkeypatch, tmp_path):
    lib_path = _touch_ctk_file(tmp_path / "cuda-12.9", "12.9.20250531", "targets/x86_64-linux/lib/libnvrtc.so.12")

    monkeypatch.setattr(compatibility_module, "_load_nvidia_dynamic_lib", lambda _libname: _loaded_dl(lib_path))

    def fail_query_driver_release_version() -> DriverReleaseVersion:
        raise QueryDriverReleaseVersionError("release query failed")

    monkeypatch.setattr(
        compatibility_module,
        "query_driver_release_version",
        fail_query_driver_release_version,
    )

    guard_rails = CompatibilityGuardRails(driver_cuda_version=_driver_cuda_version(12000))

    with pytest.raises(
        CompatibilityInsufficientMetadataError,
        match="Failed to query the display-driver release version needed for compatibility checks",
    ) as exc_info:
        guard_rails.load_nvidia_dynamic_lib("nvrtc")

    assert isinstance(exc_info.value.__cause__, QueryDriverReleaseVersionError)


def test_find_nvidia_header_directory_returns_none_when_unresolved(monkeypatch):
    monkeypatch.setattr(
        compatibility_module,
        "_locate_nvidia_header_directory",
        lambda _libname: None,
    )

    guard_rails = CompatibilityGuardRails(driver_cuda_version=_driver_cuda_version(13000))

    assert guard_rails.find_nvidia_header_directory("nvrtc") is None
