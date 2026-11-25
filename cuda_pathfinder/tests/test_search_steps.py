# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Tests for the composable search steps and cascade runner."""

from __future__ import annotations

import ctypes
import os
from ctypes import wintypes

import pytest

from cuda.pathfinder import UnsupportedArchError
from cuda.pathfinder._dynamic_libs import search_platform as search_platform_mod
from cuda.pathfinder._dynamic_libs.descriptor_catalog import WindowsSearchDirs
from cuda.pathfinder._dynamic_libs.lib_descriptor import LIB_DESCRIPTORS, LibDescriptor
from cuda.pathfinder._dynamic_libs.load_dl_common import DynamicLibNotFoundError
from cuda.pathfinder._dynamic_libs.search_platform import (
    _find_descriptor_dll_under_dir,
    linux_search_context,
    windows_search_context,
)
from cuda.pathfinder._dynamic_libs.search_steps import (
    EARLY_FIND_STEPS,
    LATE_FIND_STEPS,
    FindResult,
    NotFound,
    SearchContext,
    find_in_conda,
    find_in_cuda_path,
    find_in_install_root_env_vars,
    find_in_program_files_roots,
    find_in_site_packages,
    run_find_steps,
)
from cuda.pathfinder._utils import windows_arch as windows_arch_mod

_STEPS_MOD = "cuda.pathfinder._dynamic_libs.search_steps"
_PLAT_MOD = "cuda.pathfinder._dynamic_libs.search_platform"


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_desc(name: str = "cudart", **overrides) -> LibDescriptor:
    defaults = {
        "name": name,
        "packaged_with": "ctk",
        "linux_sonames": ("libcudart.so",),
        "windows_dlls": ("cudart64_12.dll",),
        "site_packages_linux": (os.path.join("nvidia", "cuda_runtime", "lib"),),
        "site_packages_windows": WindowsSearchDirs(
            x64=(os.path.join("nvidia", "cuda_runtime", "bin"),),
            arm64=(os.path.join("nvidia", "cuda_runtime", "bin"),),
        ),
    }
    defaults.update(overrides)
    return LibDescriptor(**defaults)


def _ctx(desc: LibDescriptor | None = None, *, windows_arch: str | None = None) -> SearchContext:
    desc = desc or _make_desc()
    if windows_arch is None:
        return linux_search_context(desc)
    return windows_search_context(desc, windows_arch)


def _patch_site_packages_search(mocker, root):
    def _find_sub_dirs(sub_dirs):
        path = root.joinpath(*sub_dirs)
        return [str(path)] if path.is_dir() else []

    return mocker.patch(f"{_PLAT_MOD}.find_sub_dirs_all_sitepackages", side_effect=_find_sub_dirs)


def _write_pe(path, machine):
    image = bytearray(0x86)
    image[:2] = b"MZ"
    image[0x3C:0x40] = (0x80).to_bytes(4, "little")
    image[0x80:0x84] = b"PE\0\0"
    image[0x84:0x86] = machine.to_bytes(2, "little")
    path.write_bytes(image)


# ---------------------------------------------------------------------------
# SearchContext
# ---------------------------------------------------------------------------


class TestSearchContext:
    def test_lib_searched_for_linux(self):
        ctx = linux_search_context(LIB_DESCRIPTORS["cublas"])
        assert ctx.lib_searched_for == "libcublas.so.13 or libcublas.so.12"

    def test_lib_searched_for_windows(self):
        ctx = windows_search_context(_make_desc(name="cublas"), "x64")
        assert ctx.lib_searched_for == "known cublas DLL"

    def test_raise_not_found_includes_messages(self):
        ctx = _ctx()
        missed = NotFound(("No such file: libcudart.so",), ('  listdir("/some/dir"):',))
        with pytest.raises(DynamicLibNotFoundError, match="No such file"):
            ctx.raise_not_found(missed)

    def test_raise_not_found_empty_messages(self):
        ctx = _ctx()
        with pytest.raises(DynamicLibNotFoundError):
            ctx.raise_not_found(NotFound())


# ---------------------------------------------------------------------------
# Windows Python architecture detection
# ---------------------------------------------------------------------------


class TestWindowsPythonArch:
    @pytest.mark.agent_authored(model="gpt-5")
    def test_linux_platform_does_not_detect_windows_arch(self, mocker):
        mocker.patch.object(search_platform_mod, "IS_WINDOWS", False)
        get_windows_arch = mocker.patch.object(search_platform_mod, "windows_python_arch")

        ctx = search_platform_mod.search_context(_make_desc())

        assert ctx.windows_arch is None
        get_windows_arch.assert_not_called()

    @pytest.mark.agent_authored(model="gpt-5")
    def test_detects_sysconfig_x64(self, mocker):
        mocker.patch.object(windows_arch_mod.sysconfig, "get_platform", return_value="win-amd64")

        assert windows_arch_mod.windows_python_arch() == "x64"

    @pytest.mark.agent_authored(model="gpt-5")
    def test_detects_sysconfig_arm64(self, mocker):
        mocker.patch.object(windows_arch_mod.sysconfig, "get_platform", return_value="win-arm64")

        assert windows_arch_mod.windows_python_arch() == "arm64"

    @pytest.mark.agent_authored(model="gpt-5")
    def test_rejects_unknown_sysconfig_tag(self, mocker):
        mocker.patch.object(windows_arch_mod.sysconfig, "get_platform", return_value="custom-win")

        with pytest.raises(
            UnsupportedArchError,
            match=r"Unsupported Windows Python platform tag: 'custom-win'.*win-amd64.*win-arm64",
        ) as exc_info:
            windows_arch_mod.windows_python_arch()
        assert exc_info.value.platform_tag == "custom-win"


class TestWindowsMachineArch:
    @pytest.mark.parametrize(
        ("native_machine", "expected"),
        ((0x8664, "x64"), (0xAA64, "arm64")),
    )
    @pytest.mark.agent_authored(model="gpt-5.6")
    def test_uses_native_pe_machine(self, mocker, native_machine, expected):
        mocker.patch.object(windows_arch_mod, "_windows_native_machine", return_value=native_machine)
        platform_machine = mocker.patch.object(windows_arch_mod.platform, "machine", return_value="AMD64")

        assert windows_arch_mod.windows_machine_arch() == expected
        platform_machine.assert_not_called()

    @pytest.mark.agent_authored(model="gpt-5.6")
    def test_rejects_unknown_native_pe_machine(self, mocker):
        mocker.patch.object(windows_arch_mod, "_windows_native_machine", return_value=0x014C)

        with pytest.raises(RuntimeError, match=r"Unsupported native Windows PE machine type: 0x014c"):
            windows_arch_mod.windows_machine_arch()

    @pytest.mark.parametrize(
        ("reported_machine", "expected"),
        (("AMD64", "x64"), ("x86_64", "x64"), ("ARM64", "arm64"), ("aarch64", "arm64")),
    )
    @pytest.mark.agent_authored(model="gpt-5.6")
    def test_old_windows_fallback_normalizes_platform_machine(self, mocker, reported_machine, expected):
        mocker.patch.object(windows_arch_mod, "_windows_native_machine", return_value=None)
        mocker.patch.object(windows_arch_mod.platform, "machine", return_value=reported_machine)

        assert windows_arch_mod.windows_machine_arch() == expected

    @pytest.mark.agent_authored(model="gpt-5.6")
    def test_native_machine_returns_none_when_is_wow64_process2_is_unavailable(self, mocker):
        kernel32 = mocker.Mock(spec=["GetCurrentProcess"])
        mocker.patch.object(ctypes, "WinDLL", create=True, return_value=kernel32)

        assert windows_arch_mod._windows_native_machine() is None

    @pytest.mark.agent_authored(model="gpt-5.6")
    def test_native_machine_configures_api_and_returns_native_machine(self, mocker):
        kernel32 = mocker.Mock()
        kernel32.GetCurrentProcess.return_value = wintypes.HANDLE(1)

        def report_native_machine(_process, _process_machine, native_machine):
            native_machine._obj.value = 0xAA64
            return True

        kernel32.IsWow64Process2.side_effect = report_native_machine
        mocker.patch.object(ctypes, "WinDLL", create=True, return_value=kernel32)

        assert windows_arch_mod._windows_native_machine() == 0xAA64
        assert kernel32.GetCurrentProcess.argtypes == ()
        assert kernel32.GetCurrentProcess.restype is wintypes.HANDLE
        assert kernel32.IsWow64Process2.argtypes == (
            wintypes.HANDLE,
            ctypes.POINTER(wintypes.USHORT),
            ctypes.POINTER(wintypes.USHORT),
        )
        assert kernel32.IsWow64Process2.restype is wintypes.BOOL

    @pytest.mark.agent_authored(model="gpt-5.6")
    def test_native_machine_raises_contextual_error_when_api_call_fails(self, mocker):
        kernel32 = mocker.Mock()
        kernel32.GetCurrentProcess.return_value = wintypes.HANDLE(1)
        kernel32.IsWow64Process2.return_value = False
        mocker.patch.object(ctypes, "WinDLL", create=True, return_value=kernel32)
        mocker.patch.object(ctypes, "get_last_error", create=True, return_value=87)
        windows_error = OSError(87, "The parameter is incorrect")
        mocker.patch.object(ctypes, "WinError", create=True, return_value=windows_error)

        with pytest.raises(
            RuntimeError,
            match=r"IsWow64Process2 failed while detecting the native Windows architecture \(Windows error 87\)",
        ) as exc_info:
            windows_arch_mod._windows_native_machine()

        assert exc_info.value.__cause__ is windows_error


@pytest.mark.parametrize(
    ("machine", "target_arch", "expected"),
    (
        (0x8664, "x64", True),
        (0x8664, "arm64", False),
        (0xAA64, "x64", False),
        (0xAA64, "arm64", True),
    ),
)
@pytest.mark.agent_authored(model="gpt-5")
def test_windows_pe_matches_arch(tmp_path, machine, target_arch, expected):
    dll = tmp_path / "test.dll"
    _write_pe(dll, machine)

    assert windows_arch_mod.windows_pe_matches_arch(str(dll), target_arch) is expected


@pytest.mark.agent_authored(model="gpt-5")
def test_windows_pe_matches_arch_rejects_malformed_file(tmp_path):
    dll = tmp_path / "test.dll"
    dll.write_bytes(b"not a PE file")

    assert windows_arch_mod.windows_pe_matches_arch(str(dll), "x64") is False


# ---------------------------------------------------------------------------
# find_in_site_packages
# ---------------------------------------------------------------------------


class TestFindInSitePackages:
    def test_misses_when_no_rel_dirs(self):
        desc = _make_desc(site_packages_linux=(), site_packages_windows=WindowsSearchDirs())
        result = find_in_site_packages(_ctx(desc))
        assert isinstance(result, NotFound)

    def test_found_linux(self, mocker, tmp_path):
        lib_dir = tmp_path / "nvidia" / "cuda_runtime" / "lib"
        lib_dir.mkdir(parents=True)
        so_file = lib_dir / "libcudart.so"
        so_file.touch()

        mocker.patch(
            f"{_PLAT_MOD}.find_sub_dirs_all_sitepackages",
            return_value=[str(lib_dir)],
        )

        desc = _make_desc(
            site_packages_linux=(os.path.join("nvidia", "cuda_runtime", "lib"),),
        )
        result = find_in_site_packages(_ctx(desc))
        assert isinstance(result, FindResult)
        assert result.abs_path == str(so_file)
        assert result.found_via == "site-packages"

    @pytest.mark.agent_authored(model="gpt-5.6-sol")
    def test_nvvm_linux_wheel_accepts_declared_unversioned_filename(self, mocker, tmp_path):
        lib_dir = tmp_path / "nvidia" / "cuda_nvcc" / "nvvm" / "lib64"
        lib_dir.mkdir(parents=True)
        so_file = lib_dir / "libnvvm.so"
        so_file.touch()

        mocker.patch(
            f"{_PLAT_MOD}.find_sub_dirs_all_sitepackages",
            return_value=[str(lib_dir)],
        )

        result = find_in_site_packages(_ctx(LIB_DESCRIPTORS["nvvm"]))

        assert isinstance(result, FindResult)
        assert result.abs_path == str(so_file)
        assert result.found_via == "site-packages"

    def test_found_windows(self, mocker, tmp_path):
        bin_dir = tmp_path / "nvidia" / "cuda_runtime" / "bin"
        bin_dir.mkdir(parents=True)
        dll = bin_dir / "cudart64_12.dll"
        dll.touch()

        mocker.patch(
            f"{_PLAT_MOD}.find_sub_dirs_all_sitepackages",
            return_value=[str(bin_dir)],
        )
        mocker.patch(f"{_PLAT_MOD}.is_suppressed_dll_file", return_value=False)

        desc = _make_desc(
            name="cudart",
            site_packages_windows=WindowsSearchDirs(
                x64=(os.path.join("nvidia", "cuda_runtime", "bin"),),
                arm64=(os.path.join("nvidia", "cuda_runtime", "bin"),),
            ),
        )
        result = find_in_site_packages(_ctx(desc, windows_arch="x64"))
        assert isinstance(result, FindResult)
        assert result.abs_path == str(dll)
        assert result.found_via == "site-packages"

    @pytest.mark.agent_authored(model="gpt-5")
    def test_windows_exact_names_reject_cudnn_sidecar(self, mocker, tmp_path):
        bin_dir = tmp_path / "nvidia" / "cudnn" / "bin"
        bin_dir.mkdir(parents=True)
        _write_pe(bin_dir / "cudnn_adv64_9.dll", windows_arch_mod.WINDOWS_PE_MACHINE_BY_ARCH["x64"])

        mocker.patch(
            f"{_PLAT_MOD}.find_sub_dirs_all_sitepackages",
            return_value=[str(bin_dir)],
        )

        result = find_in_site_packages(_ctx(LIB_DESCRIPTORS["cudnn"], windows_arch="x64"))

        assert isinstance(result, NotFound)

    @pytest.mark.agent_authored(model="gpt-5.6-sol")
    def test_windows_matching_rejects_unlisted_cupti_name(self, mocker, tmp_path):
        bin_dir = tmp_path / "nvidia" / "cuda_cupti" / "bin"
        bin_dir.mkdir(parents=True)
        (bin_dir / "cupti64_14.dll").touch()

        mocker.patch(
            f"{_PLAT_MOD}.find_sub_dirs_all_sitepackages",
            return_value=[str(bin_dir)],
        )

        result = find_in_site_packages(_ctx(LIB_DESCRIPTORS["cupti"], windows_arch="x64"))

        assert isinstance(result, NotFound)

    @pytest.mark.agent_authored(model="gpt-5.6-sol")
    def test_windows_known_cupti_names_follow_declared_preference_order(self, tmp_path):
        preferred_dll = tmp_path / "cupti64_2026.3.0.dll"
        fallback_dll = tmp_path / "cupti64_2026.2.1.dll"
        preferred_dll.touch()
        fallback_dll.touch()

        result = _find_descriptor_dll_under_dir(str(tmp_path), LIB_DESCRIPTORS["cupti"])

        assert result == str(preferred_dll)

    @pytest.mark.parametrize(
        ("libname", "sibling_dll"),
        (
            ("cublas", "cublasLt64_13.dll"),
            ("cufft", "cufftw64_12.dll"),
            ("cusolver", "cusolverMg64_12.dll"),
            ("cusparse", "cusparseLt.dll"),
            ("cutensor", "cutensorMg.dll"),
        ),
    )
    @pytest.mark.agent_authored(model="gpt-5.6-sol")
    def test_windows_matching_rejects_sibling_library(self, tmp_path, libname, sibling_dll):
        (tmp_path / sibling_dll).touch()

        result = _find_descriptor_dll_under_dir(str(tmp_path), LIB_DESCRIPTORS[libname])

        assert result is None

    @pytest.mark.agent_authored(model="gpt-5")
    def test_found_windows_arm64_prefers_cuda13_arch_dir_to_cuda12(self, mocker, tmp_path):
        x86_64_dir = tmp_path / "nvidia" / "cu13" / "bin" / "x86_64"
        arm64_dir = tmp_path / "nvidia" / "cu13" / "bin" / "arm64"
        cuda12_dir = tmp_path / "nvidia" / "cuda_runtime" / "bin"
        x86_64_dir.mkdir(parents=True)
        arm64_dir.mkdir(parents=True)
        cuda12_dir.mkdir(parents=True)
        (x86_64_dir / "cudart64_12.dll").touch()
        (cuda12_dir / "cudart64_12.dll").touch()
        arm64_dll = arm64_dir / "cudart64_12.dll"
        arm64_dll.touch()

        _patch_site_packages_search(mocker, tmp_path)
        mocker.patch(f"{_PLAT_MOD}.is_suppressed_dll_file", return_value=False)

        desc = LIB_DESCRIPTORS["cudart"]
        result = find_in_site_packages(_ctx(desc, windows_arch="arm64"))
        assert isinstance(result, FindResult)
        assert result.abs_path == str(arm64_dll)
        assert result.found_via == "site-packages"

    @pytest.mark.agent_authored(model="gpt-5")
    def test_found_windows_x64_prefers_cuda13_arch_dir_to_cuda12(self, mocker, tmp_path):
        x86_64_dir = tmp_path / "nvidia" / "cu13" / "bin" / "x86_64"
        arm64_dir = tmp_path / "nvidia" / "cu13" / "bin" / "arm64"
        cuda12_dir = tmp_path / "nvidia" / "cuda_runtime" / "bin"
        x86_64_dir.mkdir(parents=True)
        arm64_dir.mkdir(parents=True)
        cuda12_dir.mkdir(parents=True)
        x86_64_dll = x86_64_dir / "cudart64_12.dll"
        x86_64_dll.touch()
        (arm64_dir / "cudart64_12.dll").touch()
        (cuda12_dir / "cudart64_12.dll").touch()

        _patch_site_packages_search(mocker, tmp_path)
        mocker.patch(f"{_PLAT_MOD}.is_suppressed_dll_file", return_value=False)

        desc = LIB_DESCRIPTORS["cudart"]
        result = find_in_site_packages(_ctx(desc, windows_arch="x64"))
        assert isinstance(result, FindResult)
        assert result.abs_path == str(x86_64_dll)
        assert result.found_via == "site-packages"

    @pytest.mark.agent_authored(model="gpt-5")
    def test_found_windows_x64_uses_cuda12_when_cuda13_is_absent(self, mocker, tmp_path):
        cuda12_dir = tmp_path / "nvidia" / "cuda_runtime" / "bin"
        cuda12_dir.mkdir(parents=True)
        cuda12_dll = cuda12_dir / "cudart64_12.dll"
        cuda12_dll.touch()

        _patch_site_packages_search(mocker, tmp_path)
        mocker.patch(f"{_PLAT_MOD}.is_suppressed_dll_file", return_value=False)

        desc = LIB_DESCRIPTORS["cudart"]
        result = find_in_site_packages(_ctx(desc, windows_arch="x64"))
        assert isinstance(result, FindResult)
        assert result.abs_path == str(cuda12_dll)
        assert result.found_via == "site-packages"

    @pytest.mark.agent_authored(model="gpt-5")
    def test_found_windows_arm64_skips_cuda12_when_cuda13_is_absent(self, mocker, tmp_path):
        cuda12_dir = tmp_path / "nvidia" / "cuda_runtime" / "bin"
        cuda12_dir.mkdir(parents=True)
        (cuda12_dir / "cudart64_12.dll").touch()

        _patch_site_packages_search(mocker, tmp_path)
        mocker.patch(f"{_PLAT_MOD}.is_suppressed_dll_file", return_value=False)

        desc = LIB_DESCRIPTORS["cudart"]
        ctx = _ctx(desc, windows_arch="arm64")
        assert ctx.site_packages_rel_dirs == ("nvidia/cu13/bin/arm64",)

        result = find_in_site_packages(ctx)

        assert isinstance(result, NotFound)

    @pytest.mark.parametrize(
        ("target_arch", "wheel_arch", "expected_source"),
        (
            ("x64", "x64", "site-packages"),
            ("arm64", "arm64", "site-packages"),
            ("arm64", "x64", "CUDNN_PATH"),
        ),
    )
    @pytest.mark.agent_authored(model="gpt-6")
    def test_cudnn_wheel_architecture_and_archive_fallback(
        self, mocker, tmp_path, target_arch, wheel_arch, expected_source
    ):
        wheel_dir = tmp_path / "nvidia" / "cudnn" / "bin"
        wheel_dir.mkdir(parents=True)
        cudnn_dll = wheel_dir / "cudnn64_9.dll"
        _write_pe(cudnn_dll, windows_arch_mod.WINDOWS_PE_MACHINE_BY_ARCH[wheel_arch])

        archive_root = tmp_path / "archive"
        archive_dll = archive_root / "bin" / target_arch / "cudnn64_9.dll"
        archive_dll.parent.mkdir(parents=True)
        _write_pe(archive_dll, windows_arch_mod.WINDOWS_PE_MACHINE_BY_ARCH[target_arch])
        mocker.patch.dict(os.environ, {"CUDNN_PATH": str(archive_root)})

        _patch_site_packages_search(mocker, tmp_path)
        mocker.patch(f"{_PLAT_MOD}.is_suppressed_dll_file", return_value=False)

        desc = LIB_DESCRIPTORS["cudnn"]
        result = run_find_steps(
            _ctx(desc, windows_arch=target_arch),
            (find_in_site_packages, find_in_install_root_env_vars),
        )

        expected_dll = cudnn_dll if expected_source == "site-packages" else archive_dll
        assert result == FindResult(str(expected_dll), expected_source)

    def test_not_found_appends_error(self, mocker, tmp_path):
        empty_dir = tmp_path / "nvidia" / "cuda_runtime" / "lib"
        empty_dir.mkdir(parents=True)

        mocker.patch(
            f"{_PLAT_MOD}.find_sub_dirs_all_sitepackages",
            return_value=[str(empty_dir)],
        )

        ctx = _ctx()
        result = find_in_site_packages(ctx)
        assert isinstance(result, NotFound)
        assert any("No such file" in m for m in result.error_messages)

    @pytest.mark.agent_authored(model="gpt-5.6-sol")
    def test_linux_declared_versioned_soname_does_not_need_unversioned_link(self, mocker, tmp_path):
        lib_dir = tmp_path / "nvidia" / "cuda_runtime" / "lib"
        lib_dir.mkdir(parents=True)
        versioned = lib_dir / "libcudart.so.13"
        versioned.touch()

        mocker.patch(
            f"{_PLAT_MOD}.find_sub_dirs_all_sitepackages",
            return_value=[str(lib_dir)],
        )

        desc = _make_desc(linux_sonames=("libcudart.so.13", "libcudart.so.12"))
        result = find_in_site_packages(_ctx(desc))
        assert isinstance(result, FindResult)
        assert result.abs_path == str(versioned)
        assert result.found_via == "site-packages"

    @pytest.mark.agent_authored(model="gpt-5.6-sol")
    def test_linux_declared_sonames_follow_preference_order(self, mocker, tmp_path):
        lib_dir = tmp_path / "nvidia" / "cuda_runtime" / "lib"
        lib_dir.mkdir(parents=True)
        preferred = lib_dir / "libcudart.so.13"
        fallback = lib_dir / "libcudart.so.12"
        preferred.touch()
        fallback.touch()

        mocker.patch(
            f"{_PLAT_MOD}.find_sub_dirs_all_sitepackages",
            return_value=[str(lib_dir)],
        )

        desc = _make_desc(linux_sonames=("libcudart.so.13", "libcudart.so.12"))
        result = find_in_site_packages(_ctx(desc))
        assert isinstance(result, FindResult)
        assert result.abs_path == str(preferred)
        assert result.found_via == "site-packages"

    @pytest.mark.parametrize(
        "undeclared_filename",
        ("libcudart.so", "libcudart.so.14", "libcudart.so.13.5.0", "libcudart.so.13.backup"),
    )
    @pytest.mark.agent_authored(model="gpt-5.6-sol")
    def test_linux_rejects_undeclared_filename(self, mocker, tmp_path, undeclared_filename):
        lib_dir = tmp_path / "nvidia" / "cuda_runtime" / "lib"
        lib_dir.mkdir(parents=True)
        (lib_dir / undeclared_filename).touch()

        mocker.patch(
            f"{_PLAT_MOD}.find_sub_dirs_all_sitepackages",
            return_value=[str(lib_dir)],
        )

        desc = _make_desc(linux_sonames=("libcudart.so.13", "libcudart.so.12"))
        ctx = _ctx(desc)
        result = find_in_site_packages(ctx)
        assert isinstance(result, NotFound)
        assert result.error_messages
        assert all("*" not in message for message in result.error_messages)


# ---------------------------------------------------------------------------
# find_in_conda
# ---------------------------------------------------------------------------


class TestFindInConda:
    def test_misses_without_conda_prefix(self, mocker):
        mocker.patch.dict(os.environ, {}, clear=True)
        assert isinstance(find_in_conda(_ctx()), NotFound)

    def test_misses_with_empty_conda_prefix(self, mocker):
        mocker.patch.dict(os.environ, {"CONDA_PREFIX": ""})
        assert isinstance(find_in_conda(_ctx()), NotFound)

    def test_found_linux(self, mocker, tmp_path):
        lib_dir = tmp_path / "lib"
        lib_dir.mkdir()
        so_file = lib_dir / "libcudart.so"
        so_file.touch()

        mocker.patch.dict(os.environ, {"CONDA_PREFIX": str(tmp_path)})

        result = find_in_conda(_ctx())
        assert isinstance(result, FindResult)
        assert result.abs_path == str(so_file)
        assert result.found_via == "conda"

    def test_found_windows(self, mocker, tmp_path):
        bin_dir = tmp_path / "Library" / "bin"
        bin_dir.mkdir(parents=True)
        dll = bin_dir / "cudart64_12.dll"
        dll.touch()

        mocker.patch.dict(os.environ, {"CONDA_PREFIX": str(tmp_path)})

        result = find_in_conda(_ctx(windows_arch="x64"))
        assert isinstance(result, FindResult)
        assert result.abs_path == str(dll)
        assert result.found_via == "conda"

    @pytest.mark.agent_authored(model="gpt-5")
    def test_found_windows_arm64_prefers_arch_dir(self, mocker, tmp_path):
        x64_dir = tmp_path / "Library" / "bin" / "x64"
        arm64_dir = tmp_path / "Library" / "bin" / "arm64"
        x64_dir.mkdir(parents=True)
        arm64_dir.mkdir(parents=True)
        (x64_dir / "cudart64_12.dll").touch()
        arm64_dll = arm64_dir / "cudart64_12.dll"
        arm64_dll.touch()

        mocker.patch.dict(os.environ, {"CONDA_PREFIX": str(tmp_path)})

        result = find_in_conda(_ctx(windows_arch="arm64"))
        assert isinstance(result, FindResult)
        assert result.abs_path == str(arm64_dll)
        assert result.found_via == "conda"

    @pytest.mark.agent_authored(model="gpt-5.6-sol")
    def test_linux_declared_versioned_soname_found_in_lib_dir(self, mocker, tmp_path):
        lib_dir = tmp_path / "lib"
        lib_dir.mkdir()
        versioned = lib_dir / "libcudart.so.13"
        versioned.touch()

        mocker.patch.dict(os.environ, {"CONDA_PREFIX": str(tmp_path)})

        desc = _make_desc(linux_sonames=("libcudart.so.13", "libcudart.so.12"))
        result = find_in_conda(_ctx(desc))
        assert isinstance(result, FindResult)
        assert result.abs_path == str(versioned)
        assert result.found_via == "conda"

    @pytest.mark.parametrize(
        "undeclared_filename",
        ("libcudart.so", "libcudart.so.14", "libcudart.so.13.5.0", "libcudart.so.13.backup"),
    )
    @pytest.mark.agent_authored(model="gpt-5.6-sol")
    def test_linux_rejects_undeclared_filename_in_lib_dir(self, mocker, tmp_path, undeclared_filename):
        lib_dir = tmp_path / "lib"
        lib_dir.mkdir()
        (lib_dir / undeclared_filename).touch()

        mocker.patch.dict(os.environ, {"CONDA_PREFIX": str(tmp_path)})

        desc = _make_desc(linux_sonames=("libcudart.so.13", "libcudart.so.12"))
        ctx = _ctx(desc)
        result = find_in_conda(ctx)
        assert isinstance(result, NotFound)
        assert result.error_messages
        assert all("*" not in message for message in result.error_messages)


# ---------------------------------------------------------------------------
# find_in_cuda_path
# ---------------------------------------------------------------------------


class TestFindInCudaHome:
    def test_misses_without_env_var(self, mocker):
        mocker.patch(f"{_STEPS_MOD}.get_cuda_path_or_home", return_value=None)
        assert isinstance(find_in_cuda_path(_ctx()), NotFound)

    def test_found_linux(self, mocker, tmp_path):
        lib_dir = tmp_path / "lib64"
        lib_dir.mkdir()
        so_file = lib_dir / "libcudart.so"
        so_file.touch()

        mocker.patch(f"{_STEPS_MOD}.get_cuda_path_or_home", return_value=str(tmp_path))

        result = find_in_cuda_path(_ctx())
        assert isinstance(result, FindResult)
        assert result.abs_path == str(so_file)
        assert result.found_via == "CUDA_PATH"

    @pytest.mark.skipif(os.name == "nt", reason="POSIX symlink parent traversal")
    @pytest.mark.agent_authored(model="gpt-6")
    def test_symlink_parent_root_selects_correct_library(self, mocker, tmp_path):
        sdk = tmp_path / "installed" / "sdk"
        (sdk / "leaf").mkdir(parents=True)
        lib_dir = sdk / "lib64"
        lib_dir.mkdir()
        selected_library = lib_dir / "libcudart.so"
        selected_library.touch()
        alias = tmp_path / "alias"
        alias.symlink_to(sdk / "leaf", target_is_directory=True)

        wrong_lib_dir = tmp_path / "lib64"
        wrong_lib_dir.mkdir()
        (wrong_lib_dir / "libcudart.so").touch()
        mocker.patch(f"{_STEPS_MOD}.get_cuda_path_or_home", return_value=str(alias / ".."))

        result = find_in_cuda_path(_ctx())

        assert isinstance(result, FindResult)
        assert result.found_via == "CUDA_PATH"
        assert os.path.samefile(result.abs_path, selected_library)

    def test_found_windows(self, mocker, tmp_path):
        bin_dir = tmp_path / "bin"
        bin_dir.mkdir()
        dll = bin_dir / "cudart64_12.dll"
        dll.touch()

        mocker.patch(f"{_STEPS_MOD}.get_cuda_path_or_home", return_value=str(tmp_path))

        result = find_in_cuda_path(_ctx(windows_arch="x64"))
        assert isinstance(result, FindResult)
        assert result.abs_path == str(dll)
        assert result.found_via == "CUDA_PATH"

    @pytest.mark.agent_authored(model="gpt-5")
    def test_found_windows_arm64_prefers_arch_dir(self, mocker, tmp_path):
        x64_dir = tmp_path / "bin" / "x64"
        arm64_dir = tmp_path / "bin" / "arm64"
        x64_dir.mkdir(parents=True)
        arm64_dir.mkdir(parents=True)
        (x64_dir / "cudart64_12.dll").touch()
        arm64_dll = arm64_dir / "cudart64_12.dll"
        arm64_dll.touch()

        mocker.patch(f"{_STEPS_MOD}.get_cuda_path_or_home", return_value=str(tmp_path))

        result = find_in_cuda_path(_ctx(windows_arch="arm64"))
        assert isinstance(result, FindResult)
        assert result.abs_path == str(arm64_dll)
        assert result.found_via == "CUDA_PATH"

    @pytest.mark.parametrize(
        ("target_arch", "machine", "expected_found"),
        (
            ("x64", 0x8664, True),
            ("x64", 0xAA64, False),
            ("arm64", 0x8664, False),
            ("arm64", 0xAA64, True),
        ),
    )
    @pytest.mark.agent_authored(model="gpt-5")
    def test_nvvm_windows_checks_binary_arch(self, mocker, tmp_path, target_arch, machine, expected_found):
        nvvm_dir = tmp_path / "nvvm" / "bin"
        nvvm_dir.mkdir(parents=True)
        dll = nvvm_dir / "nvvm64_40_0.dll"
        _write_pe(dll, machine)

        mocker.patch(f"{_STEPS_MOD}.get_cuda_path_or_home", return_value=str(tmp_path))

        ctx = _ctx(LIB_DESCRIPTORS["nvvm"], windows_arch=target_arch)
        result = find_in_cuda_path(ctx)

        assert isinstance(result, FindResult) is expected_found
        if expected_found:
            assert isinstance(result, FindResult)
            assert result.abs_path == str(dll)
            assert result.found_via == "CUDA_PATH"
        else:
            assert any(f"No {target_arch}-compatible PE file" in message for message in result.error_messages)


# ---------------------------------------------------------------------------
# Descriptor-specific Linux install roots
# ---------------------------------------------------------------------------


class TestLinuxInstallRoots:
    @pytest.mark.parametrize(
        ("libname", "env_var", "rel_dir", "soname"),
        (
            ("cudnn", "CUDNN_PATH", "lib", "libcudnn.so.9"),
            ("cudnn", "CUDNN_PATH", "lib64", "libcudnn.so.9"),
            ("nccl", "NCCL_HOME", "lib", "libnccl.so.2"),
            ("nccl", "NCCL_HOME", "lib64", "libnccl.so.2"),
            ("nccl", "NCCL_HOME", "build/lib", "libnccl.so.2"),
        ),
    )
    @pytest.mark.agent_authored(model="gpt-5.6-sol")
    def test_product_root_finds_supported_layouts(self, mocker, tmp_path, libname, env_var, rel_dir, soname):
        root = tmp_path / libname
        lib_dir = root / rel_dir
        lib_dir.mkdir(parents=True)
        library = lib_dir / soname
        library.touch()
        env = {"CUDNN_PATH": "", "NCCL_HOME": ""}
        env[env_var] = str(root)
        mocker.patch.dict(os.environ, env)

        result = find_in_install_root_env_vars(_ctx(LIB_DESCRIPTORS[libname]))

        assert result == FindResult(str(library), env_var)

    @pytest.mark.agent_authored(model="gpt-5.6-sol")
    def test_nccl_home_prefers_canonical_installed_layout(self, mocker, tmp_path):
        libraries = []
        for rel_dir in ("lib", "lib64", "build/lib"):
            lib_dir = tmp_path / rel_dir
            lib_dir.mkdir(parents=True)
            library = lib_dir / "libnccl.so.2"
            library.touch()
            libraries.append(library)
        mocker.patch.dict(os.environ, {"CUDNN_PATH": "", "NCCL_HOME": str(tmp_path)})

        result = find_in_install_root_env_vars(_ctx(LIB_DESCRIPTORS["nccl"]))

        assert result == FindResult(str(libraries[0]), "NCCL_HOME")


# ---------------------------------------------------------------------------
# Descriptor-specific Windows install roots
# ---------------------------------------------------------------------------


class TestWindowsInstallRoots:
    @pytest.mark.parametrize(
        ("target_arch", "archive_bin_rel_dir"),
        [
            ("x64", "bin/x64"),
            ("x64", "bin"),
            ("arm64", "bin/arm64"),
        ],
    )
    @pytest.mark.agent_authored(model="gpt-5")
    def test_cudnn_path_finds_supported_archive_layouts(self, mocker, tmp_path, target_arch, archive_bin_rel_dir):
        bin_dir = tmp_path / archive_bin_rel_dir
        bin_dir.mkdir(parents=True)
        dll = bin_dir / "cudnn64_9.dll"
        _write_pe(dll, windows_arch_mod.WINDOWS_PE_MACHINE_BY_ARCH[target_arch])
        mocker.patch.dict(os.environ, {"CUDNN_PATH": str(tmp_path)})

        result = find_in_install_root_env_vars(_ctx(LIB_DESCRIPTORS["cudnn"], windows_arch=target_arch))

        assert result == FindResult(str(dll), "CUDNN_PATH")

    @pytest.mark.parametrize(
        ("target_arch", "other_arch_bin_rel_dir"),
        [
            ("x64", "bin/arm64"),
            ("arm64", "bin/x64"),
            ("arm64", "bin"),
        ],
    )
    @pytest.mark.agent_authored(model="gpt-5")
    def test_cudnn_path_does_not_cross_architectures(self, mocker, tmp_path, target_arch, other_arch_bin_rel_dir):
        bin_dir = tmp_path / other_arch_bin_rel_dir
        bin_dir.mkdir(parents=True)
        _write_pe(bin_dir / "cudnn64_9.dll", windows_arch_mod.WINDOWS_PE_MACHINE_BY_ARCH[target_arch])
        mocker.patch.dict(os.environ, {"CUDNN_PATH": str(tmp_path)})

        result = find_in_install_root_env_vars(_ctx(LIB_DESCRIPTORS["cudnn"], windows_arch=target_arch))

        assert isinstance(result, NotFound)

    @pytest.mark.agent_authored(model="gpt-5")
    def test_program_files_finds_versioned_cudnn_install(self, mocker, tmp_path):
        bin_dir = tmp_path / "NVIDIA" / "CUDNN" / "v9.24" / "bin"
        x64_bin_dir = bin_dir / "x64"
        x64_bin_dir.mkdir(parents=True)
        _write_pe(x64_bin_dir / "cudnn_adv64_9.dll", windows_arch_mod.WINDOWS_PE_MACHINE_BY_ARCH["x64"])
        dll = bin_dir / "cudnn64_9.dll"
        _write_pe(dll, windows_arch_mod.WINDOWS_PE_MACHINE_BY_ARCH["x64"])
        mocker.patch.dict(os.environ, {"PROGRAMFILES": str(tmp_path), "PROGRAMW6432": ""})

        result = find_in_program_files_roots(_ctx(LIB_DESCRIPTORS["cudnn"], windows_arch="x64"))

        assert result == FindResult(str(dll), "ProgramFiles")

    @pytest.mark.agent_authored(model="gpt-5")
    def test_program_files_prefers_newest_numeric_cudnn_version_across_layouts(self, mocker, tmp_path):
        older_dir = tmp_path / "NVIDIA" / "CUDNN" / "v9.9" / "bin" / "x64"
        newer_dir = tmp_path / "NVIDIA" / "CUDNN" / "v9.10" / "bin"
        older_dir.mkdir(parents=True)
        newer_dir.mkdir(parents=True)
        _write_pe(older_dir / "cudnn64_9.dll", windows_arch_mod.WINDOWS_PE_MACHINE_BY_ARCH["x64"])
        newer_dll = newer_dir / "cudnn64_9.dll"
        _write_pe(newer_dll, windows_arch_mod.WINDOWS_PE_MACHINE_BY_ARCH["x64"])
        mocker.patch.dict(os.environ, {"PROGRAMFILES": str(tmp_path), "PROGRAMW6432": ""})

        result = find_in_program_files_roots(_ctx(LIB_DESCRIPTORS["cudnn"], windows_arch="x64"))

        assert result == FindResult(str(newer_dll), "ProgramFiles")

    @pytest.mark.agent_authored(model="gpt-5")
    def test_cudnn_arm64_archive_layout_is_not_assumed_for_other_roots(self, mocker, tmp_path):
        conda_root = tmp_path / "conda"
        cuda_root = tmp_path / "cuda"
        program_files_root = tmp_path / "Program Files"
        for bin_dir in (
            conda_root / "Library" / "bin" / "arm64",
            cuda_root / "bin" / "arm64",
            program_files_root / "NVIDIA" / "CUDNN" / "v9.25" / "bin" / "arm64",
        ):
            bin_dir.mkdir(parents=True)
            _write_pe(bin_dir / "cudnn64_9.dll", windows_arch_mod.WINDOWS_PE_MACHINE_BY_ARCH["arm64"])
        mocker.patch.dict(
            os.environ,
            {
                "CONDA_PREFIX": str(conda_root),
                "PROGRAMFILES": str(program_files_root),
                "PROGRAMW6432": "",
            },
        )
        mocker.patch(f"{_STEPS_MOD}.get_cuda_path_or_home", return_value=str(cuda_root))
        ctx = _ctx(LIB_DESCRIPTORS["cudnn"], windows_arch="arm64")

        assert isinstance(find_in_conda(ctx), NotFound)
        assert isinstance(find_in_cuda_path(ctx), NotFound)
        assert isinstance(find_in_program_files_roots(ctx), NotFound)
        assert ctx.site_packages_rel_dirs == ("nvidia/cudnn/bin",)


# ---------------------------------------------------------------------------
# run_find_steps
# ---------------------------------------------------------------------------


class TestRunFindSteps:
    def test_returns_first_hit(self):
        hit = FindResult("/path/to/lib.so", "step-a")

        def step_a(_ctx):
            return hit

        def step_b(_ctx):
            raise AssertionError("step_b should not be called")

        result = run_find_steps(_ctx(), (step_a, step_b))
        assert result is hit

    def test_returns_not_found_when_all_miss(self):
        result = run_find_steps(_ctx(), (lambda _: NotFound(), lambda _: NotFound()))
        assert result == NotFound()

    @pytest.mark.agent_authored(model="claude-fable-5-2")
    def test_combines_misses_in_step_order(self):
        step_a_miss = NotFound(("a1", "a2"), ("listing a",))
        step_b_miss = NotFound(("b1",), ("listing b",))

        result = run_find_steps(_ctx(), (lambda _: step_a_miss, lambda _: NotFound(), lambda _: step_b_miss))

        assert result == NotFound(("a1", "a2", "b1"), ("listing a", "listing b"))

    def test_empty_steps(self):
        assert run_find_steps(_ctx(), ()) == NotFound()

    @pytest.mark.agent_authored(model="claude-fable-5-2")
    def test_both_result_types_say_whether_they_found_something(self):
        assert FindResult("/lib/libfoo.so", "conda").found is True
        assert NotFound(("No such file: libfoo.so",)).found is False

    def test_skips_misses_returns_later_hit(self):
        hit = FindResult("/later/lib.so", "step-c")
        result = run_find_steps(_ctx(), (lambda _: NotFound(), lambda _: hit))
        assert result is hit


# ---------------------------------------------------------------------------
# Step tuple sanity checks
# ---------------------------------------------------------------------------


class TestStepTuples:
    def test_early_find_steps_contains_expected(self):
        assert find_in_site_packages in EARLY_FIND_STEPS
        assert find_in_conda in EARLY_FIND_STEPS

    def test_late_find_steps_contains_expected(self):
        assert find_in_install_root_env_vars in LATE_FIND_STEPS
        assert find_in_cuda_path in LATE_FIND_STEPS
        assert find_in_program_files_roots in LATE_FIND_STEPS

    @pytest.mark.agent_authored(model="gpt-5")
    def test_late_find_steps_use_specific_roots_before_generic_roots(self):
        assert (
            find_in_install_root_env_vars,
            find_in_cuda_path,
            find_in_program_files_roots,
        ) == LATE_FIND_STEPS

    def test_early_and_late_are_disjoint(self):
        assert not set(EARLY_FIND_STEPS) & set(LATE_FIND_STEPS)


# ---------------------------------------------------------------------------
# Data-driven anchor paths
# ---------------------------------------------------------------------------


class TestAnchorRelDirs:
    """Verify that descriptor anchor paths drive directory resolution."""

    @pytest.mark.agent_authored(model="gpt-5")
    def test_windows_search_dirs_arch_only_constructors(self):
        assert WindowsSearchDirs.x64_only("first", "second") == WindowsSearchDirs(x64=("first", "second"))
        assert WindowsSearchDirs.arm64_only("first", "second") == WindowsSearchDirs(arm64=("first", "second"))

    def test_nvvm_has_custom_linux_paths(self):
        desc = LIB_DESCRIPTORS["nvvm"]
        assert desc.anchor_rel_dirs_linux == ("nvvm/lib64",)

    def test_nvvm_has_custom_windows_paths(self):
        desc = LIB_DESCRIPTORS["nvvm"]
        assert desc.anchor_rel_dirs_windows.for_arch("x64") == ("nvvm/bin/x64", "nvvm/bin")
        assert desc.anchor_rel_dirs_windows.for_arch("arm64") == ("nvvm/bin",)

    @pytest.mark.agent_authored(model="gpt-5")
    def test_cupti_has_custom_windows_paths(self):
        desc = LIB_DESCRIPTORS["cupti"]
        assert desc.anchor_rel_dirs_windows.for_arch("x64") == (
            "extras/CUPTI/lib/x64",
            "extras/CUPTI/lib64",
            "bin",
        )
        assert desc.anchor_rel_dirs_windows.for_arch("arm64") == ("extras/CUPTI/lib/arm64",)

    @pytest.mark.parametrize("libname", ["cudart", "cublas", "nvrtc"])
    def test_regular_ctk_libs_use_defaults(self, libname):
        desc = LIB_DESCRIPTORS[libname]
        assert desc.anchor_rel_dirs_linux == ("lib64", "lib")
        assert desc.anchor_rel_dirs_windows.for_arch("x64") == ("bin/x64", "bin")
        assert desc.anchor_rel_dirs_windows.for_arch("arm64") == ("bin/arm64",)

    @pytest.mark.agent_authored(model="gpt-5")
    def test_cudla_uses_arm64_only_windows_anchor(self):
        desc = LIB_DESCRIPTORS["cudla"]

        assert desc.anchor_rel_dirs_windows.for_arch("x64") == ()
        assert desc.anchor_rel_dirs_windows.for_arch("arm64") == ("bin/arm64",)

    @pytest.mark.agent_authored(model="gpt-5")
    def test_windows_anchor_dirs_select_arm64(self):
        desc = _make_desc(
            anchor_rel_dirs_windows=WindowsSearchDirs(
                x64=("bin/x64", "bin"),
                arm64=("bin/arm64", "bin"),
            )
        )
        assert windows_search_context(desc, "arm64").anchor_rel_dirs == ("bin/arm64", "bin")

    @pytest.mark.agent_authored(model="gpt-5")
    def test_windows_anchor_dirs_select_x64(self):
        desc = _make_desc(
            anchor_rel_dirs_windows=WindowsSearchDirs(
                x64=("bin/x64", "bin"),
                arm64=("bin/arm64", "bin"),
            )
        )
        assert windows_search_context(desc, "x64").anchor_rel_dirs == ("bin/x64", "bin")

    def test_nvvm_cuda_home_linux(self, mocker, tmp_path):
        """End-to-end: find_in_cuda_path resolves nvvm under its custom subdir."""
        mocker.patch(f"{_STEPS_MOD}.get_cuda_path_or_home", return_value=str(tmp_path))

        nvvm_dir = tmp_path / "nvvm" / "lib64"
        nvvm_dir.mkdir(parents=True)
        so_file = nvvm_dir / "libnvvm.so"
        so_file.touch()

        desc = _make_desc(
            name="nvvm",
            linux_sonames=("libnvvm.so",),
            anchor_rel_dirs_linux=("nvvm/lib64",),
        )
        result = find_in_cuda_path(_ctx(desc))
        assert isinstance(result, FindResult)
        assert result.abs_path == str(so_file)
        assert result.found_via == "CUDA_PATH"


# ---------------------------------------------------------------------------
# What a miss reports
# ---------------------------------------------------------------------------


class TestMissReports:
    @pytest.mark.agent_authored(model="claude-fable-5-2")
    def test_product_root_miss_reports_each_directory_searched(self, mocker, tmp_path):
        desc = LIB_DESCRIPTORS["nccl"]
        lib_dirs = []
        for rel_dir in desc.install_root_env_rel_dirs_linux[:2]:
            lib_dir = tmp_path / rel_dir
            lib_dir.mkdir(parents=True)
            (lib_dir / "unrelated.txt").touch()
            lib_dirs.append(lib_dir)
        mocker.patch.dict(os.environ, {"NCCL_HOME": str(tmp_path)})

        result = find_in_install_root_env_vars(_ctx(desc))

        searched = " or ".join(desc.linux_sonames)
        assert result == NotFound(
            (f"No such file: {searched}",) * 2,
            tuple(line for lib_dir in lib_dirs for line in (f'  listdir("{lib_dir.resolve()}"):', "    unrelated.txt")),
        )

    @pytest.mark.agent_authored(model="claude-fable-5-2")
    def test_program_files_miss_reports_each_root_searched(self, mocker, tmp_path):
        bin_dirs = []
        for version in ("v9.10", "v9.9"):  # newest first: the order the roots are searched in
            bin_dir = tmp_path / "NVIDIA" / "CUDNN" / version / "bin"
            bin_dir.mkdir(parents=True)
            (bin_dir / "unrelated.txt").touch()
            bin_dirs.append(bin_dir)
        mocker.patch.dict(os.environ, {"PROGRAMFILES": str(tmp_path), "PROGRAMW6432": ""})

        result = find_in_program_files_roots(_ctx(LIB_DESCRIPTORS["cudnn"], windows_arch="x64"))

        assert result == NotFound(
            ("No x64-compatible PE file: known cudnn DLL",) * 2,
            tuple(line for bin_dir in bin_dirs for line in (f'  listdir("{bin_dir.resolve()}"):', "    unrelated.txt")),
        )
