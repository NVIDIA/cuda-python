# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import os
import platform
from pathlib import Path
from types import SimpleNamespace

import pytest
from child_load_nvidia_dynamic_lib_helper import (
    build_child_process_failed_for_libname_message,
    run_load_nvidia_dynamic_lib_in_subprocess,
)
from conftest import skip_if_missing_libnvcudla_so
from local_helpers import have_distribution, quote_for_shell

from cuda.pathfinder import DynamicLibNotAvailableError, DynamicLibUnknownError, load_nvidia_dynamic_lib
from cuda.pathfinder._dynamic_libs import load_nvidia_dynamic_lib as load_nvidia_dynamic_lib_module
from cuda.pathfinder._dynamic_libs import supported_nvidia_libs
from cuda.pathfinder._dynamic_libs.subprocess_protocol import (
    STATUS_NOT_FOUND,
    parse_dynamic_lib_subprocess_payload,
)
from cuda.pathfinder._utils.platform_aware import IS_WINDOWS

STRICTNESS = os.environ.get("CUDA_PATHFINDER_TEST_LOAD_NVIDIA_DYNAMIC_LIB_STRICTNESS", "see_what_works")
assert STRICTNESS in ("see_what_works", "all_must_work")


@pytest.mark.agent_authored(model="gpt-5")
def test_loader_uses_all_available_libnames():
    assert supported_nvidia_libs.ALL_AVAILABLE_LIBNAMES == load_nvidia_dynamic_lib_module.ALL_AVAILABLE_LIBNAMES


_CTK_DESCRIPTORS = tuple(
    desc for desc in load_nvidia_dynamic_lib_module.LIB_DESCRIPTORS.values() if desc.packaged_with == "ctk"
)
# Every library that has file names for the platform the tests run on, in catalog order.
_LIBNAMES_WITH_FILENAMES_HERE = tuple(
    libname
    for libname, desc in load_nvidia_dynamic_lib_module.LIB_DESCRIPTORS.items()
    if (desc.windows_dlls if IS_WINDOWS else desc.linux_sonames)
)


@pytest.mark.agent_authored(model="claude-fable-5-2")
def test_supported_libnames_linux_site_packages_libdirs_ctk_consistency():
    assert tuple(sorted(supported_nvidia_libs.SUPPORTED_LIBNAMES_LINUX)) == tuple(
        sorted(desc.name for desc in _CTK_DESCRIPTORS if desc.site_packages_linux)
    )


@pytest.mark.parametrize(
    ("target_arch", "supported_libnames"),
    [
        pytest.param("x64", supported_nvidia_libs.SUPPORTED_LIBNAMES_WINDOWS_X64, id="x64"),
        pytest.param("arm64", supported_nvidia_libs.SUPPORTED_LIBNAMES_WINDOWS_ARM64, id="arm64"),
    ],
)
@pytest.mark.agent_authored(model="claude-fable-5-2")
def test_supported_libnames_windows_site_packages_libdirs_ctk_consistency(target_arch, supported_libnames):
    # Not every Windows CTK library ships in a pip wheel (e.g. cudla is loaded
    # from the local CUDA Toolkit only), so a library may legitimately omit
    # site_packages_windows. Only assert that every site-packages entry maps to
    # a supported Windows libname, not the other way around.
    site_packages_libnames = {
        desc.name for desc in _CTK_DESCRIPTORS if desc.site_packages_windows.for_arch(target_arch)
    }
    assert site_packages_libnames <= set(supported_libnames)


@pytest.mark.parametrize("filenames_field", ["linux_sonames", "windows_dlls"])
@pytest.mark.agent_authored(model="claude-fable-5-2")
def test_library_filenames_belong_to_one_libname(filenames_field):
    libname_for_value = {}
    for libname, desc in load_nvidia_dynamic_lib_module.LIB_DESCRIPTORS.items():
        for value in getattr(desc, filenames_field):
            prev_libname = libname_for_value.get(value)
            if prev_libname is not None:
                raise RuntimeError(f"Multiple libnames for {value!r}: {prev_libname}, {libname}")
            libname_for_value[value] = libname


def test_runtime_error_on_non_64bit_python(mocker):
    # Ensure this test is not affected by any prior cached calls.
    load_nvidia_dynamic_lib.cache_clear()
    mocker.patch("struct.calcsize", return_value=3)  # fake 24-bit pointer
    with pytest.raises(RuntimeError, match=r"requires 64-bit Python\. Currently running: 24-bit Python"):
        load_nvidia_dynamic_lib("cudart")


def test_unknown_libname_raises_dynamic_lib_unknown_error():
    with pytest.raises(DynamicLibUnknownError, match=r"Unknown library name: 'not_a_real_lib'.*cudart"):
        load_nvidia_dynamic_lib("not_a_real_lib")


def test_known_but_platform_unavailable_libname_raises_dynamic_lib_not_available_error(monkeypatch):
    load_nvidia_dynamic_lib.cache_clear()
    monkeypatch.setattr(load_nvidia_dynamic_lib_module, "_ALL_KNOWN_LIBNAMES", frozenset(("known_but_unavailable",)))
    monkeypatch.setattr(load_nvidia_dynamic_lib_module, "ALL_AVAILABLE_LIBNAMES", frozenset())
    monkeypatch.setattr(load_nvidia_dynamic_lib_module, "_PLATFORM_NAME", "TestOS")
    with pytest.raises(
        DynamicLibNotAvailableError,
        match=r"known_but_unavailable.*not available on TestOS",
    ):
        load_nvidia_dynamic_lib("known_but_unavailable")


IMPORTLIB_METADATA_DISTRIBUTIONS_NAMES = {
    "mathdx": r"^nvidia-libmathdx-.*$",
}


def _is_expected_load_nvidia_dynamic_lib_failure(libname):
    if libname == "nvpl_fftw" and platform.machine().lower() != "aarch64":
        return True
    if libname == "cutensorMg":
        # cuTENSOR 2.8 removed cuTENSORMg in favor of cuTENSORMp.
        return have_distribution(r"^cutensor-cu(?:12|13)$", minimum_version="2.8")
    dist_name_pattern = IMPORTLIB_METADATA_DISTRIBUTIONS_NAMES.get(libname)
    if dist_name_pattern is not None:
        return not have_distribution(dist_name_pattern)
    return False


@pytest.mark.parametrize(
    ("installed_distributions", "expected"),
    [
        ([], False),
        ([SimpleNamespace(metadata={"Name": "cutensor-cu13"}, version="2.7.0")], False),
        ([SimpleNamespace(metadata={"Name": "cutensor-cu12"}, version="2.8.0")], True),
        ([SimpleNamespace(metadata={"Name": "cutensor-cu13"}, version="2.9.0")], True),
        ([SimpleNamespace(metadata={"Name": "unrelated-package"}, version="2.8.0")], False),
    ],
)
@pytest.mark.agent_authored(model="gpt-5.6-sol")
def test_cutensor_mg_expected_failure_follows_installed_cutensor_version(
    mocker,
    installed_distributions,
    expected,
):
    mocker.patch("local_helpers.importlib.metadata.distributions", return_value=installed_distributions)
    have_distribution.cache_clear()
    try:
        assert _is_expected_load_nvidia_dynamic_lib_failure("cutensorMg") is expected
    finally:
        have_distribution.cache_clear()


# For libname=nvcudla:
# This test is not currently run in any CI job
@pytest.mark.parametrize("libname", _LIBNAMES_WITH_FILENAMES_HERE)
def test_load_nvidia_dynamic_lib(info_summary_append, libname):
    # Use a fresh Python subprocess for each load to isolate global dynamic
    # loader state and keep the tests aligned with the canary probe model.
    timeout = 120 if IS_WINDOWS else 30
    result = run_load_nvidia_dynamic_lib_in_subprocess(libname, timeout=timeout)

    def raise_child_process_failed():
        raise RuntimeError(build_child_process_failed_for_libname_message(libname, result))

    if result.returncode != 0:
        skip_if_missing_libnvcudla_so(libname, timeout=timeout)
        raise_child_process_failed()
    assert not result.stderr
    payload = parse_dynamic_lib_subprocess_payload(
        result.stdout,
        libname=libname,
        error_label="Load subprocess child process",
    )
    if payload.status == STATUS_NOT_FOUND:
        skip_if_missing_libnvcudla_so(libname, timeout=timeout)
        if STRICTNESS == "all_must_work" and not _is_expected_load_nvidia_dynamic_lib_failure(libname):
            raise_child_process_failed()
        info_summary_append(f"Not found: {libname=!r}")
    else:
        abs_path = payload.abs_path
        assert abs_path is not None
        info_summary_append(f"abs_path={quote_for_shell(abs_path)}")
        assert Path(abs_path).is_file()  # double-check the abs_path
