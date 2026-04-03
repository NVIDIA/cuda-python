# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Tests for NVIDIA driver library loading ("cuda", "nvml").

These libraries are part of the display driver, not the CUDA Toolkit.
They use a simplified system-search-only path, skipping site-packages,
conda, CUDA_HOME, and the canary probe.
"""

import os

import pytest
from child_load_nvidia_dynamic_lib_helper import (
    build_child_process_failed_for_libname_message,
    run_load_nvidia_dynamic_lib_in_subprocess,
)

from cuda.pathfinder._dynamic_libs import load_nvidia_dynamic_lib as load_nvidia_dynamic_lib_module
from cuda.pathfinder._dynamic_libs.lib_descriptor import LIB_DESCRIPTORS
from cuda.pathfinder._dynamic_libs.load_dl_common import DynamicLibNotFoundError, LoadedDL
from cuda.pathfinder._dynamic_libs.load_nvidia_dynamic_lib import (
    _DRIVER_ONLY_LIBNAMES,
    _load_driver_lib_no_cache,
    _load_lib_no_cache,
)
from cuda.pathfinder._dynamic_libs.subprocess_protocol import STATUS_NOT_FOUND, parse_dynamic_lib_subprocess_payload
from cuda.pathfinder._utils.platform_aware import IS_WINDOWS, quote_for_shell

STRICTNESS = os.environ.get("CUDA_PATHFINDER_TEST_LOAD_NVIDIA_DYNAMIC_LIB_STRICTNESS", "see_what_works")
assert STRICTNESS in ("see_what_works", "all_must_work")

_MODULE = "cuda.pathfinder._dynamic_libs.load_nvidia_dynamic_lib"
_LOADER_MODULE = "cuda.pathfinder._dynamic_libs.load_nvidia_dynamic_lib.LOADER"

_CUDA_DESC = LIB_DESCRIPTORS["cuda"]
_NVML_DESC = LIB_DESCRIPTORS["nvml"]


def _make_loaded_dl(path, found_via):
    return LoadedDL(path, False, 0xDEAD, found_via)


def _skip_if_missing_nvcudla_runtime(libname: str, *, timeout: float) -> None:
    if libname != "nvcudla":
        return
    if load_nvidia_dynamic_lib_module._loadable_via_canary_subprocess("nvcudla", timeout=timeout):
        return
    pytest.skip("libnvcudla.so is not loadable via canary subprocess on this host.")


# ---------------------------------------------------------------------------
# _load_driver_lib_no_cache
# ---------------------------------------------------------------------------


def test_driver_lib_returns_already_loaded(mocker):
    already = LoadedDL("/usr/lib/libcuda.so.1", True, 0xBEEF, "was-already-loaded-from-elsewhere")
    mocker.patch(f"{_LOADER_MODULE}.check_if_already_loaded_from_elsewhere", return_value=already)
    mocker.patch(f"{_LOADER_MODULE}.load_with_system_search")

    result = _load_driver_lib_no_cache(_CUDA_DESC)

    assert result is already
    from cuda.pathfinder._dynamic_libs.load_nvidia_dynamic_lib import LOADER

    LOADER.load_with_system_search.assert_not_called()


def test_driver_lib_falls_through_to_system_search(mocker):
    loaded = _make_loaded_dl("/usr/lib/libcuda.so.1", "system-search")
    mocker.patch(f"{_LOADER_MODULE}.check_if_already_loaded_from_elsewhere", return_value=None)
    mocker.patch(f"{_LOADER_MODULE}.load_with_system_search", return_value=loaded)

    result = _load_driver_lib_no_cache(_CUDA_DESC)

    assert result is loaded
    assert result.found_via == "system-search"


def test_driver_lib_raises_when_not_found(mocker):
    mocker.patch(f"{_LOADER_MODULE}.check_if_already_loaded_from_elsewhere", return_value=None)
    mocker.patch(f"{_LOADER_MODULE}.load_with_system_search", return_value=None)

    with pytest.raises(DynamicLibNotFoundError, match="NVIDIA driver library"):
        _load_driver_lib_no_cache(_NVML_DESC)


def test_driver_lib_does_not_search_site_packages(mocker):
    """Driver libs must not go through the CTK search cascade."""
    loaded = _make_loaded_dl("/usr/lib/libcuda.so.1", "system-search")
    mocker.patch(f"{_LOADER_MODULE}.check_if_already_loaded_from_elsewhere", return_value=None)
    mocker.patch(f"{_LOADER_MODULE}.load_with_system_search", return_value=loaded)

    spy = mocker.patch(f"{_MODULE}.run_find_steps")
    _load_driver_lib_no_cache(_CUDA_DESC)
    spy.assert_not_called()


# ---------------------------------------------------------------------------
# _load_lib_no_cache dispatches driver libs correctly
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("libname", sorted(_DRIVER_ONLY_LIBNAMES))
def test_load_lib_no_cache_dispatches_to_driver_path(libname, mocker):
    loaded = _make_loaded_dl(f"/usr/lib/fake_{libname}.so", "system-search")
    mock_driver = mocker.patch(f"{_MODULE}._load_driver_lib_no_cache", return_value=loaded)

    result = _load_lib_no_cache(libname)

    assert result is loaded
    mock_driver.assert_called_once_with(LIB_DESCRIPTORS[libname])


def test_load_lib_no_cache_does_not_dispatch_ctk_lib_to_driver_path(mocker):
    """Ensure regular CTK libs don't take the driver shortcut."""
    mock_driver = mocker.patch(f"{_MODULE}._load_driver_lib_no_cache")
    mocker.patch(f"{_MODULE}.run_find_steps", return_value=None)
    mocker.patch(f"{_LOADER_MODULE}.check_if_already_loaded_from_elsewhere", return_value=None)
    mocker.patch(f"{_MODULE}.load_dependencies")
    mocker.patch(
        f"{_LOADER_MODULE}.load_with_system_search",
        return_value=_make_loaded_dl("/usr/lib/libcudart.so.13", "system-search"),
    )

    _load_lib_no_cache("cudart")

    mock_driver.assert_not_called()


# ---------------------------------------------------------------------------
# Real loading tests (dedicated subprocess for isolation)
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("libname", sorted(_DRIVER_ONLY_LIBNAMES))
def test_real_load_driver_lib(info_summary_append, libname):
    """Load a real driver library in a dedicated subprocess.

    This complements the mock tests above: it exercises the actual OS
    loader path and logs results via INFO for CI/QA inspection.
    """
    timeout = 120 if IS_WINDOWS else 30
    result = run_load_nvidia_dynamic_lib_in_subprocess(libname, timeout=timeout)

    def raise_child_process_failed():
        raise RuntimeError(build_child_process_failed_for_libname_message(libname, result))

    if result.returncode != 0:
        raise_child_process_failed()
    assert not result.stderr
    payload = parse_dynamic_lib_subprocess_payload(
        result.stdout,
        libname=libname,
        error_label="Load subprocess child process",
    )
    if payload.status == STATUS_NOT_FOUND:
        _skip_if_missing_nvcudla_runtime(libname, timeout=timeout)
        if STRICTNESS == "all_must_work":
            raise_child_process_failed()
        info_summary_append(f"Not found: {libname=!r}")
    else:
        abs_path = payload.abs_path
        assert abs_path is not None
        info_summary_append(f"abs_path={quote_for_shell(abs_path)}")
        assert os.path.isfile(abs_path)
