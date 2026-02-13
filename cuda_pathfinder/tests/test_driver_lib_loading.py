# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Tests for NVIDIA driver library loading ("cuda", "nvml").

These libraries are part of the display driver, not the CUDA Toolkit.
They use a simplified system-search-only path, skipping site-packages,
conda, CUDA_HOME, and the canary probe.
"""

import json
import os

import pytest
import spawned_process_runner
from child_load_nvidia_dynamic_lib_helper import build_child_process_failed_for_libname_message, child_process_func

from cuda.pathfinder._dynamic_libs.load_dl_common import DynamicLibNotFoundError, LoadedDL
from cuda.pathfinder._dynamic_libs.load_nvidia_dynamic_lib import (
    _DRIVER_ONLY_LIBNAMES,
    _load_driver_lib_no_cache,
    _load_lib_no_cache,
)
from cuda.pathfinder._utils.platform_aware import IS_WINDOWS, quote_for_shell

STRICTNESS = os.environ.get("CUDA_PATHFINDER_TEST_LOAD_NVIDIA_DYNAMIC_LIB_STRICTNESS", "see_what_works")
assert STRICTNESS in ("see_what_works", "all_must_work")

_MODULE = "cuda.pathfinder._dynamic_libs.load_nvidia_dynamic_lib"


def _make_loaded_dl(path, found_via):
    return LoadedDL(path, False, 0xDEAD, found_via)


# ---------------------------------------------------------------------------
# _load_driver_lib_no_cache
# ---------------------------------------------------------------------------


def test_driver_lib_returns_already_loaded(mocker):
    already = LoadedDL("/usr/lib/libcuda.so.1", True, 0xBEEF, "was-already-loaded-from-elsewhere")
    mocker.patch(f"{_MODULE}.check_if_already_loaded_from_elsewhere", return_value=already)
    mocker.patch(f"{_MODULE}.load_with_system_search")

    result = _load_driver_lib_no_cache("cuda")

    assert result is already
    # system search should not have been called
    from cuda.pathfinder._dynamic_libs import load_nvidia_dynamic_lib as mod

    mod.load_with_system_search.assert_not_called()


def test_driver_lib_falls_through_to_system_search(mocker):
    loaded = _make_loaded_dl("/usr/lib/libcuda.so.1", "system-search")
    mocker.patch(f"{_MODULE}.check_if_already_loaded_from_elsewhere", return_value=None)
    mocker.patch(f"{_MODULE}.load_with_system_search", return_value=loaded)

    result = _load_driver_lib_no_cache("cuda")

    assert result is loaded
    assert result.found_via == "system-search"


def test_driver_lib_raises_when_not_found(mocker):
    mocker.patch(f"{_MODULE}.check_if_already_loaded_from_elsewhere", return_value=None)
    mocker.patch(f"{_MODULE}.load_with_system_search", return_value=None)

    with pytest.raises(DynamicLibNotFoundError, match="NVIDIA driver library"):
        _load_driver_lib_no_cache("nvml")


def test_driver_lib_does_not_search_site_packages(mocker):
    """Driver libs must not go through the CTK search cascade."""
    loaded = _make_loaded_dl("/usr/lib/libcuda.so.1", "system-search")
    mocker.patch(f"{_MODULE}.check_if_already_loaded_from_elsewhere", return_value=None)
    mocker.patch(f"{_MODULE}.load_with_system_search", return_value=loaded)

    from cuda.pathfinder._dynamic_libs.find_nvidia_dynamic_lib import _FindNvidiaDynamicLib

    spy = mocker.spy(_FindNvidiaDynamicLib, "try_site_packages")
    _load_driver_lib_no_cache("cuda")
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
    mock_driver.assert_called_once_with(libname)


def test_load_lib_no_cache_does_not_dispatch_ctk_lib_to_driver_path(mocker):
    """Ensure regular CTK libs don't take the driver shortcut."""
    mock_driver = mocker.patch(f"{_MODULE}._load_driver_lib_no_cache")
    # Let the normal path run far enough to prove the driver path wasn't used.
    # We'll make it fail quickly at check_if_already_loaded_from_elsewhere.
    from cuda.pathfinder._dynamic_libs.find_nvidia_dynamic_lib import _FindNvidiaDynamicLib

    mocker.patch.object(_FindNvidiaDynamicLib, "try_site_packages", return_value=None)
    mocker.patch.object(_FindNvidiaDynamicLib, "try_with_conda_prefix", return_value=None)
    mocker.patch(f"{_MODULE}.check_if_already_loaded_from_elsewhere", return_value=None)
    mocker.patch(f"{_MODULE}.load_dependencies")
    mocker.patch(
        f"{_MODULE}.load_with_system_search",
        return_value=_make_loaded_dl("/usr/lib/libcudart.so.13", "system-search"),
    )

    _load_lib_no_cache("cudart")

    mock_driver.assert_not_called()


# ---------------------------------------------------------------------------
# Real loading tests (spawned child process for isolation)
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("libname", sorted(_DRIVER_ONLY_LIBNAMES))
def test_real_load_driver_lib(info_summary_append, libname):
    """Load a real driver library in a child process.

    This complements the mock tests above: it exercises the actual OS
    loader path and logs results via INFO for CI/QA inspection.
    """
    timeout = 120 if IS_WINDOWS else 30
    result = spawned_process_runner.run_in_spawned_child_process(child_process_func, args=(libname,), timeout=timeout)

    def raise_child_process_failed():
        raise RuntimeError(build_child_process_failed_for_libname_message(libname, result))

    if result.returncode != 0:
        raise_child_process_failed()
    assert not result.stderr
    if result.stdout.startswith("CHILD_LOAD_NVIDIA_DYNAMIC_LIB_HELPER_DYNAMIC_LIB_NOT_FOUND_ERROR:"):
        if STRICTNESS == "all_must_work":
            raise_child_process_failed()
        info_summary_append(f"Not found: {libname=!r}")
    else:
        abs_path = json.loads(result.stdout.rstrip())
        info_summary_append(f"abs_path={quote_for_shell(abs_path)}")
        assert os.path.isfile(abs_path)
