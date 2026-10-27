# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import os

import pytest
from child_load_nvidia_dynamic_lib_helper import (
    build_child_process_failed_for_libname_message,
    run_load_nvidia_dynamic_lib_in_subprocess,
)

from cuda.pathfinder import (
    DynamicLibNotAvailableError,
    DynamicLibNotFoundError,
    DynamicLibUnknownError,
    LocatedDynamicLib,
    find_nvidia_dynamic_lib,
    locate_nvidia_dynamic_lib,
)
from cuda.pathfinder._dynamic_libs import find_nvidia_dynamic_lib as find_mod
from cuda.pathfinder._dynamic_libs import load_nvidia_dynamic_lib as load_mod
from cuda.pathfinder._dynamic_libs import supported_nvidia_libs
from cuda.pathfinder._dynamic_libs.load_nvidia_dynamic_lib import _resolve_system_loaded_abs_path_in_subprocess
from cuda.pathfinder._dynamic_libs.search_steps import EARLY_FIND_STEPS, LATE_FIND_STEPS, FindResult
from cuda.pathfinder._dynamic_libs.subprocess_protocol import STATUS_NOT_FOUND, parse_dynamic_lib_subprocess_payload
from cuda.pathfinder._utils.platform_aware import IS_WINDOWS, quote_for_shell

_MODULE = "cuda.pathfinder._dynamic_libs.find_nvidia_dynamic_lib"


@pytest.fixture(autouse=True)
def _clear_caches():
    find_nvidia_dynamic_lib.cache_clear()
    _resolve_system_loaded_abs_path_in_subprocess.cache_clear()
    yield
    find_nvidia_dynamic_lib.cache_clear()
    _resolve_system_loaded_abs_path_in_subprocess.cache_clear()


@pytest.fixture
def no_in_process_loading(mocker):
    for name in ("check_if_already_loaded_from_elsewhere", "load_with_system_search", "load_with_abs_path"):
        mocker.patch.object(load_mod.LOADER, name, side_effect=AssertionError(f"LOADER.{name} was called"))


def _patch_find_steps(mocker, *, early=None, late=None):
    def run_find_steps(_ctx, steps):
        if steps is EARLY_FIND_STEPS:
            return early
        assert steps is LATE_FIND_STEPS
        return late

    return mocker.patch(f"{_MODULE}.run_find_steps", side_effect=run_find_steps)


@pytest.mark.parametrize(
    "libname",
    supported_nvidia_libs.SUPPORTED_WINDOWS_DLLS if IS_WINDOWS else supported_nvidia_libs.SUPPORTED_LINUX_SONAMES,
)
@pytest.mark.agent_authored(model="claude-opus-5-5")
def test_locate_matches_load_in_fresh_process(info_summary_append, libname):
    timeout = 120 if IS_WINDOWS else 30
    result = run_load_nvidia_dynamic_lib_in_subprocess(libname, timeout=timeout)
    if result.returncode != 0:
        raise RuntimeError(build_child_process_failed_for_libname_message(libname, result))
    payload = parse_dynamic_lib_subprocess_payload(
        result.stdout,
        libname=libname,
        error_label="Load subprocess child process",
    )

    if payload.status == STATUS_NOT_FOUND:
        with pytest.raises(DynamicLibNotFoundError):
            locate_nvidia_dynamic_lib(libname)
        info_summary_append(f"Not found: {libname=!r}")
        return

    located = locate_nvidia_dynamic_lib(libname)
    info_summary_append(f"abs_path={quote_for_shell(located.abs_path)} found_via={located.found_via}")
    assert located.name == libname
    assert located.filename == os.path.basename(located.abs_path)
    assert os.path.samefile(located.abs_path, payload.abs_path)
    assert find_nvidia_dynamic_lib(libname) == located.abs_path


@pytest.mark.usefixtures("no_in_process_loading")
@pytest.mark.agent_authored(model="claude-opus-5-5")
def test_early_find_skips_system_search(mocker, tmp_path):
    lib = tmp_path / "libcudart.so.13"
    _patch_find_steps(mocker, early=FindResult(str(lib), "site-packages"))
    probe = mocker.patch(f"{_MODULE}._resolve_system_loaded_abs_path_in_subprocess")

    assert locate_nvidia_dynamic_lib("cudart") == LocatedDynamicLib(
        name="cudart",
        abs_path=str(lib),
        filename="libcudart.so.13",
        found_via="site-packages",
    )
    probe.assert_not_called()


@pytest.mark.usefixtures("no_in_process_loading")
@pytest.mark.agent_authored(model="claude-opus-5-5")
def test_system_search_runs_before_late_find_steps(mocker, tmp_path):
    lib = tmp_path / "libcudart.so.13"
    run_find_steps = _patch_find_steps(mocker, late=FindResult("unused", "CUDA_PATH"))
    probe = mocker.patch(f"{_MODULE}._resolve_system_loaded_abs_path_in_subprocess", return_value=str(lib))

    located = locate_nvidia_dynamic_lib("cudart")

    assert located.abs_path == str(lib)
    assert located.found_via == "system-search"
    probe.assert_called_once_with("cudart")
    assert [call.args[1] for call in run_find_steps.call_args_list] == [EARLY_FIND_STEPS]


@pytest.mark.usefixtures("no_in_process_loading")
@pytest.mark.agent_authored(model="claude-opus-5-5")
def test_late_find_steps_after_system_search(mocker, tmp_path):
    lib = tmp_path / "libcudart.so.13"
    _patch_find_steps(mocker, late=FindResult(str(lib), "CUDA_PATH"))
    mocker.patch(f"{_MODULE}._resolve_system_loaded_abs_path_in_subprocess", return_value=None)

    located = locate_nvidia_dynamic_lib("cudart")

    assert located.abs_path == str(lib)
    assert located.found_via == "CUDA_PATH"


@pytest.mark.usefixtures("no_in_process_loading")
@pytest.mark.agent_authored(model="claude-opus-5-5")
def test_ctk_root_canary_is_last(mocker, tmp_path):
    lib = tmp_path / "libnvvm.so.4"
    _patch_find_steps(mocker)
    mocker.patch(f"{_MODULE}._resolve_system_loaded_abs_path_in_subprocess", return_value=None)
    canary = mocker.patch(f"{_MODULE}._try_ctk_root_canary", return_value=str(lib))

    located = locate_nvidia_dynamic_lib("nvvm")

    assert located.abs_path == str(lib)
    assert located.found_via == "system-ctk-root"
    canary.assert_called_once()


@pytest.mark.usefixtures("no_in_process_loading")
@pytest.mark.agent_authored(model="claude-opus-5-5")
def test_not_found_raises(mocker):
    _patch_find_steps(mocker)
    mocker.patch(f"{_MODULE}._resolve_system_loaded_abs_path_in_subprocess", return_value=None)
    mocker.patch(f"{_MODULE}._try_ctk_root_canary", return_value=None)

    with pytest.raises(DynamicLibNotFoundError, match="Failure finding"):
        locate_nvidia_dynamic_lib("nvvm")
    with pytest.raises(DynamicLibNotFoundError, match="Failure finding"):
        find_nvidia_dynamic_lib("nvvm")


@pytest.mark.usefixtures("no_in_process_loading")
@pytest.mark.agent_authored(model="claude-opus-5-5")
def test_driver_lib_uses_system_search_only(mocker, tmp_path):
    lib = tmp_path / "libcuda.so.1"
    run_find_steps = mocker.patch(f"{_MODULE}.run_find_steps")
    mocker.patch(f"{_MODULE}._resolve_system_loaded_abs_path_in_subprocess", return_value=str(lib))

    located = locate_nvidia_dynamic_lib("cuda")

    assert located.abs_path == str(lib)
    assert located.found_via == "system-search"
    run_find_steps.assert_not_called()


@pytest.mark.usefixtures("no_in_process_loading")
@pytest.mark.agent_authored(model="claude-opus-5-5")
def test_driver_lib_not_found_raises(mocker):
    mocker.patch(f"{_MODULE}._resolve_system_loaded_abs_path_in_subprocess", return_value=None)

    with pytest.raises(DynamicLibNotFoundError, match="NVIDIA driver library"):
        locate_nvidia_dynamic_lib("cuda")


@pytest.mark.agent_authored(model="claude-opus-5-5")
def test_find_is_cached(mocker, tmp_path):
    lib = tmp_path / "libcudart.so.13"
    locate = mocker.patch(
        f"{_MODULE}.locate_nvidia_dynamic_lib",
        return_value=LocatedDynamicLib("cudart", str(lib), lib.name, "site-packages"),
    )

    assert find_mod.find_nvidia_dynamic_lib("cudart") == str(lib)
    assert find_mod.find_nvidia_dynamic_lib("cudart") == str(lib)
    locate.assert_called_once_with("cudart")


@pytest.mark.agent_authored(model="claude-opus-5-5")
def test_unknown_libname_raises_dynamic_lib_unknown_error():
    with pytest.raises(DynamicLibUnknownError, match=r"Unknown library name: 'not_a_real_lib'"):
        locate_nvidia_dynamic_lib("not_a_real_lib")


@pytest.mark.agent_authored(model="claude-opus-5-5")
def test_known_but_platform_unavailable_libname_raises_dynamic_lib_not_available_error(monkeypatch):
    monkeypatch.setattr(load_mod, "_ALL_KNOWN_LIBNAMES", frozenset(("known_but_unavailable",)))
    monkeypatch.setattr(load_mod, "ALL_AVAILABLE_LIBNAMES", frozenset())
    monkeypatch.setattr(load_mod, "_PLATFORM_NAME", "TestOS")
    with pytest.raises(DynamicLibNotAvailableError, match=r"known_but_unavailable.*not available on TestOS"):
        locate_nvidia_dynamic_lib("known_but_unavailable")
