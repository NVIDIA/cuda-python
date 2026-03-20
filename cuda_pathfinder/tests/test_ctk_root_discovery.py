# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0


import os
import subprocess
import sys
import textwrap

import pytest

from cuda.pathfinder._dynamic_libs import load_nvidia_dynamic_lib as load_mod
from cuda.pathfinder._dynamic_libs import search_steps as steps_mod
from cuda.pathfinder._dynamic_libs.lib_descriptor import LIB_DESCRIPTORS
from cuda.pathfinder._dynamic_libs.load_dl_common import DynamicLibNotFoundError, LoadedDL
from cuda.pathfinder._dynamic_libs.load_nvidia_dynamic_lib import (
    _load_lib_no_cache,
    _resolve_system_loaded_abs_path_in_subprocess,
    _try_ctk_root_canary,
)
from cuda.pathfinder._dynamic_libs.search_steps import (
    SearchContext,
    _derive_ctk_root_linux,
    _derive_ctk_root_windows,
    derive_ctk_root,
    find_via_ctk_root,
)
from cuda.pathfinder._dynamic_libs.subprocess_protocol import (
    DYNAMIC_LIB_SUBPROCESS_CWD,
    DYNAMIC_LIB_SUBPROCESS_MODULE,
    MODE_CANARY,
)
from cuda.pathfinder._utils.platform_aware import IS_WINDOWS

_MODULE = "cuda.pathfinder._dynamic_libs.load_nvidia_dynamic_lib"
_STEPS_MODULE = "cuda.pathfinder._dynamic_libs.search_steps"
_PACKAGE_ROOT = DYNAMIC_LIB_SUBPROCESS_CWD


def _ctx(libname: str = "nvvm") -> SearchContext:
    return SearchContext(LIB_DESCRIPTORS[libname])


@pytest.fixture(autouse=True)
def _clear_canary_subprocess_probe_cache():
    _resolve_system_loaded_abs_path_in_subprocess.cache_clear()
    yield
    _resolve_system_loaded_abs_path_in_subprocess.cache_clear()


# ---------------------------------------------------------------------------
# Platform-aware test helpers
# ---------------------------------------------------------------------------


def _create_nvvm_in_ctk(ctk_root):
    """Create a fake nvvm lib in the platform-appropriate CTK subdirectory."""
    if IS_WINDOWS:
        nvvm_dir = ctk_root / "nvvm" / "bin"
        nvvm_dir.mkdir(parents=True)
        nvvm_lib = nvvm_dir / "nvvm64.dll"
    else:
        nvvm_dir = ctk_root / "nvvm" / "lib64"
        nvvm_dir.mkdir(parents=True)
        nvvm_lib = nvvm_dir / "libnvvm.so"
    nvvm_lib.write_bytes(b"fake")
    return nvvm_lib


def _create_cudart_in_ctk(ctk_root):
    """Create a fake cudart lib in the platform-appropriate CTK subdirectory."""
    if IS_WINDOWS:
        lib_dir = ctk_root / "bin"
        lib_dir.mkdir(parents=True)
        lib_file = lib_dir / "cudart64_12.dll"
    else:
        lib_dir = ctk_root / "lib64"
        lib_dir.mkdir(parents=True)
        lib_file = lib_dir / "libcudart.so"
    lib_file.write_bytes(b"fake")
    return lib_file


def _fake_canary_path(ctk_root):
    """Return the path a system-loaded canary lib would resolve to."""
    if IS_WINDOWS:
        return str(ctk_root / "bin" / "cudart64_13.dll")
    return str(ctk_root / "lib64" / "libcudart.so.13")


# ---------------------------------------------------------------------------
# derive_ctk_root
# ---------------------------------------------------------------------------


def test_derive_ctk_root_linux_lib64():
    assert _derive_ctk_root_linux("/usr/local/cuda-13/lib64/libcudart.so.13") == "/usr/local/cuda-13"


def test_derive_ctk_root_linux_lib():
    assert _derive_ctk_root_linux("/opt/cuda/lib/libcudart.so.12") == "/opt/cuda"


def test_derive_ctk_root_linux_unrecognized():
    assert _derive_ctk_root_linux("/some/weird/path/libcudart.so.13") is None


def test_derive_ctk_root_linux_root_level():
    assert _derive_ctk_root_linux("/lib64/libcudart.so.13") == "/"


def test_derive_ctk_root_linux_targets_lib64():
    assert (
        _derive_ctk_root_linux("/usr/local/cuda-13.1/targets/x86_64-linux/lib64/libcudart.so.13")
        == "/usr/local/cuda-13.1"
    )


def test_derive_ctk_root_linux_targets_lib():
    assert _derive_ctk_root_linux("/opt/cuda/targets/sbsa-linux/lib/libcudart.so.12") == "/opt/cuda"


def test_derive_ctk_root_windows_ctk13():
    path = r"C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v13.0\bin\x64\cudart64_13.dll"
    assert _derive_ctk_root_windows(path) == r"C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v13.0"


def test_derive_ctk_root_windows_ctk12():
    path = r"C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v12.8\bin\cudart64_12.dll"
    assert _derive_ctk_root_windows(path) == r"C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v12.8"


def test_derive_ctk_root_windows_unrecognized():
    assert _derive_ctk_root_windows(r"C:\weird\cudart64_13.dll") is None


def test_derive_ctk_root_windows_case_insensitive_bin():
    assert _derive_ctk_root_windows(r"C:\CUDA\Bin\cudart64_12.dll") == r"C:\CUDA"


def test_derive_ctk_root_windows_case_insensitive_x64():
    assert _derive_ctk_root_windows(r"C:\CUDA\BIN\X64\cudart64_13.dll") == r"C:\CUDA"


def test_derive_ctk_root_dispatches_to_linux(mocker):
    linux_derive = mocker.patch(f"{_STEPS_MODULE}._derive_ctk_root_linux", return_value="/usr/local/cuda")
    windows_derive = mocker.patch(f"{_STEPS_MODULE}._derive_ctk_root_windows")
    assert derive_ctk_root("/usr/local/cuda/lib64/libcudart.so.13") == "/usr/local/cuda"
    linux_derive.assert_called_once_with("/usr/local/cuda/lib64/libcudart.so.13")
    windows_derive.assert_not_called()


def test_derive_ctk_root_dispatches_to_windows(mocker):
    mocker.patch(f"{_STEPS_MODULE}._derive_ctk_root_linux", return_value=None)
    windows_derive = mocker.patch(f"{_STEPS_MODULE}._derive_ctk_root_windows", return_value=r"C:\CUDA\v13")
    assert derive_ctk_root(r"C:\CUDA\v13\bin\cudart64_13.dll") == r"C:\CUDA\v13"
    windows_derive.assert_called_once_with(r"C:\CUDA\v13\bin\cudart64_13.dll")


# ---------------------------------------------------------------------------
# find_via_ctk_root
# ---------------------------------------------------------------------------


def test_try_via_ctk_root_finds_nvvm(tmp_path):
    ctk_root = tmp_path / "cuda-13"
    nvvm_lib = _create_nvvm_in_ctk(ctk_root)

    result = find_via_ctk_root(_ctx("nvvm"), str(ctk_root))
    assert result is not None
    assert result.abs_path == str(nvvm_lib)
    assert result.found_via == "system-ctk-root"


def test_try_via_ctk_root_returns_none_when_dir_missing(tmp_path):
    ctk_root = tmp_path / "cuda-13"
    ctk_root.mkdir()

    assert find_via_ctk_root(_ctx("nvvm"), str(ctk_root)) is None


def test_try_via_ctk_root_regular_lib(tmp_path):
    ctk_root = tmp_path / "cuda-13"
    cudart_lib = _create_cudart_in_ctk(ctk_root)

    result = find_via_ctk_root(_ctx("cudart"), str(ctk_root))
    assert result is not None
    assert result.abs_path == str(cudart_lib)
    assert result.found_via == "system-ctk-root"


# ---------------------------------------------------------------------------
# _resolve_system_loaded_abs_path_in_subprocess
# ---------------------------------------------------------------------------


def test_subprocess_probe_returns_abs_path_on_string_payload(mocker):
    result = subprocess.CompletedProcess(
        args=[],
        returncode=0,
        stdout='{"status": "ok", "abs_path": "/usr/local/cuda/lib64/libcudart.so.13"}\n',
        stderr="",
    )
    run_mock = mocker.patch(f"{_MODULE}.subprocess.run", return_value=result)

    assert _resolve_system_loaded_abs_path_in_subprocess("cudart") == "/usr/local/cuda/lib64/libcudart.so.13"
    run_mock.assert_called_once_with(
        [sys.executable, "-m", DYNAMIC_LIB_SUBPROCESS_MODULE, MODE_CANARY, "cudart"],
        capture_output=True,
        text=True,
        timeout=10.0,
        check=False,
        cwd=_PACKAGE_ROOT,
    )


def test_subprocess_probe_returns_none_on_null_payload(mocker):
    result = subprocess.CompletedProcess(
        args=[],
        returncode=0,
        stdout='{"status": "not-found", "abs_path": null}\n',
        stderr="",
    )
    mocker.patch(f"{_MODULE}.subprocess.run", return_value=result)

    assert _resolve_system_loaded_abs_path_in_subprocess("cudart") is None


def test_subprocess_probe_raises_on_child_failure(mocker):
    result = subprocess.CompletedProcess(args=[], returncode=1, stdout="", stderr="child failed\n")
    mocker.patch(f"{_MODULE}.subprocess.run", return_value=result)

    with pytest.raises(ChildProcessError, match="child failed"):
        _resolve_system_loaded_abs_path_in_subprocess("cudart")


def test_subprocess_probe_raises_on_timeout(mocker):
    mocker.patch(
        f"{_MODULE}.subprocess.run",
        side_effect=subprocess.TimeoutExpired(cmd=["python"], timeout=10.0, stderr="probe hung\n"),
    )
    with pytest.raises(ChildProcessError, match="timed out after 10.0 seconds"):
        _resolve_system_loaded_abs_path_in_subprocess("cudart")


def test_subprocess_probe_raises_on_empty_stdout(mocker):
    result = subprocess.CompletedProcess(args=[], returncode=0, stdout=" \n \n", stderr="")
    mocker.patch(f"{_MODULE}.subprocess.run", return_value=result)

    with pytest.raises(RuntimeError, match="produced no stdout payload"):
        _resolve_system_loaded_abs_path_in_subprocess("cudart")


def test_subprocess_probe_raises_on_invalid_json_payload(mocker):
    result = subprocess.CompletedProcess(args=[], returncode=0, stdout="not-json\n", stderr="")
    mocker.patch(f"{_MODULE}.subprocess.run", return_value=result)

    with pytest.raises(RuntimeError, match="invalid JSON payload"):
        _resolve_system_loaded_abs_path_in_subprocess("cudart")


def test_subprocess_probe_raises_on_unexpected_json_payload(mocker):
    result = subprocess.CompletedProcess(
        args=[],
        returncode=0,
        stdout='{"path": "/usr/local/cuda/lib64/libcudart.so.13"}\n',
        stderr="",
    )
    mocker.patch(f"{_MODULE}.subprocess.run", return_value=result)

    with pytest.raises(RuntimeError, match="unexpected payload"):
        _resolve_system_loaded_abs_path_in_subprocess("cudart")


def test_subprocess_probe_does_not_reenter_calling_script(tmp_path):
    script_path = tmp_path / "call_probe.py"
    run_count_path = tmp_path / "run_count.txt"
    script_path.write_text(
        textwrap.dedent(
            f"""
            from pathlib import Path

            from cuda.pathfinder._dynamic_libs.load_nvidia_dynamic_lib import (
                _resolve_system_loaded_abs_path_in_subprocess,
            )

            marker_path = Path({str(run_count_path)!r})
            run_count = int(marker_path.read_text()) if marker_path.exists() else 0
            marker_path.write_text(str(run_count + 1))

            try:
                _resolve_system_loaded_abs_path_in_subprocess("not_a_real_lib")
            except Exception:
                pass
            """
        ),
        encoding="utf-8",
    )
    env = os.environ.copy()
    existing_pythonpath = env.get("PYTHONPATH")
    env["PYTHONPATH"] = (
        str(_PACKAGE_ROOT) if not existing_pythonpath else os.pathsep.join((str(_PACKAGE_ROOT), existing_pythonpath))
    )

    result = subprocess.run(  # noqa: S603 - trusted argv: current interpreter + temp script created by this test
        [sys.executable, str(script_path)],
        capture_output=True,
        text=True,
        check=False,
        env=env,
    )

    assert result.returncode == 0, result.stderr
    assert run_count_path.read_text(encoding="utf-8") == "1"


# ---------------------------------------------------------------------------
# _try_ctk_root_canary
# ---------------------------------------------------------------------------


def _make_loaded_dl(path, found_via):
    return LoadedDL(path, False, 0xDEAD, found_via)


def test_canary_finds_nvvm(tmp_path, mocker):
    ctk_root = tmp_path / "cuda-13"
    _create_cudart_in_ctk(ctk_root)
    nvvm_lib = _create_nvvm_in_ctk(ctk_root)

    probe = mocker.patch(
        f"{_MODULE}._resolve_system_loaded_abs_path_in_subprocess",
        return_value=_fake_canary_path(ctk_root),
    )
    parent_system_loader = mocker.patch.object(load_mod.LOADER, "load_with_system_search")

    assert _try_ctk_root_canary(_ctx("nvvm")) == str(nvvm_lib)
    probe.assert_called_once_with("cudart")
    parent_system_loader.assert_not_called()


def test_canary_returns_none_when_subprocess_probe_fails(mocker):
    mocker.patch(f"{_MODULE}._resolve_system_loaded_abs_path_in_subprocess", return_value=None)
    assert _try_ctk_root_canary(_ctx("nvvm")) is None


def test_canary_returns_none_when_ctk_root_unrecognized(mocker):
    mocker.patch(
        f"{_MODULE}._resolve_system_loaded_abs_path_in_subprocess",
        return_value="/weird/path/libcudart.so.13",
    )
    assert _try_ctk_root_canary(_ctx("nvvm")) is None


def test_canary_returns_none_when_nvvm_not_in_ctk_root(tmp_path, mocker):
    ctk_root = tmp_path / "cuda-13"
    # Create only the canary lib dir, not nvvm
    _create_cudart_in_ctk(ctk_root)

    mocker.patch(
        f"{_MODULE}._resolve_system_loaded_abs_path_in_subprocess",
        return_value=_fake_canary_path(ctk_root),
    )
    assert _try_ctk_root_canary(_ctx("nvvm")) is None


def test_canary_skips_when_abs_path_none(mocker):
    mocker.patch(f"{_MODULE}._resolve_system_loaded_abs_path_in_subprocess", return_value=None)
    assert _try_ctk_root_canary(_ctx("nvvm")) is None


# ---------------------------------------------------------------------------
# _load_lib_no_cache search-order
# ---------------------------------------------------------------------------


@pytest.fixture
def _isolate_load_cascade(mocker):
    """Disable the search steps that run before system-search in _load_lib_no_cache.

    This lets the ordering tests focus on system-search, CUDA_HOME, and the
    canary probe without needing a real site-packages or conda environment.
    """

    # Skip EARLY_FIND_STEPS (site-packages + conda) so tests can focus on
    # system-search, CUDA_HOME and canary behavior.
    def _run_find_steps_with_early_disabled(ctx, steps):
        if steps is load_mod.EARLY_FIND_STEPS:
            return None
        return steps_mod.run_find_steps(ctx, steps)

    mocker.patch(f"{_MODULE}.run_find_steps", side_effect=_run_find_steps_with_early_disabled)
    # Lib not already loaded by another component
    mocker.patch.object(load_mod.LOADER, "check_if_already_loaded_from_elsewhere", return_value=None)
    # Skip transitive dependency loading
    mocker.patch(f"{_MODULE}.load_dependencies")


@pytest.mark.usefixtures("_isolate_load_cascade")
def test_cuda_home_takes_priority_over_canary(tmp_path, mocker):
    # Two competing CTK roots: one from CUDA_HOME, one the canary would find.
    cuda_home_root = tmp_path / "cuda-home"
    nvvm_home_lib = _create_nvvm_in_ctk(cuda_home_root)

    canary_root = tmp_path / "cuda-system"
    _create_cudart_in_ctk(canary_root)
    _create_nvvm_in_ctk(canary_root)

    canary_mock = mocker.MagicMock(return_value=_fake_canary_path(canary_root))

    # System search finds nothing for nvvm.
    mocker.patch.object(load_mod.LOADER, "load_with_system_search", return_value=None)
    # Canary subprocess probe would find cudart if consulted.
    mocker.patch(f"{_MODULE}._resolve_system_loaded_abs_path_in_subprocess", side_effect=canary_mock)
    # CUDA_HOME points to a separate root that also has nvvm
    mocker.patch(f"{_STEPS_MODULE}.get_cuda_path_or_home", return_value=str(cuda_home_root))
    # Capture the final load call
    mocker.patch.object(
        load_mod.LOADER,
        "load_with_abs_path",
        side_effect=lambda _desc, path, via: _make_loaded_dl(path, via),
    )

    result = _load_lib_no_cache("nvvm")

    # CUDA_HOME must win; the canary should never have been consulted
    assert result.found_via == "CUDA_HOME"
    assert result.abs_path == str(nvvm_home_lib)
    canary_mock.assert_not_called()


@pytest.mark.usefixtures("_isolate_load_cascade")
def test_canary_fires_only_after_all_earlier_steps_fail(tmp_path, mocker):
    canary_root = tmp_path / "cuda-system"
    _create_cudart_in_ctk(canary_root)
    nvvm_lib = _create_nvvm_in_ctk(canary_root)

    # System search: nvvm not on linker path.
    mocker.patch.object(load_mod.LOADER, "load_with_system_search", return_value=None)
    # Canary subprocess probe finds cudart under a system CTK root.
    mocker.patch(
        f"{_MODULE}._resolve_system_loaded_abs_path_in_subprocess",
        return_value=_fake_canary_path(canary_root),
    )
    # No CUDA_HOME set
    mocker.patch(f"{_STEPS_MODULE}.get_cuda_path_or_home", return_value=None)
    # Capture the final load call
    mocker.patch.object(
        load_mod.LOADER,
        "load_with_abs_path",
        side_effect=lambda _desc, path, via: _make_loaded_dl(path, via),
    )

    result = _load_lib_no_cache("nvvm")

    assert result.found_via == "system-ctk-root"
    assert result.abs_path == str(nvvm_lib)


@pytest.mark.usefixtures("_isolate_load_cascade")
def test_non_discoverable_lib_skips_canary_probe(mocker):
    # Force fallback path for a lib that is not canary-discoverable.
    mocker.patch.object(load_mod.LOADER, "load_with_system_search", return_value=None)
    mocker.patch(f"{_STEPS_MODULE}.get_cuda_path_or_home", return_value=None)
    canary_probe = mocker.patch(f"{_MODULE}._resolve_system_loaded_abs_path_in_subprocess")

    with pytest.raises(DynamicLibNotFoundError):
        _load_lib_no_cache("cublas")

    canary_probe.assert_not_called()
