# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import sys

import pytest

from cuda.pathfinder._dynamic_libs.descriptor_catalog import DescriptorSpec

# load_dl_linux binds dlinfo()/dlerror() from libdl at module scope, which only
# exists off Windows. platform_loader.py routes every non-win32 platform here.
pytestmark = pytest.mark.skipif(sys.platform == "win32", reason="Exercises the non-Windows dynamic library loader")

_SHM_OPEN_ERROR = "libcufile.so.0: undefined symbol: shm_open"

CUFILE_DESC = DescriptorSpec(
    name="cufile",
    packaged_with="ctk",
    linux_sonames=("libcufile.so.0",),
)


class FakeHandle:
    _handle = 0x1234


def _record_cdll_calls(monkeypatch, load_dl_linux, *, librt_loadable: bool = True) -> list[str]:
    """Capture ctypes.CDLL() calls made by the module under test."""
    loaded: list[str] = []

    def fake_cdll(name, *args, **kwargs):
        loaded.append(name)
        if not librt_loadable:
            raise OSError(f"{name}: cannot open shared object file")
        return FakeHandle()

    monkeypatch.setattr(load_dl_linux.ctypes, "CDLL", fake_cdll)
    return loaded


@pytest.mark.agent_authored(model="claude-opus-5")
def test_cufile_shm_open_failure_is_retried_after_loading_librt(monkeypatch):
    """nvidia-cufile wheels reference shm_open() without a DT_NEEDED entry for librt.

    See https://github.com/NVIDIA/cuda-python/issues/2313: on glibc < 2.34
    shm_open lives in librt, so the first dlopen() fails and only succeeds
    once librt's symbols are globally visible.
    """
    from cuda.pathfinder._dynamic_libs import load_dl_linux

    loaded = _record_cdll_calls(monkeypatch, load_dl_linux)
    attempts: list[str] = []

    def fake_load_lib(desc, filename):
        attempts.append(filename)
        if len(attempts) == 1:
            raise OSError(_SHM_OPEN_ERROR)
        return FakeHandle()

    monkeypatch.setattr(load_dl_linux, "_load_lib", fake_load_lib)

    loaded_dl = load_dl_linux.load_with_abs_path(CUFILE_DESC, "/site-packages/nvidia/cu13/lib/libcufile.so.0", "conda")

    assert loaded_dl.abs_path == "/site-packages/nvidia/cu13/lib/libcufile.so.0"
    assert loaded_dl.found_via == "conda"
    assert loaded_dl._handle_uint == FakeHandle._handle
    assert loaded == ["librt.so.1"]
    assert len(attempts) == 2


@pytest.mark.agent_authored(model="claude-opus-5")
def test_cufile_retry_reports_the_original_error_if_librt_does_not_help(monkeypatch):
    from cuda.pathfinder._dynamic_libs import load_dl_linux

    loaded = _record_cdll_calls(monkeypatch, load_dl_linux)

    def fake_load_lib(desc, filename):
        raise OSError(_SHM_OPEN_ERROR)

    monkeypatch.setattr(load_dl_linux, "_load_lib", fake_load_lib)

    with pytest.raises(RuntimeError, match="undefined symbol: shm_open"):
        load_dl_linux.load_with_abs_path(CUFILE_DESC, "/site-packages/nvidia/cu13/lib/libcufile.so.0")

    assert loaded == ["librt.so.1"]


@pytest.mark.agent_authored(model="claude-opus-5")
def test_cufile_retry_is_skipped_if_librt_cannot_be_loaded(monkeypatch):
    from cuda.pathfinder._dynamic_libs import load_dl_linux

    loaded = _record_cdll_calls(monkeypatch, load_dl_linux, librt_loadable=False)
    attempts: list[str] = []

    def fake_load_lib(desc, filename):
        attempts.append(filename)
        raise OSError(_SHM_OPEN_ERROR)

    monkeypatch.setattr(load_dl_linux, "_load_lib", fake_load_lib)

    with pytest.raises(RuntimeError, match="undefined symbol: shm_open"):
        load_dl_linux.load_with_abs_path(CUFILE_DESC, "/site-packages/nvidia/cu13/lib/libcufile.so.0")

    assert loaded == ["librt.so.1"]
    assert len(attempts) == 1


@pytest.mark.agent_authored(model="claude-opus-5")
@pytest.mark.parametrize(
    ("libname", "message"),
    [
        ("cufile", "libcufile.so.0: cannot open shared object file"),  # unrelated cufile failure
        ("cublas", _SHM_OPEN_ERROR),  # right symptom, wrong library
    ],
)
def test_unrelated_dlopen_failures_do_not_trigger_the_librt_workaround(monkeypatch, libname, message):
    from cuda.pathfinder._dynamic_libs import load_dl_linux

    loaded = _record_cdll_calls(monkeypatch, load_dl_linux)
    attempts: list[str] = []

    def fake_load_lib(desc, filename):
        attempts.append(filename)
        raise OSError(message)

    monkeypatch.setattr(load_dl_linux, "_load_lib", fake_load_lib)

    desc = DescriptorSpec(name=libname, packaged_with="ctk", linux_sonames=(f"lib{libname}.so.0",))
    with pytest.raises(RuntimeError, match="Failed to dlopen"):
        load_dl_linux.load_with_abs_path(desc, f"/site-packages/lib{libname}.so.0")

    assert loaded == []
    assert len(attempts) == 1
