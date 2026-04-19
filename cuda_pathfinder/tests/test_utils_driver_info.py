# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import ctypes

import pytest

from cuda.pathfinder._dynamic_libs.load_dl_common import LoadedDL
from cuda.pathfinder._utils import driver_info


class _FakeCuDriverGetVersion:
    def __init__(self, *, status: int, version: int):
        self.argtypes = None
        self.restype = None
        self._status = status
        self._version = version

    def __call__(self, version_ptr) -> int:
        ctypes.cast(version_ptr, ctypes.POINTER(ctypes.c_int)).contents.value = self._version
        return self._status


class _FakeDriverLib:
    def __init__(self, *, status: int, version: int):
        self.cuDriverGetVersion = _FakeCuDriverGetVersion(status=status, version=version)


def _loaded_cuda(abs_path: str) -> LoadedDL:
    return LoadedDL(
        abs_path=abs_path,
        was_already_loaded_from_elsewhere=False,
        _handle_uint=0xBEEF,
        found_via="system-search",
    )


def test_query_driver_release_version_text():
    driver_info._load_nvidia_dynamic_lib.cache_clear()
    try:
        release_version = driver_info._query_driver_release_version_text()
    finally:
        driver_info._load_nvidia_dynamic_lib.cache_clear()

    components = tuple(int(component) for component in release_version.split("."))
    assert len(components) in (2, 3)
    assert 400 <= components[0] < 1000
    for component in components[1:]:
        assert component >= 0


def test_query_driver_version_uses_windll_on_windows(monkeypatch):
    fake_driver_lib = _FakeDriverLib(status=0, version=12080)
    loaded_paths: list[str] = []

    monkeypatch.setattr(driver_info, "IS_WINDOWS", True)
    monkeypatch.setattr(
        driver_info,
        "_load_nvidia_dynamic_lib",
        lambda _libname: _loaded_cuda(r"C:\Windows\System32\nvcuda.dll"),
    )

    def fake_windll(abs_path: str):
        loaded_paths.append(abs_path)
        return fake_driver_lib

    monkeypatch.setattr(driver_info.ctypes, "WinDLL", fake_windll, raising=False)

    assert driver_info._query_driver_version_int() == 12080
    assert loaded_paths == [r"C:\Windows\System32\nvcuda.dll"]


def test_query_driver_version_returns_parsed_dataclass(monkeypatch):
    monkeypatch.setattr(driver_info, "_query_driver_version_int", lambda: 12080)

    assert driver_info.query_driver_version() == driver_info.DriverCudaVersion(
        encoded=12080,
        major=12,
        minor=8,
    )


def test_query_driver_version_int_raises_when_cuda_call_fails(monkeypatch):
    fake_driver_lib = _FakeDriverLib(status=1, version=0)

    monkeypatch.setattr(driver_info, "IS_WINDOWS", False)
    monkeypatch.setattr(driver_info, "_load_nvidia_dynamic_lib", lambda _libname: _loaded_cuda("/usr/lib/libcuda.so.1"))
    monkeypatch.setattr(driver_info.ctypes, "CDLL", lambda _abs_path: fake_driver_lib)

    with pytest.raises(RuntimeError, match=r"cuDriverGetVersion\(\) \(status=1\)"):
        driver_info._query_driver_version_int()
