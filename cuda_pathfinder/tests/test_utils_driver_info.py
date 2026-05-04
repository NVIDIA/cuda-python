# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import ctypes

import pytest

from cuda.pathfinder._dynamic_libs.load_dl_common import LoadedDL
from cuda.pathfinder._utils import driver_info


@pytest.fixture(autouse=True)
def _clear_driver_cuda_version_query_cache():
    driver_info.query_driver_cuda_version.cache_clear()
    driver_info.query_driver_release_version.cache_clear()
    yield
    driver_info.query_driver_cuda_version.cache_clear()
    driver_info.query_driver_release_version.cache_clear()


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


class _FakeNvmlFunction:
    def __init__(self, func):
        self.argtypes = None
        self.restype = None
        self._func = func

    def __call__(self, *args):
        return self._func(*args)


class _FakeNvmlLib:
    def __init__(
        self,
        *,
        init_status: int = 0,
        driver_release_version: str = "595.58.03",
        query_status: int = 0,
        shutdown_statuses: tuple[int, ...] = (0,),
    ):
        self.shutdown_calls = 0
        remaining_shutdown_statuses = list(shutdown_statuses)

        self.nvmlInit_v2 = _FakeNvmlFunction(lambda: init_status)

        def nvml_system_get_driver_version(version_buffer, _buffer_length) -> int:
            if query_status != 0:
                return query_status
            version_buffer.value = driver_release_version.encode()
            return 0

        self.nvmlSystemGetDriverVersion = _FakeNvmlFunction(nvml_system_get_driver_version)

        def nvml_shutdown() -> int:
            self.shutdown_calls += 1
            if remaining_shutdown_statuses:
                return remaining_shutdown_statuses.pop(0)
            return 0

        self.nvmlShutdown = _FakeNvmlFunction(nvml_shutdown)


def _loaded_cuda(abs_path: str) -> LoadedDL:
    return LoadedDL(
        abs_path=abs_path,
        was_already_loaded_from_elsewhere=False,
        _handle_uint=0xBEEF,
        found_via="system-search",
    )


def _loaded_nvml(abs_path: str) -> LoadedDL:
    return LoadedDL(
        abs_path=abs_path,
        was_already_loaded_from_elsewhere=False,
        _handle_uint=0xCAFE,
        found_via="system-search",
    )


def test_driver_release_version_from_text_parses_branch():
    assert driver_info.DriverReleaseVersion.from_text("595.58.03") == driver_info.DriverReleaseVersion(
        text="595.58.03",
        components=(595, 58, 3),
        branch=595,
    )


def test_query_driver_release_version_wraps_internal_failures(monkeypatch):
    root_cause = RuntimeError("low-level release query failed")

    def fail_query_driver_release_version_text() -> str:
        raise root_cause

    monkeypatch.setattr(driver_info, "_query_driver_release_version_text", fail_query_driver_release_version_text)

    with pytest.raises(
        driver_info.QueryDriverReleaseVersionError,
        match="Failed to query the display-driver release version",
    ) as exc_info:
        driver_info.query_driver_release_version()

    assert exc_info.value.__cause__ is root_cause


def test_query_driver_release_version_text_uses_nvml(monkeypatch):
    fake_nvml_lib = _FakeNvmlLib(driver_release_version="595.58.03")
    loaded_paths: list[str] = []

    monkeypatch.setattr(
        driver_info,
        "_load_nvidia_dynamic_lib",
        lambda _libname: _loaded_nvml("/usr/lib/libnvidia-ml.so.1"),
    )

    def fake_cdll(abs_path: str):
        loaded_paths.append(abs_path)
        return fake_nvml_lib

    monkeypatch.setattr(driver_info.ctypes, "CDLL", fake_cdll)

    assert driver_info._query_driver_release_version_text() == "595.58.03"
    assert loaded_paths == ["/usr/lib/libnvidia-ml.so.1"]
    assert fake_nvml_lib.shutdown_calls == 1


def test_query_driver_release_version_text_raises_when_nvml_call_fails(monkeypatch):
    fake_nvml_lib = _FakeNvmlLib(query_status=1)

    monkeypatch.setattr(
        driver_info,
        "_load_nvidia_dynamic_lib",
        lambda _libname: _loaded_nvml("/usr/lib/libnvidia-ml.so.1"),
    )
    monkeypatch.setattr(driver_info.ctypes, "CDLL", lambda _abs_path: fake_nvml_lib)

    with pytest.raises(RuntimeError, match=r"nvmlSystemGetDriverVersion\(\) \(status=1\)"):
        driver_info._query_driver_release_version_text()
    assert fake_nvml_lib.shutdown_calls == 1


def test_query_driver_cuda_version_uses_windll_on_windows(monkeypatch):
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

    assert driver_info._query_driver_cuda_version_int() == 12080
    assert loaded_paths == [r"C:\Windows\System32\nvcuda.dll"]


def test_driver_cuda_version_from_encoded_returns_subclass_instance():
    version = driver_info.DriverCudaVersion.from_encoded(12080)

    assert version == driver_info.DriverCudaVersion(
        encoded=12080,
        major=12,
        minor=8,
    )
    assert type(version) is driver_info.DriverCudaVersion


def test_query_driver_cuda_version_wraps_internal_failures(monkeypatch):
    root_cause = RuntimeError("low-level query failed")

    def fail_query_driver_cuda_version_int() -> int:
        raise root_cause

    monkeypatch.setattr(driver_info, "_query_driver_cuda_version_int", fail_query_driver_cuda_version_int)

    with pytest.raises(
        driver_info.QueryDriverCudaVersionError,
        match="Failed to query the CUDA driver version",
    ) as exc_info:
        driver_info.query_driver_cuda_version()

    assert exc_info.value.__cause__ is root_cause


def test_query_driver_cuda_version_int_raises_when_cuda_call_fails(monkeypatch):
    fake_driver_lib = _FakeDriverLib(status=1, version=0)

    monkeypatch.setattr(driver_info, "IS_WINDOWS", False)
    monkeypatch.setattr(driver_info, "_load_nvidia_dynamic_lib", lambda _libname: _loaded_cuda("/usr/lib/libcuda.so.1"))
    monkeypatch.setattr(driver_info.ctypes, "CDLL", lambda _abs_path: fake_driver_lib)

    with pytest.raises(RuntimeError, match=r"cuDriverGetVersion\(\) \(status=1\)"):
        driver_info._query_driver_cuda_version_int()
