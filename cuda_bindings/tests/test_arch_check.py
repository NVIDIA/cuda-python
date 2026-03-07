# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: LicenseRef-NVIDIA-SOFTWARE-LICENSE


import pytest

from cuda.bindings import nvml
from cuda.bindings._test_helpers import arch_check
from cuda.pathfinder import DynamicLibNotFoundError


def _raise(exc):
    def _inner():
        raise exc

    return _inner


def _make_not_supported_error():
    err = nvml.NotSupportedError.__new__(nvml.NotSupportedError)
    Exception.__init__(err, "Not supported")
    return err


def _make_lib_rm_version_mismatch_error():
    err = nvml.LibRmVersionMismatchError.__new__(nvml.LibRmVersionMismatchError)
    Exception.__init__(err, "Driver/library version mismatch")
    return err


def _make_dynamic_lib_not_found_error():
    return DynamicLibNotFoundError("Failure finding libnvml.so")


@pytest.fixture(autouse=True)
def clear_hardware_supports_nvml_cache():
    arch_check.hardware_supports_nvml.cache_clear()
    yield
    arch_check.hardware_supports_nvml.cache_clear()


def test_hardware_supports_nvml_returns_true_when_probe_succeeds(monkeypatch):
    calls = []

    monkeypatch.setattr(arch_check.nvml, "init_v2", lambda: calls.append("init"))
    monkeypatch.setattr(arch_check.nvml, "system_get_driver_branch", lambda: "560")
    monkeypatch.setattr(arch_check.nvml, "shutdown", lambda: calls.append("shutdown"))

    assert arch_check.hardware_supports_nvml() is True
    assert calls == ["init", "shutdown"]


def test_hardware_supports_nvml_returns_false_for_not_supported(monkeypatch):
    calls = []

    monkeypatch.setattr(arch_check.nvml, "init_v2", lambda: calls.append("init"))
    monkeypatch.setattr(arch_check.nvml, "system_get_driver_branch", _raise(_make_not_supported_error()))
    monkeypatch.setattr(arch_check.nvml, "shutdown", lambda: calls.append("shutdown"))

    assert arch_check.hardware_supports_nvml() is False
    assert calls == ["init", "shutdown"]


@pytest.mark.parametrize(
    "error_factory",
    [
        _make_lib_rm_version_mismatch_error,
        _make_dynamic_lib_not_found_error,
    ],
)
def test_hardware_supports_nvml_runtime_errors_skip_locally(monkeypatch, error_factory):
    monkeypatch.delenv("CI", raising=False)
    monkeypatch.setattr(arch_check.nvml, "init_v2", _raise(error_factory()))

    assert arch_check.hardware_supports_nvml() is False


@pytest.mark.parametrize(
    "error_factory",
    [
        _make_lib_rm_version_mismatch_error,
        _make_dynamic_lib_not_found_error,
    ],
)
def test_hardware_supports_nvml_runtime_errors_fail_in_ci(monkeypatch, error_factory):
    err = error_factory()
    monkeypatch.setenv("CI", "1")
    monkeypatch.setattr(arch_check.nvml, "init_v2", _raise(err))

    with pytest.raises(type(err)):
        arch_check.hardware_supports_nvml()
