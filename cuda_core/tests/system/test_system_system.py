# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

# ruff: noqa: E402

import os

import pytest
from cuda.core import system

from .conftest import skip_if_nvml_unsupported


def test_cuda_driver_version():
    cuda_driver_version = system.get_driver_version_full()
    assert isinstance(cuda_driver_version, tuple)
    assert len(cuda_driver_version) == 3

    ver_maj, ver_min, ver_patch = cuda_driver_version
    assert ver_maj >= 10
    assert 0 <= ver_min <= 99
    assert 0 <= ver_patch <= 9


@skip_if_nvml_unsupported
def test_gpu_driver_version():
    driver_version = system.get_gpu_driver_version()
    assert isinstance(driver_version, tuple)
    assert len(driver_version) in (2, 3)

    (ver_maj, ver_min, *ver_patch) = driver_version
    assert 400 <= ver_maj < 1000
    assert ver_min >= 0
    if ver_patch:
        assert 0 <= ver_patch[0] <= 99


@skip_if_nvml_unsupported
def test_nvml_version():
    nvml_version = system.get_nvml_version()
    assert isinstance(nvml_version, tuple)
    assert len(nvml_version) in (3, 4)

    (cuda_ver_maj, ver_maj, ver_min, *ver_patch) = nvml_version
    assert cuda_ver_maj >= 10
    assert 400 <= ver_maj < 1000
    assert ver_min >= 0
    if ver_patch:
        assert 0 <= ver_patch[0] <= 99


@skip_if_nvml_unsupported
def test_get_process_name():
    try:
        process_name = system.get_process_name(os.getpid())
    except system.NotFoundError:
        pytest.skip("Process not found")

    assert isinstance(process_name, str)
    assert "python" in process_name


def test_device_count():
    device_count = system.get_num_devices()
    assert isinstance(device_count, int)
    assert device_count >= 0
