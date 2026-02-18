# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: LicenseRef-NVIDIA-SOFTWARE-LICENSE

import pathlib
import sys
from importlib.metadata import PackageNotFoundError, distribution

import cuda.bindings.driver as cuda
import pytest

# Import shared test helpers for tests across subprojects.
# PLEASE KEEP IN SYNC with copies in other conftest.py in this repo.
_test_helpers_root = pathlib.Path(__file__).resolve().parents[2] / "cuda_python_test_helpers"
try:
    distribution("cuda-python-test-helpers")
except PackageNotFoundError as exc:
    if not _test_helpers_root.is_dir():
        raise RuntimeError(
            f"cuda-python-test-helpers not installed; expected checkout path {_test_helpers_root}"
        ) from exc

    test_helpers_root = str(_test_helpers_root)
    if test_helpers_root not in sys.path:
        sys.path.insert(0, test_helpers_root)


@pytest.fixture(scope="module")
def cuda_driver():
    (err,) = cuda.cuInit(0)
    assert err == cuda.CUresult.CUDA_SUCCESS


@pytest.fixture(scope="module")
def device(cuda_driver):
    err, device = cuda.cuDeviceGet(0)
    assert err == cuda.CUresult.CUDA_SUCCESS
    return device


@pytest.fixture(scope="module", autouse=True)
def ctx(device):
    # Construct context
    err, ctx = cuda.cuCtxCreate(None, 0, device)
    assert err == cuda.CUresult.CUDA_SUCCESS
    yield ctx
    (err,) = cuda.cuCtxDestroy(ctx)
    assert err == cuda.CUresult.CUDA_SUCCESS
