# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

# Fixtures for the legacy-API tests. These override the same-named fixtures in
# the parent conftest.py, which use the ``cuda.bindings._v2.driver`` API.

import pytest

import cuda.bindings.driver as cuda


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
