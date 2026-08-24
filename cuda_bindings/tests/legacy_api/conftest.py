# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Overrides the root conftest's `cuda_driver`/`device`/`ctx` fixtures with
versions backed by the legacy (non-v2) driver API.

The root conftest.py was ported to `cuda.bindings._v2.driver`, whose handles
are plain Python ints. Some legacy_api tests (e.g. test_legacy_cuda.py's repr
tests) assert on the repr of the old wrapped handle types (CUcontext,
CUdevice), so this directory needs its own copies of these fixtures using the
old API to stay frozen, per legacy_api/README.md.
"""

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
