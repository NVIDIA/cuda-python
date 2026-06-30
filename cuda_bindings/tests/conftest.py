# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: LicenseRef-NVIDIA-SOFTWARE-LICENSE

import pathlib
import sys
from contextlib import contextmanager
from importlib.metadata import PackageNotFoundError, distribution

import pytest

import cuda.bindings.driver as cuda

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


from cuda_python_test_helpers.pytest_run_parallel import (
    install_run_parallel_worker_context_patch,
    mark_item_for_worker_context,
)


def pytest_configure(config):
    install_run_parallel_worker_context_patch()


@contextmanager
def _thread_context():
    (err,) = cuda.cuInit(0)
    assert err == cuda.CUresult.CUDA_SUCCESS
    err, device = cuda.cuDeviceGet(0)
    assert err == cuda.CUresult.CUDA_SUCCESS
    err, ctx = cuda.cuCtxCreate(None, 0, device)
    assert err == cuda.CUresult.CUDA_SUCCESS
    try:
        yield device, ctx
    finally:
        (err,) = cuda.cuCtxDestroy(ctx)
        assert err == cuda.CUresult.CUDA_SUCCESS


@contextmanager
def _cuda_bindings_worker_context(*, thread_index, iteration_index, kwargs):
    with _thread_context() as (device, ctx):
        if "device" in kwargs:
            kwargs["device"] = device
        if "ctx" in kwargs:
            kwargs["ctx"] = ctx
        yield


def _is_cudla_item(item):
    nodeid = item.nodeid.replace("\\", "/")
    return nodeid.startswith("tests/cudla/") or "cuda_bindings/tests/cudla/" in nodeid


def _item_needs_thread_ctx(item):
    if _is_cudla_item(item):
        return False
    fixturenames = set(getattr(item, "fixturenames", ()))
    return bool(fixturenames & {"device", "ctx", "driver", "cufile_env_json"})


def pytest_collection_modifyitems(config, items):
    for item in items:
        if _item_needs_thread_ctx(item):
            mark_item_for_worker_context(item, _cuda_bindings_worker_context)


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
