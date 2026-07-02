# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: LicenseRef-NVIDIA-SOFTWARE-LICENSE

import functools
import inspect
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


def pytest_configure(config):
    # When using `parallel-threads` set up mini-plugin to ensure each thread has a CUDA context
    parallel_threads = getattr(config.option, "parallel_threads", 0)
    if parallel_threads == "auto" or int(parallel_threads) > 1:
        config.pluginmanager.register(_CudaBindingsParallelPlugin(), name="_cuda_bindings_parallel_plugin")


@contextmanager
def _thread_context():
    # Context setting up `device` and `ctx` for individual threads on
    # pytest-run-parallel
    err, device = cuda.cuDeviceGet(0)
    assert err == cuda.CUresult.CUDA_SUCCESS
    err, ctx = cuda.cuCtxCreate(None, 0, device)
    assert err == cuda.CUresult.CUDA_SUCCESS
    try:
        yield device, ctx
    finally:
        (err,) = cuda.cuCtxDestroy(ctx)
        assert err == cuda.CUresult.CUDA_SUCCESS


def _wrap_worker_cuda_test(func):
    if getattr(func, "_cuda_bindings_worker_cuda_wrapped", False):
        return func

    sig = inspect.signature(func)
    wants_device = "device" in sig.parameters
    wants_ctx = "ctx" in sig.parameters

    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        with _thread_context() as (device, ctx):
            # device is None when reusing an existing context (defensive path);
            # keep whatever the fixture provided in kwargs as-is.
            if wants_device and device is not None:
                kwargs["device"] = device
            if wants_ctx:
                kwargs["ctx"] = ctx
            return func(*args, **kwargs)

    wrapper._cuda_bindings_worker_cuda_wrapped = True
    return wrapper


def _item_needs_thread_ctx(item):
    fixturenames = getattr(item, "fixturenames", ())
    # The 'device' fixture is the main fixture to set up a CUDA context.
    # 'driver' is specific to the cufile tests and used there instead.
    return "device" in fixturenames or "driver" in fixturenames


class _CudaBindingsParallelPlugin:
    """A mini pytest plugin used only for pytest-run-parallel testing.
    pytest-run-parallel spawns new threads for each test and we need to
    initialize and pass the correct CUDA context for each these.

    This plugin looks for context specific fixtures and replaces them
    new context specific fixtures may have to be added.
    """

    @pytest.hookimpl()
    def pytest_collection_modifyitems(self, config, items):
        for item in items:
            if _item_needs_thread_ctx(item):
                item.obj = _wrap_worker_cuda_test(item.obj)


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
