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


def _parallel_threads_enabled(config):
    parallel_threads = getattr(config.option, "parallel_threads", 0)
    if parallel_threads == "auto":
        return True
    return parallel_threads is not None and int(parallel_threads) > 0


def pytest_configure(config):
    if _parallel_threads_enabled(config):
        config.pluginmanager.register(_CudaBindingsParallelPlugin(), name="_cuda_bindings_parallel_plugin")


@contextmanager
def _thread_context():
    # Defensive: if this worker thread already has an active context (e.g. from
    # double-wrapping), reuse it rather than pushing another one.
    # Note: fixtures never run on the test thread; this is purely a safety net.
    err, existing = cuda.cuCtxGetCurrent()
    if err == cuda.CUresult.CUDA_SUCCESS and existing and int(existing) != 0:
        yield None, existing
        return

    # cuInit(0) is idempotent; safe to call even if cuda_driver fixture already ran.
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
    # 'device' is present when the module-level ctx(device) autouse chain is
    # active (test_cuda.py, test_kernelParams.py, nvml tests, …).
    # 'driver' is present for test_cufile.py tests that use the local driver
    # fixture; their local ctx() shadows the parent ctx(device) so 'device'
    # does not appear in their fixture chain, but they still need a per-thread
    # CUDA context for cuMemAlloc and similar calls made inside the test.
    return "device" in fixturenames or "driver" in fixturenames


class _CudaBindingsParallelPlugin:
    @pytest.hookimpl(tryfirst=True)
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
