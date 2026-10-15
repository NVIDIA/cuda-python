# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import functools
import importlib
import inspect
import pathlib
import sys
import threading
from contextlib import contextmanager

import pytest

import cuda.bindings.driver as cuda

# Keep in sync with cuda_core/tests/conftest.py.
try:
    import cuda_python_test_helpers._pytest_plugin  # noqa: F401
except ImportError as e:
    # Don't call .resolve(): resolving symlinks can make parents[2] point
    # somewhere other than the monorepo root if a sub-directory is symlinked.
    _test_helpers_root = pathlib.Path(__file__).parents[2] / "cuda_python_test_helpers"
    if not _test_helpers_root.is_dir():
        raise RuntimeError(f"cuda-python-test-helpers not installed and not found at {_test_helpers_root}") from e
    for _k in list(sys.modules):
        if _k == "cuda_python_test_helpers" or _k.startswith("cuda_python_test_helpers."):
            del sys.modules[_k]
    sys.path.insert(0, str(_test_helpers_root))
    importlib.invalidate_caches()

pytest_plugins = ["cuda_python_test_helpers._pytest_plugin"]


def pytest_configure(config):
    # When using `parallel-threads` set up mini-plugin to ensure each thread has a CUDA context
    parallel_threads = getattr(config.option, "parallel_threads", 0)
    if parallel_threads == "auto" or int(parallel_threads) > 1:
        config.pluginmanager.register(_CudaBindingsParallelPlugin(), name="_cuda_bindings_parallel_plugin")


class _ThreadContextPool:
    """CUDA contexts for pytest-run-parallel worker threads, reused across tests.

    pytest-run-parallel starts new threads for every test, and each thread
    needs a current context. Creating and destroying a context per thread per
    test serializes in the driver: with four threads it cost about 0.75 s per
    test and made the free-threaded bindings run more than ten times slower.

    The pool gives each running thread a context of its own, as before, and
    keeps it for the next test. The plugin empties the pool at the end of each
    test module, so a context lives no longer than the module-scoped ``ctx``
    fixture does in a run without threads. A context that reports an error at
    the end of a test is destroyed rather than reused.
    """

    def __init__(self):
        self._lock = threading.Lock()
        self._idle = []

    @staticmethod
    def _create():
        err, device = cuda.cuDeviceGet(0)
        assert err == cuda.CUresult.CUDA_SUCCESS
        # cuCtxCreate makes the new context current; pop it so that context()
        # can push and pop it like a reused one.
        err, ctx = cuda.cuCtxCreate(None, 0, device)
        assert err == cuda.CUresult.CUDA_SUCCESS
        err, _ = cuda.cuCtxPopCurrent()
        assert err == cuda.CUresult.CUDA_SUCCESS
        return device, ctx

    @contextmanager
    def context(self):
        with self._lock:
            entry = self._idle.pop() if self._idle else None
        if entry is None:
            entry = self._create()
        device, ctx = entry
        (err,) = cuda.cuCtxPushCurrent(ctx)
        assert err == cuda.CUresult.CUDA_SUCCESS
        try:
            yield device, ctx
        finally:
            (sync_err,) = cuda.cuCtxSynchronize()
            # Pop rather than unset, so that the context that was current
            # before the test (on the main thread, for tests that are not run
            # in parallel) is current again.
            err, _ = cuda.cuCtxPopCurrent()
            assert err == cuda.CUresult.CUDA_SUCCESS
            if sync_err == cuda.CUresult.CUDA_SUCCESS:
                with self._lock:
                    self._idle.append(entry)
            else:
                (err,) = cuda.cuCtxDestroy(ctx)
                assert err == cuda.CUresult.CUDA_SUCCESS

    def clear(self):
        """Destroy the idle contexts. Call only when no test is running."""
        with self._lock:
            entries, self._idle = self._idle, []
        for _, ctx in entries:
            (err,) = cuda.cuCtxDestroy(ctx)
            assert err == cuda.CUresult.CUDA_SUCCESS


_thread_contexts = _ThreadContextPool()


def _wrap_worker_cuda_test(func):
    if getattr(func, "_cuda_bindings_worker_cuda_wrapped", False):
        return func

    sig = inspect.signature(func)
    wants_device = "device" in sig.parameters
    wants_ctx = "ctx" in sig.parameters

    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        kwargs = dict(kwargs)  # copy before mutating
        with _thread_contexts.context() as (device, ctx):
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


    This plugin approach is not ideal, it would be nicer to introduce hooks
    into pytest-run-parallel.  Once that issue is closed this would be good
    to refactor: https://github.com/Quansight-Labs/pytest-run-parallel/issues/189
    """

    @pytest.hookimpl()
    def pytest_collection_modifyitems(self, config, items):
        for item in items:
            if _item_needs_thread_ctx(item):
                item.obj = _wrap_worker_cuda_test(item.obj)

    @pytest.hookimpl(trylast=True)
    def pytest_runtest_teardown(self, item, nextitem):
        # The worker threads of this test have joined. Destroy the pooled
        # contexts when the next test is in another module, or there is none.
        if nextitem is None or getattr(nextitem, "module", None) is not getattr(item, "module", None):
            _thread_contexts.clear()


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
