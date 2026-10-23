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

import cuda.bindings.driver as legacy_driver
from cuda.bindings._v2 import driver

# BEGIN SYNCED PYTEST PLUGIN BOOTSTRAP
# Keep every block with this marker byte-for-byte identical.
# Before editing or reviewing, find all copies; see root AGENTS.md.
# Keep the shared pytest plugin available in wheel-test jobs, which run from
# the monorepo checkout without installing cuda-python-test-helpers.
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
# END SYNCED PYTEST PLUGIN BOOTSTRAP


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
    fixture does in a run without threads.

    The worker threads of a test now run at the same time. Do not add a
    synchronizing call between tests here: while another thread captures a
    stream in global mode, such a call fails and breaks that capture.

    Subclasses implement the driver calls, so that the objects passed to the
    tests have the types those tests expect.
    """

    def __init__(self):
        self._lock = threading.Lock()
        self._idle = []

    def _create(self):
        """Return ``(device, ctx)``; ``ctx`` must not be left current."""
        raise NotImplementedError

    def _push(self, ctx):
        raise NotImplementedError

    def _pop(self):
        raise NotImplementedError

    def _destroy(self, ctx):
        raise NotImplementedError

    @contextmanager
    def context(self):
        with self._lock:
            if self._idle:
                entry = self._idle.pop()
            else:
                entry = None
        if entry is None:
            entry = self._create()
        device, ctx = entry
        self._push(ctx)
        try:
            yield device, ctx
        finally:
            # Pop rather than unset, so that the context that was current
            # before the test (on the main thread, for tests that are not run
            # in parallel) is current again.
            self._pop()
            with self._lock:
                self._idle.append(entry)

    def clear(self):
        """Destroy the idle contexts. Call only when no test is running."""
        with self._lock:
            entries, self._idle = self._idle, []
        for _, ctx in entries:
            self._destroy(ctx)


class _V2ThreadContextPool(_ThreadContextPool):
    def _create(self):
        device = driver.device_get(0)
        # ctx_create_v4 makes the new context current; pop it so that
        # context() can push and pop it like a reused one.
        ctx = driver.ctx_create_v4(None, 0, device)
        driver.ctx_pop_current_v2()
        return device, ctx

    def _push(self, ctx):
        driver.ctx_push_current_v2(ctx)

    def _pop(self):
        driver.ctx_pop_current_v2()

    def _destroy(self, ctx):
        driver.ctx_destroy_v2(ctx)


class _LegacyThreadContextPool(_ThreadContextPool):
    # Uses the legacy API so that the objects passed to tests/legacy_api have
    # the types those tests expect.
    def _create(self):
        err, device = legacy_driver.cuDeviceGet(0)
        assert err == legacy_driver.CUresult.CUDA_SUCCESS
        err, ctx = legacy_driver.cuCtxCreate(None, 0, device)
        assert err == legacy_driver.CUresult.CUDA_SUCCESS
        err, _ = legacy_driver.cuCtxPopCurrent()
        assert err == legacy_driver.CUresult.CUDA_SUCCESS
        return device, ctx

    def _push(self, ctx):
        (err,) = legacy_driver.cuCtxPushCurrent(ctx)
        assert err == legacy_driver.CUresult.CUDA_SUCCESS

    def _pop(self):
        err, _ = legacy_driver.cuCtxPopCurrent()
        assert err == legacy_driver.CUresult.CUDA_SUCCESS

    def _destroy(self, ctx):
        (err,) = legacy_driver.cuCtxDestroy(ctx)
        assert err == legacy_driver.CUresult.CUDA_SUCCESS


_thread_contexts = _V2ThreadContextPool()
_legacy_thread_contexts = _LegacyThreadContextPool()


def _wrap_worker_cuda_test(func, thread_contexts):
    if getattr(func, "_cuda_bindings_worker_cuda_wrapped", False):
        return func

    sig = inspect.signature(func)
    wants_device = "device" in sig.parameters
    wants_ctx = "ctx" in sig.parameters

    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        kwargs = dict(kwargs)  # copy before mutating
        with thread_contexts.context() as (device, ctx):
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
                if "legacy_api" in item.path.parts:
                    thread_contexts = _legacy_thread_contexts
                else:
                    thread_contexts = _thread_contexts
                item.obj = _wrap_worker_cuda_test(item.obj, thread_contexts)

    @pytest.hookimpl(trylast=True)
    def pytest_runtest_teardown(self, item, nextitem):
        # The worker threads of this test have joined. Destroy the pooled
        # contexts when the next test is in another module, or there is none.
        if nextitem is None or getattr(nextitem, "module", None) is not getattr(item, "module", None):
            _thread_contexts.clear()
            _legacy_thread_contexts.clear()


@pytest.fixture(scope="module")
def cuda_driver():
    driver.init(0)


@pytest.fixture(scope="module")
def device(cuda_driver):
    return driver.device_get(0)


@pytest.fixture(scope="module", autouse=True)
def ctx(device):
    # Construct context
    ctx = driver.ctx_create_v4(None, 0, device)
    yield ctx
    driver.ctx_destroy_v2(ctx)
