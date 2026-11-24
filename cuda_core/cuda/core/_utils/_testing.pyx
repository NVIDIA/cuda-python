# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Test-only helpers that reach into ``cuda.core`` internals.

This module is **not** part of the public ``cuda.core`` API. It is built into
the package (like other private ``_utils`` modules) purely for the test suite,
so that production modules need no test hooks of their own.

Weak handles
------------

Weak handles let tests observe, deterministically, when the strong references
that keep a CUDA resource alive have all been released -- without relying on
driver- or hardware-specific side effects (for example, whether freed device
memory happens to remain readable).

Every resource handle is owned by a C++ ``std::shared_ptr``. A **weak handle**
is a non-owning ``std::weak_ptr`` observer of that control block: truthy while
some strong owner remains, falsy once the last one is gone. Use :func:`weak_handle`
to obtain a weak handle from a supported front-end object.

To support another type, add a ``cdef _weak_from_<type>`` that reads its ``cdef``
handle field (see ``*.pxd``), assigns to :ctype:`OpaqueHandle`, and extend the
``isinstance`` chain in :func:`weak_handle`. Types whose slots hold arbitrary
Python owners via ``make_opaque_py`` are not covered here -- use
:class:`weakref.ref` on a weak-referenceable owner object in tests instead.

Driver function wrappers
------------------------

The C++ layer calls the driver through a table of function pointers that it
fills once per process, on its first driver call, from
``cuda.bindings._internal.driver._inspect_function_pointers()``.
:func:`install_driver_fn_wrappers` replaces that function so that the fill
stores test wrappers for selected entries. Each wrapper calls the real driver
function unless a test arms it. Install the wrappers before ``cuda.core``'s
first driver call in the process: conftest.py does so for the test session,
and a spawned child must do so itself. The setters raise unless the table was
filled with the wrappers, so a test cannot silently run unarmed. Arm a wrapper
only after the process's first driver call through ``cuda.core``, for example
after creating a device or a stream.
"""

from libc.stdint cimport intptr_t

from cuda.bindings cimport cydriver
from cuda.core._memory._buffer cimport Buffer
from cuda.core.graph._graph_definition cimport GraphDefinition
from cuda.core._rt cimport OpaqueHandle, attach_rollback_failure
from cuda.core._utils.cuda_utils cimport HANDLE_RETURN

import cuda.bindings._internal.driver as _driver_loader


# Cython cannot spell ``weak_ptr[const void]`` inline (the ``const void``
# template argument fails to parse), so the weak type and its one constructor
# are provided by a small inline C++ shim local to this test-only module. This
# keeps the production resource_handles translation units untouched.
cdef extern from *:
    """
    #include <memory>
    namespace cuda_core_test {
    using OpaqueWeakHandle = std::weak_ptr<const void>;
    static inline OpaqueWeakHandle make_weak(const std::shared_ptr<const void>& h) {
        return OpaqueWeakHandle(h);
    }
    }  // namespace cuda_core_test
    """
    cppclass OpaqueWeakHandle "cuda_core_test::OpaqueWeakHandle":
        OpaqueWeakHandle()
        bint expired()
        long use_count()
    OpaqueWeakHandle make_weak "cuda_core_test::make_weak" (const OpaqueHandle& h)


cdef class WeakHandle:
    """Non-owning weak handle for a resource's shared control block.

    Truthy while some strong owner of the underlying resource handle remains,
    falsy once the last strong reference is released. Obtain instances via
    :func:`weak_handle` rather than constructing directly.
    """

    cdef OpaqueWeakHandle _w

    def __bool__(self):
        return not self._w.expired()

    def expired(self):
        """Return ``True`` once every strong owner of the handle is gone."""
        return self._w.expired()

    def use_count(self):
        """Number of strong owners currently sharing the handle."""
        return self._w.use_count()


cdef WeakHandle _weak_from_opaque(OpaqueHandle h):
    # Build the weak handle from a (temporary) strong handle. The strong copy
    # lives only for the duration of this call, so it does not perturb the
    # reference count the weak handle later reports.
    cdef WeakHandle wh = WeakHandle.__new__(WeakHandle)
    wh._w = make_weak(h)
    return wh


cdef WeakHandle _weak_from_buffer(Buffer buf):
    cdef OpaqueHandle h = buf._h_ptr
    if not h:
        raise ValueError("Buffer has no active allocation")
    return _weak_from_opaque(h)


cdef WeakHandle _weak_from_graph_definition(GraphDefinition graph):
    cdef OpaqueHandle h = graph._h_graph
    if not h:
        raise ValueError("GraphDefinition has no active graph")
    return _weak_from_opaque(h)


def weak_handle(obj):
    """Return a :class:`WeakHandle` observing the resource behind ``obj``.

    Currently supports :class:`~cuda.core.Buffer` (allocation handle) and
    :class:`~cuda.core.graph.GraphDefinition` (graph hierarchy handle). See
    the module docstring for how to add more types.

    Raises
    ------
    ValueError
        If ``obj`` is a :class:`~cuda.core.Buffer` with no active allocation.
    TypeError
        If ``obj`` is not a supported type.
    """
    if isinstance(obj, Buffer):
        return _weak_from_buffer(obj)
    if isinstance(obj, GraphDefinition):
        return _weak_from_graph_definition(obj)
    raise TypeError(
        f"weak_handle() does not support {type(obj).__name__!r}; "
        "supported types: Buffer, GraphDefinition"
    )


# Keeps the ``_for_testing`` suffix because it shares its name with the cimported
# ``attach_rollback_failure``, which this wrapper calls.
def attach_rollback_failure_for_testing(int status):
    """Attach a failed CUDA call to the exception being handled, or report it.

    Calls ``attach_rollback_failure()`` for a fictitious ``cuTestOperation``.
    Called inside an ``except`` block it adds a note to the exception being
    handled (Python 3.11+); anywhere else it emits a ``CUDAWarning``.
    """
    attach_rollback_failure(b"cuTestOperation", <cydriver.CUresult>status, b"failed while testing")


# The wrappers run on driver calls from any thread, with or without the GIL,
# and during interpreter shutdown, so they are plain C++ and never call Python.
cdef extern from *:
    """
    #include <atomic>
    #include <chrono>
    #include <condition_variable>
    #include <cstdint>
    #include <mutex>
    #include <cuda.h>
    namespace cuda_core_test {
    // The real pointers are atomic because the fill may run on two threads at
    // once in a free-threaded build, and each run calls the wrap functions.
    using CtxSetCurrentFn = CUresult(CUDAAPI *)(CUcontext);
    static std::atomic<CtxSetCurrentFn> real_ctx_set_current{nullptr};
    static thread_local CUresult ctx_set_current_fault = CUDA_SUCCESS;
    static thread_local CUcontext ctx_set_current_fault_target = nullptr;

    static CUresult CUDAAPI faulty_ctx_set_current(CUcontext ctx) {
        if (ctx_set_current_fault != CUDA_SUCCESS && ctx == ctx_set_current_fault_target) {
            CUresult fault = ctx_set_current_fault;
            ctx_set_current_fault = CUDA_SUCCESS;
            return fault;
        }
        return real_ctx_set_current.load()(ctx);
    }
    static std::intptr_t wrap_ctx_set_current(std::intptr_t real) {
        real_ctx_set_current.store(reinterpret_cast<CtxSetCurrentFn>(real));
        return reinterpret_cast<std::intptr_t>(&faulty_ctx_set_current);
    }
    static void set_ctx_set_current_fault(CUresult status, CUcontext target) {
        ctx_set_current_fault = status;
        ctx_set_current_fault_target = target;
    }

    // The pool release gate: the first cuMemPoolSetAccess on the target pool
    // after arming blocks until an import returns that pool (or a timeout),
    // then until the test opens the gate (or a grace period passes).
    // Only the armed path touches the mutex; the gate is never destroyed, so a
    // deleter that runs during process exit cannot reach a destroyed mutex.
    struct PoolReleaseGate {
        std::mutex mutex;
        std::condition_variable changed;
        CUmemoryPool pool = nullptr;
        std::chrono::duration<double> timeout{0};
        std::chrono::duration<double> grace{0};
        bool entered = false;
        bool imported = false;
        bool opened = false;
        bool disarmed = false;
    };
    static PoolReleaseGate &pool_release_gate() {
        static PoolReleaseGate *gate = new PoolReleaseGate;
        return *gate;
    }
    // Armed until the release enters the gate; watching until disarmed.
    static std::atomic<bool> pool_release_gate_armed{false};
    static std::atomic<bool> pool_release_gate_watching{false};

    using MemPoolSetAccessFn = CUresult(CUDAAPI *)(CUmemoryPool, const CUmemAccessDesc *, size_t);
    static std::atomic<MemPoolSetAccessFn> real_mempool_set_access{nullptr};

    static CUresult CUDAAPI gated_mempool_set_access(
            CUmemoryPool pool, const CUmemAccessDesc *map, size_t count) {
        if (pool_release_gate_armed.load()) {
            PoolReleaseGate &gate = pool_release_gate();
            std::unique_lock<std::mutex> lock(gate.mutex);
            if (pool_release_gate_armed.load() && pool == gate.pool) {
                pool_release_gate_armed.store(false);  // one-shot: later calls pass through
                gate.entered = true;
                gate.changed.notify_all();
                gate.changed.wait_for(lock, gate.timeout, [&] { return gate.imported || gate.disarmed; });
                if (gate.imported) {
                    gate.changed.wait_for(lock, gate.grace, [&] { return gate.opened || gate.disarmed; });
                }
            }
        }
        return real_mempool_set_access.load()(pool, map, count);
    }
    static std::intptr_t wrap_mempool_set_access(std::intptr_t real) {
        real_mempool_set_access.store(reinterpret_cast<MemPoolSetAccessFn>(real));
        return reinterpret_cast<std::intptr_t>(&gated_mempool_set_access);
    }

    using MemPoolImportFn = CUresult(CUDAAPI *)(CUmemoryPool *, void *, CUmemAllocationHandleType, unsigned long long);
    static std::atomic<MemPoolImportFn> real_mempool_import{nullptr};

    static CUresult CUDAAPI signaling_mempool_import(
            CUmemoryPool *pool, void *handle, CUmemAllocationHandleType type, unsigned long long flags) {
        CUresult status = real_mempool_import.load()(pool, handle, type, flags);
        if (status == CUDA_SUCCESS && pool_release_gate_watching.load()) {
            PoolReleaseGate &gate = pool_release_gate();
            std::lock_guard<std::mutex> lock(gate.mutex);
            // Only an import made while the release is held counts.
            if (gate.entered && !gate.disarmed && *pool == gate.pool) {
                gate.imported = true;
                gate.changed.notify_all();
            }
        }
        return status;
    }
    static std::intptr_t wrap_mempool_import(std::intptr_t real) {
        real_mempool_import.store(reinterpret_cast<MemPoolImportFn>(real));
        return reinterpret_cast<std::intptr_t>(&signaling_mempool_import);
    }

    static void arm_pool_release_gate(CUmemoryPool pool, double timeout_sec, double grace_sec) {
        PoolReleaseGate &gate = pool_release_gate();
        std::lock_guard<std::mutex> lock(gate.mutex);
        gate.pool = pool;
        gate.timeout = std::chrono::duration<double>(timeout_sec);
        gate.grace = std::chrono::duration<double>(grace_sec);
        gate.entered = false;
        gate.imported = false;
        gate.opened = false;
        gate.disarmed = false;
        pool_release_gate_watching.store(true);
        pool_release_gate_armed.store(true);
    }
    static bool wait_pool_release_gate_entered(double timeout_sec) {
        PoolReleaseGate &gate = pool_release_gate();
        std::unique_lock<std::mutex> lock(gate.mutex);
        return gate.changed.wait_for(
            lock, std::chrono::duration<double>(timeout_sec), [&] { return gate.entered || gate.disarmed; })
            && gate.entered;
    }
    static void open_pool_release_gate() {
        PoolReleaseGate &gate = pool_release_gate();
        std::lock_guard<std::mutex> lock(gate.mutex);
        gate.opened = true;
        gate.changed.notify_all();
    }
    static bool disarm_pool_release_gate() {
        PoolReleaseGate &gate = pool_release_gate();
        std::lock_guard<std::mutex> lock(gate.mutex);
        pool_release_gate_armed.store(false);
        pool_release_gate_watching.store(false);
        const bool overlapped = gate.entered && gate.imported;
        gate.disarmed = true;
        gate.changed.notify_all();  // release a held release or waiter
        return overlapped;
    }
    }  // namespace cuda_core_test
    """
    intptr_t _wrap_ctx_set_current "cuda_core_test::wrap_ctx_set_current" (intptr_t real) noexcept
    void _set_ctx_set_current_fault "cuda_core_test::set_ctx_set_current_fault" (
        cydriver.CUresult status, cydriver.CUcontext target) noexcept nogil
    intptr_t _wrap_mempool_set_access "cuda_core_test::wrap_mempool_set_access" (intptr_t real) noexcept
    intptr_t _wrap_mempool_import "cuda_core_test::wrap_mempool_import" (intptr_t real) noexcept
    void _arm_pool_release_gate "cuda_core_test::arm_pool_release_gate" (
        cydriver.CUmemoryPool pool, double timeout_sec, double grace_sec) noexcept nogil
    bint _wait_pool_release_gate_entered "cuda_core_test::wait_pool_release_gate_entered" (
        double timeout_sec) noexcept nogil
    void _open_pool_release_gate "cuda_core_test::open_pool_release_gate" () noexcept nogil
    bint _disarm_pool_release_gate "cuda_core_test::disarm_pool_release_gate" () noexcept nogil


_inspect_without_wrappers = None
_wrappers_installed = False


def _inspect_with_wrappers():
    global _wrappers_installed
    pointers = dict(_inspect_without_wrappers())
    pointers["__cuCtxSetCurrent"] = _wrap_ctx_set_current(pointers["__cuCtxSetCurrent"])
    pointers["__cuMemPoolSetAccess"] = _wrap_mempool_set_access(pointers["__cuMemPoolSetAccess"])
    pointers["__cuMemPoolImportFromShareableHandle"] = _wrap_mempool_import(
        pointers["__cuMemPoolImportFromShareableHandle"])
    _wrappers_installed = True
    return pointers


def install_driver_fn_wrappers():
    """Make the C++ driver function table use the test wrappers in this process.

    Call before ``cuda.core`` makes its first driver call; see the module
    docstring. Calling it again has no effect.
    """
    global _inspect_without_wrappers
    if _inspect_without_wrappers is None:
        _inspect_without_wrappers = _driver_loader._inspect_function_pointers
        _driver_loader._inspect_function_pointers = _inspect_with_wrappers


cdef int _require_driver_fn_wrappers() except -1:
    if not _wrappers_installed:
        raise RuntimeError(
            "cuda.core's driver function table does not hold the test wrappers yet. "
            "Either install_driver_fn_wrappers() was not called before cuda.core's first driver call "
            "in this process, or cuda.core has not made a driver call yet; arm the wrapper after one"
        )
    return 0


def arm_pool_release_gate(intptr_t pool, double timeout_sec, double grace_sec):
    """Hold the next release of ``pool`` while the pool is imported again.

    The C++ layer calls ``cuMemPoolSetAccess`` only to revoke access just
    before it destroys a pool (or an imported pool's imports), after its last
    reference is gone. The first such call on ``pool`` after arming blocks
    until a ``cuMemPoolImportFromShareableHandle`` from the C++ layer returns
    ``pool``, or until ``timeout_sec`` passes. Once that import has returned,
    it stays blocked until :func:`open_pool_release_gate` is called, or until
    ``grace_sec`` passes, and then proceeds. Later calls pass through.

    Open the gate once the import has finished. An import that correctly waits
    for the release cannot finish first, so the grace period ends the hold; an
    import that does not wait finishes, opens the gate, and the release then
    revokes access after the import was granted it, so the test sees the
    damage. Use :func:`wait_pool_release_gate_entered` to start the import once
    the release is held, and :func:`disarm_pool_release_gate` to check that the
    import overlapped the release.
    """
    _require_driver_fn_wrappers()
    _arm_pool_release_gate(<cydriver.CUmemoryPool>pool, timeout_sec, grace_sec)


def open_pool_release_gate():
    """Let a held release proceed once the import that overlapped it has finished."""
    _open_pool_release_gate()


def wait_pool_release_gate_entered(double timeout_sec):
    """Wait until a release is held by the gate; return ``False`` on timeout or disarm."""
    cdef bint entered
    with nogil:
        entered = _wait_pool_release_gate_entered(timeout_sec)
    return entered


def disarm_pool_release_gate():
    """Disarm the gate and let a held release proceed.

    Return ``True`` if a release was held and an import of the pool returned
    while it was held, so the import overlapped the release.
    """
    return _disarm_pool_release_gate()


def set_context_restore_fault(int status):
    """Make ``cuda.core``'s next switch back to the current context fail on this thread.

    The next ``cuCtxSetCurrent`` that the C++ layer makes on this thread to the
    context that is current now returns ``status`` without switching, so the
    context it switched to stays current, exactly as a real failure would. The
    C++ layer switches contexts only to enter an operation's context and to
    restore the caller's afterwards, and it does not switch to a context that
    is already current, so only a restoration can fail. 0 disarms the fault.
    """
    cdef cydriver.CUcontext current = NULL
    if status:
        _require_driver_fn_wrappers()
        HANDLE_RETURN(cydriver.cuCtxGetCurrent(&current))
    _set_ctx_set_current_fault(<cydriver.CUresult>status, current)
