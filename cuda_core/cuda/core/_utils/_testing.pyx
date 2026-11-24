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
and a spawned child must do so itself. The setters raise if the table was filled without the wrappers, so a
test cannot silently run unarmed.
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
    #include <cstdint>
    #include <thread>
    #include <cuda.h>
    namespace cuda_core_testing {
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

    using MemPoolSetAccessFn = CUresult(CUDAAPI *)(CUmemoryPool, const CUmemAccessDesc *, size_t);
    static std::atomic<MemPoolSetAccessFn> real_mempool_set_access{nullptr};
    static std::atomic<unsigned int> mempool_set_access_delay_us{0};

    static CUresult CUDAAPI delayed_mempool_set_access(
            CUmemoryPool pool, const CUmemAccessDesc *map, size_t count) {
        if (unsigned int delay = mempool_set_access_delay_us.load()) {
            std::this_thread::sleep_for(std::chrono::microseconds(delay));
        }
        return real_mempool_set_access.load()(pool, map, count);
    }
    static std::intptr_t wrap_mempool_set_access(std::intptr_t real) {
        real_mempool_set_access.store(reinterpret_cast<MemPoolSetAccessFn>(real));
        return reinterpret_cast<std::intptr_t>(&delayed_mempool_set_access);
    }
    static void set_mempool_set_access_delay(unsigned int microseconds) {
        mempool_set_access_delay_us.store(microseconds);
    }
    }  // namespace cuda_core_testing
    """
    intptr_t _wrap_ctx_set_current "cuda_core_testing::wrap_ctx_set_current" (intptr_t real) noexcept
    void _set_ctx_set_current_fault "cuda_core_testing::set_ctx_set_current_fault" (
        cydriver.CUresult status, cydriver.CUcontext target) noexcept nogil
    intptr_t _wrap_mempool_set_access "cuda_core_testing::wrap_mempool_set_access" (intptr_t real) noexcept
    void _set_mempool_set_access_delay "cuda_core_testing::set_mempool_set_access_delay" (
        unsigned int microseconds) noexcept nogil


_inspect_without_wrappers = None
_wrappers_installed = False


def _inspect_with_wrappers():
    global _wrappers_installed
    pointers = dict(_inspect_without_wrappers())
    pointers["__cuCtxSetCurrent"] = _wrap_ctx_set_current(pointers["__cuCtxSetCurrent"])
    pointers["__cuMemPoolSetAccess"] = _wrap_mempool_set_access(pointers["__cuMemPoolSetAccess"])
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
            "cuda.core filled its driver function table without the test wrappers; "
            "call install_driver_fn_wrappers() before cuda.core's first driver call in this process"
        )
    return 0


def set_mempool_set_access_delay(unsigned int microseconds):
    """Make every ``cuMemPoolSetAccess`` call from the C++ layer sleep first.

    The C++ layer calls ``cuMemPoolSetAccess`` only to revoke access just
    before it destroys a pool (or an imported pool's imports), so the delay
    holds a pool's release in that state. 0 disables the delay.
    """
    if microseconds:
        _require_driver_fn_wrappers()
    _set_mempool_set_access_delay(microseconds)


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
