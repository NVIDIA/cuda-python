// SPDX-FileCopyrightText: Copyright (c) 2024-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <Python.h>
#include "types.hpp"
#include <cstddef>
#include <cstdint>

namespace cuda_core::rt {

#if PY_VERSION_HEX < 0x030D0000
extern "C" int _Py_IsFinalizing(void);
#endif

// Best-effort probe for interpreter shutdown.
//
// In CPython this is not a hard guarantee: finalization can begin after this
// returns false but before a later PyGILState_Ensure() or other Python C-API
// call.
//
// If that race is lost on a non-finalizer thread, CPython's behavior is
// version-dependent: on older supported versions (3.10-3.13) it may abruptly
// terminate the current thread (historically via PyThread_exit_thread(),
// without normal C++ unwinding), while on newer versions (3.14+) it may hang
// the thread until process exit.
//
// We still use this check because the policy in this layer is to avoid Python
// work once shutdown is underway and accept an intentional leak or skipped
// Python conversion in that edge case rather than add more complex deferral
// machinery.
inline bool py_is_finalizing() noexcept {
#if PY_VERSION_HEX >= 0x030D0000
    return Py_IsFinalizing();
#else
    return _Py_IsFinalizing() != 0;
#endif
}

// as_py() - convert handle to Python wrapper object (returns new reference)
namespace detail {
// n.b. class lookup is not cached to avoid deadlock hazard, see DESIGN.md
inline PyObject* make_py(const char* module_name, const char* class_name, std::intptr_t value) noexcept {
    if (py_is_finalizing()) {
        Py_RETURN_NONE;
    }
    PyObject* mod = PyImport_ImportModule(module_name);
    if (!mod) return nullptr;
    PyObject* cls = PyObject_GetAttrString(mod, class_name);
    Py_DECREF(mod);
    if (!cls) return nullptr;
    PyObject* result = PyObject_CallFunction(cls, "L", value);
    Py_DECREF(cls);
    return result;
}
}  // namespace detail

inline PyObject* as_py(const ContextHandle& h) noexcept {
    return detail::make_py("cuda.bindings.driver", "CUcontext", as_intptr(h));
}

inline PyObject* as_py(const GreenCtxHandle& h) noexcept {
    return detail::make_py("cuda.bindings.driver", "CUgreenCtx", as_intptr(h));
}

inline PyObject* as_py(const StreamHandle& h) noexcept {
    return detail::make_py("cuda.bindings.driver", "CUstream", as_intptr(h));
}

inline PyObject* as_py(const EventHandle& h) noexcept {
    return detail::make_py("cuda.bindings.driver", "CUevent", as_intptr(h));
}

inline PyObject* as_py(const MemoryPoolHandle& h) noexcept {
    return detail::make_py("cuda.bindings.driver", "CUmemoryPool", as_intptr(h));
}

inline PyObject* as_py(const DevicePtrHandle& h) noexcept {
    return detail::make_py("cuda.bindings.driver", "CUdeviceptr", as_intptr(h));
}

inline PyObject* as_py(const LibraryHandle& h) noexcept {
    return detail::make_py("cuda.bindings.driver", "CUlibrary", as_intptr(h));
}

inline PyObject* as_py(const CUmodule& h) noexcept {
    return detail::make_py("cuda.bindings.driver", "CUmodule", as_intptr(h));
}

inline PyObject* as_py(const KernelHandle& h) noexcept {
    return detail::make_py("cuda.bindings.driver", "CUkernel", as_intptr(h));
}

inline PyObject* as_py(const GraphHandle& h) noexcept {
    return detail::make_py("cuda.bindings.driver", "CUgraph", as_intptr(h));
}

inline PyObject* as_py(const GraphExecHandle& h) noexcept {
    return detail::make_py("cuda.bindings.driver", "CUgraphExec", as_intptr(h));
}

inline PyObject* as_py(const GraphNodeHandle& h) noexcept {
    if (!as_intptr(h)) {
        Py_RETURN_NONE;
    }
    return detail::make_py("cuda.bindings.driver", "CUgraphNode", as_intptr(h));
}

inline PyObject* as_py(const NvrtcProgramHandle& h) noexcept {
    return detail::make_py("cuda.bindings.nvrtc", "nvrtcProgram", as_intptr(h));
}

inline PyObject* as_py(const NvvmProgramHandle& h) noexcept {
    // NVVM bindings use raw integers, not wrapper classes
    return PyLong_FromSsize_t(as_intptr(h));
}

inline PyObject* as_py(const NvJitLinkHandle& h) noexcept {
    // nvJitLink bindings use raw integers, not wrapper classes
    return PyLong_FromSsize_t(as_intptr(h));
}

inline PyObject* as_py(const CuLinkHandle& h) noexcept {
    return detail::make_py("cuda.bindings.driver", "CUlinkState", as_intptr(h));
}

inline PyObject* as_py(const GraphicsResourceHandle& h) noexcept {
    return detail::make_py("cuda.bindings.driver", "CUgraphicsResource", as_intptr(h));
}

inline PyObject* as_py(const FileDescriptorHandle& h) noexcept {
    return PyLong_FromSsize_t(as_intptr(h));
}

inline PyObject* as_py(const OpaqueArrayHandle& h) noexcept {
    return detail::make_py("cuda.bindings.driver", "CUarray", as_intptr(h));
}

inline PyObject* as_py(const MipmappedArrayHandle& h) noexcept {
    return detail::make_py("cuda.bindings.driver", "CUmipmappedArray", as_intptr(h));
}

inline PyObject* as_py(const TexObjectHandle& h) noexcept {
    return detail::make_py("cuda.bindings.driver", "CUtexObject", as_intptr(h));
}

inline PyObject* as_py(const SurfObjectHandle& h) noexcept {
    return detail::make_py("cuda.bindings.driver", "CUsurfObject", as_intptr(h));
}

// ============================================================================
// Python-coupled API: the prototypes that take or return PyObject*
// ============================================================================

// Register the Python warning category used by report_* (cuda.core.CUDAWarning).
// Implemented in py_report.cpp
void register_warning_category(PyObject* category) noexcept;

// Create a non-owning stream handle that prevents a Python owner from being GC'd.
// The owner's refcount is incremented; decremented when handle is released.
// The owner is responsible for keeping the stream's context alive.
// Implemented in stream.cpp
StreamHandle create_stream_handle_with_owner(CUstream stream, PyObject* owner);

// Create a non-owning device pointer handle that prevents a Python owner from being GC'd.
// The owner's refcount is incremented; decremented when handle is released.
// The pointer will NOT be freed when the handle is released.
// If owner is nullptr, equivalent to deviceptr_create_ref.
// Implemented in memory.cpp
DevicePtrHandle deviceptr_create_with_owner(CUdeviceptr ptr, PyObject* owner);

// Callback type for MemoryResource deallocation.
// Called from the shared_ptr deleter when a handle created via
// deviceptr_create_with_mr is destroyed.  The implementation is responsible
// for converting raw C types to Python objects and calling
// mr.deallocate(ptr, size, stream).
using MRDeallocCallback = void (*)(PyObject* mr, CUdeviceptr ptr,
                                   size_t size, const StreamHandle& stream);

// Register the MR deallocation callback.
// Implemented in memory.cpp
void register_mr_dealloc_callback(MRDeallocCallback cb);

// Create a device pointer handle whose destructor calls mr.deallocate()
// via the registered callback.  The mr's refcount is incremented and
// decremented when the handle is released.
// If mr is nullptr, equivalent to deviceptr_create_ref.
// Implemented in memory.cpp
DevicePtrHandle deviceptr_create_with_mr(CUdeviceptr ptr, size_t size, PyObject* mr);

// Build an OpaqueHandle from a Python object: increments its refcount now and
// decrements it (under the GIL) on release. The caller must hold the GIL.
// Implemented in graph.cpp
OpaqueHandle make_opaque_py(PyObject* obj);

}  // namespace cuda_core::rt
