// SPDX-FileCopyrightText: Copyright (c) 2024-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
//
// SPDX-License-Identifier: Apache-2.0

#include <Python.h>

#include "resource_handles.hpp"
#include <cuda.h>
#include <atomic>
#include <array>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <functional>
#include <list>
#include <map>
#include <mutex>
#include <stdexcept>
#include <thread>
#include <type_traits>
#include <unordered_map>
#include <utility>
#include <vector>

#ifndef _WIN32
#include <unistd.h>
#endif

namespace cuda_core {

// ============================================================================
// CUDA driver function pointers
//
// These are populated by _resource_handles.pyx at module import time using
// function pointers extracted from cuda.bindings.cydriver.__pyx_capi__.
// ============================================================================

decltype(&cuGetErrorName) p_cuGetErrorName = nullptr;
decltype(&cuGetErrorString) p_cuGetErrorString = nullptr;

decltype(&cuDevicePrimaryCtxRetain) p_cuDevicePrimaryCtxRetain = nullptr;
decltype(&cuDevicePrimaryCtxRelease) p_cuDevicePrimaryCtxRelease = nullptr;
decltype(&cuCtxGetCurrent) p_cuCtxGetCurrent = nullptr;
decltype(&cuCtxSetCurrent) p_cuCtxSetCurrent = nullptr;
decltype(&cuCtxSynchronize) p_cuCtxSynchronize = nullptr;
decltype(&cuCtxGetStreamPriorityRange) p_cuCtxGetStreamPriorityRange = nullptr;
decltype(&cuCtxGetDevice) p_cuCtxGetDevice = nullptr;
decltype(&cuGraphNodeSetParams) p_cuGraphNodeSetParams = nullptr;
decltype(&cuGreenCtxCreate) p_cuGreenCtxCreate = nullptr;
decltype(&cuGreenCtxDestroy) p_cuGreenCtxDestroy = nullptr;
decltype(&cuCtxFromGreenCtx) p_cuCtxFromGreenCtx = nullptr;
decltype(&cuDevResourceGenerateDesc) p_cuDevResourceGenerateDesc = nullptr;

decltype(&cuGreenCtxStreamCreate) p_cuGreenCtxStreamCreate = nullptr;

decltype(&cuStreamCreateWithPriority) p_cuStreamCreateWithPriority = nullptr;
decltype(&cuStreamDestroy) p_cuStreamDestroy = nullptr;
decltype(&cuStreamGetCtx) p_cuStreamGetCtx = nullptr;

decltype(&cuEventCreate) p_cuEventCreate = nullptr;
decltype(&cuEventDestroy) p_cuEventDestroy = nullptr;
decltype(&cuIpcOpenEventHandle) p_cuIpcOpenEventHandle = nullptr;

decltype(&cuDeviceGetCount) p_cuDeviceGetCount = nullptr;

decltype(&cuMemPoolSetAccess) p_cuMemPoolSetAccess = nullptr;
decltype(&cuMemPoolDestroy) p_cuMemPoolDestroy = nullptr;
decltype(&cuMemPoolCreate) p_cuMemPoolCreate = nullptr;
decltype(&cuDeviceGetMemPool) p_cuDeviceGetMemPool = nullptr;
decltype(&cuMemPoolImportFromShareableHandle) p_cuMemPoolImportFromShareableHandle = nullptr;

decltype(&cuMemAllocFromPoolAsync) p_cuMemAllocFromPoolAsync = nullptr;
decltype(&cuMemAllocAsync) p_cuMemAllocAsync = nullptr;
decltype(&cuMemAlloc) p_cuMemAlloc = nullptr;
decltype(&cuMemAllocHost) p_cuMemAllocHost = nullptr;

decltype(&cuMemFreeAsync) p_cuMemFreeAsync = nullptr;
decltype(&cuMemFree) p_cuMemFree = nullptr;
decltype(&cuMemFreeHost) p_cuMemFreeHost = nullptr;

decltype(&cuMemPoolImportPointer) p_cuMemPoolImportPointer = nullptr;

decltype(&cuLibraryLoadFromFile) p_cuLibraryLoadFromFile = nullptr;
decltype(&cuLibraryLoadData) p_cuLibraryLoadData = nullptr;
decltype(&cuLibraryUnload) p_cuLibraryUnload = nullptr;
decltype(&cuLibraryGetKernel) p_cuLibraryGetKernel = nullptr;

// Graph
decltype(&cuGraphDestroy) p_cuGraphDestroy = nullptr;
decltype(&cuGraphInstantiateWithParams) p_cuGraphInstantiateWithParams = nullptr;
decltype(&cuGraphExecUpdate) p_cuGraphExecUpdate = nullptr;
decltype(&cuGraphExecDestroy) p_cuGraphExecDestroy = nullptr;
decltype(&cuUserObjectCreate) p_cuUserObjectCreate = nullptr;
decltype(&cuUserObjectRelease) p_cuUserObjectRelease = nullptr;
decltype(&cuGraphRetainUserObject) p_cuGraphRetainUserObject = nullptr;
decltype(&cuGraphReleaseUserObject) p_cuGraphReleaseUserObject = nullptr;
decltype(&cuGraphNodeFindInClone) p_cuGraphNodeFindInClone = nullptr;
decltype(&cuGraphChildGraphNodeGetGraph) p_cuGraphChildGraphNodeGetGraph = nullptr;

// Linker
decltype(&cuLinkDestroy) p_cuLinkDestroy = nullptr;

// GL interop pointers
decltype(&cuGraphicsUnmapResources) p_cuGraphicsUnmapResources = nullptr;
decltype(&cuGraphicsUnregisterResource) p_cuGraphicsUnregisterResource = nullptr;

decltype(&cuArray3DCreate) p_cuArray3DCreate = nullptr;
decltype(&cuArrayDestroy) p_cuArrayDestroy = nullptr;
decltype(&cuMipmappedArrayCreate) p_cuMipmappedArrayCreate = nullptr;
decltype(&cuMipmappedArrayDestroy) p_cuMipmappedArrayDestroy = nullptr;
decltype(&cuMipmappedArrayGetLevel) p_cuMipmappedArrayGetLevel = nullptr;
decltype(&cuTexObjectCreate) p_cuTexObjectCreate = nullptr;
decltype(&cuTexObjectDestroy) p_cuTexObjectDestroy = nullptr;
decltype(&cuSurfObjectCreate) p_cuSurfObjectCreate = nullptr;
decltype(&cuSurfObjectDestroy) p_cuSurfObjectDestroy = nullptr;

// SM resource split (13.1+ — may be null on older drivers/bindings)
#if CUDA_VERSION >= 13010
decltype(&cuDevSmResourceSplit) p_cuDevSmResourceSplit = nullptr;
#else
void* p_cuDevSmResourceSplit = nullptr;
#endif

// cuMemcpyWithAttributesAsync (13.2+ — may be null on older drivers/bindings)
#if CUDA_VERSION >= 13020
decltype(&cuMemcpyWithAttributesAsync) p_cuMemcpyWithAttributesAsync = nullptr;
#else
void* p_cuMemcpyWithAttributesAsync = nullptr;
#endif

// NVRTC function pointers
decltype(&nvrtcDestroyProgram) p_nvrtcDestroyProgram = nullptr;

// NVVM function pointers (may be null if NVVM is not available)
NvvmDestroyProgramFn p_nvvmDestroyProgram = nullptr;

// nvJitLink function pointers (may be null if nvJitLink is not available)
NvJitLinkDestroyFn p_nvJitLinkDestroy = nullptr;

// ============================================================================
// GIL and scoped-context management helpers
// ============================================================================

namespace {

// Conditionally release the GIL while calling into the CUDA driver.
class GILReleaseGuard {
public:
    GILReleaseGuard() noexcept {
        if (!Py_IsInitialized() || py_is_finalizing()) {
            return;
        }
        if (PyGILState_Check()) {
            tstate_ = PyEval_SaveThread();
        }
    }

    ~GILReleaseGuard() {
        if (tstate_) {
            PyEval_RestoreThread(tstate_);
        }
    }

    GILReleaseGuard(const GILReleaseGuard&) = delete;
    GILReleaseGuard& operator=(const GILReleaseGuard&) = delete;

private:
    PyThreadState* tstate_ = nullptr;
};

// Helper to acquire the GIL when we might not hold it.
// Use in C++ destructors that need to manipulate Python objects.
class GILAcquireGuard {
public:
    GILAcquireGuard() : acquired_(false) {
        // Don't try to acquire GIL if Python is finalizing
        if (!Py_IsInitialized() || py_is_finalizing()) {
            return;
        }
        gstate_ = PyGILState_Ensure();
        acquired_ = true;
    }

    ~GILAcquireGuard() {
        if (acquired_) {
            PyGILState_Release(gstate_);
        }
    }

    bool acquired() const { return acquired_; }

    // Non-copyable, non-movable
    GILAcquireGuard(const GILAcquireGuard&) = delete;
    GILAcquireGuard& operator=(const GILAcquireGuard&) = delete;

private:
    PyGILState_STATE gstate_;
    bool acquired_;
};

// ----------------------------------------------------------------------------
// Non-propagating error reporting
//
// Deleters, CUDA callbacks and other non-propagating paths cannot raise. They
// report through report_cuda_error()/report_message(), which emit a
// cuda.core.CUDAWarning when the interpreter is usable and fall back to stderr
// otherwise. See docs/source/error_handling.rst for the policy.
// ----------------------------------------------------------------------------

// Warning category registered by _resource_handles.pyx (cuda.core.CUDAWarning).
std::atomic<PyObject*> warning_category{nullptr};

// Thread-local detail attached to the next raised CUDAError with a matching
// status (see take_last_error_detail()). Written only by propagating helpers.
// The taken copy stays valid until the next take on the same thread.
thread_local char last_error_detail[512] = {0};
thread_local char taken_error_detail[512] = {0};
thread_local CUresult last_error_detail_status = CUDA_SUCCESS;

// Thread-local fault injected into the next context restoration (tests only).
thread_local CUresult context_restore_fault = CUDA_SUCCESS;

// Format "<operation> <detail>: <NAME>: <description>" for a failed CUDA call.
void format_cuda_error(char* buffer, size_t size, const char* operation, CUresult status,
                       const char* detail) noexcept {
    const char* error_name = nullptr;
    const char* error_description = nullptr;
    bool decoded = p_cuGetErrorName && p_cuGetErrorString
                   && p_cuGetErrorName(status, &error_name) == CUDA_SUCCESS
                   && p_cuGetErrorString(status, &error_description) == CUDA_SUCCESS;
    const char* outcome = detail ? detail : "failed";
    if (decoded) {
        std::snprintf(buffer, size, "%s %s: %s: %s", operation, outcome, error_name, error_description);
    } else {
        std::snprintf(buffer, size, "%s %s (CUDA error %d)", operation, outcome, static_cast<int>(status));
    }
}

}  // namespace

// Report a message that could not be raised. Emits cuda.core.CUDAWarning via
// the Python warnings machinery; if that itself fails (for example because the
// warning was promoted to an error), the failure is written as an unraisable
// exception, the CPython convention for exceptions in destructors. Falls back
// to stderr when the interpreter cannot be used.
void report_message(const char* message) noexcept {
    PyObject* category = warning_category.load(std::memory_order_acquire);
    if (category && Py_IsInitialized() && !py_is_finalizing()) {
        GILAcquireGuard gil;
        if (gil.acquired()) {
            // Deleters can run while a Python exception is propagating; keep it.
#if PY_VERSION_HEX >= 0x030C0000
            PyObject* pending = PyErr_GetRaisedException();
#else
            PyObject *pending_type, *pending_value, *pending_tb;
            PyErr_Fetch(&pending_type, &pending_value, &pending_tb);
#endif
            if (PyErr_WarnEx(category, message, 1) != 0) {
                PyObject* subject = PyUnicode_FromString(message);
                PyErr_WriteUnraisable(subject);
                Py_XDECREF(subject);
            }
#if PY_VERSION_HEX >= 0x030C0000
            PyErr_SetRaisedException(pending);
#else
            PyErr_Restore(pending_type, pending_value, pending_tb);
#endif
            return;
        }
    }
    std::fprintf(stderr, "%s\n", message);
}

// Report a failed non-CUDA call (NVRTC, NVVM, nvJitLink) from a path that
// cannot raise.
void report_status_code(const char* operation, long code) noexcept {
    char message[256];
    std::snprintf(message, sizeof(message), "%s failed (status %ld)", operation, code);
    report_message(message);
}

void register_warning_category(PyObject* category) noexcept {
    warning_category.store(category, std::memory_order_release);
}

// Report a failed CUDA call from a path that cannot raise. CUDA_ERROR_DEINITIALIZED
// is not reported: it means the driver is shutting down, which makes cleanup
// failures expected and uninteresting.
void report_cuda_error(const char* operation, CUresult status, const char* detail) noexcept {
    if (status == CUDA_SUCCESS || status == CUDA_ERROR_DEINITIALIZED) {
        return;
    }
    char message[512];
    format_cuda_error(message, sizeof(message), operation, status, detail);
    report_message(message);
}

namespace {

// Attach `message` as a PEP 678 note to the exception currently being handled.
// Returns false when there is none or the interpreter cannot be used.
bool add_note_to_handled_exception(const char* message) noexcept {
#if PY_VERSION_HEX >= 0x030B0000
    if (!Py_IsInitialized() || py_is_finalizing()) {
        return false;
    }
    GILAcquireGuard gil;
    if (!gil.acquired()) {
        return false;
    }
    PyObject* exc = PyErr_GetHandledException();
    if (!exc) {
        return false;
    }
    PyObject* result = PyObject_CallMethod(exc, "add_note", "s", message);
    Py_DECREF(exc);
    if (!result) {
        PyErr_Clear();
        return false;
    }
    Py_DECREF(result);
    return true;
#else
    (void)message;
    return false;
#endif
}

}  // namespace

void note_or_report_cuda_error(const char* operation, CUresult status, const char* detail) noexcept {
    if (status == CUDA_SUCCESS || status == CUDA_ERROR_DEINITIALIZED) {
        return;
    }
    char message[512];
    format_cuda_error(message, sizeof(message), operation, status, detail);
    if (!add_note_to_handled_exception(message)) {
        report_message(message);
    }
}

const char* take_last_error_detail(CUresult status) noexcept {
    if (!last_error_detail[0] || status != last_error_detail_status) {
        return nullptr;
    }
    std::memcpy(taken_error_detail, last_error_detail, sizeof(taken_error_detail));
    clear_last_error_detail();
    return taken_error_detail;
}

void clear_last_error_detail() noexcept {
    last_error_detail[0] = 0;
    last_error_detail_status = CUDA_SUCCESS;
}

void set_context_restore_fault_for_testing(CUresult status) noexcept {
    context_restore_fault = status;
}

namespace {

// Make a context current and record the state needed to restore it.
// An empty handle is a no-op: the operation runs in the caller's current
// context, and nothing is restored on exit.
CUresult enter_context(const ContextHandle& h_context, CUcontext* previous, int* changed) noexcept {
    *previous = nullptr;
    *changed = 0;
    clear_last_error_detail();
    CUcontext target = as_cu(h_context);
    if (!target) {
        return CUDA_SUCCESS;
    }

    GILReleaseGuard gil;
    CUresult status = p_cuCtxGetCurrent(previous);
    if (status != CUDA_SUCCESS || *previous == target) {
        return status;
    }
    status = p_cuCtxSetCurrent(target);
    *changed = status == CUDA_SUCCESS;
    return status;
}

// Restore the caller's context. Returns the restoration status.
CUresult restore_context(CUcontext previous) noexcept {
    if (context_restore_fault != CUDA_SUCCESS) {
        // Test hook: behave as if cuCtxSetCurrent(previous) failed, leaving the
        // target context current exactly as a real failure would.
        CUresult fault = context_restore_fault;
        context_restore_fault = CUDA_SUCCESS;
        return fault;
    }
    GILReleaseGuard gil;
    return p_cuCtxSetCurrent(previous);
}

// Record that the caller's context was not restored as the detail of the
// CUresult about to be returned and raised: the operation status if the
// operation failed too, else the restoration status. For a double failure the
// detail also names the restoration error, which the raised error does not.
void note_context_not_restored(CUcontext previous, CUresult operation_status,
                               CUresult restore_status) noexcept {
    CUcontext current = nullptr;
    if (p_cuCtxGetCurrent(&current) != CUDA_SUCCESS) {
        current = nullptr;
    }
    char cause[128] = {0};
    if (operation_status != CUDA_SUCCESS) {
        const char* error_name = nullptr;
        if (p_cuGetErrorName && p_cuGetErrorName(restore_status, &error_name) == CUDA_SUCCESS) {
            std::snprintf(cause, sizeof(cause), " after this failure (cuCtxSetCurrent: %s)", error_name);
        } else {
            std::snprintf(cause, sizeof(cause), " after this failure (cuCtxSetCurrent: CUDA error %d)",
                          static_cast<int>(restore_status));
        }
    }
    std::snprintf(last_error_detail, sizeof(last_error_detail),
                  "the calling thread's CUDA context (%#llx) could not be restored%s; "
                  "context %#llx is now current. Call Device.set_current() before issuing "
                  "further CUDA work on this thread",
                  static_cast<unsigned long long>(reinterpret_cast<uintptr_t>(previous)),
                  cause,
                  static_cast<unsigned long long>(reinterpret_cast<uintptr_t>(current)));
    last_error_detail_status = operation_status != CUDA_SUCCESS ? operation_status : restore_status;
}

// Restore the previous context and preserve an earlier operation error. The
// operation error, if any, is returned; otherwise the restoration status is.
// Either way a restoration failure is recorded as the detail of the returned
// status, so the eventual CUDAError explains it (see take_last_error_detail()).
CUresult exit_context(CUcontext previous, int changed, CUresult operation_status) noexcept {
    CUresult restore_status = changed ? restore_context(previous) : CUDA_SUCCESS;
    if (restore_status == CUDA_SUCCESS) {
        return operation_status;
    }
    note_context_not_restored(previous, operation_status, restore_status);
    return operation_status != CUDA_SUCCESS ? operation_status : restore_status;
}

// Require a callable to be invocable without throwing.
#define ASSERT_NOTHROW_INVOCABLE(...) \
    static_assert(std::is_nothrow_invocable_v<__VA_ARGS__>, "operation must be noexcept")

// Store a stream and any state needed to preserve deallocation ordering.
struct DeallocationStream {
    StreamHandle h_stream;
    std::thread::id ptds_tid{};
};

// Return whether a stream handle needs a current context to resolve it.
bool is_default_stream(CUstream stream) noexcept {
    return stream == nullptr || stream == CU_STREAM_LEGACY || stream == CU_STREAM_PER_THREAD;
}

// Return the context a deallocation-stream token must run under. Real streams
// resolve their own context; default-stream tokens use the context bound at
// allocation time. Warn when PTDS deallocation crosses host threads.
ContextHandle deallocation_context(const DeallocationStream& stream) noexcept {
    if (!is_default_stream(as_cu(stream.h_stream))) {
        return {};
    }
    if (stream.ptds_tid != std::thread::id{}
            && stream.ptds_tid != std::this_thread::get_id()) {
        report_message(
            "Buffer deallocation for a per-thread default stream "
            "is running on a different host thread than the one that recorded "
            "the deallocation stream; ordering relative to the allocating "
            "thread's PTDS is not preserved");
    }
    return get_stream_context(stream.h_stream);
}

// Run an operation with the requested context current.
template <typename Fn, typename... Args>
CUresult invoke_in_context(const ContextHandle& h_context, Fn&& operation, Args&&... args) noexcept {
    ASSERT_NOTHROW_INVOCABLE(Fn&&, Args&&...);
    if (!h_context) {
        return CUDA_ERROR_INVALID_CONTEXT;
    }
    CUcontext previous = nullptr;
    int changed = 0;
    CUresult status = enter_context(h_context, &previous, &changed);
    if (status == CUDA_SUCCESS) {
        status = std::invoke(std::forward<Fn>(operation), std::forward<Args>(args)...);
    }
    return exit_context(previous, changed, status);
}

// Run a creation operation and undo it if context restoration fails.
// Context-independent undo always runs. Context-sensitive undo runs only
// after verifying that the target context remains current; otherwise the
// resource leaks rather than risking cleanup in the wrong context.
template <typename Fn, typename Undo>
CUresult invoke_in_context_or_undo(const ContextHandle& h_context, Fn&& operation,
                                   Undo&& undo, bool undo_requires_target_context) noexcept {
    ASSERT_NOTHROW_INVOCABLE(Fn&&);
    ASSERT_NOTHROW_INVOCABLE(Undo&&);
    if (!h_context) {
        return CUDA_ERROR_INVALID_CONTEXT;
    }
    CUcontext previous = nullptr;
    int changed = 0;
    CUresult status = enter_context(h_context, &previous, &changed);
    if (status != CUDA_SUCCESS) {
        return status;
    }
    status = std::invoke(std::forward<Fn>(operation));
    CUresult composite = exit_context(previous, changed, status);
    if (status == CUDA_SUCCESS && composite != CUDA_SUCCESS) {
        bool undo_ok = true;
        if (undo_requires_target_context) {
            CUcontext current = nullptr;
            undo_ok = p_cuCtxGetCurrent(&current) == CUDA_SUCCESS
                      && current == as_cu(h_context);
        }
        if (undo_ok) {
            std::invoke(std::forward<Undo>(undo));
        } else {
            report_cuda_error(
                "cuCtxSetCurrent (restoring the caller's context)", composite,
                "failed; cleanup of the new resource skipped because its context "
                "is no longer current (resource leaked)");
        }
    }
    return composite;
}

// Run cleanup with the requested context current. Warn and skip the operation
// if activation fails, and independently warn on operation or restoration
// failure. Return the operation or activation status; restoration never
// changes the return value.
template <typename Fn, typename... Args>
CUresult cleanup_in_context(const ContextHandle& h_context, const char* name,
                            Fn&& operation, Args&&... args) noexcept {
    ASSERT_NOTHROW_INVOCABLE(Fn&&, Args&&...);
    CUcontext previous = nullptr;
    int changed = 0;
    CUresult status = enter_context(h_context, &previous, &changed);
    if (status != CUDA_SUCCESS) {
        report_cuda_error(name, status,
                           "skipped (context activation failed; resource leaked)");
    } else {
        status = std::invoke(std::forward<Fn>(operation), std::forward<Args>(args)...);
        if (status != CUDA_SUCCESS) {
            report_cuda_error(name, status);
        }
    }
    CUresult restore = exit_context(previous, changed, CUDA_SUCCESS);
    if (restore != CUDA_SUCCESS) {
        // Nothing is raised here, so the detail exit_context recorded has no
        // exception to attach to: report it and drop the detail.
        report_cuda_error(name, restore, "failed while restoring the caller's context");
        clear_last_error_detail();
    }
    return status;
}

#undef ASSERT_NOTHROW_INVOCABLE

// Decorate a status-returning cleanup call to report whenever it fails. CUDA
// calls (CUresult) are reported with the error name and description; NVRTC,
// NVVM and nvJitLink calls (integer status codes) with the raw code.
template <auto& Function>
class WarnOnFailure {
public:
    explicit WarnOnFailure(const char* operation) noexcept : operation_(operation) {}

    template <typename... Args>
    auto operator()(Args&&... args) const noexcept {
        auto status = Function(std::forward<Args>(args)...);
        report(status);
        return status;
    }

private:
    void report(CUresult status) const noexcept {
        report_cuda_error(operation_, status);
    }

    template <typename Status>
    void report(Status status) const noexcept {
        if (static_cast<long>(status) != 0) {
            report_status_code(operation_, static_cast<long>(status));
        }
    }

    const char* operation_;
};

// Warning-decorated CUDA operations used by non-throwing cleanup paths.
const WarnOnFailure<p_cuStreamDestroy> pw_cuStreamDestroy{"cuStreamDestroy"};
const WarnOnFailure<p_cuEventDestroy> pw_cuEventDestroy{"cuEventDestroy"};
const WarnOnFailure<p_cuMemFree> pw_cuMemFree{"cuMemFree"};
const WarnOnFailure<p_cuMemFreeAsync> pw_cuMemFreeAsync{"cuMemFreeAsync"};
const WarnOnFailure<p_cuArrayDestroy> pw_cuArrayDestroy{"cuArrayDestroy"};
const WarnOnFailure<p_cuMipmappedArrayDestroy> pw_cuMipmappedArrayDestroy{"cuMipmappedArrayDestroy"};
const WarnOnFailure<p_cuTexObjectDestroy> pw_cuTexObjectDestroy{"cuTexObjectDestroy"};
const WarnOnFailure<p_cuSurfObjectDestroy> pw_cuSurfObjectDestroy{"cuSurfObjectDestroy"};
const WarnOnFailure<p_cuGreenCtxDestroy> pw_cuGreenCtxDestroy{"cuGreenCtxDestroy"};
const WarnOnFailure<p_cuMemPoolDestroy> pw_cuMemPoolDestroy{"cuMemPoolDestroy"};
const WarnOnFailure<p_cuMemFreeHost> pw_cuMemFreeHost{"cuMemFreeHost"};
const WarnOnFailure<p_cuGraphDestroy> pw_cuGraphDestroy{"cuGraphDestroy"};
const WarnOnFailure<p_cuGraphExecDestroy> pw_cuGraphExecDestroy{"cuGraphExecDestroy"};
const WarnOnFailure<p_cuGraphicsUnregisterResource> pw_cuGraphicsUnregisterResource{"cuGraphicsUnregisterResource"};
const WarnOnFailure<p_cuLinkDestroy> pw_cuLinkDestroy{"cuLinkDestroy"};
const WarnOnFailure<p_cuUserObjectRelease> pw_cuUserObjectRelease{"cuUserObjectRelease"};
const WarnOnFailure<p_cuGraphReleaseUserObject> pw_cuGraphReleaseUserObject{"cuGraphReleaseUserObject"};
const WarnOnFailure<p_nvrtcDestroyProgram> pw_nvrtcDestroyProgram{"nvrtcDestroyProgram"};
const WarnOnFailure<p_nvvmDestroyProgram> pw_nvvmDestroyProgram{"nvvmDestroyProgram"};
const WarnOnFailure<p_nvJitLinkDestroy> pw_nvJitLinkDestroy{"nvJitLinkDestroy"};

}  // namespace

// Synchronize the provided context.
CUresult context_synchronize(const ContextHandle& h_context) noexcept {
    GILReleaseGuard gil;
    return invoke_in_context(h_context, []() noexcept {
        return p_cuCtxSynchronize();
    });
}

// Query the stream priority range for the provided context.
CUresult context_get_stream_priority_range(const ContextHandle& h_context,
                                           int* least_priority,
                                           int* greatest_priority) noexcept {
    GILReleaseGuard gil;
    return invoke_in_context(h_context, [&]() noexcept {
        return p_cuCtxGetStreamPriorityRange(least_priority, greatest_priority);
    });
}

// Query the device of the provided context.
CUresult context_get_device(const ContextHandle& h_context, CUdevice* device) noexcept {
    return invoke_in_context(h_context, [&]() noexcept {
        return p_cuCtxGetDevice(device);
    });
}

// Set a graph node's parameters with h_context current (an empty handle runs in
// the caller's context). Returns the cuGraphNodeSetParams status. A failure to
// restore the caller's context is returned separately in *restore_status so the
// caller can publish the metadata that depends on the successful update before
// raising it; if the update itself failed, its status is returned with the
// restoration failure recorded as its detail and *restore_status is CUDA_SUCCESS.
CUresult graph_node_set_params(CUgraphNode node, CUgraphNodeParams* params,
                               const ContextHandle& h_context,
                               CUresult* restore_status) noexcept {
    *restore_status = CUDA_SUCCESS;
    if (!p_cuGraphNodeSetParams) {
        return CUDA_ERROR_NOT_SUPPORTED;
    }
    CUcontext previous = nullptr;
    int changed = 0;
    CUresult status = enter_context(h_context, &previous, &changed);
    if (status != CUDA_SUCCESS) {
        return status;
    }
    {
        GILReleaseGuard gil;
        status = p_cuGraphNodeSetParams(node, params);
    }
    if (!changed) {
        return status;
    }
    CUresult restored = restore_context(previous);
    if (restored == CUDA_SUCCESS) {
        return status;
    }
    note_context_not_restored(previous, status, restored);
    if (status == CUDA_SUCCESS) {
        *restore_status = restored;
    }
    return status;
}

// ============================================================================
// CUDA user-object deferred cleanup
//
// CUDA invokes a user-object destructor on an internal thread where CUDA
// calls are forbidden. Payload cleanup can release resource handles whose
// deleters call CUDA, so the callback only transfers a preallocated intrusive
// node to this process-lifetime queue. One coalesced pending call drains all
// queued payloads from Python's main thread.
// ============================================================================

// Intrusive base for payloads transferred out of CUDA's callback.
struct DeferredCleanupItem {
    DeferredCleanupItem* next = nullptr;
    virtual ~DeferredCleanupItem() noexcept = default;
};

namespace {

// Process-lifetime MPSC queue that drains payloads from Python's main thread.
class DeferredCleanupQueue {
public:
    // Transfer one preallocated cleanup item from a producer to the queue.
    void enqueue(DeferredCleanupItem* item) noexcept {
        DeferredCleanupItem* head = head_.load(std::memory_order_relaxed);
        do {
            item->next = head;
        } while (!head_.compare_exchange_weak(
            head, item, std::memory_order_release, std::memory_order_relaxed));
        schedule();
    }

    // Permanently disable pending-call scheduling during interpreter shutdown.
    void stop() noexcept {
        accepting_.store(false, std::memory_order_release);
    }

    // Reattempt scheduling after Py_AddPendingCall() found CPython's bounded
    // pending-call queue full and left payloads queued for a later safe entry.
    void retry_schedule() noexcept {
        schedule();
    }

private:
    // Adapt queue draining to CPython's int (*)(void*) callback ABI.
    static int pending_call(void* arg) noexcept {
        static_cast<DeferredCleanupQueue*>(arg)->drain();
        return 0;
    }

    // Coalesce all queued work behind at most one CPython pending call.
    void schedule() noexcept {
        if (!accepting_.load(std::memory_order_acquire)) {
            return;
        }
        if (!Py_IsInitialized() || py_is_finalizing()) {
            stop();
            return;
        }
        if (!head_.load(std::memory_order_acquire)) {
            return;
        }
        bool expected = false;
        if (!scheduled_.compare_exchange_strong(
                expected, true, std::memory_order_acq_rel,
                std::memory_order_relaxed)) {
            return;
        }
        if (Py_AddPendingCall(&DeferredCleanupQueue::pending_call, this) != 0) {
            // Keep every payload queued. A later enqueue or safe cuda-core
            // entry can retry without blocking CUDA's callback thread.
            scheduled_.store(false, std::memory_order_release);
        }
    }

    // Detach and destroy all queued payloads from Python's main thread.
    void drain() noexcept {
        if (!Py_IsInitialized() || py_is_finalizing()) {
            stop();
            scheduled_.store(false, std::memory_order_release);
            return;  // Intentionally leak intact payloads during shutdown.
        }

        while (DeferredCleanupItem* list =
                   head_.exchange(nullptr, std::memory_order_acquire)) {
            while (list) {
                DeferredCleanupItem* next = list->next;
                delete list;
                list = next;
            }
        }

        scheduled_.store(false, std::memory_order_release);
        if (head_.load(std::memory_order_acquire)) {
            schedule();
        }
    }

    // Head of the intrusive multi-producer, single-consumer payload stack.
    std::atomic<DeferredCleanupItem*> head_{nullptr};
    // True while one cuda-core drain callback is pending or executing.
    std::atomic<bool> scheduled_{false};
    // False once shutdown begins, causing later payloads to be leaked safely.
    std::atomic<bool> accepting_{true};
};

// Published once at module initialization and intentionally never freed.
std::atomic<DeferredCleanupQueue*> deferred_cleanup_queue{nullptr};

void ensure_deferred_cleanup_ready() {
    DeferredCleanupQueue* queue =
        deferred_cleanup_queue.load(std::memory_order_acquire);
    if (!queue) {
        throw std::runtime_error("deferred cleanup is not initialized");
    }
    queue->retry_schedule();
}

// CUDA's CUhostFn ABI is void (*)(void*); recover and enqueue the cleanup item.
void enqueue_cleanup(void* item) noexcept {
    auto* cleanup = static_cast<DeferredCleanupItem*>(item);
    if (DeferredCleanupQueue* queue =
            deferred_cleanup_queue.load(std::memory_order_acquire)) {
        queue->enqueue(cleanup);
    }
}

}  // namespace

// Module initialization calls this once with the GIL held, which serializes
// the check, allocation, and publication below.
void initialize_deferred_cleanup() {
    if (deferred_cleanup_queue.load(std::memory_order_acquire)) {
        return;
    }
    auto* queue = new DeferredCleanupQueue();
    deferred_cleanup_queue.store(queue, std::memory_order_release);
}

void retry_deferred_cleanup() noexcept {
    if (!Py_IsInitialized() || py_is_finalizing()) {
        return;
    }
    if (DeferredCleanupQueue* queue =
            deferred_cleanup_queue.load(std::memory_order_acquire)) {
        queue->retry_schedule();
    }
}

// ============================================================================
// Handle reverse-lookup registry
//
// Maps raw CUDA handles (CUevent, CUkernel, etc.) back to their owning
// shared_ptr so that _ref constructors can recover full metadata.
// Uses weak_ptr to avoid preventing destruction.
// ============================================================================

template<typename Key, typename Handle, typename Hash = std::hash<Key>>
class HandleRegistry {
public:
    using MapType = std::unordered_map<Key, std::weak_ptr<typename Handle::element_type>, Hash>;

    void register_handle(const Key& key, const Handle& h) {
        std::lock_guard<std::mutex> lock(mutex_);
        map_[key] = h;
    }

    void unregister_handle(const Key& key) noexcept {
        std::lock_guard<std::mutex> lock(mutex_);
        map_.erase(key);
    }

    void register_handles(const std::vector<Handle>& handles) {
        std::lock_guard<std::mutex> lock(mutex_);
        for (const Handle& h : handles) {
            if (h) {
                map_[*h] = h;
            }
        }
    }

    Handle lookup(const Key& key) {
        std::lock_guard<std::mutex> lock(mutex_);
        auto it = map_.find(key);
        if (it != map_.end()) {
            if (auto h = it->second.lock()) {
                return h;
            }
            map_.erase(it);
        }
        return {};
    }

    template<typename Factory>
    Handle get_or_create(const Key& key, Factory&& create) {
        std::lock_guard<std::mutex> lock(mutex_);
        auto it = map_.find(key);
        if (it != map_.end()) {
            if (Handle h = it->second.lock()) {
                return h;
            }
            map_.erase(it);
        }

        Handle h = create();
        if (h) {
            map_[key] = h;
        }
        return h;
    }

    MapType drain() noexcept {
        std::lock_guard<std::mutex> lock(mutex_);
        MapType extracted;
        extracted.swap(map_);
        return extracted;
    }

private:
    std::mutex mutex_;
    MapType map_;
};

// ============================================================================
// Thread-local error handling
// ============================================================================

// Thread-local status of the most recent CUDA API call in this module.
static thread_local CUresult err = CUDA_SUCCESS;

// Return and clear the calling thread's most recent CUDA error.
CUresult get_last_error() noexcept {
    CUresult e = err;
    err = CUDA_SUCCESS;
    return e;
}

// Return the calling thread's most recent CUDA error without clearing it.
CUresult peek_last_error() noexcept {
    return err;
}

void clear_last_error() noexcept {
    err = CUDA_SUCCESS;
}

// ============================================================================
// Context Handles
// ============================================================================

namespace {
struct ContextBox {
    CUcontext resource;
    GreenCtxHandle h_green_ctx;
};

struct GreenCtxBox {
    CUgreenCtx resource;
};

static const ContextBox* get_box(const ContextHandle& h) noexcept {
    const CUcontext* p = h.get();
    return reinterpret_cast<const ContextBox*>(
        reinterpret_cast<const char*>(p) - offsetof(ContextBox, resource)
    );
}

// See REGISTRY_DESIGN.md (Level 1: Driver Handle -> Resource Handle)
static HandleRegistry<CUcontext, ContextHandle> context_registry;

// Create a context handle reference, with optional green context as source.
ContextHandle create_context_handle_ref(CUcontext ctx, GreenCtxHandle h_green_ctx) {
    if (!ctx) {
        return {};
    }
    if (auto h = context_registry.lookup(ctx)) {
        return h;
    }
    auto box = std::shared_ptr<const ContextBox>(
        new ContextBox{ctx, std::move(h_green_ctx)},
        [](const ContextBox* b) {
            context_registry.unregister_handle(b->resource);
            delete b;
        }
    );
    ContextHandle h(box, &box->resource);
    context_registry.register_handle(ctx, h);
    return h;
}
}  // namespace

ContextHandle create_context_handle_ref(CUcontext ctx) {
    return create_context_handle_ref(ctx, {});
}

ContextHandle create_context_handle_from_green_ctx(const GreenCtxHandle& h_green_ctx) {
    GILReleaseGuard gil;
    if (!h_green_ctx) {
        return {};
    }
    if (!p_cuCtxFromGreenCtx) {
        err = CUDA_ERROR_NOT_SUPPORTED;
        return {};
    }

    CUcontext ctx = nullptr;
    if (CUDA_SUCCESS != (err = p_cuCtxFromGreenCtx(&ctx, as_cu(h_green_ctx)))) {
        return {};
    }

    return create_context_handle_ref(ctx, h_green_ctx);
}

GreenCtxHandle get_context_green_ctx(const ContextHandle& h) noexcept {
    if (!h) {
        return {};
    }
    return get_box(h)->h_green_ctx;
}

GreenCtxHandle create_green_ctx_handle(CUdevResource* resources, unsigned int nbResources,
                                       CUdevice dev, unsigned int flags) {
    GILReleaseGuard gil;
    if (!p_cuDevResourceGenerateDesc || !p_cuGreenCtxCreate || !p_cuGreenCtxDestroy) {
        err = CUDA_ERROR_NOT_SUPPORTED;
        return {};
    }

    CUdevResourceDesc desc = nullptr;
    if (CUDA_SUCCESS != (err = p_cuDevResourceGenerateDesc(&desc, resources, nbResources))) {
        return {};
    }

    CUgreenCtx green_ctx = nullptr;
    if (CUDA_SUCCESS != (err = p_cuGreenCtxCreate(&green_ctx, desc, dev, flags))) {
        return {};
    }

    auto box = std::shared_ptr<const GreenCtxBox>(
        new GreenCtxBox{green_ctx},
        [](const GreenCtxBox* b) {
            GILReleaseGuard gil;
            pw_cuGreenCtxDestroy(b->resource);
            delete b;
        }
    );
    return GreenCtxHandle(box, &box->resource);
}

GreenCtxHandle create_green_ctx_handle_ref(CUgreenCtx green_ctx) {
    if (!green_ctx) {
        return {};
    }
    auto box = std::make_shared<const GreenCtxBox>(GreenCtxBox{green_ctx});
    return GreenCtxHandle(box, &box->resource);
}

// Thread-local cache of primary contexts indexed by device ID
static thread_local std::vector<ContextHandle> primary_context_cache;

ContextHandle get_primary_context(int device_id) {
    // Check thread-local cache
    if (static_cast<size_t>(device_id) < primary_context_cache.size()) {
        if (auto cached = primary_context_cache[device_id]) {
            return cached;
        }
    }

    // Cache miss - acquire primary context from driver
    GILReleaseGuard gil;
    CUcontext ctx;
    if (CUDA_SUCCESS != (err = p_cuDevicePrimaryCtxRetain(&ctx, device_id))) {
        return {};
    }

    auto box = std::shared_ptr<const ContextBox>(
        new ContextBox{ctx, {}},
        [device_id](const ContextBox* b) {
            context_registry.unregister_handle(b->resource);
            GILReleaseGuard gil;
            p_cuDevicePrimaryCtxRelease(device_id);
            delete b;
        }
    );
    auto h = ContextHandle(box, &box->resource);
    context_registry.register_handle(ctx, h);

    // Update cache
    if (static_cast<size_t>(device_id) >= primary_context_cache.size()) {
        primary_context_cache.resize(device_id + 1);
    }
    primary_context_cache[device_id] = h;
    return h;
}

ContextHandle get_current_context() {
    GILReleaseGuard gil;
    CUcontext ctx = nullptr;
    if (CUDA_SUCCESS != (err = p_cuCtxGetCurrent(&ctx))) {
        return {};
    }
    if (!ctx) {
        return {};  // No current context (not an error)
    }
    return create_context_handle_ref(ctx);
}

// ============================================================================
// Stream Handles
// ============================================================================

namespace {
struct StreamBox {
    CUstream resource;
    ContextHandle h_context;
};

static const StreamBox* get_box(const StreamHandle& h) noexcept {
    const CUstream* p = h.get();
    return reinterpret_cast<const StreamBox*>(
        reinterpret_cast<const char*>(p) - offsetof(StreamBox, resource)
    );
}

// See REGISTRY_DESIGN.md (Level 1: Driver Handle -> Resource Handle)
static HandleRegistry<CUstream, StreamHandle> stream_registry;
}  // namespace

StreamHandle create_stream_handle(const ContextHandle& h_ctx, unsigned int flags, int priority) {
    GILReleaseGuard gil;
    CUstream stream = nullptr;
    GreenCtxHandle h_green = get_context_green_ctx(h_ctx);
    if (h_green) {
        err = p_cuGreenCtxStreamCreate
            ? p_cuGreenCtxStreamCreate(&stream, as_cu(h_green), flags, priority)
            : CUDA_ERROR_NOT_SUPPORTED;
    } else {
        err = invoke_in_context_or_undo(
            h_ctx,
            [&]() noexcept { return p_cuStreamCreateWithPriority(&stream, flags, priority); },
            [&]() noexcept { pw_cuStreamDestroy(stream); },
            /*undo_requires_target_context=*/false);
    }
    if (err != CUDA_SUCCESS) {
        return {};
    }

    auto box = std::shared_ptr<const StreamBox>(
        new StreamBox{stream, h_ctx},
        [](const StreamBox* b) {
            stream_registry.unregister_handle(b->resource);
            GILReleaseGuard gil;
            pw_cuStreamDestroy(b->resource);
            delete b;
        }
    );
    StreamHandle h(box, &box->resource);
    stream_registry.register_handle(stream, h);
    return h;
}

StreamHandle create_stream_handle_ref(CUstream stream) {
    if (auto h = stream_registry.lookup(stream)) {
        return h;
    }
    auto box = std::shared_ptr<const StreamBox>(
        new StreamBox{stream, {}},
        [](const StreamBox* b) {
            stream_registry.unregister_handle(b->resource);
            delete b;
        }
    );
    StreamHandle h(box, &box->resource);
    stream_registry.register_handle(stream, h);
    return h;
}

StreamHandle create_stream_handle_with_owner(CUstream stream, PyObject* owner) {
    if (auto h = stream_registry.lookup(stream)) {
        // Reuse handles that already carry structural context metadata, e.g.
        // cuda-core-owned streams.
        if (get_box(h)->h_context) {
            return h;
        }
    }
    if (!owner) {
        return create_stream_handle_ref(stream);
    }
    // GIL required when owner is provided
    GILAcquireGuard gil;
    if (!gil.acquired()) {
        // Python finalizing - fall back to ref version (no owner tracking)
        return create_stream_handle_ref(stream);
    }
    Py_INCREF(owner);
    // Owner-backed handles are NOT registered in the stream registry to avoid
    // corruption when multiple owners wrap the same CUstream (each stacks its
    // own Py_INCREF/Py_DECREF independently).
    auto box = std::shared_ptr<const StreamBox>(
        new StreamBox{stream, {}},
        [owner](const StreamBox* b) {
            GILAcquireGuard gil;
            if (gil.acquired()) {
                Py_DECREF(owner);
            }
            delete b;
        }
    );
    return StreamHandle(box, &box->resource);
}

void py_object_user_object_destroy(void* py_object) noexcept {
    if (!py_object) {
        return;
    }
    GILAcquireGuard gil;
    if (!gil.acquired()) {
        return;
    }
    Py_DECREF(reinterpret_cast<PyObject*>(py_object));
}

// Return the context retained by a stream handle.
ContextHandle get_stream_context(const StreamHandle& h) noexcept {
    return h ? get_box(h)->h_context : ContextHandle{};
}

StreamHandle get_legacy_stream() {
    static StreamHandle handle = create_stream_handle_ref(CU_STREAM_LEGACY);
    return handle;
}

StreamHandle get_per_thread_stream() {
    static StreamHandle handle = create_stream_handle_ref(CU_STREAM_PER_THREAD);
    return handle;
}

StreamHandle create_context_bound_legacy_stream(const ContextHandle& h_context) {
    if (!h_context) {
        return {};
    }
    // Default deleter: this handle never owns CU_STREAM_LEGACY, so nothing
    // needs to run when the last reference is released.
    auto box = std::make_shared<const StreamBox>(StreamBox{CU_STREAM_LEGACY, h_context});
    return StreamHandle(box, &box->resource);
}

// ============================================================================
// Deallocation streams
//
// A DeallocationStream is a StreamHandle used for ordering frees. It differs
// from an ordinary StreamHandle only for default-stream tokens, for which it
// stores the (de)allocation context. Ordinarily, the LEGACY and PER_THREAD
// default streams resolve to whichever context is active at the time they are
// used, but for storing deallocation recipes we need to pin the context. With
// the PER_THREAD token, it is not possible to restore the original stream when
// deallocation runs on a different thread. Therefore, in that case the
// allocating host thread id is also stored so that cross-thread frees can be
// detected and warnings can be issued.
// ============================================================================

// Real streams are copied unchanged. Default-stream tokens without an embedded
// context are bound to the current context. Returns false (and sets err) when a
// default-stream token cannot be bound because no context is current.
static bool make_deallocation_stream(
        const StreamHandle& h, DeallocationStream& out) noexcept {
    out = {};
    if (!h) {
        return true;
    }

    const CUstream stream = as_cu(h);
    if (!is_default_stream(stream)) {
        out = DeallocationStream{h, {}};
        return true;
    }

    StreamHandle h_bound = h;
    if (!get_stream_context(h)) {
        ContextHandle h_ctx = get_current_context();
        if (!h_ctx) {
            if (err == CUDA_SUCCESS) {
                err = CUDA_ERROR_INVALID_CONTEXT;
            }
            return false;
        }
        // Do not register in stream_registry: the token value alone is not
        // a unique stream identity (context is part of the meaning).
        auto box = std::shared_ptr<const StreamBox>(
            new StreamBox{stream, h_ctx});
        h_bound = StreamHandle(box, &box->resource);
    }

    std::thread::id ptds_tid{};
    if (stream == CU_STREAM_PER_THREAD) {
        ptds_tid = std::this_thread::get_id();
    }
    out = DeallocationStream{std::move(h_bound), ptds_tid};
    return true;
}

// ============================================================================
// Event Handles
// ============================================================================

namespace {
struct EventBox {
    CUevent resource;
    bool timing_enabled;
    bool is_blocking_sync;
    bool ipc_enabled;
    int device_id;
    ContextHandle h_context;
};
}  // namespace

static const EventBox* get_box(const EventHandle& h) {
    const CUevent* p = h.get();
    return reinterpret_cast<const EventBox*>(
        reinterpret_cast<const char*>(p) - offsetof(EventBox, resource)
    );
}

bool get_event_timing_enabled(const EventHandle& h) noexcept {
    return h ? get_box(h)->timing_enabled : false;
}

bool get_event_is_blocking_sync(const EventHandle& h) noexcept {
    return h ? get_box(h)->is_blocking_sync : false;
}

bool get_event_ipc_enabled(const EventHandle& h) noexcept {
    return h ? get_box(h)->ipc_enabled : false;
}

int get_event_device_id(const EventHandle& h) noexcept {
    return h ? get_box(h)->device_id : -1;
}

// Return the context retained by an event handle.
ContextHandle get_event_context(const EventHandle& h) noexcept {
    return h ? get_box(h)->h_context : ContextHandle{};
}

// See REGISTRY_DESIGN.md (Level 1: Driver Handle -> Resource Handle)
static HandleRegistry<CUevent, EventHandle> event_registry;

EventHandle create_event_handle(const ContextHandle& h_ctx, unsigned int flags,
                                bool timing_enabled, bool is_blocking_sync,
                                bool ipc_enabled, int device_id) {
    GILReleaseGuard gil;
    CUevent event = nullptr;
    err = invoke_in_context_or_undo(
        h_ctx,
        [&]() noexcept { return p_cuEventCreate(&event, flags); },
        [&]() noexcept { pw_cuEventDestroy(event); },
        /*undo_requires_target_context=*/false);
    if (err != CUDA_SUCCESS) {
        return {};
    }

    auto box = std::shared_ptr<const EventBox>(
        new EventBox{event, timing_enabled, is_blocking_sync, ipc_enabled, device_id, h_ctx},
        [](const EventBox* b) {
            event_registry.unregister_handle(b->resource);
            GILReleaseGuard gil;
            pw_cuEventDestroy(b->resource);
            delete b;
        }
    );
    EventHandle h(box, &box->resource);
    event_registry.register_handle(event, h);
    return h;
}

EventHandle create_event_handle_for_stream(CUstream stream, unsigned int flags) {
    // Resolve the stream's owning context (for default-stream tokens this is
    // the current context, per cuStreamGetCtx) and create the event there, so
    // it can be recorded on `stream` no matter which context is current.
    CUcontext ctx = nullptr;
    {
        GILReleaseGuard gil;
        err = p_cuStreamGetCtx(stream, &ctx);
    }
    if (err != CUDA_SUCCESS) {
        return {};
    }
    if (!ctx) {
        err = CUDA_ERROR_INVALID_CONTEXT;
        return {};
    }
    return create_event_handle(create_context_handle_ref(ctx), flags, false, false, false, -1);
}

EventHandle create_event_handle_ref(CUevent event) {
    if (auto h = event_registry.lookup(event)) {
        return h;
    }
    auto box = std::make_shared<const EventBox>(EventBox{event, false, false, false, -1, {}});
    return EventHandle(box, &box->resource);
}

EventHandle create_event_handle_ipc(const CUipcEventHandle& ipc_handle,
                                    bool is_blocking_sync) {
    GILReleaseGuard gil;
    CUevent event;
    if (CUDA_SUCCESS != (err = p_cuIpcOpenEventHandle(&event, ipc_handle))) {
        return {};
    }

    auto box = std::shared_ptr<const EventBox>(
        new EventBox{event, false, is_blocking_sync, true, -1, {}},
        [](const EventBox* b) {
            event_registry.unregister_handle(b->resource);
            GILReleaseGuard gil;
            pw_cuEventDestroy(b->resource);
            delete b;
        }
    );
    EventHandle h(box, &box->resource);
    event_registry.register_handle(event, h);
    return h;
}

// ============================================================================
// Memory Pool Handles
// ============================================================================

namespace {
struct MemoryPoolBox {
    CUmemoryPool resource;
};
}  // namespace

// Helper to clear peer access before destroying a memory pool.
// Works around nvbug 5698116: recycled pool handles inherit peer access state.
// Must be noexcept since it's called from a shared_ptr deleter.
static void clear_mempool_peer_access(CUmemoryPool pool) noexcept {
    try {
        int device_count = 0;
        if (p_cuDeviceGetCount(&device_count) != CUDA_SUCCESS || device_count <= 0) {
            return;
        }

        std::vector<CUmemAccessDesc> clear_access(device_count);
        for (int i = 0; i < device_count; ++i) {
            clear_access[i].location.type = CU_MEM_LOCATION_TYPE_DEVICE;
            clear_access[i].location.id = i;
            clear_access[i].flags = CU_MEM_ACCESS_FLAGS_PROT_NONE;
        }
        p_cuMemPoolSetAccess(pool, clear_access.data(), device_count);  // Best effort
    } catch (...) {
        // Swallow exceptions - this is best-effort cleanup in destructor context
    }
}

static MemoryPoolHandle wrap_mempool_owned(CUmemoryPool pool) {
    auto box = std::shared_ptr<const MemoryPoolBox>(
        new MemoryPoolBox{pool},
        [](const MemoryPoolBox* b) {
            GILReleaseGuard gil;
            clear_mempool_peer_access(b->resource);
            pw_cuMemPoolDestroy(b->resource);
            delete b;
        }
    );
    return MemoryPoolHandle(box, &box->resource);
}

MemoryPoolHandle create_mempool_handle(const CUmemPoolProps& props) {
    GILReleaseGuard gil;
    CUmemoryPool pool;
    if (CUDA_SUCCESS != (err = p_cuMemPoolCreate(&pool, &props))) {
        return {};
    }
    return wrap_mempool_owned(pool);
}

MemoryPoolHandle create_mempool_handle_ref(CUmemoryPool pool) {
    auto box = std::make_shared<const MemoryPoolBox>(MemoryPoolBox{pool});
    return MemoryPoolHandle(box, &box->resource);
}

MemoryPoolHandle get_device_mempool(int device_id) {
    GILReleaseGuard gil;
    CUmemoryPool pool;
    if (CUDA_SUCCESS != (err = p_cuDeviceGetMemPool(&pool, device_id))) {
        return {};
    }
    return create_mempool_handle_ref(pool);
}

MemoryPoolHandle create_mempool_handle_ipc(int fd, CUmemAllocationHandleType handle_type) {
    GILReleaseGuard gil;
    CUmemoryPool pool;
    auto handle_ptr = reinterpret_cast<void*>(static_cast<uintptr_t>(fd));
    if (CUDA_SUCCESS != (err = p_cuMemPoolImportFromShareableHandle(&pool, handle_ptr, handle_type, 0))) {
        return {};
    }
    return wrap_mempool_owned(pool);
}

// ============================================================================
// Device Pointer Handles
// ============================================================================

namespace {
struct DevicePtrBox {
    CUdeviceptr resource;
    // Mutable so set_deallocation_stream() can update free ordering through a
    // const DevicePtrHandle. Built with make_deallocation_stream so default-
    // stream tokens carry a bound context.
    mutable DeallocationStream deallocation;
};
}  // namespace

// Recovers the owning DevicePtrBox from the aliased CUdeviceptr pointer.
// This works because DevicePtrHandle is a shared_ptr alias pointing to
// &box->resource, so we can compute the containing struct using offsetof.
// The const_cast is safe because we only use this to access the mutable
// deallocation member or in the deleter (where the box is being destroyed).
static DevicePtrBox* get_box(const DevicePtrHandle& h) {
    const CUdeviceptr* p = h.get();
    return reinterpret_cast<DevicePtrBox*>(
        reinterpret_cast<char*>(const_cast<CUdeviceptr*>(p)) - offsetof(DevicePtrBox, resource)
    );
}

// Return the stream that orders a device pointer's deallocation.
StreamHandle deallocation_stream(const DevicePtrHandle& h) noexcept {
    return get_box(h)->deallocation.h_stream;
}

// Replace the stream that orders a device pointer's deallocation.
CUresult set_deallocation_stream(const DevicePtrHandle& h, const StreamHandle& h_stream) noexcept {
    if (!h) {
        return CUDA_ERROR_INVALID_VALUE;
    }
    DeallocationStream ds;
    if (!make_deallocation_stream(h_stream, ds)) {
        return err != CUDA_SUCCESS ? err : CUDA_ERROR_INVALID_CONTEXT;
    }
    get_box(h)->deallocation = std::move(ds);
    return CUDA_SUCCESS;
}

DevicePtrHandle deviceptr_alloc_from_pool(size_t size, const MemoryPoolHandle& h_pool, const StreamHandle& h_stream) {
    GILReleaseGuard gil;
    CUdeviceptr ptr;
    if (CUDA_SUCCESS != (err = p_cuMemAllocFromPoolAsync(&ptr, size, *h_pool, as_cu(h_stream)))) {
        return {};
    }

    DeallocationStream ds;
    if (!make_deallocation_stream(h_stream, ds)) {
        pw_cuMemFreeAsync(ptr, as_cu(h_stream));
        return {};
    }

    auto box = std::shared_ptr<DevicePtrBox>(
        new DevicePtrBox{ptr, std::move(ds)},
        [h_pool](DevicePtrBox* b) {
            GILReleaseGuard gil;
            const DeallocationStream& stream = b->deallocation;
            cleanup_in_context(
                deallocation_context(stream), "cuMemFreeAsync",
                [&]() noexcept {
                    return p_cuMemFreeAsync(
                        b->resource, as_cu(stream.h_stream));
                });
            delete b;
        }
    );
    return DevicePtrHandle(box, &box->resource);
}

DevicePtrHandle deviceptr_alloc_async(size_t size, const StreamHandle& h_stream) {
    GILReleaseGuard gil;
    CUdeviceptr ptr;
    if (CUDA_SUCCESS != (err = p_cuMemAllocAsync(&ptr, size, as_cu(h_stream)))) {
        return {};
    }

    DeallocationStream ds;
    if (!make_deallocation_stream(h_stream, ds)) {
        pw_cuMemFreeAsync(ptr, as_cu(h_stream));
        return {};
    }

    auto box = std::shared_ptr<DevicePtrBox>(
        new DevicePtrBox{ptr, std::move(ds)},
        [](DevicePtrBox* b) {
            GILReleaseGuard gil;
            const DeallocationStream& stream = b->deallocation;
            cleanup_in_context(
                deallocation_context(stream), "cuMemFreeAsync",
                [&]() noexcept {
                    return p_cuMemFreeAsync(
                        b->resource, as_cu(stream.h_stream));
                });
            delete b;
        }
    );
    return DevicePtrHandle(box, &box->resource);
}

// Allocate device memory synchronously with the provided context current.
CUresult deviceptr_alloc_raw(CUdeviceptr* ptr, size_t size,
                             const ContextHandle& h_context) noexcept {
    GILReleaseGuard gil;
    return invoke_in_context_or_undo(
        h_context,
        [&]() noexcept { return p_cuMemAlloc(ptr, size); },
        [&]() noexcept { pw_cuMemFree(*ptr); },
        /*undo_requires_target_context=*/false);
}

DevicePtrHandle deviceptr_alloc_host(size_t size) {
    GILReleaseGuard gil;
    void* ptr;
    if (CUDA_SUCCESS != (err = p_cuMemAllocHost(&ptr, size))) {
        return {};
    }

    auto box = std::shared_ptr<DevicePtrBox>(
        new DevicePtrBox{reinterpret_cast<CUdeviceptr>(ptr), DeallocationStream{}},
        [](DevicePtrBox* b) {
            GILReleaseGuard gil;
            pw_cuMemFreeHost(reinterpret_cast<void*>(b->resource));
            delete b;
        }
    );
    return DevicePtrHandle(box, &box->resource);
}

DevicePtrHandle deviceptr_create_ref(CUdeviceptr ptr) {
    auto box = std::make_shared<DevicePtrBox>(DevicePtrBox{ptr, DeallocationStream{}});
    return DevicePtrHandle(box, &box->resource);
}

DevicePtrHandle deviceptr_create_with_owner(CUdeviceptr ptr, PyObject* owner) {
    if (!owner) {
        return deviceptr_create_ref(ptr);
    }
    // GIL required when owner is provided
    GILAcquireGuard gil;
    if (!gil.acquired()) {
        // Python finalizing - fall back to ref version (no owner tracking)
        return deviceptr_create_ref(ptr);
    }
    Py_INCREF(owner);
    auto box = std::shared_ptr<DevicePtrBox>(
        new DevicePtrBox{ptr, DeallocationStream{}},
        [owner](DevicePtrBox* b) {
            GILAcquireGuard gil;
            if (gil.acquired()) {
                Py_DECREF(owner);
            }
            delete b;
        }
    );
    return DevicePtrHandle(box, &box->resource);
}

DevicePtrHandle deviceptr_create_mapped_graphics(
    CUdeviceptr ptr,
    const GraphicsResourceHandle& h_resource,
    const StreamHandle& h_stream
) {
    DeallocationStream ds;
    if (!make_deallocation_stream(h_stream, ds)) {
        return {};
    }
    auto box = std::shared_ptr<DevicePtrBox>(
        new DevicePtrBox{ptr, std::move(ds)},
        [h_resource](DevicePtrBox* b) {
            GILReleaseGuard gil;
            CUgraphicsResource resource = as_cu(h_resource);
            const DeallocationStream& stream = b->deallocation;
            cleanup_in_context(
                deallocation_context(stream), "cuGraphicsUnmapResources",
                [&]() noexcept {
                    return p_cuGraphicsUnmapResources(
                        1, &resource, as_cu(stream.h_stream));
                });
            delete b;
        }
    );
    return DevicePtrHandle(box, &box->resource);
}

// ============================================================================
// MemoryResource-owned Device Pointer Handles
// ============================================================================

static MRDeallocCallback mr_dealloc_cb = nullptr;

void register_mr_dealloc_callback(MRDeallocCallback cb) {
    mr_dealloc_cb = cb;
}

DevicePtrHandle deviceptr_create_with_mr(CUdeviceptr ptr, size_t size, PyObject* mr) {
    if (!mr) {
        return deviceptr_create_ref(ptr);
    }
    // GIL required when mr is provided
    GILAcquireGuard gil;
    if (!gil.acquired()) {
        return deviceptr_create_ref(ptr);
    }
    Py_INCREF(mr);
    auto box = std::shared_ptr<DevicePtrBox>(
        new DevicePtrBox{ptr, DeallocationStream{}},
        [mr, size](DevicePtrBox* b) {
            GILAcquireGuard gil;
            if (gil.acquired()) {
                if (mr_dealloc_cb) {
                    const DeallocationStream& stream = b->deallocation;
                    cleanup_in_context(
                        deallocation_context(stream), "MemoryResource.deallocate",
                        [&]() noexcept {
                            mr_dealloc_cb(mr, b->resource, size, stream.h_stream);
                            return CUDA_SUCCESS;
                        });
                }
                Py_DECREF(mr);
            }
            delete b;
        }
    );
    return DevicePtrHandle(box, &box->resource);
}

// ============================================================================
// IPC Pointer Cache
// ============================================================================
// This cache handles duplicate IPC imports, which behave differently depending
// on the memory type:
//
// 1. Memory pool allocations (DeviceMemoryResource):
//    Multiple imports of the same allocation succeed and return duplicate
//    pointers. However, the driver has a reference counting bug (nvbug 5570902)
//    where the first cuMemFreeAsync incorrectly unmaps the memory even when
//    imported multiple times. A driver fix is expected.
//
// 2. Pinned memory allocations (PinnedMemoryResource):
//    Duplicate imports result in CUDA_ERROR_ALREADY_MAPPED.
//
// The cache solves both issues by checking the cache before calling
// cuMemPoolImportPointer and returning the existing handle for duplicate
// imports. This provides a consistent user experience where the same IPC
// descriptor can be imported multiple times regardless of memory type.
//
// The cache key is the export_data bytes (CUmemPoolPtrExportData), not the
// returned pointer, because we must check before calling the driver API.


// TODO: When driver fix for nvbug 5570902 is available, consider whether
// the cache is still needed for memory pool allocations (it will still be
// needed for pinned memory).
static bool use_ipc_ptr_cache() {
    return true;
}

namespace {
// Wrapper for CUmemPoolPtrExportData to use as map key
struct ExportDataKey {
    CUmemPoolPtrExportData data;

    bool operator==(const ExportDataKey& other) const {
        return std::memcmp(&data, &other.data, sizeof(data)) == 0;
    }
};

struct ExportDataKeyHash {
    std::size_t operator()(const ExportDataKey& key) const {
        // Simple hash of the bytes
        std::size_t h = 0;
        const auto* bytes = reinterpret_cast<const unsigned char*>(&key.data);
        for (std::size_t i = 0; i < sizeof(key.data); ++i) {
            h = h * 31 + bytes[i];
        }
        return h;
    }
};

}

static HandleRegistry<ExportDataKey, DevicePtrHandle, ExportDataKeyHash> ipc_ptr_cache;
static std::mutex ipc_import_mutex;

DevicePtrHandle deviceptr_import_ipc(const MemoryPoolHandle& h_pool, const void* export_data, const StreamHandle& h_stream) {
    auto data = const_cast<CUmemPoolPtrExportData*>(
        reinterpret_cast<const CUmemPoolPtrExportData*>(export_data));

    if (use_ipc_ptr_cache()) {
        ExportDataKey key;
        std::memcpy(&key.data, data, sizeof(key.data));

        std::lock_guard<std::mutex> lock(ipc_import_mutex);

        if (auto h = ipc_ptr_cache.lookup(key)) {
            return h;
        }

        GILReleaseGuard gil;
        CUdeviceptr ptr;
        if (CUDA_SUCCESS != (err = p_cuMemPoolImportPointer(&ptr, *h_pool, data))) {
            return {};
        }

        DeallocationStream ds;
        if (!make_deallocation_stream(h_stream, ds)) {
            pw_cuMemFreeAsync(ptr, as_cu(h_stream));
            return {};
        }

        auto box = std::shared_ptr<DevicePtrBox>(
            new DevicePtrBox{ptr, std::move(ds)},
            [h_pool, key](DevicePtrBox* b) {
                ipc_ptr_cache.unregister_handle(key);
                GILReleaseGuard gil;
                const DeallocationStream& stream = b->deallocation;
                cleanup_in_context(
                    deallocation_context(stream), "cuMemFreeAsync",
                    [&]() noexcept {
                        return p_cuMemFreeAsync(
                            b->resource, as_cu(stream.h_stream));
                    });
                delete b;
            }
        );
        DevicePtrHandle h(box, &box->resource);
        ipc_ptr_cache.register_handle(key, h);
        return h;

    } else {
        GILReleaseGuard gil;
        CUdeviceptr ptr;
        if (CUDA_SUCCESS != (err = p_cuMemPoolImportPointer(&ptr, *h_pool, data))) {
            return {};
        }

        DeallocationStream ds;
        if (!make_deallocation_stream(h_stream, ds)) {
            pw_cuMemFreeAsync(ptr, as_cu(h_stream));
            return {};
        }

        auto box = std::shared_ptr<DevicePtrBox>(
            new DevicePtrBox{ptr, std::move(ds)},
            [h_pool](DevicePtrBox* b) {
                GILReleaseGuard gil;
                const DeallocationStream& stream = b->deallocation;
                cleanup_in_context(
                    deallocation_context(stream), "cuMemFreeAsync",
                    [&]() noexcept {
                        return p_cuMemFreeAsync(
                            b->resource, as_cu(stream.h_stream));
                    });
                delete b;
            }
        );
        return DevicePtrHandle(box, &box->resource);
    }
}

// ============================================================================
// Library Handles
// ============================================================================

namespace {
struct LibraryBox {
    CUlibrary resource;
};
}  // namespace

LibraryHandle create_library_handle_from_file(const char* path) {
    GILReleaseGuard gil;
    CUlibrary library;
    if (CUDA_SUCCESS != (err = p_cuLibraryLoadFromFile(&library, path, nullptr, nullptr, 0, nullptr, nullptr, 0))) {
        return {};
    }

    auto box = std::shared_ptr<const LibraryBox>(
        new LibraryBox{library},
        [](const LibraryBox* b) {
            GILReleaseGuard gil;
            // TODO: re-enable once LibraryBox tracks its owning context
            // p_cuLibraryUnload(b->resource);
            delete b;
        }
    );
    return LibraryHandle(box, &box->resource);
}

LibraryHandle create_library_handle_from_data(const void* data) {
    GILReleaseGuard gil;
    CUlibrary library;
    if (CUDA_SUCCESS != (err = p_cuLibraryLoadData(&library, data, nullptr, nullptr, 0, nullptr, nullptr, 0))) {
        return {};
    }

    auto box = std::shared_ptr<const LibraryBox>(
        new LibraryBox{library},
        [](const LibraryBox* b) {
            GILReleaseGuard gil;
            // TODO: re-enable once LibraryBox tracks its owning context
            // p_cuLibraryUnload(b->resource);
            delete b;
        }
    );
    return LibraryHandle(box, &box->resource);
}

LibraryHandle create_library_handle_ref(CUlibrary library) {
    auto box = std::make_shared<const LibraryBox>(LibraryBox{library});
    return LibraryHandle(box, &box->resource);
}

// ============================================================================
// Kernel Handles
// ============================================================================

namespace {
struct KernelBox {
    CUkernel resource;
    LibraryHandle h_library;
};
}  // namespace

static const KernelBox* get_box(const KernelHandle& h) {
    const CUkernel* p = h.get();
    return reinterpret_cast<const KernelBox*>(
        reinterpret_cast<const char*>(p) - offsetof(KernelBox, resource)
    );
}

// See REGISTRY_DESIGN.md (Level 1: Driver Handle -> Resource Handle)
static HandleRegistry<CUkernel, KernelHandle> kernel_registry;

KernelHandle create_kernel_handle(const LibraryHandle& h_library, const char* name) {
    GILReleaseGuard gil;
    CUkernel kernel;
    if (CUDA_SUCCESS != (err = p_cuLibraryGetKernel(&kernel, *h_library, name))) {
        return {};
    }

    auto box = std::make_shared<const KernelBox>(KernelBox{kernel, h_library});
    KernelHandle h(box, &box->resource);
    kernel_registry.register_handle(kernel, h);
    return h;
}

KernelHandle create_kernel_handle_ref(CUkernel kernel) {
    if (auto h = kernel_registry.lookup(kernel)) {
        return h;
    }
    auto box = std::make_shared<const KernelBox>(KernelBox{kernel, {}});
    return KernelHandle(box, &box->resource);
}

LibraryHandle get_kernel_library(const KernelHandle& h) noexcept {
    if (!h) return {};
    return get_box(h)->h_library;
}

// ============================================================================
// Graph Handles
// ============================================================================

namespace {

struct NodeAttachment;
using GraphAttachmentMap = std::map<CUgraphNode, NodeAttachment*>;

struct GraphHierarchy;

// Standard-layout alias target for GraphHandle.
struct GraphBoxBase {
    CUgraph resource = nullptr;
};

// Canonical state for one CUgraph. Its GraphHandle aliases resource, whose
// address remains stable for the lifetime of the hierarchy.
struct GraphBox : GraphBoxBase {
    GraphHierarchy* hierarchy = nullptr;  // Non-owning back-reference.
    GraphBox* parent = nullptr;           // Null for the root graph.
    CUgraphNode owner_node = nullptr;     // Node in parent that owns this graph.
    GraphAttachmentMap attachments;       // Non-owning attachment index.
    HandleRegistry<CUgraphNode, GraphNodeHandle> node_handles;

    GraphBox(
            CUgraph resource_,
            GraphHierarchy* hierarchy_,
            GraphBox* parent_ = nullptr,
            CUgraphNode owner_node_ = nullptr) noexcept
        : GraphBoxBase{resource_},
          hierarchy(hierarchy_),
          parent(parent_),
          owner_node(owner_node_) {}
};

// Shared owner of stable GraphBox storage. Every GraphHandle aliases the same
// control block, so any graph handle keeps the entire hierarchy alive.
struct GraphHierarchy {
    std::list<GraphBox> graphs;  // Parent boxes precede their descendants.
    std::list<GraphBox> graveyard;  // Retired child graph tombstones.

    GraphBox* root() noexcept {
        return graphs.empty() ? nullptr : &graphs.front();
    }
};

// See REGISTRY_DESIGN.md (Level 1: Driver Handle -> Resource Handle)
using GraphRegistry = HandleRegistry<CUgraph, GraphHandle>;
static GraphRegistry graph_registry;

// Immutable resource owners for one version of a graph node's parameters.
// Inheriting DeferredCleanupItem lets CUDA's user-object destructor enqueue
// the payload without destroying owners on the callback thread.
struct NodeAttachment : DeferredCleanupItem {
    CUuserObject object = nullptr;
    std::array<OpaqueHandle, 2> owners;

    NodeAttachment(OpaqueHandle owner0, OpaqueHandle owner1)
        : owners{std::move(owner0), std::move(owner1)} {}
};

// shared_ptr deleters for the payloads that need one. Typed handles convert to
// OpaqueHandle by assignment and reuse their own control block, so they need no
// deleter here. The Python deleter follows the owner-release pattern used by
// the stream/deviceptr handles above.
void py_deleter(const void* p) noexcept {
    GILAcquireGuard gil;
    if (gil.acquired()) {
        Py_DECREF(const_cast<PyObject*>(static_cast<const PyObject*>(p)));
    }
}

void free_deleter(const void* p) noexcept {
    std::free(const_cast<void*>(p));
}

GraphBox* get_box(const GraphHandle& h) noexcept {
    auto* value = reinterpret_cast<const GraphBoxBase*>(h.get());
    return const_cast<GraphBox*>(
        static_cast<const GraphBox*>(value));
}

// Rekey a staged attachment map from source nodes to their cloned nodes.
// The caller must release the GIL before calling this function.
CUresult rekey_attachments(
        GraphAttachmentMap& attachments, CUgraph cloned_graph) {
    if (!cloned_graph) {
        return CUDA_ERROR_INVALID_VALUE;
    }
    if (!p_cuGraphNodeFindInClone) {
        return CUDA_ERROR_NOT_SUPPORTED;
    }

    GraphAttachmentMap remapped;
    while (!attachments.empty()) {
        auto attachment = attachments.extract(attachments.begin());
        CUgraphNode cloned_node = nullptr;
        CUresult status = p_cuGraphNodeFindInClone(
            &cloned_node, attachment.key(), cloned_graph);
        if (status != CUDA_SUCCESS) {
            return status;
        }
        attachment.key() = cloned_node;
        if (!remapped.insert(std::move(attachment)).inserted) {
            return CUDA_ERROR_INVALID_VALUE;
        }
    }
    attachments.swap(remapped);
    return CUDA_SUCCESS;
}

struct StagedGraphMetadata {
    const GraphBox* source;
    GraphBox* clone;
    GraphAttachmentMap* attachments;
};
using StagedGraphMetadataList = std::vector<StagedGraphMetadata>;

// Copy a source hierarchy into detached metadata before CUDA mutation.
void stage_graph_metadata(
        const GraphBox& source,
        GraphBox& clone,
        GraphAttachmentMap& attachments,
        std::list<GraphBox>& subgraphs,
        StagedGraphMetadataList& staged) {
    attachments = source.attachments;
    staged.push_back({&source, &clone, &attachments});

    for (const GraphBox& source_child : source.hierarchy->graphs) {
        if (source_child.parent != &source || !source_child.resource) {
            continue;
        }
        GraphBox& cloned_child = subgraphs.emplace_back(
            nullptr,
            clone.hierarchy,
            &clone,
            nullptr);
        stage_graph_metadata(
            source_child,
            cloned_child,
            cloned_child.attachments,
            subgraphs,
            staged);
    }
}

// Bind staged metadata to a CUDA-cloned hierarchy. The root clone resource
// must be populated before entry. The caller must release the GIL.
CUresult rekey_graph_metadata(
        StagedGraphMetadataList& staged) {
    if (!p_cuGraphNodeFindInClone || !p_cuGraphChildGraphNodeGetGraph) {
        return CUDA_ERROR_NOT_SUPPORTED;
    }

    CUresult status;
    for (size_t i = 0; i < staged.size(); ++i) {
        const GraphBox& source = *staged[i].source;
        GraphBox& clone = *staged[i].clone;
        if (i != 0) {
            CUgraphNode cloned_owner = nullptr;
            status = p_cuGraphNodeFindInClone(
                &cloned_owner,
                source.owner_node,
                clone.parent->resource);
            if (status == CUDA_SUCCESS) {
                status = p_cuGraphChildGraphNodeGetGraph(
                    cloned_owner, &clone.resource);
            }
            if (status != CUDA_SUCCESS) {
                return status;
            }
            clone.owner_node = cloned_owner;
        }

        status = rekey_attachments(
            *staged[i].attachments, clone.resource);
        if (status != CUDA_SUCCESS) {
            return status;
        }
    }
    return CUDA_SUCCESS;
}

}  // namespace

OpaqueHandle make_opaque_py(PyObject* obj) {
    Py_INCREF(obj);
    return OpaqueHandle(static_cast<const void*>(obj), py_deleter);
}

OpaqueHandle make_opaque_malloc(void* buf) {
    return OpaqueHandle(static_cast<const void*>(buf), free_deleter);
}

// State held by PreparedAttachment between preparation and commit. It keeps the
// graph alive, tracks the graph-retained replacement, and holds a preallocated
// map entry so commit cannot allocate. Destroying PreparedAttachment rolls back
// the staged user-object retain unless graph_commit_attachment publishes it.
struct PreparedAttachmentState {
    GraphHandle h_graph;
    NodeAttachment* replacement = nullptr;
    GraphAttachmentMap::node_type replacement_entry;

    explicit PreparedAttachmentState(GraphHandle h_graph_)
        : h_graph(std::move(h_graph_)) {}
};

void rollback_prepared_attachment(
        PreparedAttachmentState* state) noexcept {
    if (!state) {
        return;
    }
    if (state->replacement) {
        GraphBox* box = get_box(state->h_graph);
        if (box->resource) {
            GILReleaseGuard gil;
            pw_cuGraphReleaseUserObject(
                box->resource, state->replacement->object, 1);
        }
    }
    delete state;
}

// Detached metadata for a replacement embedded graph hierarchy. Preparation
// copies every attachment map and allocates every GraphBox before CUDA destroys
// the old embedded graph. Commit only rekeys and publishes it.
struct PreparedChildGraphUpdateState {
    GraphHandle h_parent;
    GraphHandle h_source;
    GraphBox* old_root = nullptr;
    CUgraphNode owner_node = nullptr;
    std::list<GraphBox> replacement;
    StagedGraphMetadataList staged;
    std::vector<GraphHandle> handles;

    PreparedChildGraphUpdateState(
            GraphHandle h_parent_,
            GraphHandle h_source_,
            GraphBox* old_root_,
            CUgraphNode owner_node_)
        : h_parent(std::move(h_parent_)),
          h_source(std::move(h_source_)),
          old_root(old_root_),
          owner_node(owner_node_) {}
};

GraphHandle create_graph_handle(CUgraph graph) {
    if (!graph) {
        return {};
    }

    auto hierarchy = std::shared_ptr<GraphHierarchy>(
        new GraphHierarchy{},
        [](GraphHierarchy* hierarchy) {
            for (const GraphBox& box : hierarchy->graphs) {
                if (box.resource) {
                    graph_registry.unregister_handle(box.resource);
                }
            }
            GraphBox* root = hierarchy->root();
            if (root && root->resource) {
                GILReleaseGuard gil;
                pw_cuGraphDestroy(root->resource);
            }
            retry_deferred_cleanup();
            delete hierarchy;
        }
    );
    GraphBox& root = hierarchy->graphs.emplace_back(
        graph, hierarchy.get());

    GraphHandle h_graph(hierarchy, &root.resource);
    graph_registry.register_handle(graph, h_graph);
    return h_graph;
}

GraphHandle create_child_graph_handle(
        CUgraph child_graph, const GraphHandle& h_parent,
        CUgraphNode owner_node) {
    if (!child_graph || !h_parent || !owner_node) {
        return {};
    }
    if (GraphHandle h_graph = graph_registry.lookup(child_graph)) {
        return h_graph;
    }

    GraphBox* parent = get_box(h_parent);
    GraphHierarchy* hierarchy = parent->hierarchy;
    GraphBox& child = hierarchy->graphs.emplace_back(
        child_graph, hierarchy, parent, owner_node);

    GraphHandle h_child(h_parent, &child.resource);
    graph_registry.register_handle(child_graph, h_child);
    return h_child;
}

CUresult graph_prepare_child_graph_update(
        const GraphHandle& h_parent,
        const GraphHandle& h_old_child,
        CUgraphNode owner_node,
        const GraphHandle& h_source,
        PreparedChildGraphUpdate* out_prepared) {
    if (!h_parent || !h_old_child || !owner_node ||
        !h_source || !out_prepared) {
        return CUDA_ERROR_INVALID_VALUE;
    }
    out_prepared->reset();

    GraphBox* parent = get_box(h_parent);
    GraphBox* old_root = get_box(h_old_child);
    GraphBox* source = get_box(h_source);
    // A source from the destination hierarchy can include the old embedded
    // subtree whose raw node keys CUDA destroys during replacement.
    if (!parent->resource || !old_root->resource || !source->resource ||
        old_root->parent != parent ||
        old_root->owner_node != owner_node ||
        source->hierarchy == parent->hierarchy) {
        return CUDA_ERROR_INVALID_VALUE;
    }

    PreparedChildGraphUpdate prepared =
        std::make_shared<PreparedChildGraphUpdateState>(
            h_parent, h_source, old_root, owner_node);

    GraphBox& replacement_root =
        prepared->replacement.emplace_back(
            nullptr, parent->hierarchy, parent, owner_node);
    stage_graph_metadata(
        *source,
        replacement_root,
        replacement_root.attachments,
        prepared->replacement,
        prepared->staged);

    const size_t graph_count = prepared->staged.size();
    prepared->handles.reserve(graph_count);
    for (const StagedGraphMetadata& graph : prepared->staged) {
        prepared->handles.emplace_back(
            h_parent, &graph.clone->resource);
    }

    *out_prepared = std::move(prepared);
    return CUDA_SUCCESS;
}

void publish_child_graph_update(
        PreparedChildGraphUpdateState& state,
        GraphHandle* out_child) {
    GraphBox* parent = get_box(state.h_parent);
    parent->hierarchy->graphs.splice(
        parent->hierarchy->graphs.end(), state.replacement);
    *out_child = state.handles.front();
    graph_registry.register_handles(state.handles);
}

CUresult graph_commit_child_graph_update(
        PreparedChildGraphUpdate& prepared,
        GraphHandle* out_child) {
    if (!prepared || !out_child) {
        return CUDA_ERROR_INVALID_VALUE;
    }
    out_child->reset();

    PreparedChildGraphUpdateState& state = *prepared;
    GraphBox* parent = get_box(state.h_parent);
    if (!parent->resource || !state.old_root->resource) {
        prepared.reset();
        return CUDA_ERROR_INVALID_VALUE;
    }

    CUresult status = CUDA_ERROR_NOT_SUPPORTED;
    CUgraph cloned_root = nullptr;
    if (p_cuGraphChildGraphNodeGetGraph) {
        GILReleaseGuard gil;
        status = p_cuGraphChildGraphNodeGetGraph(
            state.owner_node, &cloned_root);
        if (status == CUDA_SUCCESS) {
            state.staged.front().clone->resource = cloned_root;
            status = rekey_graph_metadata(state.staged);
        }
    }

    // CUDA has already destroyed the old embedded graph. No replacement
    // metadata is visible yet, so this selects only the old generation.
    invalidate_child_graph_state(
        state.h_parent, state.owner_node);

    if (status != CUDA_SUCCESS) {
        prepared.reset();
        throw std::runtime_error(
            "failed to update graph metadata after child graph replacement");
    }

    publish_child_graph_update(state, out_child);
    prepared.reset();
    return status;
}

CUresult graph_get_attachment(
        const GraphHandle& h_graph, CUgraphNode node,
        OpaqueHandle* owner0, OpaqueHandle* owner1) {
    if (!h_graph || !node || (!owner0 && !owner1)) {
        return CUDA_ERROR_INVALID_VALUE;
    }
    if (owner0) {
        owner0->reset();
    }
    if (owner1) {
        owner1->reset();
    }

    GraphBox* box = get_box(h_graph);
    if (!box->resource) {
        return CUDA_ERROR_INVALID_VALUE;
    }
    auto it = box->attachments.find(node);
    if (it != box->attachments.end()) {
        if (owner0) {
            *owner0 = it->second->owners[0];
        }
        if (owner1) {
            *owner1 = it->second->owners[1];
        }
    }
    return CUDA_SUCCESS;
}

CUresult graph_prepare_attachment(
        const GraphHandle& h_graph,
        OpaqueHandle owner0,
        OpaqueHandle owner1,
        PreparedAttachment* out_prepared) {
    if (!out_prepared) {
        return CUDA_ERROR_INVALID_VALUE;
    }
    out_prepared->reset();
    if (!h_graph) {
        return CUDA_ERROR_INVALID_VALUE;
    }

    GraphBox* box = get_box(h_graph);
    if (!box->resource) {
        return CUDA_ERROR_INVALID_VALUE;
    }
    if (!p_cuGraphReleaseUserObject) {
        return CUDA_ERROR_NOT_SUPPORTED;
    }

    PreparedAttachment prepared(
        new PreparedAttachmentState(h_graph),
        PreparedAttachmentDeleter{rollback_prepared_attachment});
    if (owner0 || owner1) {
        if (!p_cuUserObjectCreate || !p_cuUserObjectRelease ||
            !p_cuGraphRetainUserObject) {
            return CUDA_ERROR_NOT_SUPPORTED;
        }

        ensure_deferred_cleanup_ready();
        prepared->replacement = new NodeAttachment(
            std::move(owner0), std::move(owner1));
        GraphAttachmentMap staged;
        try {
            staged.emplace(nullptr, prepared->replacement);
            prepared->replacement_entry =
                staged.extract(staged.begin());
        } catch (...) {
            delete prepared->replacement;
            prepared->replacement = nullptr;
            throw;
        }
        auto* cleanup_item =
            static_cast<DeferredCleanupItem*>(
                prepared->replacement);

        CUuserObject object = nullptr;
        CUresult status;
        {
            GILReleaseGuard gil;
            status = p_cuUserObjectCreate(
                &object, cleanup_item,
                reinterpret_cast<CUhostFn>(enqueue_cleanup),
                1, CU_USER_OBJECT_NO_DESTRUCTOR_SYNC);
            if (status != CUDA_SUCCESS) {
                prepared->replacement_entry.mapped() = nullptr;
                delete prepared->replacement;
                prepared->replacement = nullptr;
                return status;
            }
            prepared->replacement->object = object;
            status = p_cuGraphRetainUserObject(
                box->resource, object, 1, CU_GRAPH_USER_OBJECT_MOVE);
            if (status != CUDA_SUCCESS) {
                prepared->replacement_entry.mapped() = nullptr;
                prepared->replacement = nullptr;
                pw_cuUserObjectRelease(object, 1);
                return status;
            }
        }
    }

    *out_prepared = std::move(prepared);
    return CUDA_SUCCESS;
}

CUresult graph_commit_attachment(
        PreparedAttachment& prepared,
        CUgraphNode node) {
    if (!prepared) {
        return CUDA_ERROR_INVALID_VALUE;
    }

    GraphHandle h_graph = prepared->h_graph;
    GraphBox* box = get_box(h_graph);
    if (!box->resource || (!node && !prepared->replacement)) {
        delete prepared.release();
        return CUDA_ERROR_INVALID_VALUE;
    }
    if (!node) {
        delete prepared.release();
        return CUDA_SUCCESS;
    }

    // Publish the replacement or removal before releasing the previous graph
    // reference; that release can make the previous payload eligible for
    // destruction.
    NodeAttachment* previous = nullptr;
    auto it = box->attachments.find(node);
    if (it == box->attachments.end()) {
        if (prepared->replacement) {
            prepared->replacement_entry.key() = node;
            auto result = box->attachments.insert(
                std::move(prepared->replacement_entry));
            if (!result.inserted) {
                prepared->replacement_entry =
                    std::move(result.node);
                delete prepared.release();
                return CUDA_ERROR_INVALID_VALUE;
            }
        }
    } else {
        previous = it->second;
        if (prepared->replacement) {
            it->second = prepared->replacement;
        } else {
            box->attachments.erase(it);
        }
    }

    delete prepared.release();
    if (!previous) {
        return CUDA_SUCCESS;
    }
    GILReleaseGuard gil;
    return p_cuGraphReleaseUserObject(
        box->resource, previous->object, 1);
}

CUresult graph_clone_attachments(
        const GraphHandle& h_clone,
        const GraphHandle& h_source) {
    if (!h_clone || !h_source) {
        return CUDA_ERROR_INVALID_VALUE;
    }

    GraphBox* clone = get_box(h_clone);
    GraphBox* source = get_box(h_source);
    if (!clone->resource || !source->resource ||
        !clone->attachments.empty()) {
        return CUDA_ERROR_INVALID_VALUE;
    }

    // Build and rekey the clone metadata off-hierarchy so a CUDA mapping error
    // cannot partially publish it.
    GraphAttachmentMap attachments;
    std::list<GraphBox> subgraphs;
    StagedGraphMetadataList staged;
    stage_graph_metadata(
        *source, *clone, attachments, subgraphs, staged);

    std::vector<GraphHandle> handles;
    handles.reserve(subgraphs.size());
    for (GraphBox& graph : subgraphs) {
        handles.emplace_back(h_clone, &graph.resource);
    }

    CUresult status;
    {
        GILReleaseGuard gil;
        status = rekey_graph_metadata(staged);
    }
    if (status != CUDA_SUCCESS) {
        return status;
    }

    clone->attachments.swap(attachments);
    if (subgraphs.empty()) {
        return CUDA_SUCCESS;
    }

    clone->hierarchy->graphs.splice(
        clone->hierarchy->graphs.end(), subgraphs);
    graph_registry.register_handles(handles);
    return CUDA_SUCCESS;
}

// ============================================================================
// Graph Exec Handles
// ============================================================================

namespace {

// Append-only owners introduced by individual executable-node updates. CUDA
// owns this payload through a user object propagated into the CUgraphExec.
struct ExecAttachments : DeferredCleanupItem {
    CUuserObject object = nullptr;
    std::vector<OpaqueHandle> owners;
};

struct GraphExecBox {
    CUgraphExec resource = nullptr;
    ExecAttachments* attachments = nullptr;  // Non-owning.

    ~GraphExecBox() noexcept {
        if (resource) {
            GILReleaseGuard gil;
            pw_cuGraphExecDestroy(resource);
        }
        // The accumulator fields may be dangling after exec destruction.
        retry_deferred_cleanup();
    }
};

GraphExecBox* get_exec_box(const GraphExecHandle& h) noexcept {
    return const_cast<GraphExecBox*>(
        reinterpret_cast<const GraphExecBox*>(h.get()));
}

GraphExecHandle make_graph_exec_handle(
        CUgraphExec graph_exec, ExecAttachments* attachments) {
    struct RawGraphExecGuard {
        CUgraphExec resource;

        ~RawGraphExecGuard() noexcept {
            if (resource) {
                GILReleaseGuard gil;
                pw_cuGraphExecDestroy(resource);
            }
            retry_deferred_cleanup();
        }
    } guard{graph_exec};

    auto box = std::make_shared<GraphExecBox>();
    box->resource = graph_exec;
    box->attachments = attachments;
    guard.resource = nullptr;
    return GraphExecHandle(box, &box->resource);
}

// Holds a fresh accumulator retained on the source graph across a CUDA call
// that propagates user objects into an exec. Releasing drops the source's
// reference: after successful propagation the exec keeps the accumulator
// alive, and otherwise this drops its last reference.
struct ExecAttachmentStaging {
    GraphHandle h_source;
    ExecAttachments* accumulator = nullptr;

    ~ExecAttachmentStaging() noexcept {
        report_cuda_error("cuGraphReleaseUserObject", release(),
                          "failed while dropping a staged graph attachment");
    }

    CUresult release() noexcept {
        if (!h_source || !accumulator) {
            return CUDA_SUCCESS;
        }
        const CUuserObject object = accumulator->object;
        const GraphHandle source = std::move(h_source);
        accumulator = nullptr;
        GILReleaseGuard gil;
        return p_cuGraphReleaseUserObject(*source, object, 1);
    }
};

// Create an accumulator and retain it on h_source, so that a following
// instantiation or whole-graph update propagates a reference into the exec.
CUresult stage_exec_attachments(
        const GraphHandle& h_source, ExecAttachmentStaging* out_staging) {
    if (!p_cuUserObjectCreate || !p_cuUserObjectRelease ||
        !p_cuGraphRetainUserObject || !p_cuGraphReleaseUserObject) {
        return CUDA_ERROR_NOT_SUPPORTED;
    }

    ensure_deferred_cleanup_ready();
    auto* accumulator = new ExecAttachments;

    CUuserObject object = nullptr;
    CUresult status;
    {
        GILReleaseGuard gil;
        status = p_cuUserObjectCreate(
            &object,
            static_cast<DeferredCleanupItem*>(accumulator),
            reinterpret_cast<CUhostFn>(enqueue_cleanup),
            1,
            CU_USER_OBJECT_NO_DESTRUCTOR_SYNC);
        if (status != CUDA_SUCCESS) {
            delete accumulator;
            return status;
        }
        accumulator->object = object;
        status = p_cuGraphRetainUserObject(
            *h_source, object, 1, CU_GRAPH_USER_OBJECT_MOVE);
        if (status != CUDA_SUCCESS) {
            // Dropping the last reference retires the accumulator.
            pw_cuUserObjectRelease(object, 1);
            return status;
        }
    }

    out_staging->h_source = h_source;
    out_staging->accumulator = accumulator;
    return CUDA_SUCCESS;
}

}  // namespace

// State held by PreparedExecAttachment between preparation and commit. It keeps
// the exec alive and remembers the accumulator size before the append, so that
// rollback can drop owners staged for a mutation that CUDA rejected.
struct PreparedExecAttachmentState {
    GraphExecHandle h_exec;
    ExecAttachments* attachments = nullptr;
    size_t original_size = 0;

    PreparedExecAttachmentState(
            GraphExecHandle h_exec_,
            ExecAttachments* attachments_,
            size_t original_size_)
        : h_exec(std::move(h_exec_)),
          attachments(attachments_),
          original_size(original_size_) {}
};

void rollback_prepared_exec_attachment(
        PreparedExecAttachmentState* state) noexcept {
    if (!state) {
        return;
    }
    if (state->attachments) {
        while (state->attachments->owners.size() > state->original_size) {
            state->attachments->owners.pop_back();
        }
    }
    delete state;
}

GraphExecHandle create_graph_exec_handle(
        const GraphHandle& h_source,
        CUDA_GRAPH_INSTANTIATE_PARAMS* params) {
    if (!h_source || !*h_source || !params) {
        err = CUDA_ERROR_INVALID_VALUE;
        return {};
    }
    if (!p_cuGraphInstantiateWithParams) {
        err = CUDA_ERROR_NOT_SUPPORTED;
        return {};
    }

    ExecAttachmentStaging staging;
    if (CUDA_SUCCESS != (err = stage_exec_attachments(h_source, &staging))) {
        return {};
    }

    CUgraphExec graph_exec = nullptr;
    {
        GILReleaseGuard gil;
        err = p_cuGraphInstantiateWithParams(&graph_exec, *h_source, params);
    }
    if (err != CUDA_SUCCESS) {
        return {};
    }
    // CUDA can report a specific failure while returning success. The exec is
    // then unusable, so it stays unadopted for the caller to diagnose from
    // params->result_out.
    if (params->result_out != CUDA_GRAPH_INSTANTIATE_SUCCESS) {
        return {};
    }
    if (!graph_exec) {
        err = CUDA_ERROR_INVALID_VALUE;
        return {};
    }

    GraphExecHandle h_exec = make_graph_exec_handle(
        graph_exec, staging.accumulator);
    if (CUDA_SUCCESS != (err = staging.release())) {
        return {};
    }
    return h_exec;
}

CUresult graph_exec_update(
        const GraphExecHandle& h_exec,
        const GraphHandle& h_source,
        CUgraphExecUpdateResultInfo* result_info) {
    if (!h_exec || !h_source || !*h_source || !result_info) {
        return CUDA_ERROR_INVALID_VALUE;
    }
    if (!p_cuGraphExecUpdate) {
        return CUDA_ERROR_NOT_SUPPORTED;
    }

    GraphExecBox* box = get_exec_box(h_exec);
    if (!box->resource) {
        return CUDA_ERROR_INVALID_VALUE;
    }

    ExecAttachmentStaging staging;
    CUresult status = stage_exec_attachments(h_source, &staging);
    if (status != CUDA_SUCCESS) {
        return status;
    }

    {
        GILReleaseGuard gil;
        status = p_cuGraphExecUpdate(box->resource, *h_source, result_info);
    }
    if (status != CUDA_SUCCESS) {
        return status;
    }

    // CUDA may already have retired the old accumulator. Publish the new one
    // before releasing the source graph's temporary reference.
    box->attachments = staging.accumulator;
    return staging.release();
}

CUresult graph_prepare_exec_attachment(
        const GraphExecHandle& h_exec,
        OpaqueHandle owner0,
        OpaqueHandle owner1,
        PreparedExecAttachment* out_prepared) {
    if (!out_prepared) {
        return CUDA_ERROR_INVALID_VALUE;
    }
    out_prepared->reset();
    if (!h_exec) {
        return CUDA_ERROR_INVALID_VALUE;
    }

    GraphExecBox* box = get_exec_box(h_exec);
    if (!box->resource || !box->attachments) {
        return CUDA_ERROR_INVALID_VALUE;
    }

    ExecAttachments* attachments = box->attachments;
    const size_t original_size = attachments->owners.size();
    const size_t additions =
        static_cast<size_t>(static_cast<bool>(owner0)) +
        static_cast<size_t>(static_cast<bool>(owner1));
    // Reserve before staging so that rollback and commit cannot allocate.
    attachments->owners.reserve(original_size + additions);
    PreparedExecAttachment prepared(
        new PreparedExecAttachmentState(h_exec, attachments, original_size),
        PreparedExecAttachmentDeleter{rollback_prepared_exec_attachment});
    if (owner0) {
        attachments->owners.emplace_back(std::move(owner0));
    }
    if (owner1) {
        attachments->owners.emplace_back(std::move(owner1));
    }
    *out_prepared = std::move(prepared);
    return CUDA_SUCCESS;
}

void graph_commit_exec_attachment(
        PreparedExecAttachment& prepared) noexcept {
    delete prepared.release();
}

namespace {
struct GraphNodeBox {
    mutable CUgraphNode resource;
    GraphHandle h_graph;
};
}  // namespace

static const GraphNodeBox* get_box(const GraphNodeHandle& h) {
    const CUgraphNode* p = h.get();
    return reinterpret_cast<const GraphNodeBox*>(
        reinterpret_cast<const char*>(p) - offsetof(GraphNodeBox, resource)
    );
}

// graphs is ordered parent-before-child. Nulling a selected box marks its
// later descendants, whose parent pointers remain valid after list splicing.
// This permits one allocation-free sweep of the hierarchy.
void invalidate_child_graph_state(
        const GraphHandle& h_parent,
        CUgraphNode owner_node) noexcept {
    if (!h_parent || !owner_node) {
        return;
    }

    GraphBox* parent = get_box(h_parent);
    if (!parent->resource) {
        return;
    }
    GraphHierarchy& hierarchy = *parent->hierarchy;
    for (auto it = hierarchy.graphs.begin();
         it != hierarchy.graphs.end();) {
        auto graph = it++;
        bool is_owned_root = graph->parent == parent &&
                             graph->owner_node == owner_node;
        bool is_descendant = graph->parent &&
                             !graph->parent->resource;
        if (!is_owned_root && !is_descendant) {
            continue;
        }

        // Empty node_handles and invalidate each one.
        for (auto& entry : graph->node_handles.drain()) {
            if (GraphNodeHandle h_node = entry.second.lock()) {
                get_box(h_node)->resource = nullptr;
            }
        }
        graph_registry.unregister_handle(graph->resource);
        graph->resource = nullptr;
        graph->attachments.clear();
        hierarchy.graveyard.splice(
            hierarchy.graveyard.end(), hierarchy.graphs, graph);
    }
}

GraphNodeHandle create_graph_node_handle(CUgraphNode node, const GraphHandle& h_graph) {
    if (!node) {
        auto box = std::make_shared<const GraphNodeBox>(
            GraphNodeBox{nullptr, h_graph});
        return GraphNodeHandle(box, &box->resource);
    }

    GraphBox* graph = get_box(h_graph);
    return graph->node_handles.get_or_create(
        node,
        [node, &h_graph] {
            auto box = std::make_shared<const GraphNodeBox>(
                GraphNodeBox{node, h_graph});
            return GraphNodeHandle(box, &box->resource);
        });
}

GraphHandle graph_node_get_graph(const GraphNodeHandle& h) noexcept {
    return h ? get_box(h)->h_graph : GraphHandle{};
}

void invalidate_graph_node(const GraphNodeHandle& h) noexcept {
    if (!h) {
        return;
    }

    const GraphNodeBox* node_box = get_box(h);
    CUgraphNode node = node_box->resource;
    if (!node) {
        return;
    }
    GraphBox* graph = get_box(node_box->h_graph);
    graph->node_handles.unregister_handle(node);
    node_box->resource = nullptr;
}

// ============================================================================
// Graphics Resource Handles
// ============================================================================

namespace {
struct GraphicsResourceBox {
    CUgraphicsResource resource;
};
}  // namespace

GraphicsResourceHandle create_graphics_resource_handle(CUgraphicsResource resource) {
    auto box = std::shared_ptr<const GraphicsResourceBox>(
        new GraphicsResourceBox{resource},
        [](const GraphicsResourceBox* b) {
            GILReleaseGuard gil;
            pw_cuGraphicsUnregisterResource(b->resource);
            delete b;
        }
    );
    return GraphicsResourceHandle(box, &box->resource);
}

// ============================================================================
// NVRTC Program Handles
// ============================================================================

namespace {
struct NvrtcProgramBox {
    nvrtcProgram resource;
};
}  // namespace

NvrtcProgramHandle create_nvrtc_program_handle(nvrtcProgram prog) {
    auto box = std::shared_ptr<NvrtcProgramBox>(
        new NvrtcProgramBox{prog},
        [](NvrtcProgramBox* b) {
            // Note: nvrtcDestroyProgram takes nvrtcProgram* and nulls it,
            // but we're deleting the box anyway so nulling is harmless.
            if (p_nvrtcDestroyProgram) {
                GILReleaseGuard gil;
                pw_nvrtcDestroyProgram(&b->resource);
            }
            delete b;
        }
    );
    return NvrtcProgramHandle(box, &box->resource);
}

NvrtcProgramHandle create_nvrtc_program_handle_ref(nvrtcProgram prog) {
    auto box = std::make_shared<NvrtcProgramBox>(NvrtcProgramBox{prog});
    return NvrtcProgramHandle(box, &box->resource);
}

// ============================================================================
// NVVM Program Handles
// ============================================================================

namespace {
struct NvvmProgramBox {
    NvvmProgramValue resource;
};
}  // namespace

NvvmProgramHandle create_nvvm_program_handle(nvvmProgram prog) {
    auto box = std::shared_ptr<NvvmProgramBox>(
        new NvvmProgramBox{{prog}},
        [](NvvmProgramBox* b) {
            // Note: nvvmDestroyProgram takes nvvmProgram* and nulls it,
            // but we're deleting the box anyway so nulling is harmless.
            // If NVVM is not available, the function pointer is null.
            if (p_nvvmDestroyProgram) {
                GILReleaseGuard gil;
                pw_nvvmDestroyProgram(&b->resource.raw);
            }
            delete b;
        }
    );
    return NvvmProgramHandle(box, &box->resource);
}

NvvmProgramHandle create_nvvm_program_handle_ref(nvvmProgram prog) {
    auto box = std::make_shared<NvvmProgramBox>(NvvmProgramBox{{prog}});
    return NvvmProgramHandle(box, &box->resource);
}

// ============================================================================
// nvJitLink Handles
// ============================================================================

namespace {
struct NvJitLinkBox {
    NvJitLinkValue resource;
};
}  // namespace

NvJitLinkHandle create_nvjitlink_handle(nvJitLink_t handle) {
    auto box = std::shared_ptr<NvJitLinkBox>(
        new NvJitLinkBox{{handle}},
        [](NvJitLinkBox* b) {
            // Note: nvJitLinkDestroy takes nvJitLinkHandle* and nulls it,
            // but we're deleting the box anyway so nulling is harmless.
            // If nvJitLink is not available, the function pointer is null.
            if (p_nvJitLinkDestroy) {
                GILReleaseGuard gil;
                pw_nvJitLinkDestroy(&b->resource.raw);
            }
            delete b;
        }
    );
    return NvJitLinkHandle(box, &box->resource);
}

NvJitLinkHandle create_nvjitlink_handle_ref(nvJitLink_t handle) {
    auto box = std::make_shared<NvJitLinkBox>(NvJitLinkBox{{handle}});
    return NvJitLinkHandle(box, &box->resource);
}

// ============================================================================
// cuLink Handles
// ============================================================================

namespace {
struct CuLinkBox {
    CUlinkState resource;
};
}  // namespace

CuLinkHandle create_culink_handle(CUlinkState state) {
    auto box = std::shared_ptr<CuLinkBox>(
        new CuLinkBox{state},
        [](CuLinkBox* b) {
            // cuLinkDestroy takes CUlinkState by value (not pointer).
            if (p_cuLinkDestroy) {
                GILReleaseGuard gil;
                pw_cuLinkDestroy(b->resource);
            }
            delete b;
        }
    );
    return CuLinkHandle(box, &box->resource);
}

CuLinkHandle create_culink_handle_ref(CUlinkState state) {
    auto box = std::make_shared<CuLinkBox>(CuLinkBox{state});
    return CuLinkHandle(box, &box->resource);
}

// ============================================================================
// File Descriptor Handles
// ============================================================================

FileDescriptorHandle create_fd_handle(int fd) {
#ifdef _WIN32
    throw std::runtime_error("create_fd_handle is not supported on Windows");
#else
    return FileDescriptorHandle(
        new int(fd),
        [](const int* p) {
            if (::close(*p) != 0) {
                report_message("close() failed for an IPC file descriptor; the descriptor may have leaked");
            }
            delete p;
        }
    );
#endif
}

FileDescriptorHandle create_fd_handle_ref(int fd) {
#ifdef _WIN32
    throw std::runtime_error("create_fd_handle_ref is not supported on Windows");
#else
    return std::make_shared<const int>(fd);
#endif
}

// ============================================================================
// Array / mipmapped-array / texture / surface handles (PR #467)
// ============================================================================

namespace {
struct ArrayBox {
    CUarray resource;
    // Non-null only for a mipmap-level view: keeps the parent mipmap (the real
    // owner of the level's storage) alive for as long as the level is held.
    MipmappedArrayHandle h_parent;
    ContextHandle h_context;
};

struct MipmappedArrayBox {
    CUmipmappedArray resource;
    ContextHandle h_context;
};

// Texture and surface objects are per-context pool indices. Destroying one
// with the wrong context current can silently succeed without freeing it or
// can free an unrelated object, so destruction must enter the creating
// context. Handle-based resources resolve their own context and must not.
struct TexObjectBox {
    // Tagged so TexObjectHandle is a distinct C++ type from DevicePtrHandle /
    // SurfObjectHandle (all wrap `unsigned long long`).
    TexObjectValue resource;
    // Type-erased backing dependency (OpaqueArrayHandle / MipmappedArrayHandle /
    // DevicePtrHandle). The texture's resource is a union; we only need to keep
    // whichever backing it was built from alive, never to dereference it.
    std::shared_ptr<const void> h_backing;
    ContextHandle h_context;
};

struct SurfObjectBox {
    SurfObjectValue resource;
    OpaqueArrayHandle h_array;  // surfaces are always array-backed
    ContextHandle h_context;
};

// Recover an array's owning box from its aliased resource pointer.
const ArrayBox* get_box(const OpaqueArrayHandle& h) noexcept {
    const CUarray* p = h.get();
    return reinterpret_cast<const ArrayBox*>(
        reinterpret_cast<const char*>(p) - offsetof(ArrayBox, resource));
}

// Recover a mipmapped array's owning box from its aliased resource pointer.
const MipmappedArrayBox* get_box(const MipmappedArrayHandle& h) noexcept {
    const CUmipmappedArray* p = h.get();
    return reinterpret_cast<const MipmappedArrayBox*>(
        reinterpret_cast<const char*>(p)
        - offsetof(MipmappedArrayBox, resource));
}

// Wrap an array with shared owning-destruction behavior.
static OpaqueArrayHandle wrap_array_owned(CUarray arr, ContextHandle h_context) {
    auto box = std::shared_ptr<const ArrayBox>(
        new ArrayBox{arr, {}, std::move(h_context)},
        [](const ArrayBox* b) {
            GILReleaseGuard gil;
            pw_cuArrayDestroy(b->resource);
            delete b;
        }
    );
    return OpaqueArrayHandle(box, &box->resource);
}

}  // namespace

OpaqueArrayHandle create_array_handle(const ContextHandle& h_context, const CUDA_ARRAY3D_DESCRIPTOR& desc) {
    GILReleaseGuard gil;
    CUarray arr = nullptr;
    err = invoke_in_context_or_undo(
        h_context,
        [&]() noexcept { return p_cuArray3DCreate(&arr, &desc); },
        [&]() noexcept { pw_cuArrayDestroy(arr); },
        /*undo_requires_target_context=*/false);
    if (err != CUDA_SUCCESS) {
        return {};
    }
    return wrap_array_owned(arr, h_context);
}

OpaqueArrayHandle create_array_handle_ref(CUarray arr) {
    if (!arr) {
        return {};
    }
    auto box = std::make_shared<const ArrayBox>(ArrayBox{arr, {}, {}});
    return OpaqueArrayHandle(box, &box->resource);
}

OpaqueArrayHandle create_array_handle_owning(CUarray arr) {
    if (!arr) {
        return {};
    }
    return wrap_array_owned(arr, {});
}

// Return the context retained by an array handle.
ContextHandle get_array_context(const OpaqueArrayHandle& h) noexcept {
    return h ? get_box(h)->h_context : ContextHandle{};
}

OpaqueArrayHandle create_array_level_handle(const MipmappedArrayHandle& h_mip, unsigned int level) {
    GILReleaseGuard gil;
    CUarray arr;
    ContextHandle h_context = h_mip ? get_box(h_mip)->h_context : ContextHandle{};
    if (CUDA_SUCCESS != (err = p_cuMipmappedArrayGetLevel(&arr, as_cu(h_mip), level))) {
        return {};
    }
    // Non-owning level view: storage belongs to the mipmap. Embed the mipmap
    // handle so the parent outlives this level; the deleter does not destroy.
    auto box = std::shared_ptr<const ArrayBox>(
        new ArrayBox{arr, h_mip, h_context},
        [](const ArrayBox* b) { delete b; }
    );
    return OpaqueArrayHandle(box, &box->resource);
}

MipmappedArrayHandle create_mipmapped_array_handle(const ContextHandle& h_context,
                                                   const CUDA_ARRAY3D_DESCRIPTOR& desc,
                                                   unsigned int num_levels) {
    GILReleaseGuard gil;
    CUmipmappedArray mip = nullptr;
    err = invoke_in_context_or_undo(
        h_context,
        [&]() noexcept { return p_cuMipmappedArrayCreate(&mip, &desc, num_levels); },
        [&]() noexcept { pw_cuMipmappedArrayDestroy(mip); },
        /*undo_requires_target_context=*/false);
    if (err != CUDA_SUCCESS) {
        return {};
    }
    auto box = std::shared_ptr<const MipmappedArrayBox>(
        new MipmappedArrayBox{mip, h_context},
        [](const MipmappedArrayBox* b) {
            GILReleaseGuard gil;
            pw_cuMipmappedArrayDestroy(b->resource);
            delete b;
        }
    );
    return MipmappedArrayHandle(box, &box->resource);
}

// Return the context retained by a mipmapped array handle.
ContextHandle get_mipmapped_array_context(const MipmappedArrayHandle& h) noexcept {
    return h ? get_box(h)->h_context : ContextHandle{};
}

namespace {
TexObjectHandle make_tex_object_handle(const CUDA_RESOURCE_DESC& res,
                                       const CUDA_TEXTURE_DESC& tex,
                                       std::shared_ptr<const void> h_backing,
                                       const ContextHandle& h_context) {
    GILReleaseGuard gil;
    CUtexObject obj = 0;
    err = invoke_in_context_or_undo(
        h_context,
        [&]() noexcept { return p_cuTexObjectCreate(&obj, &res, &tex, nullptr); },
        [&]() noexcept { pw_cuTexObjectDestroy(obj); },
        /*undo_requires_target_context=*/true);
    if (err != CUDA_SUCCESS) {
        return {};
    }
    auto box = std::shared_ptr<const TexObjectBox>(
        new TexObjectBox{TexObjectValue{obj}, std::move(h_backing), h_context},
        [](const TexObjectBox* b) {
            GILReleaseGuard gil;
            cleanup_in_context(b->h_context, "cuTexObjectDestroy", [&]() noexcept {
                return p_cuTexObjectDestroy(b->resource.raw);
            });
            delete b;
        }
    );
    return TexObjectHandle(box, &box->resource);
}
}  // namespace

TexObjectHandle create_tex_object_handle_array(const ContextHandle& h_context,
                                               const CUDA_RESOURCE_DESC& res,
                                               const CUDA_TEXTURE_DESC& tex,
                                               const OpaqueArrayHandle& h_backing) {
    return make_tex_object_handle(res, tex, h_backing, h_context);
}

TexObjectHandle create_tex_object_handle_mipmap(const ContextHandle& h_context,
                                                const CUDA_RESOURCE_DESC& res,
                                                const CUDA_TEXTURE_DESC& tex,
                                                const MipmappedArrayHandle& h_backing) {
    return make_tex_object_handle(res, tex, h_backing, h_context);
}

TexObjectHandle create_tex_object_handle_linear(const ContextHandle& h_context,
                                                const CUDA_RESOURCE_DESC& res,
                                                const CUDA_TEXTURE_DESC& tex,
                                                const DevicePtrHandle& h_backing) {
    return make_tex_object_handle(res, tex, h_backing, h_context);
}

SurfObjectHandle create_surf_object_handle(const ContextHandle& h_context,
                                           const CUDA_RESOURCE_DESC& res,
                                           const OpaqueArrayHandle& h_backing) {
    GILReleaseGuard gil;
    CUsurfObject obj = 0;
    err = invoke_in_context_or_undo(
        h_context,
        [&]() noexcept { return p_cuSurfObjectCreate(&obj, &res); },
        [&]() noexcept { pw_cuSurfObjectDestroy(obj); },
        /*undo_requires_target_context=*/true);
    if (err != CUDA_SUCCESS) {
        return {};
    }
    auto box = std::shared_ptr<const SurfObjectBox>(
        new SurfObjectBox{SurfObjectValue{obj}, h_backing, h_context},
        [](const SurfObjectBox* b) {
            GILReleaseGuard gil;
            cleanup_in_context(b->h_context, "cuSurfObjectDestroy", [&]() noexcept {
                return p_cuSurfObjectDestroy(b->resource.raw);
            });
            delete b;
        }
    );
    return SurfObjectHandle(box, &box->resource);
}

// ============================================================================
// SM resource split wrapper
// ============================================================================

CUresult sm_resource_split(CUdevResource* result, unsigned int nbGroups,
                           const CUdevResource* input, CUdevResource* remainder,
                           unsigned int flags, void* groupParams) {
#if CUDA_VERSION >= 13010
    if (!p_cuDevSmResourceSplit) {
        return CUDA_ERROR_NOT_SUPPORTED;
    }
    return p_cuDevSmResourceSplit(
        result, nbGroups, input, remainder, flags,
        static_cast<CU_DEV_SM_RESOURCE_GROUP_PARAMS*>(groupParams));
#else
    return CUDA_ERROR_NOT_SUPPORTED;
#endif
}

bool has_sm_resource_split() noexcept {
    return p_cuDevSmResourceSplit != nullptr;
}

// ============================================================================
// cuMemcpyWithAttributesAsync wrapper
// ============================================================================

CUresult memcpy_with_attributes_async(CUdeviceptr dst, CUdeviceptr src, size_t size,
                                       void* attr, CUstream hStream) {
#if CUDA_VERSION >= 13020
    if (!p_cuMemcpyWithAttributesAsync) {
        return CUDA_ERROR_NOT_SUPPORTED;
    }
    return p_cuMemcpyWithAttributesAsync(
        dst, src, size, static_cast<CUmemcpyAttributes*>(attr), hStream);
#else
    return CUDA_ERROR_NOT_SUPPORTED;
#endif
}

bool has_memcpy_with_attributes_async() noexcept {
    return p_cuMemcpyWithAttributesAsync != nullptr;
}

}  // namespace cuda_core
