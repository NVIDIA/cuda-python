// SPDX-FileCopyrightText: Copyright (c) 2024-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "types.hpp"
#include "driver_api.hpp"
#include "error.hpp"
#include <cstddef>
#include <cstdint>
#include <functional>
#include <memory>
#include <mutex>
#include <thread>
#include <type_traits>
#include <unordered_map>
#include <utility>
#include <vector>

namespace cuda_core::rt::detail {

// Implemented in error.cpp
void format_cuda_error(char* buffer, size_t size, const char* operation, CUresult status,
                       const char* detail) noexcept;
// Implemented in error.cpp
void note_context_not_restored(CUcontext previous, CUresult operation_status,
                               CUresult restore_status) noexcept;
// Implemented in error.cpp
void format_operation(char* buffer, size_t size, const char* operation,
                      unsigned long long handle) noexcept;

// Bits of a resource handle for diagnostics: a pointer as its address, an
// integer handle (CUdeviceptr, CUtexObject, ...) as its value, and a pointer
// to a handle (nvrtcDestroyProgram(&prog) and friends) as the handle it points
// to.
template <typename T>
unsigned long long handle_bits(const T& value) noexcept {
    using U = std::remove_cv_t<std::remove_reference_t<T>>;
    if constexpr (std::is_pointer_v<U>) {
        using P = std::remove_cv_t<std::remove_pointer_t<U>>;
        if constexpr (std::is_pointer_v<P>) {
            return value ? handle_bits(*value) : 0ull;
        } else {
            return static_cast<unsigned long long>(reinterpret_cast<uintptr_t>(value));
        }
    } else if constexpr (std::is_integral_v<U> || std::is_enum_v<U>) {
        return static_cast<unsigned long long>(value);
    } else {
        return 0ull;
    }
}

// Store a stream and any state needed to preserve deallocation ordering.
struct DeallocationStream {
    StreamHandle h_stream;
    std::thread::id ptds_tid{};
};

// Implemented in stream.cpp
ContextHandle deallocation_context(const DeallocationStream& stream) noexcept;
// Implemented in stream.cpp
bool make_deallocation_stream(const StreamHandle& h, DeallocationStream& out) noexcept;

// Decorate a status-returning cleanup call to report whenever it fails. CUDA
// calls (CUresult) are reported with the error name and description; NVRTC,
// NVVM and nvJitLink calls (integer status codes) with the raw code.
//
// A pw_ wrapper is not a p_ pointer with logging. On failure it acquires the
// GIL and runs Python: the warning filters, showwarning, or sys.unraisablehook,
// any of which may be user code that calls back into cuda.core. Never invoke
// one while holding a C++ lock; the GIL must be the outermost lock. Where a
// lock must stay held, call the p_ pointer, keep the status, and report after
// the lock is released (see deviceptr_import_ipc and DESIGN.md).
template <auto& Function>
class WarnOnFailure {
public:
    explicit WarnOnFailure(const char* operation) noexcept : operation_(operation) {}

    // The first argument is the resource being released; it is named in the
    // report so that independent failures are not collapsed by the warning
    // registry (see format_operation).
    template <typename First, typename... Rest>
    auto operator()(First&& first, Rest&&... rest) const noexcept {
        const unsigned long long handle = handle_bits(first);
        auto status = Function(std::forward<First>(first), std::forward<Rest>(rest)...);
        report(status, handle);
        return status;
    }

private:
    void report(CUresult status, unsigned long long handle) const noexcept {
        if (status == CUDA_SUCCESS || status == CUDA_ERROR_DEINITIALIZED) {
            return;
        }
        char operation[160];
        format_operation(operation, sizeof(operation), operation_, handle);
        report_cuda_error(operation, status);
    }

    template <typename Status>
    void report(Status status, unsigned long long handle) const noexcept {
        if (static_cast<long>(status) != 0) {
            char operation[160];
            format_operation(operation, sizeof(operation), operation_, handle);
            report_status_code(operation, static_cast<long>(status));
        }
    }

    const char* operation_;
};

// Warning-decorated CUDA operations for deleters and cleanup paths. Each one
// may run user Python on failure (see WarnOnFailure above): no C++ lock held.
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

// Intrusive base for payloads transferred out of CUDA's callback.
struct DeferredCleanupItem {
    DeferredCleanupItem* next = nullptr;
    virtual ~DeferredCleanupItem() noexcept = default;
};

// Implemented in py_deferred_cleanup.cpp
void ensure_deferred_cleanup_ready();
// Implemented in py_deferred_cleanup.cpp
void enqueue_cleanup(void* item) noexcept;

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

}  // namespace cuda_core::rt::detail
