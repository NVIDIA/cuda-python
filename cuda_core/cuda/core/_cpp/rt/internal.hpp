// SPDX-FileCopyrightText: Copyright (c) 2024-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "types.hpp"
#include "driver_api.hpp"
#include "error.hpp"
#include <condition_variable>
#include <cstddef>
#include <cstdint>
#include <functional>
#include <memory>
#include <mutex>
#include <stdexcept>
#include <thread>
#include <type_traits>
#include <unordered_map>
#include <utility>

namespace cuda_core::rt::detail {

// Implemented in error.cpp
void format_cuda_error(char *buffer, size_t size, const char *operation, CUresult status, const char *detail) noexcept;
// Implemented in error.cpp
void note_context_not_restored(CUcontext previous, CUresult operation_status, CUresult restore_status) noexcept;
// Implemented in error.cpp
void format_operation(char *buffer, size_t size, const char *operation, unsigned long long handle) noexcept;

// Bits of a resource handle for diagnostics: a pointer as its address, an
// integer handle (CUdeviceptr, CUtexObject, ...) as its value, and a pointer
// to a handle (nvrtcDestroyProgram(&prog) and friends) as the handle it points
// to.
template <typename T>
unsigned long long
handle_bits(const T &value) noexcept
{
    using U = std::remove_cv_t<std::remove_reference_t<T>>;
    if constexpr (std::is_pointer_v<U>) {
        using P = std::remove_cv_t<std::remove_pointer_t<U>>;
        if constexpr (std::is_pointer_v<P>) {
            return value ? handle_bits(*value) : 0ULL;
        }
        else {
            return static_cast<unsigned long long>(reinterpret_cast<uintptr_t>(value));
        }
    }
    else if constexpr (std::is_integral_v<U> || std::is_enum_v<U>) {
        return static_cast<unsigned long long>(value);
    }
    else {
        return 0ULL;
    }
}

// Store a stream and any state needed to preserve deallocation ordering.
struct DeallocationStream {
    StreamHandle h_stream;
    std::thread::id ptds_tid;
};

// Implemented in stream.cpp
ContextHandle deallocation_context(const DeallocationStream &stream) noexcept;
// Implemented in stream.cpp. Not noexcept: binding a default-stream token
// allocates. May throw std::bad_alloc.
bool make_deallocation_stream(const StreamHandle &h, DeallocationStream &out);

// Implemented in virtual_memory.cpp. Synchronize the stream a VMM buffer
// recorded before its mappings are released. Skips the sync, with a report
// when a capture is the reason, if no stream was recorded, the interpreter is
// finalizing, or the sync would disturb a graph capture (VMM_DESIGN.md).
void vmm_sync_before_release(const DeallocationStream &stream) noexcept;

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
template <auto &Function, FnTable Table = FnTable::driver>
class WarnOnFailure {
   public:
    explicit WarnOnFailure(const char *operation) noexcept : operation_(operation) {}

    // The first argument is the resource being released; it is named in the
    // report so that independent failures are not collapsed by the warning
    // registry (see format_operation). The call goes through the function
    // table like DRIVER_CALL: it reports an unavailable entry and yields an
    // error status, never a null dereference.
    template <typename First, typename... Rest>
    auto
    operator()(First &&first, Rest &&...rest) const noexcept
    {
        const unsigned long long handle = handle_bits(first);
        auto status = detail::fn_or_unavailable(Function, Table, operation_)(
            std::forward<First>(first),
            std::forward<Rest>(rest)...
        );
        report(status, handle);
        return status;
    }

   private:
    void
    report(CUresult status, unsigned long long handle) const noexcept
    {
        if (status == CUDA_SUCCESS || status == CUDA_ERROR_DEINITIALIZED) {
            return;
        }
        char operation[kOperationNameSize];
        format_operation(operation, sizeof(operation), operation_, handle);
        report_cuda_error(operation, status);
    }

    template <typename Status>
    void
    report(Status status, unsigned long long handle) const noexcept
    {
        if (static_cast<long>(status) != 0) {
            char operation[kOperationNameSize];
            format_operation(operation, sizeof(operation), operation_, handle);
            report_status_code(operation, static_cast<long>(status));
        }
    }

    const char *operation_;
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
const WarnOnFailure<p_cuMemRelease> pw_cuMemRelease{"cuMemRelease"};
const WarnOnFailure<p_cuMemUnmap> pw_cuMemUnmap{"cuMemUnmap"};
const WarnOnFailure<p_cuMemAddressFree> pw_cuMemAddressFree{"cuMemAddressFree"};
const WarnOnFailure<p_cuGraphDestroy> pw_cuGraphDestroy{"cuGraphDestroy"};
const WarnOnFailure<p_cuGraphExecDestroy> pw_cuGraphExecDestroy{"cuGraphExecDestroy"};
const WarnOnFailure<p_cuGraphicsUnregisterResource> pw_cuGraphicsUnregisterResource{"cuGraphicsUnregisterResource"};
const WarnOnFailure<p_cuLinkDestroy> pw_cuLinkDestroy{"cuLinkDestroy"};
const WarnOnFailure<p_cuUserObjectRelease> pw_cuUserObjectRelease{"cuUserObjectRelease"};
const WarnOnFailure<p_cuGraphReleaseUserObject> pw_cuGraphReleaseUserObject{"cuGraphReleaseUserObject"};
const WarnOnFailure<p_nvrtcDestroyProgram, FnTable::nvrtc> pw_nvrtcDestroyProgram{"nvrtcDestroyProgram"};
const WarnOnFailure<p_nvvmDestroyProgram, FnTable::nvvm> pw_nvvmDestroyProgram{"nvvmDestroyProgram"};
const WarnOnFailure<p_nvJitLinkDestroy, FnTable::nvjitlink> pw_nvJitLinkDestroy{"nvJitLinkDestroy"};

// Intrusive base for payloads transferred out of CUDA's callback.
struct DeferredCleanupItem {
    DeferredCleanupItem *next = nullptr;
    virtual ~DeferredCleanupItem() noexcept = default;
};

// Implemented in py_deferred_cleanup.cpp
void ensure_deferred_cleanup_ready();
// Implemented in py_deferred_cleanup.cpp
void enqueue_cleanup(void *item) noexcept;

// ============================================================================
// Handle reverse-lookup registry
//
// Maps raw CUDA handles (CUevent, CUkernel, etc.) back to their owning
// shared_ptr so that _ref constructors can recover full metadata.
// Uses weak_ptr to avoid preventing destruction.
// ============================================================================

template <typename Key, typename Handle, typename Hash = std::hash<Key>>
class HandleRegistry {
   public:
    using MapType = std::unordered_map<Key, std::weak_ptr<typename Handle::element_type>, Hash>;

    void
    register_handle(const Key &key, const Handle &h)
    {
        std::scoped_lock lock(mutex_);
        map_[key] = h;
    }

    void
    unregister_handle(const Key &key) noexcept
    {
        std::scoped_lock lock(mutex_);
        map_.erase(key);
    }
    Handle
    lookup(const Key &key)
    {
        std::scoped_lock lock(mutex_);
        auto it = map_.find(key);
        if (it != map_.end()) {
            if (auto h = it->second.lock()) {
                return h;
            }
            map_.erase(it);
        }
        return {};
    }

    template <typename Factory>
    Handle
    get_or_create(const Key &key, Factory &&create)
    {
        std::scoped_lock lock(mutex_);
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

    // MSVC's std::unordered_map allocates in its default constructor, so an
    // allocation failure here terminates. Running out of memory is fatal either way.
    MapType
    // NOLINTNEXTLINE(bugprone-exception-escape)
    drain() noexcept
    {
        std::scoped_lock lock(mutex_);
        MapType extracted;
        extracted.swap(map_);
        return extracted;
    }

   private:
    std::mutex mutex_;
    MapType map_;
};

// ============================================================================
// Handle registry that waits for in-flight releases
//
// Like HandleRegistry, but an entry outlives its handle's last reference: only
// the owner's deleter removes it, by calling release() once its cleanup is
// done. get_or_create() waits while the entry under its key has expired, so a
// new owner is never created while the previous owner is still releasing the
// resource. Used where the driver hands the same resource to repeat imports
// (IPC pools and IPC pointers), so a new owner and an old owner's cleanup
// would act on one driver resource. See REGISTRY_DESIGN.md.
// ============================================================================

template <typename Key, typename Handle, typename Hash = std::hash<Key>>
class ReleasingHandleRegistry {
   public:
    // Return the live handle registered under key, waiting first while the
    // previous owner's release is in flight. Otherwise register and return the
    // handle that create() returns; an empty handle is returned unregistered.
    // create() runs under the registry's mutex, so it must not acquire the GIL
    // or release a handle whose deleter calls release().
    template <typename Factory>
    Handle
    get_or_create(const Key &key, Factory &&create)
    {
        std::unique_lock lock(mutex_);
        Lookup found;
        released_.wait(lock, [&] {
            found = find(lock, key);
            return found.state != State::Releasing;
        });
        if (found.state == State::Live) {
            return found.owner;
        }
        // Insert before creating, so a failed insertion has nothing to undo.
        auto it = map_.try_emplace(key).first;
        Handle h;
        try {
            h = create();
        }
        catch (...) {
            map_.erase(it);
            throw;
        }
        if (h) {
            it->second = Entry{h, h.get()};
        }
        else {
            map_.erase(it);
        }
        return h;
    }

    // Called by an owner's deleter after its last driver call on the resource:
    // erase key's entry if it belongs to the handle that pointed to resource,
    // and wake the importers waiting for it.
    void
    release(const Key &key, const typename Handle::element_type *resource) noexcept
    {
        {
            std::scoped_lock lock(mutex_);
            auto it = map_.find(key);
            if (it != map_.end() && it->second.resource == resource) {
                map_.erase(it);
            }
        }
        released_.notify_all();
    }

   private:
    struct Entry {
        std::weak_ptr<typename Handle::element_type> handle;
        const typename Handle::element_type *resource = nullptr;
    };

    // What is registered under a key. An entry outlives its owner's last
    // reference, so an entry whose owner has expired is a release in flight.
    enum class State {
        Vacant,     // no entry: the caller may create an owner
        Live,       // an owner holds the resource
        Releasing,  // the owner's deleter has not called release() yet
    };

    struct Lookup {
        State state = State::Vacant;
        Handle owner;  // set only when state is Live
    };

    // Takes the caller's lock as proof that it holds mutex_, and checks it.
    Lookup
    find(const std::unique_lock<std::mutex> &held, const Key &key) const
    {
        if (!held.owns_lock() || held.mutex() != &mutex_) {
            throw std::runtime_error(
                "internal cuda.core error, please report: registry lookup without the registry's lock"
            );
        }
        auto it = map_.find(key);
        if (it == map_.end()) {
            return {State::Vacant, {}};
        }
        Handle owner = it->second.handle.lock();
        if (!owner) {
            return {State::Releasing, {}};
        }
        return {State::Live, std::move(owner)};
    }

    std::mutex mutex_;
    std::condition_variable released_;
    std::unordered_map<Key, Entry, Hash> map_;
};

}  // namespace cuda_core::rt::detail
