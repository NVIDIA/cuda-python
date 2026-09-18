// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
//
// SPDX-License-Identifier: Apache-2.0

// Virtual memory management handles: physical allocations, address
// reservations, mappings, and the range of mappings a buffer owns.
// See VMM_DESIGN.md for the ownership model.

#include "py.hpp"
#include "api.hpp"
#include "context_scope.hpp"
#include "driver_api.hpp"
#include "error.hpp"
#include "internal.hpp"
#include "vmm_range.hpp"
#include <cstddef>
#include <memory>
#include <mutex>
#include <utility>
#include <vector>

namespace cuda_core::rt {

using namespace detail;

// ============================================================================
// Boxes
// ============================================================================

namespace {

struct MemAllocationBox {
    MemAllocationValue resource;          // from cuMemCreate
    size_t size;                          // the only size cuMemMap accepts for it
    std::vector<CUmemAccessDesc> access;  // applied to every mapping of this allocation
};

struct VaReservationBox {
    VaReservationValue resource;          // base address from cuMemAddressReserve
    size_t size;                          // exact reserved size; cuMemAddressFree needs both
};

struct VaMappingBox {
    VaMappingValue resource;              // mapped address
    size_t size;                          // == the allocation's size
    MemAllocationHandle h_alloc;          // released after the unmap
    VaReservationHandle h_reservation;    // freed after the unmap
};

// Recover a box from the aliased handle; the resource is the first member.
template <typename Box, typename Handle>
const Box* box_of(const Handle& h) noexcept {
    return reinterpret_cast<const Box*>(
        reinterpret_cast<const char*>(h.get()) - offsetof(Box, resource));
}

}  // namespace

// ============================================================================
// Physical allocations
// ============================================================================

MemAllocationHandle create_mem_allocation_handle(size_t size, const CUmemAllocationProp& prop,
                                                 const CUmemAccessDesc* descs, size_t count) {
    // Copy the descriptors before the driver call so a failed copy leaves
    // nothing to undo.
    std::vector<CUmemAccessDesc> access(descs, descs + count);
    GILReleaseGuard gil;
    CUmemGenericAllocationHandle handle = 0;
    if (CUDA_SUCCESS != (err = p_cuMemCreate(&handle, size, &prop, 0))) {
        return {};
    }
    auto box = std::shared_ptr<const MemAllocationBox>(
        new MemAllocationBox{{handle}, size, std::move(access)},
        [](const MemAllocationBox* b) {
            GILReleaseGuard gil;
            pw_cuMemRelease(b->resource.raw);
            delete b;
        }
    );
    return MemAllocationHandle(box, &box->resource);
}

size_t mem_allocation_size(const MemAllocationHandle& h) noexcept {
    return h ? box_of<MemAllocationBox>(h)->size : 0;
}

// ============================================================================
// Address reservations
// ============================================================================

VaReservationHandle create_va_reservation_handle(size_t size, size_t alignment, CUdeviceptr hint) {
    GILReleaseGuard gil;
    CUdeviceptr ptr = 0;
    if (CUDA_SUCCESS != (err = p_cuMemAddressReserve(&ptr, size, alignment, hint, 0))) {
        return {};
    }
    auto box = std::shared_ptr<const VaReservationBox>(
        new VaReservationBox{{ptr}, size},
        [](const VaReservationBox* b) {
            GILReleaseGuard gil;
            pw_cuMemAddressFree(b->resource.raw, b->size);
            delete b;
        }
    );
    return VaReservationHandle(box, &box->resource);
}

size_t va_reservation_size(const VaReservationHandle& h) noexcept {
    return h ? box_of<VaReservationBox>(h)->size : 0;
}

// ============================================================================
// Mappings
// ============================================================================

VaMappingHandle create_va_mapping_handle(CUdeviceptr ptr, const MemAllocationHandle& h_alloc,
                                         const VaReservationHandle& h_res) {
    if (!h_alloc || !h_res) {
        err = CUDA_ERROR_INVALID_VALUE;
        return {};
    }
    const MemAllocationBox* alloc = box_of<MemAllocationBox>(h_alloc);
    const VaReservationBox* res = box_of<VaReservationBox>(h_res);
    const CUdeviceptr base = res->resource.raw;
    if (ptr < base || ptr - base > res->size || alloc->size > res->size - (ptr - base)) {
        err = CUDA_ERROR_INVALID_VALUE;
        return {};
    }

    GILReleaseGuard gil;
    if (CUDA_SUCCESS != (err = p_cuMemMap(ptr, alloc->size, 0, alloc->resource.raw, 0))) {
        return {};
    }
    // cuMemSetAccess rejects an empty descriptor list; a mapping with no
    // descriptors is mapped but not accessible, which is what the caller asked for.
    if (!alloc->access.empty()) {
        const CUresult status = p_cuMemSetAccess(ptr, alloc->size, alloc->access.data(), alloc->access.size());
        if (status != CUDA_SUCCESS) {
            pw_cuMemUnmap(ptr, alloc->size);
            err = status;
            return {};
        }
    }
    auto box = std::shared_ptr<const VaMappingBox>(
        new VaMappingBox{{ptr}, alloc->size, h_alloc, h_res},
        [](const VaMappingBox* b) {
            GILReleaseGuard gil;
            pw_cuMemUnmap(b->resource.raw, b->size);
            delete b;  // then the allocation and the reservation release
        }
    );
    return VaMappingHandle(box, &box->resource);
}

size_t va_mapping_size(const VaMappingHandle& h) noexcept {
    return h ? box_of<VaMappingBox>(h)->size : 0;
}

MemAllocationHandle va_mapping_allocation(const VaMappingHandle& h) noexcept {
    return h ? box_of<VaMappingBox>(h)->h_alloc : MemAllocationHandle{};
}

// ============================================================================
// Ranges
// ============================================================================

// Base address -> live range, so a DevicePtrHandle can be recognized as a VMM
// buffer and its range recovered. Two buffers that share a base share the
// range. The range deleter removes the entry before it frees the reservation,
// so the address cannot be re-reserved while the key is present.
static HandleRegistry<CUdeviceptr, VmmRangeHandle> vmm_range_registry;

namespace detail {
void vmm_range_forward_stream(VmmRange& range, const DeallocationStream& stream) noexcept {
    if (!stream.h_stream) {
        return;
    }
    const CUstream s = as_cu(stream.h_stream);
    const CUcontext ctx = as_cu(get_stream_context(stream.h_stream));
    std::lock_guard<std::mutex> lock(range.mu);
    for (const DeallocationStream& recorded : range.streams) {
        if (as_cu(recorded.h_stream) == s && as_cu(get_stream_context(recorded.h_stream)) == ctx) {
            return;
        }
    }
    try {
        range.streams.push_back(stream);
    } catch (...) {
        range.stream_dropped = true;  // reported by the range deleter, outside the lock
    }
}
}  // namespace detail

// True when synchronizing `stream` would disturb a graph capture: the stream
// is capturing, or it is the legacy stream while a blocking stream in its
// context is capturing (the query reports that as
// CUDA_ERROR_STREAM_CAPTURE_IMPLICIT). cuStreamSynchronize would invalidate
// such a capture; cuStreamGetCaptureInfo does not.
static bool sync_would_disturb_capture(CUstream stream) noexcept {
    CUstreamCaptureStatus status = CU_STREAM_CAPTURE_STATUS_NONE;
#if CUDA_VERSION >= 13000
    const CUresult result = p_cuStreamGetCaptureInfo(stream, &status, nullptr, nullptr, nullptr, nullptr, nullptr);
#else
    const CUresult result = p_cuStreamGetCaptureInfo(stream, &status, nullptr, nullptr, nullptr, nullptr);
#endif
    if (result == CUDA_ERROR_STREAM_CAPTURE_IMPLICIT) {
        return true;
    }
    return result == CUDA_SUCCESS && status == CU_STREAM_CAPTURE_STATUS_ACTIVE;
}

// Synchronize a recorded deallocation stream with its bound context current,
// then restore the caller's context. Modeled on cleanup_in_context, with a
// skip message that fits this use: when the sync cannot run, nothing leaks,
// because the range is unmapped regardless. Sets `capture_skipped` instead of
// synchronizing when the sync would disturb a capture.
//
// The sync itself runs with the calling thread in relaxed capture mode.
// cuStreamSynchronize is one of the calls the driver treats as unsafe while a
// capture is active: in the thread's default (global) mode it invalidates
// every global-mode capture in the process, and any non-relaxed capture this
// thread began, on streams unrelated to `s`. Relaxed mode disables that
// interaction; the stream's own capture state is still checked above.
static void sync_recorded_stream(const DeallocationStream& ds, bool& capture_skipped) noexcept {
    const CUstream s = as_cu(ds.h_stream);
    CUcontext previous = nullptr;
    int changed = 0;
    const char* detail = nullptr;
    CUresult status = enter_context(deallocation_context(ds), &previous, &changed);
    if (status != CUDA_SUCCESS) {
        detail = "skipped (context activation failed); the range was unmapped without synchronizing it";
    } else if (sync_would_disturb_capture(s)) {
        capture_skipped = true;
    } else {
        CUstreamCaptureMode mode = CU_STREAM_CAPTURE_MODE_RELAXED;
        const CUresult swapped = p_cuThreadExchangeStreamCaptureMode(&mode);  // `mode` now holds the previous mode
        status = p_cuStreamSynchronize(s);
        if (swapped == CUDA_SUCCESS) {
            p_cuThreadExchangeStreamCaptureMode(&mode);  // restore the previous mode
        }
    }
    const CUresult restore = exit_context(previous, changed, CUDA_SUCCESS);
    if (restore != CUDA_SUCCESS) {
        // Nothing is raised here, so the detail exit_context recorded has no
        // exception to attach to; drop it.
        clear_last_error_detail();
    }
    if (status != CUDA_SUCCESS || restore != CUDA_SUCCESS) {
        char operation_name[160];
        format_operation(operation_name, sizeof(operation_name), "cuStreamSynchronize", handle_bits(s));
        if (status != CUDA_SUCCESS) {
            report_cuda_error(operation_name, status, detail);
        }
        if (restore != CUDA_SUCCESS) {
            report_cuda_error(operation_name, restore, "failed while restoring the caller's context");
        }
    }
}

VmmRangeHandle create_vmm_range(CUdeviceptr base) {
    auto range = VmmRangeHandle(
        new VmmRange(base),
        [](VmmRange* r) {
            GILReleaseGuard gil;
            vmm_range_registry.unregister_handle(r->base);
            if (!py_is_finalizing()) {
                bool capture_skipped = false;
                for (const DeallocationStream& ds : r->streams) {
                    sync_recorded_stream(ds, capture_skipped);
                }
                if (capture_skipped) {
                    report_message(
                        "a VirtualMemoryResource buffer was released while its deallocation stream "
                        "is capturing, or is the legacy stream while another stream in its context is "
                        "capturing; the range was unmapped without synchronizing that stream");
                }
                if (r->stream_dropped) {
                    report_message(
                        "a VirtualMemoryResource buffer could not record a deallocation stream "
                        "(out of memory); the range was unmapped without synchronizing it");
                }
            }
            delete r;  // mappings unmap; reservations free and allocations release
        }
    );
    vmm_range_registry.register_handle(base, range);
    return range;
}

VmmRangeHandle vmm_range(const DevicePtrHandle& h) {
    return h ? vmm_range_registry.lookup(as_cu(h)) : VmmRangeHandle{};
}

size_t vmm_range_count(const VmmRangeHandle& range) noexcept {
    return range ? range->mappings.size() : 0;
}

VaMappingHandle vmm_range_mapping(const VmmRangeHandle& range, size_t index) noexcept {
    if (!range || index >= range->mappings.size()) {
        return {};
    }
    return range->mappings[index];
}

size_t vmm_range_total(const VmmRangeHandle& range) noexcept {
    size_t total = 0;
    if (range) {
        for (const VaMappingHandle& m : range->mappings) {
            total += va_mapping_size(m);
        }
    }
    return total;
}

void vmm_range_reserve(const VmmRangeHandle& range, size_t count) {
    if (range) {
        range->mappings.reserve(count);
    }
}

void vmm_range_append(const VmmRangeHandle& range, const VaMappingHandle& mapping) {
    if (range && mapping) {
        range->mappings.push_back(mapping);
    }
}

}  // namespace cuda_core::rt
