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
#include <cstddef>
#include <memory>
#include <utility>
#include <vector>

namespace cuda_core::rt {

using namespace detail;

// ============================================================================
// Boxes
// ============================================================================

namespace {

struct MemAllocationBox {
    MemAllocationValue resource;          // from cuMemCreate or cuMemImportFromShareableHandle
    size_t size;                          // the only size cuMemMap accepts for it
    std::vector<CUmemAccessDesc> access;  // applied to every mapping of this allocation
    bool imported;                        // another process created it; it cannot be exported again
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

// Wrap an allocation handle this process holds one reference to, whether
// cuMemCreate or cuMemImportFromShareableHandle produced it. The last
// reference releases it; the memory is freed once no mapping remains.
MemAllocationHandle wrap_mem_allocation(CUmemGenericAllocationHandle handle, size_t size,
                                        std::vector<CUmemAccessDesc>&& access, bool imported) {
    auto box = std::shared_ptr<const MemAllocationBox>(
        new MemAllocationBox{{handle}, size, std::move(access), imported},
        [](const MemAllocationBox* b) {
            GILReleaseGuard gil;
            pw_cuMemRelease(b->resource.raw);
            delete b;
        }
    );
    return MemAllocationHandle(box, &box->resource);
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
    if (CUDA_SUCCESS != (err = DRIVER_CALL(cuMemCreate, &handle, size, &prop, 0))) {
        return {};
    }
    return wrap_mem_allocation(handle, size, std::move(access), false);
}

MemAllocationHandle import_mem_allocation_handle(void* os_handle, CUmemAllocationHandleType handle_type,
                                                 size_t size, const CUmemAccessDesc* descs, size_t count) {
    // Copy the descriptors before the driver call so a failed copy leaves
    // nothing to undo.
    std::vector<CUmemAccessDesc> access(descs, descs + count);
    GILReleaseGuard gil;
    CUmemGenericAllocationHandle handle = 0;
    if (CUDA_SUCCESS != (err = DRIVER_CALL(cuMemImportFromShareableHandle, &handle, os_handle, handle_type))) {
        return {};
    }
    return wrap_mem_allocation(handle, size, std::move(access), true);
}

size_t mem_allocation_size(const MemAllocationHandle& h) noexcept {
    return h ? box_of<MemAllocationBox>(h)->size : 0;
}

bool mem_allocation_is_imported(const MemAllocationHandle& h) noexcept {
    return h ? box_of<MemAllocationBox>(h)->imported : false;
}

// ============================================================================
// Address reservations
// ============================================================================

VaReservationHandle create_va_reservation_handle(size_t size, size_t alignment, CUdeviceptr hint) {
    GILReleaseGuard gil;
    CUdeviceptr ptr = 0;
    if (CUDA_SUCCESS != (err = DRIVER_CALL(cuMemAddressReserve, &ptr, size, alignment, hint, 0))) {
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
    if (CUDA_SUCCESS != (err = DRIVER_CALL(cuMemMap, ptr, alloc->size, 0, alloc->resource.raw, 0))) {
        return {};
    }
    // cuMemSetAccess rejects an empty descriptor list; a mapping with no
    // descriptors is mapped but not accessible, which is what the caller asked for.
    if (!alloc->access.empty()) {
        const CUresult status = DRIVER_CALL(cuMemSetAccess, ptr, alloc->size, alloc->access.data(), alloc->access.size());
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

// True when synchronizing `stream` would disturb a graph capture: the stream
// is capturing, or it is the legacy stream while a blocking stream in its
// context is capturing (the query reports that as
// CUDA_ERROR_STREAM_CAPTURE_IMPLICIT). cuStreamSynchronize would invalidate
// such a capture; cuStreamIsCapturing does not. It is the query
// cuStreamGetCaptureInfo makes first, with one signature on every CUDA major.
static bool sync_would_disturb_capture(CUstream stream) noexcept {
    CUstreamCaptureStatus status = CU_STREAM_CAPTURE_STATUS_NONE;
    const CUresult result = DRIVER_CALL(cuStreamIsCapturing, stream, &status);
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
        const CUresult swapped = DRIVER_CALL(cuThreadExchangeStreamCaptureMode, &mode);  // `mode` now holds the previous mode
        status = DRIVER_CALL(cuStreamSynchronize, s);
        if (swapped == CUDA_SUCCESS) {
            DRIVER_CALL(cuThreadExchangeStreamCaptureMode, &mode);  // restore the previous mode
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

namespace detail {
void vmm_sync_before_release(const DeallocationStream& stream) noexcept {
    // Order the release on this buffer's stream, as the other DevicePtrBox
    // deleters do, unless the interpreter is finalizing or the sync would
    // disturb a capture. A host-located buffer recorded no stream.
    if (!stream.h_stream || py_is_finalizing()) {
        return;
    }
    bool capture_skipped = false;
    sync_recorded_stream(stream, capture_skipped);
    if (capture_skipped) {
        report_message(
            "a VirtualMemoryResource buffer was released while its deallocation stream "
            "is capturing, or is the legacy stream while another stream in its context is "
            "capturing; the buffer was unmapped without synchronizing that stream");
    }
}
}  // namespace detail

VmmRangeHandle create_vmm_range(const std::vector<VaMappingHandle>& mappings) {
    return std::make_shared<const VmmRange>(mappings);
}

std::vector<VaMappingHandle> vmm_range_mappings(const VmmRangeHandle& range) {
    return range ? *range : VmmRange{};
}

size_t vmm_range_total(const VmmRangeHandle& range) noexcept {
    size_t total = 0;
    if (range) {
        for (const VaMappingHandle& m : *range) {
            total += va_mapping_size(m);
        }
    }
    return total;
}

}  // namespace cuda_core::rt
