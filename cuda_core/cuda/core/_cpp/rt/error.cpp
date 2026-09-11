// SPDX-FileCopyrightText: Copyright (c) 2024-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
//
// SPDX-License-Identifier: Apache-2.0

#include "error.hpp"
#include "driver_api.hpp"
#include "internal.hpp"
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <cuda.h>

namespace cuda_core::rt {

using namespace detail;

// ----------------------------------------------------------------------------
// Non-propagating error reporting
//
// Deleters, CUDA callbacks and other non-propagating paths cannot raise. They
// report through report_cuda_error()/report_message(), which emit a
// cuda.core.CUDAWarning when the interpreter is usable and fall back to stderr
// otherwise. See docs/source/error_handling.rst for the policy.
// ----------------------------------------------------------------------------

namespace {
// Thread-local detail attached to the next raised CUDAError with a matching
// status (see take_last_error_detail()). Written only by propagating helpers.
// The taken copy stays valid until the next take on the same thread.
thread_local char last_error_detail[512] = {0};
thread_local char taken_error_detail[512] = {0};
thread_local CUresult last_error_detail_status = CUDA_SUCCESS;
}  // namespace

namespace detail {
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
}  // namespace detail

// Report a failed non-CUDA call (NVRTC, NVVM, nvJitLink) from a path that
// cannot raise.
void report_status_code(const char* operation, long code) noexcept {
    char message[256];
    std::snprintf(message, sizeof(message), "%s failed (status %ld)", operation, code);
    report_message(message);
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

namespace detail {
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
}  // namespace detail

// ============================================================================
// Thread-local error handling
// ============================================================================

// Thread-local status of the most recent CUDA API call in this module.
thread_local CUresult err = CUDA_SUCCESS;

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

}  // namespace cuda_core::rt
