// SPDX-FileCopyrightText: Copyright (c) 2024-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <cuda.h>

namespace cuda_core::rt {

// ============================================================================
// Thread-local error handling
// ============================================================================

// Get and clear the last CUDA error (like cudaGetLastError)
CUresult get_last_error() noexcept;

// Get the last CUDA error without clearing it (like cudaPeekAtLastError)
CUresult peek_last_error() noexcept;

// Explicitly clear the last error
void clear_last_error() noexcept;

// Thread-local status of the most recent CUDA API call in this module. Defined
// in error.cpp; every family source writes it.
extern thread_local CUresult err;

// ============================================================================
// Non-propagating error reporting
//
// Paths that cannot raise (shared_ptr deleters, CUDA callbacks, __dealloc__)
// report failures through these functions instead of discarding them. They
// emit a cuda.core.CUDAWarning when the interpreter can be used and write to
// stderr otherwise; they never raise. See docs/source/error_handling.rst.
// ============================================================================

// Report a failed CUDA call. `detail` replaces the default "failed" wording,
// e.g. "skipped (context activation failed; resource leaked)".
// CUDA_ERROR_DEINITIALIZED (driver shutting down) is never reported.
void report_cuda_error(const char* operation, CUresult status, const char* detail = nullptr) noexcept;

// Report a message that is not tied to a CUresult.
// Implemented in py_report.cpp
void report_message(const char* message) noexcept;

// Report a failed NVRTC/NVVM/nvJitLink call by raw status code.
void report_status_code(const char* operation, long code) noexcept;

// Attach a failed CUDA call to the Python exception currently being handled
// (PEP 678 note, Python 3.11+): for rollback failures inside `except` blocks
// whose original exception is about to be re-raised. When no exception is
// being handled or notes are unavailable, falls back to report_cuda_error().
// Implemented in py_report.cpp
void note_or_report_cuda_error(const char* operation, CUresult status, const char* detail = nullptr) noexcept;

// Detail recorded by a context-scoped helper for the CUresult it is about to
// return, e.g. that the caller's context could not be restored. The Cython
// error path attaches it to the raised CUDAError as a note. Thread-local and
// keyed by status: take_ returns the detail (valid until the next take on this
// thread) and clears it when `status` is the CUresult it was recorded for, and
// returns nullptr otherwise, so a detail whose status was never raised cannot
// attach to an unrelated error.
const char* take_last_error_detail(CUresult status) noexcept;
void clear_last_error_detail() noexcept;

// Tests only: make the next context restoration on this thread fail with
// `status`, leaving the target context current as a real failure would.
// Implemented in context.cpp
void set_context_restore_fault_for_testing(CUresult status) noexcept;

}  // namespace cuda_core::rt
