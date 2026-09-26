// SPDX-FileCopyrightText: Copyright (c) 2024-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "types.hpp"
#include "driver_api.hpp"
#include "error.hpp"
#include "internal.hpp"
#include <functional>
#include <type_traits>
#include <utility>

namespace cuda_core::rt::detail {

// Implemented in context.cpp
CUresult enter_context(const ContextHandle& h_context, CUcontext* previous, int* changed) noexcept;
// Implemented in context.cpp
CUresult restore_context(CUcontext previous) noexcept;
// Implemented in context.cpp
CUresult exit_context(CUcontext previous, int changed, CUresult operation_status) noexcept;

// Require a callable to be invocable without throwing.
#define ASSERT_NOTHROW_INVOCABLE(...) \
    static_assert(std::is_nothrow_invocable_v<__VA_ARGS__>, "operation must be noexcept")

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

// Run cleanup with the requested context current, restore the caller's
// context, call `after_cleanup`, then report any failure (activation or
// operation failure first, then restoration failure). `after_cleanup` marks
// the point from which code we do not control may run: the report emits a
// CUDAWarning, which acquires the GIL and runs user code, and whatever the
// caller does next may do the same. It is called unconditionally, so a lock
// the cleanup had to run under is released at one fixed point regardless of
// outcome; pass a hook that unlocks it. Returns the operation or activation
// status; restoration never changes it.
template <typename Fn, typename AfterCleanup>
CUresult cleanup_in_context(const ContextHandle& h_context, const char* name,
                            unsigned long long handle, Fn&& operation,
                            AfterCleanup&& after_cleanup) noexcept {
    ASSERT_NOTHROW_INVOCABLE(Fn&&);
    ASSERT_NOTHROW_INVOCABLE(AfterCleanup&&);
    CUcontext previous = nullptr;
    int changed = 0;
    const char* detail = nullptr;
    CUresult status = enter_context(h_context, &previous, &changed);
    if (status != CUDA_SUCCESS) {
        detail = "skipped (context activation failed; resource leaked)";
    } else {
        status = std::invoke(std::forward<Fn>(operation));
    }
    CUresult restore = exit_context(previous, changed, CUDA_SUCCESS);
    if (restore != CUDA_SUCCESS) {
        // Nothing is raised here, so the detail exit_context recorded has no
        // exception to attach to; drop it.
        clear_last_error_detail();
    }
    std::invoke(std::forward<AfterCleanup>(after_cleanup));
    if (status != CUDA_SUCCESS || restore != CUDA_SUCCESS) {
        char operation_name[160];
        format_operation(operation_name, sizeof(operation_name), name, handle);
        if (status != CUDA_SUCCESS) {
            report_cuda_error(operation_name, status, detail);
        }
        if (restore != CUDA_SUCCESS) {
            report_cuda_error(operation_name, restore, "failed while restoring the caller's context");
        }
    }
    return status;
}

// Same, with nothing to do after the cleanup.
template <typename Fn>
CUresult cleanup_in_context(const ContextHandle& h_context, const char* name,
                            unsigned long long handle, Fn&& operation) noexcept {
    return cleanup_in_context(h_context, name, handle, std::forward<Fn>(operation),
                              []() noexcept {});
}

#undef ASSERT_NOTHROW_INVOCABLE

}  // namespace cuda_core::rt::detail
