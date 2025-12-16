// SPDX-FileCopyrightText: Copyright (c) 2024-2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <cstdint>

#include "resource_handles.hpp"

namespace cuda_core {

// C++ capsule API for cross-extension-module calls.
//
// The function-pointer table is exported from the Python extension module
// `cuda.core.experimental._resource_handles` as a PyCapsule named:
//
//   "cuda.core.experimental._resource_handles._CXX_API"
//
// Other extension modules import the capsule and dispatch through the table to
// ensure there is a single owner of all correctness-critical static/thread_local
// state in resource_handles.cpp (caches, last-error state, etc.).

static constexpr std::uint32_t RESOURCE_HANDLES_CXX_API_VERSION = 1;

struct ResourceHandlesCxxApiV1 {
    std::uint32_t abi_version;
    std::uint32_t struct_size;

    // Thread-local error handling
    CUresult (*get_last_error)() noexcept;
    CUresult (*peek_last_error)() noexcept;
    void (*clear_last_error)() noexcept;

    // Context handles
    ContextHandle (*create_context_handle_ref)(CUcontext ctx);
    ContextHandle (*get_primary_context)(int device_id) noexcept;
    ContextHandle (*get_current_context)() noexcept;

    // Stream handles
    StreamHandle (*create_stream_handle)(ContextHandle h_ctx, unsigned int flags, int priority);
    StreamHandle (*create_stream_handle_ref)(CUstream stream);
    StreamHandle (*create_stream_handle_with_owner)(CUstream stream, PyObject* owner);
    StreamHandle (*get_legacy_stream)() noexcept;
    StreamHandle (*get_per_thread_stream)() noexcept;

    // Event handles
    EventHandle (*create_event_handle)(ContextHandle h_ctx, unsigned int flags);
    EventHandle (*create_event_handle_noctx)(unsigned int flags);
    EventHandle (*create_event_handle_ipc)(const CUipcEventHandle& ipc_handle);

    // Memory pool handles
    MemoryPoolHandle (*create_mempool_handle)(const CUmemPoolProps& props);
    MemoryPoolHandle (*create_mempool_handle_ref)(CUmemoryPool pool);
    MemoryPoolHandle (*get_device_mempool)(int device_id) noexcept;
    MemoryPoolHandle (*create_mempool_handle_ipc)(int fd, CUmemAllocationHandleType handle_type);

    // Device pointer handles
    DevicePtrHandle (*deviceptr_alloc_from_pool)(
        size_t size,
        MemoryPoolHandle h_pool,
        StreamHandle h_stream);
    DevicePtrHandle (*deviceptr_alloc_async)(size_t size, StreamHandle h_stream);
    DevicePtrHandle (*deviceptr_alloc)(size_t size);
    DevicePtrHandle (*deviceptr_alloc_host)(size_t size);
    DevicePtrHandle (*deviceptr_create_ref)(CUdeviceptr ptr);
    DevicePtrHandle (*deviceptr_create_with_owner)(CUdeviceptr ptr, PyObject* owner);
    DevicePtrHandle (*deviceptr_import_ipc)(
        MemoryPoolHandle h_pool,
        const void* export_data,
        StreamHandle h_stream);
    StreamHandle (*deallocation_stream)(const DevicePtrHandle& h);
    void (*set_deallocation_stream)(const DevicePtrHandle& h, StreamHandle h_stream);
};

// Return pointer to a process-wide singleton table.
const ResourceHandlesCxxApiV1* get_resource_handles_cxx_api_v1() noexcept;

}  // namespace cuda_core

