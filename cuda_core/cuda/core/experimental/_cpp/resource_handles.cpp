// SPDX-FileCopyrightText: Copyright (c) 2024-2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
//
// SPDX-License-Identifier: Apache-2.0

#include "resource_handles.hpp"
#include <cuda.h>

namespace cuda_core {

// ============================================================================
// Context Handles
// ============================================================================

ContextHandle create_context_handle_ref(CUcontext ctx) {
    // Creates a non-owning handle that references an existing context
    // (e.g., primary context managed by CUDA driver)

    // Allocate the box containing the context resource
    ContextBox* box = new ContextBox();
    box->resource = ctx;

    // Use default deleter - it will delete the box, but not touch the CUcontext
    // CUcontext lifetime is managed externally (e.g., by CUDA driver)
    std::shared_ptr<const ContextBox> box_ptr(box);

    // Use aliasing constructor to create handle that exposes only CUcontext
    // The handle's reference count is tied to box_ptr, but it points to &box_ptr->resource
    return ContextHandle(box_ptr, &box_ptr->resource);
}

// TODO: Future owning handle for cuCtxCreate/cuCtxDestroy
// ContextHandle create_context_handle(CUdevice dev, unsigned int flags) { ... }

// ============================================================================
// Stream Handles
// ============================================================================

// TODO: Implement StreamH create_stream_handle(...) when Stream gets handle support

// ============================================================================
// Event Handles
// ============================================================================

// TODO: Implement EventH create_event_handle(...) when Event gets handle support

// ============================================================================
// Device Pointer Handles
// ============================================================================

// TODO: Implement DevicePtrH create_deviceptr_handle(...) when DevicePtr gets handle support

// ============================================================================
// Memory Pool Handles
// ============================================================================

// TODO: Implement MemPoolH create_mempool_handle(...) when MemPool gets handle support

}  // namespace cuda_core
