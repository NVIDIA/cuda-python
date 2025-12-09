// SPDX-FileCopyrightText: Copyright (c) 2024-2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <memory>
#include <cuda.h>

namespace cuda_core {

// Handle type aliases - expose only the raw CUDA resource
using ContextHandle = std::shared_ptr<const CUcontext>;

// Function to create a non-owning context handle (references existing context).
ContextHandle create_context_handle_ref(CUcontext ctx);

// ============================================================================
// Context acquisition functions (pure C++, nogil-safe)
// ============================================================================

// Get handle to the primary context for a device (with thread-local caching)
// Returns empty handle on error (caller must check)
ContextHandle get_primary_context(int dev_id) noexcept;

// Get handle to the current CUDA context
// Returns empty handle if no context is current (caller must check)
ContextHandle get_current_context() noexcept;

}  // namespace cuda_core
