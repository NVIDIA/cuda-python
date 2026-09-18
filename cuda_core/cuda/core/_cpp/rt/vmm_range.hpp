// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "types.hpp"
#include "internal.hpp"
#include <mutex>
#include <vector>

namespace cuda_core::rt {

// The mappings a VirtualMemoryResource buffer owns, in ascending address
// order, plus every deallocation stream an owner of the range recorded.
// Owned through VmmRangeHandle by the DevicePtrBox of each buffer that maps
// the range (a grow in place creates a second owner at the same base). The
// range deleter synchronizes the recorded streams and then destroys the
// mappings, which unmap, free the reservations and release the allocations
// as their last references go. See VMM_DESIGN.md.
struct VmmRange {
    explicit VmmRange(CUdeviceptr base_) noexcept : base(base_) {}

    CUdeviceptr base;                                  // registry key
    std::vector<VaMappingHandle> mappings;             // contiguous; sum of sizes = range total
    std::mutex mu;                                     // guards `streams` and `stream_dropped`
    std::vector<detail::DeallocationStream> streams;   // forwarded by dying owners, deduplicated
    bool stream_dropped = false;                       // a forward failed for lack of memory
};

namespace detail {
// Record the stream an owner of `range` used for deallocation, so the range
// deleter synchronizes it before it unmaps. Takes range.mu; nothing under the
// lock acquires the GIL. Implemented in virtual_memory.cpp.
void vmm_range_forward_stream(VmmRange& range, const DeallocationStream& stream) noexcept;
}  // namespace detail

}  // namespace cuda_core::rt
