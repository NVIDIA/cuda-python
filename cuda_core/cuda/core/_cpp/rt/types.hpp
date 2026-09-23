// SPDX-FileCopyrightText: Copyright (c) 2024-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <cuda.h>
#include <nvrtc.h>
#include <cstdint>
#include <memory>

// Forward declaration for NVVM - avoids nvvm.h dependency
// Use void* to match cuda.bindings.cynvvm's typedef
using nvvmProgram = void*;

// Forward declaration for nvJitLink - avoids nvJitLink.h dependency
// Use void* to match cuda.bindings.cynvjitlink's typedef
using nvJitLink_t = void*;

namespace cuda_core::rt {

// ============================================================================
// TaggedHandle - make void*-based handle types distinct for overloading
//
// Both nvvmProgram and nvJitLink_t are void*, so shared_ptr<const void*>
// would be the same C++ type for both. TaggedHandle<T, Tag> wraps the raw
// value with a unique tag type, making each shared_ptr type distinct.
// ============================================================================

template<typename T, int Tag>
struct TaggedHandle {
    T raw;
};

using NvvmProgramValue = TaggedHandle<nvvmProgram, 0>;
using NvJitLinkValue = TaggedHandle<nvJitLink_t, 1>;

// CUtexObject, CUsurfObject and CUdeviceptr are all `unsigned long long`, so
// shared_ptr<const CUtexObject> et al. would be the *same* C++ type as
// DevicePtrHandle (and each other), collapsing the as_cu/as_intptr/as_py
// overload sets. Tag them to keep each handle type distinct, exactly as the
// NVVM / nvJitLink handles above do.
using TexObjectValue = TaggedHandle<CUtexObject, 2>;
using SurfObjectValue = TaggedHandle<CUsurfObject, 3>;

// ============================================================================
// Handle type aliases - expose only the raw CUDA resource
// ============================================================================

using ContextHandle = std::shared_ptr<const CUcontext>;
using GreenCtxHandle = std::shared_ptr<const CUgreenCtx>;
using StreamHandle = std::shared_ptr<const CUstream>;
using EventHandle = std::shared_ptr<const CUevent>;
using MemoryPoolHandle = std::shared_ptr<const CUmemoryPool>;
using LibraryHandle = std::shared_ptr<const CUlibrary>;
using KernelHandle = std::shared_ptr<const CUkernel>;
using GraphHandle = std::shared_ptr<const CUgraph>;
using GraphExecHandle = std::shared_ptr<const CUgraphExec>;
using GraphNodeHandle = std::shared_ptr<const CUgraphNode>;
using GraphicsResourceHandle = std::shared_ptr<const CUgraphicsResource>;
using NvrtcProgramHandle = std::shared_ptr<const nvrtcProgram>;
using NvvmProgramHandle = std::shared_ptr<const NvvmProgramValue>;
using NvJitLinkHandle = std::shared_ptr<const NvJitLinkValue>;
using CuLinkHandle = std::shared_ptr<const CUlinkState>;
using FileDescriptorHandle = std::shared_ptr<const int>;
using OpaqueArrayHandle = std::shared_ptr<const CUarray>;
using MipmappedArrayHandle = std::shared_ptr<const CUmipmappedArray>;
using TexObjectHandle = std::shared_ptr<const TexObjectValue>;
using SurfObjectHandle = std::shared_ptr<const SurfObjectValue>;

using DevicePtrHandle = std::shared_ptr<const CUdeviceptr>;

// Type-erased shared owner of an attached resource. Typed handles such as
// EventHandle and KernelHandle convert to OpaqueHandle by assignment, reusing
// their existing control block; the helpers below build OpaqueHandles for the
// two cases that need a custom deleter.
using OpaqueHandle = std::shared_ptr<const void>;

struct PreparedAttachmentState;
using PreparedAttachmentRollback =
    void (*)(PreparedAttachmentState*) noexcept;
struct PreparedAttachmentDeleter {
    PreparedAttachmentRollback rollback = nullptr;

    void operator()(PreparedAttachmentState* state) const noexcept {
        rollback(state);
    }
};
using PreparedAttachment =
    std::unique_ptr<PreparedAttachmentState, PreparedAttachmentDeleter>;

struct PreparedChildGraphUpdateState;
// Opaque unpublished hierarchy transaction; releasing it discards staged
// metadata unless graph_commit_child_graph_update publishes the replacement.
using PreparedChildGraphUpdate =
    std::shared_ptr<PreparedChildGraphUpdateState>;

struct PreparedExecAttachmentState;
using PreparedExecAttachmentRollback =
    void (*)(PreparedExecAttachmentState*) noexcept;
struct PreparedExecAttachmentDeleter {
    PreparedExecAttachmentRollback rollback = nullptr;

    void operator()(PreparedExecAttachmentState* state) const noexcept {
        rollback(state);
    }
};
// Opaque append transaction. Releasing it rolls back newly appended owners
// unless graph_commit_exec_attachment has kept them.
using PreparedExecAttachment =
    std::unique_ptr<PreparedExecAttachmentState, PreparedExecAttachmentDeleter>;

// ============================================================================
// Overloaded helper functions to extract raw resources from handles
// ============================================================================

// as_cu() - extract the raw CUDA handle
inline CUcontext as_cu(const ContextHandle& h) noexcept {
    return h ? *h : nullptr;
}

inline CUgreenCtx as_cu(const GreenCtxHandle& h) noexcept {
    return h ? *h : nullptr;
}

inline CUstream as_cu(const StreamHandle& h) noexcept {
    return h ? *h : nullptr;
}

inline CUevent as_cu(const EventHandle& h) noexcept {
    return h ? *h : nullptr;
}

inline CUmemoryPool as_cu(const MemoryPoolHandle& h) noexcept {
    return h ? *h : nullptr;
}

inline CUdeviceptr as_cu(const DevicePtrHandle& h) noexcept {
    return h ? *h : 0;
}

inline CUlibrary as_cu(const LibraryHandle& h) noexcept {
    return h ? *h : nullptr;
}

inline CUmodule as_cu(const CUmodule& h) noexcept {
    return h;
}

inline CUkernel as_cu(const KernelHandle& h) noexcept {
    return h ? *h : nullptr;
}

inline CUgraph as_cu(const GraphHandle& h) noexcept {
    return h ? *h : nullptr;
}

inline CUgraphExec as_cu(const GraphExecHandle& h) noexcept {
    return h ? *h : nullptr;
}

inline CUgraphNode as_cu(const GraphNodeHandle& h) noexcept {
    return h ? *h : nullptr;
}

inline CUgraphicsResource as_cu(const GraphicsResourceHandle& h) noexcept {
    return h ? *h : nullptr;
}

inline nvrtcProgram as_cu(const NvrtcProgramHandle& h) noexcept {
    return h ? *h : nullptr;
}

inline nvvmProgram as_cu(const NvvmProgramHandle& h) noexcept {
    return h ? h->raw : nullptr;
}

inline nvJitLink_t as_cu(const NvJitLinkHandle& h) noexcept {
    return h ? h->raw : nullptr;
}

inline CUlinkState as_cu(const CuLinkHandle& h) noexcept {
    return h ? *h : nullptr;
}

inline CUarray as_cu(const OpaqueArrayHandle& h) noexcept {
    return h ? *h : nullptr;
}

inline CUmipmappedArray as_cu(const MipmappedArrayHandle& h) noexcept {
    return h ? *h : nullptr;
}

// CUtexObject / CUsurfObject are integer-valued (like CUdeviceptr); null is 0.
// The raw value lives in the tagged wrapper's `raw` field.
inline CUtexObject as_cu(const TexObjectHandle& h) noexcept {
    return h ? h->raw : 0;
}

inline CUsurfObject as_cu(const SurfObjectHandle& h) noexcept {
    return h ? h->raw : 0;
}

// as_intptr() - extract handle as intptr_t for Python interop
// Using signed intptr_t per C standard convention and issue #1342
inline std::intptr_t as_intptr(const ContextHandle& h) noexcept {
    return reinterpret_cast<std::intptr_t>(as_cu(h));
}

inline std::intptr_t as_intptr(const GreenCtxHandle& h) noexcept {
    return reinterpret_cast<std::intptr_t>(as_cu(h));
}

inline std::intptr_t as_intptr(const StreamHandle& h) noexcept {
    return reinterpret_cast<std::intptr_t>(as_cu(h));
}

inline std::intptr_t as_intptr(const EventHandle& h) noexcept {
    return reinterpret_cast<std::intptr_t>(as_cu(h));
}

inline std::intptr_t as_intptr(const MemoryPoolHandle& h) noexcept {
    return reinterpret_cast<std::intptr_t>(as_cu(h));
}

inline std::intptr_t as_intptr(const DevicePtrHandle& h) noexcept {
    return static_cast<std::intptr_t>(as_cu(h));
}

inline std::intptr_t as_intptr(const LibraryHandle& h) noexcept {
    return reinterpret_cast<std::intptr_t>(as_cu(h));
}

inline std::intptr_t as_intptr(const CUmodule& h) noexcept {
    return reinterpret_cast<std::intptr_t>(as_cu(h));
}

inline std::intptr_t as_intptr(const KernelHandle& h) noexcept {
    return reinterpret_cast<std::intptr_t>(as_cu(h));
}

inline std::intptr_t as_intptr(const GraphHandle& h) noexcept {
    return reinterpret_cast<std::intptr_t>(as_cu(h));
}

inline std::intptr_t as_intptr(const GraphExecHandle& h) noexcept {
    return reinterpret_cast<std::intptr_t>(as_cu(h));
}

inline std::intptr_t as_intptr(const GraphNodeHandle& h) noexcept {
    return reinterpret_cast<std::intptr_t>(as_cu(h));
}

inline std::intptr_t as_intptr(const GraphicsResourceHandle& h) noexcept {
    return reinterpret_cast<std::intptr_t>(as_cu(h));
}

inline std::intptr_t as_intptr(const NvrtcProgramHandle& h) noexcept {
    return reinterpret_cast<std::intptr_t>(as_cu(h));
}

inline std::intptr_t as_intptr(const NvvmProgramHandle& h) noexcept {
    return reinterpret_cast<std::intptr_t>(as_cu(h));
}

inline std::intptr_t as_intptr(const NvJitLinkHandle& h) noexcept {
    return reinterpret_cast<std::intptr_t>(as_cu(h));
}

inline std::intptr_t as_intptr(const CuLinkHandle& h) noexcept {
    return reinterpret_cast<std::intptr_t>(as_cu(h));
}

inline std::intptr_t as_intptr(const FileDescriptorHandle& h) noexcept {
    return h ? static_cast<std::intptr_t>(*h) : -1;
}

inline std::intptr_t as_intptr(const OpaqueArrayHandle& h) noexcept {
    return reinterpret_cast<std::intptr_t>(as_cu(h));
}

inline std::intptr_t as_intptr(const MipmappedArrayHandle& h) noexcept {
    return reinterpret_cast<std::intptr_t>(as_cu(h));
}

inline std::intptr_t as_intptr(const TexObjectHandle& h) noexcept {
    return static_cast<std::intptr_t>(as_cu(h));
}

inline std::intptr_t as_intptr(const SurfObjectHandle& h) noexcept {
    return static_cast<std::intptr_t>(as_cu(h));
}

}  // namespace cuda_core::rt
