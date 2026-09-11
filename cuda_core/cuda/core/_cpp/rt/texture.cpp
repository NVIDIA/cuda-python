// SPDX-FileCopyrightText: Copyright (c) 2024-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
//
// SPDX-License-Identifier: Apache-2.0

#include "py.hpp"
#include "api.hpp"
#include "context_scope.hpp"
#include "driver_api.hpp"
#include "error.hpp"
#include "internal.hpp"
#include <cstddef>
#include <memory>
#include <utility>

namespace cuda_core::rt {

using namespace detail;

// ============================================================================
// Graphics Resource Handles
// ============================================================================

namespace {
struct GraphicsResourceBox {
    CUgraphicsResource resource;
};
}  // namespace

GraphicsResourceHandle create_graphics_resource_handle(CUgraphicsResource resource) {
    auto box = std::shared_ptr<const GraphicsResourceBox>(
        new GraphicsResourceBox{resource},
        [](const GraphicsResourceBox* b) {
            GILReleaseGuard gil;
            pw_cuGraphicsUnregisterResource(b->resource);
            delete b;
        }
    );
    return GraphicsResourceHandle(box, &box->resource);
}

// ============================================================================
// Array / mipmapped-array / texture / surface handles (PR #467)
// ============================================================================

namespace {
struct ArrayBox {
    CUarray resource;
    // Non-null only for a mipmap-level view: keeps the parent mipmap (the real
    // owner of the level's storage) alive for as long as the level is held.
    MipmappedArrayHandle h_parent;
    ContextHandle h_context;
};

struct MipmappedArrayBox {
    CUmipmappedArray resource;
    ContextHandle h_context;
};

// Texture and surface objects are per-context pool indices. Destroying one
// with the wrong context current can silently succeed without freeing it or
// can free an unrelated object, so destruction must enter the creating
// context. Handle-based resources resolve their own context and must not.
struct TexObjectBox {
    // Tagged so TexObjectHandle is a distinct C++ type from DevicePtrHandle /
    // SurfObjectHandle (all wrap `unsigned long long`).
    TexObjectValue resource;
    // Type-erased backing dependency (OpaqueArrayHandle / MipmappedArrayHandle /
    // DevicePtrHandle). The texture's resource is a union; we only need to keep
    // whichever backing it was built from alive, never to dereference it.
    std::shared_ptr<const void> h_backing;
    ContextHandle h_context;
};

struct SurfObjectBox {
    SurfObjectValue resource;
    OpaqueArrayHandle h_array;  // surfaces are always array-backed
    ContextHandle h_context;
};

// Recover an array's owning box from its aliased resource pointer.
const ArrayBox* get_box(const OpaqueArrayHandle& h) noexcept {
    const CUarray* p = h.get();
    return reinterpret_cast<const ArrayBox*>(
        reinterpret_cast<const char*>(p) - offsetof(ArrayBox, resource));
}

// Recover a mipmapped array's owning box from its aliased resource pointer.
const MipmappedArrayBox* get_box(const MipmappedArrayHandle& h) noexcept {
    const CUmipmappedArray* p = h.get();
    return reinterpret_cast<const MipmappedArrayBox*>(
        reinterpret_cast<const char*>(p)
        - offsetof(MipmappedArrayBox, resource));
}

// Wrap an array with shared owning-destruction behavior.
static OpaqueArrayHandle wrap_array_owned(CUarray arr, ContextHandle h_context) {
    auto box = std::shared_ptr<const ArrayBox>(
        new ArrayBox{arr, {}, std::move(h_context)},
        [](const ArrayBox* b) {
            GILReleaseGuard gil;
            pw_cuArrayDestroy(b->resource);
            delete b;
        }
    );
    return OpaqueArrayHandle(box, &box->resource);
}

}  // namespace

OpaqueArrayHandle create_array_handle(const ContextHandle& h_context, const CUDA_ARRAY3D_DESCRIPTOR& desc) {
    GILReleaseGuard gil;
    CUarray arr = nullptr;
    err = invoke_in_context_or_undo(
        h_context,
        [&]() noexcept { return p_cuArray3DCreate(&arr, &desc); },
        [&]() noexcept { pw_cuArrayDestroy(arr); },
        /*undo_requires_target_context=*/false);
    if (err != CUDA_SUCCESS) {
        return {};
    }
    return wrap_array_owned(arr, h_context);
}

OpaqueArrayHandle create_array_handle_ref(CUarray arr) {
    if (!arr) {
        return {};
    }
    auto box = std::make_shared<const ArrayBox>(ArrayBox{arr, {}, {}});
    return OpaqueArrayHandle(box, &box->resource);
}

OpaqueArrayHandle create_array_handle_owning(CUarray arr) {
    if (!arr) {
        return {};
    }
    return wrap_array_owned(arr, {});
}

// Return the context retained by an array handle.
ContextHandle get_array_context(const OpaqueArrayHandle& h) noexcept {
    return h ? get_box(h)->h_context : ContextHandle{};
}

OpaqueArrayHandle create_array_level_handle(const MipmappedArrayHandle& h_mip, unsigned int level) {
    GILReleaseGuard gil;
    CUarray arr;
    ContextHandle h_context = h_mip ? get_box(h_mip)->h_context : ContextHandle{};
    if (CUDA_SUCCESS != (err = p_cuMipmappedArrayGetLevel(&arr, as_cu(h_mip), level))) {
        return {};
    }
    // Non-owning level view: storage belongs to the mipmap. Embed the mipmap
    // handle so the parent outlives this level; the deleter does not destroy.
    auto box = std::shared_ptr<const ArrayBox>(
        new ArrayBox{arr, h_mip, h_context},
        [](const ArrayBox* b) { delete b; }
    );
    return OpaqueArrayHandle(box, &box->resource);
}

MipmappedArrayHandle create_mipmapped_array_handle(const ContextHandle& h_context,
                                                   const CUDA_ARRAY3D_DESCRIPTOR& desc,
                                                   unsigned int num_levels) {
    GILReleaseGuard gil;
    CUmipmappedArray mip = nullptr;
    err = invoke_in_context_or_undo(
        h_context,
        [&]() noexcept { return p_cuMipmappedArrayCreate(&mip, &desc, num_levels); },
        [&]() noexcept { pw_cuMipmappedArrayDestroy(mip); },
        /*undo_requires_target_context=*/false);
    if (err != CUDA_SUCCESS) {
        return {};
    }
    auto box = std::shared_ptr<const MipmappedArrayBox>(
        new MipmappedArrayBox{mip, h_context},
        [](const MipmappedArrayBox* b) {
            GILReleaseGuard gil;
            pw_cuMipmappedArrayDestroy(b->resource);
            delete b;
        }
    );
    return MipmappedArrayHandle(box, &box->resource);
}

// Return the context retained by a mipmapped array handle.
ContextHandle get_mipmapped_array_context(const MipmappedArrayHandle& h) noexcept {
    return h ? get_box(h)->h_context : ContextHandle{};
}

namespace {
TexObjectHandle make_tex_object_handle(const CUDA_RESOURCE_DESC& res,
                                       const CUDA_TEXTURE_DESC& tex,
                                       std::shared_ptr<const void> h_backing,
                                       const ContextHandle& h_context) {
    GILReleaseGuard gil;
    CUtexObject obj = 0;
    err = invoke_in_context_or_undo(
        h_context,
        [&]() noexcept { return p_cuTexObjectCreate(&obj, &res, &tex, nullptr); },
        [&]() noexcept { pw_cuTexObjectDestroy(obj); },
        /*undo_requires_target_context=*/true);
    if (err != CUDA_SUCCESS) {
        return {};
    }
    auto box = std::shared_ptr<const TexObjectBox>(
        new TexObjectBox{TexObjectValue{obj}, std::move(h_backing), h_context},
        [](const TexObjectBox* b) {
            GILReleaseGuard gil;
            cleanup_in_context(b->h_context, "cuTexObjectDestroy", [&]() noexcept {
                return p_cuTexObjectDestroy(b->resource.raw);
            });
            delete b;
        }
    );
    return TexObjectHandle(box, &box->resource);
}
}  // namespace

TexObjectHandle create_tex_object_handle_array(const ContextHandle& h_context,
                                               const CUDA_RESOURCE_DESC& res,
                                               const CUDA_TEXTURE_DESC& tex,
                                               const OpaqueArrayHandle& h_backing) {
    return make_tex_object_handle(res, tex, h_backing, h_context);
}

TexObjectHandle create_tex_object_handle_mipmap(const ContextHandle& h_context,
                                                const CUDA_RESOURCE_DESC& res,
                                                const CUDA_TEXTURE_DESC& tex,
                                                const MipmappedArrayHandle& h_backing) {
    return make_tex_object_handle(res, tex, h_backing, h_context);
}

TexObjectHandle create_tex_object_handle_linear(const ContextHandle& h_context,
                                                const CUDA_RESOURCE_DESC& res,
                                                const CUDA_TEXTURE_DESC& tex,
                                                const DevicePtrHandle& h_backing) {
    return make_tex_object_handle(res, tex, h_backing, h_context);
}

SurfObjectHandle create_surf_object_handle(const ContextHandle& h_context,
                                           const CUDA_RESOURCE_DESC& res,
                                           const OpaqueArrayHandle& h_backing) {
    GILReleaseGuard gil;
    CUsurfObject obj = 0;
    err = invoke_in_context_or_undo(
        h_context,
        [&]() noexcept { return p_cuSurfObjectCreate(&obj, &res); },
        [&]() noexcept { pw_cuSurfObjectDestroy(obj); },
        /*undo_requires_target_context=*/true);
    if (err != CUDA_SUCCESS) {
        return {};
    }
    auto box = std::shared_ptr<const SurfObjectBox>(
        new SurfObjectBox{SurfObjectValue{obj}, h_backing, h_context},
        [](const SurfObjectBox* b) {
            GILReleaseGuard gil;
            cleanup_in_context(b->h_context, "cuSurfObjectDestroy", [&]() noexcept {
                return p_cuSurfObjectDestroy(b->resource.raw);
            });
            delete b;
        }
    );
    return SurfObjectHandle(box, &box->resource);
}

}  // namespace cuda_core::rt
