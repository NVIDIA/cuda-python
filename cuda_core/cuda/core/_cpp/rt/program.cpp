// SPDX-FileCopyrightText: Copyright (c) 2024-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
//
// SPDX-License-Identifier: Apache-2.0

#include "py.hpp"
#include "api.hpp"
#include "driver_api.hpp"
#include "error.hpp"
#include "internal.hpp"
#include <cstddef>
#include <memory>
#include <utility>

namespace cuda_core::rt {

using namespace detail;

// ============================================================================
// Library Handles
// ============================================================================

namespace {
struct LibraryBox {
    CUlibrary resource;
};
}  // namespace

LibraryHandle create_library_handle_from_file(const char* path) {
    GILReleaseGuard gil;
    CUlibrary library;
    if (CUDA_SUCCESS != (err = p_cuLibraryLoadFromFile(&library, path, nullptr, nullptr, 0, nullptr, nullptr, 0))) {
        return {};
    }

    auto box = std::shared_ptr<const LibraryBox>(
        new LibraryBox{library},
        [](const LibraryBox* b) {
            GILReleaseGuard gil;
            // TODO: re-enable once LibraryBox tracks its owning context
            // p_cuLibraryUnload(b->resource);
            delete b;
        }
    );
    return LibraryHandle(box, &box->resource);
}

LibraryHandle create_library_handle_from_data(const void* data) {
    GILReleaseGuard gil;
    CUlibrary library;
    if (CUDA_SUCCESS != (err = p_cuLibraryLoadData(&library, data, nullptr, nullptr, 0, nullptr, nullptr, 0))) {
        return {};
    }

    auto box = std::shared_ptr<const LibraryBox>(
        new LibraryBox{library},
        [](const LibraryBox* b) {
            GILReleaseGuard gil;
            // TODO: re-enable once LibraryBox tracks its owning context
            // p_cuLibraryUnload(b->resource);
            delete b;
        }
    );
    return LibraryHandle(box, &box->resource);
}

LibraryHandle create_library_handle_ref(CUlibrary library) {
    auto box = std::make_shared<const LibraryBox>(LibraryBox{library});
    return LibraryHandle(box, &box->resource);
}

// ============================================================================
// Kernel Handles
// ============================================================================

namespace {
struct KernelBox {
    CUkernel resource;
    LibraryHandle h_library;
};
}  // namespace

static const KernelBox* get_box(const KernelHandle& h) {
    const CUkernel* p = h.get();
    return reinterpret_cast<const KernelBox*>(
        reinterpret_cast<const char*>(p) - offsetof(KernelBox, resource)
    );
}

// See REGISTRY_DESIGN.md (Level 1: Driver Handle -> Resource Handle)
static HandleRegistry<CUkernel, KernelHandle> kernel_registry;

KernelHandle create_kernel_handle(const LibraryHandle& h_library, const char* name) {
    GILReleaseGuard gil;
    CUkernel kernel;
    if (CUDA_SUCCESS != (err = p_cuLibraryGetKernel(&kernel, *h_library, name))) {
        return {};
    }

    auto box = std::make_shared<const KernelBox>(KernelBox{kernel, h_library});
    KernelHandle h(box, &box->resource);
    kernel_registry.register_handle(kernel, h);
    return h;
}

KernelHandle create_kernel_handle_ref(CUkernel kernel) {
    if (auto h = kernel_registry.lookup(kernel)) {
        return h;
    }
    auto box = std::make_shared<const KernelBox>(KernelBox{kernel, {}});
    return KernelHandle(box, &box->resource);
}

LibraryHandle get_kernel_library(const KernelHandle& h) noexcept {
    if (!h) return {};
    return get_box(h)->h_library;
}

// ============================================================================
// NVRTC Program Handles
// ============================================================================

namespace {
struct NvrtcProgramBox {
    nvrtcProgram resource;
};
}  // namespace

NvrtcProgramHandle create_nvrtc_program_handle(nvrtcProgram prog) {
    auto box = std::shared_ptr<NvrtcProgramBox>(
        new NvrtcProgramBox{prog},
        [](NvrtcProgramBox* b) {
            // Note: nvrtcDestroyProgram takes nvrtcProgram* and nulls it,
            // but we're deleting the box anyway so nulling is harmless.
            if (p_nvrtcDestroyProgram) {
                GILReleaseGuard gil;
                pw_nvrtcDestroyProgram(&b->resource);
            }
            delete b;
        }
    );
    return NvrtcProgramHandle(box, &box->resource);
}

NvrtcProgramHandle create_nvrtc_program_handle_ref(nvrtcProgram prog) {
    auto box = std::make_shared<NvrtcProgramBox>(NvrtcProgramBox{prog});
    return NvrtcProgramHandle(box, &box->resource);
}

// ============================================================================
// NVVM Program Handles
// ============================================================================

namespace {
struct NvvmProgramBox {
    NvvmProgramValue resource;
};
}  // namespace

NvvmProgramHandle create_nvvm_program_handle(nvvmProgram prog) {
    auto box = std::shared_ptr<NvvmProgramBox>(
        new NvvmProgramBox{{prog}},
        [](NvvmProgramBox* b) {
            // Note: nvvmDestroyProgram takes nvvmProgram* and nulls it,
            // but we're deleting the box anyway so nulling is harmless.
            // If NVVM is not available, the function pointer is null.
            if (p_nvvmDestroyProgram) {
                GILReleaseGuard gil;
                pw_nvvmDestroyProgram(&b->resource.raw);
            }
            delete b;
        }
    );
    return NvvmProgramHandle(box, &box->resource);
}

NvvmProgramHandle create_nvvm_program_handle_ref(nvvmProgram prog) {
    auto box = std::make_shared<NvvmProgramBox>(NvvmProgramBox{{prog}});
    return NvvmProgramHandle(box, &box->resource);
}

// ============================================================================
// nvJitLink Handles
// ============================================================================

namespace {
struct NvJitLinkBox {
    NvJitLinkValue resource;
};
}  // namespace

NvJitLinkHandle create_nvjitlink_handle(nvJitLink_t handle) {
    auto box = std::shared_ptr<NvJitLinkBox>(
        new NvJitLinkBox{{handle}},
        [](NvJitLinkBox* b) {
            // Note: nvJitLinkDestroy takes nvJitLinkHandle* and nulls it,
            // but we're deleting the box anyway so nulling is harmless.
            // If nvJitLink is not available, the function pointer is null.
            if (p_nvJitLinkDestroy) {
                GILReleaseGuard gil;
                pw_nvJitLinkDestroy(&b->resource.raw);
            }
            delete b;
        }
    );
    return NvJitLinkHandle(box, &box->resource);
}

NvJitLinkHandle create_nvjitlink_handle_ref(nvJitLink_t handle) {
    auto box = std::make_shared<NvJitLinkBox>(NvJitLinkBox{{handle}});
    return NvJitLinkHandle(box, &box->resource);
}

// ============================================================================
// cuLink Handles
// ============================================================================

namespace {
struct CuLinkBox {
    CUlinkState resource;
};
}  // namespace

CuLinkHandle create_culink_handle(CUlinkState state) {
    auto box = std::shared_ptr<CuLinkBox>(
        new CuLinkBox{state},
        [](CuLinkBox* b) {
            // cuLinkDestroy takes CUlinkState by value (not pointer).
            if (p_cuLinkDestroy) {
                GILReleaseGuard gil;
                pw_cuLinkDestroy(b->resource);
            }
            delete b;
        }
    );
    return CuLinkHandle(box, &box->resource);
}

CuLinkHandle create_culink_handle_ref(CUlinkState state) {
    auto box = std::make_shared<CuLinkBox>(CuLinkBox{state});
    return CuLinkHandle(box, &box->resource);
}

}  // namespace cuda_core::rt
