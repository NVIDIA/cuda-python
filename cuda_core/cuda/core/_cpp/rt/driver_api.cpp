// SPDX-FileCopyrightText: Copyright (c) 2024-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
//
// SPDX-License-Identifier: Apache-2.0

#include "driver_api.hpp"
#include <cstddef>
#include <cuda.h>

namespace cuda_core::rt {

// The pointers. Null until ensure_fn_table() fills the table. See driver_api.hpp.
#define CUDA_CORE_DEFINE_DRIVER_FN(name, introduced) decltype(&name) p_##name = nullptr;
CUDA_CORE_DRIVER_FUNCTIONS(CUDA_CORE_DEFINE_DRIVER_FN)
#undef CUDA_CORE_DEFINE_DRIVER_FN

decltype(&nvrtcDestroyProgram) p_nvrtcDestroyProgram = nullptr;
NvvmDestroyProgramFn p_nvvmDestroyProgram = nullptr;
NvJitLinkDestroyFn p_nvJitLinkDestroy = nullptr;

namespace {

#define CUDA_CORE_STR(x) #x
#define CUDA_CORE_XSTR(x) CUDA_CORE_STR(x)

// "__" + the macro-expanded symbol that cuda.h maps the public name to, which
// is how cuda-bindings keys its table. #name is the public name, unexpanded.
#define CUDA_CORE_DRIVER_FN_ENTRY(name, introduced) \
    {"__" CUDA_CORE_XSTR(name), #name, reinterpret_cast<void**>(&p_##name), introduced},

const FnEntry driver_entries[] = {CUDA_CORE_DRIVER_FUNCTIONS(CUDA_CORE_DRIVER_FN_ENTRY)};
#undef CUDA_CORE_DRIVER_FN_ENTRY

const FnEntry nvrtc_entries[] = {
    {"__nvrtcDestroyProgram", "nvrtcDestroyProgram", reinterpret_cast<void**>(&p_nvrtcDestroyProgram), 0},
};
const FnEntry nvvm_entries[] = {
    {"__nvvmDestroyProgram", "nvvmDestroyProgram", reinterpret_cast<void**>(&p_nvvmDestroyProgram), 0},
};
const FnEntry nvjitlink_entries[] = {
    {"__nvJitLinkDestroy", "nvJitLinkDestroy", reinterpret_cast<void**>(&p_nvJitLinkDestroy), 0},
};

}  // namespace

const FnEntry* fn_table_entries(FnTable table, std::size_t* count) noexcept {
    switch (table) {
        case FnTable::driver:
            *count = sizeof(driver_entries) / sizeof(driver_entries[0]);
            return driver_entries;
        case FnTable::nvrtc:
            *count = sizeof(nvrtc_entries) / sizeof(nvrtc_entries[0]);
            return nvrtc_entries;
        case FnTable::nvvm:
            *count = sizeof(nvvm_entries) / sizeof(nvvm_entries[0]);
            return nvvm_entries;
        case FnTable::nvjitlink:
            *count = sizeof(nvjitlink_entries) / sizeof(nvjitlink_entries[0]);
            return nvjitlink_entries;
    }
    *count = 0;
    return nullptr;
}

}  // namespace cuda_core::rt
