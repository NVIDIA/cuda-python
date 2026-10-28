// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
//
// SPDX-License-Identifier: Apache-2.0

#include "py.hpp"
#include "api.hpp"
#include "driver_api.hpp"
#include "internal.hpp"
#include <new>
#include <utility>

namespace cuda_core::rt {

namespace {

struct HostLaunchRecord : detail::DeferredCleanupItem {
    CUhostFn fn;
    void *user_data;
    OpaqueHandle fn_owner;
    OpaqueHandle data_owner;

    HostLaunchRecord(CUhostFn fn, void *user_data, OpaqueHandle fn_owner, OpaqueHandle data_owner) noexcept
        : fn(fn), user_data(user_data), fn_owner(std::move(fn_owner)), data_owner(std::move(data_owner))
    {
    }
};

void
host_launch_trampoline(void *p) noexcept
{
    auto *rec = static_cast<HostLaunchRecord *>(p);
    rec->fn(rec->user_data);
    // Owners may run Python finalizers or CUDA calls, so release them on the main thread.
    detail::enqueue_cleanup(static_cast<detail::DeferredCleanupItem *>(rec));
}

}  // namespace

CUresult
launch_host_function(
    CUstream stream, CUhostFn fn, void *user_data, OpaqueHandle fn_owner, OpaqueHandle data_owner
) noexcept
{
    retry_deferred_cleanup();
    auto *record = new (std::nothrow) HostLaunchRecord(fn, user_data, std::move(fn_owner), std::move(data_owner));
    if (record == nullptr) {
        return CUDA_ERROR_OUT_OF_MEMORY;
    }
    CUresult status = DRIVER_CALL(cuLaunchHostFunc, stream, &host_launch_trampoline, record);
    if (status != CUDA_SUCCESS) {
        delete record;
        return status;
    }
    return CUDA_SUCCESS;
}

}  // namespace cuda_core::rt
