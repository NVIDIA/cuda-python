# SPDX-FileCopyrightText: Copyright (c) 2024-2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

from libc.stdint cimport uint32_t
from libc.stddef cimport size_t

from cuda.bindings cimport cydriver
from cuda.core.experimental._resource_handles cimport (
    ContextHandle,
    DevicePtrHandle,
    EventHandle,
    MemoryPoolHandle,
    StreamHandle,
)


cdef extern from "_cpp/resource_handles_cxx_api.hpp" namespace "cuda_core":
    cdef struct ResourceHandlesCxxApiV1:
        uint32_t abi_version
        uint32_t struct_size

        # Thread-local error handling
        cydriver.CUresult (*get_last_error)() nogil
        cydriver.CUresult (*peek_last_error)() nogil
        void (*clear_last_error)() nogil

        # Context handles
        ContextHandle (*create_context_handle_ref)(cydriver.CUcontext ctx) nogil
        ContextHandle (*get_primary_context)(int device_id) nogil
        ContextHandle (*get_current_context)() nogil

        # Stream handles
        StreamHandle (*create_stream_handle)(ContextHandle h_ctx, unsigned int flags, int priority) nogil
        StreamHandle (*create_stream_handle_ref)(cydriver.CUstream stream) nogil
        StreamHandle (*create_stream_handle_with_owner)(cydriver.CUstream stream, object owner)
        StreamHandle (*get_legacy_stream)() nogil
        StreamHandle (*get_per_thread_stream)() nogil

        # Event handles
        EventHandle (*create_event_handle)(ContextHandle h_ctx, unsigned int flags) nogil
        EventHandle (*create_event_handle_noctx)(unsigned int flags) nogil
        EventHandle (*create_event_handle_ipc)(const cydriver.CUipcEventHandle& ipc_handle) nogil

        # Memory pool handles
        MemoryPoolHandle (*create_mempool_handle)(const cydriver.CUmemPoolProps& props) nogil
        MemoryPoolHandle (*create_mempool_handle_ref)(cydriver.CUmemoryPool pool) nogil
        MemoryPoolHandle (*get_device_mempool)(int device_id) nogil
        MemoryPoolHandle (*create_mempool_handle_ipc)(int fd, cydriver.CUmemAllocationHandleType handle_type) nogil

        # Device pointer handles
        DevicePtrHandle (*deviceptr_alloc_from_pool)(
            size_t size,
            MemoryPoolHandle h_pool,
            StreamHandle h_stream) nogil
        DevicePtrHandle (*deviceptr_alloc_async)(size_t size, StreamHandle h_stream) nogil
        DevicePtrHandle (*deviceptr_alloc)(size_t size) nogil
        DevicePtrHandle (*deviceptr_alloc_host)(size_t size) nogil
        DevicePtrHandle (*deviceptr_create_ref)(cydriver.CUdeviceptr ptr) nogil
        DevicePtrHandle (*deviceptr_create_with_owner)(cydriver.CUdeviceptr ptr, object owner)
        DevicePtrHandle (*deviceptr_import_ipc)(
            MemoryPoolHandle h_pool,
            const void* export_data,
            StreamHandle h_stream) nogil
        StreamHandle (*deallocation_stream)(const DevicePtrHandle& h) nogil
        void (*set_deallocation_stream)(const DevicePtrHandle& h, StreamHandle h_stream) nogil

    const ResourceHandlesCxxApiV1* get_resource_handles_cxx_api_v1() nogil
