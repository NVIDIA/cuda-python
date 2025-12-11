# SPDX-FileCopyrightText: Copyright (c) 2024-2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

from libc.stdint cimport uintptr_t
from libcpp.memory cimport shared_ptr

from cuda.bindings cimport cydriver

# Declare the C++ namespace and types
cdef extern from "_cpp/resource_handles.hpp" namespace "cuda_core":
    # ========================================================================
    # Context Handle
    # ========================================================================
    ctypedef shared_ptr[const cydriver.CUcontext] ContextHandle

    # Function to create a non-owning context handle (references existing context)
    ContextHandle create_context_handle_ref(cydriver.CUcontext ctx) nogil

    # Context acquisition functions (pure C++, nogil-safe with thread-local caching)
    ContextHandle get_primary_context(int device_id) nogil
    ContextHandle get_current_context() nogil

    # ========================================================================
    # Stream Handle
    # ========================================================================
    ctypedef shared_ptr[const cydriver.CUstream] StreamHandle

    # Create an owning stream handle via cuStreamCreateWithPriority
    # Context handle establishes structural dependency (context outlives stream)
    # Returns empty handle on error (caller must check)
    StreamHandle create_stream_handle(ContextHandle h_ctx, unsigned int flags, int priority) nogil

    # Create a non-owning stream handle (stream NOT destroyed when handle released)
    # Caller is responsible for keeping the stream's context alive
    StreamHandle create_stream_handle_ref(cydriver.CUstream stream) nogil

    # Create non-owning handle that prevents Python owner from being GC'd
    # Owner is responsible for keeping the stream's context alive
    StreamHandle create_stream_handle_with_owner(cydriver.CUstream stream, object owner)

    # Get non-owning handle to the legacy default stream (no context dependency)
    StreamHandle get_legacy_stream() nogil

    # Get non-owning handle to the per-thread default stream (no context dependency)
    StreamHandle get_per_thread_stream() nogil

    # ========================================================================
    # Event Handle
    # ========================================================================
    ctypedef shared_ptr[const cydriver.CUevent] EventHandle

    # Create an owning event handle via cuEventCreate
    # Context handle establishes structural dependency (context outlives event)
    # Returns empty handle on error (caller must check)
    EventHandle create_event_handle(ContextHandle h_ctx, unsigned int flags) nogil

    # Create an owning event handle without context dependency
    # Use for temporary events that are created and destroyed in the same scope
    # Returns empty handle on error (caller must check)
    EventHandle create_event_handle(unsigned int flags) nogil

    # Create an owning event handle from IPC handle
    # The originating process owns the event and its context
    # Returns empty handle on error (caller must check)
    EventHandle create_event_handle_ipc(const cydriver.CUipcEventHandle& ipc_handle) nogil

    # ========================================================================
    # Memory Pool Handle
    # ========================================================================
    ctypedef shared_ptr[const cydriver.CUmemoryPool] MemoryPoolHandle

    # Create an owning memory pool handle via cuMemPoolCreate
    # Memory pools are device-scoped (not context-scoped)
    # Returns empty handle on error (caller must check)
    MemoryPoolHandle create_mempool_handle(const cydriver.CUmemPoolProps& props) nogil

    # Create a non-owning memory pool handle (pool NOT destroyed when released)
    # Use for device default/current pools managed by the driver
    MemoryPoolHandle create_mempool_handle_ref(cydriver.CUmemoryPool pool) nogil

    # Get non-owning handle to the current memory pool for a device
    # Returns empty handle on error (caller must check)
    MemoryPoolHandle get_device_mempool(int device_id) nogil

    # Create an owning memory pool handle from IPC import
    # File descriptor NOT owned by this handle (caller manages FD separately)
    # Returns empty handle on error (caller must check)
    MemoryPoolHandle create_mempool_handle_ipc(int fd, cydriver.CUmemAllocationHandleType handle_type) nogil

    # ========================================================================
    # Overloaded helper functions (C++ handles dispatch by type)
    # ========================================================================

    # native() - extract the raw CUDA handle
    cydriver.CUcontext native(ContextHandle h) nogil
    cydriver.CUstream native(StreamHandle h) nogil
    cydriver.CUevent native(EventHandle h) nogil
    cydriver.CUmemoryPool native(MemoryPoolHandle h) nogil

    # intptr() - extract handle as uintptr_t for Python interop
    uintptr_t intptr(ContextHandle h) nogil
    uintptr_t intptr(StreamHandle h) nogil
    uintptr_t intptr(EventHandle h) nogil
    uintptr_t intptr(MemoryPoolHandle h) nogil

    # py() - convert handle to Python driver wrapper object (requires GIL)
    object py(ContextHandle h)
    object py(StreamHandle h)
    object py(EventHandle h)
    object py(MemoryPoolHandle h)
