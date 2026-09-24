# SPDX-FileCopyrightText: Copyright (c) 2024-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

# This module compiles the C++ under _cpp/rt/ into one shared library.
# Consumer modules cimport the functions declared in _rt.pxd. Since there is
# only one copy of the C++ code (in this .so), all static and thread-local
# state is shared correctly across all consumer modules.
#
# "rt" is short for runtime: this is cuda.core's runtime support layer
# (resource handles, the driver function-pointer table, error reporting and
# deferred cleanup). It is unrelated to the CUDA Runtime API (cudart).
#
# The cdef extern from declarations below satisfy the .pxd declarations directly,
# without needing separate wrapper functions.

from cpython.object cimport PyObject
from libc.stddef cimport size_t

from cuda.bindings cimport cydriver
from cuda.bindings cimport cynvrtc
from cuda.bindings cimport cynvvm
from cuda.bindings cimport cynvjitlink


# =============================================================================
# C++ function declarations (non-inline, implemented under _cpp/rt/)
#
# These declarations satisfy the cdef function declarations in _rt.pxd.
# Consumer modules cimport these functions and calls go through this .so.
# =============================================================================

cdef extern from "_cpp/rt/rt.hpp" namespace "cuda_core::rt":
    # Thread-local error handling
    cydriver.CUresult get_last_error "cuda_core::rt::get_last_error" () noexcept nogil
    cydriver.CUresult peek_last_error "cuda_core::rt::peek_last_error" () noexcept nogil
    void clear_last_error "cuda_core::rt::clear_last_error" () noexcept nogil

    # Non-propagating error reporting
    void register_warning_category "cuda_core::rt::register_warning_category" (
        PyObject* category) noexcept
    void report_cuda_error "cuda_core::rt::report_cuda_error" (
        const char* operation, cydriver.CUresult status, const char* detail) noexcept nogil
    void report_message "cuda_core::rt::report_message" (const char* message) noexcept nogil
    void report_status_code "cuda_core::rt::report_status_code" (
        const char* operation, long code) noexcept nogil
    void attach_rollback_failure "cuda_core::rt::attach_rollback_failure" (
        const char* operation, cydriver.CUresult status, const char* detail) noexcept nogil
    # Alias for calls made from this module: calling the pxd-declared name here
    # would make Cython emit a conflicting static prototype for it.
    void _attach_rollback_failure_local "cuda_core::rt::attach_rollback_failure" (
        const char* operation, cydriver.CUresult status, const char* detail) noexcept nogil
    const char* take_last_error_detail "cuda_core::rt::take_last_error_detail" (
        cydriver.CUresult status) noexcept nogil
    void clear_last_error_detail "cuda_core::rt::clear_last_error_detail" () noexcept nogil
    void set_context_restore_fault_for_testing "cuda_core::rt::set_context_restore_fault_for_testing" (
        cydriver.CUresult status) noexcept nogil

    # Context handles
    ContextHandle create_context_handle_ref "cuda_core::rt::create_context_handle_ref" (
        cydriver.CUcontext ctx) except+ nogil
    ContextHandle create_context_handle_from_green_ctx "cuda_core::rt::create_context_handle_from_green_ctx" (
        const GreenCtxHandle& h_green_ctx) except+ nogil
    GreenCtxHandle get_context_green_ctx "cuda_core::rt::get_context_green_ctx" (
        const ContextHandle& h) noexcept nogil
    GreenCtxHandle create_green_ctx_handle "cuda_core::rt::create_green_ctx_handle" (
        cydriver.CUdevResource* resources, unsigned int nbResources,
        cydriver.CUdevice dev, unsigned int flags) except+ nogil
    GreenCtxHandle create_green_ctx_handle_ref "cuda_core::rt::create_green_ctx_handle_ref" (
        cydriver.CUgreenCtx ctx) except+ nogil
    ContextHandle get_primary_context "cuda_core::rt::get_primary_context" (
        int device_id) except+ nogil
    ContextHandle get_current_context "cuda_core::rt::get_current_context" () except+ nogil
    cydriver.CUresult context_synchronize "cuda_core::rt::context_synchronize" (
        const ContextHandle& h_context) noexcept nogil
    cydriver.CUresult context_get_stream_priority_range "cuda_core::rt::context_get_stream_priority_range" (
        const ContextHandle& h_context,
        int* least_priority,
        int* greatest_priority) noexcept nogil
    cydriver.CUresult context_get_device "cuda_core::rt::context_get_device" (
        const ContextHandle& h_context, cydriver.CUdevice* device) noexcept nogil
    cydriver.CUresult graph_node_set_params "cuda_core::rt::graph_node_set_params" (
        cydriver.CUgraphNode node, cydriver.CUgraphNodeParams* params,
        const ContextHandle& h_context, cydriver.CUresult* restore_status) noexcept nogil

    # Stream handles
    StreamHandle create_stream_handle "cuda_core::rt::create_stream_handle" (
        const ContextHandle& h_ctx, unsigned int flags, int priority) except+ nogil
    StreamHandle create_stream_handle_ref "cuda_core::rt::create_stream_handle_ref" (
        cydriver.CUstream stream) except+ nogil
    StreamHandle create_stream_handle_with_owner "cuda_core::rt::create_stream_handle_with_owner" (
        cydriver.CUstream stream, object owner) except+ nogil
    void initialize_deferred_cleanup "cuda_core::rt::initialize_deferred_cleanup" () except+
    void retry_deferred_cleanup "cuda_core::rt::retry_deferred_cleanup" () noexcept
    ContextHandle get_stream_context "cuda_core::rt::get_stream_context" (
        const StreamHandle& h) noexcept nogil
    StreamHandle get_legacy_stream "cuda_core::rt::get_legacy_stream" () except+ nogil
    StreamHandle get_per_thread_stream "cuda_core::rt::get_per_thread_stream" () except+ nogil
    StreamHandle create_context_bound_legacy_stream "cuda_core::rt::create_context_bound_legacy_stream" (
        const ContextHandle& h_context) except+ nogil

    # Event handles (note: _create_event_handle* are internal due to C++ overloading)
    EventHandle create_event_handle "cuda_core::rt::create_event_handle" (
        const ContextHandle& h_ctx, unsigned int flags,
        bint timing_enabled, bint is_blocking_sync,
        bint ipc_enabled, int device_id) except+ nogil
    EventHandle create_event_handle_for_stream "cuda_core::rt::create_event_handle_for_stream" (
        cydriver.CUstream stream, unsigned int flags) except+ nogil
    EventHandle create_event_handle_ref "cuda_core::rt::create_event_handle_ref" (
        cydriver.CUevent event) except+ nogil
    EventHandle create_event_handle_ipc "cuda_core::rt::create_event_handle_ipc" (
        const cydriver.CUipcEventHandle& ipc_handle, bint is_blocking_sync) except+ nogil

    # Event metadata getters
    bint get_event_timing_enabled "cuda_core::rt::get_event_timing_enabled" (
        const EventHandle& h) noexcept nogil
    bint get_event_is_blocking_sync "cuda_core::rt::get_event_is_blocking_sync" (
        const EventHandle& h) noexcept nogil
    bint get_event_ipc_enabled "cuda_core::rt::get_event_ipc_enabled" (
        const EventHandle& h) noexcept nogil
    int get_event_device_id "cuda_core::rt::get_event_device_id" (
        const EventHandle& h) noexcept nogil
    ContextHandle get_event_context "cuda_core::rt::get_event_context" (
        const EventHandle& h) noexcept nogil

    # Memory pool handles
    MemoryPoolHandle create_mempool_handle "cuda_core::rt::create_mempool_handle" (
        const cydriver.CUmemPoolProps& props) except+ nogil
    MemoryPoolHandle create_mempool_handle_ref "cuda_core::rt::create_mempool_handle_ref" (
        cydriver.CUmemoryPool pool) except+ nogil
    MemoryPoolHandle get_device_mempool "cuda_core::rt::get_device_mempool" (
        int device_id) except+ nogil
    MemoryPoolHandle create_mempool_handle_ipc "cuda_core::rt::create_mempool_handle_ipc" (
        int fd, cydriver.CUmemAllocationHandleType handle_type) except+ nogil

    # Device pointer handles
    DevicePtrHandle deviceptr_alloc_from_pool "cuda_core::rt::deviceptr_alloc_from_pool" (
        size_t size, const MemoryPoolHandle& h_pool, const StreamHandle& h_stream) except+ nogil
    DevicePtrHandle deviceptr_alloc_async "cuda_core::rt::deviceptr_alloc_async" (
        size_t size, const StreamHandle& h_stream) except+ nogil
    cydriver.CUresult deviceptr_alloc_raw "cuda_core::rt::deviceptr_alloc_raw" (
        cydriver.CUdeviceptr* ptr, size_t size, const ContextHandle& h_context) noexcept nogil
    DevicePtrHandle deviceptr_alloc_host "cuda_core::rt::deviceptr_alloc_host" (size_t size) except+ nogil
    DevicePtrHandle deviceptr_create_ref "cuda_core::rt::deviceptr_create_ref" (
        cydriver.CUdeviceptr ptr) except+ nogil
    DevicePtrHandle deviceptr_create_with_owner "cuda_core::rt::deviceptr_create_with_owner" (
        cydriver.CUdeviceptr ptr, object owner) except+ nogil
    DevicePtrHandle deviceptr_create_mapped_graphics "cuda_core::rt::deviceptr_create_mapped_graphics" (
        cydriver.CUdeviceptr ptr,
        const GraphicsResourceHandle& h_resource,
        const StreamHandle& h_stream) except+ nogil

    # MR deallocation callback
    void register_mr_dealloc_callback "cuda_core::rt::register_mr_dealloc_callback" (
        MRDeallocCallback cb) noexcept
    DevicePtrHandle deviceptr_create_with_mr "cuda_core::rt::deviceptr_create_with_mr" (
        cydriver.CUdeviceptr ptr, size_t size, object mr) except+ nogil

    DevicePtrHandle deviceptr_import_ipc "cuda_core::rt::deviceptr_import_ipc" (
        const MemoryPoolHandle& h_pool, const void* export_data, const StreamHandle& h_stream) except+ nogil
    StreamHandle deallocation_stream "cuda_core::rt::deallocation_stream" (
        const DevicePtrHandle& h) noexcept nogil
    cydriver.CUresult set_deallocation_stream "cuda_core::rt::set_deallocation_stream" (
        const DevicePtrHandle& h, const StreamHandle& h_stream) noexcept nogil

    # Library handles
    LibraryHandle create_library_handle_from_file "cuda_core::rt::create_library_handle_from_file" (
        const char* path) except+ nogil
    LibraryHandle create_library_handle_from_data "cuda_core::rt::create_library_handle_from_data" (
        const void* data) except+ nogil
    LibraryHandle create_library_handle_ref "cuda_core::rt::create_library_handle_ref" (
        cydriver.CUlibrary library) except+ nogil

    # Kernel handles
    KernelHandle create_kernel_handle "cuda_core::rt::create_kernel_handle" (
        const LibraryHandle& h_library, const char* name) except+ nogil
    KernelHandle create_kernel_handle_ref "cuda_core::rt::create_kernel_handle_ref" (
        cydriver.CUkernel kernel) except+ nogil
    LibraryHandle get_kernel_library "cuda_core::rt::get_kernel_library" (
        const KernelHandle& h) noexcept nogil

    # Graph handles
    GraphHandle create_graph_handle "cuda_core::rt::create_graph_handle" (
        cydriver.CUgraph graph) except+ nogil
    GraphHandle create_child_graph_handle "cuda_core::rt::create_child_graph_handle" (
        cydriver.CUgraph child_graph, const GraphHandle& h_parent,
        cydriver.CUgraphNode owner_node) except+ nogil

    # Graph node attachments
    OpaqueHandle make_opaque_py "cuda_core::rt::make_opaque_py" (object obj) except+
    OpaqueHandle make_opaque_malloc "cuda_core::rt::make_opaque_malloc" (void* buf) except+
    cydriver.CUresult graph_get_attachment "cuda_core::rt::graph_get_attachment" (
        const GraphHandle& h_graph, cydriver.CUgraphNode node,
        OpaqueHandle* owner0, OpaqueHandle* owner1) except+
    cydriver.CUresult graph_prepare_attachment "cuda_core::rt::graph_prepare_attachment" (
        const GraphHandle& h_graph, OpaqueHandle owner0, OpaqueHandle owner1,
        PreparedAttachment* out_prepared) except+
    cydriver.CUresult graph_commit_attachment "cuda_core::rt::graph_commit_attachment" (
        PreparedAttachment& prepared, cydriver.CUgraphNode node) except+
    cydriver.CUresult graph_clone_attachments "cuda_core::rt::graph_clone_attachments" (
        const GraphHandle& h_clone, const GraphHandle& h_source) except+
    cydriver.CUresult graph_prepare_child_graph_update "cuda_core::rt::graph_prepare_child_graph_update" (
        const GraphHandle& h_parent, const GraphHandle& h_old_child,
        cydriver.CUgraphNode owner_node, const GraphHandle& h_source,
        PreparedChildGraphUpdate* out_prepared) except+
    cydriver.CUresult graph_commit_child_graph_update "cuda_core::rt::graph_commit_child_graph_update" (
        PreparedChildGraphUpdate& prepared, GraphHandle* out_child) except+
    void invalidate_child_graph_state "cuda_core::rt::invalidate_child_graph_state" (
        const GraphHandle& h_parent, cydriver.CUgraphNode owner_node) noexcept
    void invalidate_root_graph_state "cuda_core::rt::invalidate_root_graph_state" (
        const GraphHandle& h_root) noexcept

    # Graph exec handles
    GraphExecHandle create_graph_exec_handle "cuda_core::rt::create_graph_exec_handle" (
        const GraphHandle& h_source,
        cydriver.CUDA_GRAPH_INSTANTIATE_PARAMS* params) except+
    cydriver.CUresult graph_exec_update "cuda_core::rt::graph_exec_update" (
        const GraphExecHandle& h_exec,
        const GraphHandle& h_source,
        cydriver.CUgraphExecUpdateResultInfo* result_info) except+
    cydriver.CUresult graph_prepare_exec_attachment "cuda_core::rt::graph_prepare_exec_attachment" (
        const GraphExecHandle& h_exec,
        OpaqueHandle owner0,
        OpaqueHandle owner1,
        PreparedExecAttachment* out_prepared) except+
    void graph_commit_exec_attachment "cuda_core::rt::graph_commit_exec_attachment" (
        PreparedExecAttachment& prepared) noexcept

    # Graph node handles
    GraphNodeHandle create_graph_node_handle "cuda_core::rt::create_graph_node_handle" (
        cydriver.CUgraphNode node, const GraphHandle& h_graph) except+ nogil
    GraphHandle graph_node_get_graph "cuda_core::rt::graph_node_get_graph" (
        const GraphNodeHandle& h) noexcept nogil
    void invalidate_graph_node "cuda_core::rt::invalidate_graph_node" (
        const GraphNodeHandle& h) noexcept nogil

    # Graphics resource handles
    GraphicsResourceHandle create_graphics_resource_handle "cuda_core::rt::create_graphics_resource_handle" (
        cydriver.CUgraphicsResource resource) except+ nogil

    # NVRTC Program handles
    NvrtcProgramHandle create_nvrtc_program_handle "cuda_core::rt::create_nvrtc_program_handle" (
        cynvrtc.nvrtcProgram prog) except+ nogil
    NvrtcProgramHandle create_nvrtc_program_handle_ref "cuda_core::rt::create_nvrtc_program_handle_ref" (
        cynvrtc.nvrtcProgram prog) except+ nogil

    # NVVM Program handles
    NvvmProgramHandle create_nvvm_program_handle "cuda_core::rt::create_nvvm_program_handle" (
        cynvvm.nvvmProgram prog) except+ nogil
    NvvmProgramHandle create_nvvm_program_handle_ref "cuda_core::rt::create_nvvm_program_handle_ref" (
        cynvvm.nvvmProgram prog) except+ nogil

    # nvJitLink handles
    NvJitLinkHandle create_nvjitlink_handle "cuda_core::rt::create_nvjitlink_handle" (
        cynvjitlink.nvJitLinkHandle handle) except+ nogil
    NvJitLinkHandle create_nvjitlink_handle_ref "cuda_core::rt::create_nvjitlink_handle_ref" (
        cynvjitlink.nvJitLinkHandle handle) except+ nogil

    # cuLink handles
    CuLinkHandle create_culink_handle "cuda_core::rt::create_culink_handle" (
        cydriver.CUlinkState state) except+ nogil
    CuLinkHandle create_culink_handle_ref "cuda_core::rt::create_culink_handle_ref" (
        cydriver.CUlinkState state) except+ nogil

    # File descriptor handles
    FileDescriptorHandle create_fd_handle "cuda_core::rt::create_fd_handle" (
        int fd) except+ nogil
    FileDescriptorHandle create_fd_handle_ref "cuda_core::rt::create_fd_handle_ref" (
        int fd) except+ nogil

    # Array / mipmapped-array / texture / surface handles (PR #467)
    OpaqueArrayHandle create_array_handle "cuda_core::rt::create_array_handle" (
        const ContextHandle& h_context, const cydriver.CUDA_ARRAY3D_DESCRIPTOR& desc) except+ nogil
    OpaqueArrayHandle create_array_handle_ref "cuda_core::rt::create_array_handle_ref" (
        cydriver.CUarray arr) except+ nogil
    OpaqueArrayHandle create_array_handle_owning "cuda_core::rt::create_array_handle_owning" (
        cydriver.CUarray arr) except+ nogil
    ContextHandle get_array_context "cuda_core::rt::get_array_context" (
        const OpaqueArrayHandle& h) noexcept nogil
    OpaqueArrayHandle create_array_level_handle "cuda_core::rt::create_array_level_handle" (
        const MipmappedArrayHandle& h_mip, unsigned int level) except+ nogil
    MipmappedArrayHandle create_mipmapped_array_handle "cuda_core::rt::create_mipmapped_array_handle" (
        const ContextHandle& h_context, const cydriver.CUDA_ARRAY3D_DESCRIPTOR& desc,
        unsigned int num_levels) except+ nogil
    ContextHandle get_mipmapped_array_context "cuda_core::rt::get_mipmapped_array_context" (
        const MipmappedArrayHandle& h) noexcept nogil
    TexObjectHandle create_tex_object_handle_array "cuda_core::rt::create_tex_object_handle_array" (
        const ContextHandle& h_context, const cydriver.CUDA_RESOURCE_DESC& res,
        const cydriver.CUDA_TEXTURE_DESC& tex, const OpaqueArrayHandle& h_backing) except+ nogil
    TexObjectHandle create_tex_object_handle_mipmap "cuda_core::rt::create_tex_object_handle_mipmap" (
        const ContextHandle& h_context, const cydriver.CUDA_RESOURCE_DESC& res,
        const cydriver.CUDA_TEXTURE_DESC& tex, const MipmappedArrayHandle& h_backing) except+ nogil
    TexObjectHandle create_tex_object_handle_linear "cuda_core::rt::create_tex_object_handle_linear" (
        const ContextHandle& h_context, const cydriver.CUDA_RESOURCE_DESC& res,
        const cydriver.CUDA_TEXTURE_DESC& tex, const DevicePtrHandle& h_backing) except+ nogil
    SurfObjectHandle create_surf_object_handle "cuda_core::rt::create_surf_object_handle" (
        const ContextHandle& h_context, const cydriver.CUDA_RESOURCE_DESC& res,
        const OpaqueArrayHandle& h_backing) except+ nogil


initialize_deferred_cleanup()


def _set_context_restore_fault_for_testing(int status):
    """Make the next context restoration on this thread fail with ``status``.

    Test hook for the context save/restore paths in the handle layer. The
    injected failure leaves the target context current, exactly as a failing
    ``cuCtxSetCurrent`` would, so callers must restore the context themselves.
    """
    set_context_restore_fault_for_testing(<cydriver.CUresult>status)


def _attach_rollback_failure_for_testing(int status):
    """Attach a failed CUDA call to the exception being handled, or report it.

    Test hook for ``attach_rollback_failure()``. Called inside an ``except``
    block it adds a note to the exception being handled (Python 3.11+); anywhere
    else it emits a ``CUDAWarning``.
    """
    _attach_rollback_failure_local(
        b"cuTestOperation", <cydriver.CUresult>status, b"failed while testing")
