// SPDX-FileCopyrightText: Copyright (c) 2024-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "types.hpp"
#include <cuda.h>
#include <nvrtc.h>
#include <cstddef>

namespace cuda_core::rt {

// ============================================================================
// Context handle functions
// ============================================================================

// Function to create a non-owning context handle (references existing context).
ContextHandle create_context_handle_ref(CUcontext ctx);

// Create a context handle for the CUcontext view of the provided green context.
// The returned ContextHandle keeps the green context alive, but the CUcontext
// view is non-owning and is not destroyed independently.
ContextHandle create_context_handle_from_green_ctx(const GreenCtxHandle& h_green_ctx);

// Return the green context dependency associated with a ContextHandle, if any.
GreenCtxHandle get_context_green_ctx(const ContextHandle& h) noexcept;

// Create an owning green context handle from a list of device resources.
GreenCtxHandle create_green_ctx_handle(CUdevResource* resources, unsigned int nbResources,
                                       CUdevice dev, unsigned int flags);

// Create a non-owning green context handle.
GreenCtxHandle create_green_ctx_handle_ref(CUgreenCtx ctx);

// Get handle to the primary context for a device (with thread-local caching)
// Returns empty handle on error (caller must check)
ContextHandle get_primary_context(int device_id);

// Get handle to the current CUDA context
// Returns empty handle if no context is current (caller must check)
ContextHandle get_current_context();

// Synchronize the provided context. Releases the GIL around the driver call.
// Returns CUDA_ERROR_INVALID_CONTEXT for an empty handle.
CUresult context_synchronize(const ContextHandle& h_context) noexcept;

// Query the stream priority range for the provided context.
// Returns CUDA_ERROR_INVALID_CONTEXT for an empty handle.
CUresult context_get_stream_priority_range(
    const ContextHandle& h_context,
    int* least_priority,
    int* greatest_priority) noexcept;

// Query the device of the provided context.
// Returns CUDA_ERROR_INVALID_CONTEXT for an empty handle.
CUresult context_get_device(const ContextHandle& h_context, CUdevice* device) noexcept;

// Call cuGraphNodeSetParams with h_context current (empty handle: the caller's
// context). Returns the update status; *restore_status receives a failure to
// restore the caller's context after a successful update, which the caller
// raises only after publishing the metadata that depends on the update.
// Returns CUDA_ERROR_NOT_SUPPORTED when the driver lacks cuGraphNodeSetParams.
// Implemented in graph.cpp
CUresult graph_node_set_params(
    CUgraphNode node,
    CUgraphNodeParams* params,
    const ContextHandle& h_context,
    CUresult* restore_status) noexcept;

// ============================================================================
// Stream handle functions
// ============================================================================

// Create an owning stream handle by calling cuStreamCreateWithPriority.
// The stream structurally depends on the provided context handle.
// When the last reference is released, cuStreamDestroy is called automatically.
// Returns empty handle on error (caller must check).
StreamHandle create_stream_handle(const ContextHandle& h_ctx, unsigned int flags, int priority);

// Create a non-owning stream handle (references existing stream).
// Use for borrowed streams (from foreign code) or built-in streams.
// The stream will NOT be destroyed when the handle is released.
// Caller is responsible for keeping the stream's context alive.
StreamHandle create_stream_handle_ref(CUstream stream);

// Initialize the process-lifetime CUDA user-object cleanup queue. Called once
// from module initialization while Python is fully initialized.
// Implemented in py_deferred_cleanup.cpp
void initialize_deferred_cleanup();
// Implemented in py_deferred_cleanup.cpp
void retry_deferred_cleanup() noexcept;

// Return the context dependency associated with a stream handle, if any.
ContextHandle get_stream_context(const StreamHandle& h) noexcept;

// Get non-owning handle to the legacy default stream (CU_STREAM_LEGACY)
// Note: Legacy stream has no specific context dependency.
StreamHandle get_legacy_stream();

// Get non-owning handle to the per-thread default stream (CU_STREAM_PER_THREAD)
// Note: Per-thread stream has no specific context dependency.
StreamHandle get_per_thread_stream();

// Wrap CU_STREAM_LEGACY with an explicit context, bypassing the "bind to
// whatever is current" resolution that a bare default-stream token uses (see
// make_deallocation_stream). Lets a resource that always operates in one
// known context (e.g. a synchronous, non-pooled allocator) record a correct
// deallocation context without requiring that context to be current when the
// token is created. Returns an empty handle for an empty h_context.
StreamHandle create_context_bound_legacy_stream(const ContextHandle& h_context);

// ============================================================================
// Event handle functions
// ============================================================================

// Create an owning event handle by calling cuEventCreate.
// The event structurally depends on the provided context handle.
// Metadata fields are stored in the EventBox for later retrieval.
// When the last reference is released, cuEventDestroy is called automatically.
// Returns empty handle on error (caller must check).
EventHandle create_event_handle(const ContextHandle& h_ctx, unsigned int flags,
                                bool timing_enabled, bool is_blocking_sync,
                                bool ipc_enabled, int device_id);

// Create an owning event in the context that owns `stream`, so it can be
// recorded on that stream regardless of which context is current. Default-
// stream tokens resolve to the current context (cuStreamGetCtx semantics).
// Use for temporary ordering events that are created and destroyed in the
// same scope; the handle carries no device id.
// When the last reference is released, cuEventDestroy is called automatically.
// Returns empty handle on error (caller must check).
EventHandle create_event_handle_for_stream(CUstream stream, unsigned int flags);

// Create an owning event handle from an IPC handle.
// The originating process owns the event and its context.
// When the last reference is released, cuEventDestroy is called automatically.
// Returns empty handle on error (caller must check).
EventHandle create_event_handle_ipc(const CUipcEventHandle& ipc_handle,
                                    bool is_blocking_sync);

// Create a non-owning event handle (references existing event).
// Use for events that are managed by the CUDA graph or another owner.
// The event will NOT be destroyed when the handle is released.
// Metadata defaults to unknown (timing_enabled=false, device_id=-1).
EventHandle create_event_handle_ref(CUevent event);

// Event metadata accessors (read from EventBox via pointer arithmetic)
bool get_event_timing_enabled(const EventHandle& h) noexcept;
bool get_event_is_blocking_sync(const EventHandle& h) noexcept;
bool get_event_ipc_enabled(const EventHandle& h) noexcept;
int get_event_device_id(const EventHandle& h) noexcept;
ContextHandle get_event_context(const EventHandle& h) noexcept;

// ============================================================================
// Memory pool handle functions
// ============================================================================

// Create an owning memory pool handle by calling cuMemPoolCreate.
// Memory pools are device-scoped (not context-scoped).
// When the last reference is released, cuMemPoolDestroy is called automatically.
// Returns empty handle on error (caller must check).
MemoryPoolHandle create_mempool_handle(const CUmemPoolProps& props);

// Create a non-owning memory pool handle (references existing pool).
// Use for device default/current pools that are managed by the driver.
// The pool will NOT be destroyed when the handle is released.
MemoryPoolHandle create_mempool_handle_ref(CUmemoryPool pool);

// Get non-owning handle to the current memory pool for a device.
// Returns empty handle on error (caller must check).
MemoryPoolHandle get_device_mempool(int device_id);

// Create an owning memory pool handle from an IPC import.
// The file descriptor is NOT owned by this handle (caller manages FD separately).
// When the last reference is released, cuMemPoolDestroy is called automatically.
// Returns empty handle on error (caller must check).
MemoryPoolHandle create_mempool_handle_ipc(int fd, CUmemAllocationHandleType handle_type);

// ============================================================================
// Device pointer handle functions
// ============================================================================

// Allocate device memory from a pool asynchronously via cuMemAllocFromPoolAsync.
// The pointer structurally depends on the provided pool handle (captured in deleter).
// When the last reference is released, cuMemFreeAsync is called on the stored stream.
// Returns empty handle on error (caller must check).
DevicePtrHandle deviceptr_alloc_from_pool(
    size_t size,
    const MemoryPoolHandle& h_pool,
    const StreamHandle& h_stream);

// Allocate device memory asynchronously via cuMemAllocAsync.
// When the last reference is released, cuMemFreeAsync is called on the stored stream.
// Returns empty handle on error (caller must check).
DevicePtrHandle deviceptr_alloc_async(size_t size, const StreamHandle& h_stream);

// Allocate device memory synchronously via cuMemAlloc with the provided
// context current. The caller owns the pointer and releases it with cuMemFree.
// Returns CUDA_ERROR_INVALID_CONTEXT for an empty handle.
CUresult deviceptr_alloc_raw(CUdeviceptr* ptr, size_t size,
                             const ContextHandle& h_context) noexcept;

// Allocate pinned host memory via cuMemAllocHost.
// When the last reference is released, cuMemFreeHost is called.
// Returns empty handle on error (caller must check).
DevicePtrHandle deviceptr_alloc_host(size_t size);

// Create a non-owning device pointer handle (references existing pointer).
// Use for foreign pointers (e.g., from external libraries).
// The pointer will NOT be freed when the handle is released.
DevicePtrHandle deviceptr_create_ref(CUdeviceptr ptr);

// Create a device pointer handle for a mapped graphics resource.
// The pointer structurally depends on the provided graphics resource handle.
// When the last reference is released, cuGraphicsUnmapResources is called on
// the stored stream, then the graphics resource may be unregistered when its
// own handle is released.
DevicePtrHandle deviceptr_create_mapped_graphics(
    CUdeviceptr ptr,
    const GraphicsResourceHandle& h_resource,
    const StreamHandle& h_stream);

// Import a device pointer from IPC via cuMemPoolImportPointer.
// When the last reference is released, cuMemFreeAsync is called on the stored stream.
// Note: Does not yet implement reference counting for nvbug 5570902.
// On error, returns empty handle and sets thread-local error (use get_last_error()).
DevicePtrHandle deviceptr_import_ipc(
    const MemoryPoolHandle& h_pool,
    const void* export_data,
    const StreamHandle& h_stream);

// Access the deallocation stream for a device pointer handle (read-only).
// For non-owning handles, the stream is not used but can still be accessed.
StreamHandle deallocation_stream(const DevicePtrHandle& h) noexcept;

// Set the deallocation stream for a device pointer handle.
// Returns CUDA_ERROR_INVALID_CONTEXT when a default-stream token cannot be
// bound because no CUDA context is current.
CUresult set_deallocation_stream(
    const DevicePtrHandle& h, const StreamHandle& h_stream) noexcept;

// ============================================================================
// Library handle functions
// ============================================================================

// Create an owning library handle by loading from a file path.
// When the last reference is released, cuLibraryUnload is called automatically.
// Returns empty handle on error (caller must check).
LibraryHandle create_library_handle_from_file(const char* path);

// Create an owning library handle by loading from memory data.
// The driver makes an internal copy of the data; caller can free it after return.
// When the last reference is released, cuLibraryUnload is called automatically.
// Returns empty handle on error (caller must check).
LibraryHandle create_library_handle_from_data(const void* data);

// Create a non-owning library handle (references existing library).
// Use for borrowed libraries (e.g., from foreign code).
// The library will NOT be unloaded when the handle is released.
LibraryHandle create_library_handle_ref(CUlibrary library);

// ============================================================================
// Kernel handle functions
// ============================================================================

// Get a kernel from a library by name.
// The kernel structurally depends on the provided library handle.
// Kernels have no explicit destroy - their lifetime is tied to the library.
// Returns empty handle on error (caller must check).
KernelHandle create_kernel_handle(const LibraryHandle& h_library, const char* name);

// Create a kernel handle from a raw CUkernel.
// If the kernel is already managed (in the registry), returns the owning
// handle with library dependency. Otherwise returns a non-owning ref.
KernelHandle create_kernel_handle_ref(CUkernel kernel);

// Get the library handle associated with a kernel (from KernelBox).
// Returns empty handle if the kernel has no library dependency.
LibraryHandle get_kernel_library(const KernelHandle& h) noexcept;

// ============================================================================
// Graph handle functions
// ============================================================================

// Create the owning handle for a root graph and its hierarchy.
GraphHandle create_graph_handle(CUgraph graph);

// Create the canonical handle for a graph whose CUDA lifetime is owned by a
// node in h_parent.
GraphHandle create_child_graph_handle(
    CUgraph child_graph, const GraphHandle& h_parent, CUgraphNode owner_node);

// ============================================================================
// Graph node attachments
//
// Each resource-bearing node has one attachment with an immutable owner bundle,
// retained on its CUgraph as a CUDA user object.
//
// Attachment mutations use prepare -> CUDA mutation -> commit. Preparation
// graph-retains a replacement and preallocates its map entry when needed; an
// empty replacement stages removal. Dropping an uncommitted PreparedAttachment
// rolls back any staged retain. Commit updates metadata before releasing the
// previous graph reference.
// graph_get_attachment lets callers carry unchanged owners into partial
// updates. The clone and invalidation helpers synchronize non-owning metadata
// after CUDA copies or destroys graph state.
// ============================================================================

// Build an OpaqueHandle from a malloc'd buffer: std::free on release.
OpaqueHandle make_opaque_malloc(void* buf);

// Copy requested owners from node's current attachment. Pass nullptr to ignore
// either owner; a missing attachment produces empty handles.
CUresult graph_get_attachment(
    const GraphHandle& h_graph,
    CUgraphNode node,
    OpaqueHandle* owner0,
    OpaqueHandle* owner1);

// Create and graph-retain a replacement attachment before a CUDA mutation.
// Destruction rolls the prepared attachment back unless it is committed.
CUresult graph_prepare_attachment(
    const GraphHandle& h_graph,
    OpaqueHandle owner0,
    OpaqueHandle owner1,
    PreparedAttachment* out_prepared);

// Publish a prepared attachment after the CUDA mutation succeeds. A null node
// retains the attachment anonymously without publishing node metadata.
CUresult graph_commit_attachment(
    PreparedAttachment& prepared,
    CUgraphNode node);

// Copy attachment metadata from a source graph hierarchy into its CUDA clone.
CUresult graph_clone_attachments(
    const GraphHandle& h_clone,
    const GraphHandle& h_source);

// Stage a complete metadata replacement before CUDA replaces an embedded
// graph. Dropping the prepared state leaves the current hierarchy unchanged.
CUresult graph_prepare_child_graph_update(
    const GraphHandle& h_parent,
    const GraphHandle& h_old_child,
    CUgraphNode owner_node,
    const GraphHandle& h_source,
    PreparedChildGraphUpdate* out_prepared);

// Rekey staged metadata to CUDA's replacement clone, retire the old embedded
// hierarchy, and publish the replacement handle.
CUresult graph_commit_child_graph_update(
    PreparedChildGraphUpdate& prepared,
    GraphHandle* out_child);

// Invalidate cuda.core state for child graphs CUDA destroyed with owner_node.
void invalidate_child_graph_state(
    const GraphHandle& h_parent,
    CUgraphNode owner_node) noexcept;

// Invalidate cuda.core state for a root graph that CUDA destroyed itself, such
// as the graph of an invalidated capture ended by cuStreamEndCapture. The
// owning handle then no longer calls cuGraphDestroy. No-op unless h_root is
// the live root of its hierarchy.
void invalidate_root_graph_state(const GraphHandle& h_root) noexcept;

// ============================================================================
// Graph exec handle functions
// ============================================================================

// Create an owning exec handle by calling cuGraphInstantiateWithParams.
// A fresh attachment accumulator is retained on h_source first, because CUDA
// propagates user object references only at instantiation; an exec cannot
// receive them afterwards. The exec is the sole owner once this returns.
// When the last reference is released, cuGraphExecDestroy is called
// automatically.
// Returns empty handle on error (caller must check). The caller reads
// params->result_out for the specific instantiation failure and
// get_last_error() for a driver status.
GraphExecHandle create_graph_exec_handle(
    const GraphHandle& h_source,
    CUDA_GRAPH_INSTANTIATE_PARAMS* params);

// Update h_exec in place by calling cuGraphExecUpdate, and publish a fresh
// accumulator when CUDA accepts the update. Writes result_info for the caller.
CUresult graph_exec_update(
    const GraphExecHandle& h_exec,
    const GraphHandle& h_source,
    CUgraphExecUpdateResultInfo* result_info);

// Append owners before an executable-node mutation. The accumulator grows
// because CUDA cannot attach user objects to an exec after instantiation, so
// old owners stay reachable. Dropping the transaction restores the accumulator
// to its original size.
CUresult graph_prepare_exec_attachment(
    const GraphExecHandle& h_exec,
    OpaqueHandle owner0,
    OpaqueHandle owner1,
    PreparedExecAttachment* out_prepared);

// Keep the owners added by graph_prepare_exec_attachment.
void graph_commit_exec_attachment(
    PreparedExecAttachment& prepared) noexcept;

// ============================================================================
// Graph node handle functions
// ============================================================================

// Create a node handle. Nodes are owned by their parent graph (not
// independently destroyable). The GraphHandle dependency ensures the
// graph outlives any node reference.
GraphNodeHandle create_graph_node_handle(CUgraphNode node, const GraphHandle& h_graph);

// Extract the owning graph handle from a node handle.
GraphHandle graph_node_get_graph(const GraphNodeHandle& h) noexcept;

// Zero the CUgraphNode resource inside the handle, marking it invalid.
void invalidate_graph_node(const GraphNodeHandle& h) noexcept;

// ============================================================================
// Graphics resource handle functions
// ============================================================================

// Create an owning graphics resource handle.
// When the last reference is released, cuGraphicsUnregisterResource is called automatically.
// Use for CUgraphicsResource handles obtained from cuGraphicsGLRegisterBuffer etc.
GraphicsResourceHandle create_graphics_resource_handle(CUgraphicsResource resource);

// ============================================================================
// NVRTC Program handle functions
// ============================================================================

// Create an owning NVRTC program handle.
// When the last reference is released, nvrtcDestroyProgram is called.
// Use this to wrap a program created via nvrtcCreateProgram.
NvrtcProgramHandle create_nvrtc_program_handle(nvrtcProgram prog);

// Create a non-owning NVRTC program handle (references existing program).
// The program will NOT be destroyed when the handle is released.
NvrtcProgramHandle create_nvrtc_program_handle_ref(nvrtcProgram prog);

// ============================================================================
// NVVM Program handle functions
// ============================================================================

// Create an owning NVVM program handle.
// When the last reference is released, nvvmDestroyProgram is called.
// Use this to wrap a program created via nvvmCreateProgram.
// Note: If NVVM is not available (p_nvvmDestroyProgram is null), the deleter is a no-op.
NvvmProgramHandle create_nvvm_program_handle(nvvmProgram prog);

// Create a non-owning NVVM program handle (references existing program).
// The program will NOT be destroyed when the handle is released.
NvvmProgramHandle create_nvvm_program_handle_ref(nvvmProgram prog);

// ============================================================================
// nvJitLink handle functions
// ============================================================================

// Create an owning nvJitLink handle.
// When the last reference is released, nvJitLinkDestroy is called.
// Use this to wrap a handle created via nvJitLinkCreate.
// Note: If nvJitLink is not available (p_nvJitLinkDestroy is null), the deleter is a no-op.
NvJitLinkHandle create_nvjitlink_handle(nvJitLink_t handle);

// Create a non-owning nvJitLink handle (references existing handle).
// The handle will NOT be destroyed when the last reference is released.
NvJitLinkHandle create_nvjitlink_handle_ref(nvJitLink_t handle);

// ============================================================================
// cuLink handle functions
// ============================================================================

// Create an owning cuLink handle.
// When the last reference is released, cuLinkDestroy is called.
// Use this to wrap a CUlinkState created via cuLinkCreate.
CuLinkHandle create_culink_handle(CUlinkState state);

// Create a non-owning cuLink handle (references existing CUlinkState).
// The handle will NOT be destroyed when the last reference is released.
CuLinkHandle create_culink_handle_ref(CUlinkState state);

// ============================================================================
// File descriptor handle functions
// ============================================================================

// Create an owning file descriptor handle.
// When the last reference is released, POSIX close() is called.
FileDescriptorHandle create_fd_handle(int fd);

// Create a non-owning file descriptor handle (caller manages the fd).
FileDescriptorHandle create_fd_handle_ref(int fd);

// ============================================================================
// Array / mipmapped-array / texture / surface handle functions (PR #467)
//
// These resources are managed exactly like every other cuda.core resource:
// the owning handle's deleter calls the matching cu*Destroy with the GIL
// released, structural dependencies are embedded in the box (so a backing
// resource always outlives a texture/surface/level built on it), and
// creation returns an empty handle + thread-local error on failure.
// ============================================================================

// Create an owning CUDA array via cuArray3DCreate.
// When the last reference is released, cuArrayDestroy is called automatically.
// Returns empty handle on error (caller must check).
OpaqueArrayHandle create_array_handle(const ContextHandle& h_context, const CUDA_ARRAY3D_DESCRIPTOR& desc);

// Create a non-owning array handle (references an existing CUarray).
// Use for arrays owned elsewhere (e.g. graphics interop). Never destroyed here.
OpaqueArrayHandle create_array_handle_ref(CUarray arr);

// Create an owning array handle adopting an existing CUarray.
// When the last reference is released, cuArrayDestroy is called automatically.
OpaqueArrayHandle create_array_handle_owning(CUarray arr);

// Return the context dependency associated with an array, if known.
ContextHandle get_array_context(const OpaqueArrayHandle& h) noexcept;

// Create a non-owning handle to a mipmap level via cuMipmappedArrayGetLevel.
// The level CUarray is owned by the mipmap; the parent MipmappedArrayHandle is
// embedded in the box so it outlives the level view. No destroy in the deleter.
// Returns empty handle on error (caller must check).
OpaqueArrayHandle create_array_level_handle(const MipmappedArrayHandle& h_mip, unsigned int level);

// Create an owning mipmapped array via cuMipmappedArrayCreate.
// When the last reference is released, cuMipmappedArrayDestroy is called.
// Returns empty handle on error (caller must check).
MipmappedArrayHandle create_mipmapped_array_handle(const ContextHandle& h_context,
                                                   const CUDA_ARRAY3D_DESCRIPTOR& desc,
                                                   unsigned int num_levels);

// Return the context dependency associated with a mipmapped array, if known.
ContextHandle get_mipmapped_array_context(const MipmappedArrayHandle& h) noexcept;

// Create an owning texture object via cuTexObjectCreate, embedding the backing
// resource handle (array / mipmapped array / linear-or-pitch2d device pointer)
// so the backing always outlives the texture. cuTexObjectDestroy runs in the
// deleter. Returns empty handle on error (caller must check).
TexObjectHandle create_tex_object_handle_array(const ContextHandle& h_context,
                                               const CUDA_RESOURCE_DESC& res,
                                               const CUDA_TEXTURE_DESC& tex,
                                               const OpaqueArrayHandle& h_backing);
TexObjectHandle create_tex_object_handle_mipmap(const ContextHandle& h_context,
                                                const CUDA_RESOURCE_DESC& res,
                                                const CUDA_TEXTURE_DESC& tex,
                                                const MipmappedArrayHandle& h_backing);
TexObjectHandle create_tex_object_handle_linear(const ContextHandle& h_context,
                                                const CUDA_RESOURCE_DESC& res,
                                                const CUDA_TEXTURE_DESC& tex,
                                                const DevicePtrHandle& h_backing);

// Create an owning surface object via cuSurfObjectCreate, embedding the backing
// array handle so it outlives the surface. cuSurfObjectDestroy runs in the
// deleter. Returns empty handle on error (caller must check).
SurfObjectHandle create_surf_object_handle(const ContextHandle& h_context,
                                           const CUDA_RESOURCE_DESC& res,
                                           const OpaqueArrayHandle& h_backing);

}  // namespace cuda_core::rt
