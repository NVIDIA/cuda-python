# SPDX-FileCopyrightText: Copyright (c) 2024-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from libc.stdint cimport uintptr_t

from cuda.bindings cimport cydriver
from cuda.core._context cimport Context
from cuda.core._memory._buffer cimport Buffer, MemoryResource
from cuda.core._resource_handles cimport (
    ContextHandle,
    create_context_bound_legacy_stream,
    deviceptr_alloc_raw,
    get_last_error,
    get_primary_context,
)
from cuda.core._stream cimport Stream, Stream_accept, Stream_is_default_token
from cuda.core._utils.cuda_utils cimport HANDLE_RETURN

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from cuda.core.graph import GraphBuilder
    from cuda.core.typing import DevicePointerType

__all__ = []


class _SynchronousMemoryResource(MemoryResource):
    __slots__ = ("_context", "_device_id")

    def __init__(self, device_id: int, context=None) -> None:
        from .._device import Device

        self._device_id = Device(device_id).device_id
        # Resolved lazily (in _resolve_context) so that construction with
        # context=None does no CUDA work; the primary context is retained
        # only once actually needed, on the first allocate()/deallocate().
        self._context = context

    def _resolve_context(self) -> Context:
        cdef ContextHandle h_context
        if self._context is None:
            h_context = get_primary_context(self._device_id)
            if not h_context:
                HANDLE_RETURN(get_last_error())
            self._context = Context._from_handle(
                Context, h_context, self._device_id)
        return self._context

    def allocate(
        self,
        size_t size,
        *,
        stream: Stream | GraphBuilder | None = None,
    ) -> Buffer:
        # cuMemAlloc/cuMemFree are synchronous; a caller-supplied stream is
        # accepted (and validated) for interface conformance and, if it is a
        # real stream, recorded as the stream that orders deallocation.
        cdef Context context = self._resolve_context()
        cdef Stream dealloc_stream = None
        if stream is not None:
            dealloc_stream = Stream_accept(stream)
        if dealloc_stream is None or Stream_is_default_token(dealloc_stream):
            # A default-stream token carries no context of its own; Buffer._init
            # would bind it to whichever context is current when it records the
            # deallocation stream (and fail if none is). Bind it to this
            # resource's context instead, so Buffer teardown frees in the right
            # context no matter what is current then. Always the legacy token:
            # a per-thread token would also arm the cross-thread PTDS warning,
            # which is noise for a synchronous resource.
            dealloc_stream = Stream._from_handle(
                Stream, create_context_bound_legacy_stream(context._h_context))

        cdef cydriver.CUdeviceptr ptr = 0
        if size:
            with nogil:
                HANDLE_RETURN(deviceptr_alloc_raw(&ptr, size, context._h_context))
        return Buffer._init(<uintptr_t>ptr, size, self, stream=dealloc_stream)

    def deallocate(
        self,
        ptr: DevicePointerType,
        size_t size,
        *,
        stream: Stream | GraphBuilder | None = None,
    ) -> None:
        if stream is not None:
            Stream_accept(stream).sync()
        # No context switch here, by design (settled in the review of #2750):
        # cuMemFree does not need a current context. The driver resolves the
        # allocation's owning context from the pointer through unified
        # addressing and frees it there (cuapiMemFree_common: "a current
        # context is not required to free the device memory"). On the Buffer
        # teardown path the C++ deleter has additionally already made the
        # recorded deallocation context, bound by allocate() above, current.
        cdef cydriver.CUdeviceptr devptr
        if size:
            devptr = <cydriver.CUdeviceptr><uintptr_t>int(ptr)
            with nogil:
                HANDLE_RETURN(cydriver.cuMemFree(devptr))

    @property
    def is_device_accessible(self) -> bool:
        return True

    @property
    def is_host_accessible(self) -> bool:
        return False

    @property
    def device_id(self) -> int:
        return self._device_id
