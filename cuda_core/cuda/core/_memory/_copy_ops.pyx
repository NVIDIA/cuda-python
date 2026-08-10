# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from collections.abc import Sequence

import warnings

IF CUDA_CORE_BUILD_MAJOR >= 13:
    from libcpp.vector cimport vector

from libc.string cimport memset

# as_cu, HANDLE_RETURN and _attr_run_starts are referenced only from the
# CUDA 13 branch of _do_copy_batch. cython-lint does not evaluate
# compile-time IF blocks, so it needs a pragma to see them as used.
from cuda.bindings cimport cydriver
from cuda.core._memory._buffer cimport Buffer
from cuda.core._resource_handles cimport as_cu  # no-cython-lint
from cuda.core._stream cimport Stream, Stream_accept
from cuda.core._utils.cuda_utils cimport HANDLE_RETURN  # no-cython-lint

from cuda.core._device import Device
from cuda.core._memory._copy_enums import CopyOptions, _attr_run_starts  # no-cython-lint
from cuda.core._memory._managed_location import _coerce_location

cdef tuple _coerce_batch_buffers(object buffers, str what):
    """Coerce ``buffers`` to a tuple[Buffer, ...]; rejects a single Buffer."""
    cdef list out
    if isinstance(buffers, Buffer):
        raise TypeError(
            f"{what}: pass a sequence of Buffers; for a single buffer use "
            f"the Buffer.copy_to / Buffer.copy_from instance method"
        )
    if isinstance(buffers, Sequence):
        if not buffers:
            raise ValueError(f"{what}: empty buffers sequence")
        out = []
        for t in buffers:
            if not isinstance(t, Buffer):
                raise TypeError(
                    f"{what}: expected Buffer, got {type(t).__name__}"
                )
            out.append(t)
        return tuple(out)
    raise TypeError(
        f"{what}: buffers must be a sequence of Buffer, "
        f"got {type(buffers).__name__}"
    )


IF CUDA_CORE_BUILD_MAJOR >= 13:
    cdef inline cydriver.CUmemLocation _to_cumemlocation(object loc):
        """Convert a _LocSpec dataclass to a cydriver.CUmemLocation struct."""
        cdef str kind = loc.kind
        if kind == "device":
            return cydriver.CUmemLocation(
                type=cydriver.CUmemLocationType.CU_MEM_LOCATION_TYPE_DEVICE,
                id=<int>loc.id)
        elif kind == "host":
            return cydriver.CUmemLocation(
                type=cydriver.CUmemLocationType.CU_MEM_LOCATION_TYPE_HOST,
                id=0)
        elif kind == "host_numa":
            return cydriver.CUmemLocation(
                type=cydriver.CUmemLocationType.CU_MEM_LOCATION_TYPE_HOST_NUMA,
                id=<int>loc.id)
        else:  # host_numa_current
            return cydriver.CUmemLocation(
                type=cydriver.CUmemLocationType.CU_MEM_LOCATION_TYPE_HOST_NUMA_CURRENT,
                id=0)
ELSE:
    cdef inline cydriver.CUmemLocation _to_cumemlocation(object loc):
        raise NotImplementedError(
            "_to_cumemlocation requires a CUDA 13 build of cuda.core"
        )


cdef cydriver.CUmemcpyAttributes _to_cu_memcpy_attributes(object attr):
    """Convert a CopyOptions to a cydriver.CUmemcpyAttributes struct."""
    cdef cydriver.CUmemcpyAttributes cu_attr
    memset(&cu_attr, 0, sizeof(cydriver.CUmemcpyAttributes))
    cu_attr.srcAccessOrder = <cydriver.CUmemcpySrcAccessOrder>(<int>attr._to_driver_enum())
    cu_attr.flags = <unsigned int>(<int>attr._to_driver_flags())

    cdef object src_loc = _coerce_location(attr.src_location_hint, allow_none=True)
    cdef object dst_loc = _coerce_location(attr.dst_location_hint, allow_none=True)

    if src_loc is not None:
        cu_attr.srcLocHint = _to_cumemlocation(src_loc)
    if dst_loc is not None:
        cu_attr.dstLocHint = _to_cumemlocation(dst_loc)

    return cu_attr


def copy_batch(
    stream: object,
    srcs: Sequence[Buffer],
    dsts: Sequence[Buffer],
    *,
    options: object = None,
) -> None:
    """Copy a batch of buffers asynchronously.

    Requires CUDA 13+. For a single buffer, use
    :meth:`Buffer.copy_to` or :meth:`Buffer.copy_from`.

    The driver provides no graph-node form of ``cuMemcpyBatchAsync``, so
    this cannot be captured into a graph. Build graph copies with
    :meth:`graph.GraphNode.memcpy` or per-buffer :meth:`Buffer.copy_to`.

    Parameters
    ----------
    stream : :class:`~_stream.Stream`
        Stream for the asynchronous copy. Passing a
        :class:`~graph.GraphBuilder` raises ``CUDAError``
        (``CUDA_ERROR_STREAM_CAPTURE_UNSUPPORTED``).
    srcs : Sequence[:class:`Buffer`]
        Source buffers.  Must be a sequence, not a single Buffer.
    dsts : Sequence[:class:`Buffer`]
        Destination buffers.  Must match ``len(srcs)``.
    options : :class:`CopyOptions` | Sequence[:class:`CopyOptions`] | None
        Per-copy options. A single value applies to every copy; a
        sequence pairs by index and must match ``len(srcs)``. ``None``
        uses stream-ordered defaults.

    Raises
    ------
    NotImplementedError
        On a CUDA 12 build of ``cuda.core``.
    ValueError
        If lengths or sizes mismatch.
    TypeError
        If a single Buffer is passed instead of a sequence.
    UserWarning
        If ``overlap_mode='prefer_overlap_with_compute'`` is requested
        on a non-integrated (discrete) GPU.
    """
    cdef tuple src_bufs = _coerce_batch_buffers(srcs, "copy_batch")
    cdef tuple dst_bufs = _coerce_batch_buffers(dsts, "copy_batch")
    cdef Py_ssize_t n = len(src_bufs)

    if len(dst_bufs) != n:
        raise ValueError(
            f"copy_batch: srcs length {n} does not match dsts length {len(dst_bufs)}"
        )

    cdef Stream s = Stream_accept(stream)

    cdef Buffer src_buf
    cdef Buffer dst_buf
    cdef Py_ssize_t i

    for i in range(n):
        src_buf = <Buffer>src_bufs[i]
        dst_buf = <Buffer>dst_bufs[i]
        if src_buf.size != dst_buf.size:
            raise ValueError(
                f"copy_batch: buffer size mismatch at index {i} "
                f"(src={src_buf.size}, dst={dst_buf.size})"
            )

    # Expand `options` to one CopyOptions per copy; the encoder below
    # collapses equal neighbours back into driver attribute runs.
    cdef tuple attr_tuple
    if options is None:
        attr_tuple = (CopyOptions(),) * n
    elif isinstance(options, CopyOptions):
        attr_tuple = (options,) * n
    elif isinstance(options, Sequence):
        if len(options) != n:
            raise ValueError(
                f"copy_batch: options length {len(options)} does not match "
                f"buffers length {n}"
            )
        attr_list = []
        for a in options:
            if not isinstance(a, CopyOptions):
                raise TypeError(
                    f"copy_batch: each options element must be CopyOptions, "
                    f"got {type(a).__name__}"
                )
            attr_list.append(a)
        attr_tuple = tuple(attr_list)
    else:
        raise TypeError(
            f"copy_batch: options must be CopyOptions or a sequence of "
            f"CopyOptions, got {type(options).__name__}"
        )

    # Check for overlap_mode warning on non-integrated GPUs
    cdef bint any_overlap = False
    cdef object ca_attr
    for i in range(n):
        ca_attr = attr_tuple[i]
        if ca_attr.overlap_mode != "default":
            any_overlap = True
            break

    if any_overlap:
        device = Device()
        if not device.properties.integrated:
            warnings.warn(
                "overlap_mode='prefer_overlap_with_compute' has no effect on "
                "non-integrated (non-Tegra) GPUs; the transfer will use "
                "default copy behavior.",
                UserWarning,
                stacklevel=2,
            )

    _do_copy_batch(src_bufs, dst_bufs, s, attr_tuple)


cdef void _do_copy_batch(tuple src_bufs, tuple dst_bufs, Stream s, tuple attr_tuple):
    IF CUDA_CORE_BUILD_MAJOR >= 13:
        cdef Py_ssize_t n = len(src_bufs)
        cdef cydriver.CUstream hstream = as_cu(s._h_stream)
        cdef vector[cydriver.CUdeviceptr] dst_ptrs
        cdef vector[cydriver.CUdeviceptr] src_ptrs
        cdef vector[size_t] sizes
        cdef vector[size_t] attrs_idxs
        dst_ptrs.resize(n)
        src_ptrs.resize(n)
        sizes.resize(n)

        cdef Buffer src_buf
        cdef Buffer dst_buf
        cdef Py_ssize_t i

        # Collapse equal neighbouring attributes into runs so a broadcast
        # attribute reaches the driver once (numAttrs == 1) instead of being
        # repeated per copy. attrs[k] applies to [attrsIdxs[k], attrsIdxs[k+1]).
        cdef list run_starts = _attr_run_starts(attr_tuple)
        cdef vector[cydriver.CUmemcpyAttributes] cu_attrs
        cdef size_t num_attrs = <size_t>len(run_starts)
        cu_attrs.reserve(num_attrs)
        attrs_idxs.reserve(num_attrs)
        for i in run_starts:
            cu_attrs.push_back(_to_cu_memcpy_attributes(attr_tuple[i]))
            attrs_idxs.push_back(<size_t>i)

        for i in range(n):
            src_buf = <Buffer>src_bufs[i]
            dst_buf = <Buffer>dst_bufs[i]
            src_ptrs[i] = as_cu(src_buf._h_ptr)
            dst_ptrs[i] = as_cu(dst_buf._h_ptr)
            sizes[i] = src_buf.size

        with nogil:
            HANDLE_RETURN(cydriver.cuMemcpyBatchAsync(
                dst_ptrs.data(),
                src_ptrs.data(),
                sizes.data(),
                <size_t>n,
                cu_attrs.data(),
                attrs_idxs.data(),
                num_attrs,
                hstream,
            ))
    ELSE:
        raise NotImplementedError(
            "copy_batch requires a CUDA 13 build of cuda.core"
        )
