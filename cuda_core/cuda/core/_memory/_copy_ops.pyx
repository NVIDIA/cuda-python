# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from collections.abc import Sequence

import warnings

IF CUDA_CORE_BUILD_MAJOR >= 13:
    from libcpp.vector cimport vector

from libc.string cimport memset

from cuda.bindings cimport cydriver
from cuda.core._memory._buffer cimport Buffer, Buffer_coerce_batch
from cuda.core._memory._location cimport to_cumemlocation
from cuda.core._resource_handles cimport as_cu
from cuda.core._stream cimport Stream, Stream_accept, Stream_is_default_token
from cuda.core._utils.cuda_utils cimport HANDLE_RETURN

# cy_driver_version and _attr_run_starts are referenced only from CUDA 13
# branches. cython-lint does not evaluate compile-time IF blocks, so they need
# a pragma to be seen as used.
from cuda.core._utils.version cimport cy_driver_version  # no-cython-lint

from cuda.core._device import Device
from cuda.core._memory._copy_enums import CopyOptions, _attr_run_starts  # no-cython-lint
from cuda.core._memory._managed_location import _coerce_location

_SINGLE_COPY_HINT = "Buffer.copy_to / Buffer.copy_from"

# Attributes reach the driver only through cuMemcpyBatchAsync. The per-copy
# cuMemcpyAsync fallback has nowhere to put them, so anything other than the
# defaults is refused rather than dropped.
_DEFAULT_COPY_OPTIONS = CopyOptions()


cdef inline bint _batch_entry_point_available():
    """Whether cuMemcpyBatchAsync can actually be called here.

    Requires ``cuda.core`` built against CUDA 13 headers (compile time) and
    a driver reporting CUDA 13.0 or newer, i.e.
    ``cuDriverGetVersion() >= 13000`` (run time).

    The run-time bound is set by the binding layer, not by when the driver
    gained the feature. CUDA 12.8 already exposed a ``cuMemcpyBatchAsync``,
    but its signature carried a ``failIdx`` out-parameter that CUDA 13.0
    dropped. ``cuda.bindings`` resolves only the 13.0 revision, via
    ``cuGetProcAddress_v2('cuMemcpyBatchAsync', ..., 13000, ...)``, so an
    older driver yields a NULL pointer even though it may implement the
    earlier entry point.
    """
    IF CUDA_CORE_BUILD_MAJOR >= 13:
        return cy_driver_version() >= (13, 0, 0)
    ELSE:
        return False


def _batch_entry_point_in_use() -> bool:
    """Internal: expose the dispatch predicate so tests can gate on it."""
    return bool(_batch_entry_point_available())


def _normalize_copy_options(
    options: CopyOptions | Sequence[CopyOptions] | None,
    Py_ssize_t n,
) -> tuple[CopyOptions, ...]:
    """Expand ``options`` to exactly one :class:`CopyOptions` per copy.

    ``None`` and a scalar broadcast; a sequence pairs by index and must
    already have length ``n``.

    Internal, but deliberately importable: options are hints that change
    how the driver stages a transfer and never the bytes it produces, so
    this expansion (and the run encoding applied to it) is the only
    observable evidence that a scalar reached every copy.
    """
    if options is None:
        return (_DEFAULT_COPY_OPTIONS,) * n
    if isinstance(options, CopyOptions):
        return (options,) * n
    if isinstance(options, Sequence):
        if len(options) != n:
            raise ValueError(
                f"copy_batch: options length {len(options)} does not match "
                f"buffers length {n}"
            )
        for a in options:
            if not isinstance(a, CopyOptions):
                raise TypeError(
                    f"copy_batch: each options element must be CopyOptions, "
                    f"got {type(a).__name__}"
                )
        return tuple(options)
    raise TypeError(
        f"copy_batch: options must be CopyOptions or a sequence of "
        f"CopyOptions, got {type(options).__name__}"
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
        cu_attr.srcLocHint = to_cumemlocation(src_loc.kind, src_loc.id)
    if dst_loc is not None:
        cu_attr.dstLocHint = to_cumemlocation(dst_loc.kind, dst_loc.id)

    return cu_attr


def copy_batch(
    stream: Stream,
    srcs: Sequence[Buffer],
    dsts: Sequence[Buffer],
    *,
    options: CopyOptions | Sequence[CopyOptions] | None = None,
) -> None:
    """Copy a batch of buffers asynchronously.

    Source buffer and destination buffer sizes must match. For a single
    buffer, use :meth:`Buffer.copy_to` or :meth:`Buffer.copy_from`.

    The driver provides no graph-node form of ``cuMemcpyBatchAsync``, so
    this cannot be captured into a graph. Build graph copies with
    :meth:`graph.GraphNode.memcpy` or per-buffer :meth:`Buffer.copy_to`.

    Parameters
    ----------
    stream : :class:`~_stream.Stream`
        Stream for the asynchronous copy. First positional and required
        (mirrors :func:`launch`). Unlike most stream-taking APIs this does
        not accept a :class:`~graph.GraphBuilder`; one is rejected with
        ``TypeError`` because the copy cannot be captured.
    srcs : Sequence[:class:`Buffer`]
        Source buffers. Must be a sequence, not a single Buffer.
    dsts : Sequence[:class:`Buffer`]
        Destination buffers. Must match ``len(srcs)``.
    options : :class:`CopyOptions` | Sequence[:class:`CopyOptions`] | None
        Per-copy options. A single value applies to every copy; a
        sequence pairs by index and must match ``len(srcs)``. ``None``
        uses stream-ordered defaults.

    Raises
    ------
    ValueError
        If lengths or sizes mismatch.
    TypeError
        If a single Buffer is passed instead of a sequence, or if a
        default-stream token (``LEGACY_DEFAULT_STREAM`` /
        ``PER_THREAD_DEFAULT_STREAM``) is passed instead of an explicit
        stream.
    NotImplementedError
        If non-default ``options`` are given where
        ``cuMemcpyBatchAsync`` is unavailable (see Notes).

    Notes
    -----
    Batching through ``cuMemcpyBatchAsync`` requires all three of:
    ``cuda.core`` built against CUDA 13 headers, ``cuda.bindings`` 13.0 or
    newer, and a driver reporting CUDA 13.0 or newer
    (``cuDriverGetVersion() >= 13000``). ``cuda.bindings`` binds only the
    CUDA 13.0 revision of the entry point, so a driver that predates it is
    refused even where it implements the earlier CUDA 12.8 signature.

    The driver may execute batch items concurrently and in any order.
    A batch must therefore not contain copies where the source range of
    one copy overlaps the destination range of another; such aliasing
    produces undefined results. Detecting overlaps at runtime is
    impractical; callers are responsible for ensuring no aliasing exists.

    On pre-CUDA 13 installs the copies fall back to a Python-level loop
    over ``cuMemcpyAsync``, so the potential performance benefit of
    asynchronous batched copies is not realized. The fallback has no way
    to convey :class:`CopyOptions` to the driver, so non-default options
    raise :class:`NotImplementedError` there rather than being silently
    ignored.

    Warns
    -----
    UserWarning
        If ``overlap_mode='prefer_overlap_with_compute'`` is requested
        on a non-integrated (discrete) GPU.
    """
    cdef tuple src_bufs = Buffer_coerce_batch(srcs, "copy_batch", _SINGLE_COPY_HINT)
    cdef tuple dst_bufs = Buffer_coerce_batch(dsts, "copy_batch", _SINGLE_COPY_HINT)
    cdef Py_ssize_t n = len(src_bufs)

    if len(dst_bufs) != n:
        raise ValueError(
            f"copy_batch: srcs length {n} does not match dsts length {len(dst_bufs)}"
        )

    cdef Stream s = Stream_accept(stream)

    if Stream_is_default_token(s):
        raise TypeError(
            "copy_batch does not accept a default-stream token "
            "(LEGACY_DEFAULT_STREAM / PER_THREAD_DEFAULT_STREAM); "
            "pass an explicit stream"
        )

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

    cdef tuple attr_tuple = _normalize_copy_options(options, n)

    # Without the batched entry point there is nowhere to put attributes,
    # so refuse them here rather than dropping them in the fallback. Doing
    # this before the overlap warning keeps the error the only diagnostic.
    if not _batch_entry_point_available():
        for i in range(n):
            if attr_tuple[i] != _DEFAULT_COPY_OPTIONS:
                raise NotImplementedError(
                    "copy_batch: non-default CopyOptions requires cuMemcpyBatchAsync, "
                    "which needs cuda.core built against CUDA 13, cuda.bindings 13.0+, "
                    "and a driver reporting CUDA 13.0 or newer; omit options to use the "
                    "per-copy fallback"
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
        # Building against CUDA 13 headers says nothing about the installed
        # driver, so the run-time version still has to be checked before
        # calling a 13.0-only entry point (see PRs #2054 / #2064).
        if _batch_entry_point_available():
            _do_copy_batch_native(src_bufs, dst_bufs, s, attr_tuple)
        else:
            _do_copy_batch_loop(src_bufs, dst_bufs, s)
    ELSE:
        _do_copy_batch_loop(src_bufs, dst_bufs, s)


cdef void _do_copy_batch_loop(tuple src_bufs, tuple dst_bufs, Stream s):
    """Per-copy cuMemcpyAsync fallback where the batch entry point is absent.

    Issues copies one at a time, so the performance benefit of batching is
    not realized. Callers guarantee the options are defaults; copy_batch
    rejects anything else before reaching here.
    """
    cdef Py_ssize_t n = len(src_bufs)
    cdef Py_ssize_t i
    cdef Buffer src_buf
    cdef Buffer dst_buf
    cdef size_t nbytes
    cdef cydriver.CUstream hstream = as_cu(s._h_stream)

    for i in range(n):
        src_buf = <Buffer>src_bufs[i]
        dst_buf = <Buffer>dst_bufs[i]
        nbytes = src_buf._size
        with nogil:
            HANDLE_RETURN(cydriver.cuMemcpyAsync(
                as_cu(dst_buf._h_ptr), as_cu(src_buf._h_ptr), nbytes, hstream))


IF CUDA_CORE_BUILD_MAJOR >= 13:
    cdef void _do_copy_batch_native(tuple src_bufs, tuple dst_bufs, Stream s, tuple attr_tuple):
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
