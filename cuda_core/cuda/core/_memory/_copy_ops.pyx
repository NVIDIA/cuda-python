# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from collections.abc import Sequence

IF CUDA_CORE_BUILD_MAJOR >= 13:
    from libcpp.vector cimport vector

from cuda.bindings cimport cydriver
from cuda.core._memory._buffer cimport Buffer, Buffer_coerce_batch
from cuda.core._memory._copy_attributes cimport _to_cu_memcpy_attributes  # no-cython-lint
from cuda.core._resource_handles cimport as_cu
from cuda.core._stream cimport Stream, Stream_accept, Stream_is_legacy_default_token
from cuda.core._utils.cuda_utils cimport HANDLE_RETURN

# cy_driver_version and _attr_run_starts are referenced only from CUDA 13
# branches. cython-lint does not evaluate compile-time IF blocks, so they need
# a pragma to be seen as used.
from cuda.core._utils.version cimport cy_driver_version  # no-cython-lint

from cuda.core._memory._copy_enums import (
    CopyOptions,
    _attr_run_starts,  # no-cython-lint
    _reject_unsupported_during_api_call,
)

_SINGLE_COPY_HINT = "Buffer.copy_to / Buffer.copy_from"


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
        return (CopyOptions(),) * n
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
    this cannot be captured into a graph. Both passing a
    :class:`~graph.GraphBuilder` and passing its underlying
    :attr:`~graph.GraphBuilder.stream` while capture is active are
    rejected. Build graph copies with
    :meth:`graph.GraphNode.memcpy` or per-buffer :meth:`Buffer.copy_to`.

    Parameters
    ----------
    stream : :class:`~_stream.Stream`
        Stream for the asynchronous copy. First positional and required
        (mirrors :func:`launch`). Does not accept a capturing stream
        (including a :class:`~graph.GraphBuilder`\'s underlying stream); use
        :meth:`graph.GraphNode.memcpy` or per-buffer
        :meth:`Buffer.copy_to` to build copies into a graph. Does not accept
        ``LEGACY_DEFAULT_STREAM``, which ``cuMemcpyBatchAsync`` rejects
        outright; ``PER_THREAD_DEFAULT_STREAM`` is a real stream to the
        driver and is accepted.
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
        If a single Buffer is passed instead of a sequence, if
        ``LEGACY_DEFAULT_STREAM`` is passed, or if the stream is currently
        in graph capture mode.
    RuntimeError
        If any copy requests ``src_access_order=DURING_API_CALL`` and the
        native ``cuMemcpyBatchAsync`` path is unavailable (see Notes): the
        per-copy ``cuMemcpyAsync`` fallback reads the source in stream
        order only, which cannot honor that guarantee.

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
    asynchronous batched copies is not realized. ``src_access_order`` values
    of ``STREAM`` and ``ANY`` are silently ignored on the fallback path
    (stream-ordered access already satisfies both); ``DURING_API_CALL``
    raises ``RuntimeError`` instead, since silently downgrading it to
    stream-ordered access would let a caller reuse the source buffer before
    the real read happens.

    """
    cdef tuple src_bufs = Buffer_coerce_batch(srcs, "copy_batch", _SINGLE_COPY_HINT)
    cdef tuple dst_bufs = Buffer_coerce_batch(dsts, "copy_batch", _SINGLE_COPY_HINT)
    cdef Py_ssize_t n = len(src_bufs)

    if len(dst_bufs) != n:
        raise ValueError(
            f"copy_batch: srcs length {n} does not match dsts length {len(dst_bufs)}"
        )

    cdef Stream s = Stream_accept(stream)

    if Stream_is_legacy_default_token(s):
        raise TypeError(
            "copy_batch does not accept LEGACY_DEFAULT_STREAM; cuMemcpyBatchAsync "
            "rejects it outright, unlike PER_THREAD_DEFAULT_STREAM, which is a real "
            "stream to the driver and is accepted. Pass an explicit stream or "
            "PER_THREAD_DEFAULT_STREAM."
        )

    cdef cydriver.CUstreamCaptureStatus _cap_status
    IF CUDA_CORE_BUILD_MAJOR >= 13:
        HANDLE_RETURN(cydriver.cuStreamGetCaptureInfo(as_cu(s._h_stream), &_cap_status,
                                                      NULL, NULL, NULL, NULL, NULL))
    ELSE:
        HANDLE_RETURN(cydriver.cuStreamGetCaptureInfo(as_cu(s._h_stream), &_cap_status,
                                                      NULL, NULL, NULL, NULL))
    if _cap_status == cydriver.CU_STREAM_CAPTURE_STATUS_ACTIVE:
        raise TypeError(
            "copy_batch does not support graph capture; "
            "use GraphNode.memcpy or per-buffer Buffer.copy_to instead"
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

    _do_copy_batch(src_bufs, dst_bufs, s, attr_tuple)


cdef void _do_copy_batch(tuple src_bufs, tuple dst_bufs, Stream s, tuple attr_tuple):
    IF CUDA_CORE_BUILD_MAJOR >= 13:
        # Building against CUDA 13 headers says nothing about the installed
        # driver, so the run-time version still has to be checked before
        # calling a 13.0-only entry point (see PRs #2054 / #2064).
        if _batch_entry_point_available():
            _do_copy_batch_native(src_bufs, dst_bufs, s, attr_tuple)
        else:
            _reject_during_api_call_fallback(attr_tuple)
            _do_copy_batch_loop(src_bufs, dst_bufs, s)
    ELSE:
        _reject_during_api_call_fallback(attr_tuple)
        _do_copy_batch_loop(src_bufs, dst_bufs, s)


cdef void _reject_during_api_call_fallback(tuple attr_tuple):
    """Raise before the per-copy cuMemcpyAsync loop if any copy needs
    DURING_API_CALL, which that fallback cannot honor (see
    _reject_unsupported_during_api_call for why this must raise rather than
    silently ignore the option, unlike STREAM and ANY).
    """
    cdef Py_ssize_t i
    for i in range(len(attr_tuple)):
        _reject_unsupported_during_api_call(
            (<object>attr_tuple[i]).src_access_order,
            "cuda.core built against CUDA 13 headers and cuda.bindings/driver "
            "13.0 or newer (cuMemcpyBatchAsync is unavailable here)",
            index=i,
        )


cdef void _do_copy_batch_loop(tuple src_bufs, tuple dst_bufs, Stream s):
    """Per-copy cuMemcpyAsync fallback where the batch entry point is absent.

    Issues copies one at a time, so the performance benefit of batching is
    not realized. STREAM and ANY are silently ignored here (satisfied by
    stream-ordered cuMemcpyAsync regardless); DURING_API_CALL is rejected by
    _reject_during_api_call_fallback before this is ever called.
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
