# SPDX-FileCopyrightText: Copyright (c) 2024-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

from libc.stdint cimport uintptr_t

from cuda.bindings cimport cydriver

from cuda.core._host_callback cimport _resolve_host_callback
from cuda.core._launch_config cimport LaunchConfig
from cuda.core._kernel_arg_handler cimport ParamHolder
from cuda.core._module cimport Kernel
from cuda.core._rt cimport as_cu, launch_host_function, OpaqueHandle
from cuda.core._stream cimport Stream_accept, Stream
from cuda.core._utils.cuda_utils cimport (
    check_or_create_options,
    HANDLE_RETURN,
)
from cuda.core._module import Kernel
from cuda.core._stream import Stream
from math import prod
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from cuda.core.graph import GraphBuilder
    from cuda.core.typing import IsStreamType

__all__ = ['launch', 'host_launch']


def launch(
    stream: Stream | GraphBuilder | IsStreamType,
    LaunchConfig config not None: LaunchConfig,
    Kernel kernel not None: Kernel,
    *kernel_args
) -> None:
    """Launches a :obj:`~_module.Kernel`
    object with launch-time configuration.

    Parameters
    ----------
    stream : :obj:`~_stream.Stream` | :obj:`~graph.GraphBuilder`
        The stream establishing the stream ordering semantic of a
        launch.
    config : :obj:`LaunchConfig`
        Launch configurations inline with options provided by
        :obj:`~_launcher.LaunchConfig` dataclass.
    kernel : :obj:`~_module.Kernel`
        Kernel to launch.
    *kernel_args : Any
        Variable length argument list that is provided to the
        launching kernel.

    """
    cdef Stream s = Stream_accept(stream, allow_stream_protocol=True)
    cdef LaunchConfig conf = check_or_create_options(LaunchConfig, config, "launch config")

    # TODO: can we ensure kernel_args is valid/safe to use here?
    # TODO: merge with HelperKernelParams?
    cdef ParamHolder ker_args = ParamHolder(kernel_args)
    cdef void** args_ptr = <void**><uintptr_t>(ker_args.ptr)

    cdef cydriver.CUfunction func_handle = <cydriver.CUfunction>as_cu(kernel._h_kernel)

    drv_cfg = conf._to_native_launch_config()
    drv_cfg.hStream = as_cu(s._h_stream)
    if conf.is_cooperative:
        _check_cooperative_launch(kernel, conf, s)
    with nogil:
        HANDLE_RETURN(cydriver.cuLaunchKernelEx(&drv_cfg, func_handle, args_ptr, NULL))


def host_launch(stream: Stream | GraphBuilder | IsStreamType, fn, *, user_data=None) -> None:
    """Enqueue a host function on a stream in stream order.

    The function runs on a driver-owned thread once all work enqueued on the
    stream before this call has completed. Work enqueued after this call waits
    until the function returns. This call returns when the function is enqueued;
    observe completion with :meth:`Stream.sync` or an :class:`Event` recorded
    after it.

    Passing a :class:`~graph.GraphBuilder` forwards to
    :meth:`~graph.GraphBuilder.callback`.

    Parameters
    ----------
    stream : Stream or graph.GraphBuilder or IsStreamType
        The stream establishing execution order, or a graph builder capturing
        the callback.
    fn : callable or ctypes function pointer
        A Python callable taking no arguments, which runs with the GIL, or a
        ctypes function pointer declared as ``CUhostFn``, for example
        ``ctypes.CFUNCTYPE(None, ctypes.c_void_p)``, which ``cuda.core``
        passes to the driver as is. A pointer to compiled code runs without
        the GIL; a ctypes callback that wraps a Python function acquires the
        GIL through ctypes and is subject to the same deadlock rule as a
        Python callable. ``cuda.core`` keeps ``fn`` alive until the function
        has run and releases it afterwards on the main thread.
    user_data : int or bytes-like, optional
        Only for ctypes function pointers. An ``int`` is a raw pointer that
        the caller must keep valid until the function has run. Bytes-like data
        is copied; ``cuda.core`` owns the copy until the function has run and
        releases it afterwards on the main thread.

    Raises
    ------
    TypeError
        If ``stream`` is ``None`` or not a stream, or if ``fn`` is not
        callable or its ctypes prototype is not ``CUhostFn``.
    ValueError
        If ``user_data`` is given for a Python callable.
    RuntimeError
        If ``stream`` is a :class:`Stream` that is capturing a graph, or a
        :class:`~graph.GraphBuilder` that is closed or not building.

    Notes
    -----
    An exception raised by a Python callable cannot propagate to the caller.
    It is reported through :func:`sys.unraisablehook` and the stream continues.

    .. warning::

        The function must not call CUDA, directly or through a library. The
        driver does not permit such calls. Such a call may fail with an error,
        or it may deadlock.

        A Python callable, or a ctypes callback that wraps a Python function,
        acquires the GIL on the driver's thread. A thread
        that holds the GIL while blocking inside a CUDA call waiting for this
        stream therefore deadlocks with the callback. ``cuda.core`` and
        ``cuda.bindings`` release the GIL around CUDA calls. A library that
        calls CUDA with the GIL held must not be used to wait on a stream with
        a pending Python callback.
    """
    from cuda.core.graph._graph_builder import GraphBuilder

    if isinstance(stream, GraphBuilder):
        return stream.callback(fn, user_data=user_data)

    cdef Stream s = Stream_accept(stream, allow_stream_protocol=True)
    cdef cydriver.CUstream c_stream = as_cu(s._h_stream)
    cdef cydriver.CUstreamCaptureStatus status
    with nogil:
        HANDLE_RETURN(cydriver.cuStreamIsCapturing(c_stream, &status))
    if status == cydriver.CU_STREAM_CAPTURE_STATUS_ACTIVE:
        raise RuntimeError(
            "host_launch cannot enqueue on a stream that is capturing a graph; "
            "pass the GraphBuilder, or call GraphBuilder.callback")

    cdef cydriver.CUhostFn c_fn
    cdef void* c_user_data = NULL
    cdef OpaqueHandle fn_owner, data_owner
    _resolve_host_callback(fn, user_data, &c_fn, &c_user_data, &fn_owner, &data_owner)
    with nogil:
        HANDLE_RETURN(launch_host_function(c_stream, c_fn, c_user_data, fn_owner, data_owner))
    return None


cdef _check_cooperative_launch(kernel: Kernel, config: LaunchConfig, stream: Stream):
    dev = stream.device
    num_sm = dev.properties.multiprocessor_count
    max_grid_size = (
        kernel.occupancy.max_active_blocks_per_multiprocessor(prod(config.block), config.shmem_size) * num_sm
    )
    if prod(config.grid) > max_grid_size:
        # For now let's try not to be smart and adjust the grid size behind users' back.
        # We explicitly ask users to adjust.
        x, y, z = config.grid
        raise ValueError(f"The specified grid size ({x} * {y} * {z}) exceeds the limit ({max_grid_size})")
