# SPDX-FileCopyrightText: Copyright (c) 2024-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

import ctypes
import itertools
import threading

from libc.stdint cimport uintptr_t

from cuda.bindings cimport cydriver

from cuda.core._launch_config cimport LaunchConfig
from cuda.core._kernel_arg_handler cimport ParamHolder
from cuda.core._module cimport Kernel
from cuda.core._resource_handles cimport as_cu
from cuda.core._stream cimport Stream_accept, Stream
from cuda.core._utils.cuda_utils cimport (
    check_or_create_options,
    HANDLE_RETURN,
)
from cuda.core._module import Kernel
from cuda.core._stream import Stream
from math import prod


class _PendingHostLaunch:
    __slots__ = ("_fn", "_payload", "_stream")

    def __init__(self, stream, fn, payload):
        self._fn = fn
        self._payload = payload
        self._stream = stream

    def invoke(self):
        if isinstance(self._fn, ctypes._CFuncPtr):
            if self._payload is None:
                self._fn(None)
            elif isinstance(self._payload, int):
                self._fn(ctypes.c_void_p(self._payload))
            else:
                self._fn(ctypes.c_void_p(ctypes.addressof(self._payload)))
            return

        self._fn()


_pending_host_launches = {}
_pending_host_launches_lock = threading.Lock()
_pending_host_launch_tokens = itertools.count(1)


cdef void _host_launch_trampoline(void* data) noexcept with gil:
    cdef object pending

    with _pending_host_launches_lock:
        pending = _pending_host_launches.pop(<uintptr_t>data, None)
    if pending is None:
        return

    pending.invoke()


def _register_host_launch(stream, fn, user_data):
    if isinstance(fn, ctypes._CFuncPtr):
        if user_data is None:
            payload = None
        elif isinstance(user_data, int):
            payload = user_data
        else:
            payload = ctypes.create_string_buffer(bytes(user_data))
    else:
        if user_data is not None:
            raise ValueError("user_data is only supported with ctypes function pointers")
        if not callable(fn):
            raise TypeError("fn must be callable")
        payload = None

    pending = _PendingHostLaunch(stream, fn, payload)
    token = next(_pending_host_launch_tokens)
    with _pending_host_launches_lock:
        _pending_host_launches[token] = pending
    return token


def _discard_host_launch(token):
    with _pending_host_launches_lock:
        _pending_host_launches.pop(token, None)


def launch(stream: Stream | GraphBuilder | IsStreamType, config: LaunchConfig, kernel: Kernel, *kernel_args):
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

    cdef Kernel ker = <Kernel>kernel
    cdef cydriver.CUfunction func_handle = <cydriver.CUfunction>as_cu(ker._h_kernel)

    drv_cfg = conf._to_native_launch_config()
    drv_cfg.hStream = as_cu(s._h_stream)
    if conf.is_cooperative:
        _check_cooperative_launch(kernel, conf, s)
    with nogil:
        HANDLE_RETURN(cydriver.cuLaunchKernelEx(&drv_cfg, func_handle, args_ptr, NULL))


def host_launch(stream: Stream | IsStreamType, fn, *, user_data=None):
    """Enqueue a host callback onto a CUDA stream.

    The callback runs on a host thread after all previously enqueued work on
    ``stream`` completes. Future work added to the same stream will not begin
    until the callback returns.

    Two callback forms are supported:

    - **Python callable**: pass any callable that takes no arguments.
    - **ctypes function pointer**: pass a ``ctypes.CFUNCTYPE``-style callable.
      The callback receives a single ``void*`` argument. ``user_data`` may be
      an ``int`` pointer value or bytes-like object.

    .. warning::

        Host callbacks must not call CUDA APIs directly or indirectly. CUDA
        driver callbacks run on an internal host thread. If that thread blocks
        waiting for the GIL while another Python thread holds the GIL and is
        blocked in a CUDA call, the process can deadlock through CUDA-driver /
        GIL lock-order inversion. CUDA-using Python libraries should release
        the GIL around CUDA calls, and host callbacks must avoid re-entering
        CUDA from Python.

    Parameters
    ----------
    stream : :obj:`~_stream.Stream`
        The stream that establishes the callback ordering.
    fn : callable or ctypes function pointer
        The callback function.
    user_data : int or bytes-like, optional
        Only for ctypes function pointers. If ``int``, passed as a raw pointer
        value. If bytes-like, the payload is copied and passed as a ``void*``
        pointer to the copied buffer.
    """
    cdef Stream s = Stream_accept(stream, allow_stream_protocol=True)
    cdef uintptr_t token = _register_host_launch(s, fn, user_data)

    try:
        with nogil:
            HANDLE_RETURN(cydriver.cuLaunchHostFunc(
                as_cu(s._h_stream),
                <cydriver.CUhostFn>_host_launch_trampoline,
                <void*>token,
            ))
    except Exception:
        _discard_host_launch(token)
        raise


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
