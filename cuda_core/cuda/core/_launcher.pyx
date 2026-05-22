# SPDX-FileCopyrightText: Copyright (c) 2024-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

from cpython.ref cimport Py_INCREF

from libc.stdint cimport uintptr_t
from libc.stdlib cimport malloc, free
from libc.string cimport memcpy as c_memcpy

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


cdef extern from "Python.h":
    void _py_decref "Py_DECREF" (void*)


cdef struct _HostLaunchCFuncState:
    cydriver.CUhostFn fn
    void* user_data
    bint owns_user_data
    void* fn_pyobj


cdef void _py_host_launch_trampoline(void* data) noexcept with gil:
    try:
        (<object>data)()
    finally:
        _py_decref(data)


cdef void _ctypes_host_launch_trampoline(void* data) noexcept with gil:
    cdef _HostLaunchCFuncState* state = <_HostLaunchCFuncState*>data

    try:
        state.fn(state.user_data)
    finally:
        if state.fn_pyobj != NULL:
            _py_decref(state.fn_pyobj)
        if state.owns_user_data and state.user_data != NULL:
            free(state.user_data)
        free(state)


cdef void _cleanup_host_launch(
        cydriver.CUhostFn fn, void* user_data) except *:
    cdef _HostLaunchCFuncState* state

    if fn == <cydriver.CUhostFn>_py_host_launch_trampoline:
        if user_data != NULL:
            _py_decref(user_data)
        return

    if fn == <cydriver.CUhostFn>_ctypes_host_launch_trampoline:
        state = <_HostLaunchCFuncState*>user_data
        if state != NULL:
            if state.fn_pyobj != NULL:
                _py_decref(state.fn_pyobj)
            if state.owns_user_data and state.user_data != NULL:
                free(state.user_data)
            free(state)


cdef void _prepare_host_launch(
        object fn, object user_data,
        cydriver.CUhostFn* out_fn, void** out_user_data) except *:
    import ctypes as ct

    cdef void* fn_pyobj = NULL
    cdef _HostLaunchCFuncState* state = NULL
    cdef bytes buf

    if isinstance(fn, ct._CFuncPtr):
        state = <_HostLaunchCFuncState*>malloc(sizeof(_HostLaunchCFuncState))
        if state == NULL:
            raise MemoryError("failed to allocate host callback state")

        state.fn = <cydriver.CUhostFn><uintptr_t>ct.cast(fn, ct.c_void_p).value
        state.user_data = NULL
        state.owns_user_data = False
        state.fn_pyobj = NULL

        # Keep the ctypes wrapper alive until the driver invokes the callback.
        Py_INCREF(fn)
        fn_pyobj = <void*>fn
        state.fn_pyobj = fn_pyobj

        out_fn[0] = <cydriver.CUhostFn>_ctypes_host_launch_trampoline

        if user_data is None:
            pass
        elif isinstance(user_data, int):
            state.user_data = <void*><uintptr_t>user_data
        else:
            buf = bytes(user_data)
            state.user_data = malloc(len(buf))
            if state.user_data == NULL:
                _py_decref(fn_pyobj)
                free(state)
                raise MemoryError("failed to allocate user_data buffer")
            c_memcpy(state.user_data, <const char*>buf, len(buf))
            state.owns_user_data = True
        out_user_data[0] = <void*>state
    else:
        if user_data is not None:
            raise ValueError("user_data is only supported with ctypes function pointers")
        if not callable(fn):
            raise TypeError("fn must be callable")
        # The driver only sees a void* payload, so we transfer the Python
        # callable itself as callback state and release it in the trampoline.
        Py_INCREF(fn)
        fn_pyobj = <void*>fn
        out_fn[0] = <cydriver.CUhostFn>_py_host_launch_trampoline
        out_user_data[0] = fn_pyobj


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
    cdef cydriver.CUhostFn c_fn
    cdef void* c_user_data = NULL

    _prepare_host_launch(fn, user_data, &c_fn, &c_user_data)
    try:
        with nogil:
            HANDLE_RETURN(cydriver.cuLaunchHostFunc(
                as_cu(s._h_stream), c_fn, c_user_data))
    except Exception:
        _cleanup_host_launch(c_fn, c_user_data)
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
