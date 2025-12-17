# SPDX-FileCopyrightText: Copyright (c) 2024-2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

from libc.stdint cimport uintptr_t

from cuda.bindings cimport cydriver

from cuda.core._launch_config cimport LaunchConfig
from cuda.core._kernel_arg_handler cimport ParamHolder
from cuda.core._resource_handles cimport native
from cuda.core._stream cimport Stream_accept, Stream
from cuda.core._utils.cuda_utils cimport (
    check_or_create_options,
    HANDLE_RETURN,
)

import threading

from cuda.core._module import Kernel
from cuda.core._stream import Stream
from cuda.core._utils.cuda_utils import (
    _reduce_3_tuple,
    get_binding_version,
)


cdef bint _inited = False
cdef bint _use_ex = False
cdef object _lock = threading.Lock()


cdef int _lazy_init() except?-1:
    global _inited, _use_ex
    if _inited:
        return 0

    cdef int _driver_ver
    with _lock:
        if _inited:
            return 0

        # binding availability depends on cuda-python version
        _py_major_minor = get_binding_version()
        HANDLE_RETURN(cydriver.cuDriverGetVersion(&_driver_ver))
        _use_ex = (_driver_ver >= 11080) and (_py_major_minor >= (11, 8))
        _inited = True

    return 0


def launch(stream: Stream | GraphBuilder | IsStreamT, config: LaunchConfig, kernel: Kernel, *kernel_args):
    """Launches a :obj:`~_module.Kernel`
    object with launch-time configuration.

    Parameters
    ----------
    stream : :obj:`~_stream.Stream` | :obj:`~_graph.GraphBuilder`
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
    _lazy_init()
    cdef LaunchConfig conf = check_or_create_options(LaunchConfig, config, "launch config")

    # TODO: can we ensure kernel_args is valid/safe to use here?
    # TODO: merge with HelperKernelParams?
    cdef ParamHolder ker_args = ParamHolder(kernel_args)
    cdef void** args_ptr = <void**><uintptr_t>(ker_args.ptr)

    # TODO: cythonize Module/Kernel/...
    # Note: CUfunction and CUkernel are interchangeable
    cdef cydriver.CUfunction func_handle = <cydriver.CUfunction>(<uintptr_t>(kernel._handle))

    # Note: CUkernel can still be launched via the old cuLaunchKernel and we do not care
    # about the CUfunction/CUkernel difference (which depends on whether the "old" or
    # "new" module loading APIs are in use). We check both binding & driver versions here
    # mainly to see if the "Ex" API is available and if so we use it, as it's more feature
    # rich.
    if _use_ex:
        drv_cfg = conf._to_native_launch_config()
        drv_cfg.hStream = native(s._h_stream)
        if conf.cooperative_launch:
            _check_cooperative_launch(kernel, conf, s)
        with nogil:
            HANDLE_RETURN(cydriver.cuLaunchKernelEx(&drv_cfg, func_handle, args_ptr, NULL))
    else:
        # TODO: check if config has any unsupported attrs
        HANDLE_RETURN(
            cydriver.cuLaunchKernel(
                func_handle,
                conf.grid[0], conf.grid[1], conf.grid[2],
                conf.block[0], conf.block[1], conf.block[2],
                conf.shmem_size, native(s._h_stream), args_ptr, NULL
            )
        )


cdef _check_cooperative_launch(kernel: Kernel, config: LaunchConfig, stream: Stream):
    dev = stream.device
    num_sm = dev.properties.multiprocessor_count
    max_grid_size = (
        kernel.occupancy.max_active_blocks_per_multiprocessor(_reduce_3_tuple(config.block), config.shmem_size) * num_sm
    )
    if _reduce_3_tuple(config.grid) > max_grid_size:
        # For now let's try not to be smart and adjust the grid size behind users' back.
        # We explicitly ask users to adjust.
        x, y, z = config.grid
        raise ValueError(f"The specified grid size ({x} * {y} * {z}) exceeds the limit ({max_grid_size})")
