# SPDX-FileCopyrightText: Copyright (c) 2024-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

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
from cuda.core._utils.cuda_utils import _reduce_3_tuple


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
    cdef LaunchConfig conf = check_or_create_options(LaunchConfig, config, "launch config")

    # TODO: can we ensure kernel_args is valid/safe to use here?
    # TODO: merge with HelperKernelParams?
    cdef ParamHolder ker_args = ParamHolder(kernel_args)
    cdef void** args_ptr = <void**><uintptr_t>(ker_args.ptr)

    cdef Kernel ker = <Kernel>kernel
    cdef cydriver.CUfunction func_handle = <cydriver.CUfunction>as_cu(ker._h_kernel)

    drv_cfg = conf._to_native_launch_config()
    drv_cfg.hStream = as_cu(s._h_stream)
    if conf.cooperative_launch:
        _check_cooperative_launch(kernel, conf, s)
    with nogil:
        HANDLE_RETURN(cydriver.cuLaunchKernelEx(&drv_cfg, func_handle, args_ptr, NULL))


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
