# Copyright (c) 2024-2025, NVIDIA CORPORATION & AFFILIATES. ALL RIGHTS RESERVED.
#
# SPDX-License-Identifier: Apache-2.0


from cuda.core.experimental._kernel_arg_handler import ParamHolder
from cuda.core.experimental._launch_config import LaunchConfig, _to_native_launch_config
from cuda.core.experimental._module import Kernel
from cuda.core.experimental._stream import Stream
from cuda.core.experimental._utils.clear_error_support import assert_type
from cuda.core.experimental._utils.cuda_utils import (
    check_or_create_options,
    driver,
    get_binding_version,
    handle_return,
)

# TODO: revisit this treatment for py313t builds
_inited = False
_use_ex = None


def _lazy_init():
    global _inited
    if _inited:
        return

    global _use_ex
    # binding availability depends on cuda-python version
    _py_major_minor = get_binding_version()
    _driver_ver = handle_return(driver.cuDriverGetVersion())
    _use_ex = (_driver_ver >= 11080) and (_py_major_minor >= (11, 8))
    _inited = True


def launch(stream, config, kernel, *kernel_args):
    """Launches a :obj:`~_module.Kernel`
    object with launch-time configuration.

    Parameters
    ----------
    stream : :obj:`~_stream.Stream`
        The stream establishing the stream ordering semantic of a
        launch.
    config : :obj:`~_launcher.LaunchConfig`
        Launch configurations inline with options provided by
        :obj:`~_launcher.LaunchConfig` dataclass.
    kernel : :obj:`~_module.Kernel`
        Kernel to launch.
    *kernel_args : Any
        Variable length argument list that is provided to the
        launching kernel.

    """
    if stream is None:
        raise ValueError("stream cannot be None, stream must either be a Stream object or support __cuda_stream__")
    if not isinstance(stream, Stream):
        try:
            stream = Stream._init(stream)
        except Exception as e:
            raise ValueError(
                f"stream must either be a Stream object or support __cuda_stream__ (got {type(stream)})"
            ) from e
    assert_type(kernel, Kernel)
    _lazy_init()
    config = check_or_create_options(LaunchConfig, config, "launch config")

    # TODO: can we ensure kernel_args is valid/safe to use here?
    # TODO: merge with HelperKernelParams?
    kernel_args = ParamHolder(kernel_args)
    args_ptr = kernel_args.ptr

    # Note: CUkernel can still be launched via the old cuLaunchKernel and we do not care
    # about the CUfunction/CUkernel difference (which depends on whether the "old" or
    # "new" module loading APIs are in use). We check both binding & driver versions here
    # mainly to see if the "Ex" API is available and if so we use it, as it's more feature
    # rich.
    if _use_ex:
        drv_cfg = _to_native_launch_config(config)
        drv_cfg.hStream = stream.handle
        handle_return(driver.cuLaunchKernelEx(drv_cfg, int(kernel._handle), args_ptr, 0))
    else:
        # TODO: check if config has any unsupported attrs
        handle_return(
            driver.cuLaunchKernel(
                int(kernel._handle), *config.grid, *config.block, config.shmem_size, stream.handle, args_ptr, 0
            )
        )
