from dataclasses import dataclass
from typing import Optional, Union

import numpy as np

from cuda import cuda, cudart
from cuda.py._utils import CUDAError, check_or_create_options, handle_return
from cuda.py._memory import Buffer
from cuda.py._module import Kernel
from cuda.py._stream import Stream


@dataclass
class LaunchConfig:
    """
    """
    grid: Union[tuple, int] = None
    block: Union[tuple, int] = None
    stream: Stream = None
    shmem_size: Optional[int] = None

    def __post_init__(self):
        self.grid = self._cast_to_3_tuple(self.grid)
        self.block = self._cast_to_3_tuple(self.block)
        # we handle "stream=None" in the launch API
        if self.stream is not None:
            if not isinstance(self.stream, Stream):
                try:
                    self.stream = Stream(self.stream)
                except Exception as e:
                    raise ValueError(
                        "stream must either be a Stream object "
                        "or support __cuda_stream__") from e
        if self.shmem_size is None:
            self.shmem_size = 0

    def _cast_to_3_tuple(self, cfg):
        if isinstance(cfg, int):
            if cfg < 1:
                raise ValueError
            return (cfg, 1, 1)
        elif isinstance(cfg, tuple):
            size = len(cfg)
            if size == 1:
                cfg = cfg[0]
                if cfg < 1:
                    raise ValueError
                return (cfg, 1, 1)
            elif size == 2:
                if cfg[0] < 1 or cfg[1] < 1:
                    raise ValueError
                return (*cfg, 1)
            elif size == 3:
                if cfg[0] < 1 or cfg[1] < 1 or cfg[2] < 1:
                    raise ValueError
                return cfg
        else:
            raise ValueError


def launch(kernel, config, *kernel_args):
    if not isinstance(kernel, Kernel):
        raise ValueError
    config = check_or_create_options(LaunchConfig, config, "launch config")
    # TODO: can we ensure kernel_args is valid/safe to use here?

    driver_ver = handle_return(cuda.cuDriverGetVersion())
    if driver_ver >= 12000:
        drv_cfg = cuda.CUlaunchConfig()
        drv_cfg.gridDimX, drv_cfg.gridDimY, drv_cfg.gridDimZ = config.grid
        drv_cfg.blockDimX, drv_cfg.blockDimY, drv_cfg.blockDimZ = config.block
        if config.stream is None:
            raise CUDAError("stream cannot be None")
        drv_cfg.hStream = config.stream._handle
        drv_cfg.sharedMemBytes = config.shmem_size
        drv_cfg.numAttrs = 0  # FIXME

        # TODO: merge with HelperKernelParams?
        num_args = len(kernel_args)
        args_ptr = 0
        if num_args:
            # FIXME: support args passed by value
            args = np.empty(num_args, dtype=np.intp)
            for i, arg in enumerate(kernel_args):
                if isinstance(arg, Buffer):
                    # this is super weird... we need the address of where the actual
                    # buffer address is stored...
                    args[i] = arg.ptr.getPtr()
                else:
                    raise NotImplementedError
            args_ptr = args.ctypes.data

        handle_return(cuda.cuLaunchKernelEx(
            drv_cfg, int(kernel._handle), args_ptr, 0))
    else:
        raise NotImplementedError("TODO")
