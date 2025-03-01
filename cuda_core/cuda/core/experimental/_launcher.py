# Copyright (c) 2024, NVIDIA CORPORATION & AFFILIATES. ALL RIGHTS RESERVED.
#
# SPDX-License-Identifier: LicenseRef-NVIDIA-SOFTWARE-LICENSE

from dataclasses import dataclass
from typing import Optional, Union

from cuda.core.experimental._clear_error_support import assert_type
from cuda.core.experimental._device import Device
from cuda.core.experimental._kernel_arg_handler import ParamHolder
from cuda.core.experimental._module import Kernel
from cuda.core.experimental._stream import Stream
from cuda.core.experimental._utils import CUDAError, check_or_create_options, driver, get_binding_version, handle_return

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


@dataclass
class LaunchConfig:
    """Customizable launch options.

    Attributes
    ----------
    grid : Union[tuple, int]
        Collection of threads that will execute a kernel function.
    cluster : Union[tuple, int]
        Group of blocks (Thread Block Cluster) that will execute on the same
        GPU Processing Cluster (GPC). Blocks within a cluster have access to
        distributed shared memory and can be explicitly synchronized.
    block : Union[tuple, int]
        Group of threads (Thread Block) that will execute on the same
        streaming multiprocessor (SM). Threads within a thread blocks have
        access to shared memory and can be explicitly synchronized.
    stream : :obj:`~_stream.Stream`
        The stream establishing the stream ordering semantic of a
        launch.
    shmem_size : int, optional
        Dynamic shared-memory size per thread block in bytes.
        (Default to size 0)

    """

    # TODO: expand LaunchConfig to include other attributes
    grid: Union[tuple, int] = None
    cluster: Union[tuple, int] = None
    block: Union[tuple, int] = None
    stream: Stream = None
    shmem_size: Optional[int] = None

    def __post_init__(self):
        _lazy_init()
        self.grid = self._cast_to_3_tuple(self.grid)
        self.block = self._cast_to_3_tuple(self.block)
        # thread block clusters are supported starting H100
        if self.cluster is not None:
            if not _use_ex:
                raise CUDAError("thread block clusters require cuda.bindings & driver 11.8+") # ACTNBL show driver version HAPPY_ONLY_EXERCISED
            if Device().compute_capability < (9, 0):
                raise CUDAError("thread block clusters are not supported on devices with compute capability < 9.0") # ACTNBL show cc version HAPPY_ONLY_EXERCISED
            self.cluster = self._cast_to_3_tuple(self.cluster)
        # we handle "stream=None" in the launch API
        if self.stream is not None and not isinstance(self.stream, Stream):
            try:
                self.stream = Stream._init(self.stream)
            except Exception as e:
                raise ValueError("stream must either be a Stream object or support __cuda_stream__") from e # ACTNBL show type(self.stream) BUT should raise from Stream._init() UNHAPPY_EXERCISED
        if self.shmem_size is None:
            self.shmem_size = 0

    def _cast_to_3_tuple(self, cfg):
        # ACTNBL rewrite UNHAPPY_EXERCISED
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
    """Launches a :obj:`~_module.Kernel`
    object with launch-time configuration.

    Parameters
    ----------
    kernel : :obj:`~_module.Kernel`
        Kernel to launch.
    config : :obj:`~_launcher.LaunchConfig`
        Launch configurations inline with options provided by
        :obj:`~_launcher.LaunchConfig` dataclass.
    *kernel_args : Any
        Variable length argument list that is provided to the
        launching kernel.

    """
    assert_type(kernel, Kernel)
    config = check_or_create_options(LaunchConfig, config, "launch config")
    if config.stream is None:
        raise CUDAError("stream cannot be None") # ACTNBL "config.stream cannot be None" FN_NOT_CALLED

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
        drv_cfg = driver.CUlaunchConfig()
        drv_cfg.gridDimX, drv_cfg.gridDimY, drv_cfg.gridDimZ = config.grid
        drv_cfg.blockDimX, drv_cfg.blockDimY, drv_cfg.blockDimZ = config.block
        drv_cfg.hStream = config.stream.handle
        drv_cfg.sharedMemBytes = config.shmem_size
        attrs = []  # TODO: support more attributes
        if config.cluster:
            attr = driver.CUlaunchAttribute()
            attr.id = driver.CUlaunchAttributeID.CU_LAUNCH_ATTRIBUTE_CLUSTER_DIMENSION
            dim = attr.value.clusterDim
            dim.x, dim.y, dim.z = config.cluster
            attrs.append(attr)
        drv_cfg.numAttrs = len(attrs)
        drv_cfg.attrs = attrs
        handle_return(driver.cuLaunchKernelEx(drv_cfg, int(kernel._handle), args_ptr, 0))
    else:
        # TODO: check if config has any unsupported attrs
        handle_return(
            driver.cuLaunchKernel(
                int(kernel._handle), *config.grid, *config.block, config.shmem_size, config.stream._handle, args_ptr, 0
            )
        )
