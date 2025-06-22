# Copyright (c) 2025, NVIDIA CORPORATION & AFFILIATES. ALL RIGHTS RESERVED.
#
# SPDX-License-Identifier: Apache-2.0

from dataclasses import dataclass
from typing import Optional, Union

from cuda.core.experimental._device import Device
from cuda.core.experimental._utils.cuda_utils import (
    CUDAError,
    cast_to_3_tuple,
    driver,
    get_binding_version,
    handle_return,
)

# TODO: revisit this treatment for py313t builds
_inited = False


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
    shmem_size : int, optional
        Dynamic shared-memory size per thread block in bytes.
        (Default to size 0)
    cooperative_launch : bool, optional
        Whether this config can be used to launch a cooperative kernel.
    """

    # TODO: expand LaunchConfig to include other attributes
    grid: Union[tuple, int] = None
    cluster: Union[tuple, int] = None
    block: Union[tuple, int] = None
    shmem_size: Optional[int] = None
    cooperative_launch: Optional[bool] = False

    def __post_init__(self):
        _lazy_init()
        self.grid = cast_to_3_tuple("LaunchConfig.grid", self.grid)
        self.block = cast_to_3_tuple("LaunchConfig.block", self.block)
        # FIXME: Calling Device() strictly speaking is not quite right; we should instead
        # look up the device from stream. We probably need to defer the checks related to
        # device compute capability or attributes.
        # thread block clusters are supported starting H100
        if self.cluster is not None:
            if not _use_ex:
                err, drvers = driver.cuDriverGetVersion()
                drvers_fmt = f" (got driver version {drvers})" if err == driver.CUresult.CUDA_SUCCESS else ""
                raise CUDAError(f"thread block clusters require cuda.bindings & driver 11.8+{drvers_fmt}")
            cc = Device().compute_capability
            if cc < (9, 0):
                raise CUDAError(
                    f"thread block clusters are not supported on devices with compute capability < 9.0 (got {cc})"
                )
            self.cluster = cast_to_3_tuple("LaunchConfig.cluster", self.cluster)
        if self.shmem_size is None:
            self.shmem_size = 0
        if self.cooperative_launch and not Device().properties.cooperative_launch:
            raise CUDAError("cooperative kernels are not supported on this device")


def _to_native_launch_config(config: LaunchConfig) -> driver.CUlaunchConfig:
    _lazy_init()
    drv_cfg = driver.CUlaunchConfig()
    drv_cfg.gridDimX, drv_cfg.gridDimY, drv_cfg.gridDimZ = config.grid
    drv_cfg.blockDimX, drv_cfg.blockDimY, drv_cfg.blockDimZ = config.block
    drv_cfg.sharedMemBytes = config.shmem_size
    attrs = []  # TODO: support more attributes
    if config.cluster:
        attr = driver.CUlaunchAttribute()
        attr.id = driver.CUlaunchAttributeID.CU_LAUNCH_ATTRIBUTE_CLUSTER_DIMENSION
        dim = attr.value.clusterDim
        dim.x, dim.y, dim.z = config.cluster
        attrs.append(attr)
    if config.cooperative_launch:
        attr = driver.CUlaunchAttribute()
        attr.id = driver.CUlaunchAttributeID.CU_LAUNCH_ATTRIBUTE_COOPERATIVE
        attr.value.cooperative = 1
        attrs.append(attr)
    drv_cfg.numAttrs = len(attrs)
    drv_cfg.attrs = attrs
    return drv_cfg
