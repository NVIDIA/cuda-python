# Copyright (c) 2024-2025, NVIDIA CORPORATION & AFFILIATES. ALL RIGHTS RESERVED.
#
# SPDX-License-Identifier: Apache-2.0

import threading
from typing import Optional, Union

from cuda.core.experimental._context import Context, ContextOptions
from cuda.core.experimental._event import Event, EventOptions
from cuda.core.experimental._graph import GraphBuilder
from cuda.core.experimental._memory import Buffer, DeviceMemoryResource, MemoryResource, _SynchronousMemoryResource
from cuda.core.experimental._stream import IsStreamT, Stream, StreamOptions, default_stream
from cuda.core.experimental._utils.clear_error_support import assert_type
from cuda.core.experimental._utils.cuda_utils import (
    ComputeCapability,
    CUDAError,
    _check_driver_error,
    driver,
    handle_return,
    precondition,
    runtime,
)

_tls = threading.local()
_lock = threading.Lock()
_is_cuInit = False


class DeviceProperties:
    """
    A class to query various attributes of a CUDA device.

    Attributes are read-only and provide information about the device.
    """

    def __new__(self, *args, **kwargs):
        raise RuntimeError("DeviceProperties cannot be instantiated directly. Please use Device APIs.")

    __slots__ = ("_handle", "_cache")

    @classmethod
    def _init(cls, handle):
        self = super().__new__(cls)
        self._handle = handle
        self._cache = {}
        return self

    def _get_attribute(self, attr):
        """Retrieve the attribute value directly from the driver."""
        return handle_return(driver.cuDeviceGetAttribute(attr, self._handle))

    def _get_cached_attribute(self, attr):
        """Retrieve the attribute value, using cache if applicable."""
        if attr not in self._cache:
            self._cache[attr] = self._get_attribute(attr)
        return self._cache[attr]

    @property
    def max_threads_per_block(self) -> int:
        """
        int: Maximum number of threads per block.
        """
        return self._get_cached_attribute(driver.CUdevice_attribute.CU_DEVICE_ATTRIBUTE_MAX_THREADS_PER_BLOCK)

    @property
    def max_block_dim_x(self) -> int:
        """
        int: Maximum x-dimension of a block.
        """
        return self._get_cached_attribute(driver.CUdevice_attribute.CU_DEVICE_ATTRIBUTE_MAX_BLOCK_DIM_X)

    @property
    def max_block_dim_y(self) -> int:
        """
        int: Maximum y-dimension of a block.
        """
        return self._get_cached_attribute(driver.CUdevice_attribute.CU_DEVICE_ATTRIBUTE_MAX_BLOCK_DIM_Y)

    @property
    def max_block_dim_z(self) -> int:
        """
        int: Maximum z-dimension of a block.
        """
        return self._get_cached_attribute(driver.CUdevice_attribute.CU_DEVICE_ATTRIBUTE_MAX_BLOCK_DIM_Z)

    @property
    def max_grid_dim_x(self) -> int:
        """
        int: Maximum x-dimension of a grid.
        """
        return self._get_cached_attribute(driver.CUdevice_attribute.CU_DEVICE_ATTRIBUTE_MAX_GRID_DIM_X)

    @property
    def max_grid_dim_y(self) -> int:
        """
        int: Maximum y-dimension of a grid.
        """
        return self._get_cached_attribute(driver.CUdevice_attribute.CU_DEVICE_ATTRIBUTE_MAX_GRID_DIM_Y)

    @property
    def max_grid_dim_z(self) -> int:
        """
        int: Maximum z-dimension of a grid.
        """
        return self._get_cached_attribute(driver.CUdevice_attribute.CU_DEVICE_ATTRIBUTE_MAX_GRID_DIM_Z)

    @property
    def max_shared_memory_per_block(self) -> int:
        """
        int: Maximum amount of shared memory available to a thread block in bytes.
        """
        return self._get_cached_attribute(driver.CUdevice_attribute.CU_DEVICE_ATTRIBUTE_MAX_SHARED_MEMORY_PER_BLOCK)

    @property
    def total_constant_memory(self) -> int:
        """
        int: Memory available on device for __constant__ variables in a CUDA C kernel in bytes.
        """
        return self._get_cached_attribute(driver.CUdevice_attribute.CU_DEVICE_ATTRIBUTE_TOTAL_CONSTANT_MEMORY)

    @property
    def warp_size(self) -> int:
        """
        int: Warp size in threads.
        """
        return self._get_cached_attribute(driver.CUdevice_attribute.CU_DEVICE_ATTRIBUTE_WARP_SIZE)

    @property
    def max_pitch(self) -> int:
        """
        int: Maximum pitch in bytes allowed by the memory copy functions that involve memory regions allocated
        through cuMemAllocPitch().
        """
        return self._get_cached_attribute(driver.CUdevice_attribute.CU_DEVICE_ATTRIBUTE_MAX_PITCH)

    @property
    def maximum_texture1d_width(self) -> int:
        """
        int: Maximum 1D texture width.
        """
        return self._get_cached_attribute(driver.CUdevice_attribute.CU_DEVICE_ATTRIBUTE_MAXIMUM_TEXTURE1D_WIDTH)

    @property
    def maximum_texture1d_linear_width(self) -> int:
        """
        int: Maximum width for a 1D texture bound to linear memory.
        """
        return self._get_cached_attribute(driver.CUdevice_attribute.CU_DEVICE_ATTRIBUTE_MAXIMUM_TEXTURE1D_LINEAR_WIDTH)

    @property
    def maximum_texture1d_mipmapped_width(self) -> int:
        """
        int: Maximum mipmapped 1D texture width.
        """
        return self._get_cached_attribute(
            driver.CUdevice_attribute.CU_DEVICE_ATTRIBUTE_MAXIMUM_TEXTURE1D_MIPMAPPED_WIDTH
        )

    @property
    def maximum_texture2d_width(self) -> int:
        """
        int: Maximum 2D texture width.
        """
        return self._get_cached_attribute(driver.CUdevice_attribute.CU_DEVICE_ATTRIBUTE_MAXIMUM_TEXTURE2D_WIDTH)

    @property
    def maximum_texture2d_height(self) -> int:
        """
        int: Maximum 2D texture height.
        """
        return self._get_cached_attribute(driver.CUdevice_attribute.CU_DEVICE_ATTRIBUTE_MAXIMUM_TEXTURE2D_HEIGHT)

    @property
    def maximum_texture2d_linear_width(self) -> int:
        """
        int: Maximum width for a 2D texture bound to linear memory.
        """
        return self._get_cached_attribute(driver.CUdevice_attribute.CU_DEVICE_ATTRIBUTE_MAXIMUM_TEXTURE2D_LINEAR_WIDTH)

    @property
    def maximum_texture2d_linear_height(self) -> int:
        """
        int: Maximum height for a 2D texture bound to linear memory.
        """
        return self._get_cached_attribute(driver.CUdevice_attribute.CU_DEVICE_ATTRIBUTE_MAXIMUM_TEXTURE2D_LINEAR_HEIGHT)

    @property
    def maximum_texture2d_linear_pitch(self) -> int:
        """
        int: Maximum pitch in bytes for a 2D texture bound to linear memory.
        """
        return self._get_cached_attribute(driver.CUdevice_attribute.CU_DEVICE_ATTRIBUTE_MAXIMUM_TEXTURE2D_LINEAR_PITCH)

    @property
    def maximum_texture2d_mipmapped_width(self) -> int:
        """
        int: Maximum mipmapped 2D texture width.
        """
        return self._get_cached_attribute(
            driver.CUdevice_attribute.CU_DEVICE_ATTRIBUTE_MAXIMUM_TEXTURE2D_MIPMAPPED_WIDTH
        )

    @property
    def maximum_texture2d_mipmapped_height(self) -> int:
        """
        int: Maximum mipmapped 2D texture height.
        """
        return self._get_cached_attribute(
            driver.CUdevice_attribute.CU_DEVICE_ATTRIBUTE_MAXIMUM_TEXTURE2D_MIPMAPPED_HEIGHT
        )

    @property
    def maximum_texture3d_width(self) -> int:
        """
        int: Maximum 3D texture width.
        """
        return self._get_cached_attribute(driver.CUdevice_attribute.CU_DEVICE_ATTRIBUTE_MAXIMUM_TEXTURE3D_WIDTH)

    @property
    def maximum_texture3d_height(self) -> int:
        """
        int: Maximum 3D texture height.
        """
        return self._get_cached_attribute(driver.CUdevice_attribute.CU_DEVICE_ATTRIBUTE_MAXIMUM_TEXTURE3D_HEIGHT)

    @property
    def maximum_texture3d_depth(self) -> int:
        """
        int: Maximum 3D texture depth.
        """
        return self._get_cached_attribute(driver.CUdevice_attribute.CU_DEVICE_ATTRIBUTE_MAXIMUM_TEXTURE3D_DEPTH)

    @property
    def maximum_texture3d_width_alternate(self) -> int:
        """
        int: Alternate maximum 3D texture width, 0 if no alternate maximum 3D texture size is supported.
        """
        return self._get_cached_attribute(
            driver.CUdevice_attribute.CU_DEVICE_ATTRIBUTE_MAXIMUM_TEXTURE3D_WIDTH_ALTERNATE
        )

    @property
    def maximum_texture3d_height_alternate(self) -> int:
        """
        int: Alternate maximum 3D texture height, 0 if no alternate maximum 3D texture size is supported.
        """
        return self._get_cached_attribute(
            driver.CUdevice_attribute.CU_DEVICE_ATTRIBUTE_MAXIMUM_TEXTURE3D_HEIGHT_ALTERNATE
        )

    @property
    def maximum_texture3d_depth_alternate(self) -> int:
        """
        int: Alternate maximum 3D texture depth, 0 if no alternate maximum 3D texture size is supported.
        """
        return self._get_cached_attribute(
            driver.CUdevice_attribute.CU_DEVICE_ATTRIBUTE_MAXIMUM_TEXTURE3D_DEPTH_ALTERNATE
        )

    @property
    def maximum_texturecubemap_width(self) -> int:
        """
        int: Maximum cubemap texture width or height.
        """
        return self._get_cached_attribute(driver.CUdevice_attribute.CU_DEVICE_ATTRIBUTE_MAXIMUM_TEXTURECUBEMAP_WIDTH)

    @property
    def maximum_texture1d_layered_width(self) -> int:
        """
        int: Maximum 1D layered texture width.
        """
        return self._get_cached_attribute(driver.CUdevice_attribute.CU_DEVICE_ATTRIBUTE_MAXIMUM_TEXTURE1D_LAYERED_WIDTH)

    @property
    def maximum_texture1d_layered_layers(self) -> int:
        """
        int: Maximum layers in a 1D layered texture.
        """
        return self._get_cached_attribute(
            driver.CUdevice_attribute.CU_DEVICE_ATTRIBUTE_MAXIMUM_TEXTURE1D_LAYERED_LAYERS
        )

    @property
    def maximum_texture2d_layered_width(self) -> int:
        """
        int: Maximum 2D layered texture width.
        """
        return self._get_cached_attribute(driver.CUdevice_attribute.CU_DEVICE_ATTRIBUTE_MAXIMUM_TEXTURE2D_LAYERED_WIDTH)

    @property
    def maximum_texture2d_layered_height(self) -> int:
        """
        int: Maximum 2D layered texture height.
        """
        return self._get_cached_attribute(
            driver.CUdevice_attribute.CU_DEVICE_ATTRIBUTE_MAXIMUM_TEXTURE2D_LAYERED_HEIGHT
        )

    @property
    def maximum_texture2d_layered_layers(self) -> int:
        """
        int: Maximum layers in a 2D layered texture.
        """
        return self._get_cached_attribute(
            driver.CUdevice_attribute.CU_DEVICE_ATTRIBUTE_MAXIMUM_TEXTURE2D_LAYERED_LAYERS
        )

    @property
    def maximum_texturecubemap_layered_width(self) -> int:
        """
        int: Maximum cubemap layered texture width or height.
        """
        return self._get_cached_attribute(
            driver.CUdevice_attribute.CU_DEVICE_ATTRIBUTE_MAXIMUM_TEXTURECUBEMAP_LAYERED_WIDTH
        )

    @property
    def maximum_texturecubemap_layered_layers(self) -> int:
        """
        int: Maximum layers in a cubemap layered texture.
        """
        return self._get_cached_attribute(
            driver.CUdevice_attribute.CU_DEVICE_ATTRIBUTE_MAXIMUM_TEXTURECUBEMAP_LAYERED_LAYERS
        )

    @property
    def maximum_surface1d_width(self) -> int:
        """
        int: Maximum 1D surface width.
        """
        return self._get_cached_attribute(driver.CUdevice_attribute.CU_DEVICE_ATTRIBUTE_MAXIMUM_SURFACE1D_WIDTH)

    @property
    def maximum_surface2d_width(self) -> int:
        """
        int: Maximum 2D surface width.
        """
        return self._get_cached_attribute(driver.CUdevice_attribute.CU_DEVICE_ATTRIBUTE_MAXIMUM_SURFACE2D_WIDTH)

    @property
    def maximum_surface2d_height(self) -> int:
        """
        int: Maximum 2D surface height.
        """
        return self._get_cached_attribute(driver.CUdevice_attribute.CU_DEVICE_ATTRIBUTE_MAXIMUM_SURFACE2D_HEIGHT)

    @property
    def maximum_surface3d_width(self) -> int:
        """
        int: Maximum 3D surface width.
        """
        return self._get_cached_attribute(driver.CUdevice_attribute.CU_DEVICE_ATTRIBUTE_MAXIMUM_SURFACE3D_WIDTH)

    @property
    def maximum_surface3d_height(self) -> int:
        """
        int: Maximum 3D surface height.
        """
        return self._get_cached_attribute(driver.CUdevice_attribute.CU_DEVICE_ATTRIBUTE_MAXIMUM_SURFACE3D_HEIGHT)

    @property
    def maximum_surface3d_depth(self) -> int:
        """
        int: Maximum 3D surface depth.
        """
        return self._get_cached_attribute(driver.CUdevice_attribute.CU_DEVICE_ATTRIBUTE_MAXIMUM_SURFACE3D_DEPTH)

    @property
    def maximum_surface1d_layered_width(self) -> int:
        """
        int: Maximum 1D layered surface width.
        """
        return self._get_cached_attribute(driver.CUdevice_attribute.CU_DEVICE_ATTRIBUTE_MAXIMUM_SURFACE1D_LAYERED_WIDTH)

    @property
    def maximum_surface1d_layered_layers(self) -> int:
        """
        int: Maximum layers in a 1D layered surface.
        """
        return self._get_cached_attribute(
            driver.CUdevice_attribute.CU_DEVICE_ATTRIBUTE_MAXIMUM_SURFACE1D_LAYERED_LAYERS
        )

    @property
    def maximum_surface2d_layered_width(self) -> int:
        """
        int: Maximum 2D layered surface width.
        """
        return self._get_cached_attribute(driver.CUdevice_attribute.CU_DEVICE_ATTRIBUTE_MAXIMUM_SURFACE2D_LAYERED_WIDTH)

    @property
    def maximum_surface2d_layered_height(self) -> int:
        """
        int: Maximum 2D layered surface height.
        """
        return self._get_cached_attribute(
            driver.CUdevice_attribute.CU_DEVICE_ATTRIBUTE_MAXIMUM_SURFACE2D_LAYERED_HEIGHT
        )

    @property
    def maximum_surface2d_layered_layers(self) -> int:
        """
        int: Maximum layers in a 2D layered surface.
        """
        return self._get_cached_attribute(
            driver.CUdevice_attribute.CU_DEVICE_ATTRIBUTE_MAXIMUM_SURFACE2D_LAYERED_LAYERS
        )

    @property
    def maximum_surfacecubemap_width(self) -> int:
        """
        int: Maximum cubemap surface width.
        """
        return self._get_cached_attribute(driver.CUdevice_attribute.CU_DEVICE_ATTRIBUTE_MAXIMUM_SURFACECUBEMAP_WIDTH)

    @property
    def maximum_surfacecubemap_layered_width(self) -> int:
        """
        int: Maximum cubemap layered surface width.
        """
        return self._get_cached_attribute(
            driver.CUdevice_attribute.CU_DEVICE_ATTRIBUTE_MAXIMUM_SURFACECUBEMAP_LAYERED_WIDTH
        )

    @property
    def maximum_surfacecubemap_layered_layers(self) -> int:
        """
        int: Maximum layers in a cubemap layered surface.
        """
        return self._get_cached_attribute(
            driver.CUdevice_attribute.CU_DEVICE_ATTRIBUTE_MAXIMUM_SURFACECUBEMAP_LAYERED_LAYERS
        )

    @property
    def max_registers_per_block(self) -> int:
        """
        int: Maximum number of 32-bit registers available to a thread block.
        """
        return self._get_cached_attribute(driver.CUdevice_attribute.CU_DEVICE_ATTRIBUTE_MAX_REGISTERS_PER_BLOCK)

    @property
    def clock_rate(self) -> int:
        """
        int: The typical clock frequency in kilohertz.
        """
        return self._get_attribute(driver.CUdevice_attribute.CU_DEVICE_ATTRIBUTE_CLOCK_RATE)

    @property
    def texture_alignment(self) -> int:
        """
        int: Alignment requirement; texture base addresses aligned to textureAlign bytes do not need an offset
        applied to texture fetches.
        """
        return self._get_cached_attribute(driver.CUdevice_attribute.CU_DEVICE_ATTRIBUTE_TEXTURE_ALIGNMENT)

    @property
    def texture_pitch_alignment(self) -> int:
        """
        int: Pitch alignment requirement for 2D texture references bound to pitched memory.
        """
        return self._get_cached_attribute(driver.CUdevice_attribute.CU_DEVICE_ATTRIBUTE_TEXTURE_PITCH_ALIGNMENT)

    @property
    def gpu_overlap(self) -> bool:
        """
        bool: True if the device can concurrently copy memory between host and device while executing a kernel,
        False if not.
        """
        return bool(self._get_cached_attribute(driver.CUdevice_attribute.CU_DEVICE_ATTRIBUTE_GPU_OVERLAP))

    @property
    def multiprocessor_count(self) -> int:
        """
        int: Number of multiprocessors on the device.
        """
        return self._get_cached_attribute(driver.CUdevice_attribute.CU_DEVICE_ATTRIBUTE_MULTIPROCESSOR_COUNT)

    @property
    def kernel_exec_timeout(self) -> bool:
        """
        bool: True if there is a run time limit for kernels executed on the device, False if not.
        """
        return bool(self._get_attribute(driver.CUdevice_attribute.CU_DEVICE_ATTRIBUTE_KERNEL_EXEC_TIMEOUT))

    @property
    def integrated(self) -> bool:
        """
        bool: True if the device is integrated with the memory subsystem, False if not.
        """
        return bool(self._get_cached_attribute(driver.CUdevice_attribute.CU_DEVICE_ATTRIBUTE_INTEGRATED))

    @property
    def can_map_host_memory(self) -> bool:
        """
        bool: True if the device can map host memory into the CUDA address space, False if not.
        """
        return bool(self._get_cached_attribute(driver.CUdevice_attribute.CU_DEVICE_ATTRIBUTE_CAN_MAP_HOST_MEMORY))

    @property
    def compute_mode(self) -> int:
        """
        int: Compute mode that device is currently in.
        """
        return self._get_attribute(driver.CUdevice_attribute.CU_DEVICE_ATTRIBUTE_COMPUTE_MODE)

    @property
    def concurrent_kernels(self) -> bool:
        """
        bool: True if the device supports executing multiple kernels within the same context simultaneously,
        False if not.
        """
        return bool(self._get_cached_attribute(driver.CUdevice_attribute.CU_DEVICE_ATTRIBUTE_CONCURRENT_KERNELS))

    @property
    def ecc_enabled(self) -> bool:
        """
        bool: True if error correction is enabled on the device, False if error correction is disabled or not
        supported by the device.
        """
        return bool(self._get_cached_attribute(driver.CUdevice_attribute.CU_DEVICE_ATTRIBUTE_ECC_ENABLED))

    @property
    def pci_bus_id(self) -> int:
        """
        int: PCI bus identifier of the device.
        """
        return self._get_cached_attribute(driver.CUdevice_attribute.CU_DEVICE_ATTRIBUTE_PCI_BUS_ID)

    @property
    def pci_device_id(self) -> int:
        """
        int: PCI device (also known as slot) identifier of the device.
        """
        return self._get_cached_attribute(driver.CUdevice_attribute.CU_DEVICE_ATTRIBUTE_PCI_DEVICE_ID)

    @property
    def pci_domain_id(self) -> int:
        """
        int: PCI domain identifier of the device.
        """
        return self._get_cached_attribute(driver.CUdevice_attribute.CU_DEVICE_ATTRIBUTE_PCI_DOMAIN_ID)

    @property
    def tcc_driver(self) -> bool:
        """
        bool: True if the device is using a TCC driver, False if not.
        """
        return bool(self._get_cached_attribute(driver.CUdevice_attribute.CU_DEVICE_ATTRIBUTE_TCC_DRIVER))

    @property
    def memory_clock_rate(self) -> int:
        """
        int: Peak memory clock frequency in kilohertz.
        """
        return self._get_attribute(driver.CUdevice_attribute.CU_DEVICE_ATTRIBUTE_MEMORY_CLOCK_RATE)

    @property
    def global_memory_bus_width(self) -> int:
        """
        int: Global memory bus width in bits.
        """
        return self._get_cached_attribute(driver.CUdevice_attribute.CU_DEVICE_ATTRIBUTE_GLOBAL_MEMORY_BUS_WIDTH)

    @property
    def l2_cache_size(self) -> int:
        """
        int: Size of L2 cache in bytes, 0 if the device doesn't have L2 cache.
        """
        return self._get_cached_attribute(driver.CUdevice_attribute.CU_DEVICE_ATTRIBUTE_L2_CACHE_SIZE)

    @property
    def max_threads_per_multiprocessor(self) -> int:
        """
        int: Maximum resident threads per multiprocessor.
        """
        return self._get_cached_attribute(driver.CUdevice_attribute.CU_DEVICE_ATTRIBUTE_MAX_THREADS_PER_MULTIPROCESSOR)

    @property
    def unified_addressing(self) -> bool:
        """
        bool: True if the device shares a unified address space with the host, False if not.
        """
        return bool(self._get_cached_attribute(driver.CUdevice_attribute.CU_DEVICE_ATTRIBUTE_UNIFIED_ADDRESSING))

    @property
    def compute_capability_major(self) -> int:
        """
        int: Major compute capability version number.
        """
        return self._get_cached_attribute(driver.CUdevice_attribute.CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MAJOR)

    @property
    def compute_capability_minor(self) -> int:
        """
        int: Minor compute capability version number.
        """
        return self._get_cached_attribute(driver.CUdevice_attribute.CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MINOR)

    @property
    def global_l1_cache_supported(self) -> bool:
        """
        True if device supports caching globals in L1 cache, False if caching globals in L1 cache is not supported
        by the device.
        """
        return bool(self._get_cached_attribute(driver.CUdevice_attribute.CU_DEVICE_ATTRIBUTE_GLOBAL_L1_CACHE_SUPPORTED))

    @property
    def local_l1_cache_supported(self) -> bool:
        """
        True if device supports caching locals in L1 cache, False if caching locals in L1 cache is not supported
        by the device.
        """
        return bool(self._get_cached_attribute(driver.CUdevice_attribute.CU_DEVICE_ATTRIBUTE_LOCAL_L1_CACHE_SUPPORTED))

    @property
    def max_shared_memory_per_multiprocessor(self) -> int:
        """
        Maximum amount of shared memory available to a multiprocessor in bytes.
        """
        return self._get_cached_attribute(
            driver.CUdevice_attribute.CU_DEVICE_ATTRIBUTE_MAX_SHARED_MEMORY_PER_MULTIPROCESSOR
        )

    @property
    def max_registers_per_multiprocessor(self) -> int:
        """
        Maximum number of 32-bit registers available to a multiprocessor.
        """
        return self._get_cached_attribute(
            driver.CUdevice_attribute.CU_DEVICE_ATTRIBUTE_MAX_REGISTERS_PER_MULTIPROCESSOR
        )

    @property
    def managed_memory(self) -> bool:
        """
        True if device supports allocating managed memory on this system, False if allocating managed memory is not
        supported by the device on this system.
        """
        return bool(self._get_cached_attribute(driver.CUdevice_attribute.CU_DEVICE_ATTRIBUTE_MANAGED_MEMORY))

    @property
    def multi_gpu_board(self) -> bool:
        """
        True if device is on a multi-GPU board, False if not.
        """
        return bool(self._get_cached_attribute(driver.CUdevice_attribute.CU_DEVICE_ATTRIBUTE_MULTI_GPU_BOARD))

    @property
    def multi_gpu_board_group_id(self) -> int:
        """
        Unique identifier for a group of devices associated with the same board.
        """
        return self._get_cached_attribute(driver.CUdevice_attribute.CU_DEVICE_ATTRIBUTE_MULTI_GPU_BOARD_GROUP_ID)

    @property
    def host_native_atomic_supported(self) -> bool:
        """
        True if Link between the device and the host supports native atomic operations, False if not.
        """
        return bool(
            self._get_cached_attribute(driver.CUdevice_attribute.CU_DEVICE_ATTRIBUTE_HOST_NATIVE_ATOMIC_SUPPORTED)
        )

    @property
    def single_to_double_precision_perf_ratio(self) -> int:
        """
        Ratio of single precision performance to double precision performance.
        """
        return self._get_attribute(driver.CUdevice_attribute.CU_DEVICE_ATTRIBUTE_SINGLE_TO_DOUBLE_PRECISION_PERF_RATIO)

    @property
    def pageable_memory_access(self) -> bool:
        """
        True if device supports coherently accessing pageable memory without calling cudaHostRegister on it,
        False if not.
        """
        return bool(self._get_cached_attribute(driver.CUdevice_attribute.CU_DEVICE_ATTRIBUTE_PAGEABLE_MEMORY_ACCESS))

    @property
    def concurrent_managed_access(self) -> bool:
        """
        True if device can coherently access managed memory concurrently with the CPU, False if not.
        """
        return bool(self._get_cached_attribute(driver.CUdevice_attribute.CU_DEVICE_ATTRIBUTE_CONCURRENT_MANAGED_ACCESS))

    @property
    def compute_preemption_supported(self) -> bool:
        """
        True if device supports Compute Preemption, False if not.
        """
        return bool(
            self._get_cached_attribute(driver.CUdevice_attribute.CU_DEVICE_ATTRIBUTE_COMPUTE_PREEMPTION_SUPPORTED)
        )

    @property
    def can_use_host_pointer_for_registered_mem(self) -> bool:
        """
        True if device can access host registered memory at the same virtual address as the CPU, False if not.
        """
        return bool(
            self._get_cached_attribute(
                driver.CUdevice_attribute.CU_DEVICE_ATTRIBUTE_CAN_USE_HOST_POINTER_FOR_REGISTERED_MEM
            )
        )

    # TODO: A few attrs are missing here (NVIDIA/cuda-python#675)

    @property
    def cooperative_launch(self) -> bool:
        """
        True if device supports launching cooperative kernels, False if not.
        """
        return bool(self._get_cached_attribute(driver.CUdevice_attribute.CU_DEVICE_ATTRIBUTE_COOPERATIVE_LAUNCH))

    # TODO: A few attrs are missing here (NVIDIA/cuda-python#675)

    @property
    def max_shared_memory_per_block_optin(self) -> int:
        """
        The maximum per block shared memory size supported on this device.
        """
        return self._get_cached_attribute(
            driver.CUdevice_attribute.CU_DEVICE_ATTRIBUTE_MAX_SHARED_MEMORY_PER_BLOCK_OPTIN
        )

    @property
    def pageable_memory_access_uses_host_page_tables(self) -> bool:
        """
        True if device accesses pageable memory via the host's page tables, False if not.
        """
        return bool(
            self._get_cached_attribute(
                driver.CUdevice_attribute.CU_DEVICE_ATTRIBUTE_PAGEABLE_MEMORY_ACCESS_USES_HOST_PAGE_TABLES
            )
        )

    @property
    def direct_managed_mem_access_from_host(self) -> bool:
        """
        True if the host can directly access managed memory on the device without migration, False if not.
        """
        return bool(
            self._get_cached_attribute(
                driver.CUdevice_attribute.CU_DEVICE_ATTRIBUTE_DIRECT_MANAGED_MEM_ACCESS_FROM_HOST
            )
        )

    @property
    def virtual_memory_management_supported(self) -> bool:
        """
        True if device supports virtual memory management APIs like cuMemAddressReserve, cuMemCreate, cuMemMap
        and related APIs, False if not.
        """
        return bool(
            self._get_cached_attribute(
                driver.CUdevice_attribute.CU_DEVICE_ATTRIBUTE_VIRTUAL_MEMORY_MANAGEMENT_SUPPORTED
            )
        )

    @property
    def handle_type_posix_file_descriptor_supported(self) -> bool:
        """
        True if device supports exporting memory to a posix file descriptor with cuMemExportToShareableHandle,
        False if not.
        """
        return bool(
            self._get_cached_attribute(
                driver.CUdevice_attribute.CU_DEVICE_ATTRIBUTE_HANDLE_TYPE_POSIX_FILE_DESCRIPTOR_SUPPORTED
            )
        )

    @property
    def handle_type_win32_handle_supported(self) -> bool:
        """
        True if device supports exporting memory to a Win32 NT handle with cuMemExportToShareableHandle,
        False if not.
        """
        return bool(
            self._get_cached_attribute(driver.CUdevice_attribute.CU_DEVICE_ATTRIBUTE_HANDLE_TYPE_WIN32_HANDLE_SUPPORTED)
        )

    @property
    def handle_type_win32_kmt_handle_supported(self) -> bool:
        """
        True if device supports exporting memory to a Win32 KMT handle with cuMemExportToShareableHandle,
        False if not.
        """
        return bool(
            self._get_cached_attribute(
                driver.CUdevice_attribute.CU_DEVICE_ATTRIBUTE_HANDLE_TYPE_WIN32_KMT_HANDLE_SUPPORTED
            )
        )

    @property
    def max_blocks_per_multiprocessor(self) -> int:
        """
        Maximum number of thread blocks that can reside on a multiprocessor.
        """
        return self._get_cached_attribute(driver.CUdevice_attribute.CU_DEVICE_ATTRIBUTE_MAX_BLOCKS_PER_MULTIPROCESSOR)

    @property
    def generic_compression_supported(self) -> bool:
        """
        True if device supports compressible memory allocation via cuMemCreate, False if not.
        """
        return bool(
            self._get_cached_attribute(driver.CUdevice_attribute.CU_DEVICE_ATTRIBUTE_GENERIC_COMPRESSION_SUPPORTED)
        )

    @property
    def max_persisting_l2_cache_size(self) -> int:
        """
        Maximum L2 persisting lines capacity setting in bytes.
        """
        return self._get_cached_attribute(driver.CUdevice_attribute.CU_DEVICE_ATTRIBUTE_MAX_PERSISTING_L2_CACHE_SIZE)

    @property
    def max_access_policy_window_size(self) -> int:
        """
        Maximum value of CUaccessPolicyWindow::num_bytes.
        """
        return self._get_cached_attribute(driver.CUdevice_attribute.CU_DEVICE_ATTRIBUTE_MAX_ACCESS_POLICY_WINDOW_SIZE)

    @property
    def gpu_direct_rdma_with_cuda_vmm_supported(self) -> bool:
        """
        True if device supports specifying the GPUDirect RDMA flag with cuMemCreate, False if not.
        """
        return bool(
            self._get_cached_attribute(
                driver.CUdevice_attribute.CU_DEVICE_ATTRIBUTE_GPU_DIRECT_RDMA_WITH_CUDA_VMM_SUPPORTED
            )
        )

    @property
    def reserved_shared_memory_per_block(self) -> int:
        """
        Amount of shared memory per block reserved by CUDA driver in bytes.
        """
        return self._get_cached_attribute(
            driver.CUdevice_attribute.CU_DEVICE_ATTRIBUTE_RESERVED_SHARED_MEMORY_PER_BLOCK
        )

    @property
    def sparse_cuda_array_supported(self) -> bool:
        """
        True if device supports sparse CUDA arrays and sparse CUDA mipmapped arrays, False if not.
        """
        return bool(
            self._get_cached_attribute(driver.CUdevice_attribute.CU_DEVICE_ATTRIBUTE_SPARSE_CUDA_ARRAY_SUPPORTED)
        )

    @property
    def read_only_host_register_supported(self) -> bool:
        """
        True if device supports using the cuMemHostRegister flag CU_MEMHOSTERGISTER_READ_ONLY to register
        memory that must be mapped as read-only to the GPU, False if not.
        """
        return bool(
            self._get_cached_attribute(driver.CUdevice_attribute.CU_DEVICE_ATTRIBUTE_READ_ONLY_HOST_REGISTER_SUPPORTED)
        )

    @property
    def memory_pools_supported(self) -> bool:
        """
        True if device supports using the cuMemAllocAsync and cuMemPool family of APIs, False if not.
        """
        return bool(self._get_cached_attribute(driver.CUdevice_attribute.CU_DEVICE_ATTRIBUTE_MEMORY_POOLS_SUPPORTED))

    @property
    def gpu_direct_rdma_supported(self) -> bool:
        """
        True if device supports GPUDirect RDMA APIs, False if not.
        """
        return bool(self._get_cached_attribute(driver.CUdevice_attribute.CU_DEVICE_ATTRIBUTE_GPU_DIRECT_RDMA_SUPPORTED))

    @property
    def gpu_direct_rdma_flush_writes_options(self) -> int:
        """
        The returned attribute shall be interpreted as a bitmask, where the individual bits are described by
        the CUflushGPUDirectRDMAWritesOptions enum.
        """
        return self._get_cached_attribute(
            driver.CUdevice_attribute.CU_DEVICE_ATTRIBUTE_GPU_DIRECT_RDMA_FLUSH_WRITES_OPTIONS
        )

    @property
    def gpu_direct_rdma_writes_ordering(self) -> int:
        """
        GPUDirect RDMA writes to the device do not need to be flushed for consumers within the scope indicated
        by the returned attribute.
        """
        return self._get_cached_attribute(driver.CUdevice_attribute.CU_DEVICE_ATTRIBUTE_GPU_DIRECT_RDMA_WRITES_ORDERING)

    @property
    def mempool_supported_handle_types(self) -> int:
        """
        Bitmask of handle types supported with mempool based IPC.
        """
        return self._get_cached_attribute(driver.CUdevice_attribute.CU_DEVICE_ATTRIBUTE_MEMPOOL_SUPPORTED_HANDLE_TYPES)

    @property
    def deferred_mapping_cuda_array_supported(self) -> bool:
        """
        True if device supports deferred mapping CUDA arrays and CUDA mipmapped arrays, False if not.
        """
        return bool(
            self._get_cached_attribute(
                driver.CUdevice_attribute.CU_DEVICE_ATTRIBUTE_DEFERRED_MAPPING_CUDA_ARRAY_SUPPORTED
            )
        )

    @property
    def numa_config(self) -> int:
        """
        NUMA configuration of a device.
        """
        return self._get_cached_attribute(driver.CUdevice_attribute.CU_DEVICE_ATTRIBUTE_NUMA_CONFIG)

    @property
    def numa_id(self) -> int:
        """
        NUMA node ID of the GPU memory.
        """
        return self._get_cached_attribute(driver.CUdevice_attribute.CU_DEVICE_ATTRIBUTE_NUMA_ID)

    @property
    def multicast_supported(self) -> bool:
        """
        True if device supports switch multicast and reduction operations, False if not.
        """
        return bool(self._get_cached_attribute(driver.CUdevice_attribute.CU_DEVICE_ATTRIBUTE_MULTICAST_SUPPORTED))


_SUCCESS = driver.CUresult.CUDA_SUCCESS
_INVALID_CTX = driver.CUresult.CUDA_ERROR_INVALID_CONTEXT


class Device:
    """Represent a GPU and act as an entry point for cuda.core features.

    This is a singleton object that helps ensure interoperability
    across multiple libraries imported in the process to both see
    and use the same GPU device.

    While acting as the entry point, many other CUDA resources can be
    allocated such as streams and buffers. Any :obj:`~_context.Context` dependent
    resource created through this device, will continue to refer to
    this device's context.

    Newly returned :obj:`~_device.Device` objects are thread-local singletons
    for a specified device.

    Note
    ----
    Will not initialize the GPU.

    Parameters
    ----------
    device_id : int, optional
        Device ordinal to return a :obj:`~_device.Device` object for.
        Default value of `None` return the currently used device.

    """

    __slots__ = ("_id", "_mr", "_has_inited", "_properties")

    def __new__(cls, device_id: Optional[int] = None):
        global _is_cuInit
        if _is_cuInit is False:
            with _lock:
                handle_return(driver.cuInit(0))
                _is_cuInit = True

        # important: creating a Device instance does not initialize the GPU!
        if device_id is None:
            err, dev = driver.cuCtxGetDevice()
            if err == _SUCCESS:
                device_id = int(dev)
            elif err == _INVALID_CTX:
                ctx = handle_return(driver.cuCtxGetCurrent())
                assert int(ctx) == 0
                device_id = 0  # cudart behavior
            else:
                _check_driver_error(err)
        elif device_id < 0:
            raise ValueError(f"device_id must be >= 0, got {device_id}")

        # ensure Device is singleton
        try:
            devices = _tls.devices
        except AttributeError:
            total = handle_return(driver.cuDeviceGetCount())
            devices = _tls.devices = []
            for dev_id in range(total):
                dev = super().__new__(cls)
                dev._id = dev_id
                # If the device is in TCC mode, or does not support memory pools for some other reason,
                # use the SynchronousMemoryResource which does not use memory pools.
                if (
                    handle_return(
                        driver.cuDeviceGetAttribute(
                            driver.CUdevice_attribute.CU_DEVICE_ATTRIBUTE_MEMORY_POOLS_SUPPORTED, dev_id
                        )
                    )
                ) == 1:
                    dev._mr = DeviceMemoryResource(dev_id)
                else:
                    dev._mr = _SynchronousMemoryResource(dev_id)

                dev._has_inited = False
                dev._properties = None
                devices.append(dev)

        try:
            return devices[device_id]
        except IndexError:
            raise ValueError(f"device_id must be within [0, {len(devices)}), got {device_id}") from None

    def _check_context_initialized(self, *args, **kwargs):
        if not self._has_inited:
            raise CUDAError(
                f"Device {self._id} is not yet initialized, perhaps you forgot to call .set_current() first?"
            )

    @property
    def device_id(self) -> int:
        """Return device ordinal."""
        return self._id

    @property
    def pci_bus_id(self) -> str:
        """Return a PCI Bus Id string for this device."""
        bus_id = handle_return(runtime.cudaDeviceGetPCIBusId(13, self._id))
        return bus_id[:12].decode()

    @property
    def uuid(self) -> str:
        """Return a UUID for the device.

        Returns 16-octets identifying the device. If the device is in
        MIG mode, returns its MIG UUID which uniquely identifies the
        subscribed MIG compute instance.

        Note
        ----
        MIG UUID is only returned when device is in MIG mode and the
        driver is older than CUDA 11.4.

        """
        driver_ver = handle_return(driver.cuDriverGetVersion())
        if driver_ver >= 11040:
            uuid = handle_return(driver.cuDeviceGetUuid_v2(self._id))
        else:
            uuid = handle_return(driver.cuDeviceGetUuid(self._id))
        uuid = uuid.bytes.hex()
        # 8-4-4-4-12
        return f"{uuid[:8]}-{uuid[8:12]}-{uuid[12:16]}-{uuid[16:20]}-{uuid[20:]}"

    @property
    def name(self) -> str:
        """Return the device name."""
        # Use 256 characters to be consistent with CUDA Runtime
        name = handle_return(driver.cuDeviceGetName(256, self._id))
        name = name.split(b"\0")[0]
        return name.decode()

    @property
    def properties(self) -> DeviceProperties:
        """Return a :obj:`~_device.DeviceProperties` class with information about the device."""
        if self._properties is None:
            self._properties = DeviceProperties._init(self._id)

        return self._properties

    @property
    def compute_capability(self) -> ComputeCapability:
        """Return a named tuple with 2 fields: major and minor."""
        if "compute_capability" in self.properties._cache:
            return self.properties._cache["compute_capability"]
        cc = ComputeCapability(self.properties.compute_capability_major, self.properties.compute_capability_minor)
        self.properties._cache["compute_capability"] = cc
        return cc

    @property
    @precondition(_check_context_initialized)
    def context(self) -> Context:
        """Return the current :obj:`~_context.Context` associated with this device.

        Note
        ----
        Device must be initialized.

        """
        ctx = handle_return(driver.cuCtxGetCurrent())
        if int(ctx) == 0:
            raise CUDAError("No context is bound to the calling CPU thread.")
        return Context._from_ctx(ctx, self._id)

    @property
    def memory_resource(self) -> MemoryResource:
        """Return :obj:`~_memory.MemoryResource` associated with this device."""
        return self._mr

    @memory_resource.setter
    def memory_resource(self, mr):
        assert_type(mr, MemoryResource)
        self._mr = mr

    @property
    def default_stream(self) -> Stream:
        """Return default CUDA :obj:`~_stream.Stream` associated with this device.

        The type of default stream returned depends on if the environment
        variable CUDA_PYTHON_CUDA_PER_THREAD_DEFAULT_STREAM is set.

        If set, returns a per-thread default stream. Otherwise returns
        the legacy stream.

        """
        return default_stream()

    def __int__(self):
        """Return device_id."""
        return self._id

    def __repr__(self):
        return f"<Device {self._id} ({self.name})>"

    def set_current(self, ctx: Context = None) -> Union[Context, None]:
        """Set device to be used for GPU executions.

        Initializes CUDA and sets the calling thread to a valid CUDA
        context. By default the primary context is used, but optional `ctx`
        parameter can be used to explicitly supply a :obj:`~_context.Context` object.

        Providing a `ctx` causes the previous set context to be popped and returned.

        Parameters
        ----------
        ctx : :obj:`~_context.Context`, optional
            Optional context to push onto this device's current thread stack.

        Returns
        -------
        Union[:obj:`~_context.Context`, None], optional
            Popped context.

        Examples
        --------
        Acts as an entry point of this object. Users always start a code by
        calling this method, e.g.

        >>> from cuda.core.experimental import Device
        >>> dev0 = Device(0)
        >>> dev0.set_current()
        >>> # ... do work on device 0 ...

        """
        if ctx is not None:
            assert_type(ctx, Context)
            if ctx._id != self._id:
                raise RuntimeError(
                    "the provided context was created on the device with"
                    f" id={ctx._id}, which is different from the target id={self._id}"
                )
            prev_ctx = handle_return(driver.cuCtxPopCurrent())
            handle_return(driver.cuCtxPushCurrent(ctx._handle))
            self._has_inited = True
            if int(prev_ctx) != 0:
                return Context._from_ctx(prev_ctx, self._id)
        else:
            ctx = handle_return(driver.cuCtxGetCurrent())
            if int(ctx) == 0:
                # use primary ctx
                ctx = handle_return(driver.cuDevicePrimaryCtxRetain(self._id))
                handle_return(driver.cuCtxPushCurrent(ctx))
            else:
                ctx_id = handle_return(driver.cuCtxGetDevice())
                if ctx_id != self._id:
                    # use primary ctx
                    ctx = handle_return(driver.cuDevicePrimaryCtxRetain(self._id))
                    handle_return(driver.cuCtxPushCurrent(ctx))
                else:
                    # no-op, a valid context already exists and is set current
                    pass
            self._has_inited = True

    def create_context(self, options: ContextOptions = None) -> Context:
        """Create a new :obj:`~_context.Context` object.

        Note
        ----
        The newly context will not be set as current.

        Parameters
        ----------
        options : :obj:`~_context.ContextOptions`, optional
            Customizable dataclass for context creation options.

        Returns
        -------
        :obj:`~_context.Context`
            Newly created context object.

        """
        raise NotImplementedError("WIP: https://github.com/NVIDIA/cuda-python/issues/189")

    @precondition(_check_context_initialized)
    def create_stream(self, obj: Optional[IsStreamT] = None, options: StreamOptions = None) -> Stream:
        """Create a Stream object.

        New stream objects can be created in two different ways:

        1) Create a new CUDA stream with customizable ``options``.
        2) Wrap an existing foreign `obj` supporting the ``__cuda_stream__`` protocol.

        Option (2) internally holds a reference to the foreign object
        such that the lifetime is managed.

        Note
        ----
        Device must be initialized.

        Parameters
        ----------
        obj : :obj:`~_stream.IsStreamT`, optional
            Any object supporting the ``__cuda_stream__`` protocol.
        options : :obj:`~_stream.StreamOptions`, optional
            Customizable dataclass for stream creation options.

        Returns
        -------
        :obj:`~_stream.Stream`
            Newly created stream object.

        """
        return Stream._init(obj=obj, options=options)

    @precondition(_check_context_initialized)
    def create_event(self, options: Optional[EventOptions] = None) -> Event:
        """Create an Event object without recording it to a Stream.

        Note
        ----
        Device must be initialized.

        Parameters
        ----------
        options : :obj:`EventOptions`, optional
            Customizable dataclass for event creation options.

        Returns
        -------
        :obj:`~_event.Event`
            Newly created event object.

        """
        return Event._init(self._id, self.context._handle, options)

    @precondition(_check_context_initialized)
    def allocate(self, size, stream: Optional[Stream] = None) -> Buffer:
        """Allocate device memory from a specified stream.

        Allocates device memory of `size` bytes on the specified `stream`
        using the memory resource currently associated with this Device.

        Parameter `stream` is optional, using a default stream by default.

        Note
        ----
        Device must be initialized.

        Parameters
        ----------
        size : int
            Number of bytes to allocate.
        stream : :obj:`~_stream.Stream`, optional
            The stream establishing the stream ordering semantic.
            Default value of `None` uses default stream.

        Returns
        -------
        :obj:`~_memory.Buffer`
            Newly created buffer object.

        """
        if stream is None:
            stream = default_stream()
        return self._mr.allocate(size, stream)

    @precondition(_check_context_initialized)
    def sync(self):
        """Synchronize the device.

        Note
        ----
        Device must be initialized.

        """
        handle_return(runtime.cudaDeviceSynchronize())

    @precondition(_check_context_initialized)
    def create_graph_builder(self) -> GraphBuilder:
        """Create a new :obj:`~_graph.GraphBuilder` object.

        Returns
        -------
        :obj:`~_graph.GraphBuilder`
            Newly created graph builder object.

        """
        return GraphBuilder._init(stream=self.create_stream(), is_stream_owner=True)
