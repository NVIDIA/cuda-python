# SPDX-FileCopyrightText: Copyright (c) 2024-2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

cimport cpython
from libc.stdint cimport uintptr_t

from cuda.bindings cimport cydriver
from cuda.core._utils.cuda_utils cimport HANDLE_RETURN

import threading

from cuda.core._context cimport Context
from cuda.core._context import ContextOptions
from cuda.core._event cimport Event as cyEvent
from cuda.core._event import Event, EventOptions
from cuda.core._resource_handles cimport (
    ContextHandle,
    create_context_handle_ref,
    get_primary_context,
    as_cu,
)

from cuda.core._graph import GraphBuilder
from cuda.core._stream import IsStreamT, Stream, StreamOptions
from cuda.core._utils.clear_error_support import assert_type
from cuda.core._utils.cuda_utils import (
    ComputeCapability,
    CUDAError,
    driver,
    handle_return,
    runtime,
)
from cuda.core._stream cimport default_stream

# TODO: I prefer to type these as "cdef object" and avoid accessing them from within Python,
# but it seems it is very convenient to expose them for testing purposes...
_tls = threading.local()
_lock = threading.Lock()
cdef bint _is_cuInit = False


cdef class DeviceProperties:
    """
    A class to query various attributes of a CUDA device.

    Attributes are read-only and provide information about the device.
    """
    cdef:
        int _handle
        dict _cache

    def __init__(self, *args, **kwargs):
        raise RuntimeError("DeviceProperties cannot be instantiated directly. Please use Device APIs.")

    @classmethod
    def _init(cls, handle):
        cdef DeviceProperties self = DeviceProperties.__new__(cls)
        self._handle = handle
        self._cache = {}
        return self

    cdef inline int _get_attribute(self, cydriver.CUdevice_attribute attr, default=0) except? -2:
        """Retrieve the attribute value directly from the driver."""
        cdef int val
        cdef cydriver.CUresult err
        with nogil:
            err = cydriver.cuDeviceGetAttribute(&val, attr, self._handle)
        if err == cydriver.CUresult.CUDA_ERROR_INVALID_VALUE and default is not None:
            return <int>default
        HANDLE_RETURN(err)
        return val

    cdef inline int _get_cached_attribute(self, attr, default=0) except? -2:
        """Retrieve the attribute value, using cache if applicable."""
        if attr not in self._cache:
            self._cache[attr] = self._get_attribute(attr, default)
        return self._cache[attr]

    @property
    def max_threads_per_block(self) -> int:
        """int: Maximum number of threads per block."""
        return self._get_cached_attribute(driver.CUdevice_attribute.CU_DEVICE_ATTRIBUTE_MAX_THREADS_PER_BLOCK)

    @property
    def max_block_dim_x(self) -> int:
        """int: Maximum block dimension X."""
        return self._get_cached_attribute(driver.CUdevice_attribute.CU_DEVICE_ATTRIBUTE_MAX_BLOCK_DIM_X)

    @property
    def max_block_dim_y(self) -> int:
        """int: Maximum block dimension Y."""
        return self._get_cached_attribute(driver.CUdevice_attribute.CU_DEVICE_ATTRIBUTE_MAX_BLOCK_DIM_Y)

    @property
    def max_block_dim_z(self) -> int:
        """int: Maximum block dimension Z."""
        return self._get_cached_attribute(driver.CUdevice_attribute.CU_DEVICE_ATTRIBUTE_MAX_BLOCK_DIM_Z)

    @property
    def max_grid_dim_x(self) -> int:
        """int: Maximum grid dimension X."""
        return self._get_cached_attribute(driver.CUdevice_attribute.CU_DEVICE_ATTRIBUTE_MAX_GRID_DIM_X)

    @property
    def max_grid_dim_y(self) -> int:
        """int: Maximum grid dimension Y."""
        return self._get_cached_attribute(driver.CUdevice_attribute.CU_DEVICE_ATTRIBUTE_MAX_GRID_DIM_Y)

    @property
    def max_grid_dim_z(self) -> int:
        """int: Maximum grid dimension Z."""
        return self._get_cached_attribute(driver.CUdevice_attribute.CU_DEVICE_ATTRIBUTE_MAX_GRID_DIM_Z)

    @property
    def max_shared_memory_per_block(self) -> int:
        """int: Maximum shared memory available per block in bytes."""
        return self._get_cached_attribute(driver.CUdevice_attribute.CU_DEVICE_ATTRIBUTE_MAX_SHARED_MEMORY_PER_BLOCK)

    @property
    def total_constant_memory(self) -> int:
        """int: Memory available on device for constant variables in a CUDA C kernel in bytes."""
        return self._get_cached_attribute(driver.CUdevice_attribute.CU_DEVICE_ATTRIBUTE_TOTAL_CONSTANT_MEMORY)

    @property
    def warp_size(self) -> int:
        """int: Warp size in threads."""
        return self._get_cached_attribute(driver.CUdevice_attribute.CU_DEVICE_ATTRIBUTE_WARP_SIZE)

    @property
    def max_pitch(self) -> int:
        """int: Maximum pitch in bytes allowed by memory copies."""
        return self._get_cached_attribute(driver.CUdevice_attribute.CU_DEVICE_ATTRIBUTE_MAX_PITCH)

    @property
    def maximum_texture1d_width(self) -> int:
        """int: Maximum 1D texture width."""
        return self._get_cached_attribute(driver.CUdevice_attribute.CU_DEVICE_ATTRIBUTE_MAXIMUM_TEXTURE1D_WIDTH)

    @property
    def maximum_texture1d_linear_width(self) -> int:
        """int: Maximum width for a 1D texture bound to linear memory."""
        return self._get_cached_attribute(driver.CUdevice_attribute.CU_DEVICE_ATTRIBUTE_MAXIMUM_TEXTURE1D_LINEAR_WIDTH)

    @property
    def maximum_texture1d_mipmapped_width(self) -> int:
        """int: Maximum mipmapped 1D texture width."""
        return self._get_cached_attribute(
            driver.CUdevice_attribute.CU_DEVICE_ATTRIBUTE_MAXIMUM_TEXTURE1D_MIPMAPPED_WIDTH
        )

    @property
    def maximum_texture2d_width(self) -> int:
        """int: Maximum 2D texture width."""
        return self._get_cached_attribute(driver.CUdevice_attribute.CU_DEVICE_ATTRIBUTE_MAXIMUM_TEXTURE2D_WIDTH)

    @property
    def maximum_texture2d_height(self) -> int:
        """int: Maximum 2D texture height."""
        return self._get_cached_attribute(driver.CUdevice_attribute.CU_DEVICE_ATTRIBUTE_MAXIMUM_TEXTURE2D_HEIGHT)

    @property
    def maximum_texture2d_linear_width(self) -> int:
        """int: Maximum width for a 2D texture bound to linear memory."""
        return self._get_cached_attribute(driver.CUdevice_attribute.CU_DEVICE_ATTRIBUTE_MAXIMUM_TEXTURE2D_LINEAR_WIDTH)

    @property
    def maximum_texture2d_linear_height(self) -> int:
        """int: Maximum height for a 2D texture bound to linear memory."""
        return self._get_cached_attribute(driver.CUdevice_attribute.CU_DEVICE_ATTRIBUTE_MAXIMUM_TEXTURE2D_LINEAR_HEIGHT)

    @property
    def maximum_texture2d_linear_pitch(self) -> int:
        """int: Maximum pitch in bytes for a 2D texture bound to linear memory."""
        return self._get_cached_attribute(driver.CUdevice_attribute.CU_DEVICE_ATTRIBUTE_MAXIMUM_TEXTURE2D_LINEAR_PITCH)

    @property
    def maximum_texture2d_mipmapped_width(self) -> int:
        """int: Maximum mipmapped 2D texture width."""
        return self._get_cached_attribute(
            driver.CUdevice_attribute.CU_DEVICE_ATTRIBUTE_MAXIMUM_TEXTURE2D_MIPMAPPED_WIDTH
        )

    @property
    def maximum_texture2d_mipmapped_height(self) -> int:
        """int: Maximum mipmapped 2D texture height."""
        return self._get_cached_attribute(
            driver.CUdevice_attribute.CU_DEVICE_ATTRIBUTE_MAXIMUM_TEXTURE2D_MIPMAPPED_HEIGHT
        )

    @property
    def maximum_texture3d_width(self) -> int:
        """int: Maximum 3D texture width."""
        return self._get_cached_attribute(driver.CUdevice_attribute.CU_DEVICE_ATTRIBUTE_MAXIMUM_TEXTURE3D_WIDTH)

    @property
    def maximum_texture3d_height(self) -> int:
        """int: Maximum 3D texture height."""
        return self._get_cached_attribute(driver.CUdevice_attribute.CU_DEVICE_ATTRIBUTE_MAXIMUM_TEXTURE3D_HEIGHT)

    @property
    def maximum_texture3d_depth(self) -> int:
        """int: Maximum 3D texture depth."""
        return self._get_cached_attribute(driver.CUdevice_attribute.CU_DEVICE_ATTRIBUTE_MAXIMUM_TEXTURE3D_DEPTH)

    @property
    def maximum_texture3d_width_alternate(self) -> int:
        """int: Alternate maximum 3D texture width, 0 if no alternate maximum 3D texture size is supported."""
        return self._get_cached_attribute(
            driver.CUdevice_attribute.CU_DEVICE_ATTRIBUTE_MAXIMUM_TEXTURE3D_WIDTH_ALTERNATE
        )

    @property
    def maximum_texture3d_height_alternate(self) -> int:
        """int: Alternate maximum 3D texture height, 0 if no alternate maximum 3D texture size is supported."""
        return self._get_cached_attribute(
            driver.CUdevice_attribute.CU_DEVICE_ATTRIBUTE_MAXIMUM_TEXTURE3D_HEIGHT_ALTERNATE
        )

    @property
    def maximum_texture3d_depth_alternate(self) -> int:
        """int: Alternate maximum 3D texture depth, 0 if no alternate maximum 3D texture size is supported."""
        return self._get_cached_attribute(
            driver.CUdevice_attribute.CU_DEVICE_ATTRIBUTE_MAXIMUM_TEXTURE3D_DEPTH_ALTERNATE
        )

    @property
    def maximum_texturecubemap_width(self) -> int:
        """int: Maximum cubemap texture width or height."""
        return self._get_cached_attribute(driver.CUdevice_attribute.CU_DEVICE_ATTRIBUTE_MAXIMUM_TEXTURECUBEMAP_WIDTH)

    @property
    def maximum_texture1d_layered_width(self) -> int:
        """int: Maximum 1D layered texture width."""
        return self._get_cached_attribute(driver.CUdevice_attribute.CU_DEVICE_ATTRIBUTE_MAXIMUM_TEXTURE1D_LAYERED_WIDTH)

    @property
    def maximum_texture1d_layered_layers(self) -> int:
        """int: Maximum layers in a 1D layered texture."""
        return self._get_cached_attribute(
            driver.CUdevice_attribute.CU_DEVICE_ATTRIBUTE_MAXIMUM_TEXTURE1D_LAYERED_LAYERS
        )

    @property
    def maximum_texture2d_layered_width(self) -> int:
        """int: Maximum 2D layered texture width."""
        return self._get_cached_attribute(driver.CUdevice_attribute.CU_DEVICE_ATTRIBUTE_MAXIMUM_TEXTURE2D_LAYERED_WIDTH)

    @property
    def maximum_texture2d_layered_height(self) -> int:
        """int: Maximum 2D layered texture height."""
        return self._get_cached_attribute(
            driver.CUdevice_attribute.CU_DEVICE_ATTRIBUTE_MAXIMUM_TEXTURE2D_LAYERED_HEIGHT
        )

    @property
    def maximum_texture2d_layered_layers(self) -> int:
        """int: Maximum layers in a 2D layered texture."""
        return self._get_cached_attribute(
            driver.CUdevice_attribute.CU_DEVICE_ATTRIBUTE_MAXIMUM_TEXTURE2D_LAYERED_LAYERS
        )

    @property
    def maximum_texturecubemap_layered_width(self) -> int:
        """int: Maximum cubemap layered texture width or height."""
        return self._get_cached_attribute(
            driver.CUdevice_attribute.CU_DEVICE_ATTRIBUTE_MAXIMUM_TEXTURECUBEMAP_LAYERED_WIDTH
        )

    @property
    def maximum_texturecubemap_layered_layers(self) -> int:
        """int: Maximum layers in a cubemap layered texture."""
        return self._get_cached_attribute(
            driver.CUdevice_attribute.CU_DEVICE_ATTRIBUTE_MAXIMUM_TEXTURECUBEMAP_LAYERED_LAYERS
        )

    @property
    def maximum_surface1d_width(self) -> int:
        """int: Maximum 1D surface width."""
        return self._get_cached_attribute(driver.CUdevice_attribute.CU_DEVICE_ATTRIBUTE_MAXIMUM_SURFACE1D_WIDTH)

    @property
    def maximum_surface2d_width(self) -> int:
        """int: Maximum 2D surface width."""
        return self._get_cached_attribute(driver.CUdevice_attribute.CU_DEVICE_ATTRIBUTE_MAXIMUM_SURFACE2D_WIDTH)

    @property
    def maximum_surface2d_height(self) -> int:
        """int: Maximum 2D surface height."""
        return self._get_cached_attribute(driver.CUdevice_attribute.CU_DEVICE_ATTRIBUTE_MAXIMUM_SURFACE2D_HEIGHT)

    @property
    def maximum_surface3d_width(self) -> int:
        """int: Maximum 3D surface width."""
        return self._get_cached_attribute(driver.CUdevice_attribute.CU_DEVICE_ATTRIBUTE_MAXIMUM_SURFACE3D_WIDTH)

    @property
    def maximum_surface3d_height(self) -> int:
        """int: Maximum 3D surface height."""
        return self._get_cached_attribute(driver.CUdevice_attribute.CU_DEVICE_ATTRIBUTE_MAXIMUM_SURFACE3D_HEIGHT)

    @property
    def maximum_surface3d_depth(self) -> int:
        """int: Maximum 3D surface depth."""
        return self._get_cached_attribute(driver.CUdevice_attribute.CU_DEVICE_ATTRIBUTE_MAXIMUM_SURFACE3D_DEPTH)

    @property
    def maximum_surface1d_layered_width(self) -> int:
        """int: Maximum 1D layered surface width."""
        return self._get_cached_attribute(driver.CUdevice_attribute.CU_DEVICE_ATTRIBUTE_MAXIMUM_SURFACE1D_LAYERED_WIDTH)

    @property
    def maximum_surface1d_layered_layers(self) -> int:
        """int: Maximum layers in a 1D layered surface."""
        return self._get_cached_attribute(
            driver.CUdevice_attribute.CU_DEVICE_ATTRIBUTE_MAXIMUM_SURFACE1D_LAYERED_LAYERS
        )

    @property
    def maximum_surface2d_layered_width(self) -> int:
        """int: Maximum 2D layered surface width."""
        return self._get_cached_attribute(driver.CUdevice_attribute.CU_DEVICE_ATTRIBUTE_MAXIMUM_SURFACE2D_LAYERED_WIDTH)

    @property
    def maximum_surface2d_layered_height(self) -> int:
        """int: Maximum 2D layered surface height."""
        return self._get_cached_attribute(
            driver.CUdevice_attribute.CU_DEVICE_ATTRIBUTE_MAXIMUM_SURFACE2D_LAYERED_HEIGHT
        )

    @property
    def maximum_surface2d_layered_layers(self) -> int:
        """int: Maximum layers in a 2D layered surface."""
        return self._get_cached_attribute(
            driver.CUdevice_attribute.CU_DEVICE_ATTRIBUTE_MAXIMUM_SURFACE2D_LAYERED_LAYERS
        )

    @property
    def maximum_surfacecubemap_width(self) -> int:
        """int: Maximum cubemap surface width."""
        return self._get_cached_attribute(driver.CUdevice_attribute.CU_DEVICE_ATTRIBUTE_MAXIMUM_SURFACECUBEMAP_WIDTH)

    @property
    def maximum_surfacecubemap_layered_width(self) -> int:
        """int: Maximum cubemap layered surface width."""
        return self._get_cached_attribute(
            driver.CUdevice_attribute.CU_DEVICE_ATTRIBUTE_MAXIMUM_SURFACECUBEMAP_LAYERED_WIDTH
        )

    @property
    def maximum_surfacecubemap_layered_layers(self) -> int:
        """int: Maximum layers in a cubemap layered surface."""
        return self._get_cached_attribute(
            driver.CUdevice_attribute.CU_DEVICE_ATTRIBUTE_MAXIMUM_SURFACECUBEMAP_LAYERED_LAYERS
        )

    @property
    def max_registers_per_block(self) -> int:
        """int: Maximum number of 32-bit registers available to a thread block."""
        return self._get_cached_attribute(driver.CUdevice_attribute.CU_DEVICE_ATTRIBUTE_MAX_REGISTERS_PER_BLOCK)

    @property
    def clock_rate(self) -> int:
        """int: Typical clock frequency in kilohertz."""
        return self._get_attribute(driver.CUdevice_attribute.CU_DEVICE_ATTRIBUTE_CLOCK_RATE)

    @property
    def texture_alignment(self) -> int:
        """int: Alignment requirement for textures."""
        return self._get_cached_attribute(driver.CUdevice_attribute.CU_DEVICE_ATTRIBUTE_TEXTURE_ALIGNMENT)

    @property
    def texture_pitch_alignment(self) -> int:
        """int: Pitch alignment requirement for textures."""
        return self._get_cached_attribute(driver.CUdevice_attribute.CU_DEVICE_ATTRIBUTE_TEXTURE_PITCH_ALIGNMENT)

    @property
    def gpu_overlap(self) -> bool:
        """bool: Device can possibly copy memory and execute a kernel concurrently. Deprecated. Use instead async_engine_count."""
        return bool(self._get_cached_attribute(driver.CUdevice_attribute.CU_DEVICE_ATTRIBUTE_GPU_OVERLAP))

    @property
    def multiprocessor_count(self) -> int:
        """int: Number of multiprocessors on device."""
        return self._get_cached_attribute(driver.CUdevice_attribute.CU_DEVICE_ATTRIBUTE_MULTIPROCESSOR_COUNT)

    @property
    def kernel_exec_timeout(self) -> bool:
        """bool: Specifies whether there is a run time limit on kernels."""
        return bool(self._get_attribute(driver.CUdevice_attribute.CU_DEVICE_ATTRIBUTE_KERNEL_EXEC_TIMEOUT))

    @property
    def integrated(self) -> bool:
        """bool: Device is integrated with host memory."""
        return bool(self._get_cached_attribute(driver.CUdevice_attribute.CU_DEVICE_ATTRIBUTE_INTEGRATED))

    @property
    def can_map_host_memory(self) -> bool:
        """bool: Device can map host memory into CUDA address space."""
        return bool(self._get_cached_attribute(driver.CUdevice_attribute.CU_DEVICE_ATTRIBUTE_CAN_MAP_HOST_MEMORY))

    @property
    def compute_mode(self) -> int:
        """int: Compute mode (See CUcomputemode for details)."""
        return self._get_attribute(driver.CUdevice_attribute.CU_DEVICE_ATTRIBUTE_COMPUTE_MODE)

    @property
    def concurrent_kernels(self) -> bool:
        """bool: Device can possibly execute multiple kernels concurrently."""
        return bool(self._get_cached_attribute(driver.CUdevice_attribute.CU_DEVICE_ATTRIBUTE_CONCURRENT_KERNELS))

    @property
    def ecc_enabled(self) -> bool:
        """bool: Device has ECC support enabled."""
        return bool(self._get_cached_attribute(driver.CUdevice_attribute.CU_DEVICE_ATTRIBUTE_ECC_ENABLED))

    @property
    def pci_bus_id(self) -> int:
        """int: PCI bus ID of the device."""
        return self._get_cached_attribute(driver.CUdevice_attribute.CU_DEVICE_ATTRIBUTE_PCI_BUS_ID)

    @property
    def pci_device_id(self) -> int:
        """int: PCI device ID of the device."""
        return self._get_cached_attribute(driver.CUdevice_attribute.CU_DEVICE_ATTRIBUTE_PCI_DEVICE_ID)

    @property
    def pci_domain_id(self) -> int:
        """int: PCI domain ID of the device."""
        return self._get_cached_attribute(driver.CUdevice_attribute.CU_DEVICE_ATTRIBUTE_PCI_DOMAIN_ID)

    @property
    def tcc_driver(self) -> bool:
        """bool: Device is using TCC driver model."""
        return bool(self._get_cached_attribute(driver.CUdevice_attribute.CU_DEVICE_ATTRIBUTE_TCC_DRIVER))

    @property
    def memory_clock_rate(self) -> int:
        """int: Peak memory clock frequency in kilohertz."""
        return self._get_attribute(driver.CUdevice_attribute.CU_DEVICE_ATTRIBUTE_MEMORY_CLOCK_RATE)

    @property
    def global_memory_bus_width(self) -> int:
        """int: Global memory bus width in bits."""
        return self._get_cached_attribute(driver.CUdevice_attribute.CU_DEVICE_ATTRIBUTE_GLOBAL_MEMORY_BUS_WIDTH)

    @property
    def l2_cache_size(self) -> int:
        """int: Size of L2 cache in bytes."""
        return self._get_cached_attribute(driver.CUdevice_attribute.CU_DEVICE_ATTRIBUTE_L2_CACHE_SIZE)

    @property
    def max_threads_per_multiprocessor(self) -> int:
        """int: Maximum resident threads per multiprocessor."""
        return self._get_cached_attribute(driver.CUdevice_attribute.CU_DEVICE_ATTRIBUTE_MAX_THREADS_PER_MULTIPROCESSOR)

    @property
    def unified_addressing(self) -> bool:
        """bool: Device shares a unified address space with the host."""
        return bool(self._get_cached_attribute(driver.CUdevice_attribute.CU_DEVICE_ATTRIBUTE_UNIFIED_ADDRESSING))

    @property
    def compute_capability_major(self) -> int:
        """int: Major compute capability version number."""
        return self._get_cached_attribute(driver.CUdevice_attribute.CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MAJOR)

    @property
    def compute_capability_minor(self) -> int:
        """int: Minor compute capability version number."""
        return self._get_cached_attribute(driver.CUdevice_attribute.CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MINOR)

    @property
    def global_l1_cache_supported(self) -> bool:
        """bool: Device supports caching globals in L1."""
        return bool(self._get_cached_attribute(driver.CUdevice_attribute.CU_DEVICE_ATTRIBUTE_GLOBAL_L1_CACHE_SUPPORTED))

    @property
    def local_l1_cache_supported(self) -> bool:
        """bool: Device supports caching locals in L1."""
        return bool(self._get_cached_attribute(driver.CUdevice_attribute.CU_DEVICE_ATTRIBUTE_LOCAL_L1_CACHE_SUPPORTED))

    @property
    def max_shared_memory_per_multiprocessor(self) -> int:
        """int: Maximum shared memory available per multiprocessor in bytes."""
        return self._get_cached_attribute(
            driver.CUdevice_attribute.CU_DEVICE_ATTRIBUTE_MAX_SHARED_MEMORY_PER_MULTIPROCESSOR
        )

    @property
    def max_registers_per_multiprocessor(self) -> int:
        """int: Maximum number of 32-bit registers available per multiprocessor."""
        return self._get_cached_attribute(
            driver.CUdevice_attribute.CU_DEVICE_ATTRIBUTE_MAX_REGISTERS_PER_MULTIPROCESSOR
        )

    @property
    def managed_memory(self) -> bool:
        """bool: Device can allocate managed memory on this system."""
        return bool(self._get_cached_attribute(driver.CUdevice_attribute.CU_DEVICE_ATTRIBUTE_MANAGED_MEMORY))

    @property
    def multi_gpu_board(self) -> bool:
        """bool: Device is on a multi-GPU board."""
        return bool(self._get_cached_attribute(driver.CUdevice_attribute.CU_DEVICE_ATTRIBUTE_MULTI_GPU_BOARD))

    @property
    def multi_gpu_board_group_id(self) -> int:
        """int: Unique id for a group of devices on the same multi-GPU board."""
        return self._get_cached_attribute(driver.CUdevice_attribute.CU_DEVICE_ATTRIBUTE_MULTI_GPU_BOARD_GROUP_ID)

    @property
    def host_native_atomic_supported(self) -> bool:
        """bool: Link between the device and the host supports all native atomic operations."""
        return bool(
            self._get_cached_attribute(driver.CUdevice_attribute.CU_DEVICE_ATTRIBUTE_HOST_NATIVE_ATOMIC_SUPPORTED)
        )

    @property
    def single_to_double_precision_perf_ratio(self) -> int:
        """int: Ratio of single precision performance (in floating-point operations per second) to double precision performance."""
        return self._get_attribute(driver.CUdevice_attribute.CU_DEVICE_ATTRIBUTE_SINGLE_TO_DOUBLE_PRECISION_PERF_RATIO)

    @property
    def pageable_memory_access(self) -> bool:
        """bool: Device supports coherently accessing pageable memory without calling cudaHostRegister on it."""
        return bool(self._get_cached_attribute(driver.CUdevice_attribute.CU_DEVICE_ATTRIBUTE_PAGEABLE_MEMORY_ACCESS))

    @property
    def concurrent_managed_access(self) -> bool:
        """bool: Device can coherently access managed memory concurrently with the CPU."""
        return bool(self._get_cached_attribute(driver.CUdevice_attribute.CU_DEVICE_ATTRIBUTE_CONCURRENT_MANAGED_ACCESS))

    @property
    def compute_preemption_supported(self) -> bool:
        """bool: Device supports compute preemption."""
        return bool(
            self._get_cached_attribute(driver.CUdevice_attribute.CU_DEVICE_ATTRIBUTE_COMPUTE_PREEMPTION_SUPPORTED)
        )

    @property
    def can_use_host_pointer_for_registered_mem(self) -> bool:
        """bool: Device can access host registered memory at the same virtual address as the CPU."""
        return bool(
            self._get_cached_attribute(
                driver.CUdevice_attribute.CU_DEVICE_ATTRIBUTE_CAN_USE_HOST_POINTER_FOR_REGISTERED_MEM
            )
        )

    # TODO: A few attrs are missing here (NVIDIA/cuda-python#675)

    @property
    def cooperative_launch(self) -> bool:
        """bool: Device supports launching cooperative kernels via cuLaunchCooperativeKernel."""
        return bool(self._get_cached_attribute(driver.CUdevice_attribute.CU_DEVICE_ATTRIBUTE_COOPERATIVE_LAUNCH))

    # TODO: A few attrs are missing here (NVIDIA/cuda-python#675)

    @property
    def max_shared_memory_per_block_optin(self) -> int:
        """int: Maximum optin shared memory per block."""
        return self._get_cached_attribute(
            driver.CUdevice_attribute.CU_DEVICE_ATTRIBUTE_MAX_SHARED_MEMORY_PER_BLOCK_OPTIN
        )

    @property
    def pageable_memory_access_uses_host_page_tables(self) -> bool:
        """bool: Device accesses pageable memory via the host's page tables."""
        return bool(
            self._get_cached_attribute(
                driver.CUdevice_attribute.CU_DEVICE_ATTRIBUTE_PAGEABLE_MEMORY_ACCESS_USES_HOST_PAGE_TABLES
            )
        )

    @property
    def direct_managed_mem_access_from_host(self) -> bool:
        """bool: The host can directly access managed memory on the device without migration."""
        return bool(
            self._get_cached_attribute(
                driver.CUdevice_attribute.CU_DEVICE_ATTRIBUTE_DIRECT_MANAGED_MEM_ACCESS_FROM_HOST
            )
        )

    @property
    def virtual_memory_management_supported(self) -> bool:
        """bool: Device supports virtual memory management APIs like cuMemAddressReserve, cuMemCreate, cuMemMap and related APIs."""
        return bool(
            self._get_cached_attribute(
                driver.CUdevice_attribute.CU_DEVICE_ATTRIBUTE_VIRTUAL_MEMORY_MANAGEMENT_SUPPORTED
            )
        )

    @property
    def handle_type_posix_file_descriptor_supported(self) -> bool:
        """bool: Device supports exporting memory to a posix file descriptor with cuMemExportToShareableHandle, if requested via cuMemCreate."""
        return bool(
            self._get_cached_attribute(
                driver.CUdevice_attribute.CU_DEVICE_ATTRIBUTE_HANDLE_TYPE_POSIX_FILE_DESCRIPTOR_SUPPORTED
            )
        )

    @property
    def handle_type_win32_handle_supported(self) -> bool:
        """bool: Device supports exporting memory to a Win32 NT handle with cuMemExportToShareableHandle, if requested via cuMemCreate."""
        return bool(
            self._get_cached_attribute(driver.CUdevice_attribute.CU_DEVICE_ATTRIBUTE_HANDLE_TYPE_WIN32_HANDLE_SUPPORTED)
        )

    @property
    def handle_type_win32_kmt_handle_supported(self) -> bool:
        """bool: Device supports exporting memory to a Win32 KMT handle with cuMemExportToShareableHandle, if requested via cuMemCreate."""
        return bool(
            self._get_cached_attribute(
                driver.CUdevice_attribute.CU_DEVICE_ATTRIBUTE_HANDLE_TYPE_WIN32_KMT_HANDLE_SUPPORTED
            )
        )

    @property
    def max_blocks_per_multiprocessor(self) -> int:
        """int: Maximum number of blocks per multiprocessor."""
        return self._get_cached_attribute(driver.CUdevice_attribute.CU_DEVICE_ATTRIBUTE_MAX_BLOCKS_PER_MULTIPROCESSOR)

    @property
    def generic_compression_supported(self) -> bool:
        """bool: Device supports compression of memory."""
        return bool(
            self._get_cached_attribute(driver.CUdevice_attribute.CU_DEVICE_ATTRIBUTE_GENERIC_COMPRESSION_SUPPORTED)
        )

    @property
    def max_persisting_l2_cache_size(self) -> int:
        """int: Maximum L2 persisting lines capacity setting in bytes."""
        return self._get_cached_attribute(driver.CUdevice_attribute.CU_DEVICE_ATTRIBUTE_MAX_PERSISTING_L2_CACHE_SIZE)

    @property
    def max_access_policy_window_size(self) -> int:
        """int: Maximum value of CUaccessPolicyWindow.num_bytes."""
        return self._get_cached_attribute(driver.CUdevice_attribute.CU_DEVICE_ATTRIBUTE_MAX_ACCESS_POLICY_WINDOW_SIZE)

    @property
    def gpu_direct_rdma_with_cuda_vmm_supported(self) -> bool:
        """bool: Device supports specifying the GPUDirect RDMA flag with cuMemCreate."""
        return bool(
            self._get_cached_attribute(
                driver.CUdevice_attribute.CU_DEVICE_ATTRIBUTE_GPU_DIRECT_RDMA_WITH_CUDA_VMM_SUPPORTED
            )
        )

    @property
    def reserved_shared_memory_per_block(self) -> int:
        """int: Shared memory reserved by CUDA driver per block in bytes."""
        return self._get_cached_attribute(
            driver.CUdevice_attribute.CU_DEVICE_ATTRIBUTE_RESERVED_SHARED_MEMORY_PER_BLOCK
        )

    @property
    def sparse_cuda_array_supported(self) -> bool:
        """bool: Device supports sparse CUDA arrays and sparse CUDA mipmapped arrays."""
        return bool(
            self._get_cached_attribute(driver.CUdevice_attribute.CU_DEVICE_ATTRIBUTE_SPARSE_CUDA_ARRAY_SUPPORTED)
        )

    @property
    def read_only_host_register_supported(self) -> bool:
        """bool: True if device supports using the cuMemHostRegister flag CU_MEMHOSTERGISTER_READ_ONLY to register memory that must be mapped as read-only to the GPU, False if not."""
        return bool(
            self._get_cached_attribute(driver.CUdevice_attribute.CU_DEVICE_ATTRIBUTE_READ_ONLY_HOST_REGISTER_SUPPORTED)
        )

    @property
    def memory_pools_supported(self) -> bool:
        """bool: Device supports using the cuMemAllocAsync and cuMemPool family of APIs."""
        return bool(self._get_cached_attribute(driver.CUdevice_attribute.CU_DEVICE_ATTRIBUTE_MEMORY_POOLS_SUPPORTED))

    @property
    def gpu_direct_rdma_supported(self) -> bool:
        """bool: Device supports GPUDirect RDMA APIs, like nvidia_p2p_get_pages (see https://docs.nvidia.com/cuda/gpudirect-rdma for more information)."""
        return bool(self._get_cached_attribute(driver.CUdevice_attribute.CU_DEVICE_ATTRIBUTE_GPU_DIRECT_RDMA_SUPPORTED))

    @property
    def gpu_direct_rdma_flush_writes_options(self) -> int:
        """int: The returned attribute shall be interpreted as a bitmask, where the individual bits are described by the CUflushGPUDirectRDMAWritesOptions enum."""
        return self._get_cached_attribute(
            driver.CUdevice_attribute.CU_DEVICE_ATTRIBUTE_GPU_DIRECT_RDMA_FLUSH_WRITES_OPTIONS
        )

    @property
    def gpu_direct_rdma_writes_ordering(self) -> int:
        """int: GPUDirect RDMA writes to the device do not need to be flushed for consumers within the scope indicated by the returned attribute. See CUGPUDirectRDMAWritesOrdering for the numerical values returned here."""
        return self._get_cached_attribute(driver.CUdevice_attribute.CU_DEVICE_ATTRIBUTE_GPU_DIRECT_RDMA_WRITES_ORDERING)

    @property
    def mempool_supported_handle_types(self) -> int:
        """int: Handle types supported with mempool based IPC."""
        return self._get_cached_attribute(driver.CUdevice_attribute.CU_DEVICE_ATTRIBUTE_MEMPOOL_SUPPORTED_HANDLE_TYPES)

    @property
    def deferred_mapping_cuda_array_supported(self) -> bool:
        """bool: Device supports deferred mapping CUDA arrays and CUDA mipmapped arrays."""
        return bool(
            self._get_cached_attribute(
                driver.CUdevice_attribute.CU_DEVICE_ATTRIBUTE_DEFERRED_MAPPING_CUDA_ARRAY_SUPPORTED
            )
        )

    @property
    def numa_config(self) -> int:
        """int: NUMA configuration of a device: value is of type CUdeviceNumaConfig enum."""
        return self._get_cached_attribute(driver.CUdevice_attribute.CU_DEVICE_ATTRIBUTE_NUMA_CONFIG)

    @property
    def numa_id(self) -> int:
        """int: NUMA node ID of the GPU memory."""
        return self._get_cached_attribute(driver.CUdevice_attribute.CU_DEVICE_ATTRIBUTE_NUMA_ID)

    @property
    def multicast_supported(self) -> bool:
        """bool: Device supports switch multicast and reduction operations."""
        return bool(self._get_cached_attribute(driver.CUdevice_attribute.CU_DEVICE_ATTRIBUTE_MULTICAST_SUPPORTED))

    @property
    def surface_alignment(self) -> int:
        """int: Surface alignment requirement in bytes."""
        return self._get_cached_attribute(driver.CUdevice_attribute.CU_DEVICE_ATTRIBUTE_SURFACE_ALIGNMENT)

    @property
    def async_engine_count(self) -> int:
        """int: Number of asynchronous engines."""
        return self._get_cached_attribute(driver.CUdevice_attribute.CU_DEVICE_ATTRIBUTE_ASYNC_ENGINE_COUNT)

    @property
    def can_tex2d_gather(self) -> bool:
        """bool: True if device supports 2D texture gather operations, False if not."""
        return bool(self._get_cached_attribute(driver.CUdevice_attribute.CU_DEVICE_ATTRIBUTE_CAN_TEX2D_GATHER))

    @property
    def maximum_texture2d_gather_width(self) -> int:
        """int: Maximum 2D texture gather width."""
        return self._get_cached_attribute(driver.CUdevice_attribute.CU_DEVICE_ATTRIBUTE_MAXIMUM_TEXTURE2D_GATHER_WIDTH)

    @property
    def maximum_texture2d_gather_height(self) -> int:
        """int: Maximum 2D texture gather height."""
        return self._get_cached_attribute(driver.CUdevice_attribute.CU_DEVICE_ATTRIBUTE_MAXIMUM_TEXTURE2D_GATHER_HEIGHT)

    @property
    def stream_priorities_supported(self) -> bool:
        """bool: True if device supports stream priorities, False if not."""
        return bool(
            self._get_cached_attribute(driver.CUdevice_attribute.CU_DEVICE_ATTRIBUTE_STREAM_PRIORITIES_SUPPORTED)
        )

    @property
    def can_flush_remote_writes(self) -> bool:
        """bool: The CU_STREAM_WAIT_VALUE_FLUSH flag and the CU_STREAM_MEM_OP_FLUSH_REMOTE_WRITES MemOp are supported on the device. See Stream Memory Operations for additional details."""
        return bool(self._get_cached_attribute(driver.CUdevice_attribute.CU_DEVICE_ATTRIBUTE_CAN_FLUSH_REMOTE_WRITES))

    @property
    def host_register_supported(self) -> bool:
        """bool: Device supports host memory registration via cudaHostRegister."""
        return bool(self._get_cached_attribute(driver.CUdevice_attribute.CU_DEVICE_ATTRIBUTE_HOST_REGISTER_SUPPORTED))

    @property
    def timeline_semaphore_interop_supported(self) -> bool:
        """bool: External timeline semaphore interop is supported on the device."""
        return bool(
            self._get_cached_attribute(
                driver.CUdevice_attribute.CU_DEVICE_ATTRIBUTE_TIMELINE_SEMAPHORE_INTEROP_SUPPORTED
            )
        )

    @property
    def cluster_launch(self) -> bool:
        """bool: Indicates device supports cluster launch."""
        return bool(self._get_cached_attribute(driver.CUdevice_attribute.CU_DEVICE_ATTRIBUTE_CLUSTER_LAUNCH))

    @property
    def can_use_64_bit_stream_mem_ops(self) -> bool:
        """bool: 64-bit operations are supported in cuStreamBatchMemOp and related MemOp APIs."""
        return bool(
            self._get_cached_attribute(driver.CUdevice_attribute.CU_DEVICE_ATTRIBUTE_CAN_USE_64_BIT_STREAM_MEM_OPS)
        )

    @property
    def can_use_stream_wait_value_nor(self) -> bool:
        """bool: CU_STREAM_WAIT_VALUE_NOR is supported by MemOp APIs."""
        return bool(
            self._get_cached_attribute(driver.CUdevice_attribute.CU_DEVICE_ATTRIBUTE_CAN_USE_STREAM_WAIT_VALUE_NOR)
        )

    @property
    def dma_buf_supported(self) -> bool:
        """bool: Device supports buffer sharing with dma_buf mechanism."""
        return bool(self._get_cached_attribute(driver.CUdevice_attribute.CU_DEVICE_ATTRIBUTE_DMA_BUF_SUPPORTED))

    # Start of CUDA 12 device attributes

    @property
    def ipc_event_supported(self) -> bool:
        """bool: Device supports IPC Events."""
        return bool(self._get_cached_attribute(driver.CUdevice_attribute.CU_DEVICE_ATTRIBUTE_IPC_EVENT_SUPPORTED))

    @property
    def mem_sync_domain_count(self) -> int:
        """int: Number of memory domains the device supports."""
        return self._get_cached_attribute(driver.CUdevice_attribute.CU_DEVICE_ATTRIBUTE_MEM_SYNC_DOMAIN_COUNT, default=1)

    @property
    def tensor_map_access_supported(self) -> bool:
        """bool: Device supports accessing memory using Tensor Map."""
        return bool(
            self._get_cached_attribute(driver.CUdevice_attribute.CU_DEVICE_ATTRIBUTE_TENSOR_MAP_ACCESS_SUPPORTED)
        )

    @property
    def handle_type_fabric_supported(self) -> bool:
        """bool: Device supports exporting memory to a fabric handle with cuMemExportToShareableHandle() or requested with cuMemCreate()."""
        return bool(
            self._get_cached_attribute(driver.CUdevice_attribute.CU_DEVICE_ATTRIBUTE_HANDLE_TYPE_FABRIC_SUPPORTED)
        )

    @property
    def unified_function_pointers(self) -> bool:
        """bool: Device supports unified function pointers."""
        return bool(self._get_cached_attribute(driver.CUdevice_attribute.CU_DEVICE_ATTRIBUTE_UNIFIED_FUNCTION_POINTERS))

    @property
    def mps_enabled(self) -> bool:
        """bool: Indicates if contexts created on this device will be shared via MPS."""
        return bool(self._get_cached_attribute(driver.CUdevice_attribute.CU_DEVICE_ATTRIBUTE_MPS_ENABLED))

    @property
    def host_numa_id(self) -> int:
        """int: NUMA ID of the host node closest to the device. Returns -1 when system does not support NUMA."""
        return self._get_cached_attribute(driver.CUdevice_attribute.CU_DEVICE_ATTRIBUTE_HOST_NUMA_ID, default=-1)

    @property
    def d3d12_cig_supported(self) -> bool:
        """bool: Device supports CIG with D3D12."""
        return bool(self._get_cached_attribute(driver.CUdevice_attribute.CU_DEVICE_ATTRIBUTE_D3D12_CIG_SUPPORTED))

    @property
    def mem_decompress_algorithm_mask(self) -> int:
        """int: The returned valued shall be interpreted as a bitmask, where the individual bits are described by the CUmemDecompressAlgorithm enum."""
        return self._get_cached_attribute(driver.CUdevice_attribute.CU_DEVICE_ATTRIBUTE_MEM_DECOMPRESS_ALGORITHM_MASK)

    @property
    def mem_decompress_maximum_length(self) -> int:
        """int: The returned valued is the maximum length in bytes of a single decompress operation that is allowed."""
        return self._get_cached_attribute(driver.CUdevice_attribute.CU_DEVICE_ATTRIBUTE_MEM_DECOMPRESS_MAXIMUM_LENGTH)

    @property
    def vulkan_cig_supported(self) -> bool:
        """bool: Device supports CIG with Vulkan."""
        return bool(self._get_cached_attribute(driver.CUdevice_attribute.CU_DEVICE_ATTRIBUTE_VULKAN_CIG_SUPPORTED))

    @property
    def gpu_pci_device_id(self) -> int:
        """int: The combined 16-bit PCI device ID and 16-bit PCI vendor ID.

        Returns 0 if the driver does not support this query.
        """
        return self._get_cached_attribute(driver.CUdevice_attribute.CU_DEVICE_ATTRIBUTE_GPU_PCI_DEVICE_ID)

    @property
    def gpu_pci_subsystem_id(self) -> int:
        """int: The combined 16-bit PCI subsystem ID and 16-bit PCI subsystem vendor ID.

        Returns 0 if the driver does not support this query.
        """
        return self._get_cached_attribute(driver.CUdevice_attribute.CU_DEVICE_ATTRIBUTE_GPU_PCI_SUBSYSTEM_ID)

    @property
    def host_numa_virtual_memory_management_supported(self) -> bool:
        """bool: Device supports HOST_NUMA location with the virtual memory management APIs like cuMemCreate, cuMemMap and related APIs."""
        return bool(
            self._get_cached_attribute(
                driver.CUdevice_attribute.CU_DEVICE_ATTRIBUTE_HOST_NUMA_VIRTUAL_MEMORY_MANAGEMENT_SUPPORTED
            )
        )

    @property
    def host_numa_memory_pools_supported(self) -> bool:
        """bool: Device supports HOST_NUMA location with the cuMemAllocAsync and cuMemPool family of APIs."""
        return bool(
            self._get_cached_attribute(driver.CUdevice_attribute.CU_DEVICE_ATTRIBUTE_HOST_NUMA_MEMORY_POOLS_SUPPORTED)
        )

    # Start of CUDA 13 device attributes

    @property
    def host_numa_multinode_ipc_supported(self) -> bool:
        """bool: Device supports HOST_NUMA location IPC between nodes in a multi-node system."""
        return bool(
            self._get_cached_attribute(driver.CUdevice_attribute.CU_DEVICE_ATTRIBUTE_HOST_NUMA_MULTINODE_IPC_SUPPORTED)
        )

    @property
    def host_memory_pools_supported(self) -> bool:
        """bool: Device suports HOST location with the cuMemAllocAsync and cuMemPool family of APIs."""
        return bool(
            self._get_cached_attribute(driver.CUdevice_attribute.CU_DEVICE_ATTRIBUTE_HOST_MEMORY_POOLS_SUPPORTED)
        )

    @property
    def host_virtual_memory_management_supported(self) -> bool:
        """bool: Device supports HOST location with the virtual memory management APIs like cuMemCreate, cuMemMap and related APIs."""
        return bool(
            self._get_cached_attribute(
                driver.CUdevice_attribute.CU_DEVICE_ATTRIBUTE_HOST_VIRTUAL_MEMORY_MANAGEMENT_SUPPORTED
            )
        )

    @property
    def host_alloc_dma_buf_supported(self) -> bool:
        """bool: Device supports page-locked host memory buffer sharing with dma_buf mechanism."""
        return bool(
            self._get_cached_attribute(driver.CUdevice_attribute.CU_DEVICE_ATTRIBUTE_HOST_ALLOC_DMA_BUF_SUPPORTED)
        )

    @property
    def only_partial_host_native_atomic_supported(self) -> bool:
        """bool: Link between the device and the host supports only some native atomic operations."""
        return bool(
            self._get_cached_attribute(
                driver.CUdevice_attribute.CU_DEVICE_ATTRIBUTE_ONLY_PARTIAL_HOST_NATIVE_ATOMIC_SUPPORTED
            )
        )


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
    __slots__ = ("_device_id", "_memory_resource", "_has_inited", "_properties", "_uuid", "_context")

    def __new__(cls, device_id: Device | int | None = None):
        if isinstance(device_id, Device):
            return device_id

        Device_ensure_cuda_initialized()
        device_id = Device_resolve_device_id(device_id)
        devices = Device_ensure_tls_devices(cls)

        try:
            return devices[device_id]
        except IndexError:
            raise ValueError(f"device_id must be within [0, {len(devices)}), got {device_id}") from None

    def _check_context_initialized(self):
        if not self._has_inited:
            raise CUDAError(
                f"Device {self._device_id} is not yet initialized, perhaps you forgot to call .set_current() first?"
            )


    @classmethod
    def get_all_devices(cls):
        """
        Query the available device instances.

        Returns
        -------
        tuple of Device
            A tuple containing instances of available devices.
        """
        from cuda.core import system
        total = system.get_num_devices()
        return tuple(cls(device_id) for device_id in range(total))

    def to_system_device(self) -> 'cuda.core.system.Device':
        """
        Get the corresponding :class:`cuda.core.system.Device` (which is used
        for NVIDIA Machine Library (NVML) access) for this
        :class:`cuda.core.Device` (which is used for CUDA access).

        The devices are mapped to one another by their UUID.

        Returns
        -------
        cuda.core.system.Device
            The corresponding system-level device instance used for NVML access.
        """
        from cuda.core.system._system import CUDA_BINDINGS_NVML_IS_COMPATIBLE

        if not CUDA_BINDINGS_NVML_IS_COMPATIBLE:
            raise RuntimeError(
                "cuda.core.system.Device requires cuda_bindings 13.1.2+ or 12.9.6+"
            )

        from cuda.core.system import Device as SystemDevice
        return SystemDevice(uuid=self.uuid)

    @property
    def device_id(self) -> int:
        """Return device ordinal."""
        return self._device_id

    @property
    def pci_bus_id(self) -> str:
        """Return a PCI Bus Id string for this device."""
        bus_id = handle_return(runtime.cudaDeviceGetPCIBusId(13, self._device_id))
        return bus_id[:12].decode()

    def can_access_peer(self, peer: Device | int) -> bool:
        """Check if this device can access memory from the specified peer device.

        Queries whether peer-to-peer memory access is supported between this
        device and the specified peer device.

        Parameters
        ----------
        peer : Device | int
            The peer device to check accessibility to. Can be a Device object or device ID.
        """
        peer = Device(peer)
        cdef int d1 = <int> self.device_id
        cdef int d2 = <int> peer.device_id
        if d1 == d2:
            return True
        cdef int value = 0
        with nogil:
            HANDLE_RETURN(cydriver.cuDeviceCanAccessPeer(&value, d1, d2))
        return bool(value)

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

        The UUID is cached after first access to avoid repeated CUDA API calls.

        """
        cdef cydriver.CUuuid uuid
        cdef cydriver.CUdevice dev
        cdef bytes uuid_b
        cdef str uuid_hex

        if self._uuid is None:
            dev = self._device_id
            with nogil:
                IF CUDA_CORE_BUILD_MAJOR == 12:
                    HANDLE_RETURN(cydriver.cuDeviceGetUuid_v2(&uuid, dev))
                ELSE:  # 13.0+
                    HANDLE_RETURN(cydriver.cuDeviceGetUuid(&uuid, dev))
            uuid_b = cpython.PyBytes_FromStringAndSize(uuid.bytes, sizeof(uuid.bytes))
            uuid_hex = uuid_b.hex()
            # 8-4-4-4-12
            self._uuid = f"{uuid_hex[:8]}-{uuid_hex[8:12]}-{uuid_hex[12:16]}-{uuid_hex[16:20]}-{uuid_hex[20:]}"
        return self._uuid

    @property
    def name(self) -> str:
        """Return the device name."""
        # Use 256 characters to be consistent with CUDA Runtime
        cdef int LENGTH = 256
        cdef bytes name = bytes(LENGTH)
        cdef char* name_ptr = name
        cdef cydriver.CUdevice this_dev = self._device_id
        with nogil:
            HANDLE_RETURN(cydriver.cuDeviceGetName(name_ptr, LENGTH, this_dev))
        name = name.split(b"\0")[0]
        return name.decode()

    @property
    def properties(self) -> DeviceProperties:
        """Return a :obj:`~_device.DeviceProperties` class with information about the device."""
        if self._properties is None:
            self._properties = DeviceProperties._init(self._device_id)

        return self._properties

    @property
    def compute_capability(self) -> ComputeCapability:
        """Return a named tuple with 2 fields: major and minor."""
        cdef DeviceProperties prop = self.properties
        if "compute_capability" in prop._cache:
            return prop._cache["compute_capability"]
        cc = ComputeCapability(prop.compute_capability_major, prop.compute_capability_minor)
        prop._cache["compute_capability"] = cc
        return cc

    @property
    def arch(self) -> str:
        """Return compute capability as a string (e.g., '75' for CC 7.5)."""
        return f"{self.compute_capability.major}{self.compute_capability.minor}"

    @property
    def context(self) -> Context:
        """Return the :obj:`~_context.Context` associated with this device.

        Note
        ----
        Device must be initialized.

        """
        self._check_context_initialized()
        return self._context

    @property
    def memory_resource(self) -> MemoryResource:
        """Return :obj:`~_memory.MemoryResource` associated with this device."""
        cdef int attr, device_id
        if self._memory_resource is None:
            # If the device is in TCC mode, or does not support memory pools for some other reason,
            # use the SynchronousMemoryResource which does not use memory pools.
            device_id = self._device_id
            with nogil:
                HANDLE_RETURN(
                    cydriver.cuDeviceGetAttribute(
                        &attr, cydriver.CUdevice_attribute.CU_DEVICE_ATTRIBUTE_MEMORY_POOLS_SUPPORTED, device_id
                    )
                )
            if attr == 1:
                from cuda.core._memory import DeviceMemoryResource
                self._memory_resource = DeviceMemoryResource(self._device_id)
            else:
                from cuda.core._memory import _SynchronousMemoryResource
                self._memory_resource = _SynchronousMemoryResource(self._device_id)

        return self._memory_resource

    @memory_resource.setter
    def memory_resource(self, mr):
        from cuda.core._memory import MemoryResource
        assert_type(mr, MemoryResource)
        self._memory_resource = mr

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
        return self._device_id

    def __repr__(self):
        return f"<Device {self._device_id} ({self.name})>"

    def __hash__(self) -> int:
        return hash(self.uuid)

    def __eq__(self, other) -> bool:
        if not isinstance(other, Device):
            return NotImplemented
        return self._device_id == other._device_id

    def __reduce__(self):
        return Device, (self.device_id,)

    def set_current(self, ctx: Context = None) -> Context | None:
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
        :obj:`~_context.Context`, optional
            Popped context.

        Examples
        --------
        Acts as an entry point of this object. Users always start a code by
        calling this method, e.g.

        >>> from cuda.core import Device
        >>> dev0 = Device(0)
        >>> dev0.set_current()
        >>> # ... do work on device 0 ...

        """
        cdef ContextHandle h_context
        cdef cydriver.CUcontext prev_ctx, curr_ctx

        if ctx is not None:
            # TODO: revisit once Context is cythonized
            assert_type(ctx, Context)
            if ctx._device_id != self._device_id:
                raise RuntimeError(
                    "the provided context was created on the device with"
                    f" id={ctx._device_id}, which is different from the target id={self._device_id}"
                )
            # prev_ctx is the previous context
            curr_ctx = as_cu(ctx._h_context)
            prev_ctx = NULL
            with nogil:
                HANDLE_RETURN(cydriver.cuCtxPopCurrent(&prev_ctx))
                HANDLE_RETURN(cydriver.cuCtxPushCurrent(curr_ctx))
            self._has_inited = True
            self._context = ctx  # Store owning context reference
            if prev_ctx != NULL:
                return Context._from_handle(Context, create_context_handle_ref(prev_ctx), self._device_id)
        else:
            # use primary ctx
            h_context = get_primary_context(self._device_id)
            if h_context.get() == NULL:
                raise ValueError("Cannot set NULL context as current")
            with nogil:
                HANDLE_RETURN(cydriver.cuCtxSetCurrent(as_cu(h_context)))
            self._has_inited = True
            self._context = Context._from_handle(Context, h_context, self._device_id)  # Store owning context

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

    def create_stream(self, obj: IsStreamT | None = None, options: StreamOptions | None = None) -> Stream:
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
        self._check_context_initialized()
        return Stream._init(obj=obj, options=options, device_id=self._device_id, ctx=self._context)

    def create_event(self, options: EventOptions | None = None) -> Event:
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
        self._check_context_initialized()
        cdef Context ctx = self._context
        return cyEvent._init(cyEvent, self._device_id, ctx._h_context, options, True)

    def allocate(self, size, stream: Stream | GraphBuilder | None = None) -> Buffer:
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
        self._check_context_initialized()
        return self.memory_resource.allocate(size, stream)

    def sync(self):
        """Synchronize the device.

        Note
        ----
        Device must be initialized.

        """
        self._check_context_initialized()
        handle_return(runtime.cudaDeviceSynchronize())

    def create_graph_builder(self) -> GraphBuilder:
        """Create a new :obj:`~_graph.GraphBuilder` object.

        Returns
        -------
        :obj:`~_graph.GraphBuilder`
            Newly created graph builder object.

        """
        self._check_context_initialized()
        return GraphBuilder._init(stream=self.create_stream(), is_stream_owner=True)


cdef inline int Device_ensure_cuda_initialized() except? -1:
    """Initialize CUDA driver and check version compatibility (once per process)."""
    global _is_cuInit
    if _is_cuInit is False:
        with _lock, nogil:
            HANDLE_RETURN(cydriver.cuInit(0))
            _is_cuInit = True
        try:
            from cuda.bindings.utils import warn_if_cuda_major_version_mismatch
        except ImportError:
            pass
        else:
            warn_if_cuda_major_version_mismatch()
    return 0


cdef inline int Device_resolve_device_id(device_id) except? -1:
    """Resolve device_id, defaulting to current device or 0."""
    cdef cydriver.CUdevice dev
    cdef cydriver.CUcontext ctx
    cdef cydriver.CUresult err
    if device_id is None:
        with nogil:
            err = cydriver.cuCtxGetDevice(&dev)
        if err == cydriver.CUresult.CUDA_SUCCESS:
            return int(dev)
        elif err == cydriver.CUresult.CUDA_ERROR_INVALID_CONTEXT:
            with nogil:
                HANDLE_RETURN(cydriver.cuCtxGetCurrent(&ctx))
            assert <void*>(ctx) == NULL
            return 0  # cudart behavior
        else:
            HANDLE_RETURN(err)
    elif device_id < 0:
        raise ValueError(f"device_id must be >= 0, got {device_id}")
    return device_id


cdef inline list Device_ensure_tls_devices(cls):
    """Ensure thread-local Device singletons exist, creating if needed."""
    cdef int total
    try:
        return _tls.devices
    except AttributeError:
        with nogil:
            HANDLE_RETURN(cydriver.cuDeviceGetCount(&total))
        devices = _tls.devices = []
        for dev_id in range(total):
            device = super(Device, cls).__new__(cls)
            device._device_id = dev_id
            device._memory_resource = None
            device._has_inited = False
            device._properties = None
            device._uuid = None
            device._context = None
            devices.append(device)
        return devices
