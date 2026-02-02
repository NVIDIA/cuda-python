# SPDX-FileCopyrightText: Copyright (c) 2024-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from libc.stddef cimport size_t

import functools
import threading
from collections import namedtuple

from cuda.core._device import Device
from cuda.core._launch_config cimport LaunchConfig
from cuda.core._launch_config import LaunchConfig
from cuda.core._stream cimport Stream
from cuda.core._resource_handles cimport (
    LibraryHandle,
    KernelHandle,
    create_library_handle_from_file,
    create_library_handle_from_data,
    create_library_handle_ref,
    create_kernel_handle,
    create_kernel_handle_ref,
    get_last_error,
    as_cu,
    as_py,
    as_intptr,
)
from cuda.core._stream import Stream
from cuda.core._utils.clear_error_support import (
    assert_type_str_or_bytes_like,
    raise_code_path_meant_to_be_unreachable,
)
from cuda.core._utils.cuda_utils cimport HANDLE_RETURN
from cuda.core._utils.cuda_utils import driver, get_binding_version
from cuda.bindings cimport cydriver

__all__ = ["Kernel", "ObjectCode"]

# Lazy initialization state and synchronization
# For Python 3.13t (free-threaded builds), we use a lock to ensure thread-safe initialization.
# For regular Python builds with GIL, the lock overhead is minimal and the code remains safe.
cdef object _init_lock = threading.Lock()
cdef bint _inited = False
cdef int _py_major_ver = 0
cdef int _py_minor_ver = 0
cdef int _driver_ver = 0
cdef tuple _kernel_ctypes = None
cdef bint _paraminfo_supported = False


cdef int _lazy_init() except -1:
    """
    Initialize module-level state in a thread-safe manner.

    This function is thread-safe and suitable for both:
    - Regular Python builds (with GIL)
    - Python 3.13t free-threaded builds (without GIL)

    Uses double-checked locking pattern for performance:
    - Fast path: check without lock if already initialized
    - Slow path: acquire lock and initialize if needed
    """
    global _inited
    # Fast path: already initialized (no lock needed for read)
    if _inited:
        return 0

    cdef int drv_ver
    # Slow path: acquire lock and initialize
    with _init_lock:
        # Double-check: another thread might have initialized while we waited
        if _inited:
            return 0

        global _py_major_ver, _py_minor_ver, _driver_ver, _kernel_ctypes, _paraminfo_supported
        # binding availability depends on cuda-python version
        _py_major_ver, _py_minor_ver = get_binding_version()
        _kernel_ctypes = (driver.CUkernel,)
        with nogil:
            HANDLE_RETURN(cydriver.cuDriverGetVersion(&drv_ver))
        _driver_ver = drv_ver
        _paraminfo_supported = _driver_ver >= 12040

        # Mark as initialized (must be last to ensure all state is set)
        _inited = True

    return 0


# Auto-initializing accessors (cdef for internal use)
cdef inline int _get_py_major_ver() except -1:
    """Get the Python binding major version, initializing if needed."""
    _lazy_init()
    return _py_major_ver


cdef inline int _get_py_minor_ver() except -1:
    """Get the Python binding minor version, initializing if needed."""
    _lazy_init()
    return _py_minor_ver


cdef inline int _get_driver_ver() except -1:
    """Get the CUDA driver version, initializing if needed."""
    _lazy_init()
    return _driver_ver


cdef inline tuple _get_kernel_ctypes():
    """Get the kernel ctypes tuple, initializing if needed."""
    _lazy_init()
    return _kernel_ctypes


cdef inline bint _is_paraminfo_supported() except -1:
    """Return True if cuKernelGetParamInfo is available (driver >= 12.4)."""
    _lazy_init()
    return _paraminfo_supported


@functools.cache
def _is_cukernel_get_library_supported() -> bool:
    """Return True when cuKernelGetLibrary is available for inverse kernel-to-library lookup.

    Requires cuda-python bindings >= 12.5 and driver >= 12.5.
    """
    return (
        (_get_py_major_ver(), _get_py_minor_ver()) >= (12, 5)
        and _get_driver_ver() >= 12050
        and hasattr(driver, "cuKernelGetLibrary")
    )


cdef inline LibraryHandle _make_empty_library_handle():
    """Create an empty LibraryHandle to indicate no library loaded."""
    return LibraryHandle()  # Empty shared_ptr


cdef class KernelAttributes:
    """Provides access to kernel attributes."""

    def __init__(self, *args, **kwargs):
        raise RuntimeError("KernelAttributes cannot be instantiated directly. Please use Kernel APIs.")

    @staticmethod
    cdef KernelAttributes _init(KernelHandle h_kernel):
        cdef KernelAttributes self = KernelAttributes.__new__(KernelAttributes)
        self._h_kernel = h_kernel
        self._cache = {}
        _lazy_init()
        return self

    cdef int _get_cached_attribute(self, int device_id, cydriver.CUfunction_attribute attribute) except? -1:
        """Helper function to get a cached attribute or fetch and cache it if not present."""
        cdef tuple cache_key = (device_id, <int>attribute)
        cached = self._cache.get(cache_key, cache_key)
        if cached is not cache_key:
            return cached
        cdef int result
        with nogil:
            HANDLE_RETURN(cydriver.cuKernelGetAttribute(&result, attribute, as_cu(self._h_kernel), device_id))
        self._cache[cache_key] = result
        return result

    cdef inline int _resolve_device_id(self, device_id) except? -1:
        """Convert Device or int to device_id int."""
        return Device(device_id).device_id

    def max_threads_per_block(self, device_id: Device | int = None) -> int:
        """int : The maximum number of threads per block.
        This attribute is read-only."""
        return self._get_cached_attribute(
            self._resolve_device_id(device_id), cydriver.CU_FUNC_ATTRIBUTE_MAX_THREADS_PER_BLOCK
        )

    def shared_size_bytes(self, device_id: Device | int = None) -> int:
        """int : The size in bytes of statically-allocated shared memory required by this function.
        This attribute is read-only."""
        return self._get_cached_attribute(
            self._resolve_device_id(device_id), cydriver.CU_FUNC_ATTRIBUTE_SHARED_SIZE_BYTES
        )

    def const_size_bytes(self, device_id: Device | int = None) -> int:
        """int : The size in bytes of user-allocated constant memory required by this function.
        This attribute is read-only."""
        return self._get_cached_attribute(
            self._resolve_device_id(device_id), cydriver.CU_FUNC_ATTRIBUTE_CONST_SIZE_BYTES
        )

    def local_size_bytes(self, device_id: Device | int = None) -> int:
        """int : The size in bytes of local memory used by each thread of this function.
        This attribute is read-only."""
        return self._get_cached_attribute(
            self._resolve_device_id(device_id), cydriver.CU_FUNC_ATTRIBUTE_LOCAL_SIZE_BYTES
        )

    def num_regs(self, device_id: Device | int = None) -> int:
        """int : The number of registers used by each thread of this function.
        This attribute is read-only."""
        return self._get_cached_attribute(
            self._resolve_device_id(device_id), cydriver.CU_FUNC_ATTRIBUTE_NUM_REGS
        )

    def ptx_version(self, device_id: Device | int = None) -> int:
        """int : The PTX virtual architecture version for which the function was compiled.
        This attribute is read-only."""
        return self._get_cached_attribute(
            self._resolve_device_id(device_id), cydriver.CU_FUNC_ATTRIBUTE_PTX_VERSION
        )

    def binary_version(self, device_id: Device | int = None) -> int:
        """int : The binary architecture version for which the function was compiled.
        This attribute is read-only."""
        return self._get_cached_attribute(
            self._resolve_device_id(device_id), cydriver.CU_FUNC_ATTRIBUTE_BINARY_VERSION
        )

    def cache_mode_ca(self, device_id: Device | int = None) -> bool:
        """bool : Whether the function has been compiled with user specified option "-Xptxas --dlcm=ca" set.
        This attribute is read-only."""
        return bool(
            self._get_cached_attribute(
                self._resolve_device_id(device_id), cydriver.CU_FUNC_ATTRIBUTE_CACHE_MODE_CA
            )
        )

    def max_dynamic_shared_size_bytes(self, device_id: Device | int = None) -> int:
        """int : The maximum size in bytes of dynamically-allocated shared memory that can be used
        by this function."""
        return self._get_cached_attribute(
            self._resolve_device_id(device_id), cydriver.CU_FUNC_ATTRIBUTE_MAX_DYNAMIC_SHARED_SIZE_BYTES
        )

    def preferred_shared_memory_carveout(self, device_id: Device | int = None) -> int:
        """int : The shared memory carveout preference, in percent of the total shared memory."""
        return self._get_cached_attribute(
            self._resolve_device_id(device_id), cydriver.CU_FUNC_ATTRIBUTE_PREFERRED_SHARED_MEMORY_CARVEOUT
        )

    def cluster_size_must_be_set(self, device_id: Device | int = None) -> bool:
        """bool : The kernel must launch with a valid cluster size specified.
        This attribute is read-only."""
        return bool(
            self._get_cached_attribute(
                self._resolve_device_id(device_id), cydriver.CU_FUNC_ATTRIBUTE_CLUSTER_SIZE_MUST_BE_SET
            )
        )

    def required_cluster_width(self, device_id: Device | int = None) -> int:
        """int : The required cluster width in blocks."""
        return self._get_cached_attribute(
            self._resolve_device_id(device_id), cydriver.CU_FUNC_ATTRIBUTE_REQUIRED_CLUSTER_WIDTH
        )

    def required_cluster_height(self, device_id: Device | int = None) -> int:
        """int : The required cluster height in blocks."""
        return self._get_cached_attribute(
            self._resolve_device_id(device_id), cydriver.CU_FUNC_ATTRIBUTE_REQUIRED_CLUSTER_HEIGHT
        )

    def required_cluster_depth(self, device_id: Device | int = None) -> int:
        """int : The required cluster depth in blocks."""
        return self._get_cached_attribute(
            self._resolve_device_id(device_id), cydriver.CU_FUNC_ATTRIBUTE_REQUIRED_CLUSTER_DEPTH
        )

    def non_portable_cluster_size_allowed(self, device_id: Device | int = None) -> bool:
        """bool : Whether the function can be launched with non-portable cluster size."""
        return bool(
            self._get_cached_attribute(
                self._resolve_device_id(device_id),
                cydriver.CU_FUNC_ATTRIBUTE_NON_PORTABLE_CLUSTER_SIZE_ALLOWED,
            )
        )

    def cluster_scheduling_policy_preference(self, device_id: Device | int = None) -> int:
        """int : The block scheduling policy of a function."""
        return self._get_cached_attribute(
            self._resolve_device_id(device_id),
            cydriver.CU_FUNC_ATTRIBUTE_CLUSTER_SCHEDULING_POLICY_PREFERENCE,
        )


MaxPotentialBlockSizeOccupancyResult = namedtuple("MaxPotential", ("min_grid_size", "max_block_size"))


cdef class KernelOccupancy:
    """This class offers methods to query occupancy metrics that help determine optimal
    launch parameters such as block size, grid size, and shared memory usage.
    """

    def __init__(self, *args, **kwargs):
        raise RuntimeError("KernelOccupancy cannot be instantiated directly. Please use Kernel APIs.")

    @staticmethod
    cdef KernelOccupancy _init(KernelHandle h_kernel):
        cdef KernelOccupancy self = KernelOccupancy.__new__(KernelOccupancy)
        self._h_kernel = h_kernel
        return self

    def max_active_blocks_per_multiprocessor(self, block_size: int, dynamic_shared_memory_size: int) -> int:
        """Occupancy of the kernel.

        Returns the maximum number of active blocks per multiprocessor for this kernel.

        Parameters
        ----------
            block_size: int
                Block size parameter used to launch this kernel.
            dynamic_shared_memory_size: int
                The amount of dynamic shared memory in bytes needed by block.
                Use `0` if block does not need shared memory.

        Returns
        -------
        int
            The maximum number of active blocks per multiprocessor.

        Note
        ----
            The fraction of the product of maximum number of active blocks per multiprocessor
            and the block size to the maximum number of threads per multiprocessor is known as
            theoretical multiprocessor utilization (occupancy).

        """
        cdef int num_blocks
        cdef int c_block_size = block_size
        cdef size_t c_shmem_size = dynamic_shared_memory_size
        cdef cydriver.CUfunction func = <cydriver.CUfunction>as_cu(self._h_kernel)
        with nogil:
            HANDLE_RETURN(cydriver.cuOccupancyMaxActiveBlocksPerMultiprocessor(
                &num_blocks, func, c_block_size, c_shmem_size
            ))
        return num_blocks

    def max_potential_block_size(
        self, dynamic_shared_memory_needed: int | driver.CUoccupancyB2DSize, block_size_limit: int
    ) -> MaxPotentialBlockSizeOccupancyResult:
        """MaxPotentialBlockSizeOccupancyResult: Suggested launch configuration for reasonable occupancy.

        Returns the minimum grid size needed to achieve the maximum occupancy and
        the maximum block size that can achieve the maximum occupancy.

        Parameters
        ----------
            dynamic_shared_memory_needed: Union[int, driver.CUoccupancyB2DSize]
                The amount of dynamic shared memory in bytes needed by block.
                Use `0` if block does not need shared memory. Use C-callable
                represented by :obj:`~driver.CUoccupancyB2DSize` to encode
                amount of needed dynamic shared memory which varies depending
                on tne block size.
            block_size_limit: int
                Known upper limit on the kernel block size. Use `0` to indicate
                the maximum block size permitted by the device / kernel instead

        Returns
        -------
        :obj:`~MaxPotentialBlockSizeOccupancyResult`
            An object with `min_grid_size` amd `max_block_size` attributes encoding
            the suggested launch configuration.

        Note
        ----
            Please be advised that use of C-callable that requires Python Global
            Interpreter Lock may lead to deadlocks.

        """
        cdef int min_grid_size, max_block_size
        cdef cydriver.CUfunction func = <cydriver.CUfunction>as_cu(self._h_kernel)
        cdef cydriver.CUoccupancyB2DSize callback
        cdef size_t c_shmem_size
        cdef int c_block_size_limit = block_size_limit
        if isinstance(dynamic_shared_memory_needed, int):
            c_shmem_size = dynamic_shared_memory_needed
            with nogil:
                HANDLE_RETURN(cydriver.cuOccupancyMaxPotentialBlockSize(
                    &min_grid_size, &max_block_size, func, NULL, c_shmem_size, c_block_size_limit
                ))
        elif isinstance(dynamic_shared_memory_needed, driver.CUoccupancyB2DSize):
            # Callback may require GIL, so don't use nogil here
            callback = <cydriver.CUoccupancyB2DSize><size_t>dynamic_shared_memory_needed.getPtr()
            HANDLE_RETURN(cydriver.cuOccupancyMaxPotentialBlockSize(
                &min_grid_size, &max_block_size, func, callback, 0, c_block_size_limit
            ))
        else:
            raise TypeError(
                "dynamic_shared_memory_needed expected to have type int, or CUoccupancyB2DSize, "
                f"got {type(dynamic_shared_memory_needed)}"
            )
        return MaxPotentialBlockSizeOccupancyResult(min_grid_size=min_grid_size, max_block_size=max_block_size)

    def available_dynamic_shared_memory_per_block(self, num_blocks_per_multiprocessor: int, block_size: int) -> int:
        """Dynamic shared memory available per block for given launch configuration.

        The amount of dynamic shared memory per block, in bytes, for given kernel launch configuration.

        Parameters
        ----------
            num_blocks_per_multiprocessor: int
                Number of blocks to be concurrently executing on a multiprocessor.
            block_size: int
                Block size parameter used to launch this kernel.

        Returns
        -------
        int
            Dynamic shared memory available per block for given launch configuration.
        """
        cdef size_t dynamic_smem_size
        cdef int c_num_blocks = num_blocks_per_multiprocessor
        cdef int c_block_size = block_size
        cdef cydriver.CUfunction func = <cydriver.CUfunction>as_cu(self._h_kernel)
        with nogil:
            HANDLE_RETURN(cydriver.cuOccupancyAvailableDynamicSMemPerBlock(
                &dynamic_smem_size, func, c_num_blocks, c_block_size
            ))
        return dynamic_smem_size

    def max_potential_cluster_size(self, config: LaunchConfig, stream: Stream | None = None) -> int:
        """Maximum potential cluster size.

        The maximum potential cluster size for this kernel and given launch configuration.

        Parameters
        ----------
            config: :obj:`~_launch_config.LaunchConfig`
                Kernel launch configuration. Cluster dimensions in the configuration are ignored.
            stream: :obj:`~Stream`, optional
                The stream on which this kernel is to be launched.

        Returns
        -------
        int
            The maximum cluster size that can be launched for this kernel and launch configuration.
        """
        cdef cydriver.CUlaunchConfig drv_cfg = (<LaunchConfig>config)._to_native_launch_config()
        cdef Stream s
        if stream is not None:
            s = <Stream>stream
            drv_cfg.hStream = as_cu(s._h_stream)
        cdef int cluster_size
        cdef cydriver.CUfunction func = <cydriver.CUfunction>as_cu(self._h_kernel)
        with nogil:
            HANDLE_RETURN(cydriver.cuOccupancyMaxPotentialClusterSize(&cluster_size, func, &drv_cfg))
        return cluster_size

    def max_active_clusters(self, config: LaunchConfig, stream: Stream | None = None) -> int:
        """Maximum number of active clusters on the target device.

        The maximum number of clusters that could concurrently execute on the target device.

        Parameters
        ----------
            config: :obj:`~_launch_config.LaunchConfig`
                Kernel launch configuration.
            stream: :obj:`~Stream`, optional
                The stream on which this kernel is to be launched.

        Returns
        -------
        int
            The maximum number of clusters that could co-exist on the target device.
        """
        cdef cydriver.CUlaunchConfig drv_cfg = (<LaunchConfig>config)._to_native_launch_config()
        cdef Stream s
        if stream is not None:
            s = <Stream>stream
            drv_cfg.hStream = as_cu(s._h_stream)
        cdef int num_clusters
        cdef cydriver.CUfunction func = <cydriver.CUfunction>as_cu(self._h_kernel)
        with nogil:
            HANDLE_RETURN(cydriver.cuOccupancyMaxActiveClusters(&num_clusters, func, &drv_cfg))
        return num_clusters


ParamInfo = namedtuple("ParamInfo", ["offset", "size"])


cdef class Kernel:
    """Represent a compiled kernel that had been loaded onto the device.

    Kernel instances can execution when passed directly into the
    :func:`~launch` function.

    Directly creating a :obj:`~_module.Kernel` is not supported, and they
    should instead be created through a :obj:`~_module.ObjectCode` object.

    """

    def __init__(self, *args, **kwargs):
        raise RuntimeError("Kernel objects cannot be instantiated directly. Please use ObjectCode APIs.")

    @staticmethod
    cdef Kernel _from_obj(KernelHandle h_kernel):
        cdef Kernel ker = Kernel.__new__(Kernel)
        ker._h_kernel = h_kernel
        ker._attributes = None
        ker._occupancy = None
        return ker

    @property
    def attributes(self) -> KernelAttributes:
        """Get the read-only attributes of this kernel."""
        if self._attributes is None:
            self._attributes = KernelAttributes._init(self._h_kernel)
        return self._attributes

    cdef tuple _get_arguments_info(self, bint param_info=False):
        if not _is_paraminfo_supported():
            driver_ver = _get_driver_ver()
            raise NotImplementedError(
                "Driver version 12.4 or newer is required for this function. "
                f"Using driver version {driver_ver // 1000}.{(driver_ver % 1000) // 10}"
            )
        cdef size_t arg_pos = 0
        cdef list param_info_data = []
        cdef cydriver.CUkernel cu_kernel = as_cu(self._h_kernel)
        cdef size_t param_offset, param_size
        cdef cydriver.CUresult err
        while True:
            with nogil:
                err = cydriver.cuKernelGetParamInfo(cu_kernel, arg_pos, &param_offset, &param_size)
            if err != cydriver.CUDA_SUCCESS:
                break
            if param_info:
                param_info_data.append(ParamInfo(offset=param_offset, size=param_size))
            arg_pos = arg_pos + 1
        if err != cydriver.CUDA_ERROR_INVALID_VALUE:
            HANDLE_RETURN(err)
        return arg_pos, param_info_data

    @property
    def num_arguments(self) -> int:
        """int : The number of arguments of this function"""
        num_args, _ = self._get_arguments_info()
        return num_args

    @property
    def arguments_info(self) -> list[ParamInfo]:
        """list[ParamInfo]: (offset, size) for each argument of this function"""
        _, param_info = self._get_arguments_info(param_info=True)
        return param_info

    @property
    def occupancy(self) -> KernelOccupancy:
        """Get the occupancy information for launching this kernel."""
        if self._occupancy is None:
            self._occupancy = KernelOccupancy._init(self._h_kernel)
        return self._occupancy

    @property
    def handle(self):
        """Return the underlying kernel handle object.

        .. caution::

            This handle is a Python object. To get the memory address of the underlying C
            handle, call ``int(Kernel.handle)``.
        """
        return as_py(self._h_kernel)

    @staticmethod
    def from_handle(handle, mod: ObjectCode = None) -> Kernel:
        """Creates a new :obj:`Kernel` object from a foreign kernel handle.

        Uses a CUkernel pointer address to create a new :obj:`Kernel` object.

        Parameters
        ----------
        handle : int
            Kernel handle representing the address of a foreign
            kernel object (CUkernel).
        mod : :obj:`ObjectCode`, optional
            The ObjectCode object associated with this kernel. If not provided,
            a placeholder ObjectCode will be created. Note that without a proper
            ObjectCode, certain operations may be limited.
        """

        # Validate that handle is an integer
        if not isinstance(handle, int):
            raise TypeError(f"handle must be an integer, got {type(handle).__name__}")

        # Convert the integer handle to CUkernel
        cdef cydriver.CUkernel cu_kernel = <cydriver.CUkernel><void*><size_t>handle
        cdef KernelHandle h_kernel
        cdef cydriver.CUlibrary cu_library
        cdef cydriver.CUresult err

        # If no module provided, create a placeholder and try to get the library
        if mod is None:
            mod = ObjectCode._init(b"", "cubin")
            if _is_cukernel_get_library_supported():
                # Try to get the owning library via cuKernelGetLibrary
                with nogil:
                    err = cydriver.cuKernelGetLibrary(&cu_library, cu_kernel)
                if err == cydriver.CUDA_SUCCESS:
                    mod._h_library = create_library_handle_ref(cu_library)

        # Create kernel handle with library dependency
        h_kernel = create_kernel_handle_ref(cu_kernel, mod._h_library)
        if not h_kernel:
            HANDLE_RETURN(get_last_error())

        return Kernel._from_obj(h_kernel)

    def __eq__(self, other) -> bool:
        if not isinstance(other, Kernel):
            return NotImplemented
        return as_intptr(self._h_kernel) == as_intptr((<Kernel>other)._h_kernel)

    def __hash__(self) -> int:
        return hash(as_intptr(self._h_kernel))

    def __repr__(self) -> str:
        return f"<Kernel handle={as_intptr(self._h_kernel):#x}>"


CodeTypeT = bytes | bytearray | str

cdef tuple _supported_code_type = ("cubin", "ptx", "ltoir", "fatbin", "object", "library")

cdef class ObjectCode:
    """Represent a compiled program to be loaded onto the device.

    This object provides a unified interface for different types of
    compiled programs that will be loaded onto the device.

    Note
    ----
    This class has no default constructor. If you already have a cubin that you would
    like to load, use the :meth:`from_cubin` alternative constructor. Constructing directly
    from all other possible code types should be avoided in favor of compilation through
    :class:`~cuda.core.Program`
    """

    def __init__(self, *args, **kwargs):
        raise RuntimeError(
            "ObjectCode objects cannot be instantiated directly. "
            "Please use ObjectCode APIs (from_cubin, from_ptx) or Program APIs (compile)."
        )

    @classmethod
    def _init(cls, module, code_type, *, name: str = "", symbol_mapping: dict | None = None):
        assert code_type in _supported_code_type, f"{code_type=} is not supported"
        cdef ObjectCode self = ObjectCode.__new__(ObjectCode)

        # _h_library is assigned during _lazy_load_module
        self._h_library = LibraryHandle()  # Empty handle
        _lazy_init()

        self._code_type = code_type
        self._module = module
        self._sym_map = {} if symbol_mapping is None else symbol_mapping
        self._name = name if name else ""

        return self

    @classmethod
    def _reduce_helper(cls, module, code_type, name, symbol_mapping):
        # just for forwarding kwargs
        return cls._init(module, code_type, name=name if name else "", symbol_mapping=symbol_mapping)

    def __reduce__(self):
        return ObjectCode._reduce_helper, (self._module, self._code_type, self._name, self._sym_map)

    @staticmethod
    def from_cubin(module: bytes | str, *, name: str = "", symbol_mapping: dict | None = None) -> ObjectCode:
        """Create an :class:`ObjectCode` instance from an existing cubin.

        Parameters
        ----------
        module : Union[bytes, str]
            Either a bytes object containing the in-memory cubin to load, or
            a file path string pointing to the on-disk cubin to load.
        name : Optional[str]
            A human-readable identifier representing this code object.
        symbol_mapping : Optional[dict]
            A dictionary specifying how the unmangled symbol names (as keys)
            should be mapped to the mangled names before trying to retrieve
            them (default to no mappings).
        """
        return ObjectCode._init(module, "cubin", name=name, symbol_mapping=symbol_mapping)

    @staticmethod
    def from_ptx(module: bytes | str, *, name: str = "", symbol_mapping: dict | None = None) -> ObjectCode:
        """Create an :class:`ObjectCode` instance from an existing PTX.

        Parameters
        ----------
        module : Union[bytes, str]
            Either a bytes object containing the in-memory ptx code to load, or
            a file path string pointing to the on-disk ptx file to load.
        name : Optional[str]
            A human-readable identifier representing this code object.
        symbol_mapping : Optional[dict]
            A dictionary specifying how the unmangled symbol names (as keys)
            should be mapped to the mangled names before trying to retrieve
            them (default to no mappings).
        """
        return ObjectCode._init(module, "ptx", name=name, symbol_mapping=symbol_mapping)

    @staticmethod
    def from_ltoir(module: bytes | str, *, name: str = "", symbol_mapping: dict | None = None) -> ObjectCode:
        """Create an :class:`ObjectCode` instance from an existing LTOIR.

        Parameters
        ----------
        module : Union[bytes, str]
            Either a bytes object containing the in-memory ltoir code to load, or
            a file path string pointing to the on-disk ltoir file to load.
        name : Optional[str]
            A human-readable identifier representing this code object.
        symbol_mapping : Optional[dict]
            A dictionary specifying how the unmangled symbol names (as keys)
            should be mapped to the mangled names before trying to retrieve
            them (default to no mappings).
        """
        return ObjectCode._init(module, "ltoir", name=name, symbol_mapping=symbol_mapping)

    @staticmethod
    def from_fatbin(module: bytes | str, *, name: str = "", symbol_mapping: dict | None = None) -> ObjectCode:
        """Create an :class:`ObjectCode` instance from an existing fatbin.

        Parameters
        ----------
        module : Union[bytes, str]
            Either a bytes object containing the in-memory fatbin to load, or
            a file path string pointing to the on-disk fatbin to load.
        name : Optional[str]
            A human-readable identifier representing this code object.
        symbol_mapping : Optional[dict]
            A dictionary specifying how the unmangled symbol names (as keys)
            should be mapped to the mangled names before trying to retrieve
            them (default to no mappings).
        """
        return ObjectCode._init(module, "fatbin", name=name, symbol_mapping=symbol_mapping)

    @staticmethod
    def from_object(module: bytes | str, *, name: str = "", symbol_mapping: dict | None = None) -> ObjectCode:
        """Create an :class:`ObjectCode` instance from an existing object code.

        Parameters
        ----------
        module : Union[bytes, str]
            Either a bytes object containing the in-memory object code to load, or
            a file path string pointing to the on-disk object code to load.
        name : Optional[str]
            A human-readable identifier representing this code object.
        symbol_mapping : Optional[dict]
            A dictionary specifying how the unmangled symbol names (as keys)
            should be mapped to the mangled names before trying to retrieve
            them (default to no mappings).
        """
        return ObjectCode._init(module, "object", name=name, symbol_mapping=symbol_mapping)

    @staticmethod
    def from_library(module: bytes | str, *, name: str = "", symbol_mapping: dict | None = None) -> ObjectCode:
        """Create an :class:`ObjectCode` instance from an existing library.

        Parameters
        ----------
        module : Union[bytes, str]
            Either a bytes object containing the in-memory library to load, or
            a file path string pointing to the on-disk library to load.
        name : Optional[str]
            A human-readable identifier representing this code object.
        symbol_mapping : Optional[dict]
            A dictionary specifying how the unmangled symbol names (as keys)
            should be mapped to the mangled names before trying to retrieve
            them (default to no mappings).
        """
        return ObjectCode._init(module, "library", name=name, symbol_mapping=symbol_mapping)

    # TODO: do we want to unload in a finalizer? Probably not..

    cdef int _lazy_load_module(self) except -1:
        if self._h_library:
            return 0
        module = self._module
        assert_type_str_or_bytes_like(module)
        cdef bytes path_bytes
        if isinstance(module, str):
            path_bytes = module.encode()
            self._h_library = create_library_handle_from_file(<const char*>path_bytes)
            if not self._h_library:
                HANDLE_RETURN(get_last_error())
            return 0
        if isinstance(module, (bytes, bytearray)):
            self._h_library = create_library_handle_from_data(<const void*><char*>module)
            if not self._h_library:
                HANDLE_RETURN(get_last_error())
            return 0
        raise_code_path_meant_to_be_unreachable()
        return -1

    def get_kernel(self, name) -> Kernel:
        """Return the :obj:`~_module.Kernel` of a specified name from this object code.

        Parameters
        ----------
        name : Any
            Name of the kernel to retrieve.

        Returns
        -------
        :obj:`~_module.Kernel`
            Newly created kernel object.

        """
        self._lazy_load_module()
        supported_code_types = ("cubin", "ptx", "fatbin")
        if self._code_type not in supported_code_types:
            raise RuntimeError(f'Unsupported code type "{self._code_type}" ({supported_code_types=})')
        try:
            name = self._sym_map[name]
        except KeyError:
            name = name.encode()

        cdef KernelHandle h_kernel = create_kernel_handle(self._h_library, <const char*>name)
        if not h_kernel:
            HANDLE_RETURN(get_last_error())
        return Kernel._from_obj(h_kernel)

    @property
    def code(self) -> CodeTypeT:
        """Return the underlying code object."""
        return self._module

    @property
    def name(self) -> str:
        """Return a human-readable name of this code object."""
        return self._name

    @property
    def code_type(self) -> str:
        """Return the type of the underlying code object."""
        return self._code_type

    @property
    def symbol_mapping(self) -> dict:
        """Return a copy of the symbol mapping dictionary."""
        return dict(self._sym_map)

    @property
    def handle(self):
        """Return the underlying handle object.

        .. caution::

            This handle is a Python object. To get the memory address of the underlying C
            handle, call ``int(ObjectCode.handle)``.
        """
        self._lazy_load_module()
        return as_py(self._h_library)

    def __eq__(self, other) -> bool:
        if not isinstance(other, ObjectCode):
            return NotImplemented
        # Trigger lazy load for both objects to compare handles
        self._lazy_load_module()
        (<ObjectCode>other)._lazy_load_module()
        return as_intptr(self._h_library) == as_intptr((<ObjectCode>other)._h_library)

    def __hash__(self) -> int:
        # Trigger lazy load to get the handle
        self._lazy_load_module()
        return hash(as_intptr(self._h_library))

    def __repr__(self) -> str:
        # Trigger lazy load to get the handle
        self._lazy_load_module()
        return f"<ObjectCode handle={as_intptr(self._h_library):#x} code_type='{self._code_type}'>"
