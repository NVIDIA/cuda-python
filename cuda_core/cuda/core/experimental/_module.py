# SPDX-FileCopyrightText: Copyright (c) 2024-2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

import weakref
from collections import namedtuple
from typing import Optional, Union
from warnings import warn

from cuda.core.experimental._launch_config import LaunchConfig, _to_native_launch_config
from cuda.core.experimental._stream import Stream
from cuda.core.experimental._utils.clear_error_support import (
    assert_type,
    assert_type_str_or_bytes_like,
    raise_code_path_meant_to_be_unreachable,
)
from cuda.core.experimental._utils.cuda_utils import driver, get_binding_version, handle_return, precondition

_backend = {
    "old": {
        "file": driver.cuModuleLoad,
        "data": driver.cuModuleLoadDataEx,
        "kernel": driver.cuModuleGetFunction,
        "attribute": driver.cuFuncGetAttribute,
    },
}


# TODO: revisit this treatment for py313t builds
_inited = False
_py_major_ver = None
_driver_ver = None
_kernel_ctypes = None


def _lazy_init():
    global _inited
    if _inited:
        return

    global _py_major_ver, _driver_ver, _kernel_ctypes
    # binding availability depends on cuda-python version
    _py_major_ver, _ = get_binding_version()
    if _py_major_ver >= 12:
        _backend["new"] = {
            "file": driver.cuLibraryLoadFromFile,
            "data": driver.cuLibraryLoadData,
            "kernel": driver.cuLibraryGetKernel,
            "attribute": driver.cuKernelGetAttribute,
        }
        _kernel_ctypes = (driver.CUfunction, driver.CUkernel)
    else:
        _kernel_ctypes = (driver.CUfunction,)
    _driver_ver = handle_return(driver.cuDriverGetVersion())
    if _py_major_ver >= 12 and _driver_ver >= 12040:
        _backend["new"]["paraminfo"] = driver.cuKernelGetParamInfo
    _inited = True


class KernelAttributes:
    def __new__(self, *args, **kwargs):
        raise RuntimeError("KernelAttributes cannot be instantiated directly. Please use Kernel APIs.")

    slots = ("_kernel", "_cache", "_backend_version", "_loader")

    @classmethod
    def _init(cls, kernel):
        self = super().__new__(cls)
        self._kernel = weakref.ref(kernel)
        self._cache = {}

        self._backend_version = "new" if (_py_major_ver >= 12 and _driver_ver >= 12000) else "old"
        self._loader = _backend[self._backend_version]
        return self

    def _get_cached_attribute(self, device_id: int, attribute: driver.CUfunction_attribute) -> int:
        """Helper function to get a cached attribute or fetch and cache it if not present."""
        cache_key = device_id, attribute
        result = self._cache.get(cache_key, cache_key)
        if result is not cache_key:
            return result
        kernel = self._kernel()
        if kernel is None:
            raise RuntimeError("Cannot access kernel attributes for expired Kernel object")
        if self._backend_version == "new":
            result = handle_return(self._loader["attribute"](attribute, kernel._handle, device_id))
        else:  # "old" backend
            warn(
                "Device ID argument is ignored when getting attribute from kernel when cuda version < 12. ",
                RuntimeWarning,
                stacklevel=2,
            )
            result = handle_return(self._loader["attribute"](attribute, kernel._handle))
        self._cache[cache_key] = result
        return result

    def max_threads_per_block(self, device_id: int = None) -> int:
        """int : The maximum number of threads per block.
        This attribute is read-only."""
        return self._get_cached_attribute(
            device_id, driver.CUfunction_attribute.CU_FUNC_ATTRIBUTE_MAX_THREADS_PER_BLOCK
        )

    def shared_size_bytes(self, device_id: int = None) -> int:
        """int : The size in bytes of statically-allocated shared memory required by this function.
        This attribute is read-only."""
        return self._get_cached_attribute(device_id, driver.CUfunction_attribute.CU_FUNC_ATTRIBUTE_SHARED_SIZE_BYTES)

    def const_size_bytes(self, device_id: int = None) -> int:
        """int : The size in bytes of user-allocated constant memory required by this function.
        This attribute is read-only."""
        return self._get_cached_attribute(device_id, driver.CUfunction_attribute.CU_FUNC_ATTRIBUTE_CONST_SIZE_BYTES)

    def local_size_bytes(self, device_id: int = None) -> int:
        """int : The size in bytes of local memory used by each thread of this function.
        This attribute is read-only."""
        return self._get_cached_attribute(device_id, driver.CUfunction_attribute.CU_FUNC_ATTRIBUTE_LOCAL_SIZE_BYTES)

    def num_regs(self, device_id: int = None) -> int:
        """int : The number of registers used by each thread of this function.
        This attribute is read-only."""
        return self._get_cached_attribute(device_id, driver.CUfunction_attribute.CU_FUNC_ATTRIBUTE_NUM_REGS)

    def ptx_version(self, device_id: int = None) -> int:
        """int : The PTX virtual architecture version for which the function was compiled.
        This attribute is read-only."""
        return self._get_cached_attribute(device_id, driver.CUfunction_attribute.CU_FUNC_ATTRIBUTE_PTX_VERSION)

    def binary_version(self, device_id: int = None) -> int:
        """int : The binary architecture version for which the function was compiled.
        This attribute is read-only."""
        return self._get_cached_attribute(device_id, driver.CUfunction_attribute.CU_FUNC_ATTRIBUTE_BINARY_VERSION)

    def cache_mode_ca(self, device_id: int = None) -> bool:
        """bool : Whether the function has been compiled with user specified option "-Xptxas --dlcm=ca" set.
        This attribute is read-only."""
        return bool(self._get_cached_attribute(device_id, driver.CUfunction_attribute.CU_FUNC_ATTRIBUTE_CACHE_MODE_CA))

    def max_dynamic_shared_size_bytes(self, device_id: int = None) -> int:
        """int : The maximum size in bytes of dynamically-allocated shared memory that can be used
        by this function."""
        return self._get_cached_attribute(
            device_id, driver.CUfunction_attribute.CU_FUNC_ATTRIBUTE_MAX_DYNAMIC_SHARED_SIZE_BYTES
        )

    def preferred_shared_memory_carveout(self, device_id: int = None) -> int:
        """int : The shared memory carveout preference, in percent of the total shared memory."""
        return self._get_cached_attribute(
            device_id, driver.CUfunction_attribute.CU_FUNC_ATTRIBUTE_PREFERRED_SHARED_MEMORY_CARVEOUT
        )

    def cluster_size_must_be_set(self, device_id: int = None) -> bool:
        """bool : The kernel must launch with a valid cluster size specified.
        This attribute is read-only."""
        return bool(
            self._get_cached_attribute(
                device_id, driver.CUfunction_attribute.CU_FUNC_ATTRIBUTE_CLUSTER_SIZE_MUST_BE_SET
            )
        )

    def required_cluster_width(self, device_id: int = None) -> int:
        """int : The required cluster width in blocks."""
        return self._get_cached_attribute(
            device_id, driver.CUfunction_attribute.CU_FUNC_ATTRIBUTE_REQUIRED_CLUSTER_WIDTH
        )

    def required_cluster_height(self, device_id: int = None) -> int:
        """int : The required cluster height in blocks."""
        return self._get_cached_attribute(
            device_id, driver.CUfunction_attribute.CU_FUNC_ATTRIBUTE_REQUIRED_CLUSTER_HEIGHT
        )

    def required_cluster_depth(self, device_id: int = None) -> int:
        """int : The required cluster depth in blocks."""
        return self._get_cached_attribute(
            device_id, driver.CUfunction_attribute.CU_FUNC_ATTRIBUTE_REQUIRED_CLUSTER_DEPTH
        )

    def non_portable_cluster_size_allowed(self, device_id: int = None) -> bool:
        """bool : Whether the function can be launched with non-portable cluster size."""
        return bool(
            self._get_cached_attribute(
                device_id, driver.CUfunction_attribute.CU_FUNC_ATTRIBUTE_NON_PORTABLE_CLUSTER_SIZE_ALLOWED
            )
        )

    def cluster_scheduling_policy_preference(self, device_id: int = None) -> int:
        """int : The block scheduling policy of a function."""
        return self._get_cached_attribute(
            device_id, driver.CUfunction_attribute.CU_FUNC_ATTRIBUTE_CLUSTER_SCHEDULING_POLICY_PREFERENCE
        )


MaxPotentialBlockSizeOccupancyResult = namedtuple("MaxPotential", ("min_grid_size", "max_block_size"))


class KernelOccupancy:
    """ """

    def __new__(self, *args, **kwargs):
        raise RuntimeError("KernelOccupancy cannot be instantiated directly. Please use Kernel APIs.")

    slots = ("_handle",)

    @classmethod
    def _init(cls, handle):
        self = super().__new__(cls)
        self._handle = handle

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
        return handle_return(
            driver.cuOccupancyMaxActiveBlocksPerMultiprocessor(self._handle, block_size, dynamic_shared_memory_size)
        )

    def max_potential_block_size(
        self, dynamic_shared_memory_needed: Union[int, driver.CUoccupancyB2DSize], block_size_limit: int
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
        if isinstance(dynamic_shared_memory_needed, int):
            min_grid_size, max_block_size = handle_return(
                driver.cuOccupancyMaxPotentialBlockSize(
                    self._handle, None, dynamic_shared_memory_needed, block_size_limit
                )
            )
        elif isinstance(dynamic_shared_memory_needed, driver.CUoccupancyB2DSize):
            min_grid_size, max_block_size = handle_return(
                driver.cuOccupancyMaxPotentialBlockSize(
                    self._handle, dynamic_shared_memory_needed.getPtr(), 0, block_size_limit
                )
            )
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
        return handle_return(
            driver.cuOccupancyAvailableDynamicSMemPerBlock(self._handle, num_blocks_per_multiprocessor, block_size)
        )

    def max_potential_cluster_size(self, config: LaunchConfig, stream: Optional[Stream] = None) -> int:
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
        drv_cfg = _to_native_launch_config(config)
        if stream is not None:
            drv_cfg.hStream = stream.handle
        return handle_return(driver.cuOccupancyMaxPotentialClusterSize(self._handle, drv_cfg))

    def max_active_clusters(self, config: LaunchConfig, stream: Optional[Stream] = None) -> int:
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
        drv_cfg = _to_native_launch_config(config)
        if stream is not None:
            drv_cfg.hStream = stream.handle
        return handle_return(driver.cuOccupancyMaxActiveClusters(self._handle, drv_cfg))


ParamInfo = namedtuple("ParamInfo", ["offset", "size"])


class Kernel:
    """Represent a compiled kernel that had been loaded onto the device.

    Kernel instances can execution when passed directly into the
    :func:`~launch` function.

    Directly creating a :obj:`~_module.Kernel` is not supported, and they
    should instead be created through a :obj:`~_module.ObjectCode` object.

    """

    __slots__ = ("_handle", "_module", "_attributes", "_occupancy", "__weakref__")

    def __new__(self, *args, **kwargs):
        raise RuntimeError("Kernel objects cannot be instantiated directly. Please use ObjectCode APIs.")

    @classmethod
    def _from_obj(cls, obj, mod):
        assert_type(obj, _kernel_ctypes)
        assert_type(mod, ObjectCode)
        ker = super().__new__(cls)
        ker._handle = obj
        ker._module = mod
        ker._attributes = None
        ker._occupancy = None
        return ker

    @property
    def attributes(self) -> KernelAttributes:
        """Get the read-only attributes of this kernel."""
        if self._attributes is None:
            self._attributes = KernelAttributes._init(self)
        return self._attributes

    def _get_arguments_info(self, param_info=False) -> tuple[int, list[ParamInfo]]:
        attr_impl = self.attributes
        if attr_impl._backend_version != "new":
            raise NotImplementedError("New backend is required")
        if "paraminfo" not in attr_impl._loader:
            raise NotImplementedError(
                "Driver version 12.4 or newer is required for this function. "
                f"Using driver version {_driver_ver // 1000}.{(_driver_ver % 1000) // 10}"
            )
        arg_pos = 0
        param_info_data = []
        while True:
            result = attr_impl._loader["paraminfo"](self._handle, arg_pos)
            if result[0] != driver.CUresult.CUDA_SUCCESS:
                break
            if param_info:
                p_info = ParamInfo(offset=result[1], size=result[2])
                param_info_data.append(p_info)
            arg_pos = arg_pos + 1
        if result[0] != driver.CUresult.CUDA_ERROR_INVALID_VALUE:
            handle_return(result)
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
            self._occupancy = KernelOccupancy._init(self._handle)
        return self._occupancy

    # TODO: implement from_handle()


CodeTypeT = Union[bytes, bytearray, str]


class ObjectCode:
    """Represent a compiled program to be loaded onto the device.

    This object provides a unified interface for different types of
    compiled programs that will be loaded onto the device.

    Note
    ----
    This class has no default constructor. If you already have a cubin that you would
    like to load, use the :meth:`from_cubin` alternative constructor. Constructing directly
    from all other possible code types should be avoided in favor of compilation through
    :class:`~cuda.core.experimental.Program`

    Note
    ----
    Usage under CUDA 11.x will only load to the current device
    context.
    """

    __slots__ = ("_handle", "_backend_version", "_code_type", "_module", "_loader", "_sym_map", "_name")
    _supported_code_type = ("cubin", "ptx", "ltoir", "fatbin", "object", "library")

    def __new__(self, *args, **kwargs):
        raise RuntimeError(
            "ObjectCode objects cannot be instantiated directly. "
            "Please use ObjectCode APIs (from_cubin, from_ptx) or Program APIs (compile)."
        )

    @classmethod
    def _init(cls, module, code_type, *, name: str = "", symbol_mapping: Optional[dict] = None):
        self = super().__new__(cls)
        assert code_type in self._supported_code_type, f"{code_type=} is not supported"
        _lazy_init()

        # handle is assigned during _lazy_load
        self._handle = None

        self._backend_version = "new" if (_py_major_ver >= 12 and _driver_ver >= 12000) else "old"
        self._loader = _backend[self._backend_version]

        self._code_type = code_type
        self._module = module
        self._sym_map = {} if symbol_mapping is None else symbol_mapping
        self._name = name

        return self

    @classmethod
    def _reduce_helper(self, module, code_type, name, symbol_mapping):
        # just for forwarding kwargs
        return ObjectCode._init(module, code_type, name=name, symbol_mapping=symbol_mapping)

    def __reduce__(self):
        return ObjectCode._reduce_helper, (self._module, self._code_type, self._name, self._sym_map)

    @staticmethod
    def from_cubin(module: Union[bytes, str], *, name: str = "", symbol_mapping: Optional[dict] = None) -> "ObjectCode":
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
    def from_ptx(module: Union[bytes, str], *, name: str = "", symbol_mapping: Optional[dict] = None) -> "ObjectCode":
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
    def from_ltoir(module: Union[bytes, str], *, name: str = "", symbol_mapping: Optional[dict] = None) -> "ObjectCode":
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
    def from_fatbin(
        module: Union[bytes, str], *, name: str = "", symbol_mapping: Optional[dict] = None
    ) -> "ObjectCode":
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
    def from_object(
        module: Union[bytes, str], *, name: str = "", symbol_mapping: Optional[dict] = None
    ) -> "ObjectCode":
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
    def from_library(
        module: Union[bytes, str], *, name: str = "", symbol_mapping: Optional[dict] = None
    ) -> "ObjectCode":
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

    def _lazy_load_module(self, *args, **kwargs):
        if self._handle is not None:
            return
        module = self._module
        assert_type_str_or_bytes_like(module)
        if isinstance(module, str):
            if self._backend_version == "new":
                self._handle = handle_return(self._loader["file"](module.encode(), [], [], 0, [], [], 0))
            else:  # "old" backend
                self._handle = handle_return(self._loader["file"](module.encode()))
            return
        if isinstance(module, (bytes, bytearray)):
            if self._backend_version == "new":
                self._handle = handle_return(self._loader["data"](module, [], [], 0, [], [], 0))
            else:  # "old" backend
                self._handle = handle_return(self._loader["data"](module, 0, [], []))
            return
        raise_code_path_meant_to_be_unreachable()

    @precondition(_lazy_load_module)
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
        supported_code_types = ("cubin", "ptx", "fatbin")
        if self._code_type not in supported_code_types:
            raise RuntimeError(f'Unsupported code type "{self._code_type}" ({supported_code_types=})')
        try:
            name = self._sym_map[name]
        except KeyError:
            name = name.encode()

        data = handle_return(self._loader["kernel"](self._handle, name))
        return Kernel._from_obj(data, self)

    @property
    def code(self) -> CodeTypeT:
        """Return the underlying code object."""
        return self._module

    @property
    def name(self) -> str:
        """Return a human-readable name of this code object."""
        return self._name

    @property
    @precondition(_lazy_load_module)
    def handle(self):
        """Return the underlying handle object.

        .. caution::

            This handle is a Python object. To get the memory address of the underlying C
            handle, call ``int(ObjectCode.handle)``.
        """
        return self._handle
