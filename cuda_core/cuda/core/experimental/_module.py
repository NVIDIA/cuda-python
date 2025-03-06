# Copyright (c) 2024-2025, NVIDIA CORPORATION & AFFILIATES. ALL RIGHTS RESERVED.
#
# SPDX-License-Identifier: LicenseRef-NVIDIA-SOFTWARE-LICENSE

from typing import Optional, Union
from warnings import warn

from cuda.core.experimental._utils.clear_error_support import (
    assert_type,
    assert_type_str_or_bytes,
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
    _inited = True


class KernelAttributes:
    def __new__(self, *args, **kwargs):
        raise RuntimeError("KernelAttributes cannot be instantiated directly. Please use Kernel APIs.")

    slots = ("_handle", "_cache", "_backend_version", "_loader")

    @classmethod
    def _init(cls, handle):
        self = super().__new__(cls)
        self._handle = handle
        self._cache = {}

        self._backend_version = "new" if (_py_major_ver >= 12 and _driver_ver >= 12000) else "old"
        self._loader = _backend[self._backend_version]
        return self

    def _get_cached_attribute(self, device_id: int, attribute: driver.CUfunction_attribute) -> int:
        """Helper function to get a cached attribute or fetch and cache it if not present."""
        if device_id in self._cache and attribute in self._cache[device_id]:
            return self._cache[device_id][attribute]
        if self._backend_version == "new":
            result = handle_return(self._loader["attribute"](attribute, self._handle, device_id))
        else:  # "old" backend
            warn(
                "Device ID argument is ignored when getting attribute from kernel when cuda version < 12. ",
                RuntimeWarning,
                stacklevel=2,
            )
            result = handle_return(self._loader["attribute"](attribute, self._handle))
        if device_id not in self._cache:
            self._cache[device_id] = {}
        self._cache[device_id][attribute] = result
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


class Kernel:
    """Represent a compiled kernel that had been loaded onto the device.

    Kernel instances can execution when passed directly into the
    :func:`~launch` function.

    Directly creating a :obj:`~_module.Kernel` is not supported, and they
    should instead be created through a :obj:`~_module.ObjectCode` object.

    """

    __slots__ = ("_handle", "_module", "_attributes")

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
        return ker

    @property
    def attributes(self) -> KernelAttributes:
        """Get the read-only attributes of this kernel."""
        if self._attributes is None:
            self._attributes = KernelAttributes._init(self._handle)
        return self._attributes

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

    __slots__ = ("_handle", "_backend_version", "_code_type", "_module", "_loader", "_sym_map")
    _supported_code_type = ("cubin", "ptx", "ltoir", "fatbin")

    def __new__(self, *args, **kwargs):
        raise RuntimeError(
            "ObjectCode objects cannot be instantiated directly. "
            "Please use ObjectCode APIs (from_cubin, from_ptx) or Program APIs (compile)."
        )

    @classmethod
    def _init(cls, module, code_type, *, symbol_mapping: Optional[dict] = None):
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

        return self

    @staticmethod
    def from_cubin(module: Union[bytes, str], *, symbol_mapping: Optional[dict] = None) -> "ObjectCode":
        """Create an :class:`ObjectCode` instance from an existing cubin.

        Parameters
        ----------
        module : Union[bytes, str]
            Either a bytes object containing the in-memory cubin to load, or
            a file path string pointing to the on-disk cubin to load.
        symbol_mapping : Optional[dict]
            A dictionary specifying how the unmangled symbol names (as keys)
            should be mapped to the mangled names before trying to retrieve
            them (default to no mappings).
        """
        return ObjectCode._init(module, "cubin", symbol_mapping=symbol_mapping)

    @staticmethod
    def from_ptx(module: Union[bytes, str], *, symbol_mapping: Optional[dict] = None) -> "ObjectCode":
        """Create an :class:`ObjectCode` instance from an existing PTX.

        Parameters
        ----------
        module : Union[bytes, str]
            Either a bytes object containing the in-memory ptx code to load, or
            a file path string pointing to the on-disk ptx file to load.
        symbol_mapping : Optional[dict]
            A dictionary specifying how the unmangled symbol names (as keys)
            should be mapped to the mangled names before trying to retrieve
            them (default to no mappings).
        """
        return ObjectCode._init(module, "ptx", symbol_mapping=symbol_mapping)

    # TODO: do we want to unload in a finalizer? Probably not..

    def _lazy_load_module(self, *args, **kwargs):
        if self._handle is not None:
            return
        module = self._module
        assert_type_str_or_bytes(module)
        if isinstance(module, str):
            if self._backend_version == "new":
                self._handle = handle_return(self._loader["file"](module.encode(), [], [], 0, [], [], 0))
            else:  # "old" backend
                self._handle = handle_return(self._loader["file"](module.encode()))
            return
        if isinstance(module, bytes):
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
    @precondition(_lazy_load_module)
    def handle(self):
        """Return the underlying handle object."""
        return self._handle
