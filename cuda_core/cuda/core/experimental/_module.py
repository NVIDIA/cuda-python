# Copyright (c) 2024, NVIDIA CORPORATION & AFFILIATES. ALL RIGHTS RESERVED.
#
# SPDX-License-Identifier: LicenseRef-NVIDIA-SOFTWARE-LICENSE

import importlib.metadata

from cuda import cuda
from cuda.core.experimental._utils import handle_return, precondition

_backend = {
    "old": {
        "file": cuda.cuModuleLoad,
        "data": cuda.cuModuleLoadDataEx,
        "kernel": cuda.cuModuleGetFunction,
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
    _py_major_ver = int(importlib.metadata.version("cuda-python").split(".")[0])
    if _py_major_ver >= 12:
        _backend["new"] = {
            "file": cuda.cuLibraryLoadFromFile,
            "data": cuda.cuLibraryLoadData,
            "kernel": cuda.cuLibraryGetKernel,
        }
        _kernel_ctypes = (cuda.CUfunction, cuda.CUkernel)
    else:
        _kernel_ctypes = (cuda.CUfunction,)
    _driver_ver = handle_return(cuda.cuDriverGetVersion())
    _inited = True


class Kernel:
    """Represent a compiled kernel that had been loaded onto the device.

    Kernel instances can execution when passed directly into the
    :func:`~launch` function.

    Directly creating a :obj:`~_module.Kernel` is not supported, and they
    should instead be created through a :obj:`~_module.ObjectCode` object.

    """

    __slots__ = ("_handle", "_module")

    def __init__(self):
        raise RuntimeError("directly constructing a Kernel instance is not supported")

    @staticmethod
    def _from_obj(obj, mod):
        assert isinstance(obj, _kernel_ctypes)
        assert isinstance(mod, ObjectCode)
        ker = Kernel.__new__(Kernel)
        ker._handle = obj
        ker._module = mod
        return ker

    # Kernel attribute getters and setters
    @property
    def max_threads_per_block(self):
        """Get the maximum number of threads per block."""
        with self._exception_manager():
            return handle_return(
                cuda.cuKernelGetAttribute(
                    cuda.CUfunction_attribute.CU_FUNC_ATTRIBUTE_MAX_THREADS_PER_BLOCK, self._handle, None
                )
            )

    @max_threads_per_block.setter
    def max_threads_per_block(self, value: int):
        raise AttributeError("max_threads_per_block is read only")

    @property
    def shared_size_bytes(self):
        """Get the size in bytes of statically-allocated shared memory required by this function."""
        with self._exception_manager():
            return handle_return(
                cuda.cuKernelGetAttribute(
                    cuda.CUfunction_attribute.CU_FUNC_ATTRIBUTE_SHARED_SIZE_BYTES, self._handle, None
                )
            )

    @shared_size_bytes.setter
    def shared_size_bytes(self, value: int):
        raise AttributeError("shared_size_bytes is read only")

    @property
    def const_size_bytes(self):
        """Get the size in bytes of user-allocated constant memory required by this function."""
        with self._exception_manager():
            return handle_return(
                cuda.cuKernelGetAttribute(
                    cuda.CUfunction_attribute.CU_FUNC_ATTRIBUTE_CONST_SIZE_BYTES, self._handle, None
                )
            )

    @const_size_bytes.setter
    def const_size_bytes(self, value: int):
        raise AttributeError("const_size_bytes is read only")

    @property
    def local_size_bytes(self):
        """Get the size in bytes of local memory used by each thread of this function."""
        with self._exception_manager():
            return handle_return(
                cuda.cuKernelGetAttribute(
                    cuda.CUfunction_attribute.CU_FUNC_ATTRIBUTE_LOCAL_SIZE_BYTES, self._handle, None
                )
            )

    @local_size_bytes.setter
    def local_size_bytes(self, value: int):
        raise AttributeError("local_size_bytes is read only")

    @property
    def num_regs(self):
        """Get the number of registers used by each thread of this function."""
        with self._exception_manager():
            return handle_return(
                cuda.cuKernelGetAttribute(cuda.CUfunction_attribute.CU_FUNC_ATTRIBUTE_NUM_REGS, self._handle, None)
            )

    @num_regs.setter
    def num_regs(self, value: int):
        raise AttributeError("num_regs is read only")

    @property
    def ptx_version(self):
        """Get the PTX virtual architecture version for which the function was compiled."""
        with self._exception_manager():
            return handle_return(
                cuda.cuKernelGetAttribute(cuda.CUfunction_attribute.CU_FUNC_ATTRIBUTE_PTX_VERSION, self._handle, None)
            )

    @ptx_version.setter
    def ptx_version(self, value: int):
        raise AttributeError("ptx_version is read only")

    @property
    def binary_version(self):
        """Get the binary architecture version for which the function was compiled."""
        with self._exception_manager():
            return handle_return(
                cuda.cuKernelGetAttribute(
                    cuda.CUfunction_attribute.CU_FUNC_ATTRIBUTE_BINARY_VERSION, self._handle, None
                )
            )

    @binary_version.setter
    def binary_version(self, value: int):
        raise AttributeError("binary_version is read only")

    @property
    def cache_mode_ca(self):
        """Get whether the function has been compiled with user specified option "-Xptxas --dlcm=ca" set."""
        with self._exception_manager():
            return handle_return(
                cuda.cuKernelGetAttribute(cuda.CUfunction_attribute.CU_FUNC_ATTRIBUTE_CACHE_MODE_CA, self._handle, None)
            )

    @cache_mode_ca.setter
    def cache_mode_ca(self, value: bool):
        raise AttributeError("cache_mode_ca is read only")

    @property
    def max_dynamic_shared_size_bytes(self):
        """Get the maximum size in bytes of dynamically-allocated shared memory that can be used by this function."""
        with self._exception_manager():
            return handle_return(
                cuda.cuKernelGetAttribute(
                    cuda.CUfunction_attribute.CU_FUNC_ATTRIBUTE_MAX_DYNAMIC_SHARED_SIZE_BYTES, self._handle, None
                )
            )

    @max_dynamic_shared_size_bytes.setter
    def max_dynamic_shared_size_bytes(self, value: int):
        """Set the maximum size in bytes of dynamically-allocated shared memory that can be used by this function."""
        with self._exception_manager():
            handle_return(
                cuda.cuKernelSetAttribute(
                    cuda.CUfunction_attribute.CU_FUNC_ATTRIBUTE_MAX_DYNAMIC_SHARED_SIZE_BYTES, value, self._handle, None
                )
            )

    @property
    def preferred_shared_memory_carveout(self):
        """Get the shared memory carveout preference, in percent of the total shared memory."""
        with self._exception_manager():
            return handle_return(
                cuda.cuKernelGetAttribute(
                    cuda.CUfunction_attribute.CU_FUNC_ATTRIBUTE_PREFERRED_SHARED_MEMORY_CARVEOUT, self._handle, None
                )
            )

    @preferred_shared_memory_carveout.setter
    def preferred_shared_memory_carveout(self, value: int):
        """Set the shared memory carveout preference, in percent of the total shared memory."""
        with self._exception_manager():
            handle_return(
                cuda.cuKernelSetAttribute(
                    cuda.CUfunction_attribute.CU_FUNC_ATTRIBUTE_PREFERRED_SHARED_MEMORY_CARVEOUT,
                    value,
                    self._handle,
                    None,
                )
            )

    @property
    def cluster_size_must_be_set(self):
        """Get whether the kernel must launch with a valid cluster size specified."""
        with self._exception_manager():
            return handle_return(
                cuda.cuKernelGetAttribute(
                    cuda.CUfunction_attribute.CU_FUNC_ATTRIBUTE_CLUSTER_SIZE_MUST_BE_SET, self._handle, None
                )
            )

    @cluster_size_must_be_set.setter
    def cluster_size_must_be_set(self, value: bool):
        raise AttributeError("cluster_size_must_be_set is read only")

    @property
    def required_cluster_width(self):
        """Get the required cluster width in blocks."""
        with self._exception_manager():
            return handle_return(
                cuda.cuKernelGetAttribute(
                    cuda.CUfunction_attribute.CU_FUNC_ATTRIBUTE_REQUIRED_CLUSTER_WIDTH, self._handle, None
                )
            )

    @required_cluster_width.setter
    def required_cluster_width(self, value: int):
        """Set the required cluster width in blocks."""
        with self._exception_manager():
            handle_return(
                cuda.cuKernelSetAttribute(
                    cuda.CUfunction_attribute.CU_FUNC_ATTRIBUTE_REQUIRED_CLUSTER_WIDTH, value, self._handle, None
                )
            )

    @property
    def required_cluster_height(self):
        """Get the required cluster height in blocks."""
        with self._exception_manager():
            return handle_return(
                cuda.cuKernelGetAttribute(
                    cuda.CUfunction_attribute.CU_FUNC_ATTRIBUTE_REQUIRED_CLUSTER_HEIGHT, self._handle, None
                )
            )

    @required_cluster_height.setter
    def required_cluster_height(self, value: int):
        """Set the required cluster height in blocks."""
        with self._exception_manager():
            handle_return(
                cuda.cuKernelSetAttribute(
                    cuda.CUfunction_attribute.CU_FUNC_ATTRIBUTE_REQUIRED_CLUSTER_HEIGHT, value, self._handle, None
                )
            )

    @property
    def required_cluster_depth(self):
        """Get the required cluster depth in blocks."""
        with self._exception_manager():
            return handle_return(
                cuda.cuKernelGetAttribute(
                    cuda.CUfunction_attribute.CU_FUNC_ATTRIBUTE_REQUIRED_CLUSTER_DEPTH, self._handle, None
                )
            )

    @required_cluster_depth.setter
    def required_cluster_depth(self, value: int):
        """Set the required cluster depth in blocks."""
        with self._exception_manager():
            handle_return(
                cuda.cuKernelSetAttribute(
                    cuda.CUfunction_attribute.CU_FUNC_ATTRIBUTE_REQUIRED_CLUSTER_DEPTH, value, self._handle, None
                )
            )

    @property
    def non_portable_cluster_size_allowed(self):
        """Get whether the function can be launched with non-portable cluster size."""
        with self._exception_manager():
            return handle_return(
                cuda.cuKernelGetAttribute(
                    cuda.CUfunction_attribute.CU_FUNC_ATTRIBUTE_NON_PORTABLE_CLUSTER_SIZE_ALLOWED, self._handle, None
                )
            )

    @non_portable_cluster_size_allowed.setter
    def non_portable_cluster_size_allowed(self, value: bool):
        """Set whether the function can be launched with non-portable cluster size."""
        with self._exception_manager():
            handle_return(
                cuda.cuKernelSetAttribute(
                    cuda.CUfunction_attribute.CU_FUNC_ATTRIBUTE_NON_PORTABLE_CLUSTER_SIZE_ALLOWED,
                    value,
                    self._handle,
                    None,
                )
            )

    @property
    def cluster_scheduling_policy_preference(self):
        """Get the block scheduling policy of a function."""
        with self._exception_manager():
            return handle_return(
                cuda.cuKernelGetAttribute(
                    cuda.CUfunction_attribute.CU_FUNC_ATTRIBUTE_CLUSTER_SCHEDULING_POLICY_PREFERENCE, self._handle, None
                )
            )

    @cluster_scheduling_policy_preference.setter
    def cluster_scheduling_policy_preference(self, value: int):
        """Set the block scheduling policy of a function."""
        with self._exception_manager():
            handle_return(
                cuda.cuKernelSetAttribute(
                    cuda.CUfunction_attribute.CU_FUNC_ATTRIBUTE_CLUSTER_SCHEDULING_POLICY_PREFERENCE,
                    value,
                    self._handle,
                    None,
                )
            )


class ObjectCode:
    """Represent a compiled program that was loaded onto the device.

    This object provides a unified interface for different types of
    compiled programs that are loaded onto the device.

    Loads the module library with specified module code and JIT options.

    Note
    ----
    Usage under CUDA 11.x will only load to the current device
    context.

    Parameters
    ----------
    module : Union[bytes, str]
        Either a bytes object containing the module to load, or
        a file path string containing that module for loading.
    code_type : Any
        String of the compiled type.
        Supported options are "ptx", "cubin", "ltoir" and "fatbin".
    jit_options : Optional
        Mapping of JIT options to use during module loading.
        (Default to no options)
    symbol_mapping : Optional
        Keyword argument dictionary specifying how symbol names
        should be mapped before trying to retrieve them.
        (Default to no mappings)

    """

    __slots__ = ("_handle", "_backend_version", "_jit_options", "_code_type", "_module", "_loader", "_sym_map")
    _supported_code_type = ("cubin", "ptx", "ltoir", "fatbin")

    def __init__(self, module, code_type, jit_options=None, *, symbol_mapping=None):
        if code_type not in self._supported_code_type:
            raise ValueError
        _lazy_init()

        # handle is assigned during _lazy_load
        self._handle = None
        self._jit_options = jit_options

        self._backend_version = "new" if (_py_major_ver >= 12 and _driver_ver >= 12000) else "old"
        self._loader = _backend[self._backend_version]

        self._code_type = code_type
        self._module = module
        self._sym_map = {} if symbol_mapping is None else symbol_mapping

    # TODO: do we want to unload in a finalizer? Probably not..

    def _lazy_load_module(self, *args, **kwargs):
        if self._handle is not None:
            return
        jit_options = self._jit_options
        module = self._module
        if isinstance(module, str):
            # TODO: this option is only taken by the new library APIs, but we have
            # a bug that we can't easily support it just yet (NVIDIA/cuda-python#73).
            if jit_options is not None:
                raise ValueError
            self._handle = handle_return(self._loader["file"](module))
        else:
            assert isinstance(module, bytes)
            if jit_options is None:
                jit_options = {}
            if self._backend_version == "new":
                args = (
                    module,
                    list(jit_options.keys()),
                    list(jit_options.values()),
                    len(jit_options),
                    # TODO: support library options
                    [],
                    [],
                    0,
                )
            else:  # "old" backend
                args = (
                    module,
                    len(jit_options),
                    list(jit_options.keys()),
                    list(jit_options.values()),
                )
            self._handle = handle_return(self._loader["data"](*args))

    @precondition(_lazy_load_module)
    def get_kernel(self, name):
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
        try:
            name = self._sym_map[name]
        except KeyError:
            name = name.encode()

        data = handle_return(self._loader["kernel"](self._handle, name))
        return Kernel._from_obj(data, self)

    # TODO: implement from_handle()
