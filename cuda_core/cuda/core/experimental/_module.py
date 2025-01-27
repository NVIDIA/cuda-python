# Copyright (c) 2024, NVIDIA CORPORATION & AFFILIATES. ALL RIGHTS RESERVED.
#
# SPDX-License-Identifier: LicenseRef-NVIDIA-SOFTWARE-LICENSE


from cuda.core.experimental._utils import driver, get_binding_version, handle_return, precondition

_backend = {
    "old": {
        "file": driver.cuModuleLoad,
        "data": driver.cuModuleLoadDataEx,
        "kernel": driver.cuModuleGetFunction,
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
        }
        _kernel_ctypes = (driver.CUfunction, driver.CUkernel)
    else:
        _kernel_ctypes = (driver.CUfunction,)
    _driver_ver = handle_return(driver.cuDriverGetVersion())
    _inited = True


class KernelAttributes:
    def __init__(self):
        raise RuntimeError("KernelAttributes should not be instantiated directly")

    slots = "_handle"

    def _init(handle):
        self = KernelAttributes.__new__(KernelAttributes)
        self._handle = handle
        return self

    @property
    def max_threads_per_block(self) -> int:
        """int : The maximum number of threads per block.
        This attribute is read-only."""
        return handle_return(
            driver.cuKernelGetAttribute(
                driver.CUfunction_attribute.CU_FUNC_ATTRIBUTE_MAX_THREADS_PER_BLOCK, self._handle, None
            )
        )

    @property
    def shared_size_bytes(self) -> int:
        """int : The size in bytes of statically-allocated shared memory required by this function.
        This attribute is read-only."""
        return handle_return(
            driver.cuKernelGetAttribute(
                driver.CUfunction_attribute.CU_FUNC_ATTRIBUTE_SHARED_SIZE_BYTES, self._handle, None
            )
        )

    @property
    def const_size_bytes(self) -> int:
        """int : The size in bytes of user-allocated constant memory required by this function.
        This attribute is read-only."""
        return handle_return(
            driver.cuKernelGetAttribute(
                driver.CUfunction_attribute.CU_FUNC_ATTRIBUTE_CONST_SIZE_BYTES, self._handle, None
            )
        )

    @property
    def local_size_bytes(self) -> int:
        """int : The size in bytes of local memory used by each thread of this function.
        This attribute is read-only."""
        return handle_return(
            driver.cuKernelGetAttribute(
                driver.CUfunction_attribute.CU_FUNC_ATTRIBUTE_LOCAL_SIZE_BYTES, self._handle, None
            )
        )

    @property
    def num_regs(self) -> int:
        """int : The number of registers used by each thread of this function.
        This attribute is read-only."""
        return handle_return(
            driver.cuKernelGetAttribute(driver.CUfunction_attribute.CU_FUNC_ATTRIBUTE_NUM_REGS, self._handle, None)
        )

    @property
    def ptx_version(self) -> int:
        """int : The PTX virtual architecture version for which the function was compiled.
        This attribute is read-only."""
        return handle_return(
            driver.cuKernelGetAttribute(driver.CUfunction_attribute.CU_FUNC_ATTRIBUTE_PTX_VERSION, self._handle, None)
        )

    @property
    def binary_version(self) -> int:
        """int : The binary architecture version for which the function was compiled.
        This attribute is read-only."""
        return handle_return(
            driver.cuKernelGetAttribute(
                driver.CUfunction_attribute.CU_FUNC_ATTRIBUTE_BINARY_VERSION, self._handle, None
            )
        )

    @property
    def cache_mode_ca(self) -> bool:
        """bool : Whether the function has been compiled with user specified option "-Xptxas --dlcm=ca" set.
        This attribute is read-only."""
        return bool(
            handle_return(
                driver.cuKernelGetAttribute(
                    driver.CUfunction_attribute.CU_FUNC_ATTRIBUTE_CACHE_MODE_CA, self._handle, None
                )
            )
        )

    @property
    def max_dynamic_shared_size_bytes(self) -> int:
        """int : The maximum size in bytes of dynamically-allocated shared memory that can be used
        by this function."""
        return handle_return(
            driver.cuKernelGetAttribute(
                driver.CUfunction_attribute.CU_FUNC_ATTRIBUTE_MAX_DYNAMIC_SHARED_SIZE_BYTES, self._handle, None
            )
        )

    @property
    def preferred_shared_memory_carveout(self) -> int:
        """int : The shared memory carveout preference, in percent of the total shared memory."""
        return handle_return(
            driver.cuKernelGetAttribute(
                driver.CUfunction_attribute.CU_FUNC_ATTRIBUTE_PREFERRED_SHARED_MEMORY_CARVEOUT, self._handle, None
            )
        )

    @property
    def cluster_size_must_be_set(self) -> bool:
        """bool : The kernel must launch with a valid cluster size specified.
        This attribute is read-only."""
        return bool(
            handle_return(
                driver.cuKernelGetAttribute(
                    driver.CUfunction_attribute.CU_FUNC_ATTRIBUTE_CLUSTER_SIZE_MUST_BE_SET, self._handle, None
                )
            )
        )

    @property
    def required_cluster_width(self) -> int:
        """int : The required cluster width in blocks."""
        return handle_return(
            driver.cuKernelGetAttribute(
                driver.CUfunction_attribute.CU_FUNC_ATTRIBUTE_REQUIRED_CLUSTER_WIDTH, self._handle, None
            )
        )

    @property
    def required_cluster_height(self) -> int:
        """int : The required cluster height in blocks."""
        return handle_return(
            driver.cuKernelGetAttribute(
                driver.CUfunction_attribute.CU_FUNC_ATTRIBUTE_REQUIRED_CLUSTER_HEIGHT, self._handle, None
            )
        )

    @property
    def required_cluster_depth(self) -> int:
        """int : The required cluster depth in blocks."""
        return handle_return(
            driver.cuKernelGetAttribute(
                driver.CUfunction_attribute.CU_FUNC_ATTRIBUTE_REQUIRED_CLUSTER_DEPTH, self._handle, None
            )
        )

    @property
    def non_portable_cluster_size_allowed(self) -> bool:
        """bool : Whether the function can be launched with non-portable cluster size."""
        return bool(
            handle_return(
                driver.cuKernelGetAttribute(
                    driver.CUfunction_attribute.CU_FUNC_ATTRIBUTE_NON_PORTABLE_CLUSTER_SIZE_ALLOWED, self._handle, None
                )
            )
        )

    @property
    def cluster_scheduling_policy_preference(self) -> int:
        """int : The block scheduling policy of a function."""
        return handle_return(
            driver.cuKernelGetAttribute(
                driver.CUfunction_attribute.CU_FUNC_ATTRIBUTE_CLUSTER_SCHEDULING_POLICY_PREFERENCE, self._handle, None
            )
        )


class Kernel:
    """Represent a compiled kernel that had been loaded onto the device.

    Kernel instances can execution when passed directly into the
    :func:`~launch` function.

    Directly creating a :obj:`~_module.Kernel` is not supported, and they
    should instead be created through a :obj:`~_module.ObjectCode` object.

    """

    __slots__ = ("_handle", "_module", "_attributes")

    def __init__(self):
        raise RuntimeError("directly constructing a Kernel instance is not supported")

    @staticmethod
    def _from_obj(obj, mod):
        assert isinstance(obj, _kernel_ctypes)
        assert isinstance(mod, ObjectCode)
        ker = Kernel.__new__(Kernel)
        ker._handle = obj
        ker._module = mod
        ker._attributes = None
        return ker

    @property
    def attributes(self):
        """Get the read-only attributes of this kernel."""
        if self._attributes is None:
            self._attributes = KernelAttributes._init(self._handle)
        return self._attributes

    # TODO: implement from_handle()


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
