# Copyright (c) 2024, NVIDIA CORPORATION & AFFILIATES. ALL RIGHTS RESERVED.
#
# SPDX-License-Identifier: LicenseRef-NVIDIA-SOFTWARE-LICENSE

import importlib.metadata

from cuda import cuda, cudart
from cuda.core.experimental._utils import handle_return


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
    :func:`~cuda.core.experimental.launch` function.

    Directly creating a :obj:`Kernel` is not supported, and they
    should instead be created through a :obj:`ObjectCode` object.

    """

    __slots__ = ("_handle", "_module",)

    def __init__(self):
        raise NotImplementedError("directly constructing a Kernel instance is not supported")

    @staticmethod
    def _from_obj(obj, mod):
        assert isinstance(obj, _kernel_ctypes)
        assert isinstance(mod, ObjectCode)
        ker = Kernel.__new__(Kernel)
        ker._handle = obj
        ker._module = mod
        return ker

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

    __slots__ = ("_handle", "_code_type", "_module", "_loader", "_sym_map")
    _supported_code_type = ("cubin", "ptx", "ltoir", "fatbin")

    def __init__(self, module, code_type, jit_options=None, *,
                 symbol_mapping=None):
        if code_type not in self._supported_code_type:
            raise ValueError
        _lazy_init()
        self._handle = None

        backend = "new" if (_py_major_ver >= 12 and _driver_ver >= 12000) else "old"
        self._loader = _backend[backend]

        if isinstance(module, str):
            # TODO: this option is only taken by the new library APIs, but we have
            # a bug that we can't easily support it just yet (NVIDIA/cuda-python#73).
            if jit_options is not None:
                raise ValueError
            module = module.encode()
            self._handle = handle_return(self._loader["file"](module))
        else:
            assert isinstance(module, bytes)
            if jit_options is None:
                jit_options = {}
            if backend == "new":
                args = (module, list(jit_options.keys()), list(jit_options.values()), len(jit_options),
                        # TODO: support library options
                        [], [], 0)
            else:  # "old" backend
                args = (module, len(jit_options), list(jit_options.keys()), list(jit_options.values()))
            self._handle = handle_return(self._loader["data"](*args))

        self._code_type = code_type
        self._module = module
        self._sym_map = {} if symbol_mapping is None else symbol_mapping

    def __del__(self):
        # TODO: do we want to unload? Probably not..
        pass

    def get_kernel(self, name):
        """Return the :obj:`Kernel` of a specified name from this object code.

        Parameters
        ----------
        name : Any
            Name of the kernel to retrieve.

        Returns
        -------
        :obj:`Kernel`
            Newly created kernel object.

        """
        try:
            name = self._sym_map[name]
        except KeyError:
            name = name.encode()
        data = handle_return(self._loader["kernel"](self._handle, name))
        return Kernel._from_obj(data, self)

    # TODO: implement from_handle()
