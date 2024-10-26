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

# binding availability depends on cuda-python version
py_major_ver = int(importlib.metadata.version("cuda-python").split(".")[0])
if py_major_ver >= 12:
    _backend["new"] = {
        "file": cuda.cuLibraryLoadFromFile,
        "data": cuda.cuLibraryLoadData,
        "kernel": cuda.cuLibraryGetKernel,
    }
    _kernel_ctypes = (cuda.CUfunction, cuda.CUkernel)
else:
    _kernel_ctypes = (cuda.CUfunction,)
driver_ver = handle_return(cuda.cuDriverGetVersion())


class Kernel:

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

    __slots__ = ("_handle", "_code_type", "_module", "_loader", "_sym_map")
    _supported_code_type = ("cubin", "ptx", "fatbin")

    def __init__(self, module, code_type, jit_options=None, *,
                 symbol_mapping=None):
        if code_type not in self._supported_code_type:
            raise ValueError
        self._handle = None

        backend = "new" if (py_major_ver >= 12 and driver_ver >= 12000) else "old"
        self._loader = _backend[backend]

        if isinstance(module, str):
            if driver_ver < 12000 and jit_options is not None:
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
        try:
            name = self._sym_map[name]
        except KeyError:
            name = name.encode()
        data = handle_return(self._loader["kernel"](self._handle, name))
        return Kernel._from_obj(data, self)

    # TODO: implement from_handle()
