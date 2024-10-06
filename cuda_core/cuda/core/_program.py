# Copyright (c) 2024, NVIDIA CORPORATION & AFFILIATES. ALL RIGHTS RESERVED.
#
# SPDX-License-Identifier: LicenseRef-NVIDIA-SOFTWARE-LICENSE

from cuda import nvrtc
from cuda.core._utils import handle_return
from cuda.core._module import ObjectCode


class Program:

    __slots__ = ("_handle", "_backend", )
    _supported_code_type = ("c++", )
    _supported_target_type = ("ptx", "cubin", "ltoir", )

    def __init__(self, code, code_type):
        if code_type not in self._supported_code_type:
            raise NotImplementedError
        self._handle = None

        if code_type.lower() == "c++":
            if not isinstance(code, str):
                raise TypeError
            # TODO: support pre-loaded headers & include names
            # TODO: allow tuples once NVIDIA/cuda-python#72 is resolved
            self._handle = handle_return(
                nvrtc.nvrtcCreateProgram(code.encode(), b"", 0, [], []))
            self._backend = "nvrtc"
        else:
            raise NotImplementedError

    def __del__(self):
        self.close()

    def close(self):
        if self._handle is not None:
            handle_return(nvrtc.nvrtcDestroyProgram(self._handle))
            self._handle = None

    def compile(self, target_type, options=(), name_expressions=(), logs=None):
        if target_type not in self._supported_target_type:
            raise NotImplementedError

        if self._backend == "nvrtc":
            if name_expressions:
                for n in name_expressions:
                    handle_return(
                        nvrtc.nvrtcAddNameExpression(self._handle, n.encode()),
                        handle=self._handle)
            # TODO: allow tuples once NVIDIA/cuda-python#72 is resolved
            options = list(o.encode() for o in options)
            handle_return(
                nvrtc.nvrtcCompileProgram(self._handle, len(options), options),
                handle=self._handle)

            size_func = getattr(nvrtc, f"nvrtcGet{target_type.upper()}Size")
            comp_func = getattr(nvrtc, f"nvrtcGet{target_type.upper()}")
            size = handle_return(size_func(self._handle), handle=self._handle)
            data = b" " * size
            handle_return(comp_func(self._handle, data), handle=self._handle)

            symbol_mapping = {}
            if name_expressions:
                for n in name_expressions:
                    symbol_mapping[n] = handle_return(nvrtc.nvrtcGetLoweredName(
                        self._handle, n.encode()))

            if logs is not None:
                logsize = handle_return(nvrtc.nvrtcGetProgramLogSize(self._handle))
                if logsize > 1:
                    log = b" " * logsize
                    handle_return(nvrtc.nvrtcGetProgramLog(self._handle, log))
                    logs.write(log.decode())

            # TODO: handle jit_options for ptx?

            return ObjectCode(data, target_type, symbol_mapping=symbol_mapping)

    @property
    def backend(self):
        return self._backend

    @property
    def handle(self):
        return self._handle
