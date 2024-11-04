# Copyright (c) 2024, NVIDIA CORPORATION & AFFILIATES. ALL RIGHTS RESERVED.
#
# SPDX-License-Identifier: LicenseRef-NVIDIA-SOFTWARE-LICENSE

from cuda import nvrtc
from cuda.core.experimental._utils import handle_return
from cuda.core.experimental._module import ObjectCode


class Program:
    """Represent a compilation machinery to process programs into :obj:`ObjectCode`.

    This object provides a unified interface to multiple underlying
    compiler libraries. Compilation support is enabled for a wide
    range of code types and compilation types.

    Parameters
    ----------
    code : Any
        String of the CUDA Runtime Compilation program.
    code_type : Any
        String of the code type. Only "c++" is currently supported.

    """

    __slots__ = ("_handle", "_backend", )
    _supported_code_type = ("c++", )
    _supported_target_type = ("ptx", "cubin", "ltoir", )

    def __init__(self, code, code_type):
        self._handle = None
        if code_type not in self._supported_code_type:
            raise NotImplementedError

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
        """Return close(self)."""
        self.close()

    def close(self):
        """Destroy this program."""
        if self._handle is not None:
            handle_return(nvrtc.nvrtcDestroyProgram(self._handle))
            self._handle = None

    def compile(self, target_type, options=(), name_expressions=(), logs=None):
        """Compile the program with a specific compilation type.

        Parameters
        ----------
        target_type : Any
            String of the targeted compilation type.
            Supported options are "ptx", "cubin" and "ltoir".
        options : Union[List, Tuple], optional
            List of compilation options associated with the backend
            of this :obj:`Program`. (Default to no options)
        name_expressions : Union[List, Tuple], optional
            List of explicit name expressions to become accessible.
            (Default to no expressions)
        logs : Any, optional
            Object with a write method to receive the logs generated
            from compilation.
            (Default to no logs)

        Returns
        -------
        :obj:`ObjectCode`
            Newly created code object.

        """
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
                        self._handle, n.encode()), handle=self._handle)

            if logs is not None:
                logsize = handle_return(nvrtc.nvrtcGetProgramLogSize(self._handle),
                                        handle=self._handle)
                if logsize > 1:
                    log = b" " * logsize
                    handle_return(nvrtc.nvrtcGetProgramLog(self._handle, log),
                                  handle=self._handle)
                    logs.write(log.decode())

            # TODO: handle jit_options for ptx?

            return ObjectCode(data, target_type, symbol_mapping=symbol_mapping)

    @property
    def backend(self):
        """Return the backend type string associated with this program."""
        return self._backend

    @property
    def handle(self):
        """Return the program handle object."""
        return self._handle
