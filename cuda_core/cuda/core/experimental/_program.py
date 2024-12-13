# Copyright (c) 2024, NVIDIA CORPORATION & AFFILIATES. ALL RIGHTS RESERVED.
#
# SPDX-License-Identifier: LicenseRef-NVIDIA-SOFTWARE-LICENSE

import weakref

from cuda import nvrtc
from cuda.core.experimental._module import ObjectCode
from cuda.core.experimental._utils import handle_return


class Program:
    """Represent a compilation machinery to process programs into
    :obj:`~_module.ObjectCode`.

    This object provides a unified interface to multiple underlying
    compiler libraries. Compilation support is enabled for a wide
    range of code types and compilation types.

    Parameters
    ----------
    code : Any
        String of the CUDA Runtime Compilation program.
    code_type : Any
        String of the code type. Currently only ``"c++"`` is supported.

    """

    class _MembersNeededForFinalize:
        __slots__ = ("handle",)

        def __init__(self, program_obj, handle):
            self.handle = handle
            weakref.finalize(program_obj, self.close)

        def close(self):
            if self.handle is not None:
                handle_return(nvrtc.nvrtcDestroyProgram(self.handle))
                self.handle = None

    __slots__ = ("__weakref__", "_mnff", "_backend")
    _supported_code_type = ("c++",)
    _supported_target_type = ("ptx", "cubin", "ltoir")

    def __init__(self, code, code_type):
        self._mnff = Program._MembersNeededForFinalize(self, None)

        if code_type not in self._supported_code_type:
            raise NotImplementedError

        if code_type.lower() == "c++":
            if not isinstance(code, str):
                raise TypeError
            # TODO: support pre-loaded headers & include names
            # TODO: allow tuples once NVIDIA/cuda-python#72 is resolved
            self._mnff.handle = handle_return(nvrtc.nvrtcCreateProgram(code.encode(), b"", 0, [], []))
            self._backend = "nvrtc"
        else:
            raise NotImplementedError

    def close(self):
        """Destroy this program."""
        self._mnff.close()

    def compile(self, target_type, options=(), name_expressions=(), logs=None):
        """Compile the program with a specific compilation type.

        Parameters
        ----------
        target_type : Any
            String of the targeted compilation type.
            Supported options are "ptx", "cubin" and "ltoir".
        options : Union[List, Tuple], optional
            List of compilation options associated with the backend
            of this :obj:`~_program.Program`. (Default to no options)
        name_expressions : Union[List, Tuple], optional
            List of explicit name expressions to become accessible.
            (Default to no expressions)
        logs : Any, optional
            Object with a write method to receive the logs generated
            from compilation.
            (Default to no logs)

        Returns
        -------
        :obj:`~_module.ObjectCode`
            Newly created code object.

        """
        if target_type not in self._supported_target_type:
            raise NotImplementedError

        if self._backend == "nvrtc":
            if name_expressions:
                for n in name_expressions:
                    handle_return(nvrtc.nvrtcAddNameExpression(self._mnff.handle, n.encode()), handle=self._mnff.handle)
            # TODO: allow tuples once NVIDIA/cuda-python#72 is resolved
            options = list(o.encode() for o in options)
            handle_return(nvrtc.nvrtcCompileProgram(self._mnff.handle, len(options), options), handle=self._mnff.handle)

            size_func = getattr(nvrtc, f"nvrtcGet{target_type.upper()}Size")
            comp_func = getattr(nvrtc, f"nvrtcGet{target_type.upper()}")
            size = handle_return(size_func(self._mnff.handle), handle=self._mnff.handle)
            data = b" " * size
            handle_return(comp_func(self._mnff.handle, data), handle=self._mnff.handle)

            symbol_mapping = {}
            if name_expressions:
                for n in name_expressions:
                    symbol_mapping[n] = handle_return(
                        nvrtc.nvrtcGetLoweredName(self._mnff.handle, n.encode()), handle=self._mnff.handle
                    )

            if logs is not None:
                logsize = handle_return(nvrtc.nvrtcGetProgramLogSize(self._mnff.handle), handle=self._mnff.handle)
                if logsize > 1:
                    log = b" " * logsize
                    handle_return(nvrtc.nvrtcGetProgramLog(self._mnff.handle, log), handle=self._mnff.handle)
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
        return self._mnff.handle
