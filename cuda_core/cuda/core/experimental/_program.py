# Copyright (c) 2024-2025, NVIDIA CORPORATION & AFFILIATES. ALL RIGHTS RESERVED.
#
# SPDX-License-Identifier: LicenseRef-NVIDIA-SOFTWARE-LICENSE

import weakref
from dataclasses import dataclass
from typing import List, Optional, Tuple, Union

from cuda.core.experimental._device import Device
from cuda.core.experimental._module import ObjectCode
from cuda.core.experimental._utils import (
    _handle_boolean_option,
    check_or_create_options,
    handle_return,
    is_nested_sequence,
    is_sequence,
    nvrtc,
)


@dataclass
class ProgramOptions:
    """Customizable options for configuring `Program`.

    Attributes
    ----------
    arch : str, optional
        Pass the SM architecture value, such as ``sm_<CC>`` (for generating CUBIN) or
        ``compute_<CC>`` (for generating PTX). If not provided, the current device's architecture
        will be used.
    relocatable_device_code : bool, optional
        Enable (disable) the generation of relocatable device code.
        Default: False
        Maps to: ``--relocatable-device-code={true|false}`` (``-rdc``)
    extensible_whole_program : bool, optional
        Do extensible whole program compilation of device code.
        Default: False
        Maps to: ``--extensible-whole-program`` (``-ewp``)
    debug : bool, optional
        Generate debug information. If --dopt is not specified, then turns off all optimizations.
        Default: False
        Maps to: ``--device-debug`` (``-G``)
    lineinfo: bool, optional
        Generate line-number information.
        Default: False
        Maps to: ``--generate-line-info`` (``-lineinfo``)
    device_code_optimize : bool, optional
        Enable device code optimization. When specified along with ‘-G’, enables limited debug information generation
        for optimized device code.
        Default: None
        Maps to: ``--dopt on`` (``-dopt``)
    ptxas_options : Union[str, List[str]], optional
        Specify one or more options directly to ptxas, the PTX optimizing assembler. Options should be strings.
        For example ["-v", "-O2"].
        Default: None
        Maps to: ``--ptxas-options <options>`` (``-Xptxas``)
    max_register_count : int, optional
        Specify the maximum amount of registers that GPU functions can use.
        Default: None
        Maps to: ``--maxrregcount=<N>`` (``-maxrregcount``)
    ftz : bool, optional
        When performing single-precision floating-point operations, flush denormal values to zero or preserve denormal
        values.
        Default: False
        Maps to: ``--ftz={true|false}`` (``-ftz``)
    prec_sqrt : bool, optional
        For single-precision floating-point square root, use IEEE round-to-nearest mode or use a faster approximation.
        Default: True
        Maps to: ``--prec-sqrt={true|false}`` (``-prec-sqrt``)
    prec_div : bool, optional
        For single-precision floating-point division and reciprocals, use IEEE round-to-nearest mode or use a faster
        approximation.
        Default: True
        Maps to: ``--prec-div={true|false}`` (``-prec-div``)
    fma : bool, optional
        Enables (disables) the contraction of floating-point multiplies and adds/subtracts into floating-point
        multiply-add operations.
        Default: True
        Maps to: ``--fmad={true|false}`` (``-fmad``)
    use_fast_math : bool, optional
        Make use of fast math operations.
        Default: False
        Maps to: ``--use_fast_math`` (``-use_fast_math``)
    extra_device_vectorization : bool, optional
        Enables more aggressive device code vectorization in the NVVM optimizer.
        Default: False
        Maps to: ``--extra-device-vectorization`` (``-extra-device-vectorization``)
    link_time_optimization : bool, optional
        Generate intermediate code for later link-time optimization.
        Default: False
        Maps to: ``--dlink-time-opt`` (``-dlto``)
    gen_opt_lto : bool, optional
        Run the optimizer passes before generating the LTO IR.
        Default: False
        Maps to: ``--gen-opt-lto`` (``-gen-opt-lto``)
    define_macro : Union[str, Tuple[str, str], List[Union[str, Tuple[str, str]]]], optional
        Predefine a macro. Can be either a string, in which case that macro will be set to 1, a 2 element tuple of
        strings, in which case the first element is defined as the second, or a list of strings or tuples.
        Default: None
        Maps to: ``--define-macro=<def>`` (``-D``)
    undefine_macro : Union[str, List[str]], optional
        Cancel any previous definition of a macro, or list of macros.
        Default: None
        Maps to: ``--undefine-macro=<def>`` (``-U``)
    include_path : Union[str, List[str]], optional
        Add the directory or directories to the list of directories to be searched for headers.
        Default: None
        Maps to: ``--include-path=<dir>`` (``-I``)
    pre_include : Union[str, List[str]], optional
        Preinclude one or more headers during preprocessing. Can be either a string or a list of strings.
        Default: None
        Maps to: ``--pre-include=<header>`` (``-include``)
    no_source_include : bool, optional
        Disable the default behavior of adding the directory of each input source to the include path.
        Default: False
        Maps to: ``--no-source-include`` (``-no-source-include``)
    std : str, optional
        Set language dialect to C++03, C++11, C++14, C++17 or C++20.
        Default: c++17
        Maps to: ``--std={c++03|c++11|c++14|c++17|c++20}`` (``-std``)
    builtin_move_forward : bool, optional
        Provide builtin definitions of std::move and std::forward.
        Default: True
        Maps to: ``--builtin-move-forward={true|false}`` (``-builtin-move-forward``)
    builtin_initializer_list : bool, optional
        Provide builtin definitions of std::initializer_list class and member functions.
        Default: True
        Maps to: ``--builtin-initializer-list={true|false}`` (``-builtin-initializer-list``)
    disable_warnings : bool, optional
        Inhibit all warning messages.
        Default: False
        Maps to: ``--disable-warnings`` (``-w``)
    restrict : bool, optional
        Programmer assertion that all kernel pointer parameters are restrict pointers.
        Default: False
        Maps to: ``--restrict`` (``-restrict``)
    device_as_default_execution_space : bool, optional
        Treat entities with no execution space annotation as __device__ entities.
        Default: False
        Maps to: ``--device-as-default-execution-space`` (``-default-device``)
    device_int128 : bool, optional
        Allow the __int128 type in device code.
        Default: False
        Maps to: ``--device-int128`` (``-device-int128``)
    optimization_info : str, optional
        Provide optimization reports for the specified kind of optimization.
        Default: None
        Maps to: ``--optimization-info=<kind>`` (``-opt-info``)
    no_display_error_number : bool, optional
        Disable the display of a diagnostic number for warning messages.
        Default: False
        Maps to: ``--no-display-error-number`` (``-no-err-no``)
    diag_error : Union[int, List[int]], optional
        Emit error for a specified diagnostic message number or comma separated list of numbers.
        Default: None
        Maps to: ``--diag-error=<error-number>, ...`` (``-diag-error``)
    diag_suppress : Union[int, List[int]], optional
        Suppress a specified diagnostic message number or comma separated list of numbers.
        Default: None
        Maps to: ``--diag-suppress=<error-number>,…`` (``-diag-suppress``)
    diag_warn : Union[int, List[int]], optional
        Emit warning for a specified diagnostic message number or comma separated lis of numbers.
        Default: None
        Maps to: ``--diag-warn=<error-number>,…`` (``-diag-warn``)
    brief_diagnostics : bool, optional
        Disable or enable showing source line and column info in a diagnostic.
        Default: False
        Maps to: ``--brief-diagnostics={true|false}`` (``-brief-diag``)
    time : str, optional
        Generate a CSV table with the time taken by each compilation phase.
        Default: None
        Maps to: ``--time=<file-name>`` (``-time``)
    split_compile : int, optional
        Perform compiler optimizations in parallel.
        Default: 1
        Maps to: ``--split-compile= <number of threads>`` (``-split-compile``)
    fdevice_syntax_only : bool, optional
        Ends device compilation after front-end syntax checking.
        Default: False
        Maps to: ``--fdevice-syntax-only`` (``-fdevice-syntax-only``)
    minimal : bool, optional
        Omit certain language features to reduce compile time for small programs.
        Default: False
        Maps to: ``--minimal`` (``-minimal``)
    """

    arch: Optional[str] = None
    relocatable_device_code: Optional[bool] = None
    extensible_whole_program: Optional[bool] = None
    debug: Optional[bool] = None
    lineinfo: Optional[bool] = None
    device_code_optimize: Optional[bool] = None
    ptxas_options: Optional[Union[str, List[str], Tuple[str]]] = None
    max_register_count: Optional[int] = None
    ftz: Optional[bool] = None
    prec_sqrt: Optional[bool] = None
    prec_div: Optional[bool] = None
    fma: Optional[bool] = None
    use_fast_math: Optional[bool] = None
    extra_device_vectorization: Optional[bool] = None
    link_time_optimization: Optional[bool] = None
    gen_opt_lto: Optional[bool] = None
    define_macro: Optional[
        Union[str, Tuple[str, str], List[Union[str, Tuple[str, str]]], Tuple[Union[str, Tuple[str, str]]]]
    ] = None
    undefine_macro: Optional[Union[str, List[str], Tuple[str]]] = None
    include_path: Optional[Union[str, List[str], Tuple[str]]] = None
    pre_include: Optional[Union[str, List[str], Tuple[str]]] = None
    no_source_include: Optional[bool] = None
    std: Optional[str] = None
    builtin_move_forward: Optional[bool] = None
    builtin_initializer_list: Optional[bool] = None
    disable_warnings: Optional[bool] = None
    restrict: Optional[bool] = None
    device_as_default_execution_space: Optional[bool] = None
    device_int128: Optional[bool] = None
    optimization_info: Optional[str] = None
    no_display_error_number: Optional[bool] = None
    diag_error: Optional[Union[int, List[int], Tuple[int]]] = None
    diag_suppress: Optional[Union[int, List[int], Tuple[int]]] = None
    diag_warn: Optional[Union[int, List[int], Tuple[int]]] = None
    brief_diagnostics: Optional[bool] = None
    time: Optional[str] = None
    split_compile: Optional[int] = None
    fdevice_syntax_only: Optional[bool] = None
    minimal: Optional[bool] = None

    def __post_init__(self):
        self._formatted_options = []
        if self.arch is not None:
            self._formatted_options.append(f"--gpu-architecture={self.arch}")
        else:
            self._formatted_options.append(
                "--gpu-architecture=sm_" + "".join(f"{i}" for i in Device().compute_capability)
            )
        if self.relocatable_device_code is not None:
            self._formatted_options.append(
                f"--relocatable-device-code={_handle_boolean_option(self.relocatable_device_code)}"
            )
        if self.extensible_whole_program is not None and self.extensible_whole_program:
            self._formatted_options.append("--extensible-whole-program")
        if self.debug is not None and self.debug:
            self._formatted_options.append("--device-debug")
        if self.lineinfo is not None and self.lineinfo:
            self._formatted_options.append("--generate-line-info")
        if self.device_code_optimize is not None:
            self._formatted_options.append(f"--dopt={'on' if self.device_code_optimize else 'off'}")
        if self.ptxas_options is not None:
            self._formatted_options.append("--ptxas-options")
            if isinstance(self.ptxas_options, str):
                self._formatted_options.append(self.ptxas_options)
            elif is_sequence(self.ptxas_options):
                for option in self.ptxas_options:
                    self._formatted_options.append(option)
        if self.max_register_count is not None:
            self._formatted_options.append(f"--maxrregcount={self.max_register_count}")
        if self.ftz is not None:
            self._formatted_options.append(f"--ftz={_handle_boolean_option(self.ftz)}")
        if self.prec_sqrt is not None:
            self._formatted_options.append(f"--prec-sqrt={_handle_boolean_option(self.prec_sqrt)}")
        if self.prec_div is not None:
            self._formatted_options.append(f"--prec-div={_handle_boolean_option(self.prec_div)}")
        if self.fma is not None:
            self._formatted_options.append(f"--fmad={_handle_boolean_option(self.fma)}")
        if self.use_fast_math is not None and self.use_fast_math:
            self._formatted_options.append("--use_fast_math")
        if self.extra_device_vectorization is not None and self.extra_device_vectorization:
            self._formatted_options.append("--extra-device-vectorization")
        if self.link_time_optimization is not None and self.link_time_optimization:
            self._formatted_options.append("--dlink-time-opt")
        if self.gen_opt_lto is not None and self.gen_opt_lto:
            self._formatted_options.append("--gen-opt-lto")
        if self.define_macro is not None:
            if isinstance(self.define_macro, str):
                self._formatted_options.append(f"--define-macro={self.define_macro}")
            elif isinstance(self.define_macro, tuple):
                assert len(self.define_macro) == 2
                self._formatted_options.append(f"--define-macro={self.define_macro[0]}={self.define_macro[1]}")
            elif is_nested_sequence(self.define_macro):
                for macro in self.define_macro:
                    if isinstance(macro, tuple):
                        assert len(macro) == 2
                        self._formatted_options.append(f"--define-macro={macro[0]}={macro[1]}")
                    else:
                        self._formatted_options.append(f"--define-macro={macro}")

        if self.undefine_macro is not None:
            if isinstance(self.undefine_macro, str):
                self._formatted_options.append(f"--undefine-macro={self.undefine_macro}")
            elif is_sequence(self.undefine_macro):
                for macro in self.undefine_macro:
                    self._formatted_options.append(f"--undefine-macro={macro}")
        if self.include_path is not None:
            if isinstance(self.include_path, str):
                self._formatted_options.append(f"--include-path={self.include_path}")
            elif is_sequence(self.include_path):
                for path in self.include_path:
                    self._formatted_options.append(f"--include-path={path}")
        if self.pre_include is not None:
            if isinstance(self.pre_include, str):
                self._formatted_options.append(f"--pre-include={self.pre_include}")
            elif is_sequence(self.pre_include):
                for header in self.pre_include:
                    self._formatted_options.append(f"--pre-include={header}")

        if self.no_source_include is not None and self.no_source_include:
            self._formatted_options.append("--no-source-include")
        if self.std is not None:
            self._formatted_options.append(f"--std={self.std}")
        if self.builtin_move_forward is not None:
            self._formatted_options.append(
                f"--builtin-move-forward={_handle_boolean_option(self.builtin_move_forward)}"
            )
        if self.builtin_initializer_list is not None:
            self._formatted_options.append(
                f"--builtin-initializer-list={_handle_boolean_option(self.builtin_initializer_list)}"
            )
        if self.disable_warnings is not None and self.disable_warnings:
            self._formatted_options.append("--disable-warnings")
        if self.restrict is not None and self.restrict:
            self._formatted_options.append("--restrict")
        if self.device_as_default_execution_space is not None and self.device_as_default_execution_space:
            self._formatted_options.append("--device-as-default-execution-space")
        if self.device_int128 is not None and self.device_int128:
            self._formatted_options.append("--device-int128")
        if self.optimization_info is not None:
            self._formatted_options.append(f"--optimization-info={self.optimization_info}")
        if self.no_display_error_number is not None and self.no_display_error_number:
            self._formatted_options.append("--no-display-error-number")
        if self.diag_error is not None:
            if isinstance(self.diag_error, int):
                self._formatted_options.append(f"--diag-error={self.diag_error}")
            elif is_sequence(self.diag_error):
                for error in self.diag_error:
                    self._formatted_options.append(f"--diag-error={error}")
        if self.diag_suppress is not None:
            if isinstance(self.diag_suppress, int):
                self._formatted_options.append(f"--diag-suppress={self.diag_suppress}")
            elif is_sequence(self.diag_suppress):
                for suppress in self.diag_suppress:
                    self._formatted_options.append(f"--diag-suppress={suppress}")
        if self.diag_warn is not None:
            if isinstance(self.diag_warn, int):
                self._formatted_options.append(f"--diag-warn={self.diag_warn}")
            elif is_sequence(self.diag_warn):
                for warn in self.diag_warn:
                    self._formatted_options.append(f"--diag-warn={warn}")
        if self.brief_diagnostics is not None:
            self._formatted_options.append(f"--brief-diagnostics={_handle_boolean_option(self.brief_diagnostics)}")
        if self.time is not None:
            self._formatted_options.append(f"--time={self.time}")
        if self.split_compile is not None:
            self._formatted_options.append(f"--split-compile={self.split_compile}")
        if self.fdevice_syntax_only is not None and self.fdevice_syntax_only:
            self._formatted_options.append("--fdevice-syntax-only")
        if self.minimal is not None and self.minimal:
            self._formatted_options.append("--minimal")

    def _as_bytes(self):
        # TODO: allow tuples once NVIDIA/cuda-python#72 is resolved
        return list(o.encode() for o in self._formatted_options)

    def __repr__(self):
        # __TODO__ improve this
        return self._formatted_options


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
    options : ProgramOptions, optional
        A ProgramOptions object to customize the compilation process.
        See :obj:`ProgramOptions` for more information.
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

    __slots__ = ("__weakref__", "_mnff", "_backend", "_options")
    _supported_code_type = ("c++",)
    _supported_target_type = ("ptx", "cubin", "ltoir")

    def __init__(self, code, code_type, options: ProgramOptions = None):
        self._mnff = Program._MembersNeededForFinalize(self, None)

        self._options = options = check_or_create_options(ProgramOptions, options, "Program options")

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

    def compile(self, target_type, name_expressions=(), logs=None):
        """Compile the program with a specific compilation type.

        Parameters
        ----------
        target_type : Any
            String of the targeted compilation type.
            Supported options are "ptx", "cubin" and "ltoir".
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
            options = self._options._as_bytes()
            handle_return(
                nvrtc.nvrtcCompileProgram(self._mnff.handle, len(options), options),
                handle=self._mnff.handle,
            )

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
