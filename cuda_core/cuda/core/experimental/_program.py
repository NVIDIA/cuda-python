# SPDX-FileCopyrightText: Copyright (c) 2024-2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import weakref
from contextlib import contextmanager
from dataclasses import dataclass
from typing import TYPE_CHECKING, Union
from warnings import warn

if TYPE_CHECKING:
    import cuda.bindings

from cuda.core.experimental._device import Device
from cuda.core.experimental._linker import Linker, LinkerHandleT, LinkerOptions
from cuda.core.experimental._module import ObjectCode
from cuda.core.experimental._utils.clear_error_support import assert_type
from cuda.core.experimental._utils.cuda_utils import (
    _handle_boolean_option,
    check_or_create_options,
    driver,
    get_binding_version,
    handle_return,
    is_nested_sequence,
    is_sequence,
    nvrtc,
)


@contextmanager
def _nvvm_exception_manager(self):
    """
    Taken from _linker.py
    """
    try:
        yield
    except Exception as e:
        error_log = ""
        if hasattr(self, "_mnff"):
            try:
                nvvm = _get_nvvm_module()
                logsize = nvvm.get_program_log_size(self._mnff.handle)
                if logsize > 1:
                    log = bytearray(logsize)
                    nvvm.get_program_log(self._mnff.handle, log)
                    error_log = log.decode("utf-8", errors="backslashreplace")
            except Exception:
                error_log = ""
        # Starting Python 3.11 we could also use Exception.add_note() for the same purpose, but
        # unfortunately we are still supporting Python 3.9/3.10...
        e.args = (e.args[0] + (f"\nNVVM program log: {error_log}" if error_log else ""), *e.args[1:])
        raise e


_nvvm_module = None
_nvvm_import_attempted = False


def _get_nvvm_module():
    """
    Handles the import of NVVM module with version and availability checks.
    NVVM bindings were added in cuda-bindings 12.9.0, so we need to handle cases where:
    1. cuda.bindings is not new enough (< 12.9.0)
    2. libnvvm is not found in the Python environment

    Returns:
        The nvvm module if available and working

    Raises:
        RuntimeError: If NVVM is not available due to version or library issues
    """
    global _nvvm_module, _nvvm_import_attempted

    if _nvvm_import_attempted:
        if _nvvm_module is None:
            raise RuntimeError("NVVM module is not available (previous import attempt failed)")
        return _nvvm_module

    _nvvm_import_attempted = True

    try:
        version = get_binding_version()
        if version < (12, 9):
            raise RuntimeError(
                f"NVVM bindings require cuda-bindings >= 12.9.0, but found {version[0]}.{version[1]}.x. "
                "Please update cuda-bindings to use NVVM features."
            )

        from cuda.bindings import nvvm
        from cuda.bindings._internal.nvvm import _inspect_function_pointer

        if _inspect_function_pointer("__nvvmCreateProgram") == 0:
            raise RuntimeError("NVVM library (libnvvm) is not available in this Python environment. ")

        _nvvm_module = nvvm
        return _nvvm_module

    except RuntimeError as e:
        _nvvm_module = None
        raise e


def _process_define_macro_inner(formatted_options, macro):
    if isinstance(macro, str):
        formatted_options.append(f"--define-macro={macro}")
        return True
    if isinstance(macro, tuple):
        if len(macro) != 2 or any(not isinstance(val, str) for val in macro):
            raise RuntimeError(f"Expected define_macro tuple[str, str], got {macro}")
        formatted_options.append(f"--define-macro={macro[0]}={macro[1]}")
        return True
    return False


def _process_define_macro(formatted_options, macro):
    union_type = "Union[str, tuple[str, str]]"
    if _process_define_macro_inner(formatted_options, macro):
        return
    if is_nested_sequence(macro):
        for seq_macro in macro:
            if not _process_define_macro_inner(formatted_options, seq_macro):
                raise RuntimeError(f"Expected define_macro {union_type}, got {seq_macro}")
        return
    raise RuntimeError(f"Expected define_macro {union_type}, list[{union_type}], got {macro}")


@dataclass
class ProgramOptions:
    """Customizable options for configuring `Program`.

    Attributes
    ----------
    name : str, optional
        Name of the program. If the compilation succeeds, the name is passed down to the generated `ObjectCode`.
    arch : str, optional
        Pass the SM architecture value, such as ``sm_<CC>`` (for generating CUBIN) or
        ``compute_<CC>`` (for generating PTX). If not provided, the current device's architecture
        will be used.
    relocatable_device_code : bool, optional
        Enable (disable) the generation of relocatable device code.
        Default: False
    extensible_whole_program : bool, optional
        Do extensible whole program compilation of device code.
        Default: False
    debug : bool, optional
        Generate debug information. If --dopt is not specified, then turns off all optimizations.
        Default: False
    lineinfo: bool, optional
        Generate line-number information.
        Default: False
    device_code_optimize : bool, optional
        Enable device code optimization. When specified along with ‘-G’, enables limited debug information generation
        for optimized device code.
        Default: None
    ptxas_options : Union[str, list[str]], optional
        Specify one or more options directly to ptxas, the PTX optimizing assembler. Options should be strings.
        For example ["-v", "-O2"].
        Default: None
    max_register_count : int, optional
        Specify the maximum amount of registers that GPU functions can use.
        Default: None
    ftz : bool, optional
        When performing single-precision floating-point operations, flush denormal values to zero or preserve denormal
        values.
        Default: False
    prec_sqrt : bool, optional
        For single-precision floating-point square root, use IEEE round-to-nearest mode or use a faster approximation.
        Default: True
    prec_div : bool, optional
        For single-precision floating-point division and reciprocals, use IEEE round-to-nearest mode or use a faster
        approximation.
        Default: True
    fma : bool, optional
        Enables (disables) the contraction of floating-point multiplies and adds/subtracts into floating-point
        multiply-add operations.
        Default: True
    use_fast_math : bool, optional
        Make use of fast math operations.
        Default: False
    extra_device_vectorization : bool, optional
        Enables more aggressive device code vectorization in the NVVM optimizer.
        Default: False
    link_time_optimization : bool, optional
        Generate intermediate code for later link-time optimization.
        Default: False
    gen_opt_lto : bool, optional
        Run the optimizer passes before generating the LTO IR.
        Default: False
    define_macro : Union[str, tuple[str, str], list[Union[str, tuple[str, str]]]], optional
        Predefine a macro. Can be either a string, in which case that macro will be set to 1, a 2 element tuple of
        strings, in which case the first element is defined as the second, or a list of strings or tuples.
        Default: None
    undefine_macro : Union[str, list[str]], optional
        Cancel any previous definition of a macro, or list of macros.
        Default: None
    include_path : Union[str, list[str]], optional
        Add the directory or directories to the list of directories to be searched for headers.
        Default: None
    pre_include : Union[str, list[str]], optional
        Preinclude one or more headers during preprocessing. Can be either a string or a list of strings.
        Default: None
    no_source_include : bool, optional
        Disable the default behavior of adding the directory of each input source to the include path.
        Default: False
    std : str, optional
        Set language dialect to C++03, C++11, C++14, C++17 or C++20.
        Default: c++17
    builtin_move_forward : bool, optional
        Provide builtin definitions of std::move and std::forward.
        Default: True
    builtin_initializer_list : bool, optional
        Provide builtin definitions of std::initializer_list class and member functions.
        Default: True
    disable_warnings : bool, optional
        Inhibit all warning messages.
        Default: False
    restrict : bool, optional
        Programmer assertion that all kernel pointer parameters are restrict pointers.
        Default: False
    device_as_default_execution_space : bool, optional
        Treat entities with no execution space annotation as __device__ entities.
        Default: False
    device_int128 : bool, optional
        Allow the __int128 type in device code.
        Default: False
    optimization_info : str, optional
        Provide optimization reports for the specified kind of optimization.
        Default: None
    no_display_error_number : bool, optional
        Disable the display of a diagnostic number for warning messages.
        Default: False
    diag_error : Union[int, list[int]], optional
        Emit error for a specified diagnostic message number or comma separated list of numbers.
        Default: None
    diag_suppress : Union[int, list[int]], optional
        Suppress a specified diagnostic message number or comma separated list of numbers.
        Default: None
    diag_warn : Union[int, list[int]], optional
        Emit warning for a specified diagnostic message number or comma separated lis of numbers.
        Default: None
    brief_diagnostics : bool, optional
        Disable or enable showing source line and column info in a diagnostic.
        Default: False
    time : str, optional
        Generate a CSV table with the time taken by each compilation phase.
        Default: None
    split_compile : int, optional
        Perform compiler optimizations in parallel.
        Default: 1
    fdevice_syntax_only : bool, optional
        Ends device compilation after front-end syntax checking.
        Default: False
    minimal : bool, optional
        Omit certain language features to reduce compile time for small programs.
        Default: False
    """

    name: str | None = "<default program>"
    arch: str | None = None
    relocatable_device_code: bool | None = None
    extensible_whole_program: bool | None = None
    debug: bool | None = None
    lineinfo: bool | None = None
    device_code_optimize: bool | None = None
    ptxas_options: Union[str, list[str], tuple[str]] | None = None
    max_register_count: int | None = None
    ftz: bool | None = None
    prec_sqrt: bool | None = None
    prec_div: bool | None = None
    fma: bool | None = None
    use_fast_math: bool | None = None
    extra_device_vectorization: bool | None = None
    link_time_optimization: bool | None = None
    gen_opt_lto: bool | None = None
    define_macro: (
        Union[str, tuple[str, str], list[Union[str, tuple[str, str]]], tuple[Union[str, tuple[str, str]]]] | None
    ) = None
    undefine_macro: Union[str, list[str], tuple[str]] | None = None
    include_path: Union[str, list[str], tuple[str]] | None = None
    pre_include: Union[str, list[str], tuple[str]] | None = None
    no_source_include: bool | None = None
    std: str | None = None
    builtin_move_forward: bool | None = None
    builtin_initializer_list: bool | None = None
    disable_warnings: bool | None = None
    restrict: bool | None = None
    device_as_default_execution_space: bool | None = None
    device_int128: bool | None = None
    optimization_info: str | None = None
    no_display_error_number: bool | None = None
    diag_error: Union[int, list[int], tuple[int]] | None = None
    diag_suppress: Union[int, list[int], tuple[int]] | None = None
    diag_warn: Union[int, list[int], tuple[int]] | None = None
    brief_diagnostics: bool | None = None
    time: str | None = None
    split_compile: int | None = None
    fdevice_syntax_only: bool | None = None
    minimal: bool | None = None

    def __post_init__(self):
        self._name = self.name.encode()

        self._formatted_options = []
        if self.arch is not None:
            self._formatted_options.append(f"-arch={self.arch}")
        else:
            self.arch = f"sm_{Device().arch}"
            self._formatted_options.append(f"-arch={self.arch}")
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
        if self.device_code_optimize is not None and self.device_code_optimize:
            self._formatted_options.append("--dopt=on")
        if self.ptxas_options is not None:
            opt_name = "--ptxas-options"
            if isinstance(self.ptxas_options, str):
                self._formatted_options.append(f"{opt_name}={self.ptxas_options}")
            elif is_sequence(self.ptxas_options):
                for opt_value in self.ptxas_options:
                    self._formatted_options.append(f"{opt_name}={opt_value}")
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
            _process_define_macro(self._formatted_options, self.define_macro)
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
        return str(self._formatted_options)


ProgramHandleT = Union["cuda.bindings.nvrtc.nvrtcProgram", LinkerHandleT]


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
        String of the code type. Currently ``"ptx"``, ``"c++"``, and ``"nvvm"`` are supported.
    options : ProgramOptions, optional
        A ProgramOptions object to customize the compilation process.
        See :obj:`ProgramOptions` for more information.
    """

    class _MembersNeededForFinalize:
        __slots__ = "handle", "backend"

        def __init__(self, program_obj, handle, backend):
            self.handle = handle
            self.backend = backend
            weakref.finalize(program_obj, self.close)

        def close(self):
            if self.handle is not None:
                if self.backend == "NVRTC":
                    handle_return(nvrtc.nvrtcDestroyProgram(self.handle))
                elif self.backend == "NVVM":
                    nvvm = _get_nvvm_module()
                    nvvm.destroy_program(self.handle)
                self.handle = None

    __slots__ = ("__weakref__", "_mnff", "_backend", "_linker", "_options")

    def __init__(self, code, code_type, options: ProgramOptions = None):
        self._mnff = Program._MembersNeededForFinalize(self, None, None)

        self._options = options = check_or_create_options(ProgramOptions, options, "Program options")
        code_type = code_type.lower()

        if code_type == "c++":
            assert_type(code, str)
            # TODO: support pre-loaded headers & include names
            # TODO: allow tuples once NVIDIA/cuda-python#72 is resolved

            self._mnff.handle = handle_return(nvrtc.nvrtcCreateProgram(code.encode(), options._name, 0, [], []))
            self._mnff.backend = "NVRTC"
            self._backend = "NVRTC"
            self._linker = None

        elif code_type == "ptx":
            assert_type(code, str)
            self._linker = Linker(
                ObjectCode._init(code.encode(), code_type), options=self._translate_program_options(options)
            )
            self._backend = self._linker.backend

        elif code_type == "nvvm":
            if isinstance(code, str):
                code = code.encode("utf-8")
            elif not isinstance(code, (bytes, bytearray)):
                raise TypeError("NVVM IR code must be provided as str, bytes, or bytearray")

            nvvm = _get_nvvm_module()
            self._mnff.handle = nvvm.create_program()
            self._mnff.backend = "NVVM"
            nvvm.add_module_to_program(self._mnff.handle, code, len(code), options._name.decode())
            self._backend = "NVVM"
            self._linker = None

        else:
            supported_code_types = ("c++", "ptx", "nvvm")
            assert code_type not in supported_code_types, f"{code_type=}"
            raise RuntimeError(f"Unsupported {code_type=} ({supported_code_types=})")

    def _translate_program_options(self, options: ProgramOptions) -> LinkerOptions:
        return LinkerOptions(
            name=options.name,
            arch=options.arch,
            max_register_count=options.max_register_count,
            time=options.time,
            debug=options.debug,
            lineinfo=options.lineinfo,
            ftz=options.ftz,
            prec_div=options.prec_div,
            prec_sqrt=options.prec_sqrt,
            fma=options.fma,
            link_time_optimization=options.link_time_optimization,
            split_compile=options.split_compile,
            ptxas_options=options.ptxas_options,
        )

    def _translate_program_options_to_nvvm(self, options: ProgramOptions) -> list[str]:
        """Translate ProgramOptions to NVVM-specific compilation options."""
        nvvm_options = []

        assert options.arch is not None
        arch = options.arch
        if arch.startswith("sm_"):
            arch = f"compute_{arch[3:]}"
        nvvm_options.append(f"-arch={arch}")
        if options.debug:
            nvvm_options.append("-g")
        if options.device_code_optimize is False:
            nvvm_options.append("-opt=0")
        elif options.device_code_optimize is True:
            nvvm_options.append("-opt=3")
        # NVVM is not consistent with NVRTC, it uses 0/1 instead...
        if options.ftz is not None:
            nvvm_options.append(f"-ftz={'1' if options.ftz else '0'}")
        if options.prec_sqrt is not None:
            nvvm_options.append(f"-prec-sqrt={'1' if options.prec_sqrt else '0'}")
        if options.prec_div is not None:
            nvvm_options.append(f"-prec-div={'1' if options.prec_div else '0'}")
        if options.fma is not None:
            nvvm_options.append(f"-fma={'1' if options.fma else '0'}")

        return nvvm_options

    def close(self):
        """Destroy this program."""
        if self._linker:
            self._linker.close()
        self._mnff.close()

    @staticmethod
    def _can_load_generated_ptx():
        driver_ver = handle_return(driver.cuDriverGetVersion())
        nvrtc_major, nvrtc_minor = handle_return(nvrtc.nvrtcVersion())
        return nvrtc_major * 1000 + nvrtc_minor * 10 <= driver_ver

    def compile(self, target_type, name_expressions=(), logs=None):
        """Compile the program with a specific compilation type.

        Parameters
        ----------
        target_type : Any
            String of the targeted compilation type.
            Supported options are "ptx", "cubin" and "ltoir".
        name_expressions : Union[list, tuple], optional
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
        supported_target_types = ("ptx", "cubin", "ltoir")
        if target_type not in supported_target_types:
            raise ValueError(f'Unsupported target_type="{target_type}" ({supported_target_types=})')

        if self._backend == "NVRTC":
            if target_type == "ptx" and not self._can_load_generated_ptx():
                warn(
                    "The CUDA driver version is older than the backend version. "
                    "The generated ptx will not be loadable by the current driver.",
                    stacklevel=1,
                    category=RuntimeWarning,
                )
            if name_expressions:
                for n in name_expressions:
                    handle_return(
                        nvrtc.nvrtcAddNameExpression(self._mnff.handle, n.encode()),
                        handle=self._mnff.handle,
                    )
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
                    logs.write(log.decode("utf-8", errors="backslashreplace"))

            return ObjectCode._init(data, target_type, symbol_mapping=symbol_mapping, name=self._options.name)

        elif self._backend == "NVVM":
            if target_type not in ("ptx", "ltoir"):
                raise ValueError(f'NVVM backend only supports target_type="ptx", "ltoir", got "{target_type}"')

            nvvm_options = self._translate_program_options_to_nvvm(self._options)
            if target_type == "ltoir" and "-gen-lto" not in nvvm_options:
                nvvm_options.append("-gen-lto")
            nvvm = _get_nvvm_module()
            with _nvvm_exception_manager(self):
                nvvm.verify_program(self._mnff.handle, len(nvvm_options), nvvm_options)
                nvvm.compile_program(self._mnff.handle, len(nvvm_options), nvvm_options)

            size = nvvm.get_compiled_result_size(self._mnff.handle)
            data = bytearray(size)
            nvvm.get_compiled_result(self._mnff.handle, data)

            if logs is not None:
                logsize = nvvm.get_program_log_size(self._mnff.handle)
                if logsize > 1:
                    log = bytearray(logsize)
                    nvvm.get_program_log(self._mnff.handle, log)
                    logs.write(log.decode("utf-8", errors="backslashreplace"))

            return ObjectCode._init(data, target_type, name=self._options.name)

        supported_backends = ("nvJitLink", "driver")
        if self._backend not in supported_backends:
            raise ValueError(f'Unsupported backend="{self._backend}" ({supported_backends=})')
        return self._linker.link(target_type)

    @property
    def backend(self) -> str:
        """Return this Program instance's underlying backend."""
        return self._backend

    @property
    def handle(self) -> ProgramHandleT:
        """Return the underlying handle object.

        .. note::

           The type of the returned object depends on the backend.

        .. caution::

            This handle is a Python object. To get the memory address of the underlying C
            handle, call ``int(Program.handle)``.
        """
        return self._mnff.handle
