# SPDX-FileCopyrightText: Copyright (c) 2024-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0
"""Compilation machinery for CUDA programs.

This module provides :class:`Program` for compiling source code into
:class:`~cuda.core.ObjectCode`, with :class:`ProgramOptions` for configuration.
"""

from __future__ import annotations

from dataclasses import dataclass
import threading
from warnings import warn

from cuda.bindings import driver, nvrtc

from libcpp.vector cimport vector

from ._resource_handles cimport (
    as_cu,
    as_py,
    create_nvrtc_program_handle,
    create_nvvm_program_handle,
)
from cuda.bindings cimport cynvrtc, cynvvm
from cuda.core._utils.cuda_utils cimport HANDLE_RETURN_NVRTC, HANDLE_RETURN_NVVM
from cuda.core._device import Device
from cuda.core._linker import Linker, LinkerHandleT, LinkerOptions
from cuda.core._module import ObjectCode
from cuda.core._utils.clear_error_support import assert_type
from cuda.core._utils.cuda_utils import (
    CUDAError,
    _handle_boolean_option,
    check_or_create_options,
    get_binding_version,
    handle_return,
    is_nested_sequence,
    is_sequence,
)

__all__ = ["Program", "ProgramOptions"]

ProgramHandleT = nvrtc.nvrtcProgram | int | LinkerHandleT
"""Type alias for program handle types across different backends.

The ``int`` type covers NVVM handles, which don't have a wrapper class.
"""


# =============================================================================
# Principal Class
# =============================================================================


cdef class Program:
    """Represent a compilation machinery to process programs into
    :class:`~cuda.core.ObjectCode`.

    This object provides a unified interface to multiple underlying
    compiler libraries. Compilation support is enabled for a wide
    range of code types and compilation types.

    Parameters
    ----------
    code : str | bytes | bytearray
        The source code to compile. For C++ and PTX, must be a string.
        For NVVM IR, can be str, bytes, or bytearray.
    code_type : str
        The type of source code. Must be one of ``"c++"``, ``"ptx"``, or ``"nvvm"``.
    options : :class:`ProgramOptions`, optional
        Options to customize the compilation process.
    """

    def __init__(self, code: str | bytes | bytearray, code_type: str, options: ProgramOptions | None = None):
        Program_init(self, code, code_type, options)

    def close(self):
        """Destroy this program."""
        if self._linker:
            self._linker.close()
        # Reset handles - the C++ shared_ptr destructor handles cleanup
        self._h_nvrtc.reset()
        self._h_nvvm.reset()

    def compile(
        self, target_type: str, name_expressions: tuple | list = (), logs = None
    ) -> ObjectCode:
        """Compile the program to the specified target type.

        Parameters
        ----------
        target_type : str
            The compilation target. Must be one of ``"ptx"``, ``"cubin"``, or ``"ltoir"``.
        name_expressions : tuple | list, optional
            Sequence of name expressions to make accessible in the compiled code.
            Used for template instantiation and similar cases.
        logs : object, optional
            Object with a ``write`` method to receive compilation logs.

        Returns
        -------
        :class:`~cuda.core.ObjectCode`
            The compiled object code.
        """
        return Program_compile(self, target_type, name_expressions, logs)

    @property
    def pch_status(self) -> str | None:
        """PCH creation outcome from the most recent :meth:`compile` call.

        Possible values:

        * ``"created"`` — PCH file was written successfully.
        * ``"not_attempted"`` — PCH creation was not attempted (e.g. the
          compiler decided not to, or automatic PCH processing skipped it).
        * ``"failed"`` — an error prevented PCH creation.
        * ``None`` — PCH was not requested, the program has not been
          compiled yet, the backend is not NVRTC (e.g. PTX or NVVM),
          or the NVRTC bindings are too old to report status.

        When ``create_pch`` is set in :class:`ProgramOptions` and the PCH
        heap is too small, :meth:`compile` automatically resizes the heap
        and retries, so ``"created"`` should be the common outcome.

        .. note::

           PCH is only supported for ``code_type="c++"`` programs that
           use the NVRTC backend. For PTX and NVVM programs this property
           always returns ``None``.
        """
        return self._pch_status

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
        if self._backend == "NVRTC":
            return as_py(self._h_nvrtc)
        elif self._backend == "NVVM":
            return as_py(self._h_nvvm)  # returns int (NVVM uses raw integers)
        else:
            return self._linker.handle

    def __repr__(self) -> str:
        return f"<Program backend='{self._backend}'>"


# =============================================================================
# Other Public Classes
# =============================================================================


@dataclass
class ProgramOptions:
    """Customizable options for configuring :class:`Program`.

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
        Enable device code optimization. When specified along with '-G', enables limited debug information generation
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
    no_cache : bool, optional
        Disable compiler caching.
        Default: False
    fdevice_time_trace : str, optional
        Generate time trace JSON for profiling compilation (NVRTC only).
        Default: None
    device_float128 : bool, optional
        Allow __float128 type in device code (NVRTC only).
        Default: False
    frandom_seed : str, optional
        Set random seed for randomized optimizations (NVRTC only).
        Default: None
    ofast_compile : str, optional
        Fast compilation mode: "0", "min", "mid", or "max" (NVRTC only).
        Default: None
    pch : bool, optional
        Use default precompiled header (NVRTC only, CUDA 12.8+).
        Default: False
    create_pch : str, optional
        Create precompiled header file (NVRTC only, CUDA 12.8+).
        Default: None
    use_pch : str, optional
        Use specific precompiled header file (NVRTC only, CUDA 12.8+).
        Default: None
    pch_dir : str, optional
        PCH directory location (NVRTC only, CUDA 12.8+).
        Default: None
    pch_verbose : bool, optional
        Verbose PCH output (NVRTC only, CUDA 12.8+).
        Default: False
    pch_messages : bool, optional
        Control PCH diagnostic messages (NVRTC only, CUDA 12.8+).
        Default: False
    instantiate_templates_in_pch : bool, optional
        Control template instantiation in PCH (NVRTC only, CUDA 12.8+).
        Default: False
    extra_sources : list of 2-tuples or tuple of 2-tuples, optional
        Additional NVVM IR modules to compile together with the main program, specified as
        ``((name1, source1), (name2, source2), ...)``. Each name is a string identifier used
        in diagnostic messages. Each source can be a string (textual LLVM IR) or bytes/bytearray
        (LLVM bitcode). Only supported for the NVVM backend.
        Default: None
    use_libdevice : bool, optional
        Load NVIDIA's `libdevice <https://docs.nvidia.com/cuda/libdevice-users-guide/>`_
        math builtins library. Only supported for the NVVM backend.
        Default: False
    """

    name: str | None = "default_program"
    arch: str | None = None
    relocatable_device_code: bool | None = None
    extensible_whole_program: bool | None = None
    debug: bool | None = None
    lineinfo: bool | None = None
    device_code_optimize: bool | None = None
    ptxas_options: str | list[str] | tuple[str] | None = None
    max_register_count: int | None = None
    ftz: bool | None = None
    prec_sqrt: bool | None = None
    prec_div: bool | None = None
    fma: bool | None = None
    use_fast_math: bool | None = None
    extra_device_vectorization: bool | None = None
    link_time_optimization: bool | None = None
    gen_opt_lto: bool | None = None
    define_macro: str | tuple[str, str] | list[str | tuple[str, str]] | tuple[str | tuple[str, str], ...] | None = None
    undefine_macro: str | list[str] | tuple[str] | None = None
    include_path: str | list[str] | tuple[str] | None = None
    pre_include: str | list[str] | tuple[str] | None = None
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
    diag_error: int | list[int] | tuple[int] | None = None
    diag_suppress: int | list[int] | tuple[int] | None = None
    diag_warn: int | list[int] | tuple[int] | None = None
    brief_diagnostics: bool | None = None
    time: str | None = None
    split_compile: int | None = None
    fdevice_syntax_only: bool | None = None
    minimal: bool | None = None
    no_cache: bool | None = None
    fdevice_time_trace: str | None = None
    device_float128: bool | None = None
    frandom_seed: str | None = None
    ofast_compile: str | None = None
    pch: bool | None = None
    create_pch: str | None = None
    use_pch: str | None = None
    pch_dir: str | None = None
    pch_verbose: bool | None = None
    pch_messages: bool | None = None
    instantiate_templates_in_pch: bool | None = None
    extra_sources: list[tuple[str, str | bytes | bytearray]] | tuple[tuple[str, str | bytes | bytearray], ...] | None = None
    use_libdevice: bool | None = None  # For libdevice execution
    numba_debug: bool | None = None  # Custom option for Numba debugging

    def __post_init__(self):
        self._name = self.name.encode()
        # Set arch to default if not provided
        if self.arch is None:
            self.arch = f"sm_{Device().arch}"

    def _prepare_nvrtc_options(self) -> list[bytes]:
        return _prepare_nvrtc_options_impl(self)

    def _prepare_nvvm_options(self, as_bytes: bool = True) -> list[bytes] | list[str]:
        return _prepare_nvvm_options_impl(self, as_bytes)

    def as_bytes(self, backend: str, target_type: str | None = None) -> list[bytes]:
        """Convert program options to bytes format for the specified backend.

        This method transforms the program options into a format suitable for the
        specified compiler backend. Different backends may use different option names
        and formats even for the same conceptual options.

        Parameters
        ----------
        backend : str
            The compiler backend to prepare options for. Must be either "nvrtc" or "nvvm".
        target_type : str, optional
            The compilation target type (e.g., "ptx", "cubin", "ltoir"). Some backends
            require additional options based on the target type.

        Returns
        -------
        list[bytes]
            List of option strings encoded as bytes.

        Raises
        ------
        ValueError
            If an unknown backend is specified.
        CUDAError
            If an option incompatible with the specified backend is set.

        Examples
        --------
        >>> options = ProgramOptions(arch="sm_80", debug=True)
        >>> nvrtc_options = options.as_bytes("nvrtc")
        """
        backend = backend.lower()
        if backend == "nvrtc":
            return self._prepare_nvrtc_options()
        elif backend == "nvvm":
            options = self._prepare_nvvm_options(as_bytes=True)
            if target_type == "ltoir" and b"-gen-lto" not in options:
                options.append(b"-gen-lto")
            return options
        else:
            raise ValueError(f"Unknown backend '{backend}'. Must be one of: 'nvrtc', 'nvvm'")

    def __repr__(self):
        return f"ProgramOptions(name={self.name!r}, arch={self.arch!r})"


# =============================================================================
# Private Classes and Helper Functions
# =============================================================================

# Module-level state for NVVM lazy loading
cdef object_nvvm_module = None
cdef bint _nvvm_import_attempted = False


def _get_nvvm_module():
    """Get the NVVM module, importing it lazily with availability checks."""
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

def _find_libdevice_path():
    """Find libdevice*.bc for NVVM compilation using cuda.pathfinder."""
    from cuda.pathfinder import find_bitcode_lib
    return find_bitcode_lib("device")




cdef inline bint _process_define_macro_inner(list options, object macro) except? -1:
    """Process a single define macro, returning True if successful."""
    if isinstance(macro, str):
        options.append(f"--define-macro={macro}")
        return True
    if isinstance(macro, tuple):
        if len(macro) != 2 or any(not isinstance(val, str) for val in macro):
            raise RuntimeError(f"Expected define_macro tuple[str, str], got {macro}")
        options.append(f"--define-macro={macro[0]}={macro[1]}")
        return True
    return False


cdef inline void _process_define_macro(list options, object macro) except *:
    """Process define_macro option which can be str, tuple, or list thereof."""
    union_type = "Union[str, tuple[str, str]]"
    if _process_define_macro_inner(options, macro):
        return
    if is_nested_sequence(macro):
        for seq_macro in macro:
            if not _process_define_macro_inner(options, seq_macro):
                raise RuntimeError(f"Expected define_macro {union_type}, got {seq_macro}")
        return
    raise RuntimeError(f"Expected define_macro {union_type}, list[{union_type}], got {macro}")


cpdef bint _can_load_generated_ptx() except? -1:
    """Check if the driver can load PTX generated by the current NVRTC version."""
    driver_ver = handle_return(driver.cuDriverGetVersion())
    nvrtc_major, nvrtc_minor = handle_return(nvrtc.nvrtcVersion())
    return nvrtc_major * 1000 + nvrtc_minor * 10 <= driver_ver


cdef inline object _translate_program_options(object options):
    """Translate ProgramOptions to LinkerOptions for PTX compilation."""
    return LinkerOptions(
        name=options.name,
        arch=options.arch,
        max_register_count=options.max_register_count,
        time=options.time,
        link_time_optimization=options.link_time_optimization,
        debug=options.debug,
        lineinfo=options.lineinfo,
        ftz=options.ftz,
        prec_div=options.prec_div,
        prec_sqrt=options.prec_sqrt,
        fma=options.fma,
        split_compile=options.split_compile,
        ptxas_options=options.ptxas_options,
        no_cache=options.no_cache,
    )


cdef inline int Program_init(Program self, object code, str code_type, object options) except -1:
    """Initialize a Program instance."""
    cdef cynvrtc.nvrtcProgram nvrtc_prog
    cdef cynvvm.nvvmProgram nvvm_prog
    cdef bytes code_bytes
    cdef const char* code_ptr
    cdef const char* name_ptr
    cdef size_t code_len
    cdef bytes module_bytes
    cdef const char* module_ptr
    cdef size_t module_len

    self._options = options = check_or_create_options(ProgramOptions, options, "Program options")
    code_type = code_type.lower()
    self._compile_lock = threading.Lock()
    self._use_libdevice = False
    self._libdevice_added = False

    self._pch_status = None

    if code_type == "c++":
        assert_type(code, str)
        if options.extra_sources is not None:
            raise ValueError("extra_sources is not supported by the NVRTC backend (C++ code_type)")

        # TODO: support pre-loaded headers & include names
        code_bytes = code.encode()
        code_ptr = <const char*>code_bytes
        name_ptr = <const char*>options._name

        with nogil:
            HANDLE_RETURN_NVRTC(NULL, cynvrtc.nvrtcCreateProgram(
                &nvrtc_prog, code_ptr, name_ptr, 0, NULL, NULL))
        self._h_nvrtc = create_nvrtc_program_handle(nvrtc_prog)
        self._nvrtc_code = code_bytes
        self._backend = "NVRTC"
        self._linker = None

    elif code_type == "ptx":
        assert_type(code, str)
        if options.extra_sources is not None:
            raise ValueError("extra_sources is not supported by the PTX backend.")
        self._linker = Linker(
            ObjectCode._init(code.encode(), code_type), options=_translate_program_options(options)
        )
        self._backend = self._linker.backend

    elif code_type == "nvvm":
        _get_nvvm_module()  # Validate NVVM availability
        if isinstance(code, str):
            code = code.encode("utf-8")
        elif not isinstance(code, (bytes, bytearray)):
            raise TypeError("NVVM IR code must be provided as str, bytes, or bytearray")

        code_ptr = <const char*>(<bytes>code)
        name_ptr = <const char*>options._name
        code_len = len(code)

        with nogil:
            HANDLE_RETURN_NVVM(NULL, cynvvm.nvvmCreateProgram(&nvvm_prog))
        self._h_nvvm = create_nvvm_program_handle(nvvm_prog)  # RAII from here
        with nogil:
            HANDLE_RETURN_NVVM(nvvm_prog, cynvvm.nvvmAddModuleToProgram(nvvm_prog, code_ptr, code_len, name_ptr))

        # Add extra modules if provided
        if options.extra_sources is not None:
            if not is_sequence(options.extra_sources):
                raise TypeError(
                    "extra_sources must be a sequence of 2-tuples: ((name1, source1), (name2, source2), ...)"
                )
            for i, module in enumerate(options.extra_sources):
                if not isinstance(module, tuple) or len(module) != 2:
                    raise TypeError(
                        f"Each extra module must be a 2-tuple (name, source)"
                        f", got {type(module).__name__} at index {i}"
                    )

                module_name, module_source = module

                if not isinstance(module_name, str):
                    raise TypeError(f"Module name at index {i} must be a string, got {type(module_name).__name__}")

                if isinstance(module_source, str):
                    # Textual LLVM IR - encode to UTF-8 bytes
                    module_source = module_source.encode("utf-8")
                elif not isinstance(module_source, (bytes, bytearray)):
                    raise TypeError(
                        f"Module source at index {i} must be str (textual LLVM IR), bytes (textual LLVM IR or bitcode), "
                        f"or bytearray, got {type(module_source).__name__}"
                    )

                if len(module_source) == 0:
                    raise ValueError(f"Module source for '{module_name}' (index {i}) cannot be empty")

                # Add the module using NVVM API
                module_bytes = module_source if isinstance(module_source, bytes) else bytes(module_source)
                module_ptr = <const char*>module_bytes
                module_len = len(module_bytes)
                module_name_bytes = module_name.encode()
                module_name_ptr = <const char*>module_name_bytes

                with nogil:
                    HANDLE_RETURN_NVVM(nvvm_prog, cynvvm.nvvmAddModuleToProgram(
                        nvvm_prog, module_ptr, module_len, module_name_ptr))

        # Store use_libdevice flag
        if options.use_libdevice:
            self._use_libdevice = True

        self._backend = "NVVM"
        self._linker = None

    else:
        supported_code_types = ("c++", "ptx", "nvvm")
        assert code_type not in supported_code_types, f"{code_type=}"
        if options.use_libdevice:
            raise ValueError("use_libdevice is only supported by the NVVM backend")
        raise RuntimeError(f"Unsupported {code_type=} ({supported_code_types=})")

    return 0


cdef object _nvrtc_compile_and_extract(
    cynvrtc.nvrtcProgram prog, str target_type, object name_expressions,
    object logs, list options_list, str name,
):
    """Run nvrtcCompileProgram on *prog* and extract the output.

    This is the inner compile+extract loop, factored out so the PCH
    auto-retry path can call it on a fresh program handle.
    """
    cdef size_t output_size = 0
    cdef size_t logsize = 0
    cdef vector[const char*] options_vec
    cdef char* data_ptr = NULL
    cdef bytes name_bytes
    cdef const char* name_ptr = NULL
    cdef const char* lowered_name = NULL
    cdef dict symbol_mapping = {}

    # Add name expressions before compilation
    if name_expressions:
        for n in name_expressions:
            name_bytes = n.encode() if isinstance(n, str) else n
            name_ptr = <const char*>name_bytes
            HANDLE_RETURN_NVRTC(prog, cynvrtc.nvrtcAddNameExpression(prog, name_ptr))

    # Build options array
    options_vec.resize(len(options_list))
    for i in range(len(options_list)):
        options_vec[i] = <const char*>(<bytes>options_list[i])

    # Compile
    with nogil:
        HANDLE_RETURN_NVRTC(prog, cynvrtc.nvrtcCompileProgram(prog, <int>options_vec.size(), options_vec.data()))

    # Get compiled output based on target type
    if target_type == "ptx":
        HANDLE_RETURN_NVRTC(prog, cynvrtc.nvrtcGetPTXSize(prog, &output_size))
        data = bytearray(output_size)
        data_ptr = <char*>(<bytearray>data)
        with nogil:
            HANDLE_RETURN_NVRTC(prog, cynvrtc.nvrtcGetPTX(prog, data_ptr))
    elif target_type == "cubin":
        HANDLE_RETURN_NVRTC(prog, cynvrtc.nvrtcGetCUBINSize(prog, &output_size))
        data = bytearray(output_size)
        data_ptr = <char*>(<bytearray>data)
        with nogil:
            HANDLE_RETURN_NVRTC(prog, cynvrtc.nvrtcGetCUBIN(prog, data_ptr))
    else:  # ltoir
        HANDLE_RETURN_NVRTC(prog, cynvrtc.nvrtcGetLTOIRSize(prog, &output_size))
        data = bytearray(output_size)
        data_ptr = <char*>(<bytearray>data)
        with nogil:
            HANDLE_RETURN_NVRTC(prog, cynvrtc.nvrtcGetLTOIR(prog, data_ptr))

    # Get lowered names after compilation
    if name_expressions:
        for n in name_expressions:
            name_bytes = n.encode() if isinstance(n, str) else n
            name_ptr = <const char*>name_bytes
            HANDLE_RETURN_NVRTC(prog, cynvrtc.nvrtcGetLoweredName(prog, name_ptr, &lowered_name))
            symbol_mapping[n] = lowered_name if lowered_name != NULL else None

    # Get compilation log if requested
    if logs is not None:
        HANDLE_RETURN_NVRTC(prog, cynvrtc.nvrtcGetProgramLogSize(prog, &logsize))
        if logsize > 1:
            log = bytearray(logsize)
            data_ptr = <char*>(<bytearray>log)
            with nogil:
                HANDLE_RETURN_NVRTC(prog, cynvrtc.nvrtcGetProgramLog(prog, data_ptr))
            logs.write(log.decode("utf-8", errors="backslashreplace"))

    return ObjectCode._init(bytes(data), target_type, symbol_mapping=symbol_mapping, name=name)


cdef int _nvrtc_pch_apis_cached = -1  # -1 = unchecked

cdef bint _has_nvrtc_pch_apis():
    global _nvrtc_pch_apis_cached
    if _nvrtc_pch_apis_cached < 0:
        _nvrtc_pch_apis_cached = hasattr(nvrtc, "nvrtcGetPCHCreateStatus")
    return _nvrtc_pch_apis_cached


cdef str _PCH_STATUS_CREATED = "created"
cdef str _PCH_STATUS_NOT_ATTEMPTED = "not_attempted"
cdef str _PCH_STATUS_FAILED = "failed"


cdef str _read_pch_status(cynvrtc.nvrtcProgram prog):
    """Query nvrtcGetPCHCreateStatus and translate to a high-level string."""
    cdef cynvrtc.nvrtcResult err
    with nogil:
        err = cynvrtc.nvrtcGetPCHCreateStatus(prog)
    if err == cynvrtc.nvrtcResult.NVRTC_SUCCESS:
        return _PCH_STATUS_CREATED
    if err == cynvrtc.nvrtcResult.NVRTC_ERROR_PCH_CREATE_HEAP_EXHAUSTED:
        return None  # sentinel: caller should auto-retry
    if err == cynvrtc.nvrtcResult.NVRTC_ERROR_NO_PCH_CREATE_ATTEMPTED:
        return _PCH_STATUS_NOT_ATTEMPTED
    return _PCH_STATUS_FAILED


cdef object Program_compile_nvrtc(Program self, str target_type, object name_expressions, object logs):
    """Compile using NVRTC backend and return ObjectCode."""
    cdef cynvrtc.nvrtcProgram prog = as_cu(self._h_nvrtc)
    cdef list options_list = self._options.as_bytes("nvrtc", target_type)

    result = _nvrtc_compile_and_extract(
        prog, target_type, name_expressions, logs, options_list, self._options.name,
    )

    cdef bint pch_creation_possible = self._options.create_pch or self._options.pch
    if not pch_creation_possible or not _has_nvrtc_pch_apis():
        self._pch_status = None
        return result

    try:
        status = _read_pch_status(prog)
    except RuntimeError as e:
        raise RuntimeError(
            "PCH was requested but the runtime libnvrtc does not support "
            "PCH APIs. Update to CUDA toolkit 12.8 or newer."
        ) from e

    if status is not None:
        self._pch_status = status
        return result

    # Heap exhausted — auto-resize and retry with a fresh program
    cdef size_t required = 0
    with nogil:
        HANDLE_RETURN_NVRTC(prog, cynvrtc.nvrtcGetPCHHeapSizeRequired(prog, &required))
        HANDLE_RETURN_NVRTC(NULL, cynvrtc.nvrtcSetPCHHeapSize(required))

    cdef cynvrtc.nvrtcProgram retry_prog
    cdef const char* code_ptr = <const char*>self._nvrtc_code
    cdef const char* name_ptr = <const char*>self._options._name
    with nogil:
        HANDLE_RETURN_NVRTC(NULL, cynvrtc.nvrtcCreateProgram(
            &retry_prog, code_ptr, name_ptr, 0, NULL, NULL))
    self._h_nvrtc = create_nvrtc_program_handle(retry_prog)

    result = _nvrtc_compile_and_extract(
        retry_prog, target_type, name_expressions, logs, options_list, self._options.name,
    )

    status = _read_pch_status(retry_prog)
    self._pch_status = status if status is not None else _PCH_STATUS_FAILED
    return result


cdef object Program_compile_nvvm(Program self, str target_type, object logs):
    """Compile using NVVM backend and return ObjectCode."""
    cdef cynvvm.nvvmProgram prog = as_cu(self._h_nvvm)
    cdef size_t output_size = 0
    cdef size_t logsize = 0
    cdef vector[const char*] options_vec
    cdef char* data_ptr = NULL
    cdef bytes libdevice_bytes
    cdef const char* libdevice_ptr
    cdef size_t libdevice_len
    # Build options array
    options_list = self._options.as_bytes("nvvm", target_type)
    options_vec.resize(len(options_list))
    for i in range(len(options_list)):
        options_vec[i] = <const char*>(<bytes>options_list[i])

    # Serialize NVVM program mutation/use per Program instance.
    with self._compile_lock:
        with nogil:
            HANDLE_RETURN_NVVM(prog, cynvvm.nvvmVerifyProgram(prog, <int>options_vec.size(), options_vec.data()))

        # Load libdevice if requested - following numba-cuda.
        if self._use_libdevice and not self._libdevice_added:
            libdevice_path = _find_libdevice_path()
            with open(libdevice_path, "rb") as f:
                libdevice_bytes = f.read()
            libdevice_ptr = <const char*>libdevice_bytes
            libdevice_len = len(libdevice_bytes)
            with nogil:
                HANDLE_RETURN_NVVM(prog, cynvvm.nvvmLazyAddModuleToProgram(
                    prog, libdevice_ptr, libdevice_len, NULL))
            self._libdevice_added = True

        with nogil:
            HANDLE_RETURN_NVVM(prog, cynvvm.nvvmCompileProgram(prog, <int>options_vec.size(), options_vec.data()))

        HANDLE_RETURN_NVVM(prog, cynvvm.nvvmGetCompiledResultSize(prog, &output_size))
        data = bytearray(output_size)
        data_ptr = <char*>(<bytearray>data)
        with nogil:
            HANDLE_RETURN_NVVM(prog, cynvvm.nvvmGetCompiledResult(prog, data_ptr))

        # Get compilation log if requested
        if logs is not None:
            HANDLE_RETURN_NVVM(prog, cynvvm.nvvmGetProgramLogSize(prog, &logsize))
            if logsize > 1:
                log = bytearray(logsize)
                data_ptr = <char*>(<bytearray>log)
                with nogil:
                    HANDLE_RETURN_NVVM(prog, cynvvm.nvvmGetProgramLog(prog, data_ptr))
                logs.write(log.decode("utf-8", errors="backslashreplace"))

        return ObjectCode._init(bytes(data), target_type, name=self._options.name)

# Supported target types per backend
cdef dict SUPPORTED_TARGETS = {
    "NVRTC": ("ptx", "cubin", "ltoir"),
    "NVVM": ("ptx", "ltoir"),
    "nvJitLink": ("cubin", "ptx"),
    "driver": ("cubin", "ptx"),
}


cdef object Program_compile(Program self, str target_type, object name_expressions, object logs):
    """Compile the program to the specified target type."""
    # Validate target_type for this backend
    supported = SUPPORTED_TARGETS.get(self._backend)
    if supported is None:
        raise ValueError(f'Unknown backend="{self._backend}"')
    if target_type not in supported:
        raise ValueError(
            f'Unsupported target_type="{target_type}" for {self._backend} '
            f'(supported: {", ".join(repr(t) for t in supported)})'
        )

    if self._backend == "NVRTC":
        if target_type == "ptx" and not _can_load_generated_ptx():
            warn(
                "The CUDA driver version is older than the backend version. "
                "The generated ptx will not be loadable by the current driver.",
                stacklevel=2,
                category=RuntimeWarning,
            )
        return Program_compile_nvrtc(self, target_type, name_expressions, logs)

    elif self._backend == "NVVM":
        return Program_compile_nvvm(self, target_type, logs)

    else:
        return self._linker.link(target_type)


cdef inline list _prepare_nvrtc_options_impl(object opts):
    """Build NVRTC-specific compiler options."""
    options = [f"-arch={opts.arch}"]
    if opts.relocatable_device_code is not None:
        options.append(f"--relocatable-device-code={_handle_boolean_option(opts.relocatable_device_code)}")
    if opts.extensible_whole_program is not None and opts.extensible_whole_program:
        options.append("--extensible-whole-program")
    if opts.debug is not None and opts.debug:
        options.append("--device-debug")
    if opts.lineinfo is not None and opts.lineinfo:
        options.append("--generate-line-info")
    if opts.device_code_optimize is not None and opts.device_code_optimize:
        options.append("--dopt=on")
    if opts.ptxas_options is not None:
        opt_name = "--ptxas-options"
        if isinstance(opts.ptxas_options, str):
            options.append(f"{opt_name}={opts.ptxas_options}")
        elif is_sequence(opts.ptxas_options):
            for opt_value in opts.ptxas_options:
                options.append(f"{opt_name}={opt_value}")
    if opts.max_register_count is not None:
        options.append(f"--maxrregcount={opts.max_register_count}")
    if opts.ftz is not None:
        options.append(f"--ftz={_handle_boolean_option(opts.ftz)}")
    if opts.prec_sqrt is not None:
        options.append(f"--prec-sqrt={_handle_boolean_option(opts.prec_sqrt)}")
    if opts.prec_div is not None:
        options.append(f"--prec-div={_handle_boolean_option(opts.prec_div)}")
    if opts.fma is not None:
        options.append(f"--fmad={_handle_boolean_option(opts.fma)}")
    if opts.use_fast_math is not None and opts.use_fast_math:
        options.append("--use_fast_math")
    if opts.extra_device_vectorization is not None and opts.extra_device_vectorization:
        options.append("--extra-device-vectorization")
    if opts.link_time_optimization is not None and opts.link_time_optimization:
        options.append("--dlink-time-opt")
    if opts.gen_opt_lto is not None and opts.gen_opt_lto:
        options.append("--gen-opt-lto")
    if opts.define_macro is not None:
        _process_define_macro(options, opts.define_macro)
    if opts.undefine_macro is not None:
        if isinstance(opts.undefine_macro, str):
            options.append(f"--undefine-macro={opts.undefine_macro}")
        elif is_sequence(opts.undefine_macro):
            for macro in opts.undefine_macro:
                options.append(f"--undefine-macro={macro}")
    if opts.include_path is not None:
        if isinstance(opts.include_path, str):
            options.append(f"--include-path={opts.include_path}")
        elif is_sequence(opts.include_path):
            for path in opts.include_path:
                options.append(f"--include-path={path}")
    if opts.pre_include is not None:
        if isinstance(opts.pre_include, str):
            options.append(f"--pre-include={opts.pre_include}")
        elif is_sequence(opts.pre_include):
            for header in opts.pre_include:
                options.append(f"--pre-include={header}")
    if opts.no_source_include is not None and opts.no_source_include:
        options.append("--no-source-include")
    if opts.std is not None:
        options.append(f"--std={opts.std}")
    if opts.builtin_move_forward is not None:
        options.append(f"--builtin-move-forward={_handle_boolean_option(opts.builtin_move_forward)}")
    if opts.builtin_initializer_list is not None:
        options.append(f"--builtin-initializer-list={_handle_boolean_option(opts.builtin_initializer_list)}")
    if opts.disable_warnings is not None and opts.disable_warnings:
        options.append("--disable-warnings")
    if opts.restrict is not None and opts.restrict:
        options.append("--restrict")
    if opts.device_as_default_execution_space is not None and opts.device_as_default_execution_space:
        options.append("--device-as-default-execution-space")
    if opts.device_int128 is not None and opts.device_int128:
        options.append("--device-int128")
    if opts.device_float128 is not None and opts.device_float128:
        options.append("--device-float128")
    if opts.optimization_info is not None:
        options.append(f"--optimization-info={opts.optimization_info}")
    if opts.no_display_error_number is not None and opts.no_display_error_number:
        options.append("--no-display-error-number")
    if opts.diag_error is not None:
        if isinstance(opts.diag_error, int):
            options.append(f"--diag-error={opts.diag_error}")
        elif is_sequence(opts.diag_error):
            for error in opts.diag_error:
                options.append(f"--diag-error={error}")
    if opts.diag_suppress is not None:
        if isinstance(opts.diag_suppress, int):
            options.append(f"--diag-suppress={opts.diag_suppress}")
        elif is_sequence(opts.diag_suppress):
            for suppress in opts.diag_suppress:
                options.append(f"--diag-suppress={suppress}")
    if opts.diag_warn is not None:
        if isinstance(opts.diag_warn, int):
            options.append(f"--diag-warn={opts.diag_warn}")
        elif is_sequence(opts.diag_warn):
            for w in opts.diag_warn:
                options.append(f"--diag-warn={w}")
    if opts.brief_diagnostics is not None:
        options.append(f"--brief-diagnostics={_handle_boolean_option(opts.brief_diagnostics)}")
    if opts.time is not None:
        options.append(f"--time={opts.time}")
    if opts.split_compile is not None:
        options.append(f"--split-compile={opts.split_compile}")
    if opts.fdevice_syntax_only is not None and opts.fdevice_syntax_only:
        options.append("--fdevice-syntax-only")
    if opts.minimal is not None and opts.minimal:
        options.append("--minimal")
    if opts.no_cache is not None and opts.no_cache:
        options.append("--no-cache")
    if opts.fdevice_time_trace is not None:
        options.append(f"--fdevice-time-trace={opts.fdevice_time_trace}")
    if opts.frandom_seed is not None:
        options.append(f"--frandom-seed={opts.frandom_seed}")
    if opts.ofast_compile is not None:
        options.append(f"--Ofast-compile={opts.ofast_compile}")
    # PCH options (CUDA 12.8+)
    if opts.pch is not None and opts.pch:
        options.append("--pch")
    if opts.create_pch is not None:
        options.append(f"--create-pch={opts.create_pch}")
    if opts.use_pch is not None:
        options.append(f"--use-pch={opts.use_pch}")
    if opts.pch_dir is not None:
        options.append(f"--pch-dir={opts.pch_dir}")
    if opts.pch_verbose is not None:
        options.append(f"--pch-verbose={_handle_boolean_option(opts.pch_verbose)}")
    if opts.pch_messages is not None:
        options.append(f"--pch-messages={_handle_boolean_option(opts.pch_messages)}")
    if opts.instantiate_templates_in_pch is not None:
        options.append(
            f"--instantiate-templates-in-pch={_handle_boolean_option(opts.instantiate_templates_in_pch)}"
        )
    if opts.numba_debug:
        options.append("--numba-debug")
    return [o.encode() for o in options]


cdef inline object _prepare_nvvm_options_impl(object opts, bint as_bytes):
    """Build NVVM-specific compiler options."""
    options = []

    # Options supported by NVVM
    assert opts.arch is not None
    arch = opts.arch
    if arch.startswith("sm_"):
        arch = f"compute_{arch[3:]}"
    options.append(f"-arch={arch}")
    if opts.debug is not None and opts.debug:
        options.append("-g")
    if opts.device_code_optimize is False:
        options.append("-opt=0")
    elif opts.device_code_optimize is True:
        options.append("-opt=3")
    # NVVM uses 0/1 instead of true/false for boolean options
    if opts.ftz is not None:
        options.append(f"-ftz={'1' if opts.ftz else '0'}")
    if opts.prec_sqrt is not None:
        options.append(f"-prec-sqrt={'1' if opts.prec_sqrt else '0'}")
    if opts.prec_div is not None:
        options.append(f"-prec-div={'1' if opts.prec_div else '0'}")
    if opts.fma is not None:
        options.append(f"-fma={'1' if opts.fma else '0'}")

    # Check for unsupported options and raise error if they are set
    unsupported = []
    if opts.relocatable_device_code is not None:
        unsupported.append("relocatable_device_code")
    if opts.extensible_whole_program is not None and opts.extensible_whole_program:
        unsupported.append("extensible_whole_program")
    if opts.lineinfo is not None and opts.lineinfo:
        unsupported.append("lineinfo")
    if opts.ptxas_options is not None:
        unsupported.append("ptxas_options")
    if opts.max_register_count is not None:
        unsupported.append("max_register_count")
    if opts.use_fast_math is not None and opts.use_fast_math:
        unsupported.append("use_fast_math")
    if opts.extra_device_vectorization is not None and opts.extra_device_vectorization:
        unsupported.append("extra_device_vectorization")
    if opts.gen_opt_lto is not None and opts.gen_opt_lto:
        unsupported.append("gen_opt_lto")
    if opts.define_macro is not None:
        unsupported.append("define_macro")
    if opts.undefine_macro is not None:
        unsupported.append("undefine_macro")
    if opts.include_path is not None:
        unsupported.append("include_path")
    if opts.pre_include is not None:
        unsupported.append("pre_include")
    if opts.no_source_include is not None and opts.no_source_include:
        unsupported.append("no_source_include")
    if opts.std is not None:
        unsupported.append("std")
    if opts.builtin_move_forward is not None:
        unsupported.append("builtin_move_forward")
    if opts.builtin_initializer_list is not None:
        unsupported.append("builtin_initializer_list")
    if opts.disable_warnings is not None and opts.disable_warnings:
        unsupported.append("disable_warnings")
    if opts.restrict is not None and opts.restrict:
        unsupported.append("restrict")
    if opts.device_as_default_execution_space is not None and opts.device_as_default_execution_space:
        unsupported.append("device_as_default_execution_space")
    if opts.device_int128 is not None and opts.device_int128:
        unsupported.append("device_int128")
    if opts.optimization_info is not None:
        unsupported.append("optimization_info")
    if opts.no_display_error_number is not None and opts.no_display_error_number:
        unsupported.append("no_display_error_number")
    if opts.diag_error is not None:
        unsupported.append("diag_error")
    if opts.diag_suppress is not None:
        unsupported.append("diag_suppress")
    if opts.diag_warn is not None:
        unsupported.append("diag_warn")
    if opts.brief_diagnostics is not None:
        unsupported.append("brief_diagnostics")
    if opts.time is not None:
        unsupported.append("time")
    if opts.split_compile is not None:
        unsupported.append("split_compile")
    if opts.fdevice_syntax_only is not None and opts.fdevice_syntax_only:
        unsupported.append("fdevice_syntax_only")
    if opts.minimal is not None and opts.minimal:
        unsupported.append("minimal")
    if opts.numba_debug is not None and opts.numba_debug:
        unsupported.append("numba_debug")
    if unsupported:
        raise CUDAError(f"The following options are not supported by NVVM backend: {', '.join(unsupported)}")

    if as_bytes:
        return [o.encode() for o in options]
    else:
        return options
