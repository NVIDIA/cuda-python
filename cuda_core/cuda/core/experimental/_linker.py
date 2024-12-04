# Copyright (c) 2024, NVIDIA CORPORATION & AFFILIATES. ALL RIGHTS RESERVED.
#
# SPDX-License-Identifier: LicenseRef-NVIDIA-SOFTWARE-LICENSE

import ctypes
import weakref
from dataclasses import dataclass
from typing import List, Optional

from cuda import cuda
from cuda.core.experimental._module import ObjectCode
from cuda.core.experimental._utils import check_or_create_options, handle_return

# TODO: revisit this treatment for py313t builds
_driver = None  # populated if nvJitLink cannot be used
_driver_input_types = None  # populated if nvJitLink cannot be used
_driver_ver = None
_inited = False
_nvjitlink = None  # populated if nvJitLink can be used
_nvjitlink_input_types = None  # populated if nvJitLink cannot be used


def _lazy_init():
    global _inited
    if _inited:
        return

    global _driver, _driver_input_types, _driver_ver, _nvjitlink, _nvjitlink_input_types
    _driver_ver = handle_return(cuda.cuDriverGetVersion())
    _driver_ver = (_driver_ver // 1000, (_driver_ver % 1000) // 10)
    try:
        raise ImportError
        from cuda.bindings import nvjitlink
        from cuda.bindings._internal import nvjitlink as inner_nvjitlink
    except ImportError:
        # binding is not available
        nvjitlink = None
    else:
        if inner_nvjitlink._inspect_function_pointer("__nvJitLinkVersion") == 0:
            # binding is available, but nvJitLink is not installed
            nvjitlink = None
        elif _driver_ver > nvjitlink.version():
            # TODO: nvJitLink is not new enough, warn?
            pass
    if nvjitlink:
        _nvjitlink = nvjitlink
        _nvjitlink_input_types = {
            "ptx": _nvjitlink.InputType.PTX,
            "cubin": _nvjitlink.InputType.CUBIN,
            "fatbin": _nvjitlink.InputType.FATBIN,
            "ltoir": _nvjitlink.InputType.LTOIR,
            "object": _nvjitlink.InputType.OBJECT,
        }
    else:
        from cuda import cuda as _driver

        _driver_input_types = {
            "ptx": _driver.CUjitInputType.CU_JIT_INPUT_PTX,
            "cubin": _driver.CUjitInputType.CU_JIT_INPUT_CUBIN,
            "fatbin": _driver.CUjitInputType.CU_JIT_INPUT_FATBINARY,
            "object": _driver.CUjitInputType.CU_JIT_INPUT_OBJECT,
        }
    _inited = True


@dataclass
class LinkerOptions:
    """Customizable :obj:`LinkerOptions` for nvJitLink.

    Attributes
    ----------
    arch : str
        Pass SM architecture value. Can use compute_<N> value instead if only generating PTX.
        This is a required option.
        Acceptable value type: str
        Maps to: -arch=sm_<N>
    max_register_count : int, optional
        Maximum register count.
        Default: None
        Acceptable value type: int
        Maps to: -maxrregcount=<N>
    time : bool, optional
        Print timing information to InfoLog.
        Default: False
        Acceptable value type: bool
        Maps to: -time
    verbose : bool, optional
        Print verbose messages to InfoLog.
        Default: False
        Acceptable value type: bool
        Maps to: -verbose
    link_time_optimization : bool, optional
        Perform link time optimization.
        Default: False
        Acceptable value type: bool
        Maps to: -lto
    ptx : bool, optional
        Emit PTX after linking instead of CUBIN; only supported with -lto.
        Default: False
        Acceptable value type: bool
        Maps to: -ptx
    optimization_level : int, optional
        Set optimization level. Only 0 and 3 are accepted.
        Default: None
        Acceptable value type: int
        Maps to: -O<N>
    debug : bool, optional
        Generate debug information.
        Default: False
        Acceptable value type: bool
        Maps to: -g
    lineinfo : bool, optional
        Generate line information.
        Default: False
        Acceptable value type: bool
        Maps to: -lineinfo
    ftz : bool, optional
        Flush denormal values to zero.
        Default: False
        Acceptable value type: bool
        Maps to: -ftz=<n>
    prec_div : bool, optional
        Use precise division.
        Default: True
        Acceptable value type: bool
        Maps to: -prec-div=<n>
    prec_sqrt : bool, optional
        Use precise square root.
        Default: True
        Acceptable value type: bool
        Maps to: -prec-sqrt=<n>
    fma : bool, optional
        Use fast multiply-add.
        Default: True
        Acceptable value type: bool
        Maps to: -fma=<n>
    kernels_used : List[str], optional
        Pass list of kernels that are used; any not in the list can be removed. This option can be specified multiple
        times.
        Default: None
        Acceptable value type: list of str
        Maps to: -kernels-used=<name>
    variables_used : List[str], optional
        Pass list of variables that are used; any not in the list can be removed. This option can be specified multiple
        times.
        Default: None
        Acceptable value type: list of str
        Maps to: -variables-used=<name>
    optimize_unused_variables : bool, optional
        Assume that if a variable is not referenced in device code, it can be removed.
        Default: False
        Acceptable value type: bool
        Maps to: -optimize-unused-variables
    xptxas : List[str], optional
        Pass options to PTXAS. This option can be called multiple times.
        Default: None
        Acceptable value type: list of str
        Maps to: -Xptxas=<opt>
    split_compile : int, optional
        Split compilation maximum thread count. Use 0 to use all available processors. Value of 1 disables split
        compilation (default).
        Default: 1
        Acceptable value type: int
        Maps to: -split-compile=<N>
    split_compile_extended : int, optional
        A more aggressive form of split compilation available in LTO mode only. Accepts a maximum thread count value.
        Use 0 to use all available processors. Value of 1 disables extended split compilation (default). Note: This
        option can potentially impact performance of the compiled binary.
        Default: 1
        Acceptable value type: int
        Maps to: -split-compile-extended=<N>
    no_cache : bool, optional
        Do not cache the intermediate steps of nvJitLink.
        Default: False
        Acceptable value type: bool
        Maps to: -no-cache
    """

    arch: str
    max_register_count: Optional[int] = None
    time: Optional[bool] = None
    verbose: Optional[bool] = None
    link_time_optimization: Optional[bool] = None
    ptx: Optional[bool] = None
    optimization_level: Optional[int] = None
    debug: Optional[bool] = None
    lineinfo: Optional[bool] = None
    ftz: Optional[bool] = None
    prec_div: Optional[bool] = None
    prec_sqrt: Optional[bool] = None
    fma: Optional[bool] = None
    kernels_used: Optional[List[str]] = None
    variables_used: Optional[List[str]] = None
    optimize_unused_variables: Optional[bool] = None
    xptxas: Optional[List[str]] = None
    split_compile: Optional[int] = None
    split_compile_extended: Optional[int] = None
    no_cache: Optional[bool] = None

    def __post_init__(self):
        _lazy_init()
        self.formatted_options = []
        if _nvjitlink:
            self._init_nvjitlink()
        else:
            self._init_driver()

    def _init_nvjitlink(self):
        if self.arch is not None:
            self.formatted_options.append(f"-arch={self.arch}")
        if self.max_register_count is not None:
            self.formatted_options.append(f"-maxrregcount={self.max_register_count}")
        if self.time is not None:
            self.formatted_options.append("-time")
        if self.verbose is not None:
            self.formatted_options.append("-verbose")
        if self.link_time_optimization is not None:
            self.formatted_options.append("-lto")
        if self.ptx is not None:
            self.formatted_options.append("-ptx")
        if self.optimization_level is not None:
            self.formatted_options.append(f"-O{self.optimization_level}")
        if self.debug is not None:
            self.formatted_options.append("-g")
        if self.lineinfo is not None:
            self.formatted_options.append("-lineinfo")
        if self.ftz is not None:
            self.formatted_options.append(f"-ftz={'true' if self.ftz else 'false'}")
        if self.prec_div is not None:
            self.formatted_options.append(f"-prec-div={'true' if self.prec_div else 'false'}")
        if self.prec_sqrt is not None:
            self.formatted_options.append(f"-prec-sqrt={'true' if self.prec_sqrt else 'false'}")
        if self.fma is not None:
            self.formatted_options.append(f"-fma={'true' if self.fma else 'false'}")
        if self.kernels_used is not None:
            for kernel in self.kernels_used:
                self.formatted_options.append(f"-kernels-used={kernel}")
        if self.variables_used is not None:
            for variable in self.variables_used:
                self.formatted_options.append(f"-variables-used={variable}")
        if self.optimize_unused_variables is not None:
            self.formatted_options.append("-optimize-unused-variables")
        if self.xptxas is not None:
            for opt in self.xptxas:
                self.formatted_options.append(f"-Xptxas={opt}")
        if self.split_compile is not None:
            self.formatted_options.append(f"-split-compile={self.split_compile}")
        if self.split_compile_extended is not None:
            self.formatted_options.append(f"-split-compile-extended={self.split_compile_extended}")
        if self.no_cache is not None:
            self.formatted_options.append("-no-cache")

    def _init_driver(self):
        self.option_keys = []
        # allocate 4 KiB each for info/error logs
        size = 4194304
        self.formatted_options.extend((bytearray(size), size, bytearray(size), size))
        self.option_keys.extend(
            (
                _driver.CUjit_option.CU_JIT_INFO_LOG_BUFFER,
                _driver.CUjit_option.CU_JIT_INFO_LOG_BUFFER_SIZE_BYTES,
                _driver.CUjit_option.CU_JIT_ERROR_LOG_BUFFER,
                _driver.CUjit_option.CU_JIT_ERROR_LOG_BUFFER_SIZE_BYTES,
            )
        )

        if self.arch is not None:
            arch = self.arch.split("_")[-1].upper()
            self.formatted_options.append(getattr(_driver.CUjit_target, f"CU_TARGET_COMPUTE_{arch}"))
            self.option_keys.append(_driver.CUjit_option.CU_JIT_TARGET)
        if self.max_register_count is not None:
            self.formatted_options.append(self.max_register_count)
            self.option_keys.append(_driver.CUjit_option.CU_JIT_MAX_REGISTERS)
        if self.time is not None:
            self.formatted_options.append(1)  # ctypes.c_int32(1)
            self.option_keys.append(_driver.CUjit_option.CU_JIT_WALL_TIME)
        if self.verbose is not None:
            self.formatted_options.append(1)  # ctypes.c_int32(1)
            self.option_keys.append(_driver.CUjit_option.CU_JIT_LOG_VERBOSE)
        if self.link_time_optimization is not None:
            self.formatted_options.append(1)  # ctypes.c_int32(1)
            self.option_keys.append(_driver.CUjit_option.CU_JIT_LTO)
        if self.ptx is not None:
            self.formatted_options.append(1)  # ctypes.c_int32(1)
            self.option_keys.append(_driver.CUjit_option.CU_JIT_GENERATE_LINE_INFO)
        if self.optimization_level is not None:
            self.formatted_options.append(self.optimization_level)
            self.option_keys.append(_driver.CUjit_option.CU_JIT_OPTIMIZATION_LEVEL)
        if self.debug is not None:
            self.formatted_options.append(1)  # ctypes.c_int32(1)
            self.option_keys.append(_driver.CUjit_option.CU_JIT_GENERATE_DEBUG_INFO)
        if self.lineinfo is not None:
            self.formatted_options.append(1)  # ctypes.c_int32(1)
            self.option_keys.append(_driver.CUjit_option.CU_JIT_GENERATE_LINE_INFO)
        if self.ftz is not None:
            self.formatted_options.append(1 if self.ftz else 0)
            self.option_keys.append(_driver.CUjit_option.CU_JIT_FTZ)
        if self.prec_div is not None:
            self.formatted_options.append(1 if self.prec_div else 0)
            self.option_keys.append(_driver.CUjit_option.CU_JIT_PREC_DIV)
        if self.prec_sqrt is not None:
            self.formatted_options.append(1 if self.prec_sqrt else 0)
            self.option_keys.append(_driver.CUjit_option.CU_JIT_PREC_SQRT)
        if self.fma is not None:
            self.formatted_options.append(1 if self.fma else 0)
            self.option_keys.append(_driver.CUjit_option.CU_JIT_FMA)
        if self.kernels_used is not None:
            for kernel in self.kernels_used:
                self.formatted_options.append(kernel.encode())
                self.option_keys.append(_driver.CUjit_option.CU_JIT_REFERENCED_KERNEL_NAMES)
        if self.variables_used is not None:
            for variable in self.variables_used:
                self.formatted_options.append(variable.encode())
                self.option_keys.append(_driver.CUjit_option.CU_JIT_REFERENCED_VARIABLE_NAMES)
        if self.optimize_unused_variables is not None:
            self.formatted_options.append(1)  # ctypes.c_int32(1)
            self.option_keys.append(_driver.CUjit_option.CU_JIT_OPTIMIZE_UNUSED_DEVICE_VARIABLES)
        if self.xptxas is not None:
            for opt in self.xptxas:
                raise NotImplementedError("TODO: implement xptxas option")
        if self.split_compile_extended is not None:
            self.formatted_options.append(self.split_compile_extended)
            self.option_keys.append(_driver.CUjit_option.CU_JIT_MIN_CTA_PER_SM)
        if self.no_cache is not None:
            self.formatted_options.append(1)  # ctypes.c_int32(1)
            self.option_keys.append(_driver.CUjit_option.CU_JIT_CACHE_MODE)


class Linker:
    """
    Linker class for managing the linking of object codes with specified options.

    Parameters
    ----------
    object_codes : ObjectCode
        One or more ObjectCode objects to be linked.
    options : LinkerOptions, optional
        Options for the linker. If not provided, default options will be used.
    """

    class _MembersNeededForFinalize:
        __slots__ = ("handle", "use_nvjitlink")

        def __init__(self, program_obj, handle, use_nvjitlink):
            self.handle = handle
            self.use_nvjitlink = use_nvjitlink
            weakref.finalize(program_obj, self.close)

        def close(self):
            if self.handle is not None:
                if self.use_nvjitlink:
                    _nvjitlink.destroy(self.handle)
                else:
                    handle_return(_driver.cuLinkDestroy(self.handle))
                self.handle = None

    __slots__ = ("__weakref__", "_mnff", "_options")

    def __init__(self, *object_codes: ObjectCode, options: LinkerOptions = None):
        if len(object_codes) == 0:
            raise ValueError("At least one ObjectCode object must be provided")

        self._options = options = check_or_create_options(LinkerOptions, options, "Linker options")
        if _nvjitlink:
            handle = _nvjitlink.create(len(options.formatted_options), options.formatted_options)
            use_nvjitlink = True
        else:
            handle = handle_return(
                _driver.cuLinkCreate(len(options.formatted_options), options.option_keys, options.formatted_options)
            )
            use_nvjitlink = False
        self._mnff = Linker._MembersNeededForFinalize(self, handle, use_nvjitlink)

        for code in object_codes:
            assert isinstance(code, ObjectCode)
            self._add_code_object(code)

    def _add_code_object(self, object_code: ObjectCode):
        data = object_code._module
        assert isinstance(data, bytes)
        if _nvjitlink:
            _nvjitlink.add_data(
                self._mnff.handle,
                self._input_type_from_code_type(object_code._code_type),
                data,
                len(data),
                f"{object_code._handle}_{object_code._code_type}",
            )
        else:
            handle_return(
                _driver.cuLinkAddData(
                    self._mnff.handle,
                    self._input_type_from_code_type(object_code._code_type),
                    data,
                    len(data),
                    f"{object_code._handle}_{object_code._code_type}".encode(),
                    0,
                    None,
                    None,
                )
            )

    def link(self, target_type) -> ObjectCode:
        if target_type not in ("cubin", "ptx"):
            raise ValueError(f"Unsupported target type: {target_type}")
        if _nvjitlink:
            _nvjitlink.complete(self._mnff.handle)
            if target_type == "cubin":
                get_size = _nvjitlink.get_linked_cubin_size
                get_code = _nvjitlink.get_linked_cubin
            else:
                get_size = _nvjitlink.get_linked_ptx_size
                get_code = _nvjitlink.get_linked_ptx

            size = get_size(self._mnff.handle)
            code = bytearray(size)
            get_code(self._mnff.handle, code)
        else:
            addr, size = handle_return(_driver.cuLinkComplete(self._mnff.handle))
            code = (ctypes.c_char * size).from_address(addr)

        return ObjectCode(bytes(code), target_type)

    def get_error_log(self) -> str:
        if _nvjitlink:
            log_size = _nvjitlink.get_error_log_size(self._mnff.handle)
            log = bytearray(log_size)
            _nvjitlink.get_error_log(self._mnff.handle, log)
        else:
            log = self._options.formatted_options[2]
        return log.decode()

    def get_info_log(self) -> str:
        if _nvjitlink:
            log_size = _nvjitlink.get_info_log_size(self._mnff.handle)
            log = bytearray(log_size)
            _nvjitlink.get_info_log(self._mnff.handle, log)
        else:
            log = self._options.formatted_options[0]
        return log.decode()

    def _input_type_from_code_type(self, code_type: str):
        # this list is based on the supported values for code_type in the ObjectCode class definition.
        # nvJitLink/driver support other options for input type
        input_type = _nvjitlink_input_types.get(code_type) if _nvjitlink else _driver_input_types.get(code_type)

        if input_type is None:
            raise ValueError(f"Unknown code_type associated with ObjectCode: {code_type}")
        return input_type

    @property
    def handle(self) -> int:
        return self._mnff.handle

    def close(self):
        self._mnff.close()