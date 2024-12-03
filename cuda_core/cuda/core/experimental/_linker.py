# Copyright (c) 2024, NVIDIA CORPORATION & AFFILIATES. ALL RIGHTS RESERVED.
#
# SPDX-License-Identifier: LicenseRef-NVIDIA-SOFTWARE-LICENSE

import weakref
from dataclasses import dataclass
from typing import List, Optional

from cuda.bindings import nvjitlink
from cuda.core.experimental._module import ObjectCode
from cuda.core.experimental._utils import check_or_create_options


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
        self.formatted_options = []
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


class Linker:
    """
    Linker class for managing the linking of object codes with specified options.

    Parameters
    ----------
    object_codes : ObjectCode
        One or more ObjectCode objects to be linked.
    options : LinkerOptions, optional
        Options for the linker. If not provided, default options will be used.

    Attributes
    ----------
    _options : LinkerOptions
        The options used for the linker.
    _handle : handle
        The handle to the linker created by nvjitlink.

    Methods
    -------
    _add_code_object(object_code)
        Adds an object code to the linker.
    close()
        Closes the linker and releases resources.
    """

    class _MembersNeededForFinalize:
        __slots__ = ("handle",)

        def __init__(self, program_obj, handle):
            self.handle = handle
            weakref.finalize(program_obj, self.close)

        def close(self):
            if self.handle is not None:
                nvjitlink.destroy(self.handle)
                self.handle = None

    __slots__ = ("__weakref__", "_mnff", "_options")

    def __init__(self, *object_codes: ObjectCode, options: LinkerOptions = None):
        self._options = options = check_or_create_options(LinkerOptions, options, "Linker options")
        self._mnff = Linker._MembersNeededForFinalize(
            self, nvjitlink.create(len(options.formatted_options), options.formatted_options)
        )

        if len(object_codes) == 0:
            raise ValueError("At least one ObjectCode object must be provided")

        for code in object_codes:
            assert isinstance(code, ObjectCode)
            self._add_code_object(code)

    def _add_code_object(self, object_code: ObjectCode):
        data = object_code._module
        assert isinstance(data, bytes)
        nvjitlink.add_data(
            self._mnff.handle,
            self._input_type_from_code_type(object_code._code_type),
            data,
            len(data),
            f"{object_code._handle}_{object_code._code_type}",
        )

    _get_linked_methods = {
        "cubin": (nvjitlink.get_linked_cubin_size, nvjitlink.get_linked_cubin),
        "ptx": (nvjitlink.get_linked_ptx_size, nvjitlink.get_linked_ptx),
    }

    def link(self, target_type) -> ObjectCode:
        nvjitlink.complete(self._mnff.handle)
        get_linked = self._get_linked_methods.get(target_type)
        if get_linked is None:
            raise ValueError(f"Unsupported target type: {target_type}")

        get_size, get_code = get_linked
        size = get_size(self._mnff.handle)
        code = bytearray(size)
        get_code(self._mnff.handle, code)

        return ObjectCode(bytes(code), target_type)

    def get_error_log(self) -> str:
        log_size = nvjitlink.get_error_log_size(self._mnff.handle)
        log = bytearray(log_size)
        nvjitlink.get_error_log(self._mnff.handle, log)
        return log.decode()

    def get_info_log(self) -> str:
        log_size = nvjitlink.get_info_log_size(self._mnff.handle)
        log = bytearray(log_size)
        nvjitlink.get_info_log(self._mnff.handle, log)
        return log.decode()

    _input_types = {
        "ptx": nvjitlink.InputType.PTX,
        "cubin": nvjitlink.InputType.CUBIN,
        "fatbin": nvjitlink.InputType.FATBIN,
        "ltoir": nvjitlink.InputType.LTOIR,
        "object": nvjitlink.InputType.OBJECT,
    }

    def _input_type_from_code_type(self, code_type: str) -> nvjitlink.InputType:
        # this list is based on the supported values for code_type in the ObjectCode class definition.
        # nvjitlink supports other options for input type
        input_type = self._input_types.get(code_type)

        if input_type is None:
            raise ValueError(f"Unknown code_type associated with ObjectCode: {code_type}")
        return input_type

    @property
    def handle(self) -> int:
        return self._mnff.handle

    def close(self):
        self._mnff.close()
