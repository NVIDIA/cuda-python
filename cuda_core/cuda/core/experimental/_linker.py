# SPDX-FileCopyrightText: Copyright (c) 2024-2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import ctypes
import weakref
from contextlib import contextmanager
from dataclasses import dataclass
from typing import TYPE_CHECKING, Union
from warnings import warn

if TYPE_CHECKING:
    import cuda.bindings

from cuda.core.experimental._device import Device
from cuda.core.experimental._module import ObjectCode
from cuda.core.experimental._utils.clear_error_support import assert_type
from cuda.core.experimental._utils.cuda_utils import check_or_create_options, driver, handle_return, is_sequence

# TODO: revisit this treatment for py313t builds
_driver = None  # populated if nvJitLink cannot be used
_driver_input_types = None  # populated if nvJitLink cannot be used
_driver_ver = None
_inited = False
_nvjitlink = None  # populated if nvJitLink can be used
_nvjitlink_input_types = None  # populated if nvJitLink cannot be used


# Note: this function is reused in the tests
def _decide_nvjitlink_or_driver() -> bool:
    """Returns True if falling back to the cuLink* driver APIs."""
    global _driver_ver, _driver, _nvjitlink
    if _driver or _nvjitlink:
        return _driver is not None

    _driver_ver = handle_return(driver.cuDriverGetVersion())
    _driver_ver = (_driver_ver // 1000, (_driver_ver % 1000) // 10)
    try:
        from cuda.bindings import nvjitlink as _nvjitlink
        from cuda.bindings._internal import nvjitlink as inner_nvjitlink
    except ImportError:
        # binding is not available
        _nvjitlink = None
    else:
        if inner_nvjitlink._inspect_function_pointer("__nvJitLinkVersion") == 0:
            # binding is available, but nvJitLink is not installed
            _nvjitlink = None

    if _nvjitlink is None:
        warn(
            "nvJitLink is not installed or too old (<12.3). Therefore it is not usable "
            "and the culink APIs will be used instead.",
            stacklevel=3,
            category=RuntimeWarning,
        )
        _driver = driver
        return True
    else:
        return False


def _lazy_init():
    global _inited, _nvjitlink_input_types, _driver_input_types
    if _inited:
        return

    _decide_nvjitlink_or_driver()
    if _nvjitlink:
        if _driver_ver > _nvjitlink.version():
            # TODO: nvJitLink is not new enough, warn?
            pass
        _nvjitlink_input_types = {
            "ptx": _nvjitlink.InputType.PTX,
            "cubin": _nvjitlink.InputType.CUBIN,
            "fatbin": _nvjitlink.InputType.FATBIN,
            "ltoir": _nvjitlink.InputType.LTOIR,
            "object": _nvjitlink.InputType.OBJECT,
            "library": _nvjitlink.InputType.LIBRARY,
        }
    else:
        _driver_input_types = {
            "ptx": _driver.CUjitInputType.CU_JIT_INPUT_PTX,
            "cubin": _driver.CUjitInputType.CU_JIT_INPUT_CUBIN,
            "fatbin": _driver.CUjitInputType.CU_JIT_INPUT_FATBINARY,
            "object": _driver.CUjitInputType.CU_JIT_INPUT_OBJECT,
            "library": _driver.CUjitInputType.CU_JIT_INPUT_LIBRARY,
        }
    _inited = True


@dataclass
class LinkerOptions:
    """Customizable :obj:`Linker` options.

    Since the linker would choose to use nvJitLink or the driver APIs as the linking backed,
    not all options are applicable. When the system's installed nvJitLink is too old (<12.3),
    or not installed, the driver APIs (cuLink) will be used instead.

    Attributes
    ----------
    name : str, optional
        Name of the linker. If the linking succeeds, the name is passed down to the generated `ObjectCode`.
    arch : str, optional
        Pass the SM architecture value, such as ``sm_<CC>`` (for generating CUBIN) or
        ``compute_<CC>`` (for generating PTX). If not provided, the current device's architecture
        will be used.
    max_register_count : int, optional
        Maximum register count.
    time : bool, optional
        Print timing information to the info log.
        Default: False.
    verbose : bool, optional
        Print verbose messages to the info log.
        Default: False.
    link_time_optimization : bool, optional
        Perform link time optimization.
        Default: False.
    ptx : bool, optional
        Emit PTX after linking instead of CUBIN; only supported with ``link_time_optimization=True``.
        Default: False.
    optimization_level : int, optional
        Set optimization level. Only 0 and 3 are accepted.
    debug : bool, optional
        Generate debug information.
        Default: False.
    lineinfo : bool, optional
        Generate line information.
        Default: False.
    ftz : bool, optional
        Flush denormal values to zero.
        Default: False.
    prec_div : bool, optional
        Use precise division.
        Default: True.
    prec_sqrt : bool, optional
        Use precise square root.
        Default: True.
    fma : bool, optional
        Use fast multiply-add.
        Default: True.
    kernels_used : [Union[str, tuple[str], list[str]]], optional
        Pass a kernel or sequence of kernels that are used; any not in the list can be removed.
    variables_used : [Union[str, tuple[str], list[str]]], optional
        Pass a variable or sequence of variables that are used; any not in the list can be removed.
    optimize_unused_variables : bool, optional
        Assume that if a variable is not referenced in device code, it can be removed.
        Default: False.
    ptxas_options : [Union[str, tuple[str], list[str]]], optional
        Pass options to PTXAS.
    split_compile : int, optional
        Split compilation maximum thread count. Use 0 to use all available processors. Value of 1 disables split
        compilation (default).
        Default: 1.
    split_compile_extended : int, optional
        A more aggressive form of split compilation available in LTO mode only. Accepts a maximum thread count value.
        Use 0 to use all available processors. Value of 1 disables extended split compilation (default). Note: This
        option can potentially impact performance of the compiled binary.
        Default: 1.
    no_cache : bool, optional
        Do not cache the intermediate steps of nvJitLink.
        Default: False.
    """

    name: str | None = "<default linker>"
    arch: str | None = None
    max_register_count: int | None = None
    time: bool | None = None
    verbose: bool | None = None
    link_time_optimization: bool | None = None
    ptx: bool | None = None
    optimization_level: int | None = None
    debug: bool | None = None
    lineinfo: bool | None = None
    ftz: bool | None = None
    prec_div: bool | None = None
    prec_sqrt: bool | None = None
    fma: bool | None = None
    kernels_used: Union[str, tuple[str], list[str]] | None = None
    variables_used: Union[str, tuple[str], list[str]] | None = None
    optimize_unused_variables: bool | None = None
    ptxas_options: Union[str, tuple[str], list[str]] | None = None
    split_compile: int | None = None
    split_compile_extended: int | None = None
    no_cache: bool | None = None

    def __post_init__(self):
        _lazy_init()
        self._name = self.name.encode()
        self.formatted_options = []
        if _nvjitlink:
            self._init_nvjitlink()
        else:
            self._init_driver()

    def _init_nvjitlink(self):
        if self.arch is not None:
            self.formatted_options.append(f"-arch={self.arch}")
        else:
            self.formatted_options.append("-arch=sm_" + "".join(f"{i}" for i in Device().compute_capability))
        if self.max_register_count is not None:
            self.formatted_options.append(f"-maxrregcount={self.max_register_count}")
        if self.time is not None:
            self.formatted_options.append("-time")
        if self.verbose:
            self.formatted_options.append("-verbose")
        if self.link_time_optimization:
            self.formatted_options.append("-lto")
        if self.ptx:
            self.formatted_options.append("-ptx")
        if self.optimization_level is not None:
            self.formatted_options.append(f"-O{self.optimization_level}")
        if self.debug:
            self.formatted_options.append("-g")
        if self.lineinfo:
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
            if isinstance(self.kernels_used, str):
                self.formatted_options.append(f"-kernels-used={self.kernels_used}")
            elif isinstance(self.kernels_used, list):
                for kernel in self.kernels_used:
                    self.formatted_options.append(f"-kernels-used={kernel}")
        if self.variables_used is not None:
            if isinstance(self.variables_used, str):
                self.formatted_options.append(f"-variables-used={self.variables_used}")
            elif isinstance(self.variables_used, list):
                for variable in self.variables_used:
                    self.formatted_options.append(f"-variables-used={variable}")
        if self.optimize_unused_variables is not None:
            self.formatted_options.append("-optimize-unused-variables")
        if self.ptxas_options is not None:
            if isinstance(self.ptxas_options, str):
                self.formatted_options.append(f"-Xptxas={self.ptxas_options}")
            elif is_sequence(self.ptxas_options):
                for opt in self.ptxas_options:
                    self.formatted_options.append(f"-Xptxas={opt}")
        if self.split_compile is not None:
            self.formatted_options.append(f"-split-compile={self.split_compile}")
        if self.split_compile_extended is not None:
            self.formatted_options.append(f"-split-compile-extended={self.split_compile_extended}")
        if self.no_cache is True:
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
            raise ValueError("time option is not supported by the driver API")
        if self.verbose:
            self.formatted_options.append(1)
            self.option_keys.append(_driver.CUjit_option.CU_JIT_LOG_VERBOSE)
        if self.link_time_optimization:
            self.formatted_options.append(1)
            self.option_keys.append(_driver.CUjit_option.CU_JIT_LTO)
        if self.ptx:
            raise ValueError("ptx option is not supported by the driver API")
        if self.optimization_level is not None:
            self.formatted_options.append(self.optimization_level)
            self.option_keys.append(_driver.CUjit_option.CU_JIT_OPTIMIZATION_LEVEL)
        if self.debug:
            self.formatted_options.append(1)
            self.option_keys.append(_driver.CUjit_option.CU_JIT_GENERATE_DEBUG_INFO)
        if self.lineinfo:
            self.formatted_options.append(1)
            self.option_keys.append(_driver.CUjit_option.CU_JIT_GENERATE_LINE_INFO)
        if self.ftz is not None:
            warn("ftz option is deprecated in the driver API", DeprecationWarning, stacklevel=3)
        if self.prec_div is not None:
            warn("prec_div option is deprecated in the driver API", DeprecationWarning, stacklevel=3)
        if self.prec_sqrt is not None:
            warn("prec_sqrt option is deprecated in the driver API", DeprecationWarning, stacklevel=3)
        if self.fma is not None:
            warn("fma options is deprecated in the driver API", DeprecationWarning, stacklevel=3)
        if self.kernels_used is not None:
            warn("kernels_used is deprecated in the driver API", DeprecationWarning, stacklevel=3)
        if self.variables_used is not None:
            warn("variables_used is deprecated in the driver API", DeprecationWarning, stacklevel=3)
        if self.optimize_unused_variables is not None:
            warn("optimize_unused_variables is deprecated in the driver API", DeprecationWarning, stacklevel=3)
        if self.ptxas_options is not None:
            raise ValueError("ptxas_options option is not supported by the driver API")
        if self.split_compile is not None:
            raise ValueError("split_compile option is not supported by the driver API")
        if self.split_compile_extended is not None:
            raise ValueError("split_compile_extended option is not supported by the driver API")
        if self.no_cache is True:
            self.formatted_options.append(_driver.CUjit_cacheMode.CU_JIT_CACHE_OPTION_NONE)
            self.option_keys.append(_driver.CUjit_option.CU_JIT_CACHE_MODE)


# This needs to be a free function not a method, as it's disallowed by contextmanager.
@contextmanager
def _exception_manager(self):
    """
    A helper function to improve the error message of exceptions raised by the linker backend.
    """
    try:
        yield
    except Exception as e:
        error_log = ""
        if hasattr(self, "_mnff"):
            # our constructor could raise, in which case there's no handle available
            error_log = self.get_error_log()
        # Starting Python 3.11 we could also use Exception.add_note() for the same purpose, but
        # unfortunately we are still supporting Python 3.9/3.10...
        # Here we rely on both CUDAError and nvJitLinkError have the error string placed in .args[0].
        e.args = (e.args[0] + (f"\nLinker error log: {error_log}" if error_log else ""), *e.args[1:])
        raise e


nvJitLinkHandleT = int
LinkerHandleT = Union[nvJitLinkHandleT, "cuda.bindings.driver.CUlinkState"]


class Linker:
    """Represent a linking machinery to link one or multiple object codes into
    :obj:`~cuda.core.experimental._module.ObjectCode` with the specified options.

    This object provides a unified interface to multiple underlying
    linker libraries (such as nvJitLink or cuLink* from CUDA driver).

    Parameters
    ----------
    object_codes : ObjectCode
        One or more ObjectCode objects to be linked.
    options : LinkerOptions, optional
        Options for the linker. If not provided, default options will be used.
    """

    class _MembersNeededForFinalize:
        __slots__ = ("handle", "use_nvjitlink", "const_char_keep_alive")

        def __init__(self, program_obj, handle, use_nvjitlink):
            self.handle = handle
            self.use_nvjitlink = use_nvjitlink
            self.const_char_keep_alive = []
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
        with _exception_manager(self):
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
            assert_type(code, ObjectCode)
            self._add_code_object(code)

    def _add_code_object(self, object_code: ObjectCode):
        data = object_code._module
        assert_type(data, bytes)
        with _exception_manager(self):
            name_str = f"{object_code.name}"
            if _nvjitlink:
                _nvjitlink.add_data(
                    self._mnff.handle,
                    self._input_type_from_code_type(object_code._code_type),
                    data,
                    len(data),
                    name_str,
                )
            else:
                name_bytes = name_str.encode()
                handle_return(
                    _driver.cuLinkAddData(
                        self._mnff.handle,
                        self._input_type_from_code_type(object_code._code_type),
                        data,
                        len(data),
                        name_bytes,
                        0,
                        None,
                        None,
                    )
                )
                self._mnff.const_char_keep_alive.append(name_bytes)

    def link(self, target_type) -> ObjectCode:
        """
        Links the provided object codes into a single output of the specified target type.

        Parameters
        ----------
        target_type : str
            The type of the target output. Must be either "cubin" or "ptx".

        Returns
        -------
        ObjectCode
            The linked object code of the specified target type.

        Note
        ------
        See nvrtc compiler options documnetation to ensure the input object codes are
        correctly compiled for linking.
        """
        if target_type not in ("cubin", "ptx"):
            raise ValueError(f"Unsupported target type: {target_type}")
        with _exception_manager(self):
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

        return ObjectCode._init(bytes(code), target_type, name=self._options.name)

    def get_error_log(self) -> str:
        """Get the error log generated by the linker.

        Returns
        -------
        str
            The error log.
        """
        if _nvjitlink:
            log_size = _nvjitlink.get_error_log_size(self._mnff.handle)
            log = bytearray(log_size)
            _nvjitlink.get_error_log(self._mnff.handle, log)
        else:
            log = self._options.formatted_options[2]
        return log.decode("utf-8", errors="backslashreplace")

    def get_info_log(self) -> str:
        """Get the info log generated by the linker.

        Returns
        -------
        str
            The info log.
        """
        if _nvjitlink:
            log_size = _nvjitlink.get_info_log_size(self._mnff.handle)
            log = bytearray(log_size)
            _nvjitlink.get_info_log(self._mnff.handle, log)
        else:
            log = self._options.formatted_options[0]
        return log.decode("utf-8", errors="backslashreplace")

    def _input_type_from_code_type(self, code_type: str):
        # this list is based on the supported values for code_type in the ObjectCode class definition.
        # nvJitLink/driver support other options for input type
        input_type = _nvjitlink_input_types.get(code_type) if _nvjitlink else _driver_input_types.get(code_type)

        if input_type is None:
            raise ValueError(f"Unknown code_type associated with ObjectCode: {code_type}")
        return input_type

    @property
    def handle(self) -> LinkerHandleT:
        """Return the underlying handle object.

        .. note::

           The type of the returned object depends on the backend.

        .. caution::

            This handle is a Python object. To get the memory address of the underlying C
            handle, call ``int(Linker.handle)``.
        """
        return self._mnff.handle

    @property
    def backend(self) -> str:
        """Return this Linker instance's underlying backend."""
        return "nvJitLink" if self._mnff.use_nvjitlink else "driver"

    def close(self):
        """Destroy this linker."""
        self._mnff.close()
