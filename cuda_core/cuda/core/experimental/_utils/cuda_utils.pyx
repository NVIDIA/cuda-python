# SPDX-FileCopyrightText: Copyright (c) 2024-2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

import functools
import importlib.metadata
from collections import namedtuple
from collections.abc import Sequence
from typing import Callable

try:
    from cuda.bindings import driver, nvrtc, runtime
except ImportError:
    from cuda import cuda as driver
    from cuda import cudart as runtime
    from cuda import nvrtc

from cuda.core.experimental._utils.driver_cu_result_explanations import DRIVER_CU_RESULT_EXPLANATIONS
from cuda.core.experimental._utils.runtime_cuda_error_explanations import RUNTIME_CUDA_ERROR_EXPLANATIONS


class CUDAError(Exception):
    pass


class NVRTCError(CUDAError):
    pass


ComputeCapability = namedtuple("ComputeCapability", ("major", "minor"))


def cast_to_3_tuple(label, cfg):
    cfg_orig = cfg
    if isinstance(cfg, int):
        cfg = (cfg,)
    else:
        common = "must be an int, or a tuple with up to 3 ints"
        if not isinstance(cfg, tuple):
            raise ValueError(f"{label} {common} (got {type(cfg)})")
        if len(cfg) > 3:
            raise ValueError(f"{label} {common} (got tuple with length {len(cfg)})")
        if any(not isinstance(val, int) for val in cfg):
            raise ValueError(f"{label} {common} (got {cfg})")
    if any(val < 1 for val in cfg):
        plural_s = "" if len(cfg) == 1 else "s"
        raise ValueError(f"{label} value{plural_s} must be >= 1 (got {cfg_orig})")
    return cfg + (1,) * (3 - len(cfg))


def _reduce_3_tuple(t: tuple):
    return t[0] * t[1] * t[2]


cdef object _DRIVER_SUCCESS = driver.CUresult.CUDA_SUCCESS
cdef object _RUNTIME_SUCCESS = runtime.cudaError_t.cudaSuccess
cdef object _NVRTC_SUCCESS = nvrtc.nvrtcResult.NVRTC_SUCCESS


cpdef inline int _check_driver_error(error) except?-1:
    if error == _DRIVER_SUCCESS:
        return 0
    name_err, name = driver.cuGetErrorName(error)
    if name_err != _DRIVER_SUCCESS:
        raise CUDAError(f"UNEXPECTED ERROR CODE: {error}")
    name = name.decode()
    expl = DRIVER_CU_RESULT_EXPLANATIONS.get(int(error))
    if expl is not None:
        raise CUDAError(f"{name}: {expl}")
    desc_err, desc = driver.cuGetErrorString(error)
    if desc_err != _DRIVER_SUCCESS:
        raise CUDAError(f"{name}")
    desc = desc.decode()
    raise CUDAError(f"{name}: {desc}")


cpdef inline int _check_runtime_error(error) except?-1:
    if error == _RUNTIME_SUCCESS:
        return 0
    name_err, name = runtime.cudaGetErrorName(error)
    if name_err != _RUNTIME_SUCCESS:
        raise CUDAError(f"UNEXPECTED ERROR CODE: {error}")
    name = name.decode()
    expl = RUNTIME_CUDA_ERROR_EXPLANATIONS.get(int(error))
    if expl is not None:
        raise CUDAError(f"{name}: {expl}")
    desc_err, desc = runtime.cudaGetErrorString(error)
    if desc_err != _RUNTIME_SUCCESS:
        raise CUDAError(f"{name}")
    desc = desc.decode()
    raise CUDAError(f"{name}: {desc}")


cpdef inline int _check_nvrtc_error(error, handle=None) except?-1:
    if error == _NVRTC_SUCCESS:
        return 0
    err = f"{error}: {nvrtc.nvrtcGetErrorString(error)[1].decode()}"
    if handle is not None:
        _, logsize = nvrtc.nvrtcGetProgramLogSize(handle)
        log = b" " * logsize
        _ = nvrtc.nvrtcGetProgramLog(handle, log)
        err += f", compilation log:\n\n{log.decode('utf-8', errors='backslashreplace')}"
    raise NVRTCError(err)


cdef inline int _check_error(error, handle=None) except?-1:
    if isinstance(error, driver.CUresult):
        return _check_driver_error(error)
    elif isinstance(error, runtime.cudaError_t):
        return _check_runtime_error(error)
    elif isinstance(error, nvrtc.nvrtcResult):
        return _check_nvrtc_error(error, handle=handle)
    else:
        raise RuntimeError(f"Unknown error type: {error}")


def handle_return(tuple result, handle=None):
    _check_error(result[0], handle=handle)
    cdef int out_len = len(result)
    if out_len == 1:
        return
    elif out_len == 2:
        return result[1]
    else:
        return result[1:]


cpdef check_or_create_options(type cls, options, str options_description="", bint keep_none=False):
    """
    Create the specified options dataclass from a dictionary of options or None.
    """
    if options is None:
        if keep_none:
            return options
        return cls()
    elif isinstance(options, cls):
        return options
    elif isinstance(options, dict):
        return cls(**options)
    else:
        raise TypeError(
            f"The {options_description} must be provided as an object "
            f"of type {cls.__name__} or as a dict with valid {options_description}. "
            f"The provided object is '{options}'."
        )


def _handle_boolean_option(option: bool) -> str:
    """
    Convert a boolean option to a string representation.
    """
    return "true" if bool(option) else "false"


def precondition(checker: Callable[..., None], str what="") -> Callable:
    """
    A decorator that adds checks to ensure any preconditions are met.

    Args:
        checker: The function to call to check whether the preconditions are met. It has
        the same signature as the wrapped function with the addition of the keyword argument `what`.
        what: A string that is passed in to `checker` to provide context information.

    Returns:
        Callable: A decorator that creates the wrapping.
    """

    def outer(wrapped_function):
        """
        A decorator that actually wraps the function for checking preconditions.
        """

        @functools.wraps(wrapped_function)
        def inner(*args, **kwargs):
            """
            Check preconditions and if they are met, call the wrapped function.
            """
            checker(*args, **kwargs, what=what)
            result = wrapped_function(*args, **kwargs)

            return result

        return inner

    return outer


def get_device_from_ctx(ctx_handle) -> int:
    """Get device ID from the given ctx."""
    from cuda.core.experimental._device import Device  # avoid circular import

    prev_ctx = Device().context._handle
    switch_context = int(ctx_handle) != int(prev_ctx)
    if switch_context:
        assert prev_ctx == handle_return(driver.cuCtxPopCurrent())
        handle_return(driver.cuCtxPushCurrent(ctx_handle))
    device_id = int(handle_return(driver.cuCtxGetDevice()))
    if switch_context:
        assert ctx_handle == handle_return(driver.cuCtxPopCurrent())
        handle_return(driver.cuCtxPushCurrent(prev_ctx))
    return device_id


def is_sequence(obj):
    """
    Check if the given object is a sequence (list or tuple).
    """
    return isinstance(obj, Sequence)


def is_nested_sequence(obj):
    """
    Check if the given object is a nested sequence (list or tuple with atleast one list or tuple element).
    """
    return is_sequence(obj) and any(is_sequence(elem) for elem in obj)


@functools.lru_cache
def get_binding_version():
    try:
        major_minor = importlib.metadata.version("cuda-bindings").split(".")[:2]
    except importlib.metadata.PackageNotFoundError:
        major_minor = importlib.metadata.version("cuda-python").split(".")[:2]
    return tuple(int(v) for v in major_minor)
