# Copyright (c) 2024-2025, NVIDIA CORPORATION & AFFILIATES. ALL RIGHTS RESERVED.
#
# SPDX-License-Identifier: LicenseRef-NVIDIA-SOFTWARE-LICENSE

import functools
import importlib.metadata
from collections import namedtuple
from collections.abc import Sequence
from typing import Callable, Dict

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


def _check_driver_error(error):
    if error == driver.CUresult.CUDA_SUCCESS:
        return
    name_err, name = driver.cuGetErrorName(error)
    if name_err != driver.CUresult.CUDA_SUCCESS:
        raise CUDAError(f"UNEXPECTED ERROR CODE: {error}")
    name = name.decode()
    expl = DRIVER_CU_RESULT_EXPLANATIONS.get(int(error))
    if expl is not None:
        raise CUDAError(f"{name}: {expl}")
    desc_err, desc = driver.cuGetErrorString(error)
    if desc_err != driver.CUresult.CUDA_SUCCESS:
        raise CUDAError(f"{name}")
    desc = desc.decode()
    raise CUDAError(f"{name}: {desc}")


def _check_runtime_error(error):
    if error == runtime.cudaError_t.cudaSuccess:
        return
    name_err, name = runtime.cudaGetErrorName(error)
    if name_err != runtime.cudaError_t.cudaSuccess:
        raise CUDAError(f"UNEXPECTED ERROR CODE: {error}")
    name = name.decode()
    expl = RUNTIME_CUDA_ERROR_EXPLANATIONS.get(int(error))
    if expl is not None:
        raise CUDAError(f"{name}: {expl}")
    desc_err, desc = runtime.cudaGetErrorString(error)
    if desc_err != runtime.cudaError_t.cudaSuccess:
        raise CUDAError(f"{name}")
    desc = desc.decode()
    raise CUDAError(f"{name}: {desc}")


def _check_error(error, handle=None):
    if isinstance(error, driver.CUresult):
        _check_driver_error(error)
    elif isinstance(error, runtime.cudaError_t):
        _check_runtime_error(error)
    elif isinstance(error, nvrtc.nvrtcResult):
        if error == nvrtc.nvrtcResult.NVRTC_SUCCESS:
            return
        err = f"{error}: {nvrtc.nvrtcGetErrorString(error)[1].decode()}"
        if handle is not None:
            _, logsize = nvrtc.nvrtcGetProgramLogSize(handle)
            log = b" " * logsize
            _ = nvrtc.nvrtcGetProgramLog(handle, log)
            err += f", compilation log:\n\n{log.decode('utf-8', errors='backslashreplace')}"
        raise NVRTCError(err)
    else:
        raise RuntimeError(f"Unknown error type: {error}")


def handle_return(result, handle=None):
    _check_error(result[0], handle=handle)
    if len(result) == 1:
        return
    elif len(result) == 2:
        return result[1]
    else:
        return result[1:]


def check_or_create_options(cls, options, options_description, *, keep_none=False):
    """
    Create the specified options dataclass from a dictionary of options or None.
    """

    if options is None:
        if keep_none:
            return options
        options = cls()
    elif isinstance(options, Dict):
        options = cls(**options)

    if not isinstance(options, cls):
        raise TypeError(
            f"The {options_description} must be provided as an object "
            f"of type {cls.__name__} or as a dict with valid {options_description}. "
            f"The provided object is '{options}'."
        )

    return options


def _handle_boolean_option(option: bool) -> str:
    """
    Convert a boolean option to a string representation.
    """
    return "true" if bool(option) else "false"


def precondition(checker: Callable[..., None], what: str = "") -> Callable:
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
