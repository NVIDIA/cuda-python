# Copyright (c) 2024, NVIDIA CORPORATION & AFFILIATES. ALL RIGHTS RESERVED.
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


class CUDAError(Exception):
    pass


class NVRTCError(CUDAError):
    pass


ComputeCapability = namedtuple("ComputeCapability", ("major", "minor"))


# CUDA Toolkit v12.8.0
# https://docs.nvidia.com/cuda/cuda-driver-api/group__CUDA__TYPES.html#group__CUDA__TYPES
# noqa: E501
_DRIVER_CU_RESULT_EXPLAINED = {
    0: (
        "The API call returned with no errors. In the case of query calls, this also means that the operation being "
        "queried is complete (see cuEventQuery() and cuStreamQuery())."
    ),
    1: "The parameters passed to the API call are not within an acceptable range of values.",
    2: (
        "The API call failed because it was unable to allocate enough memory or other resources to perform the "
        "requested operation."
    ),
    3: "The CUDA driver has not been initialized with cuInit() or initialization has failed.",
    4: "The CUDA driver is in the process of shutting down.",
    5: (
        "The profiler is not initialized for this run. This can happen when the application is running with external "
        "profiling tools like visual profiler."
    ),
    34: (
        "The CUDA driver that the application has loaded is a stub library. Applications that run with the stub "
        "rather than a real driver loaded will result in CUDA API returning this error."
    ),
    46: (
        "The requested CUDA device is unavailable at the current time. Devices are often unavailable due to use of "
        "CU_COMPUTEMODE_EXCLUSIVE_PROCESS or CU_COMPUTEMODE_PROHIBITED."
    ),
    100: "No CUDA-capable devices were detected by the installed CUDA driver.",
    101: (
        "The device ordinal supplied by the user does not correspond to a valid CUDA device or the action requested "
        "is invalid for the specified device."
    ),
    102: "The Grid license is not applied.",
    200: "The device kernel image is invalid. This can also indicate an invalid CUDA module.",
    201: (
        "There is no context bound to the current thread. This can also be returned if the context passed to an API "
        "call is not a valid handle, if a user mixes different API versions, or if the green context passed to an "
        "API call was not converted to a CUcontext using cuCtxFromGreenCtx API."
    ),
    226: (
        "An exception occurred on the device that is now contained by the GPU's error containment capability. This "
        "leaves the process in an inconsistent state, and any further CUDA work will return the same error. The "
        "process must be terminated and relaunched."
    ),
    300: (
        "The device kernel source is invalid. This includes compilation/linker errors encountered in device code or "
        "user error."
    ),
    500: (
        "A named symbol was not found. Examples include global/constant variable names, driver function names, "
        "texture names, and surface names."
    ),
    700: (
        "While executing a kernel, the device encountered a load or store instruction on an invalid memory address. "
        "The process must be terminated and relaunched."
    ),
    999: "An unknown internal error has occurred.",
}


def _driver_error_info(error):
    expl = _DRIVER_CU_RESULT_EXPLAINED.get(error)
    err, name = driver.cuGetErrorName(error)
    if err == driver.CUresult.CUDA_SUCCESS:
        err, desc = driver.cuGetErrorString(error)
        assert err == driver.CUresult.CUDA_SUCCESS
        return (name.decode(), desc.decode(), expl)
    else:
        return ("INVALID ERROR CODE", None, expl)


def _check_driver_error(error):
    if error == driver.CUresult.CUDA_SUCCESS:
        return
    print(f"\nLOOOK driver.CUresult {error=}", flush=True)
    err, name = driver.cuGetErrorName(error)
    if err == driver.CUresult.CUDA_SUCCESS:
        err, desc = driver.cuGetErrorString(error)
    if err == driver.CUresult.CUDA_SUCCESS:
        raise CUDAError(f"{name.decode()}: {desc.decode()}")
    else:
        raise CUDAError(f"unknown error: {error}")


def _check_runtime_error(error):
    if error == runtime.cudaError_t.cudaSuccess:
        return
    print(f"\nLOOOK runtime.cudaError_t {error=}", flush=True)
    err, name = runtime.cudaGetErrorName(error)
    if err == runtime.cudaError_t.cudaSuccess:
        err, desc = runtime.cudaGetErrorString(error)
    if err == runtime.cudaError_t.cudaSuccess:
        raise CUDAError(f"{name.decode()}: {desc.decode()}")
    else:
        raise CUDAError(f"unknown error: {error}")


def _check_nvrtc_error(error, handle):
    if error == nvrtc.nvrtcResult.NVRTC_SUCCESS:
        return
    print(f"\nLOOOK nvrtc.nvrtcResult {error=}", flush=True)
    err = f"{error}: {nvrtc.nvrtcGetErrorString(error)[1].decode()}"
    if handle is not None:
        _, logsize = nvrtc.nvrtcGetProgramLogSize(handle)
        log = b" " * logsize
        _ = nvrtc.nvrtcGetProgramLog(handle, log)
        err += f", compilation log:\n\n{log.decode()}"
    raise NVRTCError(err)


def _check_error(error, handle=None):
    if isinstance(error, driver.CUresult):
        _check_driver_error(error)
    elif isinstance(error, runtime.cudaError_t):
        _check_runtime_error(error)
    elif isinstance(error, nvrtc.nvrtcResult):
        _check_nvrtc_error(error, handle)
    else:
        raise RuntimeError(f"Unknown error type: {error}")


def _strip_return_tuple(result):
    if len(result) == 1:
        return
    elif len(result) == 2:
        return result[1]
    else:
        return result[1:]


def handle_return(result, handle=None):
    _check_error(result[0], handle=handle)
    return _strip_return_tuple(result)


def raise_if_driver_error(return_tuple):
    _check_driver_error(return_tuple[0])
    return _strip_return_tuple(return_tuple)


def raise_if_runtime_error(return_tuple):
    _check_runtime_error(return_tuple[0])
    return _strip_return_tuple(return_tuple)


def raise_if_nvrtc_error(return_tuple, handle):
    _check_nvrtc_error(return_tuple[0], handle)
    return _strip_return_tuple(return_tuple)


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


def get_binding_version():
    try:
        major_minor = importlib.metadata.version("cuda-bindings").split(".")[:2]
    except importlib.metadata.PackageNotFoundError:
        major_minor = importlib.metadata.version("cuda-python").split(".")[:2]
    return tuple(int(v) for v in major_minor)
