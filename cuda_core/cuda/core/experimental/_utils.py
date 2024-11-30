# Copyright (c) 2024, NVIDIA CORPORATION & AFFILIATES. ALL RIGHTS RESERVED.
#
# SPDX-License-Identifier: LicenseRef-NVIDIA-SOFTWARE-LICENSE

import functools
from collections import namedtuple
from typing import Callable, Dict

from cuda import cuda, cudart, nvrtc


class CUDAError(Exception):
    pass


class NVRTCError(CUDAError):
    pass


ComputeCapability = namedtuple("ComputeCapability", ("major", "minor"))


def _check_error(error, handle=None):
    if isinstance(error, cuda.CUresult):
        if error == cuda.CUresult.CUDA_SUCCESS:
            return
        err, name = cuda.cuGetErrorName(error)
        if err == cuda.CUresult.CUDA_SUCCESS:
            err, desc = cuda.cuGetErrorString(error)
        if err == cuda.CUresult.CUDA_SUCCESS:
            raise CUDAError(f"{name.decode()}: {desc.decode()}")
        else:
            raise CUDAError(f"unknown error: {error}")
    elif isinstance(error, cudart.cudaError_t):
        if error == cudart.cudaError_t.cudaSuccess:
            return
        err, name = cudart.cudaGetErrorName(error)
        if err == cudart.cudaError_t.cudaSuccess:
            err, desc = cudart.cudaGetErrorString(error)
        if err == cudart.cudaError_t.cudaSuccess:
            raise CUDAError(f"{name.decode()}: {desc.decode()}")
        else:
            raise CUDAError(f"unknown error: {error}")
    elif isinstance(error, nvrtc.nvrtcResult):
        if error == nvrtc.nvrtcResult.NVRTC_SUCCESS:
            return
        err = f"{error}: {nvrtc.nvrtcGetErrorString(error)[1].decode()}"
        if handle is not None:
            _, logsize = nvrtc.nvrtcGetProgramLogSize(handle)
            log = b" " * logsize
            _ = nvrtc.nvrtcGetProgramLog(handle, log)
            err += f", compilation log:\n\n{log.decode()}"
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
        assert prev_ctx == handle_return(cuda.cuCtxPopCurrent())
        handle_return(cuda.cuCtxPushCurrent(ctx_handle))
    device_id = int(handle_return(cuda.cuCtxGetDevice()))
    if switch_context:
        assert ctx_handle == handle_return(cuda.cuCtxPopCurrent())
        handle_return(cuda.cuCtxPushCurrent(prev_ctx))
    return device_id
