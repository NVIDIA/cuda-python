from typing import Dict

from cuda import cuda, cudart, nvrtc


class CUDAError(Exception): pass


class NVRTCError(Exception): pass


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
        assert handle is not None
        _, logsize = nvrtc.nvrtcGetProgramLogSize(handle)
        log = b" " * logsize
        _ = nvrtc.nvrtcGetProgramLog(handle, log)
        err = f"{error}: {nvrtc.nvrtcGetErrorString(error)[1].decode()}, " \
              f"compilation log:\n\n{log.decode()}"
        raise NVRTCError(err)
    else:
        raise RuntimeError('Unknown error type: {}'.format(error))


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
        raise TypeError(f"The {options_description} must be provided as an object "
                        f"of type {cls.__name__} or as a dict with valid {options_description}. "
                        f"The provided object is '{options}'.")

    return options
