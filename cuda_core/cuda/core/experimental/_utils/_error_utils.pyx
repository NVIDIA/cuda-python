from cython cimport NULL
from libc.string cimport strlen 

import cuda.bindings
from cuda.bindings import driver, runtime

from cuda.bindings cimport cydriver as cdriver
from cuda.bindings cimport cyruntime as cruntime

from cuda.core.experimental._utils.driver_cu_result_explanations import DRIVER_CU_RESULT_EXPLANATIONS
from cuda.core.experimental._utils.runtime_cuda_error_explanations import RUNTIME_CUDA_ERROR_EXPLANATIONS


class CUDAError(Exception):
    pass



cpdef _check_driver_error(cdriver.CUresult error):
    if error == cdriver.CUresult.CUDA_SUCCESS:
        return
    cdef const char *c_name = NULL
    name_err = driver.cuGetErrorName(error, &c_name)
    if name_err != driver.CUresult.CUDA_SUCCESS:
        raise CUDAError(f"UNEXPECTED ERROR CODE: {error}")
    name: str = (<char *>c_name)[:strlen(c_name)].decode()
    #name = name.decode()
    expl = DRIVER_CU_RESULT_EXPLANATIONS.get(int(error))
    if expl is not None:
        raise CUDAError(f"{name}: {expl}")
    cdef const char *c_desc = NULL
    desc_err = driver.cuGetErrorString(error, &c_desc)
    if desc_err != driver.CUresult.CUDA_SUCCESS:
        raise CUDAError(f"{name}")
    desc: str = (<char *>c_desc)[:strlen(c_desc)].decode()
    #desc = desc.decode()
    raise CUDAError(f"{name}: {desc}")


cpdef _check_runtime_error(cruntime.cudaError_t error):
    if error == cruntime.cudaError_t.cudaSuccess:
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


