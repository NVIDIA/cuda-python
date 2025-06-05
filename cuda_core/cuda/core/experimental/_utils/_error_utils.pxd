from cuda.bindings cimport cydriver as driver
from cuda.bindings cimport cyruntime as runtime

cpdef _check_driver_error(driver.CUresult result)
cpdef _check_runtime_error(runtime.cudaError_t error)
