# distutils: language = c++
# distutils: extra_compile_args = -std=c++17

from libc.stdint cimport intptr_t

from cuda.bindings.driver cimport (CUstream as pyCUstream,
                                   CUevent as pyCUevent)
from cuda.bindings.cydriver cimport CUstream, CUevent

from cuda.core.experimental import Device


cdef extern from "utility.hpp":
    void* get_cuda_native_handle[T](T)


def test_get_cuda_native_handle():
    dev = Device(0)
    dev.set_current()
    
    s = dev.create_stream()
    cdef pyCUstream s_py = s.handle
    cdef CUstream s_c = <CUstream>get_cuda_native_handle(s_py)
    assert <intptr_t>(s_c) == <intptr_t>(int(s_py))

    e = s.record()
    cdef pyCUevent e_py = e.handle
    cdef CUevent e_c = <CUevent>get_cuda_native_handle(e_py)
    assert <intptr_t>(e_c) == <intptr_t>(int(e_py))

    e.close()
    s.close()
