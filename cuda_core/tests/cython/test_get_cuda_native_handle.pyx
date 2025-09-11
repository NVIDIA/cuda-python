# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# SPDX-License-Identifier: LicenseRef-NVIDIA-SOFTWARE-LICENSE

# distutils: language = c++
# distutils: extra_compile_args = -std=c++17

from libc.stdint cimport intptr_t

from cuda.bindings.driver cimport (CUstream as pyCUstream,
                                   CUevent as pyCUevent)
from cuda.bindings.nvrtc cimport nvrtcProgram as pynvrtcProgram
from cuda.bindings.cydriver cimport CUstream, CUevent
from cuda.bindings.cynvrtc cimport nvrtcProgram

from cuda.core.experimental import Device, Program


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

    prog = Program("extern \"C\" __global__ void dummy() {}", "c++")
    assert prog.backend == "NVRTC"
    cdef pynvrtcProgram prog_py = prog.handle
    cdef nvrtcProgram prog_c = <nvrtcProgram>get_cuda_native_handle(prog_py)
    assert <intptr_t>(prog_c) == <intptr_t>(int(prog_py))

    prog.close()
    e.close()
    s.close()
