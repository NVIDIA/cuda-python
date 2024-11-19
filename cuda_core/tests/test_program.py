# Copyright 2024 NVIDIA Corporation.  All rights reserved.
#
# Please refer to the NVIDIA end user license agreement (EULA) associated
# with this source code for terms and conditions that govern your use of
# this software. Any use, reproduction, disclosure, or distribution of
# this software and related documentation outside the terms of the EULA
# is strictly prohibited.

from cuda.core.experimental import Program
from cuda.core.experimental._module import ObjectCode, Kernel
import pytest

def test_program_init_valid_code_type():
    code = "extern \"C\" __global__ void my_kernel() {}"
    program = Program(code, "c++")
    assert program.backend == "nvrtc"
    assert program.handle is not None

def test_program_init_invalid_code_type():
    code = "extern \"C\" __global__ void my_kernel() {}"
    with pytest.raises(NotImplementedError):
        Program(code, "python")

def test_program_init_invalid_code_format():
    code = 12345
    with pytest.raises(TypeError):
        Program(code, "c++")

def test_program_compile_valid_target_type():
    code = "extern \"C\" __global__ void my_kernel() {}"
    program = Program(code, "c++")
    object_code = program.compile("ptx")
    kernel = object_code.get_kernel("my_kernel")
    assert isinstance(object_code, ObjectCode)
    assert isinstance(kernel, Kernel)

def test_program_compile_invalid_target_type():
    code = "extern \"C\" __global__ void my_kernel() {}"
    program = Program(code, "c++")
    with pytest.raises(NotImplementedError):
        program.compile("invalid_target")

def test_program_backend_property():
    code = "extern \"C\" __global__ void my_kernel() {}"
    program = Program(code, "c++")
    assert program.backend == "nvrtc"

def test_program_handle_property():
    code = "extern \"C\" __global__ void my_kernel() {}"
    program = Program(code, "c++")
    assert program.handle is not None

def test_program_close():
    code = "extern \"C\" __global__ void my_kernel() {}"
    program = Program(code, "c++")
    program.close()
    assert program.handle is None
