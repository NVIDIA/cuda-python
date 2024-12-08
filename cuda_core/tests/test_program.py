# Copyright 2024 NVIDIA Corporation.  All rights reserved.
#
# Please refer to the NVIDIA end user license agreement (EULA) associated
# with this source code for terms and conditions that govern your use of
# this software. Any use, reproduction, disclosure, or distribution of
# this software and related documentation outside the terms of the EULA
# is strictly prohibited.

import pytest

from cuda import cuda, nvrtc
from cuda.core.experimental import Device, Program
from cuda.core.experimental._module import Kernel, ObjectCode


@pytest.fixture
def can_load_generated_ptx():
    _, driver_ver = cuda.cuDriverGetVersion()
    _, nvrtc_major, nvrtc_minor = nvrtc.nvrtcVersion()
    if nvrtc_major * 1000 + nvrtc_minor * 10 > driver_ver:
        return False
    return True


def test_program_init_valid_code_type():
    code = 'extern "C" __global__ void my_kernel() {}'
    program = Program(code, "c++")
    assert program.backend == "nvrtc"
    assert program.handle is not None


def test_program_init_invalid_code_type():
    code = 'extern "C" __global__ void my_kernel() {}'
    with pytest.raises(NotImplementedError):
        Program(code, "python")


def test_program_init_invalid_code_format():
    code = 12345
    with pytest.raises(TypeError):
        Program(code, "c++")


# TODO: incorporate this check in Program
@pytest.mark.xfail(not can_load_generated_ptx, reason="PTX version too new")
def test_program_compile_valid_target_type():
    code = 'extern "C" __global__ void my_kernel() {}'
    program = Program(code, "c++")
    arch = "".join(str(i) for i in Device().compute_capability)
    object_code = program.compile("ptx", options=(f"-arch=compute_{arch}",))
    print(object_code._module.decode())
    kernel = object_code.get_kernel("my_kernel")
    assert isinstance(object_code, ObjectCode)
    assert isinstance(kernel, Kernel)


def test_program_compile_invalid_target_type():
    code = 'extern "C" __global__ void my_kernel() {}'
    program = Program(code, "c++")
    with pytest.raises(NotImplementedError):
        program.compile("invalid_target")


def test_program_backend_property():
    code = 'extern "C" __global__ void my_kernel() {}'
    program = Program(code, "c++")
    assert program.backend == "nvrtc"


def test_program_handle_property():
    code = 'extern "C" __global__ void my_kernel() {}'
    program = Program(code, "c++")
    assert program.handle is not None


def test_program_close():
    code = 'extern "C" __global__ void my_kernel() {}'
    program = Program(code, "c++")
    program.close()
    assert program.handle is None
