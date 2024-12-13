# Copyright 2024 NVIDIA Corporation.  All rights reserved.
#
# Please refer to the NVIDIA end user license agreement (EULA) associated
# with this source code for terms and conditions that govern your use of
# this software. Any use, reproduction, disclosure, or distribution of
# this software and related documentation outside the terms of the EULA
# is strictly prohibited.


import pytest
from conftest import can_load_generated_ptx

from cuda.core.experimental import Program


@pytest.mark.xfail(not can_load_generated_ptx(), reason="PTX version too new")
def test_get_kernel():
    kernel = """
extern __device__ int B();
extern __device__ int C(int a, int b);
__global__ void A() { int result = C(B(), 1);}
"""
    object_code = Program(kernel, "c++").compile("ptx", options=("-rdc=true",))
    assert object_code._handle is None
    kernel = object_code.get_kernel("A")
    assert object_code._handle is not None
    assert kernel._handle is not None
