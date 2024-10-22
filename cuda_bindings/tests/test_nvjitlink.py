# Copyright 2021-2024 NVIDIA Corporation.  All rights reserved.
#
# Please refer to the NVIDIA end user license agreement (EULA) associated
# with this source code for terms and conditions that govern your use of
# this software. Any use, reproduction, disclosure, or distribution of
# this software and related documentation outside the terms of the EULA
# is strictly prohibited.

import pytest
import os
import cuda.bindings



ptx_code = """
.version 8.5
.target sm_90
.address_size 64

.visible .entry _Z6kernelPi(
    .param .u64 _Z6kernelPi_param_0
)
{
    .reg .pred  %p<2>;
    .reg .b32   %r<3>;
    .reg .b64   %rd<3>;

    ld.param.u64    %rd1, [_Z6kernelPi_param_0];
    cvta.to.global.u64  %rd2, %rd1;
    mov.u32     %r1, %tid.x;
    st.global.u32   [%rd2+0], %r1;
    ret;
}
"""

minimal_kernel = """
.version 6.4
.target sm_75
.address_size 64

.visible .entry _kernel() {
    ret;
}
"""

# Convert PTX code to bytes
ptx_bytes = ptx_code.encode('utf-8')
minimal_kernel_bytes = minimal_kernel.encode('utf-8')

def test_unrecognized_option_error():
    with pytest.raises(cuda.bindings.nvjitlink.nvJitLinkError, match="ERROR_UNRECOGNIZED_OPTION"):
        cuda.bindings.nvjitlink.create(1, ["-fictitious_option"])


def test_invalid_arch_error():
    # sm_XX is not a valid architecture
    with pytest.raises(cuda.bindings.nvjitlink.nvJitLinkError, match="ERROR_UNRECOGNIZED_OPTION"):
        cuda.bindings.nvjitlink.create(1, ["-arch=sm_XX"])


def test_create_and_destroy():
    handle = cuda.bindings.nvjitlink.create(1, ["-arch=sm_53"])
    assert handle != 0
    cuda.bindings.nvjitlink.destroy(handle)


def test_complete_empty():
    handle = cuda.bindings.nvjitlink.create(1, ["-arch=sm_90"])
    cuda.bindings.nvjitlink.complete(handle)
    cuda.bindings.nvjitlink.destroy(handle)

def test_add_data():
    handle = cuda.bindings.nvjitlink.create(1, ["-arch=sm_90"])
    data = ptx_bytes
    cuda.bindings.nvjitlink.add_data(handle, cuda.bindings.nvjitlink.InputType.ANY, data, len(data), "test_data")
    cuda.bindings.nvjitlink.complete(handle)
    cuda.bindings.nvjitlink.destroy(handle)


def test_add_file():
    handle = cuda.bindings.nvjitlink.create(1, ["-arch=sm_90"])
    file_path = "test_file.cubin"
    with open (file_path, "wb") as f:
        f.write(ptx_bytes)

    cuda.bindings.nvjitlink.add_file(handle, cuda.bindings.nvjitlink.InputType.ANY, str(file_path))
    cuda.bindings.nvjitlink.complete(handle)
    cuda.bindings.nvjitlink.destroy(handle)
    
    os.remove(file_path)


def test_get_error_log():
    handle = cuda.bindings.nvjitlink.create(1, ["-arch=sm_90"])
    cuda.bindings.nvjitlink.complete(handle)
    log_size = cuda.bindings.nvjitlink.get_error_log_size(handle)
    log = bytearray(log_size)
    cuda.bindings.nvjitlink.get_error_log(handle, log)
    assert len(log) == log_size
    cuda.bindings.nvjitlink.destroy(handle)


def test_get_info_log():
    handle = cuda.bindings.nvjitlink.create(1, ["-arch=sm_90"])
    cuda.bindings.nvjitlink.complete(handle)
    log_size = cuda.bindings.nvjitlink.get_info_log_size(handle)
    log = bytearray(log_size)
    cuda.bindings.nvjitlink.get_info_log(handle, log)
    assert len(log) == log_size
    cuda.bindings.nvjitlink.destroy(handle)


def test_get_linked_cubin():
    handle = cuda.bindings.nvjitlink.create(1, ["-arch=sm_90"])
    cuda.bindings.nvjitlink.complete(handle)
    cubin_size = cuda.bindings.nvjitlink.get_linked_cubin_size(handle)
    cubin = bytearray(cubin_size)
    cuda.bindings.nvjitlink.get_linked_cubin(handle, cubin)
    assert len(cubin) == cubin_size
    cuda.bindings.nvjitlink.destroy(handle)

#TODO add a ptx test

def test_package_version():
    ver = cuda.bindings.nvjitlink.version()
    assert len(ver) == 2
    assert ver >= (12, 0)