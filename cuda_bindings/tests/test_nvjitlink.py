# Copyright 2021-2024 NVIDIA Corporation.  All rights reserved.
#
# Please refer to the NVIDIA end user license agreement (EULA) associated
# with this source code for terms and conditions that govern your use of
# this software. Any use, reproduction, disclosure, or distribution of
# this software and related documentation outside the terms of the EULA
# is strictly prohibited.

import pytest
import os
from cuda.bindings import nvjitlink


ptx_kernel = """
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

minimal_ptx_kernel = """
.version 8.5
.target sm_90
.address_size 64

.func _MinimalKernel()
{
    ret;
}
"""

ptx_kernel_bytes = ptx_kernel.encode('utf-8')
minimal_ptx_kernel_bytes = minimal_ptx_kernel.encode('utf-8')

def test_unrecognized_option_error():
    with pytest.raises(nvjitlink.nvJitLinkError, match="ERROR_UNRECOGNIZED_OPTION"):
        nvjitlink.create(1, ["-fictitious_option"])


def test_invalid_arch_error():
    with pytest.raises(nvjitlink.nvJitLinkError, match="ERROR_UNRECOGNIZED_OPTION"):
        nvjitlink.create(1, ["-arch=sm_XX"])


def test_create_and_destroy():
    handle = nvjitlink.create(1, ["-arch=sm_53"])
    assert handle != 0
    nvjitlink.destroy(handle)


def test_complete_empty():
    handle = nvjitlink.create(1, ["-arch=sm_90"])
    nvjitlink.complete(handle)
    nvjitlink.destroy(handle)


def test_add_data():
    handle = nvjitlink.create(1, ["-arch=sm_90"])
    nvjitlink.add_data(handle, nvjitlink.InputType.ANY, ptx_kernel_bytes, len(ptx_kernel_bytes), "test_data")
    nvjitlink.add_data(handle, nvjitlink.InputType.ANY, minimal_ptx_kernel_bytes, len(minimal_ptx_kernel_bytes), "minimal_test_data")
    nvjitlink.complete(handle)
    nvjitlink.destroy(handle)


def test_add_file():
    handle = nvjitlink.create(1, ["-arch=sm_90"])
    file_path = "test_file.cubin"
    with open (file_path, "wb") as f:
        f.write(ptx_kernel_bytes)

    nvjitlink.add_file(handle, nvjitlink.InputType.ANY, str(file_path))
    nvjitlink.complete(handle)
    nvjitlink.destroy(handle)
    os.remove(file_path)


def test_get_error_log():
    handle = nvjitlink.create(1, ["-arch=sm_90"])
    nvjitlink.complete(handle)
    log_size = nvjitlink.get_error_log_size(handle)
    log = bytearray(log_size)
    nvjitlink.get_error_log(handle, log)
    assert len(log) == log_size
    nvjitlink.destroy(handle)


def test_get_info_log():
    handle = nvjitlink.create(1, ["-arch=sm_90"])
    nvjitlink.add_data(handle, nvjitlink.InputType.ANY, ptx_kernel_bytes, len(ptx_kernel_bytes), "test_data")
    nvjitlink.complete(handle)
    log_size = nvjitlink.get_info_log_size(handle)
    log = bytearray(log_size)
    nvjitlink.get_info_log(handle, log)
    assert len(log) == log_size
    nvjitlink.destroy(handle)


def test_get_linked_cubin():
    handle = nvjitlink.create(1, ["-arch=sm_90"])
    nvjitlink.add_data(handle, nvjitlink.InputType.ANY, ptx_kernel_bytes, len(ptx_kernel_bytes), "test_data")
    nvjitlink.complete(handle)
    cubin_size = nvjitlink.get_linked_cubin_size(handle)
    cubin = bytearray(cubin_size)
    nvjitlink.get_linked_cubin(handle, cubin)
    assert len(cubin) == cubin_size
    nvjitlink.destroy(handle)


def test_get_linked_ptx():
    # TODO improve this test to call get_linked_ptx without this error
    handle = nvjitlink.create(2, ["-arch=sm_90", "-lto"])
    with pytest.raises(nvjitlink.nvJitLinkError, match="ERROR_NVVM_COMPILE"):
        nvjitlink.complete(handle)


def test_package_version():
    ver = nvjitlink.version()
    assert len(ver) == 2
    assert ver >= (12, 0)
