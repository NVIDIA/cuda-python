# Copyright (c) 2024, NVIDIA CORPORATION & AFFILIATES. ALL RIGHTS RESERVED.
#
# SPDX-License-Identifier: LicenseRef-NVIDIA-SOFTWARE-LICENSE

import pytest
import os
from cuda.bindings import nvjitlink

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
    with pytest.raises(nvjitlink.nvJitLinkError, match="ERROR_UNRECOGNIZED_OPTION"):
        nvjitlink.create(1, ["-fictitious_option"])


def test_invalid_arch_error():
    # sm_XX is not a valid architecture
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
    data = ptx_bytes
    nvjitlink.add_data(handle, nvjitlink.InputType.ANY, data, len(data), "test_data")


def test_add_file():
    handle = nvjitlink.create(1, ["-arch=sm_90"])
    file_path = "test_file.cubin"
    with open (file_path, "wb") as f:
        f.write(ptx_bytes)

    nvjitlink.add_file(handle, nvjitlink.InputType.ANY, str(file_path))
    nvjitlink.complete(handle)
    nvjitlink.destroy(handle)
    
    os.remove(file_path)


def test_get_error_log():
    handle = nvjitlink.create(1, ["-arch=sm_90"])
    nvjitlink.complete(handle)
    log_size = nvjitlink.get_error_log_size(handle)
    log = nvjitlink.get_error_log(handle)
    assert len(log) == log_size
    nvjitlink.destroy(handle)


def test_get_info_log():
    handle = nvjitlink.create(1, ["-arch=sm_90"])
    nvjitlink.complete(handle)
    log_size = nvjitlink.get_info_log_size(handle)
    log = nvjitlink.get_info_log(handle)
    assert len(log) == log_size
    nvjitlink.destroy(handle)


def test_get_linked_cubin():
    handle = nvjitlink.create(1, ["-arch=sm_90"])
    nvjitlink.complete(handle)
    cubin_size = nvjitlink.get_linked_cubin_size(handle)
    cubin = nvjitlink.get_linked_cubin(handle)
    assert len(cubin) == cubin_size
    nvjitlink.destroy(handle)


def test_package_version():
    ver = nvjitlink.version()
    assert len(ver) == 2
    assert ver >= (12, 0)
