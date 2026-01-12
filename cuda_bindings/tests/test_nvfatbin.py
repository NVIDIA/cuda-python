# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: LicenseRef-NVIDIA-SOFTWARE-LICENSE

from cuda.bindings import nvfatbin

import pytest

ARCHITECTURES = ["sm_75", "sm_80", "sm_90", "sm_100"]
PTX_VERSIONS = ["6.4", "7.0", "8.5", "8.8"]

PTX_TEMPLATE = """
.version {PTX_VERSION}
.target {ARCH}
.address_size 64

        // .globl       _Z6kernelPi

.visible .entry _Z6kernelPi(
        .param .u64 _Z6kernelPi_param_0
)
{{
        .reg .b32       %r<7>;
        .reg .b64       %rd<5>;


        ld.param.u64    %rd1, [_Z6kernelPi_param_0];
        cvta.to.global.u64      %rd2, %rd1;
        mov.u32         %r1, %tid.x;
        mov.u32         %r2, %ctaid.x;
        mov.u32         %r3, %ntid.x;
        mad.lo.s32      %r4, %r2, %r3, %r1;
        mul.wide.s32    %rd3, %r4, 4;
        add.s64         %rd4, %rd2, %rd3;
        ld.global.u32   %r5, [%rd4];
        add.s32         %r6, %r5, 1;
        st.global.u32   [%rd4], %r6;
        ret;

}}
"""

@pytest.fixture(params=ARCHITECTURES)
def arch(request):
    return request.param

@pytest.fixture(params=PTX_VERSIONS)
def ptx_version(request):
    return request.param

@pytest.fixture
def PTX(arch, ptx_version):
    return PTX_TEMPLATE.format(PTX_VERSION=ptx_version, ARCH=arch)

def test_nvfatbin_get_version():
    major, minor = nvfatbin.version()
    assert major is not None
    assert minor is not None

def test_nvfatbin_empty_create_and_destroy():
    handle = nvfatbin.create([], 0)
    assert handle is not None
    nvfatbin.destroy(handle)

def test_nvfatbin_invalid_input_create():
    with pytest.raises(nvfatbin.nvfatbinError, match="ERROR_UNRECOGNIZED_OPTION"):
        nvfatbin.create(["--unsupported_option"], 1)


def test_nvfatbin_get_empty():
    handle = nvfatbin.create([], 0)
    size = nvfatbin.size(handle)

    buffer = bytearray(size)
    nvfatbin.get(handle, buffer)

    nvfatbin.destroy(handle)


def test_nvfatbin_add_ptx(PTX, arch):
    arch_numeric = arch.split("_")[1]

    handle = nvfatbin.create([], 0)
    nvfatbin.add_ptx(handle, PTX.encode(), len(PTX), arch_numeric, "add", f"-arch={arch}")

    buffer = bytearray(nvfatbin.size(handle))

    nvfatbin.get(handle, buffer)
    nvfatbin.destroy(handle)

