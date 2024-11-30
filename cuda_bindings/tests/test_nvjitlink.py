# Copyright (c) 2024, NVIDIA CORPORATION & AFFILIATES. ALL RIGHTS RESERVED.
#
# SPDX-License-Identifier: LicenseRef-NVIDIA-SOFTWARE-LICENSE

import pytest

from cuda.bindings import nvjitlink, nvrtc

# Establish a handful of compatible architectures and PTX versions to test with
ARCHITECTURES = ["sm_60", "sm_75", "sm_80", "sm_90"]
PTX_VERSIONS = ["5.0", "6.4", "7.0", "8.5"]


def ptx_header(version, arch):
    return f"""
.version {version}
.target {arch}
.address_size 64
"""


ptx_kernel = """
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
.func _MinimalKernel()
{
    ret;
}
"""

ptx_kernel_bytes = [
    (ptx_header(version, arch) + ptx_kernel).encode("utf-8") for version, arch in zip(PTX_VERSIONS, ARCHITECTURES)
]
minimal_ptx_kernel_bytes = [
    (ptx_header(version, arch) + minimal_ptx_kernel).encode("utf-8")
    for version, arch in zip(PTX_VERSIONS, ARCHITECTURES)
]


# create a valid LTOIR input for testing
@pytest.fixture
def get_dummy_ltoir():
    def CHECK_NVRTC(err):
        if err != nvrtc.nvrtcResult.NVRTC_SUCCESS:
            raise RuntimeError(f"Nvrtc Error: {err}")

    empty_cplusplus_kernel = "__global__ void A() {}"
    err, program_handle = nvrtc.nvrtcCreateProgram(empty_cplusplus_kernel.encode(), b"", 0, [], [])
    CHECK_NVRTC(err)
    nvrtc.nvrtcCompileProgram(program_handle, 1, [b"-dlto"])
    err, size = nvrtc.nvrtcGetLTOIRSize(program_handle)
    CHECK_NVRTC(err)
    empty_kernel_ltoir = b" " * size
    (err,) = nvrtc.nvrtcGetLTOIR(program_handle, empty_kernel_ltoir)
    CHECK_NVRTC(err)
    (err,) = nvrtc.nvrtcDestroyProgram(program_handle)
    CHECK_NVRTC(err)
    return empty_kernel_ltoir


def test_unrecognized_option_error():
    with pytest.raises(nvjitlink.nvJitLinkError, match="ERROR_UNRECOGNIZED_OPTION"):
        nvjitlink.create(1, ["-fictitious_option"])


def test_invalid_arch_error():
    with pytest.raises(nvjitlink.nvJitLinkError, match="ERROR_UNRECOGNIZED_OPTION"):
        nvjitlink.create(1, ["-arch=sm_XX"])


@pytest.mark.parametrize("option", ARCHITECTURES)
def test_create_and_destroy(option):
    handle = nvjitlink.create(1, [f"-arch={option}"])
    assert handle != 0
    nvjitlink.destroy(handle)


@pytest.mark.parametrize("option", ARCHITECTURES)
def test_complete_empty(option):
    handle = nvjitlink.create(1, [f"-arch={option}"])
    nvjitlink.complete(handle)
    nvjitlink.destroy(handle)


@pytest.mark.parametrize("option, ptx_bytes", zip(ARCHITECTURES, ptx_kernel_bytes))
def test_add_data(option, ptx_bytes):
    handle = nvjitlink.create(1, [f"-arch={option}"])
    nvjitlink.add_data(handle, nvjitlink.InputType.ANY, ptx_bytes, len(ptx_bytes), "test_data")
    nvjitlink.complete(handle)
    nvjitlink.destroy(handle)


@pytest.mark.parametrize("option, ptx_bytes", zip(ARCHITECTURES, ptx_kernel_bytes))
def test_add_file(option, ptx_bytes, tmp_path):
    handle = nvjitlink.create(1, [f"-arch={option}"])
    file_path = tmp_path / "test_file.cubin"
    file_path.write_bytes(ptx_bytes)
    nvjitlink.add_file(handle, nvjitlink.InputType.ANY, str(file_path))
    nvjitlink.complete(handle)
    nvjitlink.destroy(handle)


@pytest.mark.parametrize("option", ARCHITECTURES)
def test_get_error_log(option):
    handle = nvjitlink.create(1, [f"-arch={option}"])
    nvjitlink.complete(handle)
    log_size = nvjitlink.get_error_log_size(handle)
    log = bytearray(log_size)
    nvjitlink.get_error_log(handle, log)
    assert len(log) == log_size
    nvjitlink.destroy(handle)


@pytest.mark.parametrize("option, ptx_bytes", zip(ARCHITECTURES, ptx_kernel_bytes))
def test_get_info_log(option, ptx_bytes):
    handle = nvjitlink.create(1, [f"-arch={option}"])
    nvjitlink.add_data(handle, nvjitlink.InputType.ANY, ptx_bytes, len(ptx_bytes), "test_data")
    nvjitlink.complete(handle)
    log_size = nvjitlink.get_info_log_size(handle)
    log = bytearray(log_size)
    nvjitlink.get_info_log(handle, log)
    assert len(log) == log_size
    nvjitlink.destroy(handle)


@pytest.mark.parametrize("option, ptx_bytes", zip(ARCHITECTURES, ptx_kernel_bytes))
def test_get_linked_cubin(option, ptx_bytes):
    handle = nvjitlink.create(1, [f"-arch={option}"])
    nvjitlink.add_data(handle, nvjitlink.InputType.ANY, ptx_bytes, len(ptx_bytes), "test_data")
    nvjitlink.complete(handle)
    cubin_size = nvjitlink.get_linked_cubin_size(handle)
    cubin = bytearray(cubin_size)
    nvjitlink.get_linked_cubin(handle, cubin)
    assert len(cubin) == cubin_size
    nvjitlink.destroy(handle)


@pytest.mark.parametrize("option", ARCHITECTURES)
def test_get_linked_ptx(option, get_dummy_ltoir):
    handle = nvjitlink.create(3, [f"-arch={option}", "-lto", "-ptx"])
    nvjitlink.add_data(handle, nvjitlink.InputType.LTOIR, get_dummy_ltoir, len(get_dummy_ltoir), "test_data")
    nvjitlink.complete(handle)
    ptx_size = nvjitlink.get_linked_ptx_size(handle)
    ptx = bytearray(ptx_size)
    nvjitlink.get_linked_ptx(handle, ptx)
    assert len(ptx) == ptx_size
    nvjitlink.destroy(handle)


def test_package_version():
    ver = nvjitlink.version()
    assert len(ver) == 2
    assert ver >= (12, 0)
