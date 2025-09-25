# SPDX-FileCopyrightText: Copyright (c) 2024-2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: LicenseRef-NVIDIA-SOFTWARE-LICENSE

import pytest
from cuda.bindings import nvjitlink, nvrtc

# Establish a handful of compatible architectures and PTX versions to test with
ARCHITECTURES = ["sm_75", "sm_80", "sm_90", "sm_100"]
PTX_VERSIONS = ["6.4", "7.0", "8.5", "8.8"]


PTX_HEADER = """\
.version {VERSION}
.target {ARCH}
.address_size 64
"""

PTX_KERNEL = """
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


def _build_arch_ptx_parametrized_callable():
    av = tuple(zip(ARCHITECTURES, PTX_VERSIONS))
    return pytest.mark.parametrize(
        ("arch", "ptx_bytes"),
        [(a, (PTX_HEADER.format(VERSION=v, ARCH=a) + PTX_KERNEL).encode("utf-8")) for a, v in av],
        ids=[f"{a}_{v}" for a, v in av],
    )


ARCH_PTX_PARAMETRIZED_CALLABLE = _build_arch_ptx_parametrized_callable()


def arch_ptx_parametrized(func):
    return ARCH_PTX_PARAMETRIZED_CALLABLE(func)


def check_nvjitlink_usable():
    from cuda.bindings._internal import nvjitlink as inner_nvjitlink

    return inner_nvjitlink._inspect_function_pointer("__nvJitLinkVersion") != 0


pytestmark = pytest.mark.skipif(
    not check_nvjitlink_usable(), reason="nvJitLink not usable, maybe not installed or too old (<12.3)"
)


# create a valid LTOIR input for testing
@pytest.fixture
def get_dummy_ltoir():
    def CHECK_NVRTC(err):
        if err != nvrtc.nvrtcResult.NVRTC_SUCCESS:
            raise RuntimeError(repr(err))

    empty_cplusplus_kernel = "__global__ void A() {}"
    err, program_handle = nvrtc.nvrtcCreateProgram(empty_cplusplus_kernel.encode(), b"", 0, [], [])
    CHECK_NVRTC(err)
    err = nvrtc.nvrtcCompileProgram(program_handle, 1, [b"-dlto"])[0]
    CHECK_NVRTC(err)
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


@arch_ptx_parametrized
def test_add_data(arch, ptx_bytes):
    handle = nvjitlink.create(1, [f"-arch={arch}"])
    nvjitlink.add_data(handle, nvjitlink.InputType.ANY, ptx_bytes, len(ptx_bytes), "test_data")
    nvjitlink.complete(handle)
    nvjitlink.destroy(handle)


@arch_ptx_parametrized
def test_add_file(arch, ptx_bytes, tmp_path):
    handle = nvjitlink.create(1, [f"-arch={arch}"])
    file_path = tmp_path / "test_file.cubin"
    file_path.write_bytes(ptx_bytes)
    nvjitlink.add_file(handle, nvjitlink.InputType.ANY, str(file_path))
    nvjitlink.complete(handle)
    nvjitlink.destroy(handle)


@pytest.mark.parametrize("arch", ARCHITECTURES)
def test_get_error_log(arch):
    handle = nvjitlink.create(1, [f"-arch={arch}"])
    nvjitlink.complete(handle)
    log_size = nvjitlink.get_error_log_size(handle)
    log = bytearray(log_size)
    nvjitlink.get_error_log(handle, log)
    assert len(log) == log_size
    nvjitlink.destroy(handle)


@arch_ptx_parametrized
def test_get_info_log(arch, ptx_bytes):
    handle = nvjitlink.create(1, [f"-arch={arch}"])
    nvjitlink.add_data(handle, nvjitlink.InputType.ANY, ptx_bytes, len(ptx_bytes), "test_data")
    nvjitlink.complete(handle)
    log_size = nvjitlink.get_info_log_size(handle)
    log = bytearray(log_size)
    nvjitlink.get_info_log(handle, log)
    assert len(log) == log_size
    nvjitlink.destroy(handle)


@arch_ptx_parametrized
def test_get_linked_cubin(arch, ptx_bytes):
    handle = nvjitlink.create(1, [f"-arch={arch}"])
    nvjitlink.add_data(handle, nvjitlink.InputType.ANY, ptx_bytes, len(ptx_bytes), "test_data")
    nvjitlink.complete(handle)
    cubin_size = nvjitlink.get_linked_cubin_size(handle)
    cubin = bytearray(cubin_size)
    nvjitlink.get_linked_cubin(handle, cubin)
    assert len(cubin) == cubin_size
    nvjitlink.destroy(handle)


@pytest.mark.parametrize("arch", ARCHITECTURES)
def test_get_linked_ptx(arch, get_dummy_ltoir):
    handle = nvjitlink.create(3, [f"-arch={arch}", "-lto", "-ptx"])
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
