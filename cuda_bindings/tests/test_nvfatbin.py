# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: LicenseRef-NVIDIA-SOFTWARE-LICENSE


import base64
import shutil
import subprocess

import pytest
from cuda.bindings import nvfatbin, nvrtc

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

CODE = """
int __device__ inc(int x) {
    return x + 1;
}
"""

# This is the base64 encoded TileIR from sample VectorAddition program
TILEIR_b64 = (
    "f1RpbGVJUgANAQAAgssDCAECBgYBCwEECgC/A0QHEAUAEAUBBgQIEAAABgUMAQABBgUIEAAVBgUMAQAC"
    "BgUMAQADBgUIBAAYBgQIEAAFBgUMAQAGBgUIEAAbBgUMAQAHBgUMAQAIBgUIBAAeBgQIEAAKBgUMAQAL"
    "BgUIEAAhBgUMAQAMBgUMAQANBgUIBAAkMAUFBTAFBQVOBQAmEjoIWwgsAwgALi1OBQAqEzoJWwgwCwky"
    "AwkAMzFbCi9bCzQlDQE1Cw43JQ8BFlsNOQsOOg8QAgA4OyUPARlbDT0LDj5ODgA4PyUOATYlDwEXWw1C"
    "Cw5DDxACAEFEBBA8RQMOAEBBWxEUCxJIURJJRxATAlsUSwsVTD0VBxwASkZNESUNATULDlAlDwEcWw1S"
    "Cw5TDxACAFFUJQ8BH1sNVgsOV04OAFFYJQ4BNiUPAR1bDVsLDlwPEAIAWl0EEFVeAw4AWVpbERoLEmFR"
    "EmJgEBMCWxRkCxVlPRUHHABjX2YRAhUAAE5nJQ0BNQsOaiUPASJbDWwLDm0PEAIAa24lDwElWw1wCw5x"
    "Tg4Aa3IlDgE2JQ8BI1sNdQsOdg8QAgB0dwQQb3gDDgBzdFsRIAsSe1ESfHplBwwAfWl5EVwAAIQvCMvL"
    "A8vLy8vLy8sAAAAAAAAAAAUAAAAAAAAACgAAAAAAAAAEAQAAAAQABAAABAAAAACD/QcIy8vLy8sBy8vL"
    "AAAAAGrLy8vLy8vLBAAAAAAAAAAEAAAAAAAAAAQAAAAAAAAABAAAAAAAAAAEAAAAAAAAAAQAAAAAAAAA"
    "BAAAAAAAAAAEAAAAAAAAAAQAAAAAAAAABAAAAAAAAAAEAAAAAAAAAAQAAAAAAAAABAAAAAAAAAAEAAAA"
    "AAAAAAQAAAAAAAAABAAAAAAAAAAEAAAAAAAAAAQAAAAAAAAABAAAAAAAAAAEAAAAAAAAAAQAAAAAAAAA"
    "BAAAAAAAAAAFAAAAAAAAAAYAAAAAAAAABwAAAAAAAAAIAAAAAAAAAAcAAAAAAAAABwAAAAAAAAAJAAAA"
    "AAAAAAoAAAAAAAAACQAAAAAAAAAJAAAAAAAAAAkAAAAAAAAACwAAAAAAAAAMAAAAAAAAAA0AAAAAAAAA"
    "DQAAAAAAAAANAAAAAAAAAA0AAAAAAAAADQAAAAAAAAANAAAAAAAAAA0AAAAAAAAADQAAAAAAAAANAAAA"
    "AAAAAA0AAAAAAAAADQAAAAAAAAANAAAAAAAAAA0AAAAAAAAADQAAAAAAAAANAAAAAAAAAA0AAAAAAAAA"
    "DQAAAAAAAAANAAAAAAAAAA0AAAAAAAAADQAAAAAAAAANAAAAAAAAAA0AAAAAAAAADQAAAAAAAAANAAAA"
    "AAAAAA4AAAAAAAAADgAAAAAAAAAOAAAAAAAAAA4AAAAAAAAADgAAAAAAAAAOAAAAAAAAAA4AAAAAAAAA"
    "DgAAAAAAAAAOAAAAAAAAAA4AAAAAAAAADgAAAAAAAAAOAAAAAAAAAA4AAAAAAAAADgAAAAAAAAAOAAAA"
    "AAAAAA4AAAAAAAAADgAAAAAAAAAOAAAAAAAAAA4AAAAAAAAADgAAAAAAAAAOAAAAAAAAAA4AAAAAAAAA"
    "DgAAAAAAAAAOAAAAAAAAAA8AAAAAAAAAEAAAAAAAAAAQAAAAAAAAABAAAAAAAAAAEAAAAAAAAAAQAAAA"
    "AAAAABAAAAAAAAAAEAAAAAAAAAAQAAAAAAAAABAAAAAAAAAAEAAAAAAAAAAQAAAAAAAAABAAAAAAAAAA"
    "EAAAAAAAAAAQAAAAAAAAABAAAAAAAAAAEAAAAAAAAAAQAAAAAAAAABAAAAAAAAAAEAAAAAAAAAAQAAAA"
    "AAAAABAAAAAAAAAABAAAAAAAAAAQy8vLAAAAAAMAAAAFAAAADAAAABEAAAAXAAAAHQAAACMAAAApAAAA"
    "LwAAADUAAAA7AAAAQQAAAEcAAABNAAAAUwAAAAIAAQEBBQF9AgICfQQDA34ABAMDkQEMBAMDkgEMBAMD"
    "lQEIBAMDlQEZBAMDlgEIBAMDlgEZBAMDmgEIBAMDmwEIBAMDnwENBAMDoAENBAMDowEPBAMDpwEEhcQC"
    "BMvLyxbLy8sAAAAAAQAAAAIAAAADAAAABQAAAAgAAAALAAAAHwAAACAAAAArAAAANgAAAEkAAABcAAAA"
    "XQAAAHAAAACDAAAAhgAAAJkAAACsAAAAvwAAAMIAAADVAAAAAAMHDAINAwANAQAQEQQFBQUFBAUFBQUE"
    "BQUFBQUFABENAQEBAAAAAAAAAA0BAQAEAAAAAAAADQECAQAAAAAAAAABAAAAAAAAAA0BAgEAAAAAAAAA"
    "AAQAAAAAAAAEDQwCAQAAAAAAAAABAAAAAAAAAA0MAgEAAAAAAAAAAAQAAAAAAAANDAANAAIBAAAAAAAA"
    "AAAEAAAAAAAADQMCAQAAAAAAAAABAAAAAAAAAA0DAgEAAAAAAAAAAAQAAAAAAAANAgANAgIBAAAAAAAA"
    "AAEAAAAAAAAADQICAQAAAAAAAAAABAAAAAAAAIGxAQQFy8vLAAAAABEAAAA9AAAAVQAAAJMAAABWZWN0"
    "b3JBZGRpdGlvbi5weS9sb2NhbGhvbWUvbG9jYWwtd2FuZ20vY3V0aWxlLXB5dGhvbi9zYW1wbGVzdmVj"
    "X2FkZF9rZXJuZWxfMmRfZ2F0aGVyL2xvY2FsaG9tZS9sb2NhbC13YW5nbS9jdXRpbGUtcHl0aG9uL3Nh"
    "bXBsZXMvVmVjdG9yQWRkaXRpb24ucHlzbV8xMjAA"
)


@pytest.fixture(params=ARCHITECTURES)
def arch(request):
    return request.param


@pytest.fixture(params=PTX_VERSIONS)
def ptx_version(request):
    return request.param


@pytest.fixture
def PTX(arch, ptx_version):
    return PTX_TEMPLATE.format(PTX_VERSION=ptx_version, ARCH=arch)


@pytest.fixture
def CUBIN(arch):
    def CHECK_NVRTC(err):
        if err != nvrtc.nvrtcResult.NVRTC_SUCCESS:
            raise RuntimeError(repr(err))

    err, program_handle = nvrtc.nvrtcCreateProgram(CODE.encode(), b"", 0, [], [])
    CHECK_NVRTC(err)
    err = nvrtc.nvrtcCompileProgram(program_handle, 1, [f"-arch={arch}".encode()])[0]
    CHECK_NVRTC(err)
    err, size = nvrtc.nvrtcGetCUBINSize(program_handle)
    CHECK_NVRTC(err)
    cubin = b" " * size
    (err,) = nvrtc.nvrtcGetCUBIN(program_handle, cubin)
    CHECK_NVRTC(err)
    (err,) = nvrtc.nvrtcDestroyProgram(program_handle)
    CHECK_NVRTC(err)
    return cubin


# create a valid LTOIR input for testing
@pytest.fixture
def LTOIR(arch):
    arch = arch.replace("sm", "compute")

    def CHECK_NVRTC(err):
        if err != nvrtc.nvrtcResult.NVRTC_SUCCESS:
            raise RuntimeError(repr(err))

    empty_cplusplus_kernel = "__global__ void A() {}"
    err, program_handle = nvrtc.nvrtcCreateProgram(empty_cplusplus_kernel.encode(), b"", 0, [], [])
    CHECK_NVRTC(err)
    err = nvrtc.nvrtcCompileProgram(program_handle, 1, [b"-dlto", f"-arch={arch}".encode()])[0]
    CHECK_NVRTC(err)
    err, size = nvrtc.nvrtcGetLTOIRSize(program_handle)
    CHECK_NVRTC(err)
    empty_kernel_ltoir = b" " * size
    (err,) = nvrtc.nvrtcGetLTOIR(program_handle, empty_kernel_ltoir)
    CHECK_NVRTC(err)
    (err,) = nvrtc.nvrtcDestroyProgram(program_handle)
    CHECK_NVRTC(err)
    return empty_kernel_ltoir


@pytest.fixture
def OBJECT(arch, tmpdir):
    empty_cplusplus_kernel = "__global__ void A() {} int main() { return 0; }"
    with open(tmpdir / "object.cu", "w") as f:
        f.write(empty_cplusplus_kernel)

    nvcc = shutil.which("nvcc")
    if nvcc is None:
        pytest.skip("nvcc not found on PATH")

    # This is a test fixture that intentionally invokes a trusted tool (`nvcc`) to
    # compile a temporary CUDA translation unit.
    subprocess.run(  # noqa: S603
        [nvcc, "-arch", arch, "-o", str(tmpdir / "object.o"), str(tmpdir / "object.cu")],
        check=True,
        capture_output=True,
    )
    with open(tmpdir / "object.o", "rb") as f:
        object = f.read()

    return object


@pytest.fixture
def TILEIR(tmpdir):
    binary_data = base64.b64decode(TILEIR_b64)
    return binary_data


@pytest.mark.parametrize("error_enum", nvfatbin.Result)
def test_get_error_string(error_enum):
    es = nvfatbin.get_error_string(error_enum)

    if error_enum is nvfatbin.Result.SUCCESS:
        assert es == ""
    else:
        assert es != ""


def test_nvfatbin_get_version():
    major, minor = nvfatbin.version()
    assert major is not None
    assert minor is not None


def test_nvfatbin_empty_create_and_destroy():
    handle = nvfatbin.create([], 0)
    assert handle is not None
    nvfatbin.destroy(handle)


def test_nvfatbin_invalid_input_create():
    with pytest.raises(nvfatbin.nvFatbinError, match="ERROR_UNRECOGNIZED_OPTION"):
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


@pytest.mark.parametrize("arch", ["sm_80"], indirect=True)
def test_nvfatbin_add_cubin_ELF_SIZE_MISMATCH(CUBIN, arch):
    handle = nvfatbin.create([], 0)
    with pytest.raises(nvfatbin.nvFatbinError, match="ERROR_ELF_ARCH_MISMATCH"):
        nvfatbin.add_cubin(handle, CUBIN, len(CUBIN), "75", "inc")

    nvfatbin.destroy(handle)


def test_nvfatbin_add_cubin(CUBIN, arch):
    arch_numeric = arch.split("_")[1]

    handle = nvfatbin.create([], 0)
    nvfatbin.add_cubin(handle, CUBIN, len(CUBIN), arch_numeric, "inc")

    buffer = bytearray(nvfatbin.size(handle))

    nvfatbin.get(handle, buffer)
    nvfatbin.destroy(handle)


@pytest.mark.parametrize("arch", ["sm_80"], indirect=True)
def test_nvfatbin_add_cubin_ELF_ARCH_MISMATCH(CUBIN, arch):
    handle = nvfatbin.create([], 0)
    with pytest.raises(nvfatbin.nvFatbinError, match="ERROR_ELF_ARCH_MISMATCH"):
        nvfatbin.add_cubin(handle, CUBIN, len(CUBIN), "75", "inc")

    nvfatbin.destroy(handle)


def test_nvdfatbin_add_ltoir(LTOIR, arch):
    arch_numeric = arch.split("_")[1]

    handle = nvfatbin.create([], 0)
    nvfatbin.add_ltoir(handle, LTOIR, len(LTOIR), arch_numeric, "inc", "")

    buffer = bytearray(nvfatbin.size(handle))

    nvfatbin.get(handle, buffer)
    nvfatbin.destroy(handle)


def test_nvfatbin_add_reloc(OBJECT):
    handle = nvfatbin.create([], 0)
    nvfatbin.add_reloc(handle, OBJECT, len(OBJECT))

    buffer = bytearray(nvfatbin.size(handle))

    nvfatbin.get(handle, buffer)
    nvfatbin.destroy(handle)


def test_nvfatbin_add_tile_ir(TILEIR):
    handle = nvfatbin.create([], 0)
    nvfatbin.add_tile_ir(handle, TILEIR, len(TILEIR), "VectorAdd", "")

    buffer = bytearray(nvfatbin.size(handle))

    nvfatbin.get(handle, buffer)
    nvfatbin.destroy(handle)
