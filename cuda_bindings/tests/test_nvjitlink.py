# Copyright (c) 2024, NVIDIA CORPORATION & AFFILIATES. ALL RIGHTS RESERVED.
#
# SPDX-License-Identifier: LicenseRef-NVIDIA-SOFTWARE-LICENSE

import pytest

from cuda.bindings import nvjitlink


def test_invalid_arch_error():
    # sm_XX is not a valid architecture
    with pytest.raises(nvjitlink.nvJitLinkError, match="ERROR_UNRECOGNIZED_OPTION"):
        nvjitlink.create(1, ["-arch=sm_XX"])


def test_unrecognized_option_error():
    with pytest.raises(nvjitlink.nvJitLinkError, match="ERROR_UNRECOGNIZED_OPTION"):
        nvjitlink.create(1, ["-fictitious_option"])


def test_invalid_option_type_error():
    with pytest.raises(TypeError, match="Expecting only strings"):
        nvjitlink.create("-arch", 53)


def test_create_and_destroy():
    handle = nvjitlink.create("-arch=sm_53")
    assert handle != 0
    nvjitlink.destroy(handle)


def test_complete_empty():
    handle = nvjitlink.create("-arch=sm_75")
    nvjitlink.complete(handle)
    nvjitlink.destroy(handle)


@pytest.mark.parametrize(
    "input_file,input_type",
    [
        ("device_functions_cubin", nvjitlink.InputType.CUBIN),
        ("device_functions_fatbin", nvjitlink.InputType.FATBIN),
        ("device_functions_ptx", nvjitlink.InputType.PTX),
        ("device_functions_object", nvjitlink.InputType.OBJECT),
        ("device_functions_archive", nvjitlink.InputType.LIBRARY),
    ],
)
def test_add_file(input_file, input_type, gpu_arch_flag, request):
    filename, data = request.getfixturevalue(input_file)

    handle = nvjitlink.create(gpu_arch_flag)
    nvjitlink.add_data(handle, input_type, data, filename)
    nvjitlink.destroy(handle)


# We test the LTO input case separately as it requires the `-lto` flag. The
# OBJECT input type is used because the LTO-IR container is packaged in an ELF
# object when produced by NVCC.
def test_add_file_lto(device_functions_ltoir_object, gpu_arch_flag):
    filename, data = device_functions_ltoir_object

    handle = nvjitlink.create(gpu_arch_flag, "-lto")
    nvjitlink.add_data(handle, nvjitlink.InputType.OBJECT, data, filename)
    nvjitlink.destroy(handle)


def test_get_error_log(undefined_extern_cubin, gpu_arch_flag):
    handle = nvjitlink.create(gpu_arch_flag)
    filename, data = undefined_extern_cubin
    input_type = nvjitlink.InputType.CUBIN
    nvjitlink.add_data(handle, input_type, data, filename)
    with pytest.raises(RuntimeError):
        nvjitlink.complete(handle)
    error_log = nvjitlink.get_error_log(handle)
    nvjitlink.destroy(handle)
    assert (
        "Undefined reference to '_Z5undefff' "
        "in 'undefined_extern.cubin'" in error_log
    )


def test_get_info_log(device_functions_cubin, gpu_arch_flag):
    handle = nvjitlink.create(gpu_arch_flag)
    filename, data = device_functions_cubin
    input_type = nvjitlink.InputType.CUBIN
    nvjitlink.add_data(handle, input_type, data, filename)
    nvjitlink.complete(handle)
    info_log = nvjitlink.get_info_log(handle)
    nvjitlink.destroy(handle)
    # Info log is empty
    assert "" == info_log


def test_get_linked_cubin(device_functions_cubin, gpu_arch_flag):
    handle = nvjitlink.create(gpu_arch_flag)
    filename, data = device_functions_cubin
    input_type = nvjitlink.InputType.CUBIN
    nvjitlink.add_data(handle, input_type, data, filename)
    nvjitlink.complete(handle)
    cubin = nvjitlink.get_linked_cubin(handle)
    nvjitlink.destroy(handle)

    # Just check we got something that looks like an ELF
    assert cubin[:4] == b"\x7fELF"


def test_get_linked_cubin_link_not_complete_error(
    device_functions_cubin, gpu_arch_flag
):
    handle = nvjitlink.create(gpu_arch_flag)
    filename, data = device_functions_cubin
    input_type = nvjitlink.InputType.CUBIN
    nvjitlink.add_data(handle, input_type, data, filename)
    with pytest.raises(RuntimeError, match="NVJITLINK_ERROR_INTERNAL error"):
        nvjitlink.get_linked_cubin(handle)
    nvjitlink.destroy(handle)


def test_get_linked_cubin_from_lto(device_functions_ltoir_object, gpu_arch_flag):
    filename, data = device_functions_ltoir_object
    # device_functions_ltoir_object is a host object containing a fatbin
    # containing an LTOIR container, because that is what NVCC produces when
    # LTO is requested. So we need to use the OBJECT input type, and the linker
    # retrieves the LTO IR from it because we passed the -lto flag.
    input_type = nvjitlink.InputType.OBJECT
    handle = nvjitlink.create(gpu_arch_flag, "-lto")
    nvjitlink.add_data(handle, input_type, data, filename)
    nvjitlink.complete(handle)
    cubin = nvjitlink.get_linked_cubin(handle)
    nvjitlink.destroy(handle)

    # Just check we got something that looks like an ELF
    assert cubin[:4] == b"\x7fELF"


def test_get_linked_ptx_from_lto(device_functions_ltoir_object, gpu_arch_flag):
    filename, data = device_functions_ltoir_object
    # device_functions_ltoir_object is a host object containing a fatbin
    # containing an LTOIR container, because that is what NVCC produces when
    # LTO is requested. So we need to use the OBJECT input type, and the linker
    # retrieves the LTO IR from it because we passed the -lto flag.
    input_type = nvjitlink.InputType.OBJECT
    handle = nvjitlink.create(gpu_arch_flag, "-lto", "-ptx")
    nvjitlink.add_data(handle, input_type, data, filename)
    nvjitlink.complete(handle)
    nvjitlink.get_linked_ptx(handle)
    nvjitlink.destroy(handle)


def test_get_linked_ptx_link_not_complete_error(
    device_functions_ltoir_object, gpu_arch_flag
):
    handle = nvjitlink.create(gpu_arch_flag, "-lto", "-ptx")
    filename, data = device_functions_ltoir_object
    input_type = nvjitlink.InputType.OBJECT
    nvjitlink.add_data(handle, input_type, data, filename)
    with pytest.raises(RuntimeError, match="NVJITLINK_ERROR_INTERNAL error"):
        nvjitlink.get_linked_ptx(handle)
    nvjitlink.destroy(handle)


def test_package_version():
    ver = nvjitlink.version()
    assert len(ver) == 2
    assert ver >= (12, 0)
