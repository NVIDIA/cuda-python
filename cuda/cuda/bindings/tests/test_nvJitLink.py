import pytest
from cuda import nvJitLink

def test_create_no_arch_error():
    # nvjitlink expects at least the architecture to be specified.
    with pytest.raises(RuntimeError, match="NVJITLINK_ERROR_MISSING_ARCH error"):
        nvJitLink.create()


def test_invalid_arch_error():
    # sm_XX is not a valid architecture
    with pytest.raises(RuntimeError, match="NVJITLINK_ERROR_UNRECOGNIZED_OPTION error"):
        nvJitLink.create("-arch=sm_XX")


def test_unrecognized_option_error():
    with pytest.raises(RuntimeError, match="NVJITLINK_ERROR_UNRECOGNIZED_OPTION error"):
        nvJitLink.create("-fictitious_option")


def test_invalid_option_type_error():
    with pytest.raises(TypeError, match="Expecting only strings"):
        nvJitLink.create("-arch", 53)


def test_create_and_destroy():
    handle = nvJitLink.create("-arch=sm_53")
    assert handle != 0
    nvJitLink.destroy(handle)


def test_complete_empty():
    handle = nvJitLink.create("-arch=sm_75")
    nvJitLink.complete(handle)
    nvJitLink.destroy(handle)


@pytest.mark.parametrize(
    "input_file,input_type",
    [
        ("device_functions_cubin", nvJitLink.InputType.CUBIN),
        ("device_functions_fatbin", InputType.FATBIN),
        ("device_functions_ptx", InputType.PTX),
        ("device_functions_object", InputType.OBJECT),
        ("device_functions_archive", InputType.LIBRARY),
    ],
)
def test_add_file(input_file, input_type, gpu_arch_flag, request):
    filename, data = request.getfixturevalue(input_file)

    handle = nvJitLink.create(gpu_arch_flag)
    nvJitLink.add_data(handle, input_type.value, data, filename)
    nvJitLink.destroy(handle)


# We test the LTO input case separately as it requires the `-lto` flag. The
# OBJECT input type is used because the LTO-IR container is packaged in an ELF
# object when produced by NVCC.
def test_add_file_lto(device_functions_ltoir_object, gpu_arch_flag):
    filename, data = device_functions_ltoir_object

    handle = nvJitLink.create(gpu_arch_flag, "-lto")
    nvJitLink.add_data(handle, InputType.OBJECT.value, data, filename)
    nvJitLink.destroy(handle)


def test_get_error_log(undefined_extern_cubin, gpu_arch_flag):
    handle = nvJitLink.create(gpu_arch_flag)
    filename, data = undefined_extern_cubin
    input_type = InputType.CUBIN.value
    nvJitLink.add_data(handle, input_type, data, filename)
    with pytest.raises(RuntimeError):
        nvJitLink.complete(handle)
    error_log = nvJitLink.get_error_log(handle)
    nvJitLink.destroy(handle)
    assert (
        "Undefined reference to '_Z5undefff' "
        "in 'undefined_extern.cubin'" in error_log
    )


def test_get_info_log(device_functions_cubin, gpu_arch_flag):
    handle = nvJitLink.create(gpu_arch_flag)
    filename, data = device_functions_cubin
    input_type = InputType.CUBIN.value
    nvJitLink.add_data(handle, input_type, data, filename)
    nvJitLink.complete(handle)
    info_log = nvJitLink.get_info_log(handle)
    nvJitLink.destroy(handle)
    # Info log is empty
    assert "" == info_log


def test_get_linked_cubin(device_functions_cubin, gpu_arch_flag):
    handle = nvJitLink.create(gpu_arch_flag)
    filename, data = device_functions_cubin
    input_type = InputType.CUBIN.value
    nvJitLink.add_data(handle, input_type, data, filename)
    nvJitLink.complete(handle)
    cubin = nvJitLink.get_linked_cubin(handle)
    nvJitLink.destroy(handle)

    # Just check we got something that looks like an ELF
    assert cubin[:4] == b"\x7fELF"


def test_get_linked_cubin_link_not_complete_error(
    device_functions_cubin, gpu_arch_flag
):
    handle = nvJitLink.create(gpu_arch_flag)
    filename, data = device_functions_cubin
    input_type = InputType.CUBIN.value
    nvJitLink.add_data(handle, input_type, data, filename)
    with pytest.raises(RuntimeError, match="NVJITLINK_ERROR_INTERNAL error"):
        nvJitLink.get_linked_cubin(handle)
    nvJitLink.destroy(handle)


def test_get_linked_cubin_from_lto(device_functions_ltoir_object, gpu_arch_flag):
    filename, data = device_functions_ltoir_object
    # device_functions_ltoir_object is a host object containing a fatbin
    # containing an LTOIR container, because that is what NVCC produces when
    # LTO is requested. So we need to use the OBJECT input type, and the linker
    # retrieves the LTO IR from it because we passed the -lto flag.
    input_type = InputType.OBJECT.value
    handle = nvJitLink.create(gpu_arch_flag, "-lto")
    nvJitLink.add_data(handle, input_type, data, filename)
    nvJitLink.complete(handle)
    cubin = nvJitLink.get_linked_cubin(handle)
    nvJitLink.destroy(handle)

    # Just check we got something that looks like an ELF
    assert cubin[:4] == b"\x7fELF"


def test_get_linked_ptx_from_lto(device_functions_ltoir_object, gpu_arch_flag):
    filename, data = device_functions_ltoir_object
    # device_functions_ltoir_object is a host object containing a fatbin
    # containing an LTOIR container, because that is what NVCC produces when
    # LTO is requested. So we need to use the OBJECT input type, and the linker
    # retrieves the LTO IR from it because we passed the -lto flag.
    input_type = InputType.OBJECT.value
    handle = nvJitLink.create(gpu_arch_flag, "-lto", "-ptx")
    nvJitLink.add_data(handle, input_type, data, filename)
    nvJitLink.complete(handle)
    nvJitLink.get_linked_ptx(handle)
    nvJitLink.destroy(handle)


def test_get_linked_ptx_link_not_complete_error(
    device_functions_ltoir_object, gpu_arch_flag
):
    handle = nvJitLink.create(gpu_arch_flag, "-lto", "-ptx")
    filename, data = device_functions_ltoir_object
    input_type = InputType.OBJECT.value
    nvJitLink.add_data(handle, input_type, data, filename)
    with pytest.raises(RuntimeError, match="NVJITLINK_ERROR_INTERNAL error"):
        nvJitLink.get_linked_ptx(handle)
    nvJitLink.destroy(handle)


def test_package_version():
    assert pynvjitlink.__version__ is not None
    assert len(str(pynvjitlink.__version__)) > 0