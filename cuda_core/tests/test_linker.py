# SPDX-FileCopyrightText: Copyright (c) 2024-2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# SPDX-License-Identifier: LicenseRef-NVIDIA-SOFTWARE-LICENSE

import pytest
from cuda.core.experimental import Device, Linker, LinkerOptions, Program, ProgramOptions, _linker
from cuda.core.experimental._module import ObjectCode
from cuda.core.experimental._utils.cuda_utils import CUDAError

ARCH = "sm_" + "".join(f"{i}" for i in Device().compute_capability)

kernel_a = """
extern __device__ int B();
extern __device__ int C(int a, int b);
__global__ void A() { int result = C(B(), 1);}
"""
device_function_b = "__device__ int B() { return 0; }"
device_function_c = "__device__ int C(int a, int b) { return a + b; }"

is_culink_backend = _linker._decide_nvjitlink_or_driver()
if not is_culink_backend:
    from cuda.bindings import nvjitlink

    nvJitLinkError = nvjitlink.nvJitLinkError
else:

    class nvJitLinkError(Exception):
        pass


@pytest.fixture(scope="function")
def compile_ptx_functions(init_cuda):
    # Without -rdc (relocatable device code) option, the generated ptx will not included any unreferenced
    # device functions, causing the link to fail
    object_code_a_ptx = Program(kernel_a, "c++", ProgramOptions(relocatable_device_code=True)).compile("ptx")
    object_code_b_ptx = Program(device_function_b, "c++", ProgramOptions(relocatable_device_code=True)).compile("ptx")
    object_code_c_ptx = Program(device_function_c, "c++", ProgramOptions(relocatable_device_code=True)).compile("ptx")

    return object_code_a_ptx, object_code_b_ptx, object_code_c_ptx


@pytest.fixture(scope="function")
def compile_ltoir_functions(init_cuda):
    object_code_a_ltoir = Program(kernel_a, "c++", ProgramOptions(link_time_optimization=True)).compile("ltoir")
    object_code_b_ltoir = Program(device_function_b, "c++", ProgramOptions(link_time_optimization=True)).compile(
        "ltoir"
    )
    object_code_c_ltoir = Program(device_function_c, "c++", ProgramOptions(link_time_optimization=True)).compile(
        "ltoir"
    )

    return object_code_a_ltoir, object_code_b_ltoir, object_code_c_ltoir


options = [
    LinkerOptions(),
    LinkerOptions(arch=ARCH, verbose=True),
    LinkerOptions(arch=ARCH, max_register_count=32),
    LinkerOptions(arch=ARCH, optimization_level=3),
    LinkerOptions(arch=ARCH, debug=True),
    LinkerOptions(arch=ARCH, lineinfo=True),
]
if not is_culink_backend:
    options += [
        LinkerOptions(arch=ARCH, time=True),
        LinkerOptions(arch=ARCH, optimize_unused_variables=True),
        LinkerOptions(arch=ARCH, ptxas_options="-v"),
        LinkerOptions(arch=ARCH, ptxas_options=["-v", "--verbose"]),
        LinkerOptions(arch=ARCH, ptxas_options=("-v", "--verbose")),
        LinkerOptions(arch=ARCH, split_compile=0),
        LinkerOptions(arch=ARCH, split_compile_extended=1),
        # The following options are supported by nvjitlink and deprecated by culink
        LinkerOptions(arch=ARCH, ftz=True),
        LinkerOptions(arch=ARCH, prec_div=True),
        LinkerOptions(arch=ARCH, prec_sqrt=True),
        LinkerOptions(arch=ARCH, fma=True),
        LinkerOptions(arch=ARCH, kernels_used="A"),
        LinkerOptions(arch=ARCH, kernels_used=["C", "B"]),
        LinkerOptions(arch=ARCH, kernels_used=("C", "B")),
        LinkerOptions(arch=ARCH, variables_used="var1"),
        LinkerOptions(arch=ARCH, variables_used=["var1", "var2"]),
        LinkerOptions(arch=ARCH, variables_used=("var1", "var2")),
    ]
    version = nvjitlink.version()
    if version >= (12, 5):
        options.append(LinkerOptions(arch=ARCH, no_cache=True))


@pytest.mark.parametrize("options", options)
def test_linker_init(compile_ptx_functions, options):
    linker = Linker(*compile_ptx_functions, options=options)
    object_code = linker.link("cubin")
    assert isinstance(object_code, ObjectCode)
    assert linker.backend == ("driver" if is_culink_backend else "nvJitLink")


def test_linker_init_invalid_arch(compile_ptx_functions):
    err = AttributeError if is_culink_backend else nvjitlink.nvJitLinkError
    with pytest.raises(err):
        options = LinkerOptions(arch="99", ptx=True)
        Linker(*compile_ptx_functions, options=options)


@pytest.mark.skipif(is_culink_backend, reason="culink does not support ptx option")
def test_linker_link_ptx_nvjitlink(compile_ltoir_functions):
    options = LinkerOptions(arch=ARCH, link_time_optimization=True, ptx=True)
    linker = Linker(*compile_ltoir_functions, options=options)
    linked_code = linker.link("ptx")
    assert isinstance(linked_code, ObjectCode)
    assert linked_code.name == options.name


@pytest.mark.skipif(not is_culink_backend, reason="nvjitlink requires lto for ptx linking")
def test_linker_link_ptx_culink(compile_ptx_functions):
    options = LinkerOptions(arch=ARCH)
    linker = Linker(*compile_ptx_functions, options=options)
    linked_code = linker.link("ptx")
    assert isinstance(linked_code, ObjectCode)
    assert linked_code.name == options.name


def test_linker_link_cubin(compile_ptx_functions):
    options = LinkerOptions(arch=ARCH)
    linker = Linker(*compile_ptx_functions, options=options)
    linked_code = linker.link("cubin")
    assert isinstance(linked_code, ObjectCode)
    assert linked_code.name == options.name


def test_linker_link_ptx_multiple(compile_ptx_functions):
    ptxes = tuple(ObjectCode.from_ptx(obj.code) for obj in compile_ptx_functions)
    options = LinkerOptions(arch=ARCH)
    linker = Linker(*ptxes, options=options)
    linked_code = linker.link("cubin")
    assert isinstance(linked_code, ObjectCode)
    assert linked_code.name == options.name


def test_linker_link_invalid_target_type(compile_ptx_functions):
    options = LinkerOptions(arch=ARCH)
    linker = Linker(*compile_ptx_functions, options=options)
    with pytest.raises(ValueError):
        linker.link("invalid_target")


def test_linker_get_error_log(compile_ptx_functions):
    options = LinkerOptions(name="ABC", arch=ARCH)

    replacement_kernel = """
extern __device__ int Z();
extern __device__ int C(int a, int b);
__global__ void A() { int result = C(Z(), 1);}
"""
    dummy_program = Program(
        replacement_kernel, "c++", ProgramOptions(name="CBA", relocatable_device_code=True)
    ).compile("ptx")
    linker = Linker(dummy_program, *(compile_ptx_functions[1:]), options=options)
    try:
        linker.link("cubin")

    except (nvJitLinkError, CUDAError):
        log = linker.get_error_log()
        assert isinstance(log, str)
        # TODO when 4902246 is addressed, we can update this to cover nvjitlink as well
        # The error is coming from the input object that's being linked (CBA), not the output object (ABC).
        if is_culink_backend:
            assert log.rstrip("\x00") == "error   : Undefined reference to '_Z1Zv' in 'CBA'"


def test_linker_get_info_log(compile_ptx_functions):
    options = LinkerOptions(arch=ARCH)
    linker = Linker(*compile_ptx_functions, options=options)
    linker.link("cubin")
    log = linker.get_info_log()
    assert isinstance(log, str)
