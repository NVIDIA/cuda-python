import pytest

from cuda.core.experimental import Linker, LinkerOptions, Program, _linker
from cuda.core.experimental._module import ObjectCode

ARCH = "sm_80"  # use sm_80 for testing the oop nvJitLink wrapper

kernel_a = """
extern __device__ int B();
extern __device__ int C(int a, int b);
__global__ void A() { int result = C(B(), 1);}
"""
device_function_b = "__device__ int B() { return 0; }"
device_function_c = "__device__ int C(int a, int b) { return a + b; }"

culink_backend = _linker._decide_nvjitlink_or_driver()


@pytest.fixture(scope="function")
def compile_ptx_functions(init_cuda):
    # Without -rdc (relocatable device code) option, the generated ptx will not included any unreferenced
    # device functions, causing the link to fail
    object_code_a_ptx = Program(kernel_a, "c++").compile("ptx", options=("-rdc=true",))
    object_code_b_ptx = Program(device_function_b, "c++").compile("ptx", options=("-rdc=true",))
    object_code_c_ptx = Program(device_function_c, "c++").compile("ptx", options=("-rdc=true",))

    return object_code_a_ptx, object_code_b_ptx, object_code_c_ptx


@pytest.fixture(scope="function")
def compile_ltoir_functions(init_cuda):
    object_code_a_ltoir = Program(kernel_a, "c++").compile("ltoir", options=("-dlto",))
    object_code_b_ltoir = Program(device_function_b, "c++").compile("ltoir", options=("-dlto",))
    object_code_c_ltoir = Program(device_function_c, "c++").compile("ltoir", options=("-dlto",))

    return object_code_a_ltoir, object_code_b_ltoir, object_code_c_ltoir


culink_options = [
    LinkerOptions(arch=ARCH, verbose=True),
    LinkerOptions(arch=ARCH, max_register_count=32),
    LinkerOptions(arch=ARCH, optimization_level=3),
    LinkerOptions(arch=ARCH, debug=True),
    LinkerOptions(arch=ARCH, lineinfo=True),
    LinkerOptions(arch=ARCH, no_cache=True),
]


@pytest.mark.parametrize(
    "options",
    culink_options
    if culink_backend
    else culink_options
    + [
        LinkerOptions(arch=ARCH, time=True),
        LinkerOptions(arch=ARCH, ftz=True),
        LinkerOptions(arch=ARCH, prec_div=True),
        LinkerOptions(arch=ARCH, prec_sqrt=True),
        LinkerOptions(arch=ARCH, fma=True),
        LinkerOptions(arch=ARCH, kernels_used=["A"]),
        LinkerOptions(arch=ARCH, kernels_used=["C", "B"]),
        LinkerOptions(arch=ARCH, variables_used=["var1"]),
        LinkerOptions(arch=ARCH, variables_used=["var1", "var2"]),
        LinkerOptions(arch=ARCH, optimize_unused_variables=True),
        LinkerOptions(arch=ARCH, xptxas=["-v"]),
        LinkerOptions(arch=ARCH, split_compile=0),
        LinkerOptions(arch=ARCH, split_compile_extended=1),
    ],
)
def test_linker_init(compile_ptx_functions, options):
    linker = Linker(*compile_ptx_functions, options=options)
    object_code = linker.link("cubin")
    assert isinstance(object_code, ObjectCode)


def test_linker_init_invalid_arch():
    options = LinkerOptions(arch=None)
    with pytest.raises(TypeError):
        Linker(options)


@pytest.mark.skipif(culink_backend, reason="culink does not support ptx option")
def test_linker_link_ptx_nvjitlink(compile_ltoir_functions):
    options = LinkerOptions(arch=ARCH, link_time_optimization=True, ptx=True)
    linker = Linker(*compile_ltoir_functions, options=options)
    linked_code = linker.link("ptx")
    assert isinstance(linked_code, ObjectCode)


@pytest.mark.skipif(not culink_backend, reason="nvjitlink requires lto for ptx linking")
def test_linker_link_ptx_culink(compile_ptx_functions):
    options = LinkerOptions(arch=ARCH)
    linker = Linker(*compile_ptx_functions, options=options)
    linked_code = linker.link("ptx")
    assert isinstance(linked_code, ObjectCode)


def test_linker_link_cubin(compile_ptx_functions):
    options = LinkerOptions(arch=ARCH)
    linker = Linker(*compile_ptx_functions, options=options)
    linked_code = linker.link("cubin")
    assert isinstance(linked_code, ObjectCode)


def test_linker_link_invalid_target_type(compile_ptx_functions):
    options = LinkerOptions(arch=ARCH)
    linker = Linker(*compile_ptx_functions, options=options)
    with pytest.raises(ValueError):
        linker.link("invalid_target")


def test_linker_get_error_log(compile_ptx_functions):
    options = LinkerOptions(arch=ARCH)
    linker = Linker(*compile_ptx_functions, options=options)
    linker.link("cubin")
    log = linker.get_error_log()
    assert isinstance(log, str)


def test_linker_get_info_log(compile_ptx_functions):
    options = LinkerOptions(arch=ARCH)
    linker = Linker(*compile_ptx_functions, options=options)
    linker.link("cubin")
    log = linker.get_info_log()
    assert isinstance(log, str)
