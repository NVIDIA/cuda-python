# SPDX-FileCopyrightText: Copyright (c) 2024-2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# SPDX-License-Identifier: LicenseRef-NVIDIA-SOFTWARE-LICENSE

import warnings

import pytest

from cuda.core.experimental import _linker
from cuda.core.experimental._module import Kernel, ObjectCode
from cuda.core.experimental._program import Program, ProgramOptions
from cuda.core.experimental._utils.cuda_utils import driver, handle_return

cuda_driver_version = handle_return(driver.cuDriverGetVersion())
is_culink_backend = _linker._decide_nvjitlink_or_driver()


def _is_nvvm_available():
    """Check if NVVM is available."""
    try:
        from cuda.core.experimental._program import _get_nvvm_module

        _get_nvvm_module()
        return True
    except RuntimeError:
        return False


nvvm_available = pytest.mark.skipif(
    not _is_nvvm_available(), reason="NVVM not available (libNVVM not found or cuda-bindings < 12.9.0)"
)

try:
    from cuda.core.experimental._utils.cuda_utils import driver, handle_return

    _cuda_driver_version = handle_return(driver.cuDriverGetVersion())
except Exception:
    _cuda_driver_version = 0

_libnvvm_version = None
_libnvvm_version_attempted = False

precheck_nvvm_ir = """target triple = "nvptx64-unknown-cuda"
target datalayout = "e-p:64:64:64-i1:8:8-i8:8:8-i16:16:16-i32:32:32-i64:64:64-i128:128:128-f32:32:32-f64:64:64-v16:16:16-v32:32:32-v64:64:64-v128:128:128-n16:32:64"

define void @dummy_kernel() {{
  entry:
    ret void
}}

!nvvm.annotations = !{{!0}}
!0 = !{{void ()* @dummy_kernel, !"kernel", i32 1}}

!nvvmir.version = !{{!1}}
!1 = !{{i32 {major}, i32 {minor}, i32 {debug_major}, i32 {debug_minor}}}
"""  # noqa: E501


def _get_libnvvm_version_for_tests():
    """
    Detect libNVVM version by compiling dummy IR and analyzing the PTX output.

    Workaround for the lack of direct libNVVM version API (nvbugs 5312315).
    The approach:
    - Compile a small dummy NVVM IR to PTX
    - Use PTX version analysis APIs if available to infer libNVVM version
    - Cache the result for future use
    """
    global _libnvvm_version, _libnvvm_version_attempted

    if _libnvvm_version_attempted:
        return _libnvvm_version

    _libnvvm_version_attempted = True

    try:
        from cuda.core.experimental._program import _get_nvvm_module

        nvvm = _get_nvvm_module()

        try:
            from cuda.bindings.utils import get_minimal_required_cuda_ver_from_ptx_ver, get_ptx_ver
        except ImportError:
            _libnvvm_version = None
            return _libnvvm_version

        program = nvvm.create_program()
        try:
            major, minor, debug_major, debug_minor = nvvm.ir_version()
            global precheck_nvvm_ir
            precheck_nvvm_ir = precheck_nvvm_ir.format(
                major=major, minor=minor, debug_major=debug_major, debug_minor=debug_minor
            )
            precheck_ir_bytes = precheck_nvvm_ir.encode("utf-8")
            nvvm.add_module_to_program(program, precheck_ir_bytes, len(precheck_ir_bytes), "precheck.ll")

            options = ["-arch=compute_90"]
            nvvm.verify_program(program, len(options), options)
            nvvm.compile_program(program, len(options), options)

            ptx_size = nvvm.get_compiled_result_size(program)
            ptx_data = bytearray(ptx_size)
            nvvm.get_compiled_result(program, ptx_data)
            ptx_str = ptx_data.decode("utf-8")
            ptx_version = get_ptx_ver(ptx_str)
            cuda_version = get_minimal_required_cuda_ver_from_ptx_ver(ptx_version)
            _libnvvm_version = cuda_version
            return _libnvvm_version
        finally:
            nvvm.destroy_program(program)

    except Exception:
        _libnvvm_version = None
        return _libnvvm_version


@pytest.fixture(scope="session")
def nvvm_ir():
    """Generate working NVVM IR with proper version metadata.
    The try clause here is used for older nvvm modules which
    might not have an ir_version() method. In which case the
    fallback assumes no version metadata will be present in
    the input nvvm ir
    """
    from cuda.core.experimental._program import _get_nvvm_module

    nvvm = _get_nvvm_module()
    major, minor, debug_major, debug_minor = nvvm.ir_version()

    nvvm_ir_template = """target triple = "nvptx64-unknown-cuda"
target datalayout = "e-p:64:64:64-i1:8:8-i8:8:8-i16:16:16-i32:32:32-i64:64:64-i128:128:128-f32:32:32-f64:64:64-v16:16:16-v32:32:32-v64:64:64-v128:128:128-n16:32:64"

define i32 @ave(i32 %a, i32 %b) {{
entry:
  %add = add nsw i32 %a, %b
  %div = sdiv i32 %add, 2
  ret i32 %div
}}

define void @simple(i32* %data) {{
entry:
  %0 = call i32 @llvm.nvvm.read.ptx.sreg.ctaid.x()
  %1 = call i32 @llvm.nvvm.read.ptx.sreg.ntid.x()
  %mul = mul i32 %0, %1
  %2 = call i32 @llvm.nvvm.read.ptx.sreg.tid.x()
  %add = add i32 %mul, %2
  %call = call i32 @ave(i32 %add, i32 %add)
  %idxprom = sext i32 %add to i64
  store i32 %call, i32* %data, align 4
  ret void
}}

declare i32 @llvm.nvvm.read.ptx.sreg.ctaid.x() nounwind readnone
declare i32 @llvm.nvvm.read.ptx.sreg.ntid.x() nounwind readnone
declare i32 @llvm.nvvm.read.ptx.sreg.tid.x() nounwind readnone

!nvvm.annotations = !{{!0}}
!0 = !{{void (i32*)* @simple, !"kernel", i32 1}}

!nvvmir.version = !{{!1}}
!1 = !{{i32 {major}, i32 {minor}, i32 {debug_major}, i32 {debug_minor}}}
"""  # noqa: E501
    return nvvm_ir_template.format(major=major, minor=minor, debug_major=debug_major, debug_minor=debug_minor)


@pytest.fixture(scope="module")
def ptx_code_object():
    code = 'extern "C" __global__ void my_kernel() {}'
    program = Program(code, "c++")
    ptx_object_code = program.compile("ptx")
    return ptx_object_code


@pytest.mark.parametrize(
    "options",
    [
        ProgramOptions(name="abc"),
        ProgramOptions(device_code_optimize=True, debug=True),
        ProgramOptions(relocatable_device_code=True, max_register_count=32),
        ProgramOptions(ftz=True, prec_sqrt=False, prec_div=False),
        ProgramOptions(fma=False, use_fast_math=True),
        ProgramOptions(extra_device_vectorization=True),
        ProgramOptions(link_time_optimization=True),
        ProgramOptions(define_macro="MY_MACRO"),
        ProgramOptions(define_macro=("MY_MACRO", "99")),
        ProgramOptions(define_macro=[("MY_MACRO", "99")]),
        ProgramOptions(define_macro=[("MY_MACRO", "99"), ("MY_OTHER_MACRO", "100")]),
        ProgramOptions(undefine_macro=["MY_MACRO", "MY_OTHER_MACRO"]),
        ProgramOptions(undefine_macro="MY_MACRO", include_path="/usr/local/include"),
        ProgramOptions(builtin_initializer_list=False, disable_warnings=True),
        ProgramOptions(restrict=True, device_as_default_execution_space=True),
        ProgramOptions(device_int128=True, optimization_info="inline"),
        ProgramOptions(no_display_error_number=True),
        ProgramOptions(diag_error=1234, diag_suppress=1234),
        ProgramOptions(diag_error=[1234, 1223], diag_suppress=(1234, 1223)),
        ProgramOptions(diag_warn=1000),
        ProgramOptions(std="c++11", ptxas_options=["-v"]),
        ProgramOptions(std="c++11", ptxas_options=["-v", "-O2"]),
    ],
)
def test_cpp_program_with_various_options(init_cuda, options):
    code = 'extern "C" __global__ void my_kernel() {}'
    program = Program(code, "c++", options)
    assert program.backend == "NVRTC"
    program.compile("ptx")
    program.close()
    assert program.handle is None


options = [
    ProgramOptions(max_register_count=32),
    ProgramOptions(debug=True),
    ProgramOptions(lineinfo=True),
    ProgramOptions(ftz=True),
    ProgramOptions(prec_div=True),
    ProgramOptions(prec_sqrt=True),
    ProgramOptions(fma=True),
]
if not is_culink_backend:
    options += [
        ProgramOptions(time=True),
        ProgramOptions(split_compile=True),
    ]


@pytest.mark.parametrize("options", options)
def test_ptx_program_with_various_options(init_cuda, ptx_code_object, options):
    program = Program(ptx_code_object._module.decode(), "ptx", options=options)
    assert program.backend == ("driver" if is_culink_backend else "nvJitLink")
    program.compile("cubin")
    program.close()
    assert program.handle is None


def test_program_init_valid_code_type():
    code = 'extern "C" __global__ void my_kernel() {}'
    program = Program(code, "c++")
    assert program.backend == "NVRTC"
    assert program.handle is not None


def test_program_init_invalid_code_type():
    code = "goto 100"
    with pytest.raises(
        RuntimeError, match=r"^Unsupported code_type='fortran' \(supported_code_types=\('c\+\+', 'ptx', 'nvvm'\)\)$"
    ):
        Program(code, "FORTRAN")


def test_program_init_invalid_code_format():
    code = 12345
    with pytest.raises(TypeError):
        Program(code, "c++")


# This is tested against the current device's arch
def test_program_compile_valid_target_type(init_cuda):
    code = 'extern "C" __global__ void my_kernel() {}'
    program = Program(code, "c++", options={"name": "42"})

    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        ptx_object_code = program.compile("ptx")
        assert isinstance(ptx_object_code, ObjectCode)
        assert ptx_object_code.name == "42"
        if any("The CUDA driver version is older than the backend version" in str(warning.message) for warning in w):
            pytest.skip("PTX version too new for current driver")
        ptx_kernel = ptx_object_code.get_kernel("my_kernel")
        assert isinstance(ptx_kernel, Kernel)

    program = Program(ptx_object_code._module.decode(), "ptx", options={"name": "24"})
    cubin_object_code = program.compile("cubin")
    assert isinstance(cubin_object_code, ObjectCode)
    assert cubin_object_code.name == "24"
    cubin_kernel = cubin_object_code.get_kernel("my_kernel")
    assert isinstance(cubin_kernel, Kernel)


def test_program_compile_invalid_target_type():
    code = 'extern "C" __global__ void my_kernel() {}'
    program = Program(code, "c++")
    with pytest.raises(ValueError):
        program.compile("invalid_target")


def test_program_backend_property():
    code = 'extern "C" __global__ void my_kernel() {}'
    program = Program(code, "c++")
    assert program.backend == "NVRTC"


def test_program_handle_property():
    code = 'extern "C" __global__ void my_kernel() {}'
    program = Program(code, "c++")
    assert program.handle is not None


def test_program_close():
    code = 'extern "C" __global__ void my_kernel() {}'
    program = Program(code, "c++")
    program.close()
    assert program.handle is None


@nvvm_available
def test_nvvm_deferred_import():
    """Test that our deferred NVVM import works correctly"""
    from cuda.core.experimental._program import _get_nvvm_module

    nvvm = _get_nvvm_module()
    assert nvvm is not None


@nvvm_available
def test_nvvm_program_creation_compilation(nvvm_ir):
    """Test basic NVVM program creation"""
    program = Program(nvvm_ir, "nvvm")
    assert program.backend == "NVVM"
    assert program.handle is not None
    obj = program.compile("ptx")
    ker = obj.get_kernel("simple")  # noqa: F841
    program.close()


@nvvm_available
def test_nvvm_compile_invalid_target(nvvm_ir):
    """Test that NVVM programs reject invalid compilation targets"""
    program = Program(nvvm_ir, "nvvm")
    with pytest.raises(ValueError, match='NVVM backend only supports target_type="ptx"'):
        program.compile("cubin")
    program.close()


@nvvm_available
@pytest.mark.parametrize(
    "options",
    [
        ProgramOptions(name="test1", arch="sm_90", device_code_optimize=False),
        ProgramOptions(name="test2", arch="sm_100", device_code_optimize=False),
        pytest.param(
            ProgramOptions(name="test_sm110_1", arch="sm_110", device_code_optimize=False),
            marks=pytest.mark.skipif(
                (_get_libnvvm_version_for_tests() or 0) < 13000,
                reason="Compute capability 110 requires libNVVM >= 13.0",
            ),
        ),
        pytest.param(
            ProgramOptions(
                name="test_sm110_2",
                arch="sm_110",
                ftz=True,
                prec_sqrt=False,
                prec_div=False,
                fma=True,
                device_code_optimize=True,
            ),
            marks=pytest.mark.skipif(
                (_get_libnvvm_version_for_tests() or 0) < 13000,
                reason="Compute capability 110 requires libNVVM >= 13.0",
            ),
        ),
        pytest.param(
            ProgramOptions(name="test_sm110_3", arch="sm_110", link_time_optimization=True),
            marks=pytest.mark.skipif(
                (_get_libnvvm_version_for_tests() or 0) < 13000,
                reason="Compute capability 110 requires libNVVM >= 13.0",
            ),
        ),
    ],
)
def test_nvvm_program_options(init_cuda, nvvm_ir, options):
    """Test NVVM programs with different options"""
    program = Program(nvvm_ir, "nvvm", options)
    assert program.backend == "NVVM"

    ptx_code = program.compile("ptx")
    assert isinstance(ptx_code, ObjectCode)
    assert ptx_code.name == options.name

    code_content = ptx_code.code
    ptx_text = code_content.decode() if isinstance(code_content, bytes) else str(code_content)
    assert ".visible .entry simple(" in ptx_text

    program.close()
