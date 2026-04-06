# SPDX-FileCopyrightText: Copyright (c) 2024-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# SPDX-License-Identifier: LicenseRef-NVIDIA-SOFTWARE-LICENSE

import contextlib
import re
import warnings

import pytest

from cuda.core import _linker
from cuda.core._device import Device
from cuda.core._module import Kernel, ObjectCode
from cuda.core._program import Program, ProgramOptions
from cuda.core._utils.cuda_utils import CUDAError, handle_return

pytest_plugins = ("cuda_python_test_helpers.nvvm_bitcode",)

is_culink_backend = _linker._decide_nvjitlink_or_driver()


def _is_nvvm_available():
    """Check if NVVM is available."""
    try:
        from cuda.core._program import _get_nvvm_module

        _get_nvvm_module()
        return True
    except RuntimeError:
        return False


nvvm_available = pytest.mark.skipif(
    not _is_nvvm_available(), reason="NVVM not available (libNVVM not found or cuda-bindings < 12.9.0)"
)

with contextlib.suppress(Exception):
    from cuda.core._utils.cuda_utils import nvrtc


def _get_nvrtc_version_for_tests():
    """
    Get NVRTC version.

    Returns:
        int: Version in format major * 1000 + minor * 100 (e.g., 13200 for CUDA 13.2)
        None: If NVRTC is not available
    """
    try:
        nvrtc_major, nvrtc_minor = handle_return(nvrtc.nvrtcVersion())
        version = nvrtc_major * 1000 + nvrtc_minor * 100
        return version
    except Exception:
        return None


def _has_nvrtc_pch_apis_for_tests():
    required = (
        "nvrtcGetPCHHeapSize",
        "nvrtcSetPCHHeapSize",
        "nvrtcGetPCHCreateStatus",
        "nvrtcGetPCHHeapSizeRequired",
    )
    return all(hasattr(nvrtc, name) for name in required)


nvrtc_pch_available = pytest.mark.skipif(
    (_get_nvrtc_version_for_tests() or 0) < 12800 or not _has_nvrtc_pch_apis_for_tests(),
    reason="PCH runtime APIs require NVRTC >= 12.8 bindings",
)


def _check_nvvm_arch(arch: str) -> bool:
    """Check if the given NVVM arch is supported by the installed libNVVM."""
    try:
        from cuda.bindings.utils import check_nvvm_compiler_options
    except ModuleNotFoundError as exc:
        if exc.name == "cuda":
            return False
        raise
    return check_nvvm_compiler_options([f"-arch={arch}"])


@pytest.fixture(scope="session")
def nvvm_ir():
    """Generate working NVVM IR with proper version metadata.
    The try clause here is used for older nvvm modules which
    might not have an ir_version() method. In which case the
    fallback assumes no version metadata will be present in
    the input nvvm ir
    """
    from cuda.core._program import _get_nvvm_module

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
"""
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
        pytest.param(
            ProgramOptions(debug=True, numba_debug=True),
            marks=pytest.mark.skipif(
                (_get_nvrtc_version_for_tests() or 0) < 13200,
                reason="numba_debug requires NVRTC >= 13.2",
            ),
        ),
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
        ProgramOptions(brief_diagnostics=True),
        ProgramOptions(builtin_move_forward=False),
        ProgramOptions(extensible_whole_program=True),
        ProgramOptions(fdevice_syntax_only=True),
        ProgramOptions(gen_opt_lto=True),
        ProgramOptions(minimal=True),
        ProgramOptions(no_source_include=True),
        # TODO: Add test for pre_include once we have a suitable header in the test environment
        # ProgramOptions(pre_include="cuda_runtime.h"),
        ProgramOptions(no_cache=True),
        pytest.param(
            ProgramOptions(arch="sm_100", device_float128=True),
            marks=pytest.mark.skipif(
                Device().compute_capability < (100, 0),
                reason="device_float128 requires sm_100 or later",
            ),
        ),
        ProgramOptions(frandom_seed="12345"),
        ProgramOptions(ofast_compile="min"),
        pytest.param(
            ProgramOptions(pch=True),
            marks=pytest.mark.skipif(
                (_get_nvrtc_version_for_tests() or 0) < 12800,
                reason="PCH requires NVRTC >= 12.8",
            ),
        ),
        # TODO: pch_dir requires actual PCH directory to exist - needs integration test
        # pytest.param(
        #     ProgramOptions(pch_dir="/tmp/pch"),
        #     marks=pytest.mark.skipif(
        #         (_get_nvrtc_version_for_tests() or 0) < 12800,
        #         reason="PCH requires NVRTC >= 12.8",
        #     ),
        # ),
        pytest.param(
            ProgramOptions(pch_verbose=True),
            marks=pytest.mark.skipif(
                (_get_nvrtc_version_for_tests() or 0) < 12800,
                reason="PCH requires NVRTC >= 12.8",
            ),
        ),
        pytest.param(
            ProgramOptions(pch_messages=False),
            marks=pytest.mark.skipif(
                (_get_nvrtc_version_for_tests() or 0) < 12800,
                reason="PCH requires NVRTC >= 12.8",
            ),
        ),
        pytest.param(
            ProgramOptions(instantiate_templates_in_pch=True),
            marks=pytest.mark.skipif(
                (_get_nvrtc_version_for_tests() or 0) < 12800,
                reason="PCH requires NVRTC >= 12.8",
            ),
        ),
    ],
)
def test_cpp_program_with_various_options(init_cuda, options):
    code = 'extern "C" __global__ void my_kernel() {}'
    program = Program(code, "c++", options)
    assert program.backend == "NVRTC"
    program.compile("ptx")
    program.close()


@pytest.mark.skipif(
    (_get_nvrtc_version_for_tests() or 0) < 13000,
    reason="buggy with NVRTC < 13.0 (File 'trace.json.json' could not be opened)",
)
def test_cpp_program_with_trace_option(init_cuda, tmp_path):
    code = 'extern "C" __global__ void my_kernel() {}'
    path = tmp_path / "trace"
    options = ProgramOptions(fdevice_time_trace=path)
    program = Program(code, "c++", options)
    assert program.backend == "NVRTC"
    program.compile("ptx")
    program.close()


@pytest.mark.skipif((_get_nvrtc_version_for_tests() or 0) < 12800, reason="PCH requires NVRTC >= 12.8")
def test_cpp_program_with_pch_options(init_cuda, tmp_path):
    code = 'extern "C" __global__ void my_kernel() {}'

    path = str(tmp_path / "test.pch")

    for opts in ({"create_pch": path}, {"use_pch": path}):
        options = ProgramOptions(**opts)
        program = Program(code, "c++", options)
        assert program.backend == "NVRTC"
        program.compile("ptx")
        program.close()


@nvrtc_pch_available
def test_cpp_program_pch_auto_creates(init_cuda, tmp_path):
    code = 'extern "C" __global__ void my_kernel() {}'
    pch_path = str(tmp_path / "test.pch")
    program = Program(code, "c++", ProgramOptions(create_pch=pch_path))
    assert program.pch_status is None  # not compiled yet
    program.compile("ptx")
    assert program.pch_status in ("created", "not_attempted", "failed")
    program.close()


def test_cpp_program_pch_status_none_without_pch(init_cuda):
    code = 'extern "C" __global__ void my_kernel() {}'
    program = Program(code, "c++")
    program.compile("ptx")
    assert program.pch_status is None
    program.close()


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
    program = Program(ptx_code_object.code.decode(), "ptx", options=options)
    assert program.backend == ("driver" if is_culink_backend else "nvJitLink")
    program.compile("cubin")
    program.close()


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

    program = Program(ptx_object_code.code.decode(), "ptx", options={"name": "24"})
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


def test_nvrtc_compile_invalid_code(init_cuda):
    """Compiling invalid C++ exercises the HANDLE_RETURN_NVRTC error path with compilation log."""
    from cuda.core._utils.cuda_utils import NVRTCError

    code = 'extern "C" __global__ void bad_kernel() { this_symbol_is_undefined(); }'
    program = Program(code, "c++")
    try:
        with pytest.raises(NVRTCError, match="compilation log"):
            program.compile("ptx")
    finally:
        program.close()


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
    # close() is idempotent
    program.close()


@nvvm_available
def test_nvvm_deferred_import():
    """Test that our deferred NVVM import works correctly"""
    from cuda.core._program import _get_nvvm_module

    nvvm = _get_nvvm_module()
    assert nvvm is not None


@nvvm_available
def test_nvvm_program_creation_compilation(nvvm_ir):
    """Test basic NVVM program creation"""
    program = Program(nvvm_ir, "nvvm")
    assert program.backend == "NVVM"
    assert program.handle is not None
    obj = program.compile("ptx")
    try:
        ker = obj.get_kernel("simple")
    except CUDAError as e:
        if re.search(r"CUDA_ERROR_UNSUPPORTED_PTX_VERSION", str(e)):
            pytest.xfail("PTX version not supported by current CUDA Driver")
        raise
    program.close()


@nvvm_available
def test_nvvm_compile_invalid_target(nvvm_ir):
    """Test that NVVM programs reject invalid compilation targets"""
    program = Program(nvvm_ir, "nvvm")
    with pytest.raises(ValueError, match='Unsupported target_type="cubin" for NVVM'):
        program.compile("cubin")
    program.close()


@nvvm_available
def test_nvvm_compile_invalid_ir():
    """Compiling invalid NVVM IR exercises the HANDLE_RETURN_NVVM error path."""
    from cuda.bindings.nvvm import nvvmError

    bad_ir = "this is not valid NVVM IR"
    program = Program(bad_ir, "nvvm")
    try:
        with pytest.raises(nvvmError):
            program.compile("ptx")
    finally:
        program.close()


@nvvm_available
@pytest.mark.parametrize("target_type", ["ptx", "ltoir"])
@pytest.mark.parametrize(
    "options",
    [
        ProgramOptions(name="test1", arch="sm_90", device_code_optimize=False),
        ProgramOptions(name="test2", arch="sm_100", device_code_optimize=False),
        ProgramOptions(name="test3", arch="sm_100", link_time_optimization=True),
        ProgramOptions(
            name="test4",
            arch="sm_90",
            ftz=True,
            prec_sqrt=False,
            prec_div=False,
            fma=True,
            device_code_optimize=True,
            link_time_optimization=True,
        ),
        pytest.param(
            ProgramOptions(name="test_sm110_1", arch="sm_110", device_code_optimize=False),
            marks=pytest.mark.skipif(
                not _check_nvvm_arch("compute_110"),
                reason="Compute capability 110 not supported by installed libNVVM",
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
                not _check_nvvm_arch("compute_110"),
                reason="Compute capability 110 not supported by installed libNVVM",
            ),
        ),
        pytest.param(
            ProgramOptions(name="test_sm110_3", arch="sm_110", link_time_optimization=True),
            marks=pytest.mark.skipif(
                not _check_nvvm_arch("compute_110"),
                reason="Compute capability 110 not supported by installed libNVVM",
            ),
        ),
    ],
)
def test_nvvm_program_options(init_cuda, nvvm_ir, options, target_type):
    """Test NVVM programs with different options and target types (ptx/ltoir)"""
    program = Program(nvvm_ir, "nvvm", options)
    assert program.backend == "NVVM"

    result = program.compile(target_type)
    assert isinstance(result, ObjectCode)
    assert result.name == options.name

    code_content = result.code
    assert len(code_content) > 0

    if target_type == "ptx":
        ptx_text = code_content.decode() if isinstance(code_content, bytes) else str(code_content)
        assert ".visible .entry simple(" in ptx_text

    program.close()


@nvvm_available
def test_nvvm_program_with_single_extra_source(nvvm_ir):
    """Test NVVM program with a single extra source"""
    from cuda.core._program import _get_nvvm_module

    nvvm = _get_nvvm_module()
    major, minor, debug_major, debug_minor = nvvm.ir_version()
    # helper nvvm ir for multiple module loading
    helper_nvvmir = f"""target triple = "nvptx64-unknown-cuda"
target datalayout = "e-p:64:64:64-i1:8:8-i8:8:8-i16:16:16-i32:32:32-i64:64:64-i128:128:128-f32:32:32-f64:64:64-v16:16:16-v32:32:32-v64:64:64-v128:128:128-n16:32:64"

define i32 @helper_add(i32 %x) {{
entry:
  %result = add i32 %x, 1
  ret i32 %result
}}

!nvvmir.version = !{{!0}}
!0 = !{{i32 {major}, i32 {minor}, i32 {debug_major}, i32 {debug_minor}}}
"""

    options = ProgramOptions(
        name="multi_module_test",
        extra_sources=[
            ("helper", helper_nvvmir),
        ],
    )
    program = Program(nvvm_ir, "nvvm", options)

    assert program.backend == "NVVM"

    ptx_code = program.compile("ptx")
    assert isinstance(ptx_code, ObjectCode)
    assert ptx_code.name == "multi_module_test"

    program.close()


@nvvm_available
def test_nvvm_program_with_multiple_extra_sources():
    """Test NVVM program with multiple extra sources"""
    from cuda.core._program import _get_nvvm_module

    nvvm = _get_nvvm_module()
    major, minor, debug_major, debug_minor = nvvm.ir_version()

    main_nvvm_ir = f"""target triple = "nvptx64-unknown-cuda"
target datalayout = "e-p:64:64:64-i1:8:8-i8:8:8-i16:16:16-i32:32:32-i64:64:64-i128:128:128-f32:32:32-f64:64:64-v16:16:16-v32:32:32-v64:64:64-v128:128:128-n16:32:64"

declare i32 @helper_add(i32) nounwind readnone
declare i32 @helper_mul(i32) nounwind readnone

define void @main_kernel(i32* %data) {{
entry:
  %tid = call i32 @llvm.nvvm.read.ptx.sreg.tid.x()
  %ptr = getelementptr inbounds i32, i32* %data, i32 %tid
  %val = load i32, i32* %ptr, align 4

  %val1 = call i32 @helper_add(i32 %val)
  %val2 = call i32 @helper_mul(i32 %val1)

  store i32 %val2, i32* %ptr, align 4
  ret void
}}

declare i32 @llvm.nvvm.read.ptx.sreg.tid.x() nounwind readnone

!nvvm.annotations = !{{!0}}
!0 = !{{void (i32*)* @main_kernel, !"kernel", i32 1}}

!nvvmir.version = !{{!1}}
!1 = !{{i32 {major}, i32 {minor}, i32 {debug_major}, i32 {debug_minor}}}
"""

    helper1_ir = f"""target triple = "nvptx64-unknown-cuda"
target datalayout = "e-p:64:64:64-i1:8:8-i8:8:8-i16:16:16-i32:32:32-i64:64:64-i128:128:128-f32:32:32-f64:64:64-v16:16:16-v32:32:32-v64:64:64-v128:128:128-n16:32:64"

define i32 @helper_add(i32 %x) nounwind readnone {{
entry:
  %result = add i32 %x, 1
  ret i32 %result
}}

!nvvmir.version = !{{!0}}
!0 = !{{i32 {major}, i32 {minor}, i32 {debug_major}, i32 {debug_minor}}}
"""

    helper2_ir = f"""target triple = "nvptx64-unknown-cuda"
target datalayout = "e-p:64:64:64-i1:8:8-i8:8:8-i16:16:16-i32:32:32-i64:64:64-i128:128:128-f32:32:32-f64:64:64-v16:16:16-v32:32:32-v64:64:64-v128:128:128-n16:32:64"

define i32 @helper_mul(i32 %x) nounwind readnone {{
entry:
  %result = mul i32 %x, 2
  ret i32 %result
}}

!nvvmir.version = !{{!0}}
!0 = !{{i32 {major}, i32 {minor}, i32 {debug_major}, i32 {debug_minor}}}
"""

    options = ProgramOptions(
        name="nvvm_multi_helper_test",
        extra_sources=[
            ("helper1", helper1_ir),
            ("helper2", helper2_ir),
        ],
    )
    program = Program(main_nvvm_ir, "nvvm", options)

    assert program.backend == "NVVM"
    ptx_code = program.compile("ptx")
    assert isinstance(ptx_code, ObjectCode)
    assert ptx_code.name == "nvvm_multi_helper_test"

    ltoir_code = program.compile("ltoir")
    assert isinstance(ltoir_code, ObjectCode)
    assert ltoir_code.name == "nvvm_multi_helper_test"

    program.close()


@nvvm_available
def test_bitcode_format(minimal_nvvmir):
    from contextlib import ExitStack, closing

    if len(minimal_nvvmir) < 4:
        pytest.skip("Bitcode file is not valid or empty")

    options = ProgramOptions(name="minimal_nvvmir_bitcode_test", arch="sm_90")

    with ExitStack() as stack:
        program = stack.enter_context(closing(Program(minimal_nvvmir, "nvvm", options)))
        assert program.backend == "NVVM"
        ptx_result = program.compile("ptx")
        assert isinstance(ptx_result, ObjectCode)
        assert ptx_result.name == "minimal_nvvmir_bitcode_test"
        assert len(ptx_result.code) > 0

        program_lto = stack.enter_context(closing(Program(minimal_nvvmir, "nvvm", options)))
        ltoir_result = program_lto.compile("ltoir")
        assert isinstance(ltoir_result, ObjectCode)
        assert len(ltoir_result.code) > 0


def test_cpp_program_with_extra_sources():
    # negative test with NVRTC with multiple sources
    code = 'extern "C" __global__ void my_kernel(){}'
    helper = 'extern "C" __global__ void helper(){}'
    options = ProgramOptions(extra_sources=[("helper", helper)])
    with pytest.raises(ValueError, match="extra_sources is not supported by the NVRTC backend"):
        Program(code, "c++", options)


def test_program_options_as_bytes_nvrtc():
    """Test ProgramOptions.as_bytes() for NVRTC backend"""
    options = ProgramOptions(arch="sm_80", debug=True, lineinfo=True, ftz=True)
    nvrtc_options = options.as_bytes("nvrtc")

    # Should return list of bytes
    assert isinstance(nvrtc_options, list)
    assert all(isinstance(opt, bytes) for opt in nvrtc_options)

    # Decode to check content
    options_str = [opt.decode() for opt in nvrtc_options]
    assert "-arch=sm_80" in options_str
    assert "--device-debug" in options_str
    assert "--generate-line-info" in options_str
    assert "--ftz=true" in options_str


@nvvm_available
def test_program_options_as_bytes_nvvm():
    """Test ProgramOptions.as_bytes() for NVVM backend"""
    options = ProgramOptions(arch="sm_80", debug=True, ftz=True, device_code_optimize=True)
    nvvm_options = options.as_bytes("nvvm")

    # Should return list of bytes (same as other backends)
    assert isinstance(nvvm_options, list)
    assert all(isinstance(opt, bytes) for opt in nvvm_options)

    # Decode to check content
    options_str = [opt.decode() for opt in nvvm_options]
    assert "-arch=compute_80" in options_str
    assert "-g" in options_str
    assert "-ftz=1" in options_str
    assert "-opt=3" in options_str


def test_program_options_as_bytes_invalid_backend():
    """Test ProgramOptions.as_bytes() with invalid backend"""
    options = ProgramOptions(arch="sm_80")
    with pytest.raises(ValueError, match="Unknown backend 'invalid'"):
        options.as_bytes("invalid")


@nvvm_available
def test_program_options_as_bytes_nvvm_unsupported_option():
    """Test that unsupported options raise CUDAError for NVVM backend"""
    options = ProgramOptions(arch="sm_80", lineinfo=True)
    with pytest.raises(CUDAError, match="not supported by NVVM backend"):
        options.as_bytes("nvvm")
