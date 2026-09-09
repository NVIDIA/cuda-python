# SPDX-FileCopyrightText: Copyright (c) 2024-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

import re
import shutil
import subprocess
import sys
import warnings

import pytest

from cuda.bindings._internal.utils import FunctionNotFoundError
from cuda.core import _linker
from cuda.core._device import Device
from cuda.core._module import Kernel, ObjectCode
from cuda.core._program import Program, ProgramOptions
from cuda.core._utils.cuda_utils import CUDAError, handle_return, nvrtc
from cuda.core.typing import CompilerBackendType, PCHStatusType
from cuda.pathfinder import DynamicLibNotFoundError

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


def _get_nvrtc_version_for_tests():
    """
    Get NVRTC version.

    Returns:
        int: Version in format major * 1000 + minor * 100 (e.g., 13200 for CUDA 13.2)
        None: If NVRTC is not available
    """
    try:
        nvrtc_major, nvrtc_minor = handle_return(nvrtc.nvrtcVersion())
        return nvrtc_major * 1000 + nvrtc_minor * 100
    except (DynamicLibNotFoundError, FunctionNotFoundError):
        # libnvrtc not loadable, or nvrtcVersion symbol missing.
        return None
    # CUDAError from a successfully loaded library propagates (real bug).


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

bundled_headers_available = pytest.mark.skipif(
    (_get_nvrtc_version_for_tests() or 0) < 13300,
    reason="use_bundled_headers requires NVRTC >= 13.3",
)


def _has_check_nvvm_compiler_options():
    try:
        import cuda.bindings.utils as utils
    except ModuleNotFoundError:
        return False
    return hasattr(utils, "check_nvvm_compiler_options")


has_nvvm_option_checker = pytest.mark.skipif(
    not _has_check_nvvm_compiler_options(),
    reason="cuda.bindings.utils.check_nvvm_compiler_options not available (cuda-bindings too old?)",
)


def _check_nvvm_arch(arch: str) -> bool:
    """Check if the given NVVM arch is supported by the installed libNVVM."""
    from cuda.bindings.utils import check_nvvm_compiler_options

    return check_nvvm_compiler_options([f"-arch={arch}"])


def _check_nvvm_supports_numba_debug() -> bool:
    """Check if the installed libNVVM recognizes -numba-debug.

    libNVVM only accepts single-dashed options, so the double-dashed spelling
    used by NVRTC is rejected by every libNVVM version.
    """
    if not _has_check_nvvm_compiler_options():
        return False
    from cuda.bindings.utils import check_nvvm_compiler_options

    return check_nvvm_compiler_options(["-numba-debug"])


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
    assert isinstance(program.backend, CompilerBackendType)
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
    assert isinstance(program.pch_status, PCHStatusType)
    program.close()


@bundled_headers_available
@pytest.mark.agent_authored(model="claude-sonnet-5")
def test_use_bundled_headers_installs_and_compiles(init_cuda, tmp_path, monkeypatch):
    """``use_bundled_headers`` should install NVRTC's bundled CUDA/CCCL headers into the
    (monkeypatched) cache directory and make them available on the include path, without
    a CUDA Toolkit or any user-supplied ``include_path``."""
    import cuda.core._program as _program_module

    cache_root = tmp_path / "cache-root"
    monkeypatch.setattr(_program_module, "_default_cache_dir", lambda: cache_root)

    code = """
#include <cuda/std/type_traits>
extern "C" __global__ void my_kernel(int *out) {
    *out = cuda::std::is_integral<int>::value;
}
"""
    headers_dir = cache_root / "nvrtc-bundled-headers"
    assert not headers_dir.exists()

    # Sanity check: without use_bundled_headers, the CCCL header isn't found (proves the
    # option -- not some ambient CUDA Toolkit install -- is what makes the compile below work).
    program = Program(code, "c++")
    try:
        with pytest.raises(CUDAError, match="could not open source file"):
            program.compile("ptx")
    finally:
        program.close()

    program = Program(code, "c++", ProgramOptions(use_bundled_headers=True))
    try:
        object_code = program.compile("ptx")
    finally:
        program.close()
    assert isinstance(object_code, ObjectCode)

    assert headers_dir.is_dir()
    assert (headers_dir / ".nvrtc_headers_version").is_file()
    assert (headers_dir / "cccl").is_dir()
    assert (headers_dir / "cccl" / "cuda" / "std" / "type_traits").is_file()


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
    # ``numba_debug`` is deliberately absent: it was listed here as a link-time
    # no-op (#1287), but no linker backend accepts it, so it was dropped
    # silently (#2640). The PTX path now warns; see
    # test_ptx_program_numba_debug_warns_and_is_ignored.
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


# arch is passed explicitly so the current device is not queried.
@pytest.mark.agent_authored(model="claude-opus-5")
@pytest.mark.parametrize("name", [None, "my_program"])
def test_program_options_name_accepts_none(name):
    options = ProgramOptions(name=name, arch="sm_90")
    expected = "default_program" if name is None else name
    assert options.name == expected
    assert options._name == expected.encode()


# This is tested against the current device's arch
def test_program_compile_valid_target_type(init_cuda):
    code = 'extern "C" __global__ void my_kernel() {}'
    program = Program(code, "c++", options=ProgramOptions(name="42"))

    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        ptx_object_code = program.compile("ptx")
        assert isinstance(ptx_object_code, ObjectCode)
        assert ptx_object_code.name == "42"
        if any("The CUDA driver version is older than the backend version" in str(warning.message) for warning in w):
            pytest.skip("PTX version too new for current driver")
        ptx_kernel = ptx_object_code.get_kernel("my_kernel")
        assert isinstance(ptx_kernel, Kernel)

    program = Program(ptx_object_code.code.decode(), "ptx", options=ProgramOptions(name="24"))
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


@pytest.mark.agent_authored(model="gpt-5.6")
def test_closed_program_rejects_compile():
    program = Program('extern "C" __global__ void my_kernel() {}', "c++")
    assert not program.is_closed
    program.close()

    assert program.is_closed
    with pytest.raises(RuntimeError, match="Program has been closed"):
        program.compile("ptx")


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
def test_nvvm_accepts_bytearray_input(nvvm_ir):
    """Program(..., 'nvvm') must accept bytearray input.

    Regression for a bug where the NVVM init branch retained the coerced
    ``self._code`` as bytes but still cast the original ``code`` object to
    ``<bytes>`` for the C pointer -- tripping a runtime type error for
    bytearray inputs before nvvmAddModuleToProgram was called.
    """
    program = Program(bytearray(nvvm_ir, "utf-8"), "nvvm")
    try:
        assert program.backend == "NVVM"
        assert program.handle is not None
    finally:
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
            marks=[
                has_nvvm_option_checker,
                pytest.mark.skipif(
                    _has_check_nvvm_compiler_options() and not _check_nvvm_arch("compute_110"),
                    reason="Compute capability 110 not supported by installed libNVVM",
                ),
            ],
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
            marks=[
                has_nvvm_option_checker,
                pytest.mark.skipif(
                    _has_check_nvvm_compiler_options() and not _check_nvvm_arch("compute_110"),
                    reason="Compute capability 110 not supported by installed libNVVM",
                ),
            ],
        ),
        pytest.param(
            ProgramOptions(name="test_sm110_3", arch="sm_110", link_time_optimization=True),
            marks=[
                has_nvvm_option_checker,
                pytest.mark.skipif(
                    _has_check_nvvm_compiler_options() and not _check_nvvm_arch("compute_110"),
                    reason="Compute capability 110 not supported by installed libNVVM",
                ),
            ],
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
    nvrtc_options = options.as_bytes(CompilerBackendType.NVRTC)
    assert isinstance(nvrtc_options, list)
    assert all(isinstance(opt, bytes) for opt in nvrtc_options)
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
    assert isinstance(nvvm_options, list)
    assert all(isinstance(opt, bytes) for opt in nvvm_options)
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
def test_nvvm_program_options_as_bytes_numba_debug():
    """numba_debug must be plumbed through to libNVVM as -numba-debug
    (see #1287, #2570). libNVVM rejects the double-dashed spelling."""
    options = ProgramOptions(arch="sm_80", debug=True, numba_debug=True)
    nvvm_bytes = options.as_bytes("nvvm")
    assert b"-numba-debug" in nvvm_bytes
    assert b"--numba-debug" not in nvvm_bytes
    assert b"-g" in nvvm_bytes


@pytest.mark.agent_authored(model="claude-opus-5[1m]")
def test_nvvm_options_reject_double_dash():
    """The guard must name a double-dashed option rather than let libNVVM
    reject it with an opaque error (see #2570)."""
    from cuda.core._program import _assert_single_dashed_nvvm_options

    _assert_single_dashed_nvvm_options(["-arch=compute_80", "-g", "-numba-debug"])

    with pytest.raises(RuntimeError, match=r"--numba-debug.*double-dashed"):
        _assert_single_dashed_nvvm_options(["-arch=compute_80", "--numba-debug"])


@nvvm_available
@pytest.mark.agent_authored(model="claude-opus-5[1m]")
def test_nvvm_program_options_as_bytes_all_single_dashed():
    """Every option cuda.core emits to libNVVM must be single-dashed, because
    libNVVM rejects the double-dashed spelling of all of them (see #2570).
    This covers every NVVM-supported field of ProgramOptions."""
    options = ProgramOptions(
        arch="sm_80",
        debug=True,
        numba_debug=True,
        device_code_optimize=True,
        ftz=True,
        prec_sqrt=True,
        prec_div=True,
        fma=True,
    )
    nvvm_bytes = options.as_bytes("nvvm")
    assert nvvm_bytes, "expected at least one emitted option"
    offenders = [o for o in nvvm_bytes if o.startswith(b"--")]
    assert not offenders, f"double-dashed options are rejected by libNVVM: {offenders}"


@nvvm_available
@pytest.mark.skipif(
    not _check_nvvm_supports_numba_debug(),
    reason="installed libNVVM does not recognize -numba-debug",
)
def test_nvvm_program_numba_debug(init_cuda, nvvm_ir):
    options = ProgramOptions(arch="sm_80", debug=True, numba_debug=True)
    program = Program(nvvm_ir, "nvvm", options)
    try:
        assert program.backend == "NVVM"
        result = program.compile("ptx")
        assert isinstance(result, ObjectCode)
        assert len(result.code) > 0
    finally:
        program.close()


def test_program_options_repr():
    """ProgramOptions.__repr__ returns a human-readable string."""
    opts = ProgramOptions(name="mykernel", arch="sm_80")
    r = repr(opts)
    assert "ProgramOptions" in r
    assert "mykernel" in r
    assert "sm_80" in r


def test_program_options_bad_define_macro_short_tuple():
    """define_macro with a 1-element tuple raises RuntimeError."""
    opts = ProgramOptions(name="test", arch="sm_80", define_macro=("ONLY_NAME",))
    with pytest.raises(RuntimeError, match="Expected define_macro tuple"):
        opts.as_bytes("nvrtc")


def test_program_options_bad_define_macro_non_str_value():
    """define_macro tuple with a non-string value raises RuntimeError."""
    opts = ProgramOptions(name="test", arch="sm_80", define_macro=("MY_MACRO", 99))
    with pytest.raises(RuntimeError, match="Expected define_macro tuple"):
        opts.as_bytes("nvrtc")


def test_program_options_bad_define_macro_list_non_str():
    """define_macro list containing a non-str/non-tuple item raises RuntimeError."""
    opts = ProgramOptions(name="test", arch="sm_80", define_macro=[42])
    with pytest.raises(RuntimeError, match="Expected define_macro"):
        opts.as_bytes("nvrtc")


def test_program_options_bad_define_macro_list_bad_tuple():
    """define_macro list with a malformed tuple inside raises RuntimeError."""
    opts = ProgramOptions(name="test", arch="sm_80", define_macro=[("ONLY_NAME",)])
    with pytest.raises(RuntimeError, match="Expected define_macro"):
        opts.as_bytes("nvrtc")


def test_ptx_program_extra_sources_unsupported(ptx_code_object):
    """PTX backend raises ValueError when extra_sources is specified."""
    options = ProgramOptions(extra_sources=[("module1", b"data")])
    with pytest.raises(ValueError, match="extra_sources is not supported by the PTX backend"):
        Program(ptx_code_object.code.decode(), "ptx", options)


@pytest.mark.agent_authored(model="claude-opus-5")
def test_ptx_program_numba_debug_warns_and_is_ignored(init_cuda, ptx_code_object):
    """PTX inputs go to the linker, which cannot honor numba_debug (#2640).

    It used to be forwarded into ``LinkerOptions`` and dropped without a word,
    so the compile appeared to succeed with the option applied. It is still
    ignored -- no linker can do anything with it -- but no longer silently.

    ``UserWarning``, not ``DeprecationWarning``: ``ProgramOptions.numba_debug``
    is not deprecated, it is supported on NVVM/NVRTC and merely inapplicable to
    this backend.
    """
    with pytest.warns(UserWarning, match="numba_debug is ignored for code_type='ptx'"):
        program = Program(ptx_code_object.code.decode(), "ptx", ProgramOptions(numba_debug=True))
    assert program.compile("cubin") is not None


@pytest.mark.agent_authored(model="claude-opus-5")
@pytest.mark.parametrize("value", [None, False])
def test_ptx_program_numba_debug_unset_or_false_does_not_warn(init_cuda, ptx_code_object, value):
    """The gate is truthiness: only an enabled ``numba_debug`` asks for
    something the PTX path cannot deliver, so ``False`` is not worth a warning."""
    with warnings.catch_warnings():
        warnings.simplefilter("error", UserWarning)
        program = Program(ptx_code_object.code.decode(), "ptx", ProgramOptions(numba_debug=value))
    assert program.compile("cubin") is not None


def test_ptx_program_handle_is_linker_handle(init_cuda, ptx_code_object):
    """Program.handle for the PTX backend delegates to the linker handle."""
    program = Program(ptx_code_object.code.decode(), "ptx")
    handle = program.handle
    assert handle is not None
    assert int(handle) != 0
    program.close()


@nvvm_available
def test_nvvm_program_wrong_code_type():
    """NVVM backend raises TypeError when code is not str/bytes/bytearray."""
    with pytest.raises(TypeError, match="NVVM IR code must be provided as str, bytes, or bytearray"):
        Program(42, "nvvm")


def test_extra_sources_not_sequence():
    """extra_sources must be a sequence; non-sequence raises TypeError."""
    with pytest.raises(TypeError, match="extra_sources must be a sequence of 2-tuples"):
        ProgramOptions(name="test", arch="sm_80", extra_sources=42)


def test_extra_sources_bad_module_not_tuple():
    """extra_sources items must be 2-tuples; non-tuple item raises TypeError."""
    with pytest.raises(TypeError, match="Each extra module must be a 2-tuple"):
        ProgramOptions(name="test", arch="sm_80", extra_sources=["not_a_tuple"])


def test_extra_sources_bad_module_name_not_str():
    """extra_sources module name must be a string; non-str raises TypeError."""
    with pytest.raises(TypeError, match="Module name at index 0 must be a string"):
        ProgramOptions(name="test", arch="sm_80", extra_sources=[(42, b"source")])


def test_extra_sources_bad_module_source_wrong_type():
    """extra_sources module source must be str/bytes/bytearray."""
    with pytest.raises(TypeError, match="Module source at index 0 must be str"):
        ProgramOptions(name="test", arch="sm_80", extra_sources=[("mod", 42)])


def test_extra_sources_empty_source():
    """extra_sources module source cannot be empty bytes."""
    with pytest.raises(ValueError, match="Module source for 'mod'.*cannot be empty"):
        ProgramOptions(name="test", arch="sm_80", extra_sources=[("mod", b"")])


@pytest.mark.parametrize(
    ("extra_sources", "expected"),
    [
        (None, None),
        ([("mod_s", "kernel-as-string")], [(b"mod_s", b"kernel-as-string")]),
        (
            [("mod_ba", bytearray(b"\x00\x01module-as-bytearray"))],
            [(b"mod_ba", b"\x00\x01module-as-bytearray")],
        ),
        ([("mod_b", b"\x00\x01module-as-bytes")], [(b"mod_b", b"\x00\x01module-as-bytes")]),
    ],
    ids=["none", "str", "bytearray", "bytes"],
)
def test_prepare_extra_sources_bytes(extra_sources, expected):
    """_prepare_extra_sources_bytes converts each input type to (bytes, bytes) tuples (None passthrough)."""
    # arch is set to skip __post_init__'s Device() lookup, keeping this a pure unit test.
    opts = ProgramOptions(name="t", arch="sm_80", extra_sources=extra_sources)
    result = opts._prepare_extra_sources_bytes()
    assert result == expected
    # bytearray == bytes by content, so == alone misses type regressions.
    if result is not None:
        for name, source in result:
            assert isinstance(name, bytes), f"name should be bytes, got {type(name).__name__}"
            assert isinstance(source, bytes), f"source should be bytes, got {type(source).__name__}"


def test_find_libdevice_path_delegates_to_pathfinder(monkeypatch):
    """_find_libdevice_path calls cuda.pathfinder.find_bitcode_lib('device') and returns its result."""
    import cuda.pathfinder
    from cuda.core import _program

    captured = []
    sentinel = "/fake/path/libdevice.10.bc"

    def fake_find(name):
        captured.append(name)
        return sentinel

    monkeypatch.setattr(cuda.pathfinder, "find_bitcode_lib", fake_find)
    assert _program._find_libdevice_path() == sentinel
    assert captured == ["device"]


@pytest.mark.agent_authored(model="cursor-grok-4.6")
def test_nvrtc_debug_materializes_source_to_temp_file(init_cuda, tmp_path):
    """debug/lineinfo writes NVRTC source to a real path; off and explicit name= do not."""
    import os

    code = 'extern "C" __global__ void matmul() {}'

    # case 1: (debug=False, lineinfo=False)
    off = Program(code, "c++", ProgramOptions(arch="sm_80"))
    assert off.compile("ptx").name == "default_program"
    off.close()

    # case 2: (debug=True or lineinfo=True) and explicit_name is provided
    explicit_name = str(tmp_path / "user_kernel.cu")
    named = Program(code, "c++", ProgramOptions(name=explicit_name, debug=True, arch="sm_80"))
    assert named.compile("ptx").name == explicit_name
    assert not os.path.isfile(explicit_name)
    named.close()

    # case 3: (debug=True or lineinfo=True) and explicit_name is not provided
    default_named = Program(code, "c++", ProgramOptions(debug=True, arch="sm_80"))
    implicit_name = default_named.compile("ptx").name
    try:
        assert os.path.isfile(implicit_name)
        assert re.fullmatch(r"test_program_matmul_[a-z0-9_]{8}\.cu", os.path.basename(implicit_name))
        with open(implicit_name, encoding="utf-8") as fh:
            assert fh.read() == code
    finally:
        default_named.close()
    assert not os.path.isfile(implicit_name)


@pytest.mark.agent_authored(model="cursor-grok-4.6")
@pytest.mark.thread_unsafe(reason="monkeypatches tempfile.mkstemp on the Program module")
def test_nvrtc_debug_falls_back_when_tmp_not_writable(init_cuda, monkeypatch):
    """debug=True still compiles if the temp dir cannot be written (issue #2422)."""
    from cuda.core import _program

    def _denied(*_args, **_kwargs):
        raise OSError(30, "Read-only file system")

    monkeypatch.setattr(_program.tempfile, "mkstemp", _denied)

    code = 'extern "C" __global__ void matmul() {}'
    prog = Program(code, "c++", ProgramOptions(debug=True, arch="sm_80"))
    try:
        assert prog.compile("ptx").name == "default_program"
    finally:
        prog.close()


@pytest.mark.agent_authored(model="cursor-grok-4.6")
def test_nvrtc_debug_concurrent_compile_uses_unique_temp_files(init_cuda):
    """Same kernel compiled concurrently gets distinct mkstemp paths (issue #2422)."""
    import os
    import threading
    from concurrent.futures import ThreadPoolExecutor

    code = 'extern "C" __global__ void matmul() {}'
    n = 2
    barrier = threading.Barrier(n)

    def _compile_one():
        Device().set_current()
        barrier.wait()
        prog = Program(code, "c++", ProgramOptions(debug=True, arch="sm_80"))
        name = prog.compile("ptx").name
        return prog, name

    with ThreadPoolExecutor(max_workers=n) as pool:
        futures = [pool.submit(_compile_one) for _ in range(n)]
        results = [fut.result() for fut in futures]

    progs, names = zip(*results)
    try:
        assert len(set(names)) == n
        for name in names:
            assert os.path.isfile(name)
            assert re.fullmatch(r"test_program_matmul_[a-z0-9_]{8}\.cu", os.path.basename(name))
            with open(name, encoding="utf-8") as fh:
                assert fh.read() == code
    finally:
        for prog in progs:
            prog.close()
    for name in names:
        assert not os.path.isfile(name)


@pytest.mark.agent_authored(model="claude-opus-5")
def test_nvrtc_debug_preserves_quoted_include_resolution(init_cuda, tmp_path, monkeypatch):
    """A quoted #include keeps resolving once debug redirects the NVRTC name (issue #2422).

    NVRTC looks for #include "..." in the directory of the name it was handed, so
    pointing that name at a temp .cu moves the search away from where the header
    lives and turning debug on alone breaks a compile that worked without it.
    """
    import os

    (tmp_path / "local.h").write_text("#define BUMP 7\n", encoding="utf-8")
    monkeypatch.chdir(tmp_path)
    code = '#include "local.h"\nextern "C" __global__ void matmul(int* out) { *out = BUMP; }\n'

    for debug in (False, True):
        prog = Program(code, "c++", ProgramOptions(arch="sm_80", debug=debug))
        try:
            name = prog.compile("ptx").name
        finally:
            prog.close()
        if debug:
            # Only a regression test while the name really does move out of the
            # directory holding local.h; otherwise it would pass for free.
            assert os.path.dirname(os.path.realpath(name)) != os.path.realpath(tmp_path)


@pytest.mark.agent_authored(model="claude-opus-5")
@pytest.mark.parametrize("debug", [False, True])
def test_nvrtc_debug_keeps_file_the_caller_named(init_cuda, tmp_path, debug):
    """Program only unlinks a temp file it wrote itself (issue #2422).

    The name handed to NVRTC doubled as the cleanup target, so a name pointing at
    a file that already existed made teardown delete the caller's own source.
    """
    import gc

    source = tmp_path / "matmul.cu"
    contents = "// the caller's own file\n"
    source.write_text(contents, encoding="utf-8")
    code = 'extern "C" __global__ void matmul() {}'
    options = ProgramOptions(arch="sm_80", name=str(source), debug=debug)

    prog = Program(code, "c++", options)
    prog.compile("ptx")
    prog.close()
    assert source.is_file(), "close() deleted a file the caller owns"

    # __dealloc__ runs the same cleanup, so collection must spare it too.
    prog = Program(code, "c++", options)
    prog.compile("ptx")
    del prog
    gc.collect()
    assert source.is_file(), "collection deleted a file the caller owns"
    assert source.read_text(encoding="utf-8") == contents


@pytest.mark.agent_authored(model="cursor-grok-4.6")
def test_cuda_gdb_shows_nvrtc_debug_source_lines(init_cuda):
    import pathlib

    cuda_gdb = shutil.which("cuda-gdb")
    if cuda_gdb is None:
        pytest.skip("cuda-gdb is not on PATH")

    child = pathlib.Path(__file__).resolve().parent / "helpers" / "cuda_gdb_src.py"
    proc = subprocess.run(  # noqa: S603 - trusted argv: cuda-gdb + this interpreter + in-tree helper
        [
            cuda_gdb,
            "--batch",
            "-ex",
            "set cuda break_on_launch application",
            "-ex",
            "run",
            "-ex",
            "list",
            "--args",
            sys.executable,
            "-u",
            str(child),
        ],
        check=False,
        capture_output=True,
        text=True,
        timeout=120,
    )
    output = (proc.stdout or "") + (proc.stderr or "")
    lowered = output.lower()
    if "operation not permitted" in lowered or "ptrace" in lowered or "debugging is not possible" in lowered:
        pytest.xfail("cuda-gdb is not usable for debugging on this machine: " + output)
    assert re.search(r"cuda_gdb_src_kernel_\w+\.cu", output), output
    assert "ISSUE_2422_SOURCE_LINE" in output, output
    assert "No such file or directory" not in output, output


def test_nvrtc_compile_with_logs_capture(init_cuda):
    """Program.compile with logs= exercises the NVRTC program-log reading path."""
    import io

    # #warning generates a non-empty NVRTC program log, ensuring logsize > 1.
    code = '#warning "test log capture"\nextern "C" __global__ void my_kernel() {}'
    program = Program(code, "c++")
    logs = io.StringIO()
    result = program.compile("ptx", logs=logs)
    assert isinstance(result, ObjectCode)
    assert logs.getvalue(), "Expected non-empty compilation log from #warning directive"
    program.close()


@pytest.mark.agent_authored(model="gpt-5.6-sol")
def test_program_options_bad_define_macro_nested_list_invalid_element():
    """Nested define_macro list with a non-processable element raises at the element."""
    # [("MACRO", "1")] makes is_nested_sequence True; 42 fails the inner processor.
    opts = ProgramOptions(name="test", arch="sm_80", define_macro=[("MACRO", "1"), 42])
    with pytest.raises(RuntimeError, match=r"Expected define_macro.*got 42"):
        opts.as_bytes("nvrtc")


@pytest.mark.parametrize(
    "kwargs",
    [
        {"relocatable_device_code": True},
        {"extensible_whole_program": True},
        {"lineinfo": True},
        {"ptxas_options": "-v"},
        {"max_register_count": 32},
        {"use_fast_math": True},
        {"extra_device_vectorization": True},
        {"gen_opt_lto": True},
        {"define_macro": "M"},
        {"undefine_macro": "M"},
        {"include_path": "include-dir"},
        pytest.param({"use_bundled_headers": True}, marks=bundled_headers_available),
        {"pre_include": "header.h"},
        {"no_source_include": True},
        {"std": "c++17"},
        {"builtin_move_forward": False},
        {"builtin_initializer_list": False},
        {"disable_warnings": True},
        {"restrict": True},
        {"device_as_default_execution_space": True},
        {"device_int128": True},
        {"optimization_info": "inline"},
        {"no_display_error_number": True},
        {"diag_error": 1},
        {"diag_suppress": 1},
        {"diag_warn": 1},
        {"brief_diagnostics": True},
        {"time": "timing.csv"},
        {"split_compile": 2},
        {"fdevice_syntax_only": True},
        {"minimal": True},
    ],
    ids=lambda kw: next(iter(kw)),
)
@pytest.mark.agent_authored(model="gpt-5.6-sol")
def test_nvvm_options_reject_each_unsupported_flag(kwargs):
    """Every NVVM-unsupported option is rejected, named, and reported alone."""
    # This table mirrors _prepare_nvvm_options_impl's rejection list one-for-one.
    options = ProgramOptions(arch="sm_80", **kwargs)
    name = next(iter(kwargs))
    with pytest.raises(CUDAError, match=rf"^The following options are not supported by NVVM backend: {name}$"):
        options.as_bytes("nvvm")


@pytest.mark.agent_authored(model="gpt-5.6-sol")
def test_nvrtc_as_bytes_emits_sequence_and_uncommon_flags():
    """as_bytes emits the NVRTC spellings that compile-option tests do not hit."""
    options = ProgramOptions(
        arch="sm_80",
        ptxas_options="-v",
        pre_include=["a.h", "b.h"],
        device_float128=True,
        diag_warn=[1000, 1001],
        time="timing.csv",
        split_compile=2,
        pch_dir="pch-cache",
    )
    flags = [opt.decode() for opt in options.as_bytes("nvrtc")]
    assert "--ptxas-options=-v" in flags
    assert "--pre-include=a.h" in flags
    assert "--pre-include=b.h" in flags
    assert "--device-float128" in flags
    assert "--diag-warn=1000" in flags
    assert "--diag-warn=1001" in flags
    assert "--time=timing.csv" in flags
    assert "--split-compile=2" in flags
    assert "--pch-dir=pch-cache" in flags

    single_pre = ProgramOptions(arch="sm_80", pre_include="only.h")
    assert "--pre-include=only.h" in [opt.decode() for opt in single_pre.as_bytes("nvrtc")]


@pytest.mark.thread_unsafe(reason="patches the process-global os.fdopen and tempfile.mkstemp")
@pytest.mark.agent_authored(model="gpt-5.6-sol")
def test_nvrtc_debug_falls_back_when_temp_file_write_fails(init_cuda, monkeypatch):
    """A write failure removes the temporary source and falls back to the default name."""
    import contextlib
    import os

    from cuda.core import _program

    real_fdopen = os.fdopen
    real_mkstemp = _program.tempfile.mkstemp
    temp_paths = []

    class _FailingWriter:
        def write(self, _code):
            raise OSError("No space left on device")

    @contextlib.contextmanager
    def _write_fails(fd, *args, **kwargs):
        with real_fdopen(fd, *args, **kwargs):
            yield _FailingWriter()

    def _record_mkstemp(*args, **kwargs):
        fd, path = real_mkstemp(*args, **kwargs)
        temp_paths.append(path)
        return fd, path

    monkeypatch.setattr(_program.os, "fdopen", _write_fails)
    monkeypatch.setattr(_program.tempfile, "mkstemp", _record_mkstemp)

    code = 'extern "C" __global__ void matmul() {}'
    prog = Program(code, "c++", ProgramOptions(debug=True, arch="sm_80"))
    try:
        assert len(temp_paths) == 1
        assert not os.path.exists(temp_paths[0])
        assert prog.compile("ptx").name == "default_program"
    finally:
        prog.close()


@nvvm_available
@pytest.mark.agent_authored(model="gpt-5.6-sol")
def test_nvvm_compile_with_libdevice(nvvm_ir):
    """use_libdevice resolves a referenced libdevice function into the generated PTX."""
    store = "  store i32 %call, i32* %data, align 4"
    declaration = "declare i32 @llvm.nvvm.read.ptx.sreg.ctaid.x()"
    assert store in nvvm_ir and declaration in nvvm_ir
    libdevice_ir = nvvm_ir.replace(
        store,
        """  %arg = sitofp i32 %call to double
  %result = call double @__nv_sin(double %arg)
  %converted = fptosi double %result to i32
  store i32 %converted, i32* %data, align 4""",
    ).replace(
        declaration,
        """declare double @__nv_sin(double)

declare i32 @llvm.nvvm.read.ptx.sreg.ctaid.x()""",
    )
    from cuda.pathfinder import BitcodeLibNotFoundError

    program = Program(libdevice_ir, "nvvm", ProgramOptions(use_libdevice=True, arch="sm_80"))
    try:
        try:
            obj = program.compile("ptx")
        except BitcodeLibNotFoundError:
            pytest.skip("libdevice bitcode not found")
        assert isinstance(obj, ObjectCode)
        assert obj.code
        # Without libdevice, NVVM leaves an external __nv_sin declaration in PTX.
        assert not any(b".extern" in line and b"__nv_sin" in line for line in obj.code.splitlines())
    finally:
        program.close()
