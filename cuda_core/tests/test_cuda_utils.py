# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# SPDX-License-Identifier: LicenseRef-NVIDIA-SOFTWARE-LICENSE

import dataclasses

import pytest

from cuda.bindings import driver, runtime
from cuda.core._utils import cuda_utils
from cuda.core._utils.clear_error_support import assert_type_str_or_bytes_like, raise_code_path_meant_to_be_unreachable


def test_driver_cu_result_explanations_health():
    expl_dict = cuda_utils.DRIVER_CU_RESULT_EXPLANATIONS

    # Ensure all CUresult enums are in expl_dict
    known_codes = set()
    for error in driver.CUresult:
        code = int(error)
        assert code in expl_dict
        known_codes.add(code)

    if cuda_utils.get_binding_version() >= (13, 0):
        # Ensure expl_dict has no codes not known as a CUresult enum
        extra_expl = sorted(set(expl_dict.keys()) - known_codes)
        assert not extra_expl


def test_runtime_cuda_error_explanations_health():
    expl_dict = cuda_utils.RUNTIME_CUDA_ERROR_EXPLANATIONS

    # Ensure all cudaError_t enums are in expl_dict
    known_codes = set()
    for error in runtime.cudaError_t:
        code = int(error)
        assert code in expl_dict
        known_codes.add(code)

    if cuda_utils.get_binding_version() >= (13, 0):
        # Ensure expl_dict has no codes not known as a cudaError_t enum
        extra_expl = sorted(set(expl_dict.keys()) - known_codes)
        assert not extra_expl


def test_check_driver_error():
    num_unexpected = 0
    for error in driver.CUresult:
        if error == driver.CUresult.CUDA_SUCCESS:
            assert cuda_utils._check_driver_error(error) == 0
        else:
            with pytest.raises(cuda_utils.CUDAError) as e:
                cuda_utils._check_driver_error(error)
            msg = str(e)
            if "UNEXPECTED ERROR CODE" in msg:
                num_unexpected += 1
            else:
                # Example repr(error): <CUresult.CUDA_ERROR_UNKNOWN: 999>
                enum_name = repr(error).split(".", 1)[1].split(":", 1)[0]
                assert enum_name in msg
    # Smoke test: We don't want most to be unexpected.
    assert num_unexpected < len(driver.CUresult) * 0.5


def test_check_runtime_error():
    num_unexpected = 0
    for error in runtime.cudaError_t:
        if error == runtime.cudaError_t.cudaSuccess:
            assert cuda_utils._check_runtime_error(error) == 0
        else:
            with pytest.raises(cuda_utils.CUDAError) as e:
                cuda_utils._check_runtime_error(error)
            msg = str(e)
            if "UNEXPECTED ERROR CODE" in msg:
                num_unexpected += 1
            else:
                # Example repr(error): <cudaError_t.cudaErrorUnknown: 999>
                enum_name = repr(error).split(".", 1)[1].split(":", 1)[0]
                assert enum_name in msg
    # Smoke test: We don't want most to be unexpected.
    assert num_unexpected < len(driver.CUresult) * 0.5


def test_precondition():
    def checker(*args, what=""):
        if args[0] < 0:
            raise ValueError(f"{what}: negative")

    @cuda_utils.precondition(checker, what="value check")
    def my_func(x):
        return x * 2

    assert my_func(5) == 10
    with pytest.raises(ValueError, match="negative"):
        my_func(-1)


@dataclasses.dataclass
class _DummyOptions:
    x: int = 1
    y: str = "hello"


def test_check_nvrtc_error_without_handle():
    from cuda.bindings import nvrtc

    assert cuda_utils._check_nvrtc_error(nvrtc.nvrtcResult.NVRTC_SUCCESS) == 0
    with pytest.raises(cuda_utils.NVRTCError):
        cuda_utils._check_nvrtc_error(nvrtc.nvrtcResult.NVRTC_ERROR_COMPILATION)


def test_check_nvrtc_error_with_handle(init_cuda):
    from cuda.bindings import nvrtc

    err, prog = nvrtc.nvrtcCreateProgram(b"invalid code!@#$", b"test.cu", 0, [], [])
    assert err == nvrtc.nvrtcResult.NVRTC_SUCCESS
    try:
        (compile_result,) = nvrtc.nvrtcCompileProgram(prog, 0, [])
        assert compile_result != nvrtc.nvrtcResult.NVRTC_SUCCESS
        with pytest.raises(cuda_utils.NVRTCError, match="compilation log"):
            cuda_utils._check_nvrtc_error(compile_result, handle=prog)
    finally:
        nvrtc.nvrtcDestroyProgram(prog)


def test_check_or_create_options_invalid_type():
    with pytest.raises(TypeError, match="must be provided as an object"):
        cuda_utils.check_or_create_options(_DummyOptions, 12345, options_description="test options")


def test_assert_type_str_or_bytes_like_rejects_non_str_bytes():
    with pytest.raises(TypeError, match="Expected type str or bytes or bytearray"):
        assert_type_str_or_bytes_like(12345)


def test_raise_code_path_meant_to_be_unreachable():
    with pytest.raises(RuntimeError, match="This code path is meant to be unreachable"):
        raise_code_path_meant_to_be_unreachable()
