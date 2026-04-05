# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# SPDX-License-Identifier: LicenseRef-NVIDIA-SOFTWARE-LICENSE

import dataclasses

import pytest

from cuda.bindings import driver, runtime
from cuda.core._utils import cuda_utils
from cuda.core._utils.clear_error_support import assert_type_str_or_bytes_like, raise_code_path_meant_to_be_unreachable


def _skip_if_bindings_pre_enum_docstrings():
    from cuda.core._utils.version import binding_version

    if binding_version() < (13, 2, 0):
        pytest.skip("cuda-bindings < 13.2.0 may not expose enum __doc__ strings")


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


def test_driver_error_enum_has_non_empty_docstring():
    _skip_if_bindings_pre_enum_docstrings()

    doc = driver.CUresult.CUDA_ERROR_INVALID_VALUE.__doc__
    assert doc is not None
    assert doc.strip() != ""


def test_runtime_error_enum_has_non_empty_docstring():
    _skip_if_bindings_pre_enum_docstrings()

    doc = runtime.cudaError_t.cudaErrorInvalidValue.__doc__
    assert doc is not None
    assert doc.strip() != ""


def test_check_driver_error_attaches_explanation():
    error = driver.CUresult.CUDA_ERROR_INVALID_VALUE
    name_err, name = driver.cuGetErrorName(error)
    assert name_err == driver.CUresult.CUDA_SUCCESS
    desc_err, desc = driver.cuGetErrorString(error)
    assert desc_err == driver.CUresult.CUDA_SUCCESS
    expl = cuda_utils.DRIVER_CU_RESULT_EXPLANATIONS.get(int(error))
    assert expl is not None
    assert expl != desc.decode()

    with pytest.raises(cuda_utils.CUDAError) as e:
        cuda_utils._check_driver_error(error)

    assert str(e.value) == f"{name.decode()}: {expl}"
    assert str(e.value) != f"{name.decode()}: {desc.decode()}"


def test_check_runtime_error_attaches_explanation():
    error = runtime.cudaError_t.cudaErrorInvalidValue
    name_err, name = runtime.cudaGetErrorName(error)
    assert name_err == runtime.cudaError_t.cudaSuccess
    desc_err, desc = runtime.cudaGetErrorString(error)
    assert desc_err == runtime.cudaError_t.cudaSuccess
    expl = cuda_utils.RUNTIME_CUDA_ERROR_EXPLANATIONS.get(int(error))
    assert expl is not None
    assert expl != desc.decode()

    with pytest.raises(cuda_utils.CUDAError) as e:
        cuda_utils._check_runtime_error(error)

    assert str(e.value) == f"{name.decode()}: {expl}"
    assert str(e.value) != f"{name.decode()}: {desc.decode()}"


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
