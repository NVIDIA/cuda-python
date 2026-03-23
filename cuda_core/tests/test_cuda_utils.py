# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# SPDX-License-Identifier: LicenseRef-NVIDIA-SOFTWARE-LICENSE

import dataclasses
import importlib

import pytest

from cuda.bindings import driver, runtime
from cuda.core._utils import cuda_utils
from cuda.core._utils.clear_error_support import assert_type_str_or_bytes_like, raise_code_path_meant_to_be_unreachable

_EXPLANATION_MODULES = [
    ("driver_cu_result_explanations", "DRIVER_CU_RESULT_EXPLANATIONS"),
    ("runtime_cuda_error_explanations", "RUNTIME_CUDA_ERROR_EXPLANATIONS"),
]


@pytest.mark.parametrize("module_name,public_name", _EXPLANATION_MODULES)
def test_explanations_smoke(module_name, public_name):
    expl = getattr(cuda_utils, public_name)
    for code in (0, 1, 2):
        assert code in expl
        assert isinstance(expl[code], str)


@pytest.mark.parametrize("module_name,public_name", _EXPLANATION_MODULES)
def test_explanations_ctk_version(module_name, public_name):
    del public_name  # unused
    core_mod = importlib.import_module(f"cuda.core._utils.{module_name}")
    try:
        bindings_mod = importlib.import_module(f"cuda.bindings._utils.{module_name}")
    except ModuleNotFoundError:
        pytest.skip("cuda.bindings._utils not available")
    bindings_path = f"cuda_bindings/cuda/bindings/_utils/{module_name}.py"
    core_path = f"cuda_core/cuda/core/_utils/{module_name}.py"
    if core_mod._CTK_MAJOR_MINOR_PATCH < bindings_mod._CTK_MAJOR_MINOR_PATCH:
        raise RuntimeError(
            f"cuda_core copy is older ({core_mod._CTK_MAJOR_MINOR_PATCH})"
            f" than cuda_bindings ({bindings_mod._CTK_MAJOR_MINOR_PATCH})."
            f" Please copy the _EXPLANATIONS dict from {bindings_path} to {core_path}"
        )
    if (
        core_mod._CTK_MAJOR_MINOR_PATCH == bindings_mod._CTK_MAJOR_MINOR_PATCH
        and core_mod._FALLBACK_EXPLANATIONS != bindings_mod._EXPLANATIONS
    ):
        raise RuntimeError(
            f"The cuda_core copy of the cuda_bindings _EXPLANATIONS dict is out of sync"
            f" (both at CTK {core_mod._CTK_MAJOR_MINOR_PATCH})."
            f" Please copy the _EXPLANATIONS dict from {bindings_path} to {core_path}"
        )


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
