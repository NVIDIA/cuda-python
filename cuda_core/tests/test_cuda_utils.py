# Copyright (c) 2025, NVIDIA CORPORATION & AFFILIATES. ALL RIGHTS RESERVED.
#
# SPDX-License-Identifier: LicenseRef-NVIDIA-SOFTWARE-LICENSE

import pytest
from conftest import skipif_testing_with_compute_sanitizer

from cuda.bindings import driver, runtime
from cuda.core.experimental._utils import cuda_utils


def test_driver_cu_result_explanations_health():
    expl_dict = cuda_utils.DRIVER_CU_RESULT_EXPLANATIONS

    # Ensure all CUresult enums are in expl_dict
    known_codes = set()
    for error in driver.CUresult:
        code = int(error)
        assert code in expl_dict
        known_codes.add(code)

    if cuda_utils.get_binding_version() >= (12, 0):
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

    if cuda_utils.get_binding_version() >= (12, 0):
        # Ensure expl_dict has no codes not known as a cudaError_t enum
        extra_expl = sorted(set(expl_dict.keys()) - known_codes)
        assert not extra_expl


# this test causes an API error when the driver is too old to know about all of the error codes
@skipif_testing_with_compute_sanitizer
def test_check_driver_error():
    num_unexpected = 0
    for error in driver.CUresult:
        if error == driver.CUresult.CUDA_SUCCESS:
            assert cuda_utils._check_driver_error(error) is None
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
            assert cuda_utils._check_runtime_error(error) is None
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
