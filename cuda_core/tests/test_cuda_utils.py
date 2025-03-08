# Copyright (c) 2025, NVIDIA CORPORATION & AFFILIATES. ALL RIGHTS RESERVED.
#
# SPDX-License-Identifier: LicenseRef-NVIDIA-SOFTWARE-LICENSE

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
