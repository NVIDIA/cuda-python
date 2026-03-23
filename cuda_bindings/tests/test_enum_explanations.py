# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: LicenseRef-NVIDIA-SOFTWARE-LICENSE

import importlib.metadata

from cuda.bindings import driver, runtime
from cuda.bindings._utils.driver_cu_result_explanations import DRIVER_CU_RESULT_EXPLANATIONS
from cuda.bindings._utils.runtime_cuda_error_explanations import RUNTIME_CUDA_ERROR_EXPLANATIONS


def _get_binding_version():
    try:
        major_minor = importlib.metadata.version("cuda-bindings").split(".")[:2]
    except importlib.metadata.PackageNotFoundError:
        major_minor = importlib.metadata.version("cuda-python").split(".")[:2]
    return tuple(int(v) for v in major_minor)


def test_driver_cu_result_explanations_health():
    expl_dict = DRIVER_CU_RESULT_EXPLANATIONS

    known_codes = set()
    for error in driver.CUresult:
        code = int(error)
        assert code in expl_dict
        known_codes.add(code)

    if _get_binding_version() >= (13, 0):
        extra_expl = sorted(set(expl_dict.keys()) - known_codes)
        assert not extra_expl


def test_runtime_cuda_error_explanations_health():
    expl_dict = RUNTIME_CUDA_ERROR_EXPLANATIONS

    known_codes = set()
    for error in runtime.cudaError_t:
        code = int(error)
        assert code in expl_dict
        known_codes.add(code)

    if _get_binding_version() >= (13, 0):
        extra_expl = sorted(set(expl_dict.keys()) - known_codes)
        assert not extra_expl
