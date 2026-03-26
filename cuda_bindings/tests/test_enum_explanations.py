# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: LicenseRef-NVIDIA-SOFTWARE-LICENSE

import importlib
import importlib.metadata

import pytest

from cuda.bindings import driver, runtime

_EXPLANATION_MODULES = [
    ("driver_cu_result_explanations", "DRIVER_CU_RESULT_EXPLANATIONS", driver.CUresult),
    ("runtime_cuda_error_explanations", "RUNTIME_CUDA_ERROR_EXPLANATIONS", runtime.cudaError_t),
]


def _get_binding_version():
    try:
        major_minor = importlib.metadata.version("cuda-bindings").split(".")[:2]
    except importlib.metadata.PackageNotFoundError:
        major_minor = importlib.metadata.version("cuda-python").split(".")[:2]
    return tuple(int(v) for v in major_minor)


@pytest.mark.parametrize("module_name,dict_name,enum_type", _EXPLANATION_MODULES)
def test_explanations_health(module_name, dict_name, enum_type):
    mod = importlib.import_module(f"cuda.bindings._utils.{module_name}")
    expl_dict = getattr(mod, dict_name)

    known_codes = set()
    for error in enum_type:
        code = int(error)
        assert code in expl_dict
        known_codes.add(code)

    if _get_binding_version() >= (13, 0):
        extra_expl = sorted(set(expl_dict.keys()) - known_codes)
        assert not extra_expl
