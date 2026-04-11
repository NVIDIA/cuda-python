# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: LicenseRef-NVIDIA-SOFTWARE-LICENSE

from cuda.bindings import driver
from cuda.core._utils.enum_explanations_helpers import get_best_available_explanations


def _load_fallback_explanations():
    from cuda.core._utils.driver_cu_result_explanations_frozen import _FALLBACK_EXPLANATIONS

    return _FALLBACK_EXPLANATIONS


DRIVER_CU_RESULT_EXPLANATIONS = get_best_available_explanations(driver.CUresult, _load_fallback_explanations)
