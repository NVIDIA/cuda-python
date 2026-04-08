# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: LicenseRef-NVIDIA-SOFTWARE-LICENSE

from cuda.bindings import runtime
from cuda.core._utils.enum_explanations_helpers import get_best_available_explanations


def _load_fallback_explanations():
    from cuda.core._utils.runtime_cuda_error_explanations_frozen import _FALLBACK_EXPLANATIONS

    return _FALLBACK_EXPLANATIONS


RUNTIME_CUDA_ERROR_EXPLANATIONS = get_best_available_explanations(runtime.cudaError_t, _load_fallback_explanations)
