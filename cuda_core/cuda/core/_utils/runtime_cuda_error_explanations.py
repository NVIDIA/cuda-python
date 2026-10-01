# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from cuda.bindings import runtime
from cuda.core._utils.enum_explanations_helpers import DocstringBackedExplanations

RUNTIME_CUDA_ERROR_EXPLANATIONS = DocstringBackedExplanations(runtime.cudaError_t)
