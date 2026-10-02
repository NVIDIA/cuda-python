# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from cuda.bindings import driver
from cuda.core._utils.enum_explanations_helpers import DocstringBackedExplanations

DRIVER_CU_RESULT_EXPLANATIONS = DocstringBackedExplanations(driver.CUresult)
