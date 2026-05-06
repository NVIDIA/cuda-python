# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: LicenseRef-NVIDIA-SOFTWARE-LICENSE

import pytest


@pytest.fixture(scope="module", autouse=True)
def ctx():
    """Override the parent conftest's ``ctx`` fixture which creates a CUDA
    context.  cuDLA tests do not require a CUDA context, so this no-op
    prevents ``cuInit`` / ``cuCtxCreate`` from running (and failing on
    machines without a CUDA-capable GPU)."""
    return None
