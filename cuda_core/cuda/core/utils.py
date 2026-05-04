# SPDX-FileCopyrightText: Copyright (c) 2024-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

from cuda.core._memory._managed_memory_ops import (
    discard_batch,
    discard_prefetch_batch,
    prefetch_batch,
)
from cuda.core._memoryview import StridedMemoryView, args_viewable_as_strided_memory

__all__ = [
    "StridedMemoryView",
    "args_viewable_as_strided_memory",
    "discard_batch",
    "discard_prefetch_batch",
    "prefetch_batch",
]
