# SPDX-FileCopyrightText: Copyright (c) 2024-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

from cuda.core._memory._managed_memory_ops import advise, discard, discard_prefetch, prefetch
from cuda.core._memory._managed_memory_options import (
    AdviseOptions,
    DiscardOptions,
    DiscardPrefetchOptions,
    PrefetchOptions,
)
from cuda.core._memoryview import StridedMemoryView, args_viewable_as_strided_memory

__all__ = [
    "AdviseOptions",
    "DiscardOptions",
    "DiscardPrefetchOptions",
    "PrefetchOptions",
    "StridedMemoryView",
    "advise",
    "args_viewable_as_strided_memory",
    "discard",
    "discard_prefetch",
    "prefetch",
]
