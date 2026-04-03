# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

import time

from runner.runtime import alloc_persistent

from cuda.bindings import driver as cuda

# Allocate memory used by the tests
PTR = alloc_persistent(1 << 18)
ATTRIBUTE = cuda.CUpointer_attribute.CU_POINTER_ATTRIBUTE_MEMORY_TYPE


def bench_pointer_get_attribute(loops: int) -> float:
    # Local references to avoid global lookups in the hot loop
    _cuPointerGetAttribute = cuda.cuPointerGetAttribute
    _attr = ATTRIBUTE
    _ptr = PTR

    t0 = time.perf_counter()
    for _ in range(loops):
        _cuPointerGetAttribute(_attr, _ptr)
    return time.perf_counter() - t0
