# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

from cuda.bindings import driver as cuda

from runner.runtime import alloc_persistent, assert_drv


# Allocate memory used by the tests
PTR = alloc_persistent(1 << 18)
ATTRIBUTE = cuda.CUpointer_attribute.CU_POINTER_ATTRIBUTE_MEMORY_TYPE


def bench_pointer_get_attribute() -> None:
    err, _ = cuda.cuPointerGetAttribute(ATTRIBUTE, PTR)
    assert_drv(err)
