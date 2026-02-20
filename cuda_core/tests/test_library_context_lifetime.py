# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: LicenseRef-NVIDIA-SOFTWARE-LICENSE

"""Tests for library/kernel lifetime vs context lifetime.

These tests exercise scenarios where a CUDA library (CUlibrary) or kernel
(CUkernel) outlives the context in which it was created. This can happen
when a non-primary context is explicitly destroyed while libraries or
kernels loaded within it are still alive.

Currently, LibraryBox in resource_handles.cpp does NOT store a
ContextHandle, so nothing prevents the context from being destroyed
before the library. The library's RAII deleter then calls cuLibraryUnload
on a library whose owning context is gone, which may segfault on some
driver versions.
"""

import gc

import pytest
from cuda.bindings import driver
from cuda.core import Program
from cuda.core._utils.cuda_utils import handle_return

KERNEL_SOURCE = 'extern "C" __global__ void test_kernel() {}'


def _compile_and_get_kernel():
    """Compile a trivial kernel and return (ObjectCode, Kernel)."""
    prog = Program(KERNEL_SOURCE, "c++")
    obj = prog.compile("cubin")
    kernel = obj.get_kernel("test_kernel")
    return obj, kernel


def _create_nonprimary_context(dev=0):
    """Create a non-primary context, handling CUDA 12.x vs 13.x API differences."""
    try:
        return handle_return(driver.cuCtxCreate(None, 0, dev))
    except TypeError:
        return handle_return(driver.cuCtxCreate(0, dev))


@pytest.fixture(autouse=True)
def _restore_primary_context():
    """Re-establish a valid primary context after each test."""
    yield
    ctx = handle_return(driver.cuDevicePrimaryCtxRetain(0))
    handle_return(driver.cuCtxSetCurrent(ctx))


class TestNonPrimaryContextDestroy:
    """Library/kernel destroyed after non-primary context is destroyed."""

    def test_objectcode_outlives_nonprimary_context(self):
        ctx = _create_nonprimary_context()

        obj, kernel = _compile_and_get_kernel()
        del kernel

        handle_return(driver.cuCtxDestroy(ctx))

        del obj
        gc.collect()

    def test_kernel_outlives_nonprimary_context(self):
        ctx = _create_nonprimary_context()

        obj, kernel = _compile_and_get_kernel()
        del obj

        handle_return(driver.cuCtxDestroy(ctx))

        del kernel
        gc.collect()

    def test_kernel_outlives_objectcode_and_nonprimary_context(self):
        ctx = _create_nonprimary_context()

        obj, kernel = _compile_and_get_kernel()
        del obj
        gc.collect()

        handle_return(driver.cuCtxDestroy(ctx))

        del kernel
        gc.collect()
