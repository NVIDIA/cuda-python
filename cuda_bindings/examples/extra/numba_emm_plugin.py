# Copyright 2021-2025 NVIDIA Corporation.  All rights reserved.
# SPDX-License-Identifier: LicenseRef-NVIDIA-SOFTWARE-LICENSE

"""Numba EMM Plugin using the CUDA Python Driver API.

This example provides an External Memory Management (EMM) Plugin for Numba (see
https://numba.readthedocs.io/en/stable/cuda/external-memory.html) that uses the
NVIDIA CUDA Python Driver API for all on-device allocations and frees. For
other operations interacting with the driver, Numba uses its internal ctypes
wrapper. This serves as an example of interoperability between the NVIDIA CUDA
Python Driver API, and other implementations of driver API wrappers (in this
case Numba's ctypes wrapper), and demonstrates an on-ramp to using the NVIDIA
CUDA Python Driver API wrapper by showing that it can co-exist with other
wrappers - it is not necessary to replace all wrappers in all libraries to
start using the NVIDIA wrapper.

The current version of Numba passes all tests using this plugin (with a small
patch to recognize CUDA 11.3 as a supported version). The Numba test suite can
be run with the plugin by executing:

    NUMBA_CUDA_MEMORY_MANAGER=numba_emm_plugin \\
        python -m numba.runtests numba.cuda.tests -vf -m

when the directory containing this example is on the PYTHONPATH. When tests are
run, the test summary is expected to be close to:

    Ran 1121 tests in 159.572s

    OK (skipped=17, expected failures=1)

The number of tests may vary with changes between commits in Numba, but the
main result is that there are no unexpected failures.

This example can also be run standalone with:

    python numba_emm_plugin.py

in which case it sets up Numba to use the included EMM plugin, then creates and
destroys a device array. When run standalone, the output may look like:

    Free before creating device array: 50781159424
    Free after creating device array: 50779062272
    Free after freeing device array: 50781159424

The initial value may vary, but the expectation is that 2097152 bytes (2MB)
should be taken up by the device array creation, and the original value should
be restored after freeing it.
"""

from ctypes import c_size_t

from cuda.bindings import driver as cuda
from cuda.bindings import driver as cuda_driver
from numba.cuda import (
    GetIpcHandleMixin,
    HostOnlyCUDAMemoryManager,
    MemoryInfo,
    MemoryPointer,
)

# Python functions for allocation, deallocation, and memory info via the NVIDIA
# CUDA Python Driver API


def driver_alloc(size):
    """
    Allocate `size` bytes of device memory and return a device pointer to the
    allocated memory.
    """
    err, ptr = cuda_driver.cuMemAlloc(size)
    if err != cuda_driver.CUresult.CUDA_SUCCESS:
        raise RuntimeError(f"Unexpected error code {err} from cuMemAlloc")
    return ptr


def driver_free(ptr):
    """
    Free device memory pointed to by `ptr`.
    """
    (err,) = cuda_driver.cuMemFree(ptr)
    if err != cuda_driver.CUresult.CUDA_SUCCESS:
        raise RuntimeError(f"Unexpected error code {err} from cuMemFree")


def driver_memory_info():
    """
    Return the free and total amount of device memory in bytes as a tuple.
    """
    err, free, total = cuda_driver.cuMemGetInfo()
    if err != cuda_driver.CUresult.CUDA_SUCCESS:
        raise RuntimeError(f"Unexpected error code {err} from cuMemGetInfo")
    return free, total


# EMM Plugin implementation. For documentation of the methods implemented here,
# see:
#
#    https://numba.readthedocs.io/en/stable/cuda/external-memory.html#numba.cuda.BaseCUDAMemoryManager


class DriverEMMPlugin(GetIpcHandleMixin, HostOnlyCUDAMemoryManager):
    def memalloc(self, size):
        ptr = driver_alloc(size)
        ctx = self.context
        finalizer = make_finalizer(ptr)
        # We wrap the pointer value in a c_size_t because Numba expects ctypes
        # objects
        wrapped_ptr = c_size_t(int(ptr))
        return MemoryPointer(ctx, wrapped_ptr, size, finalizer=finalizer)

    def initialize(self):
        # No setup required to use the EMM Plugin in a given context
        pass

    def get_memory_info(self):
        free, total = driver_memory_info()
        return MemoryInfo(free=free, total=total)

    @property
    def interface_version(self):
        return 1


def make_finalizer(ptr):
    def finalizer():
        driver_free(ptr)

    return finalizer


# If NUMBA_CUDA_MEMORY_MANAGER is set to this module (e.g.
# `NUMBA_CUDA_MEMORY_MANAGER=numba_emm_plugin`), then Numba will look at the
# _numba_memory_manager global to determine what class to use for memory
# management.

_numba_memory_manager = DriverEMMPlugin


def main():
    """
    A simple test / demonstration setting the memory manager and
    allocating/deleting an array.
    """

    cuda.set_memory_manager(DriverEMMPlugin)
    ctx = cuda.current_context()
    print(f"Free before creating device array: {ctx.get_memory_info().free}")
    x = cuda.device_array(1000)
    print(f"Free after creating device array: {ctx.get_memory_info().free}")
    del x
    print(f"Free after freeing device array: {ctx.get_memory_info().free}")


if __name__ == "__main__":
    import argparse

    formatter = argparse.RawDescriptionHelpFormatter
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=formatter)
    parser.parse_args()
    main()
