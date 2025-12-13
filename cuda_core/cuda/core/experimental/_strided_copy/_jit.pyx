# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

from libc.stdint cimport intptr_t

import os
import threading

from cuda.core.experimental._device import Device
from cuda.core.experimental._program import Program, ProgramOptions


_tls = threading.local()

# In multithreaded environment we share the compiled and loaded modules between threads.
# Each thread has its own cache mapping arch -> kernel_code_str -> Kernel ptr,
# on a cache miss, we first take a look into the shared cache guarded with _kernel_lock
# and eventually compile if needed.
_kernel_lock = threading.Lock()
_kernel_cache = {}  # arch -> kernel_code_str -> Kernel


cpdef str get_strided_copy_include_dir(object logger):
    """
    Finds and caches the absolute path for the strided copy includes.
    """
    # TODO(ktokarski) Once Program API supports passing includes as strings and names,
    # read all the headers once and cache them.
    cdef str strided_copy_include_dir = getattr(_tls, "strided_copy_include_dir", None)
    if strided_copy_include_dir is not None:
        return strided_copy_include_dir
    cdef str current_dir = os.path.dirname(os.path.abspath(__file__))
    cdef str copy_kernel_dir = os.path.normpath(os.path.join(current_dir, os.pardir, "include", "strided_copy"))
    _tls.strided_copy_include_dir = copy_kernel_dir
    if logger is not None:
        logger.debug(f"Strided copy include dir: {copy_kernel_dir}")
    return copy_kernel_dir


cdef inline str get_device_arch(int device_id):
    # device_id -> arch
    cdef dict device_ccs = getattr(_tls, "device_ccs", None)
    if device_ccs is None:
        device_ccs = {}
        _tls.device_ccs = device_ccs
    cdef str arch = device_ccs.get(device_id)
    if arch is None:
        arch = f"sm_{Device(device_id).arch}"
        device_ccs[device_id] = arch
    return arch


cdef compile_load_kernel(str kernel_code, str arch, object logger):
    cdef str include_dir = get_strided_copy_include_dir(logger)
    cdef options = ProgramOptions(arch=arch, include_path=include_dir)
    cdef program = Program(kernel_code, code_type="c++", options=options)
    cdef object_code = program.compile("cubin")
    cdef kernel = object_code.get_kernel("execute")
    return kernel


cdef inline intptr_t _get_or_compile_kernel(str kernel_code, str arch, object logger) except? 0:
    cdef dict cc_cache = _kernel_cache.get(arch)
    if cc_cache is None:
        cc_cache = {}
        _kernel_cache[arch] = cc_cache

    cdef kernel_obj = cc_cache.get(kernel_code)
    if kernel_obj is None:
        kernel_obj = compile_load_kernel(kernel_code, arch, logger)
        cc_cache[kernel_code] = kernel_obj
        if logger is not None:
            logger.debug(f"Stored kernel ({kernel_obj}) (arch={arch}) in global cache.\n{kernel_code}")
    elif logger is not None:
        logger.debug(f"Loaded kernel ({kernel_obj}) (arch={arch}) from global cache.\n{kernel_code}")
    return int(kernel_obj._handle)


cdef inline intptr_t get_or_compile_kernel(str kernel_code, str arch, object logger) except? 0:
    with _kernel_lock:
        return _get_or_compile_kernel(kernel_code, arch, logger)


cdef intptr_t get_kernel(str kernel_code, int device_id, object logger) except? 0:
    """
    Returns a pointer to the kernel function for a given kernel code and device id.

    In multithreaded environment, each thread has its own cache with pointers to the loaded
    modules, if the cache is not populated, the shared cache guarded with _kernel_lock is used.
    """
    cdef str arch = get_device_arch(device_id)
    cdef dict local_kernel_cache = getattr(_tls, "local_kernel_cache", None)
    if local_kernel_cache is None:
        local_kernel_cache = {}
        _tls.local_kernel_cache = local_kernel_cache
    cdef dict local_cc_cache = local_kernel_cache.get(arch)
    if local_cc_cache is None:
        local_cc_cache = {}
        local_kernel_cache[arch] = local_cc_cache

    cdef kernel_ptr = local_cc_cache.get(kernel_code)
    if kernel_ptr is None:
        kernel_ptr = get_or_compile_kernel(kernel_code, arch, logger)
        local_cc_cache[kernel_code] = kernel_ptr
    elif logger is not None:
        logger.debug(f"Loaded kernel ({kernel_ptr}) for device {device_id=} ({arch=}) from thread local cache.\n{kernel_code}")
    return kernel_ptr
