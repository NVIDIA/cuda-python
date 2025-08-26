# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import functools
import struct
import sys

from cuda.pathfinder._dynamic_libs.find_nvidia_dynamic_lib import _FindNvidiaDynamicLib
from cuda.pathfinder._dynamic_libs.load_dl_common import LoadedDL, load_dependencies
from cuda.pathfinder._dynamic_libs.supported_nvidia_libs import IS_WINDOWS

if IS_WINDOWS:
    from cuda.pathfinder._dynamic_libs.load_dl_windows import (
        check_if_already_loaded_from_elsewhere,
        load_with_abs_path,
        load_with_system_search,
    )
else:
    from cuda.pathfinder._dynamic_libs.load_dl_linux import (
        check_if_already_loaded_from_elsewhere,
        load_with_abs_path,
        load_with_system_search,
    )


def _load_lib_no_cache(libname: str) -> LoadedDL:
    found = _FindNvidiaDynamicLib(libname)
    have_abs_path = found.abs_path is not None

    # If the library was already loaded by someone else, reproduce any OS-specific
    # side-effects we would have applied on a direct absolute-path load (e.g.,
    # AddDllDirectory on Windows for libs that require it).
    loaded = check_if_already_loaded_from_elsewhere(libname, have_abs_path)

    # Load dependencies regardless of who loaded the primary lib first.
    # Doing this *after* the side-effect ensures dependencies resolve consistently
    # relative to the actually loaded location.
    load_dependencies(libname, load_nvidia_dynamic_lib)

    if loaded is not None:
        return loaded

    if not have_abs_path:
        loaded = load_with_system_search(libname)
        if loaded is not None:
            return loaded
        found.retry_with_cuda_home_priority_last()
        found.raise_if_abs_path_is_None()

    assert found.abs_path is not None  # for mypy
    return load_with_abs_path(libname, found.abs_path)


@functools.cache
def load_nvidia_dynamic_lib(libname: str) -> LoadedDL:
    """Load an NVIDIA dynamic library by name.

    Args:
        libname (str): The short name of the library to load (e.g., ``"cudart"``,
            ``"nvvm"``, etc.).

    Returns:
        LoadedDL: Object containing the OS library handle and absolute path.

    Raises:
        DynamicLibNotFoundError: If the library cannot be found or loaded.
        RuntimeError: If Python is not 64-bit.

    Search order:
        0. **Already loaded in the current process**

           - If a matching library is already loaded by some other component,
             return its absolute path and handle and skip the rest of the search.

        1. **NVIDIA Python wheels**

           - Scan installed distributions (``site-packages``) to find libraries
             shipped in NVIDIA wheels.

        2. **OS default mechanisms / Conda environments**

           - Fall back to the native loader:

             - Linux: ``dlopen()``

             - Windows: ``LoadLibraryW()``

           - Conda installations are commonly discovered via:

             - Linux: ``$ORIGIN/../lib`` in the ``RPATH`` of the ``python`` binary
               (note: this can take precedence over ``LD_LIBRARY_PATH`` and
               ``/etc/ld.so.conf.d/``).

             - Windows: ``%CONDA_PREFIX%\\Library\\bin`` on the system ``PATH``.

           - CUDA Toolkit (CTK) system installs with system config updates are often
             discovered via:

             - Linux: ``/etc/ld.so.conf.d/*cuda*.conf``

             - Windows: ``C:\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDA\\vX.Y\\bin``
               on the system ``PATH``.

        3. **Environment variables**

           - If set, use ``CUDA_HOME`` or ``CUDA_PATH`` (in that order).

    Notes:
        The search is performed **per library**. There is currently no mechanism to
        guarantee that multiple libraries are all resolved from the same location.

    """
    pointer_size_bits = struct.calcsize("P") * 8
    if pointer_size_bits != 64:
        raise RuntimeError(
            f"cuda.pathfinder.load_nvidia_dynamic_lib() requires 64-bit Python."
            f" Currently running: {pointer_size_bits}-bit Python"
            f" {sys.version_info.major}.{sys.version_info.minor}"
        )
    return _load_lib_no_cache(libname)
