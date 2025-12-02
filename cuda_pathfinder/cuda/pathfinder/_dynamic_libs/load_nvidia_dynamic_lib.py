# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import functools
import struct
import sys

from cuda.pathfinder._dynamic_libs.find_nvidia_dynamic_lib import _FindNvidiaDynamicLib
from cuda.pathfinder._dynamic_libs.load_dl_common import LoadedDL, load_dependencies
from cuda.pathfinder._utils.platform_aware import IS_WINDOWS

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
    finder = _FindNvidiaDynamicLib(libname)
    abs_path = finder.try_site_packages()
    if abs_path is not None:
        found_via = "site-packages"
    else:
        abs_path = finder.try_with_conda_prefix()
        if abs_path is not None:
            found_via = "conda"

    # If the library was already loaded by someone else, reproduce any OS-specific
    # side-effects we would have applied on a direct absolute-path load (e.g.,
    # AddDllDirectory on Windows for libs that require it).
    loaded = check_if_already_loaded_from_elsewhere(libname, abs_path is not None)

    # Load dependencies regardless of who loaded the primary lib first.
    # Doing this *after* the side-effect ensures dependencies resolve consistently
    # relative to the actually loaded location.
    load_dependencies(libname, load_nvidia_dynamic_lib)

    if loaded is not None:
        return loaded

    if abs_path is None:
        loaded = load_with_system_search(libname)
        if loaded is not None:
            return loaded
        abs_path = finder.try_with_cuda_home()
        if abs_path is None:
            finder.raise_not_found_error()
        else:
            found_via = "CUDA_HOME"

    return load_with_abs_path(libname, abs_path, found_via)


@functools.cache
def load_nvidia_dynamic_lib(libname: str) -> LoadedDL:
    """Load an NVIDIA dynamic library by name.

    Args:
        libname (str): The short name of the library to load (e.g., ``"cudart"``,
            ``"nvvm"``, etc.).

    Returns:
        LoadedDL: Object containing the OS library handle and absolute path.

        **Important:**

        **Never close the returned handle.** Do **not** call ``dlclose`` (Linux) or
        ``FreeLibrary`` (Windows) on the ``LoadedDL._handle_uint``.

        **Why:** the return value is cached (``functools.cache``) and shared across the
        process. Closing the handle can unload the module while other code still uses
        it, leading to crashes or subtle failures.

        This applies to Linux and Windows. For context, see issue #1011:
        https://github.com/NVIDIA/cuda-python/issues/1011

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

        2. **Conda environment**

           - Conda installations are discovered via ``CONDA_PREFIX``, which is
             defined automatically in activated conda environments (see
             https://docs.conda.io/projects/conda-build/en/stable/user-guide/environment-variables.html).

        3. **OS default mechanisms**

           - Fall back to the native loader:

             - Linux: ``dlopen()``

             - Windows: ``LoadLibraryW()``

           - CUDA Toolkit (CTK) system installs with system config updates are often
             discovered via:

             - Linux: ``/etc/ld.so.conf.d/*cuda*.conf``

             - Windows: ``C:\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDA\\vX.Y\\bin``
               on the system ``PATH``.

        4. **Environment variables**

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
