# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import functools
import json
import struct
import sys

from cuda.pathfinder._dynamic_libs.canary_probe_subprocess import probe_canary_abs_path_and_print_json
from cuda.pathfinder._dynamic_libs.find_nvidia_dynamic_lib import (
    _FindNvidiaDynamicLib,
    derive_ctk_root,
)
from cuda.pathfinder._dynamic_libs.load_dl_common import (
    DynamicLibNotAvailableError,
    DynamicLibNotFoundError,
    DynamicLibUnknownError,
    LoadedDL,
    load_dependencies,
)
from cuda.pathfinder._dynamic_libs.supported_nvidia_libs import (
    _CTK_ROOT_CANARY_ANCHOR_LIBNAMES,
    _CTK_ROOT_CANARY_DISCOVERABLE_LIBNAMES,
    SUPPORTED_LINUX_SONAMES,
    SUPPORTED_WINDOWS_DLLS,
)
from cuda.pathfinder._utils.platform_aware import IS_WINDOWS
from cuda.pathfinder._utils.spawned_process_runner import run_in_spawned_child_process

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

# All libnames recognized by load_nvidia_dynamic_lib, across all categories
# (CTK, third-party, driver).  Built from the platform-appropriate soname/DLL
# registry so that platform-specific libs (e.g. cufile on Linux) are included
# only where they apply.
_ALL_SUPPORTED_LIBNAMES: frozenset[str] = frozenset(
    (SUPPORTED_WINDOWS_DLLS if IS_WINDOWS else SUPPORTED_LINUX_SONAMES).keys()
)
_ALL_KNOWN_LIBNAMES: frozenset[str] = frozenset(SUPPORTED_LINUX_SONAMES) | frozenset(SUPPORTED_WINDOWS_DLLS)
_PLATFORM_NAME = "Windows" if IS_WINDOWS else "Linux"

# Driver libraries: shipped with the NVIDIA display driver, always on the
# system linker path.  These skip all CTK search steps (site-packages,
# conda, CUDA_HOME, canary) and go straight to system search.
_DRIVER_ONLY_LIBNAMES = frozenset(("cuda", "nvml"))


def _load_driver_lib_no_cache(libname: str) -> LoadedDL:
    """Load an NVIDIA driver library (system-search only).

    Driver libs (libcuda, libnvidia-ml) are part of the display driver, not
    the CUDA Toolkit.  They are always on the system linker path, so the
    full CTK search cascade (site-packages, conda, CUDA_HOME, canary) is
    unnecessary.
    """
    loaded = check_if_already_loaded_from_elsewhere(libname, False)
    if loaded is not None:
        return loaded
    loaded = load_with_system_search(libname)
    if loaded is not None:
        return loaded
    raise DynamicLibNotFoundError(
        f'"{libname}" is an NVIDIA driver library and can only be found via'
        f" system search. Ensure the NVIDIA display driver is installed."
    )


@functools.cache
def _resolve_system_loaded_abs_path_in_subprocess(libname: str) -> str | None:
    """Resolve a library's system-search absolute path in a child process.

    This runs in a spawned (not forked) child process. Spawning is important
    because it starts from a fresh interpreter state, so the child does not
    inherit already-loaded CUDA dynamic libraries from the parent process
    (especially the well-known canary probe library).

    That keeps any side-effects of loading the canary library scoped to the
    child process instead of polluting the current process, and ensures the
    canary probe is an independent system-search attempt.
    """
    result = run_in_spawned_child_process(
        probe_canary_abs_path_and_print_json,
        args=(libname,),
        timeout=10.0,
        rethrow=True,
    )

    # Read the final non-empty stdout line in case earlier lines are emitted.
    lines = [line for line in result.stdout.splitlines() if line.strip()]
    if not lines:
        raise RuntimeError(f"Canary probe child process produced no stdout payload for {libname!r}")
    try:
        payload = json.loads(lines[-1])
    except json.JSONDecodeError:
        raise RuntimeError(
            f"Canary probe child process emitted invalid JSON payload for {libname!r}: {lines[-1]!r}"
        ) from None
    if isinstance(payload, str):
        return payload
    if payload is None:
        return None
    raise RuntimeError(f"Canary probe child process emitted unexpected payload for {libname!r}: {payload!r}")


def _try_ctk_root_canary(finder: _FindNvidiaDynamicLib) -> str | None:
    """Derive the CTK root from a system-installed canary lib.

    For discoverable libs (currently nvvm) whose shared object doesn't reside
    on the standard linker path, we locate a well-known CTK lib that IS on
    the linker path via system search, derive the CTK installation root from
    its resolved path, and then look for the target lib relative to that root.

    The canary load is performed in a subprocess to avoid introducing loader
    state into the current process.
    """
    for canary_libname in _CTK_ROOT_CANARY_ANCHOR_LIBNAMES:
        canary_abs_path = _resolve_system_loaded_abs_path_in_subprocess(canary_libname)
        if canary_abs_path is None:
            continue
        ctk_root = derive_ctk_root(canary_abs_path)
        if ctk_root is None:
            continue
        abs_path: str | None = finder.try_via_ctk_root(ctk_root)
        if abs_path is not None:
            return abs_path
    return None


def _load_lib_no_cache(libname: str) -> LoadedDL:
    if libname in _DRIVER_ONLY_LIBNAMES:
        return _load_driver_lib_no_cache(libname)

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
        if abs_path is not None:
            found_via = "CUDA_HOME"
        else:
            if libname not in _CTK_ROOT_CANARY_DISCOVERABLE_LIBNAMES:
                finder.raise_not_found_error()

            # Canary probe (discoverable libs only): if the direct system
            # search and CUDA_HOME both failed (e.g. nvvm isn't on the linker
            # path and CUDA_HOME is unset), try to discover the CTK root by
            # loading a well-known CTK lib in a subprocess, then look for the
            # target lib relative to that root.
            abs_path = _try_ctk_root_canary(finder)
            if abs_path is not None:
                found_via = "system-ctk-root"
            else:
                finder.raise_not_found_error()

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
        DynamicLibUnknownError: If ``libname`` is not a recognized library name.
        DynamicLibNotAvailableError: If ``libname`` is recognized but not
            supported on this platform.
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

        5. **CTK root canary probe (discoverable libs only)**

           - For selected libraries whose shared object doesn't reside on the
             standard linker path (currently ``nvvm``),
             attempt to discover the CTK installation root by system-loading a
             well-known CTK library (``cudart``) in a subprocess, then derive
             the root from its resolved absolute path.

    **Driver libraries** (``"cuda"``, ``"nvml"``):

        These are part of the NVIDIA display driver (not the CUDA Toolkit) and
        are always on the system linker path.  For these libraries the search
        is simplified to:

        0. Already loaded in the current process
        1. OS default mechanisms (``dlopen`` / ``LoadLibraryW``)

        The CTK-specific steps (site-packages, conda, ``CUDA_HOME``, canary
        probe) are skipped entirely.

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
    if libname not in _ALL_KNOWN_LIBNAMES:
        raise DynamicLibUnknownError(f"Unknown library name: {libname!r}. Known names: {sorted(_ALL_KNOWN_LIBNAMES)}")
    if libname not in _ALL_SUPPORTED_LIBNAMES:
        raise DynamicLibNotAvailableError(
            f"Library name {libname!r} is known but not available on {_PLATFORM_NAME}. "
            f"Supported names on {_PLATFORM_NAME}: {sorted(_ALL_SUPPORTED_LIBNAMES)}"
        )
    return _load_lib_no_cache(libname)
