# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import functools
import os

from cuda.pathfinder._binaries import supported_nvidia_binaries
from cuda.pathfinder._utils.ctk_root_canary import CTK_ROOT_CANARY_ANCHOR_LIBNAMES
from cuda.pathfinder._utils.env_vars import get_cuda_path_or_home
from cuda.pathfinder._utils.find_sub_dirs import find_sub_dirs_all_sitepackages
from cuda.pathfinder._utils.platform_aware import IS_WINDOWS


class UnsupportedBinaryError(Exception):
    def __init__(self, utility: str) -> None:
        super().__init__(utility)
        self.utility = utility

    def __str__(self) -> str:
        supported_utilities = ", ".join(supported_nvidia_binaries.SUPPORTED_BINARIES)
        return f"Binary '{self.utility}' is not supported. Supported utilities are: {supported_utilities}"


def _normalize_utility_name(utility_name: str) -> str:
    """Normalize utility name by adding .exe on Windows if needed."""
    if IS_WINDOWS and not utility_name.lower().endswith((".exe", ".bat", ".cmd")):
        return f"{utility_name}.exe"
    return utility_name


def _is_executable_candidate(path: str) -> bool:
    if not os.path.isfile(path):
        return False
    if IS_WINDOWS:
        return True
    return os.access(path, os.X_OK)


def _ctk_bin_subdirs(root: str) -> list[str]:
    if IS_WINDOWS:
        return [
            os.path.join(root, "bin", "x64"),
            os.path.join(root, "bin", "x86_64"),
            os.path.join(root, "bin"),
        ]
    return [os.path.join(root, "bin")]


def _resolve_ctk_root_via_canary() -> str | None:
    from cuda.pathfinder._dynamic_libs.load_nvidia_dynamic_lib import resolve_ctk_root_via_canary

    ctk_root: str | None = resolve_ctk_root_via_canary(CTK_ROOT_CANARY_ANCHOR_LIBNAMES[0])
    return ctk_root


def _resolve_in_trusted_dirs(normalized_name: str, dirs: list[str]) -> str | None:
    """Resolve ``normalized_name`` against ``dirs`` in order."""
    seen: set[str] = set()
    for directory in dirs:
        if directory in seen:
            continue
        assert directory
        seen.add(directory)
        candidate = os.path.join(directory, normalized_name)
        if _is_executable_candidate(candidate):
            # Return an absolute path, as the docstring promises (a relative
            # search dir would otherwise leak a relative result).
            return os.path.abspath(candidate)
    return None


@functools.cache
def find_nvidia_binary_utility(utility_name: str) -> str | None:
    """Locate a CUDA binary utility executable.

    Args:
        utility_name (str): The name of the binary utility to find
            (e.g., ``"nvdisasm"``, ``"cuobjdump"``). On Windows, the ``.exe``
            extension will be automatically appended if not present. The function
            also recognizes ``.bat`` and ``.cmd`` files on Windows.

    Returns:
        str or None: Absolute path to the discovered executable, or ``None``
        if the utility cannot be found. The returned path is normalized
        (absolute and with resolved separators).

    Raises:
        UnsupportedBinaryError: If ``utility_name`` is not in the supported set
            (see ``SUPPORTED_BINARY_UTILITIES``).

    Search order:
        1. **NVIDIA Python wheels**

           - Scan installed distributions (``site-packages``) for binary layouts
             shipped in NVIDIA wheels (e.g., ``cuda-nvcc``).

        2. **Conda environments**

           - Check Conda-style installation prefixes via ``CONDA_PREFIX``
             environment variable, which use platform-specific bin directory
             layouts (``Library/bin`` on Windows, ``bin`` on Linux).

        3. **CUDA Toolkit environment variables**

           - Use ``CUDA_HOME`` or ``CUDA_PATH`` (in that order), searching
             ``bin/x64``, ``bin/x86_64``, and ``bin`` subdirectories on Windows,
             or just ``bin`` on Linux.

        4. **CTK-root canary fallback**

           - Only when steps 1-3 miss: resolve the ``cudart`` library through the
             OS dynamic loader, derive the CUDA Toolkit root from it, and search
             that root's bin layout.

    Note:
        Results are cached using ``@functools.cache`` for performance. The cache
        persists for the lifetime of the process.

        On Windows, executables are identified by their file extensions
        (``.exe``, ``.bat``, ``.cmd``). On Unix-like systems, executables
        are identified by the ``X_OK`` (execute) permission bit.

        Lookup is restricted to the trusted directories and the canary-derived
        CTK root listed above.

    Example:
        >>> from cuda.pathfinder import find_nvidia_binary_utility
        >>> nvdisasm = find_nvidia_binary_utility("nvdisasm")
        >>> if nvdisasm:
        ...     print(f"Found nvdisasm at: {nvdisasm}")
    """
    if utility_name not in supported_nvidia_binaries.SUPPORTED_BINARIES:
        raise UnsupportedBinaryError(utility_name)

    # 1. Search in site-packages (NVIDIA wheels)
    candidate_dirs = supported_nvidia_binaries.SITE_PACKAGES_BINDIRS.get(utility_name, ())
    dirs = []

    for sub_dir in candidate_dirs:
        dirs.extend(find_sub_dirs_all_sitepackages(sub_dir.split(os.sep)))

    # 2. Search in Conda environment
    if (conda_prefix := os.environ.get("CONDA_PREFIX")) is not None:
        if IS_WINDOWS:
            dirs.append(os.path.join(conda_prefix, "Library", "bin"))
        else:
            dirs.append(os.path.join(conda_prefix, "bin"))

    # 3. Search in CUDA Toolkit (CUDA_HOME/CUDA_PATH)
    if (cuda_home := get_cuda_path_or_home()) is not None:
        dirs.extend(_ctk_bin_subdirs(cuda_home))

    normalized_name = _normalize_utility_name(utility_name)
    found = _resolve_in_trusted_dirs(normalized_name, dirs)
    if found is not None:
        return found

    # 4. CTK-root canary fallback.
    ctk_root = _resolve_ctk_root_via_canary()
    if ctk_root is not None:
        return _resolve_in_trusted_dirs(normalized_name, _ctk_bin_subdirs(ctk_root))
    return None
