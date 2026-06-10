# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import functools
import os

from cuda.pathfinder._binaries import supported_nvidia_binaries
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


def _is_executable_file(path: str) -> bool:
    """Return True if ``path`` is a file the OS would run as an executable.

    On Windows executability is determined by the file extension (the
    candidate name already carries one), so existence is sufficient. On POSIX
    the execute permission bit must be set, matching ``shutil.which``.
    """
    if not os.path.isfile(path):
        return False
    if IS_WINDOWS:
        return True
    return os.access(path, os.X_OK)


def _resolve_in_trusted_dirs(normalized_name: str, dirs: list[str]) -> str | None:
    """Resolve ``normalized_name`` against ``dirs`` only, in order.

    Unlike ``shutil.which``, this never consults the current working directory
    or the ambient ``PATH``. On Windows ``shutil.which`` prepends the process
    CWD to the search even when an explicit ``path=`` is supplied, which lets a
    binary sitting in an arbitrary CWD shadow the trusted CUDA / Conda / wheel
    binary that pathfinder is contracted to discover. Searching the trusted
    directories explicitly keeps the lookup deterministic and bounded.
    """
    seen: set[str] = set()
    for directory in dirs:
        if not directory or directory in seen:
            continue
        seen.add(directory)
        candidate = os.path.join(directory, normalized_name)
        if _is_executable_file(candidate):
            return candidate
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

    Note:
        Results are cached using ``@functools.cache`` for performance. The cache
        persists for the lifetime of the process.

        On Windows, executables are identified by their file extensions
        (``.exe``, ``.bat``, ``.cmd``). On Unix-like systems, executables
        are identified by the ``X_OK`` (execute) permission bit.

        Lookup is restricted to the trusted directories listed above; the
        process working directory and the ambient ``PATH`` are never consulted.

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
        if IS_WINDOWS:
            dirs.append(os.path.join(cuda_home, "bin", "x64"))
            dirs.append(os.path.join(cuda_home, "bin", "x86_64"))
        dirs.append(os.path.join(cuda_home, "bin"))

    normalized_name = _normalize_utility_name(utility_name)
    return _resolve_in_trusted_dirs(normalized_name, dirs)
