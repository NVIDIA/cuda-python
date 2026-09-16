# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import functools
import os
from collections.abc import Iterable

from cuda.pathfinder._binaries import supported_nvidia_binaries, windows_nsight
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


def _resolve_candidate_paths(candidates: Iterable[str]) -> str | None:
    """Return the first executable candidate, preserving candidate order."""
    seen: set[str] = set()
    for candidate in candidates:
        if candidate in seen:
            continue
        seen.add(candidate)
        if _is_executable_candidate(candidate):
            return os.path.abspath(candidate)
    return None


def _find_windows_compute_sanitizer(ctk_root: str) -> str | None:
    return _resolve_candidate_paths(
        (
            os.path.join(ctk_root, "bin", "compute-sanitizer.bat"),
            os.path.join(ctk_root, "compute-sanitizer", "compute-sanitizer.exe"),
        )
    )


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


def _resolve_names_in_trusted_dirs(candidate_names: tuple[str, ...], dirs: list[str]) -> str | None:
    """Resolve ordered candidate names within each trusted directory."""
    seen: set[str] = set()
    for directory in dirs:
        if directory in seen:
            continue
        assert directory
        seen.add(directory)
        found = _resolve_candidate_paths(os.path.join(directory, name) for name in candidate_names)
        if found is not None:
            return found
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
        RuntimeError: If a native Windows architecture needed for an
            architecture-specific utility layout cannot be determined, or an
            installed Nsight product has incomplete or invalid registry data.

    Windows on ARM (WoA) Note:
        Binary utilities execute in separate processes and do not need to match
        the Python process architecture. When choosing among architecture-specific
        Windows layouts, this API deliberately targets the native machine
        architecture rather than the Python interpreter architecture. For
        example, standalone ``nsys`` and ``ncu`` discovery under x64 Python on an
        Arm64 machine selects the Arm64 target. This differs from
        ``load_nvidia_dynamic_lib`` and ``find_static_lib``, which target the
        Python interpreter architecture.

    Search order:
        1. **NVIDIA Python wheels**

           - Scan installed distributions (``site-packages``) for binary layouts
             shipped in NVIDIA wheels (e.g., ``cuda-nvcc``).

        2. **Conda environments**

           - Check Conda-style installation prefixes via ``CONDA_PREFIX``
             environment variable, which use platform-specific bin directory
             layouts (``Library/bin`` on Windows, ``bin`` on Linux).

        3. **Library-specific standalone installations**

           - Search the installation paths for the CUDA Toolkit, Nsight Systems,
             and Nsight Compute.

           3.1. **Nsight installations**: On Windows, locate Nsight Systems and
                Nsight Compute from their installer registry entries. Select
                architecture-specific binaries using the native machine
                architecture, independent of Python. Lookup of the standalone
                ``nsys`` and ``ncu`` CLIs is terminal; a miss does not fall
                through to CUDA Toolkit locations.

           3.2. **CUDA Toolkit installation**: Use ``CUDA_PATH`` or ``CUDA_HOME``
                (in that order), searching ``bin/x64``, ``bin/x86_64``, and
                ``bin`` subdirectories on Windows, or just ``bin`` on Linux.

        4. **CTK-root canary fallback**

           - For utilities that reach this step after the earlier searches miss,
             resolve the ``cudart`` library through the OS dynamic loader, derive
             the CUDA Toolkit root from it, and search that root's bin layout.

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

    normalized_name = _normalize_utility_name(utility_name)
    if IS_WINDOWS and utility_name in ("compute-sanitizer", "ncu"):
        candidate_names = (f"{utility_name}.bat", normalized_name)
        found = _resolve_names_in_trusted_dirs(candidate_names, dirs)
    else:
        found = _resolve_in_trusted_dirs(normalized_name, dirs)
    if found is not None:
        return found

    # 3. Search library-specific standalone installations.
    # 3.1. Standalone Nsight CLI lookup is terminal; CTK does not contain nsys/ncu.
    if IS_WINDOWS and utility_name == "nsys":
        return _resolve_candidate_paths(windows_nsight.nsys_candidate_paths())
    if IS_WINDOWS and utility_name == "ncu":
        return _resolve_candidate_paths(windows_nsight.ncu_candidate_paths())

    # 3.2. Search in CUDA Toolkit (CUDA_PATH/CUDA_HOME).
    if (cuda_path := get_cuda_path_or_home()) is not None:
        if IS_WINDOWS and utility_name == "compute-sanitizer":
            found = _find_windows_compute_sanitizer(cuda_path)
        else:
            found = _resolve_in_trusted_dirs(normalized_name, _ctk_bin_subdirs(cuda_path))
        if found is not None:
            return found

    # 4. CTK-root canary fallback.
    ctk_root = _resolve_ctk_root_via_canary()
    if ctk_root is not None:
        if IS_WINDOWS and utility_name == "compute-sanitizer":
            return _find_windows_compute_sanitizer(ctk_root)
        return _resolve_in_trusted_dirs(normalized_name, _ctk_bin_subdirs(ctk_root))
    return None
