# Copyright 2024-2025 NVIDIA Corporation.  All rights reserved.
#
# SPDX-License-Identifier: LicenseRef-NVIDIA-SOFTWARE-LICENSE

import os

from .find_dl_common import FindResult, get_cuda_paths_info


def find_dll_under_dir(lib_searched_for: str, dir_path: str) -> FindResult:
    """Find a .dll file under a specific directory.

    Args:
        lib_searched_for: The library name to search for
        dir_path: The directory to search in

    Returns:
        FindResult containing the search results
    """
    result = FindResult(lib_searched_for)
    file_wild = f"{lib_searched_for}.dll"

    if not os.path.isdir(dir_path):
        result.error_messages.append(f"No such directory: {dir_path}")
        return result

    for node in sorted(os.listdir(dir_path)):
        if node.lower() == file_wild.lower():
            result.abs_path = os.path.join(dir_path, node)
            return result

    result.error_messages.append(f"No such file: {file_wild}")
    result.attachments.append(f'  listdir("{dir_path}"):')
    for node in sorted(os.listdir(dir_path)):
        result.attachments.append(f"    {node}")
    return result


def find_dll_using_cudalib_dir(lib_searched_for: str) -> FindResult:
    """Find a .dll file using the CUDA library directory.

    Args:
        lib_searched_for: The library name to search for

    Returns:
        FindResult containing the search results
    """
    result = FindResult(lib_searched_for)
    cudalib_dir = get_cuda_paths_info("cudalib_dir", result.error_messages)
    if not cudalib_dir:
        return result

    return find_dll_under_dir(lib_searched_for, cudalib_dir)


def find_dll_using_cuda_path(lib_searched_for: str) -> FindResult:
    """Find a .dll file using the CUDA path.

    Args:
        lib_searched_for: The library name to search for

    Returns:
        FindResult containing the search results
    """
    result = FindResult(lib_searched_for)
    cuda_path = get_cuda_paths_info("cuda_path", result.error_messages)
    if not cuda_path:
        return result

    bin_path = os.path.join(cuda_path, "bin")
    return find_dll_under_dir(lib_searched_for, bin_path)


def find_nvidia_dynamic_library(lib_searched_for: str) -> FindResult:
    """Find a NVIDIA dynamic library on Windows.

    Args:
        lib_searched_for: The library name to search for

    Returns:
        FindResult containing the search results
    """
    # Try CUDA library directory first
    result = find_dll_using_cudalib_dir(lib_searched_for)
    if result.abs_path:
        return result

    # Then try CUDA path
    result = find_dll_using_cuda_path(lib_searched_for)
    return result
