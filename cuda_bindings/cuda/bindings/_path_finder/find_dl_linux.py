# Copyright 2024-2025 NVIDIA Corporation.  All rights reserved.
#
# SPDX-License-Identifier: LicenseRef-NVIDIA-SOFTWARE-LICENSE

import os

from .find_dl_common import FindResult, get_cuda_paths_info, no_such_file_in_sub_dirs
from .sys_path_find_sub_dirs import sys_path_find_sub_dirs


def find_so_using_nvidia_lib_dirs(lib_searched_for: str) -> FindResult:
    """Find a .so file using NVIDIA library directories.

    Args:
        lib_searched_for: The library name to search for

    Returns:
        FindResult containing the search results
    """
    result = FindResult(lib_searched_for)
    file_wild = f"lib{lib_searched_for}.so*"
    sub_dirs = ("lib", "lib64")

    for sub_dir in sys_path_find_sub_dirs(sub_dirs):
        for node in sorted(os.listdir(sub_dir)):
            if node.startswith(f"lib{lib_searched_for}.so"):
                result.abs_path = os.path.join(sub_dir, node)
                return result

    no_such_file_in_sub_dirs(sub_dirs, file_wild, result.error_messages, result.attachments)
    return result


def find_so_using_cudalib_dir(lib_searched_for: str) -> FindResult:
    """Find a .so file using the CUDA library directory.

    Args:
        lib_searched_for: The library name to search for

    Returns:
        FindResult containing the search results
    """
    result = FindResult(lib_searched_for)
    cudalib_dir = get_cuda_paths_info("cudalib_dir", result.error_messages)
    if not cudalib_dir:
        return result

    file_wild = f"lib{lib_searched_for}.so*"
    for node in sorted(os.listdir(cudalib_dir)):
        if node.startswith(f"lib{lib_searched_for}.so"):
            result.abs_path = os.path.join(cudalib_dir, node)
            return result

    result.error_messages.append(f"No such file: {file_wild}")
    result.attachments.append(f'  listdir("{cudalib_dir}"):')
    for node in sorted(os.listdir(cudalib_dir)):
        result.attachments.append(f"    {node}")
    return result


def find_so_using_cuda_path(lib_searched_for: str) -> FindResult:
    """Find a .so file using the CUDA path.

    Args:
        lib_searched_for: The library name to search for

    Returns:
        FindResult containing the search results
    """
    result = FindResult(lib_searched_for)
    cuda_path = get_cuda_paths_info("cuda_path", result.error_messages)
    if not cuda_path:
        return result

    file_wild = f"lib{lib_searched_for}.so*"
    for sub_dir in ("lib", "lib64"):
        path = os.path.join(cuda_path, sub_dir)
        if not os.path.isdir(path):
            continue
        for node in sorted(os.listdir(path)):
            if node.startswith(f"lib{lib_searched_for}.so"):
                result.abs_path = os.path.join(path, node)
                return result

    result.error_messages.append(f"No such file: {file_wild}")
    for sub_dir in ("lib", "lib64"):
        path = os.path.join(cuda_path, sub_dir)
        if os.path.isdir(path):
            result.attachments.append(f'  listdir("{path}"):')
            for node in sorted(os.listdir(path)):
                result.attachments.append(f"    {node}")
    return result


def find_nvidia_dynamic_library(lib_searched_for: str) -> FindResult:
    """Find a NVIDIA dynamic library on Linux.

    Args:
        lib_searched_for: The library name to search for

    Returns:
        FindResult containing the search results
    """
    # Try NVIDIA library directories first
    result = find_so_using_nvidia_lib_dirs(lib_searched_for)
    if result.abs_path:
        return result

    # Then try CUDA library directory
    result = find_so_using_cudalib_dir(lib_searched_for)
    if result.abs_path:
        return result

    # Finally try CUDA path
    result = find_so_using_cuda_path(lib_searched_for)
    return result
