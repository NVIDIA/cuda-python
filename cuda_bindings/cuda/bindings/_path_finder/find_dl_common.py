# Copyright 2024-2025 NVIDIA Corporation.  All rights reserved.
#
# SPDX-License-Identifier: LicenseRef-NVIDIA-SOFTWARE-LICENSE

import os
from typing import List, Optional

from .cuda_paths import get_cuda_paths
from .sys_path_find_sub_dirs import sys_path_find_sub_dirs


def no_such_file_in_sub_dirs(
    sub_dirs: tuple[str, ...], file_wild: str, error_messages: List[str], attachments: List[str]
) -> None:
    """Report that a file was not found in the given subdirectories.

    Args:
        sub_dirs: Tuple of subdirectory names to search
        file_wild: The file pattern to search for
        error_messages: List to append error messages to
        attachments: List to append directory listings to
    """
    error_messages.append(f"No such file: {file_wild}")
    for sub_dir in sys_path_find_sub_dirs(sub_dirs):
        attachments.append(f'  listdir("{sub_dir}"):')
        for node in sorted(os.listdir(sub_dir)):
            attachments.append(f"    {node}")


def get_cuda_paths_info(key: str, error_messages: List[str]) -> Optional[str]:
    """Get information from cuda_paths for a given key.

    Args:
        key: The key to look up in cuda_paths
        error_messages: List to append error messages to

    Returns:
        The path info if found, None otherwise
    """
    env_path_tuple = get_cuda_paths()[key]
    if not env_path_tuple:
        error_messages.append(f'Failure obtaining get_cuda_paths()["{key}"]')
        return None
    if not env_path_tuple.info:
        error_messages.append(f'Failure obtaining get_cuda_paths()["{key}"].info')
        return None
    return env_path_tuple.info


class FindResult:
    """Result of a library search operation.

    Attributes:
        abs_path: The absolute path to the found library, or None if not found
        error_messages: List of error messages encountered during the search
        attachments: List of additional information (e.g. directory listings)
        lib_searched_for: The library name that was searched for
    """

    def __init__(self, lib_searched_for: str):
        self.abs_path: Optional[str] = None
        self.error_messages: List[str] = []
        self.attachments: List[str] = []
        self.lib_searched_for = lib_searched_for

    def raise_if_abs_path_is_None(self) -> str:
        """Raise an error if no library was found.

        Returns:
            The absolute path to the found library

        Raises:
            RuntimeError: If no library was found
        """
        if self.abs_path:
            return self.abs_path
        err = ", ".join(self.error_messages)
        att = "\n".join(self.attachments)
        raise RuntimeError(f'Failure finding "{self.lib_searched_for}": {err}\n{att}')
