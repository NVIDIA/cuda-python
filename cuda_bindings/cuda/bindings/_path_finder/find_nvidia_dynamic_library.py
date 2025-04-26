# Copyright 2024-2025 NVIDIA Corporation.  All rights reserved.
#
# SPDX-License-Identifier: LicenseRef-NVIDIA-SOFTWARE-LICENSE

import functools
import sys
from typing import Dict

from .cuda_paths import get_cuda_paths
from .find_dl_linux import find_nvidia_dynamic_library as find_nvidia_dynamic_library_linux
from .find_dl_windows import find_nvidia_dynamic_library as find_nvidia_dynamic_library_windows


class FindNvidiaDynamicLibrary:
    """Class for finding NVIDIA dynamic libraries.

    This class maintains the same interface as the original _find_nvidia_dynamic_library
    class for backward compatibility.
    """

    def __init__(self, libname: str):
        """Initialize the finder with a library name.

        Args:
            libname: The name of the library to find
        """
        self.libname = libname
        self.error_messages = []
        self.attachments = []
        self.abs_path = None
        self.lib_searched_for = f"lib{libname}.so" if sys.platform != "win32" else f"{libname}.dll"

        # Special case for nvvm
        if libname == "nvvm":
            nvvm_path = get_cuda_paths()["nvvm"]
            if nvvm_path and nvvm_path.info:
                self.abs_path = nvvm_path.info
                return

        if sys.platform == "linux":
            result = find_nvidia_dynamic_library_linux(libname)
        elif sys.platform == "win32":
            result = find_nvidia_dynamic_library_windows(libname)
        else:
            raise NotImplementedError(f"Platform {sys.platform} is not supported")

        self.abs_path = result.abs_path
        self.error_messages = result.error_messages
        self.attachments = result.attachments

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


# Cache for found libraries
_found_libraries: Dict[str, str] = {}


@functools.cache
def find_nvidia_dynamic_library(libname: str) -> str:
    """Find a NVIDIA dynamic library.

    This function will cache the results of successful lookups to avoid repeated searches.

    Args:
        libname: The library name to search for (e.g. "cudart", "nvvm")

    Returns:
        The absolute path to the found library

    Raises:
        RuntimeError: If the library cannot be found
        NotImplementedError: If the current platform is not supported
    """
    # Check cache first
    if libname in _found_libraries:
        return _found_libraries[libname]

    # Use the class-based approach for backward compatibility
    finder = FindNvidiaDynamicLibrary(libname)
    result = finder.raise_if_abs_path_is_None()

    # Cache the result
    _found_libraries[libname] = result
    return result
