# Copyright 2025 NVIDIA Corporation.  All rights reserved.
# SPDX-License-Identifier: LicenseRef-NVIDIA-SOFTWARE-LICENSE

from dataclasses import dataclass
from typing import Callable, Optional

from .supported_libs import DIRECT_DEPENDENCIES


@dataclass
class LoadedDL:
    """Represents a loaded dynamic library.

    Attributes:
        handle: The library handle (can be converted to void* in Cython)
        abs_path: The absolute path to the library file
        was_already_loaded_from_elsewhere: Whether the library was already loaded
    """

    # ATTENTION: To convert `handle` back to `void*` in cython:
    #     Linux:   `cdef void* ptr = <void*><uintptr_t>`
    #     Windows: `cdef void* ptr = <void*><intptr_t>`
    handle: int
    abs_path: Optional[str]
    was_already_loaded_from_elsewhere: bool


def load_dependencies(libname: str, load_func: Callable[[str], LoadedDL]) -> None:
    """Load all dependencies for a given library.

    Args:
        libname: The name of the library whose dependencies should be loaded
        load_func: The function to use for loading libraries (e.g. load_nvidia_dynamic_library)

    Example:
        >>> load_dependencies("cudart", load_nvidia_dynamic_library)
        # This will load all dependencies of cudart using the provided loading function
    """
    for dep in DIRECT_DEPENDENCIES.get(libname, ()):
        load_func(dep)
