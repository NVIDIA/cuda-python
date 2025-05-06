# Copyright 2025 NVIDIA Corporation.  All rights reserved.
# SPDX-License-Identifier: LicenseRef-NVIDIA-SOFTWARE-LICENSE

from dataclasses import dataclass
from typing import Callable, Optional

from cuda.bindings._path_finder.run_python_code_safely import run_python_code_safely
from cuda.bindings._path_finder.supported_libs import DIRECT_DEPENDENCIES, IS_WINDOWS

if IS_WINDOWS:
    import pywintypes

    HandleType = pywintypes.HANDLE
else:
    HandleType = int


@dataclass
class LoadedDL:
    handle: HandleType
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


def load_in_subprocess(python_code, timeout=30):
    # This is to avoid loading libraries into the parent process.
    return run_python_code_safely(python_code, timeout=timeout)


def build_subprocess_failed_for_libname_message(libname, result):
    return (
        f"Subprocess failed for {libname=!r} with exit code {result.returncode}\n"
        f"--- stdout-from-subprocess ---\n{result.stdout}<end-of-stdout-from-subprocess>\n"
        f"--- stderr-from-subprocess ---\n{result.stderr}<end-of-stderr-from-subprocess>\n"
    )
