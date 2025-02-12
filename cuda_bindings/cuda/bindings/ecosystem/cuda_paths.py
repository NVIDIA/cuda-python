# Forked from:
# https://github.com/NVIDIA/numba-cuda/blob/bf487d78a40eea87f009d636882a5000a7524c95/numba_cuda/numba/cuda/cuda_paths.py

import os
import sys
from collections import namedtuple

from findlib import find_lib

IS_WIN32 = sys.platform.startswith("win32")

_env_path_tuple = namedtuple("_env_path_tuple", ["by", "info"])


def _find_valid_path(options):
    """Find valid path from *options*, which is a list of 2-tuple of
    (name, path).  Return first pair where *path* is not None.
    If no valid path is found, return ('<unknown>', None)
    """
    for by, data in options:
        if data is not None:
            return by, data
    else:
        return "<unknown>", None


def _nvvm_lib_dir():
    if IS_WIN32:
        return "nvvm", "bin"
    else:
        return "nvvm", "lib64"


def _get_nvvm_path_decision():
    options = [
        ("Conda environment", get_conda_ctk()),
        ("Conda environment (NVIDIA package)", get_nvidia_nvvm_ctk()),
        ("CUDA_HOME", get_cuda_home(*_nvvm_lib_dir())),
        ("System", get_system_ctk(*_nvvm_lib_dir())),
    ]
    by, path = _find_valid_path(options)
    return by, path


def _cudalib_path():
    if IS_WIN32:
        return "bin"
    else:
        return "lib64"


def get_system_ctk(*subdirs):
    """Return path to system-wide cudatoolkit; or, None if it doesn't exist."""
    # Linux?
    if sys.platform.startswith("linux"):
        # Is cuda alias to /usr/local/cuda?
        # We are intentionally not getting versioned cuda installation.
        base = "/usr/local/cuda"
        if os.path.exists(base):
            return os.path.join(base, *subdirs)


def get_conda_ctk():
    """Return path to directory containing the shared libraries of cudatoolkit."""
    is_conda_env = os.path.exists(os.path.join(sys.prefix, "conda-meta"))
    if not is_conda_env:
        return
    # Assume the existence of NVVM to imply cudatoolkit installed
    paths = find_lib("nvvm")
    if not paths:
        return
    # Use the directory name of the max path
    return os.path.dirname(max(paths))


def get_nvidia_nvvm_ctk():
    """Return path to directory containing the NVVM shared library."""
    is_conda_env = os.path.exists(os.path.join(sys.prefix, "conda-meta"))
    if not is_conda_env:
        return

    # Assume the existence of NVVM in the conda env implies that a CUDA toolkit
    # conda package is installed.

    # First, try the location used on Linux and the Windows 11.x packages
    libdir = os.path.join(sys.prefix, "nvvm", _cudalib_path())
    if not os.path.exists(libdir) or not os.path.isdir(libdir):
        # If that fails, try the location used for Windows 12.x packages
        libdir = os.path.join(sys.prefix, "Library", "nvvm", _cudalib_path())
        if not os.path.exists(libdir) or not os.path.isdir(libdir):
            # If that doesn't exist either, assume we don't have the NVIDIA
            # conda package
            return

    paths = find_lib("nvvm", libdir=libdir)
    if not paths:
        return
    # Use the directory name of the max path
    return os.path.dirname(max(paths))


def get_cuda_home(*subdirs):
    """Get paths of CUDA_HOME.
    If *subdirs* are the subdirectory name to be appended in the resulting
    path.
    """
    cuda_home = os.environ.get("CUDA_HOME")
    if cuda_home is None:
        # Try Windows CUDA installation without Anaconda
        cuda_home = os.environ.get("CUDA_PATH")
    if cuda_home is not None:
        return os.path.join(cuda_home, *subdirs)


def _get_nvvm_path():
    by, path = _get_nvvm_path_decision()
    candidates = find_lib("nvvm", path)
    path = max(candidates) if candidates else None
    return _env_path_tuple(by, path)
