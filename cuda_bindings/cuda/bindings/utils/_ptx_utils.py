# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: LicenseRef-NVIDIA-SOFTWARE-LICENSE

import re

# Mapping based on the official PTX ISA <-> CUDA Release table
# https://docs.nvidia.com/cuda/parallel-thread-execution/#release-notes-ptx-release-history
_ptx_to_cuda = {
    "1.0": (1, 0),
    "1.1": (1, 1),
    "1.2": (2, 0),
    "1.3": (2, 1),
    "1.4": (2, 2),
    "2.0": (3, 0),
    "2.1": (3, 1),
    "2.2": (3, 2),
    "2.3": (4, 0),
    "3.0": (4, 1),
    "3.1": (5, 0),
    "3.2": (5, 5),
    "4.0": (6, 0),
    "4.1": (6, 5),
    "4.2": (7, 0),
    "4.3": (7, 5),
    "5.0": (8, 0),
    "6.0": (9, 0),
    "6.1": (9, 1),
    "6.2": (9, 2),
    "6.3": (10, 0),
    "6.4": (10, 1),
    "6.5": (10, 2),
    "7.0": (11, 0),
    "7.1": (11, 1),
    "7.2": (11, 2),
    "7.3": (11, 3),
    "7.4": (11, 4),
    "7.5": (11, 5),
    "7.6": (11, 6),
    "7.7": (11, 7),
    "7.8": (11, 8),
    "8.0": (12, 0),
    "8.1": (12, 1),
    "8.2": (12, 2),
    "8.3": (12, 3),
    "8.4": (12, 4),
    "8.5": (12, 5),
    "8.6": (12, 7),
    "8.7": (12, 8),
    "8.8": (12, 9),
}


def get_minimal_required_cuda_ver_from_ptx_ver(ptx_version: str) -> int:
    """
    Maps the PTX ISA version to the minimal CUDA driver, nvPTXCompiler, or nvJitLink version
    that is needed to load a PTX of the given ISA version.

    Parameters
    ----------
    ptx_version : str
        PTX ISA version as a string, e.g. "8.8" for PTX ISA 8.8. This is the ``.version``
        directive in the PTX header.

    Returns
    -------
    int
        Minimal CUDA version as 1000 * major + 10 * minor, e.g. 12090 for CUDA 12.9.

    Raises
    ------
    ValueError
        If the PTX version is unknown.

    Examples
    --------
    >>> get_minimal_required_driver_ver_from_ptx_ver("8.8")
    12090
    >>> get_minimal_required_driver_ver_from_ptx_ver("7.0")
    11000
    """
    try:
        major, minor = _ptx_to_cuda[ptx_version]
        return 1000 * major + 10 * minor
    except KeyError:
        raise ValueError(f"Unknown or unsupported PTX ISA version: {ptx_version}") from None


# Regex pattern to match .version directive and capture the version number
# TODO: if import speed is a concern, consider lazy-initializing it.
_ptx_ver_pattern = re.compile(r"\.version\s+([0-9]+\.[0-9]+)")


def get_ptx_ver(ptx: str) -> str:
    """
    Extract the PTX ISA version string from PTX source code.

    Parameters
    ----------
    ptx : str
        The PTX assembly source code as a string.

    Returns
    -------
    str
        The PTX ISA version string, e.g., "8.8".

    Raises
    ------
    ValueError
        If the .version directive is not found in the PTX source.

    Examples
    --------
    >>> ptx = r'''
    ... .version 8.8
    ... .target sm_86
    ... .address_size 64
    ...
    ... .visible .entry test_kernel()
    ... {
    ...     ret;
    ... }
    ... '''
    >>> get_ptx_ver(ptx)
    '8.8'
    """
    m = _ptx_ver_pattern.search(ptx)
    if m:
        return m.group(1)
    else:
        raise ValueError("No .version directive found in PTX source. Is it a valid PTX?")
