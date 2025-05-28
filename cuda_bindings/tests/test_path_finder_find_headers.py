# Copyright 2025 NVIDIA Corporation.  All rights reserved.
# SPDX-License-Identifier: LicenseRef-NVIDIA-SOFTWARE-LICENSE

import pytest

from cuda.bindings._path_finder import find_nvidia_headers


def test_find_nvidia_header_directory(info_summary_append):
    with pytest.raises(RuntimeError, match="^UNKNOWN libname='unknown-libname'$"):
        find_nvidia_headers.find_nvidia_header_directory("unknown-libname")

    hdr_dir = find_nvidia_headers.find_nvidia_header_directory("nvshmem")
    # TODO: Find ways to test more meaningfully, and how to avoid HARD-WIRED PATHS in particular.
    assert hdr_dir in [
        # pip install nvidia-nvshmem-cu12
        "/home/rgrossekunst/forked/cuda-python/venvs/scratch/lib/python3.12/site-packages/nvidia/nvshmem/include",
        #
        # conda create -y -n nvshmem python=3.12
        # conda activate nvshmem
        # conda install -y conda-forge::libnvshmem3 conda-forge::libnvshmem-dev
        "/home/rgrossekunst/miniforge3/envs/nvshmem/include",
        #
        # sudo apt install libnvshmem3-cuda-12 libnvshmem3-dev-cuda-12
        "/usr/include/nvshmem_12",
        #
        # nvshmem not available
        None,
    ]
    info_summary_append(f"{hdr_dir=!r}")
