# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

# Currently these installations are only manually tested:

# pip install nvidia-nvshmem-cu12
# pip install nvidia-nvshmem-cu13

# conda create -y -n nvshmem python=3.12
# conda activate nvshmem
# conda install -y conda-forge::libnvshmem3 conda-forge::libnvshmem-dev

# wget https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2404/x86_64/cuda-keyring_1.1-1_all.deb
# sudo dpkg -i cuda-keyring_1.1-1_all.deb
# sudo apt update
# sudo apt install libnvshmem3-cuda-12 libnvshmem3-dev-cuda-12
# sudo apt install libnvshmem3-cuda-13 libnvshmem3-dev-cuda-13

import pytest

from cuda.pathfinder import _find_nvidia_header_directory as find_nvidia_header_directory


def test_find_nvidia_header_directory(info_summary_append):
    with pytest.raises(RuntimeError, match=r"^UNKNOWN libname='unknown-libname'$"):
        find_nvidia_header_directory("unknown-libname")

    hdr_dir = find_nvidia_header_directory("nvshmem")
    # TODO: Find ways to test more meaningfully.

    info_summary_append(f"{hdr_dir=!r}")
