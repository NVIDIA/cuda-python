# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

# Currently these installations are only manually tested:

# conda create -y -n nvshmem python=3.12
# conda activate nvshmem
# conda install -y conda-forge::libnvshmem3 conda-forge::libnvshmem-dev

# wget https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2404/x86_64/cuda-keyring_1.1-1_all.deb
# sudo dpkg -i cuda-keyring_1.1-1_all.deb
# sudo apt update
# sudo apt install libnvshmem3-cuda-12 libnvshmem3-dev-cuda-12
# sudo apt install libnvshmem3-cuda-13 libnvshmem3-dev-cuda-13

import functools
import importlib.metadata
import os
import re

import pytest

from cuda.pathfinder import find_nvidia_header_directory
from cuda.pathfinder._headers.supported_nvidia_headers import (
    IS_WINDOWS,
    SUPPORTED_HEADERS_CTK,
    SUPPORTED_HEADERS_CTK_ALL,
    SUPPORTED_SITE_PACKAGE_HEADER_DIRS_CTK,
)

STRICTNESS = os.environ.get("CUDA_PATHFINDER_TEST_FIND_NVIDIA_HEADERS_STRICTNESS", "see_what_works")
assert STRICTNESS in ("see_what_works", "all_must_work")


@functools.cache
def have_nvidia_nvshmem_package() -> bool:
    pattern = re.compile(r"^nvidia-nvshmem-.*$")
    return any(
        pattern.match(dist.metadata["Name"]) for dist in importlib.metadata.distributions() if "Name" in dist.metadata
    )


def test_unknown_libname():
    with pytest.raises(RuntimeError, match=r"^UNKNOWN libname='unknown-libname'$"):
        find_nvidia_header_directory("unknown-libname")


def test_find_libname_nvshmem(info_summary_append):
    hdr_dir = find_nvidia_header_directory("nvshmem")
    info_summary_append(f"{hdr_dir=!r}")
    if IS_WINDOWS:
        assert hdr_dir is None
        pytest.skip("nvshmem has no Windows support.")
    if hdr_dir:
        assert os.path.isdir(hdr_dir)
        assert os.path.isfile(os.path.join(hdr_dir, "nvshmem.h"))
    if STRICTNESS == "all_must_work" or have_nvidia_nvshmem_package():
        assert hdr_dir is not None
        if have_nvidia_nvshmem_package():
            hdr_dir_parts = hdr_dir.split(os.path.sep)
            assert "site-packages" in hdr_dir_parts
        elif conda_prefix := os.getenv("CONDA_PREFIX"):
            assert hdr_dir.startswith(conda_prefix)
        else:
            assert hdr_dir.startswith("/usr/include/nvshmem_")


def test_supported_headers_site_packages_ctk_consistency():
    assert tuple(sorted(SUPPORTED_HEADERS_CTK_ALL)) == tuple(sorted(SUPPORTED_SITE_PACKAGE_HEADER_DIRS_CTK.keys()))


@pytest.mark.parametrize("libname", SUPPORTED_HEADERS_CTK.keys())
def test_find_ctk_headers(info_summary_append, libname):
    hdr_dir = find_nvidia_header_directory(libname)
    info_summary_append(f"{hdr_dir=!r}")
    if hdr_dir:
        assert os.path.isdir(hdr_dir)
        h_filename = SUPPORTED_HEADERS_CTK[libname]
        assert os.path.isfile(os.path.join(hdr_dir, h_filename))
    if STRICTNESS == "all_must_work":
        assert hdr_dir is not None
