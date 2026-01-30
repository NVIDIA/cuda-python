# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

# Currently these installations are only manually tested:

# ../toolshed/conda_create_for_pathfinder_testing.*

# wget https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2404/x86_64/cuda-keyring_1.1-1_all.deb
# sudo dpkg -i cuda-keyring_1.1-1_all.deb
# sudo apt update
# sudo apt install libnvshmem3-cuda-12 libnvshmem3-dev-cuda-12
# sudo apt install libnvshmem3-cuda-13 libnvshmem3-dev-cuda-13

import functools
import glob
import importlib.metadata
import os
import re

import pytest

from cuda.pathfinder import FoundHeaderDir, find_nvidia_header_directory, locate_nvidia_header_directory
from cuda.pathfinder._headers.supported_nvidia_headers import (
    SUPPORTED_HEADERS_CTK,
    SUPPORTED_HEADERS_CTK_ALL,
    SUPPORTED_HEADERS_NON_CTK,
    SUPPORTED_HEADERS_NON_CTK_ALL,
    SUPPORTED_INSTALL_DIRS_NON_CTK,
    SUPPORTED_SITE_PACKAGE_HEADER_DIRS_CTK,
)

STRICTNESS = os.environ.get("CUDA_PATHFINDER_TEST_FIND_NVIDIA_HEADERS_STRICTNESS", "see_what_works")
assert STRICTNESS in ("see_what_works", "all_must_work")

NON_CTK_IMPORTLIB_METADATA_DISTRIBUTIONS_NAMES = {
    "cusparseLt": r"^nvidia-cusparselt-.*$",
    "cutensor": r"^cutensor-.*$",
    "nvshmem": r"^nvidia-nvshmem-.*$",
}


def test_unknown_libname():
    with pytest.raises(RuntimeError, match=r"^UNKNOWN libname='unknown-libname'$"):
        find_nvidia_header_directory("unknown-libname")


def test_non_ctk_importlib_metadata_distributions_names():
    # Ensure the dict keys above stay in sync with supported_nvidia_headers
    assert sorted(NON_CTK_IMPORTLIB_METADATA_DISTRIBUTIONS_NAMES) == sorted(SUPPORTED_HEADERS_NON_CTK_ALL)


@functools.cache
def have_distribution_for(libname: str) -> bool:
    pattern = re.compile(NON_CTK_IMPORTLIB_METADATA_DISTRIBUTIONS_NAMES[libname])
    return any(
        pattern.match(dist.metadata["Name"]) for dist in importlib.metadata.distributions() if "Name" in dist.metadata
    )


@pytest.mark.parametrize("libname", SUPPORTED_HEADERS_NON_CTK.keys())
def test_locate_non_ctk_headers(info_summary_append, libname):
    found_header_dir = locate_nvidia_header_directory(libname)
    info_summary_append(f"{None if not found_header_dir else found_header_dir.abs_path=!r}")
    if found_header_dir:
        assert isinstance(found_header_dir, FoundHeaderDir)
        assert found_header_dir.found_via in ("site-packages", "conda", "CUDA_HOME")
        # old api
        hdr_dir = find_nvidia_header_directory(libname)
        assert hdr_dir == found_header_dir.abs_path

        assert os.path.isdir(found_header_dir.abs_path)
        assert os.path.isfile(os.path.join(found_header_dir.abs_path, SUPPORTED_HEADERS_NON_CTK[libname]))
    if have_distribution_for(libname):
        assert found_header_dir.abs_path is not None
        found_header_dir_parts = found_header_dir.abs_path.split(os.path.sep)
        assert "site-packages" in found_header_dir_parts
    elif STRICTNESS == "all_must_work":
        assert found_header_dir is not None
        if conda_prefix := os.environ.get("CONDA_PREFIX"):
            assert found_header_dir.abs_path.startswith(conda_prefix)
        else:
            inst_dirs = SUPPORTED_INSTALL_DIRS_NON_CTK.get(libname)
            if inst_dirs is not None:
                for inst_dir in inst_dirs:
                    globbed = glob.glob(inst_dir)
                    if found_header_dir.abs_path in globbed:
                        break
                else:
                    raise RuntimeError(f"{found_header_dir.abs_path=} does not match any {inst_dirs=}")


def test_supported_headers_site_packages_ctk_consistency():
    assert tuple(sorted(SUPPORTED_HEADERS_CTK_ALL)) == tuple(sorted(SUPPORTED_SITE_PACKAGE_HEADER_DIRS_CTK.keys()))


@pytest.mark.parametrize("libname", SUPPORTED_HEADERS_CTK.keys())
def test_locate_ctk_headers(info_summary_append, libname):
    found_header_dir = locate_nvidia_header_directory(libname)
    info_summary_append(f"{None if not found_header_dir else found_header_dir.abs_path=!r}")
    if found_header_dir:
        # old api
        hdr_dir = find_nvidia_header_directory(libname)
        assert hdr_dir == found_header_dir.abs_path

        assert os.path.isdir(found_header_dir.abs_path)
        h_filename = SUPPORTED_HEADERS_CTK[libname]
        assert os.path.isfile(os.path.join(found_header_dir.abs_path, h_filename))
    if STRICTNESS == "all_must_work":
        assert found_header_dir.abs_path is not None
