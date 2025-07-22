# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: LicenseRef-NVIDIA-SOFTWARE-LICENSE

import pytest

from cuda.bindings.utils import get_minimal_required_cuda_ver_from_ptx_ver, get_ptx_ver

ptx_88_kernel = r"""
.version 8.8
.target sm_75
.address_size 64

	// .globl	empty_kernel

.visible .entry empty_kernel()
{
	ret;
}
"""


ptx_72_kernel = r"""
.version  7.2
.target sm_75
.address_size 64

	// .globl	empty_kernel

.visible .entry empty_kernel()
{
	ret;
}
"""


@pytest.mark.parametrize(
    "kernel,actual_ptx_ver,min_cuda_ver", ((ptx_88_kernel, "8.8", 12090), (ptx_72_kernel, "7.2", 11020))
)
def test_ptx_utils(kernel, actual_ptx_ver, min_cuda_ver):
    ptx_ver = get_ptx_ver(kernel)
    assert ptx_ver == actual_ptx_ver
    cuda_ver = get_minimal_required_cuda_ver_from_ptx_ver(ptx_ver)
    assert cuda_ver == min_cuda_ver
