# Copyright (c) 2025, NVIDIA CORPORATION & AFFILIATES. ALL RIGHTS RESERVED.
#
# SPDX-License-Identifier: LicenseRef-NVIDIA-SOFTWARE-LICENSE


from cuda.bindings import nvvm


def test_nvvm_version():
    ver = nvvm.version()
    assert len(ver) == 2
    assert ver >= (12, 0)
