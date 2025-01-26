# Copyright (c) 2025, NVIDIA CORPORATION & AFFILIATES. ALL RIGHTS RESERVED.
#
# SPDX-License-Identifier: LicenseRef-NVIDIA-SOFTWARE-LICENSE


from cuda.bindings import nvvm


def test_nvvm_version():
    ver = nvvm.version()
    assert len(ver) == 2
    assert ver >= (2, 0)


def test_nvvm_ir_version():
    ver = nvvm.ir_version()
    assert len(ver) == 4
    assert ver >= (2, 0, 3, 1)
