# Copyright 2024 NVIDIA Corporation.  All rights reserved.
#
# Please refer to the NVIDIA end user license agreement (EULA) associated
# with this source code for terms and conditions that govern your use of
# this software. Any use, reproduction, disclosure, or distribution of
# this software and related documentation outside the terms of the EULA
# is strictly prohibited.

from cuda.core.experimental._module import ObjectCode
import pytest
import importlib

@pytest.mark.skipif(int(importlib.metadata.version("cuda-python").split(".")[0]) < 12, reason='Module loading for older drivers validate require valid module code.')
def test_object_code_initialization():
    # Test with supported code types
    for code_type in ["cubin", "ptx", "fatbin"]:
        module_data = b"dummy_data"
        obj_code = ObjectCode(module_data, code_type)
        assert obj_code._code_type == code_type
        assert obj_code._module == module_data
        assert obj_code._handle is not None

    # Test with unsupported code type
    with pytest.raises(ValueError):
        ObjectCode(b"dummy_data", "unsupported_code_type")

#TODO add ObjectCode tests which provide the appropriate data for cuLibraryLoadFromFile
def test_object_code_initialization_with_str():
    assert True

def test_object_code_initialization_with_jit_options():
    assert True

def test_object_code_get_kernel():
    assert True

def test_kernel_from_obj():
    assert True
