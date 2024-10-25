import pytest
from cuda import cuda
from cuda.core.experimental._device import Device
from cuda.core.experimental._module import Kernel, ObjectCode
from cuda.core.experimental._utils import handle_return

@pytest.fixture(scope='module')
def init_cuda():
    Device().set_current()

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
