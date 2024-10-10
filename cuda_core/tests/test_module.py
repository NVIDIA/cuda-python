from cuda.py._module import Module

def test_module_initialization():
    module_code = b"dummy_code"
    code_type = "ptx"
    module = Module(module=module_code, code_type=code_type)
    assert module._handle is not None
    assert module._code_type == code_type
    assert module._module == module_code
    assert module._loader is not None
    assert module._sym_map == {}