# Copyright (c) 2025, NVIDIA CORPORATION & AFFILIATES. ALL RIGHTS RESERVED.
#
# SPDX-License-Identifier: LicenseRef-NVIDIA-SOFTWARE-LICENSE

import pytest

from cuda.bindings import nvvm


def test_nvvm_version():
    ver = nvvm.version()
    assert len(ver) == 2
    assert ver >= (2, 0)


def test_nvvm_ir_version():
    ver = nvvm.ir_version()
    assert len(ver) == 4
    assert ver >= (2, 0, 3, 1)


def test_create_and_destroy():
    prog = nvvm.create_program()
    try:
        assert isinstance(prog, int)
        assert prog != 0
    finally:
        nvvm.destroy_program(prog)


def test_add_module_to_program():
    prog = nvvm.create_program()
    try:
        with pytest.raises(nvvm.nvvmError, match=r"^ERROR_INVALID_INPUT \(4\)$"):
            nvvm.add_module_to_program(prog, [], 0, "SomeName")
    finally:
        nvvm.destroy_program(prog)


def test_lazy_add_module_to_program():
    prog = nvvm.create_program()
    try:
        with pytest.raises(nvvm.nvvmError, match=r"^ERROR_INVALID_INPUT \(4\)$"):
            nvvm.lazy_add_module_to_program(prog, [], 0, "SomeName")
    finally:
        nvvm.destroy_program(prog)


def test_compile_program():
    prog = nvvm.create_program()
    try:
        with pytest.raises(nvvm.nvvmError, match=r"^ERROR_NO_MODULE_IN_PROGRAM \(8\)$"):
            nvvm.compile_program(prog, 0, [])
    finally:
        nvvm.destroy_program(prog)


def test_verify_program():
    prog = nvvm.create_program()
    try:
        with pytest.raises(nvvm.nvvmError, match=r"^ERROR_NO_MODULE_IN_PROGRAM \(8\)$"):
            nvvm.verify_program(prog, 0, [])
    finally:
        nvvm.destroy_program(prog)


def test_get_compiled_result():
    prog = nvvm.create_program()
    try:
        result_size = nvvm.get_compiled_result_size(prog)
        assert result_size == 1
        buffer = bytearray(result_size)
        nvvm.get_compiled_result(prog, buffer)
        assert buffer == b"\x00"
    finally:
        nvvm.destroy_program(prog)


def test_get_program_log():
    prog = nvvm.create_program()
    try:
        log_size = nvvm.get_program_log_size(prog)
        assert log_size == 1
        buffer = bytearray(log_size)
        nvvm.get_program_log(prog, buffer)
        assert buffer == b"\x00"
    finally:
        nvvm.destroy_program(prog)
