# Copyright (c) 2025, NVIDIA CORPORATION & AFFILIATES. ALL RIGHTS RESERVED.
#
# SPDX-License-Identifier: LicenseRef-NVIDIA-SOFTWARE-LICENSE

import pytest

from cuda.bindings import nvvm

MINIMAL_NVVMIR = b"""\
target datalayout = "e-p:64:64:64-i1:8:8-i8:8:8-i16:16:16-i32:32:32-i64:64:64-i128:128:128-f32:32:32-f64:64:64-v16:16:16-v32:32:32-v64:64:64-v128:128:128-n16:32:64"

target triple = "nvptx64-nvidia-cuda"

define void @kernel() {
entry:
  ret void
}

!nvvm.annotations = !{!0}
!0 = !{void ()* @kernel, !"kernel", i32 1}

!nvvmir.version = !{!1}
!1 = !{i32 2, i32 0, i32 3, i32 1}
"""  # noqa: E501


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
        nvvm_ll = b"This is not NVVM IR"
        # TODO(rwgk): Find an input that generates an ERROR.
        # with pytest.raises(nvvm.nvvmError, match=r"^ERROR_INVALID_INPUT \(4\)$"):
        nvvm.add_module_to_program(prog, nvvm_ll, len(nvvm_ll), "SomeName")
    finally:
        nvvm.destroy_program(prog)


def test_lazy_add_module_to_program():
    prog = nvvm.create_program()
    try:
        nvvm_ll = b"This is not NVVM IR"
        # TODO(rwgk): Find an input that generates an ERROR.
        # with pytest.raises(nvvm.nvvmError, match=r"^ERROR_INVALID_INPUT \(4\)$"):
        nvvm.lazy_add_module_to_program(prog, nvvm_ll, len(nvvm_ll), "SomeName")
    finally:
        nvvm.destroy_program(prog)


def test_compile_program_fail():
    prog = nvvm.create_program()
    try:
        with pytest.raises(nvvm.nvvmError, match=r"^ERROR_NO_MODULE_IN_PROGRAM \(8\)$"):
            nvvm.compile_program(prog, 0, [])
    finally:
        nvvm.destroy_program(prog)


def test_verify_program_fail():
    prog = nvvm.create_program()
    try:
        with pytest.raises(nvvm.nvvmError, match=r"^ERROR_NO_MODULE_IN_PROGRAM \(8\)$"):
            nvvm.verify_program(prog, 0, [])
    finally:
        nvvm.destroy_program(prog)


def test_get_compiled_result_empty():
    prog = nvvm.create_program()
    try:
        result_size = nvvm.get_compiled_result_size(prog)
        assert result_size == 1
        buffer = bytearray(result_size)
        nvvm.get_compiled_result(prog, buffer)
        assert buffer == b"\x00"
    finally:
        nvvm.destroy_program(prog)


def test_get_program_log_empty():
    prog = nvvm.create_program()
    try:
        log_size = nvvm.get_program_log_size(prog)
        assert log_size == 1
        buffer = bytearray(log_size)
        nvvm.get_program_log(prog, buffer)
        assert buffer == b"\x00"
    finally:
        nvvm.destroy_program(prog)


def test_compile_program_with_minimal_nnvm_ir():
    prog = nvvm.create_program()
    try:
        nvvm.add_module_to_program(prog, MINIMAL_NVVMIR, len(MINIMAL_NVVMIR), "FileNameHere.ll")
        try:
            nvvm.compile_program(prog, 0, [])
        except nvvm.nvvmError as e:
            buffer = bytearray(nvvm.get_program_log_size(prog))
            nvvm.get_program_log(prog, buffer)
            raise RuntimeError(buffer.decode(errors="backslashreplace")) from e
        else:
            log_size = nvvm.get_program_log_size(prog)
            assert log_size == 1
            buffer = bytearray(log_size)
            nvvm.get_program_log(prog, buffer)
            assert buffer == b"\x00"
        result_size = nvvm.get_compiled_result_size(prog)
        buffer = bytearray(result_size)
        nvvm.get_compiled_result(prog, buffer)
        assert ".visible .entry kernel()" in buffer.decode("utf-8")
    finally:
        nvvm.destroy_program(prog)


def test_verify_program_with_minimal_nnvm_ir():
    prog = nvvm.create_program()
    try:
        nvvm.add_module_to_program(prog, MINIMAL_NVVMIR, len(MINIMAL_NVVMIR), "FileNameHere.ll")
        nvvm.verify_program(prog, 0, [])
    finally:
        nvvm.destroy_program(prog)
