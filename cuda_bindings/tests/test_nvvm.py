# Copyright (c) 2025, NVIDIA CORPORATION & AFFILIATES. ALL RIGHTS RESERVED.
#
# SPDX-License-Identifier: LicenseRef-NVIDIA-SOFTWARE-LICENSE

import re

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


def noregex(s):
    return "^" + re.escape(s) + "$"


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


@pytest.mark.parametrize("add_fn", [nvvm.add_module_to_program, nvvm.lazy_add_module_to_program])
def test_add_module_to_program_fail(add_fn):
    prog = nvvm.create_program()
    try:
        # Passing a C NULL pointer generates "ERROR_INVALID_INPUT (4)",
        # but that is not possible through our Python bindings.
        # The ValueError originates from the cython bindings code.
        with pytest.raises(ValueError):
            add_fn(prog, None, 0, "FileNameHere.ll")
    finally:
        nvvm.destroy_program(prog)


@pytest.mark.parametrize("c_or_v", [nvvm.compile_program, nvvm.verify_program])
def test_c_or_v_program_fail_no_module(c_or_v):
    prog = nvvm.create_program()
    try:
        with pytest.raises(nvvm.nvvmError, match=noregex("ERROR_NO_MODULE_IN_PROGRAM (8)")):
            c_or_v(prog, 0, [])
    finally:
        nvvm.destroy_program(prog)


@pytest.mark.parametrize(
    ("c_or_v", "expected_error"),
    [
        (nvvm.compile_program, "ERROR_COMPILATION (9)"),
        (nvvm.verify_program, "ERROR_INVALID_IR (6)"),
    ],
)
def test_c_or_v_program_fail_invalid_ir(c_or_v, expected_error):
    prog = nvvm.create_program()
    nvvm_ll = b"This is not NVVM IR"
    nvvm.add_module_to_program(prog, nvvm_ll, len(nvvm_ll), "FileNameHere.ll")
    try:
        with pytest.raises(nvvm.nvvmError, match=noregex(expected_error)):
            c_or_v(prog, 0, [])
        buffer = bytearray(nvvm.get_program_log_size(prog))
        nvvm.get_program_log(prog, buffer)
        assert buffer.decode(errors="backslashreplace") == "FileNameHere.ll (1, 0): parse expected top-level entity\x00"
    finally:
        nvvm.destroy_program(prog)


@pytest.mark.parametrize("c_or_v", [nvvm.compile_program, nvvm.verify_program])
def test_c_or_v_program_fail_bad_option(c_or_v):
    prog = nvvm.create_program()
    nvvm.add_module_to_program(prog, MINIMAL_NVVMIR, len(MINIMAL_NVVMIR), "FileNameHere.ll")
    try:
        with pytest.raises(nvvm.nvvmError, match=noregex("ERROR_INVALID_OPTION (7)")):
            c_or_v(prog, 1, ["BadOption"])
        buffer = bytearray(nvvm.get_program_log_size(prog))
        nvvm.get_program_log(prog, buffer)
        assert buffer.decode(errors="backslashreplace") == "libnvvm : error: BadOption is an unsupported option\x00"
    finally:
        nvvm.destroy_program(prog)


@pytest.mark.parametrize(
    ("get_size", "get_buffer"),
    [
        (nvvm.get_compiled_result_size, nvvm.get_compiled_result),
        (nvvm.get_program_log_size, nvvm.get_program_log),
    ],
)
def test_get_buffer_empty(get_size, get_buffer):
    prog = nvvm.create_program()
    try:
        buffer_size = get_size(prog)
        assert buffer_size == 1
        buffer = bytearray(buffer_size)
        get_buffer(prog, buffer)
        assert buffer == b"\x00"
    finally:
        nvvm.destroy_program(prog)


@pytest.mark.parametrize("options", [[], ["-opt=0"], ["-opt=3", "-g"]])
def test_compile_program_with_minimal_nnvm_ir(options):
    prog = nvvm.create_program()
    try:
        nvvm.add_module_to_program(prog, MINIMAL_NVVMIR, len(MINIMAL_NVVMIR), "FileNameHere.ll")
        try:
            nvvm.compile_program(prog, len(options), options)
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


@pytest.mark.parametrize("options", [[], ["-opt=0"], ["-opt=3", "-g"]])
def test_verify_program_with_minimal_nnvm_ir(options):
    prog = nvvm.create_program()
    try:
        nvvm.add_module_to_program(prog, MINIMAL_NVVMIR, len(MINIMAL_NVVMIR), "FileNameHere.ll")
        nvvm.verify_program(prog, len(options), options)
    finally:
        nvvm.destroy_program(prog)
