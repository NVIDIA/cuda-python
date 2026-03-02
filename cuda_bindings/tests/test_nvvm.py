# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# SPDX-License-Identifier: LicenseRef-NVIDIA-SOFTWARE-LICENSE

import re
from contextlib import contextmanager

import pytest
from cuda.bindings import nvvm

pytest_plugins = ("cuda_python_test_helpers.nvvm_bitcode",)


@pytest.fixture(params=[nvvm.compile_program, nvvm.verify_program])
def compile_or_verify(request):
    return request.param


def match_exact(s):
    return "^" + re.escape(s) + "$"


@contextmanager
def nvvm_program() -> int:
    prog: int = nvvm.create_program()
    try:
        yield prog
    finally:
        nvvm.destroy_program(prog)


def get_program_log(prog):
    buffer = bytearray(nvvm.get_program_log_size(prog))
    nvvm.get_program_log(prog, buffer)
    return buffer.decode(errors="backslashreplace")


def test_get_error_string():
    num_success = 0
    num_errors = 0
    for enum_obj in nvvm.Result:
        es = nvvm.get_error_string(enum_obj)
        if enum_obj is nvvm.Result.SUCCESS:
            num_success += 1
        else:
            assert es.startswith("NVVM_ERROR")
            num_errors += 1
    assert num_success == 1
    assert num_errors > 1  # smoke check is sufficient


def test_nvvm_version():
    ver = nvvm.version()
    assert len(ver) == 2
    assert ver >= (1, 0)


def test_nvvm_ir_version():
    ver = nvvm.ir_version()
    assert len(ver) == 4
    assert ver >= (1, 0, 0, 0)


def test_create_and_destroy():
    with nvvm_program() as prog:
        assert isinstance(prog, int)
        assert prog != 0


@pytest.mark.parametrize("add_fn", [nvvm.add_module_to_program, nvvm.lazy_add_module_to_program])
def test_add_module_to_program_fail(add_fn):
    with nvvm_program() as prog, pytest.raises(ValueError):
        # Passing a C NULL pointer generates "ERROR_INVALID_INPUT (4)",
        # but that is not possible through our Python bindings.
        # The ValueError originates from the cython bindings code.
        add_fn(prog, None, 0, "FileNameHere.ll")


def test_c_or_v_program_fail_no_module(compile_or_verify):
    with nvvm_program() as prog, pytest.raises(nvvm.nvvmError, match=match_exact("ERROR_NO_MODULE_IN_PROGRAM (8)")):
        compile_or_verify(prog, 0, [])


def test_c_or_v_program_fail_invalid_ir(compile_or_verify):
    expected_error = "ERROR_COMPILATION (9)" if compile_or_verify is nvvm.compile_program else "ERROR_INVALID_IR (6)"
    nvvm_ll = b"This is not NVVM IR"
    with nvvm_program() as prog:
        nvvm.add_module_to_program(prog, nvvm_ll, len(nvvm_ll), "FileNameHere.ll")
        with pytest.raises(nvvm.nvvmError, match=match_exact(expected_error)):
            compile_or_verify(prog, 0, [])
        assert get_program_log(prog) == "FileNameHere.ll (1, 0): parse expected top-level entity\x00"


def test_c_or_v_program_fail_bad_option(minimal_nvvmir, compile_or_verify):  # noqa: F401, F811
    with nvvm_program() as prog:
        nvvm.add_module_to_program(prog, minimal_nvvmir, len(minimal_nvvmir), "FileNameHere.ll")
        with pytest.raises(nvvm.nvvmError, match=match_exact("ERROR_INVALID_OPTION (7)")):
            compile_or_verify(prog, 1, ["BadOption"])
        assert get_program_log(prog) == "libnvvm : error: BadOption is an unsupported option\x00"


@pytest.mark.parametrize(
    ("get_size", "get_buffer"),
    [
        (nvvm.get_compiled_result_size, nvvm.get_compiled_result),
        (nvvm.get_program_log_size, nvvm.get_program_log),
    ],
)
def test_get_buffer_empty(get_size, get_buffer):
    with nvvm_program() as prog:
        buffer_size = get_size(prog)
        assert buffer_size == 1
        buffer = bytearray(buffer_size)
        get_buffer(prog, buffer)
        assert buffer == b"\x00"


@pytest.mark.parametrize("options", [[], ["-opt=0"], ["-opt=3", "-g"]])
def test_compile_program_with_minimal_nvvm_ir(minimal_nvvmir, options):  # noqa: F401, F811
    with nvvm_program() as prog:
        nvvm.add_module_to_program(prog, minimal_nvvmir, len(minimal_nvvmir), "FileNameHere.ll")
        try:
            nvvm.compile_program(prog, len(options), options)
        except nvvm.nvvmError as e:
            raise RuntimeError(get_program_log(prog)) from e
        else:
            log_size = nvvm.get_program_log_size(prog)
            assert log_size == 1
            buffer = bytearray(log_size)
            nvvm.get_program_log(prog, buffer)
            assert buffer == b"\x00"
        result_size = nvvm.get_compiled_result_size(prog)
        buffer = bytearray(result_size)
        nvvm.get_compiled_result(prog, buffer)
        assert ".visible .entry kernel()" in buffer.decode()


@pytest.mark.parametrize("options", [[], ["-opt=0"], ["-opt=3", "-g"]])
def test_verify_program_with_minimal_nvvm_ir(minimal_nvvmir, options):  # noqa: F401, F811
    with nvvm_program() as prog:
        nvvm.add_module_to_program(prog, minimal_nvvmir, len(minimal_nvvmir), "FileNameHere.ll")
        nvvm.verify_program(prog, len(options), options)
