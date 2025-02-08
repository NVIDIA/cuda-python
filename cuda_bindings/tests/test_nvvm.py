# Copyright (c) 2025, NVIDIA CORPORATION & AFFILIATES. ALL RIGHTS RESERVED.
#
# SPDX-License-Identifier: LicenseRef-NVIDIA-SOFTWARE-LICENSE

import base64
import re
from contextlib import contextmanager

import pytest

from cuda.bindings import nvvm

MINIMAL_NVVMIR_TXT = b"""\
target datalayout = "e-p:64:64:64-i1:8:8-i8:8:8-i16:16:16-i32:32:32-i64:64:64-i128:128:128-f32:32:32-f64:64:64-v16:16:16-v32:32:32-v64:64:64-v128:128:128-n16:32:64"

target triple = "nvptx64-nvidia-cuda"

define void @kernel() {
entry:
  ret void
}

!nvvm.annotations = !{!0}
!0 = !{void ()* @kernel, !"kernel", i32 1}

!nvvmir.version = !{!1}
!1 = !{i32 %d, i32 %d, i32 %d, i32 %d}
"""  # noqa: E501

# Equivalent to MINIMAL_NVVMIR_TXT % (2, 0, 3, 1)
MINIMAL_NVVMIR_BITCODE_2_0_3_1 = base64.b64decode("""
QkPA3jUUAAAFAAAAYgwwJElZvmbu034tRAEyBQAAAAAhDAAAJAEAAAsCIQACAAAAEwAAAAeBI5FB
yARJBhAyOZIBhAwlBQgZHgSLYoAMRQJCkgtCZBAyFDgIGEsKMjKISJAUIENGiKUAGTJC5EgOkJEh
xFBBUYGM4YPligQZRgaJIAAACwAAADIiyAggZIUEkyGkhASTIeOEoZAUEkyGjAuEZEwQFCMAJQBl
IGCOAAwAAAAAEyZ3sAd4oAd8sAM6aAN3sId0IId0CIc2GId6IIdw2OAS5dAG8KAHdkAHemAHdKAH
dkAHbZAOcaAHeKAHeNAG6YAHeoAHeoAHbZAOcWAHehAHdqAHcWAHbZAOcyAHejAHcqAHcyAHbZAO
dkAHemAHdKAHdkAHbZAOcSAHeKAHcSAHeKAHcSAHeNAG5jAHcqAHcyAHejAHctAG5mAHdKAHdkAH
emAHdNAG9hAHdqAHcWAHehAHdtAG9jAHcqAHcyAHejAHctAG9mAHdKAHdkAHemAHdNAG9hAHcoAH
ehAHcoAHehAHcoAHbeAOcWAHejAHcqAHdkAHGiEMGTFIgzDA8jdVxSCRvyxDIsAIAAAAAAAAAAAA
AEBig0BRlAAAgCwQBgAAADIemAwZEUyQjAkmR8YEQ2IJFMEIQBkAALEYAABtAAAAMwiAHMThHGYU
AT2IQziEw4xCgAd5eAdzmHEM5gAP7RAO9IAOMwxCHsLBHc6hHGYwBT2IQziEgxvMAz3IQz2MAz3M
eIx0cAd7CAd5SIdwcAd6cAN2eIdwIIcZzBEO7JAO4TAPbjAP4/AO8FAOMxDEHd4hHNghHcJhHmYw
iTu8gzvQQzm0Azy8gzyEAzvM8BR2YAd7aAc3aIdyaAc3gIdwkIdwYAd2KAd2+AV2eId3gIdfCIdx
GIdymId5mIEs7vAO7uAO9cAO7DADYsihHOShHMyhHOShHNxhHMohHMSBHcphBtaQQznIQzmYQznI
Qzm4wziUQziIAzuUwy+8gzz8gjvUAzuwwwzHaYdwWIdycIN0aAd4YId0GId0oIcZzlMP7gAP8lAO
5JAO40AP4SAO7FAOMyAoHdzBHsJBHtIhHNyBHtzgHOThHeoBHmYYUTiwQzqcgzvMUCR2YAd7aAc3
YId3eAd4mFFM9JAP8FAOMx5qHsphHOghHd7BHX4BHuShHMwhHfBhBlSFgzjMwzuwQz3QQzn8wjzk
QzuIwzuww4zFCod5mId3GId0CAd6KAdyAAAAAHkgAAAeAAAAYh5IIEOIDBk5GSSQkUDGyMhoIlAI
FDKeGBkhR8iQUQwIBQAABgAAAGtlcm5lbAAAIwgCMIJABCMIhDCCQAwjCAQxwyAEwwwEURiDjAQm
KCE3O7s2lzA3tze6MLq0N7e5UQIjHTc7u7Y0ORe7Mrm5tDe3UYIDAAAAqRgAAAsAAAALCnIoh3eA
B3pYcJhDPbjDOLBDOdDDguYcxqEN6EEewsEd5iEd6CEd3sEdANEQAAAGAAAAB8w8pIM7nAM7lAM9
oIM8lEM4kMMBAAAAYSAAAAYAAAATBAGGAwEAAAIAAAAHUBDNFGEAAAAAAABxIAAAAwAAADIOECKE
AKACAAAAAAAAAABlDAAAHQAAABIDlOgAAAAAAAAAAAYAAAAFAAAARAAAAAEAAABQAAAAAAAAAFAA
AAABAAAAaAAAAAAAAAALAAAAEwAAAB4AAAARAAAALwAAAAAAAAAAAAAAAQAAAAAAAAAAAAAABgAA
AAAAAAAGAAAA/////wAkAAAAAAAAXQwAAA8AAAASA5RvAAAAAGtlcm5lbDUuMC4xbnZwdHg2NC1u
dmlkaWEtY3VkYW1pbmltYWxfbnZ2bWlyLmxsAAAAAAA=
""")
# To regenerate, pull and start a docker container:
#     docker pull centos/llvm-toolset-7-centos7
#     docker run -it centos/llvm-toolset-7-centos7 /bin/bash
# In the docker container, copy MINIMAL_NVVMIR_TXT to a file with name minimal_nvvmir.ll
# Then run:
#     llvm-as minimal_nvvmir.ll -o minimal_nvvmir.bc
# Save this to encode.py:
#     import base64, sys, textwrap
#     bitcode = open(sys.argv[1], "rb").read()
#     encoded_bitcode = base64.b64encode(bitcode).decode("ascii")
#     wrapped_base64 = "\n".join(textwrap.wrap(encoded_bitcode, width=76))
#     print(wrapped_base64)
# Then run:
#     python encode.py minimal_nvvmir.bc


@pytest.fixture(params=["txt", "bitcode"])
def minimal_nvvmir(request):
    ir_vers = nvvm.ir_version()
    if request.param == "txt":
        return MINIMAL_NVVMIR_TXT % ir_vers
    if ir_vers[:2] != (3, 0):
        pytest.skip(f"MINIMAL_NVVMIR_BITCODE_2_0_3_1 vs {ir_vers} IR version incompatibility")
    return MINIMAL_NVVMIR_BITCODE_2_0_3_1


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


def test_c_or_v_program_fail_bad_option(minimal_nvvmir, compile_or_verify):
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
def test_compile_program_with_minimal_nnvm_ir(minimal_nvvmir, options):
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
def test_verify_program_with_minimal_nnvm_ir(minimal_nvvmir, options):
    with nvvm_program() as prog:
        nvvm.add_module_to_program(prog, minimal_nvvmir, len(minimal_nvvmir), "FileNameHere.ll")
        nvvm.verify_program(prog, len(options), options)
