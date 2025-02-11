# Copyright (c) 2025, NVIDIA CORPORATION & AFFILIATES. ALL RIGHTS RESERVED.
#
# SPDX-License-Identifier: LicenseRef-NVIDIA-SOFTWARE-LICENSE

import binascii
import re
import textwrap
from contextlib import contextmanager

import pytest

from cuda.bindings import nvvm

MINIMAL_NVVMIR_FIXTURE_PARAMS = ["txt", "bitcode_static"]
try:
    import llvmlite.binding as llvmlite_binding  # Optional test dependency.
except ImportError:
    llvmlite_binding = None
else:
    MINIMAL_NVVMIR_FIXTURE_PARAMS.append("bitcode_dynamic")

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
!1 = !{i32 %d, i32 0, i32 %d, i32 0}
"""  # noqa: E501

MINIMAL_NVVMIR_BITCODE_STATIC = {
    (1, 3):  # (major, debug_major)
    "4243c0de3514000005000000620c30244a59be669dfbb4bf0b51804c01000000210c00007f010000"
    "0b02210002000000160000000781239141c80449061032399201840c250508191e048b62800c4502"
    "42920b42641032143808184b0a3232884870c421234412878c1041920264c808b1142043468820c9"
    "01323284182a282a90317cb05c9120c3c8000000892000000b0000003222c80820624600212b2498"
    "0c212524980c19270c85a4906032645c20246382a01801300128030173046000132677b00778a007"
    "7cb0033a680377b0877420877408873618877a208770d8e012e5d006f0a0077640077a600774a007"
    "7640076d900e71a00778a00778d006e980077a80077a80076d900e7160077a100776a0077160076d"
    "900e7320077a300772a0077320076d900e7640077a600774a0077640076d900e71200778a0077120"
    "0778a00771200778d006e6300772a0077320077a300772d006e6600774a0077640077a600774d006"
    "f6100776a0077160077a100776d006f6300772a0077320077a300772d006f6600774a0077640077a"
    "600774d006f610077280077a10077280077a10077280076de00e7160077a300772a0077640071a21"
    "4c0e11de9c2e4fbbcfbe211560040000000000000000000000000620b141a0e86000004016080000"
    "06000000321e980c19114c908c092647c6044362098c009401000000b1180000ac0000003308801c"
    "c4e11c6614013d88433884c38c4280077978077398710ce6000fed100ef4800e330c421ec2c11dce"
    "a11c6630053d88433884831bcc033dc8433d8c033dcc788c7470077b08077948877070077a700376"
    "788770208719cc110eec900ee1300f6e300fe3f00ef0500e3310c41dde211cd8211dc2611e663089"
    "3bbc833bd04339b4033cbc833c84033bccf0147660077b6807376887726807378087709087706007"
    "76280776f8057678877780875f08877118877298877998812ceef00eeee00ef5c00eec300362c8a1"
    "1ce4a11ccca11ce4a11cdc611cca211cc4811dca6106d6904339c84339984339c84339b8c3389443"
    "3888033b94c32fbc833cfc823bd4033bb0c30cc7698770588772708374680778608774188774a087"
    "19ce530fee000ff2500ee4900ee3400fe1200eec500e3320281ddcc11ec2411ed2211cdc811edce0"
    "1ce4e11dea011e66185138b0433a9c833bcc50247660077b68073760877778077898514cf4900ff0"
    "500e331e6a1eca611ce8211ddec11d7e011ee4a11ccc211df0610654858338ccc33bb0433dd04339"
    "fcc23ce4433b88c33bb0c38cc50a877998877718877408077a28077298815ce3100eecc00ee5500e"
    "f33023c1d2411ee4e117d8e11dde011e6648193bb0833db4831b84c3388c4339ccc33cb8c139c8c3"
    "3bd4033ccc48b471080776600771088771588719dbc60eec600fede006f0200fe5300fe5200ff650"
    "0e6e100ee3300ee5300ff3e006e9e00ee4500ef83023e2ec611cc2811dd8e117ec211de6211dc421"
    "1dd8211de8211f66209d3bbc433db80339948339cc58bc7070077778077a08077a488777708719cb"
    "e70eef300fe1e00ee9400fe9a00fe530c3010373a8077718875f988770708774a08774d087729881"
    "844139e0c338b0433d904339cc40c4a01dcaa11de0411edec11c662463300ee1c00eec300fe9400f"
    "e5000000792000001d000000721e482043880c19097232482023818c9191d144a01028643c313242"
    "8e9021a318100a00060000006b65726e656c0000230802308240042308843082400c330c4230cc40"
    "0c4441c84860821272b3b36b730973737ba30ba34b7b739b1b2528d271b3b36b4b9373b12b939b4b"
    "7b731b2530000000a9180000250000000b0a7228877780077a587098433db8c338b04339d0c382e6"
    "1cc6a10de8411ec2c11de6211de8211ddec11d1634e3600ee7500fe1200fe4400fe1200fe7500ef4"
    "b08081077928877060077678877108077a28077258709cc338b4013ba4833d94c3026b1cd8211cdc"
    "e11cdc201ce4611cdc201ce8811ec2611cd0a11cc8611cc2811dd861c1010ff4200fe1500ff4800e"
    "00000000d11000000600000007cc3ca4833b9c033b94033da0833c94433890c30100000061200000"
    "06000000130481860301000002000000075010cd14610000000000007120000003000000320e1022"
    "8400fb020000000000000000650c00001f000000120394f000000000030000000600000006000000"
    "4c000000010000005800000000000000580000000100000070000000000000000c00000013000000"
    "1f000000080000000600000000000000700000000000000000000000010000000000000000000000"
    "060000000000000006000000ffffffff00240000000000005d0c00000d0000001203946700000000"
    "6b65726e656c31352e302e376e7670747836342d6e76696469612d637564613c737472696e673e00"
    "00000000",
    (2, 3):  # (major, debug_major)
    "4243c0de3514000005000000620c30244a59be669dfbb4bf0b51804c01000000210c000080010000"
    "0b02210002000000160000000781239141c80449061032399201840c250508191e048b62800c4502"
    "42920b42641032143808184b0a3232884870c421234412878c1041920264c808b1142043468820c9"
    "01323284182a282a90317cb05c9120c3c8000000892000000b0000003222c80820624600212b2498"
    "0c212524980c19270c85a4906032645c20246382a01801300128030173046000132677b00778a007"
    "7cb0033a680377b0877420877408873618877a208770d8e012e5d006f0a0077640077a600774a007"
    "7640076d900e71a00778a00778d006e980077a80077a80076d900e7160077a100776a0077160076d"
    "900e7320077a300772a0077320076d900e7640077a600774a0077640076d900e71200778a0077120"
    "0778a00771200778d006e6300772a0077320077a300772d006e6600774a0077640077a600774d006"
    "f6100776a0077160077a100776d006f6300772a0077320077a300772d006f6600774a0077640077a"
    "600774d006f610077280077a10077280077a10077280076de00e7160077a300772a0077640071a21"
    "4c0e11de9c2e4fbbcfbe211560040000000000000000000000000620b141a0286100004016080000"
    "06000000321e980c19114c908c092647c60443620914c10840190000b1180000ac0000003308801c"
    "c4e11c6614013d88433884c38c4280077978077398710ce6000fed100ef4800e330c421ec2c11dce"
    "a11c6630053d88433884831bcc033dc8433d8c033dcc788c7470077b08077948877070077a700376"
    "788770208719cc110eec900ee1300f6e300fe3f00ef0500e3310c41dde211cd8211dc2611e663089"
    "3bbc833bd04339b4033cbc833c84033bccf0147660077b6807376887726807378087709087706007"
    "76280776f8057678877780875f08877118877298877998812ceef00eeee00ef5c00eec300362c8a1"
    "1ce4a11ccca11ce4a11cdc611cca211cc4811dca6106d6904339c84339984339c84339b8c3389443"
    "3888033b94c32fbc833cfc823bd4033bb0c30cc7698770588772708374680778608774188774a087"
    "19ce530fee000ff2500ee4900ee3400fe1200eec500e3320281ddcc11ec2411ed2211cdc811edce0"
    "1ce4e11dea011e66185138b0433a9c833bcc50247660077b68073760877778077898514cf4900ff0"
    "500e331e6a1eca611ce8211ddec11d7e011ee4a11ccc211df0610654858338ccc33bb0433dd04339"
    "fcc23ce4433b88c33bb0c38cc50a877998877718877408077a28077298815ce3100eecc00ee5500e"
    "f33023c1d2411ee4e117d8e11dde011e6648193bb0833db4831b84c3388c4339ccc33cb8c139c8c3"
    "3bd4033ccc48b471080776600771088771588719dbc60eec600fede006f0200fe5300fe5200ff650"
    "0e6e100ee3300ee5300ff3e006e9e00ee4500ef83023e2ec611cc2811dd8e117ec211de6211dc421"
    "1dd8211de8211f66209d3bbc433db80339948339cc58bc7070077778077a08077a488777708719cb"
    "e70eef300fe1e00ee9400fe9a00fe530c3010373a8077718875f988770708774a08774d087729881"
    "844139e0c338b0433d904339cc40c4a01dcaa11de0411edec11c662463300ee1c00eec300fe9400f"
    "e5000000792000001e000000721e482043880c19097232482023818c9191d144a01028643c313242"
    "8e9021a318100a00060000006b65726e656c0000230802308240042308843082400c23080431c320"
    "04c30c045118858c04262821373bbb36973037b737ba30bab437b7b95102231d373bbbb6343917bb"
    "32b9b9b437b7518203000000a9180000250000000b0a7228877780077a587098433db8c338b04339"
    "d0c382e61cc6a10de8411ec2c11de6211de8211ddec11d1634e3600ee7500fe1200fe4400fe1200f"
    "e7500ef4b08081077928877060077678877108077a28077258709cc338b4013ba4833d94c3026b1c"
    "d8211cdce11cdc201ce4611cdc201ce8811ec2611cd0a11cc8611cc2811dd861c1010ff4200fe150"
    "0ff4800e00000000d11000000600000007cc3ca4833b9c033b94033da0833c94433890c301000000"
    "6120000006000000130481860301000002000000075010cd14610000000000007120000003000000"
    "320e10228400fc020000000000000000650c00001f000000120394f0000000000300000006000000"
    "060000004c000000010000005800000000000000580000000100000070000000000000000c000000"
    "130000001f0000000800000006000000000000007000000000000000000000000100000000000000"
    "00000000060000000000000006000000ffffffff00240000000000005d0c00000d00000012039467"
    "000000006b65726e656c31352e302e376e7670747836342d6e76696469612d637564613c73747269"
    "6e673e0000000000",
}

MINIMAL_NVVMIR_CACHE = {}


@pytest.fixture(params=MINIMAL_NVVMIR_FIXTURE_PARAMS)
def minimal_nvvmir(request):
    for pass_counter in range(2):
        nvvmir = MINIMAL_NVVMIR_CACHE.get(request.param, -1)
        if nvvmir != -1:
            if nvvmir is None:
                pytest.skip(f"UNAVAILABLE: {request.param}")
            return nvvmir
        if pass_counter:
            raise AssertionError("This code path is meant to be unreachable.")
        # Build cache entries, then try again (above).
        major, minor, debug_major, debug_minor = nvvm.ir_version()
        txt = MINIMAL_NVVMIR_TXT % (major, debug_major)
        if llvmlite_binding is None:
            bitcode_dynamic = None
        else:
            bitcode_dynamic = llvmlite_binding.parse_assembly(txt.decode()).as_bitcode()
        bitcode_static = MINIMAL_NVVMIR_BITCODE_STATIC.get((major, debug_major))
        if bitcode_static is not None:
            bitcode_static = binascii.unhexlify(bitcode_static)
        MINIMAL_NVVMIR_CACHE["txt"] = txt
        MINIMAL_NVVMIR_CACHE["bitcode_dynamic"] = bitcode_dynamic
        MINIMAL_NVVMIR_CACHE["bitcode_static"] = bitcode_static
        if bitcode_static is None:
            if bitcode_dynamic is None:
                raise RuntimeError("Please `pip install llvmlite` to generate `bitcode_static` (see PR #443)")
            bitcode_hex = binascii.hexlify(bitcode_dynamic).decode("ascii")
            print("\n\nMINIMAL_NVVMIR_BITCODE_STATIC = { # PLEASE ADD TO test_nvvm.py")
            print(f"    ({major}, {debug_major}):  # (major, debug_major)")
            lines = textwrap.wrap(bitcode_hex, width=80)
            for line in lines[:-1]:
                print(f'    "{line}"')
            print(f'    "{lines[-1]}",')
            print("}\n", flush=True)


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
def test_compile_program_with_minimal_nvvm_ir(minimal_nvvmir, options):
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
def test_verify_program_with_minimal_nvvm_ir(minimal_nvvmir, options):
    with nvvm_program() as prog:
        nvvm.add_module_to_program(prog, minimal_nvvmir, len(minimal_nvvmir), "FileNameHere.ll")
        nvvm.verify_program(prog, len(options), options)
