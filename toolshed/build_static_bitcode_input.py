#!/usr/bin/env python3

# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES.
# SPDX-License-Identifier: LicenseRef-NVIDIA-SOFTWARE-LICENSE

"""
Helper to produce static bitcode input for test_nvvm.py.

Usage:
    python toolshed/build_static_bitcode_input.py

It will print a ready-to-paste MINIMAL_NVVMIR_BITCODE_STATIC entry for the
current NVVM IR version detected at runtime.
"""

import binascii
import os
import sys
import textwrap

import llvmlite.binding  # HINT: pip install llvmlite
from cuda.bindings import nvvm


def get_minimal_nvvmir_txt_template():
    cuda_bindings_tests_dir = os.path.normpath("cuda_bindings/tests")
    assert os.path.isdir(cuda_bindings_tests_dir), (
        "Please run this helper script from the cuda-python top-level directory."
    )
    sys.path.insert(0, os.path.abspath(cuda_bindings_tests_dir))
    import test_nvvm

    return test_nvvm.MINIMAL_NVVMIR_TXT_TEMPLATE


def main():
    major, _minor, debug_major, _debug_minor = nvvm.ir_version()
    txt = get_minimal_nvvmir_txt_template() % (major, debug_major)
    bitcode_dynamic = llvmlite.binding.parse_assembly(txt.decode()).as_bitcode()
    bitcode_hex = binascii.hexlify(bitcode_dynamic).decode("ascii")
    print("\n\nMINIMAL_NVVMIR_BITCODE_STATIC = { # PLEASE ADD TO test_nvvm.py")
    print(f"    ({major}, {debug_major}):  # (major, debug_major)")
    lines = textwrap.wrap(bitcode_hex, width=80)
    for line in lines[:-1]:
        print(f'    "{line}"')
    print(f'    "{lines[-1]}",')
    print("}\n", flush=True)
    print()


if __name__ == "__main__":
    assert len(sys.argv) == 1, "This helper script does not take any arguments."
    main()
