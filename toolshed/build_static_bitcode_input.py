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
import sys
import textwrap

try:
    import llvmlite.binding
except Exception:
    sys.exit("HINT: pip install llvmlite")

from cuda.bindings import nvvm

# Keep this template in sync with test_nvvm.py
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


def main():
    major, _minor, debug_major, _debug_minor = nvvm.ir_version()
    txt = MINIMAL_NVVMIR_TXT % (major, debug_major)
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
