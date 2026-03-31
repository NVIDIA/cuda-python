# SPDX-FileCopyrightText: Copyright (c) 2026-2027 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: LicenseRef-NVIDIA-SOFTWARE-LICENSE

import contextlib
from typing import Sequence

_PRECHECK_NVVM_IR = """target triple = "nvptx64-unknown-cuda"
target datalayout = "e-p:64:64:64-i1:8:8-i8:8:8-i16:16:16-i32:32:32-i64:64:64-i128:128:128-f32:32:32-f64:64:64-v16:16:16-v32:32:32-v64:64:64-v128:128:128-n16:32:64"

define void @dummy_kernel() {{
entry:
  ret void
}}

!nvvm.annotations = !{{!0}}
!0 = !{{void ()* @dummy_kernel, !"kernel", i32 1}}

!nvvmir.version = !{{!1}}
!1 = !{{i32 {major}, i32 {minor}, i32 {debug_major}, i32 {debug_minor}}}
"""


def check_nvvm_options(options: Sequence[bytes]) -> bool:
    """
    Abstracted from https://github.com/NVIDIA/numba-cuda/pull/681

    Check if the specified options are supported by the current libNVVM version.

    The options are a list of bytes, each representing a compiler option.

    If the test program fails to compile, the options are not supported and False
    is returned.

    If the test program compiles successfully, True is returned.

    cuda.bindings.nvvm returns exceptions instead of return codes.

    Parameters
    ----------
    options : Sequence[bytes]
        List of compiler options as bytes (e.g., [b"-arch=compute_90", b"-g"]).

    Returns
    -------
    bool
        True if the options are supported, False otherwise.

    Examples
    --------
    >>> from cuda.bindings.utils import check_nvvm_options
    >>> check_nvvm_options([b"-arch=compute_90", b"-g"])
    True
    >>> check_nvvm_options([b"-arch=compute_90", b"-numba-debug"])
    True  # if -numba-debug is supported by the installed libNVVM
    """
    try:
        from cuda.bindings import nvvm
        from cuda.bindings._internal.nvvm import _inspect_function_pointer

        if _inspect_function_pointer("__nvvmCreateProgram") == 0:
            return False
    except Exception:
        return False

    program = None
    try:
        program = nvvm.create_program()

        major, minor, debug_major, debug_minor = nvvm.ir_version()
        precheck_ir = _PRECHECK_NVVM_IR.format(
            major=major,
            minor=minor,
            debug_major=debug_major,
            debug_minor=debug_minor,
        )
        precheck_ir_bytes = precheck_ir.encode("utf-8")
        nvvm.add_module_to_program(
            program,
            precheck_ir_bytes,
            len(precheck_ir_bytes),
            "precheck.ll",
        )

        options_list = [opt.decode("utf-8") if isinstance(opt, bytes) else opt for opt in options]
        nvvm.verify_program(program, len(options_list), options_list)
        nvvm.compile_program(program, len(options_list), options_list)
    except Exception:
        return False
    finally:
        if program is not None:
            with contextlib.suppress(Exception):
                nvvm.destroy_program(program)
    return True
