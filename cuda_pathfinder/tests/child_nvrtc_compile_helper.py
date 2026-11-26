# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Load NVRTC from the given path via pathfinder, compile a trivial program,
and print a JSON report including where libnvrtc-builtins was mapped from."""

import ctypes
import json
import os
import sys

from cuda.pathfinder._dynamic_libs.lib_descriptor import LIB_DESCRIPTORS
from cuda.pathfinder._dynamic_libs.load_dl_linux import load_with_abs_path


def _mapped_paths(basename_prefix: str) -> list[str]:
    paths = set()
    with open("/proc/self/maps") as maps:
        for line in maps:
            fields = line.split(maxsplit=5)
            if len(fields) == 6 and os.path.basename(fields[5].strip()).startswith(basename_prefix):
                paths.add(os.path.realpath(fields[5].strip()))
    return sorted(paths)


def main(nvrtc_path: str) -> None:
    load_with_abs_path(LIB_DESCRIPTORS["nvrtc"], nvrtc_path, "CUDA_PATH")
    nvrtc = ctypes.CDLL(nvrtc_path)

    prog = ctypes.c_void_p()
    status = nvrtc.nvrtcCreateProgram(ctypes.byref(prog), b"__global__ void k() {}\n", b"k.cu", 0, None, None)
    if status != 0:
        raise RuntimeError(f"nvrtcCreateProgram failed (status={status})")
    compile_status = nvrtc.nvrtcCompileProgram(prog, 0, None)
    log_size = ctypes.c_size_t()
    nvrtc.nvrtcGetProgramLogSize(prog, ctypes.byref(log_size))
    log = ctypes.create_string_buffer(log_size.value)
    nvrtc.nvrtcGetProgramLog(prog, log)
    nvrtc.nvrtcDestroyProgram(ctypes.byref(prog))

    print(
        json.dumps(
            {
                "compile_status": compile_status,
                "log": log.value.decode(errors="replace"),
                "builtins_paths": _mapped_paths("libnvrtc-builtins"),
            }
        )
    )


if __name__ == "__main__":
    main(sys.argv[1])
