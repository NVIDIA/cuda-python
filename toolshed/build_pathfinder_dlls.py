#!/usr/bin/env python3

# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

# Input for this script: .txt files generated with:
# for exe in *.exe; do 7z l $exe > "${exe%.exe}.txt"; done

# The output of this script is expected to be usable as-is.

import collections
import sys
from pathlib import Path

# ATTENTION: Ambiguous shorter names need to appear after matching longer names
#            (e.g. "cufft" after "cufftw")
LIBNAMES_IN_SCOPE_OF_CUDA_PATHFINDER = (
    "nvJitLink",
    "nvrtc",
    "nvvm",
    "cudart",
    "nvfatbin",
    "cublasLt",
    "cublas",
    "cufftw",
    "cufft",
    "curand",
    "cusolverMg",
    "cusolver",
    "cusparse",
    "nppc",
    "nppial",
    "nppicc",
    "nppidei",
    "nppif",
    "nppig",
    "nppim",
    "nppist",
    "nppisu",
    "nppitc",
    "npps",
    "nvblas",
    "nvjpeg",
)


def is_suppressed_dll(libname, dll):
    if libname == "cudart":
        if dll.startswith("cudart32_"):
            return True
        if dll == "cudart64_65.dll":
            # PhysX/files/Common/cudart64_65.dll from CTK 6.5, but shipped with CTK 12.0-12.9
            return True
        if dll == "cudart64_101.dll":
            # GFExperience.NvStreamSrv/amd64/server/cudart64_101.dll from CTK 10.1, but shipped with CTK 12.0-12.6
            return True
    elif libname == "nvrtc":
        if dll.endswith(".alt.dll"):
            return True
        if dll.startswith("nvrtc-builtins"):
            return True
    elif libname == "nvvm" and dll == "nvvm32.dll":
        return True
    return False


def run(args):
    dlls_from_files = set()
    for filename in args:
        lines_iter = iter(Path(filename).read_text().splitlines())
        for line in lines_iter:
            if line.startswith("-------------------"):
                break
        else:
            raise RuntimeError("------------------- NOT FOUND")
        for line in lines_iter:
            if line.startswith("-------------------"):
                break
            assert line[52] == " ", line
            assert line[53] != " ", line
            path = line[53:]
            if path.endswith(".dll"):
                dll = path.rsplit("/", 1)[1]
                dlls_from_files.add(dll)
        else:
            raise RuntimeError("------------------- NOT FOUND")

    print("DLLs in scope of cuda.pathfinder")
    print("================================")
    dlls_in_scope = set()
    dlls_by_libname = collections.defaultdict(list)
    suppressed_dlls = set()
    for libname in LIBNAMES_IN_SCOPE_OF_CUDA_PATHFINDER:
        for dll in sorted(dlls_from_files):
            if dll not in dlls_in_scope and dll.startswith(libname):
                if is_suppressed_dll(libname, dll):
                    suppressed_dlls.add(dll)
                else:
                    dlls_by_libname[libname].append(dll)
                dlls_in_scope.add(dll)
    for libname, dlls in sorted(dlls_by_libname.items()):
        print(f'"{libname}": (')
        for dll in dlls:
            print(f'    "{dll}",')
        print("),")
    print()

    print("Suppressed DLLs")
    print("===============")
    for dll in sorted(suppressed_dlls):
        print(dll)
    print()

    print("DLLs out of scope")
    print("=================")
    for dll in sorted(dlls_from_files - dlls_in_scope):
        print(dll)
    print()


if __name__ == "__main__":
    run(args=sys.argv[1:])
