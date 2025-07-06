#!/usr/bin/env python3

# Copyright (c) 2025, NVIDIA CORPORATION & AFFILIATES. ALL RIGHTS RESERVED.
#
# SPDX-License-Identifier: Apache-2.0

# Input for this script: .txt files generated with:
# for exe in *.exe; do 7z l $exe > "${exe%.exe}.txt"; done

# The output of this script
# requires obvious manual edits to remove duplicates and unwanted dlls.
# TODO: filter out cudart32_*.dll, nvvm32.dll

import sys

LIBNAMES_IN_SCOPE_OF_CUDA_PATHFINDER = (
    "nvJitLink",
    "nvrtc",
    "nvvm",
    "cudart",
    "nvfatbin",
    "cublas",
    "cublasLt",
    "cufft",
    "cufftw",
    "curand",
    "cusolver",
    "cusolverMg",
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
    "cufile",
    "cufile_rdma",
    "nvjpeg",
)


def run(args):
    dlls_from_files = set()
    for filename in args:
        lines_iter = iter(open(filename).read().splitlines())
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
    for libname in sorted(LIBNAMES_IN_SCOPE_OF_CUDA_PATHFINDER):
        print(f'"{libname}": (')
        for dll in sorted(dlls_from_files):
            if dll.startswith(libname):
                dlls_in_scope.add(dll)
                print(f'    "{dll}",')
        print("),")
    print()

    print("DLLs out of scope")
    print("=================")
    for dll in sorted(dlls_from_files - dlls_in_scope):
        print(dll)
    print()


if __name__ == "__main__":
    run(args=sys.argv[1:])
