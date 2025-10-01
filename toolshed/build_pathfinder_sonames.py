#!/usr/bin/env python3

# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

# Input for this script:
# output of toolshed/find_sonames.sh

# The output of this script is expected to be usable as-is.

import sys
from pathlib import Path

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
    assert len(args) == 1, "output-of-find_sonames.sh"

    sonames_from_file = set()
    for line in Path(args[0]).read_text().splitlines():
        flds = line.split()
        assert len(flds) == 3, flds
        if flds[-1] != "SONAME_NOT_SET":
            sonames_from_file.add(flds[-1])

    print("SONAMEs in scope of cuda.pathfinder")
    print("===================================")
    sonames_in_scope = set()
    for libname in sorted(LIBNAMES_IN_SCOPE_OF_CUDA_PATHFINDER):
        print(f'"{libname}": (')
        lib_so = "lib" + libname + ".so"
        for soname in sorted(sonames_from_file):
            if soname.startswith(lib_so):
                sonames_in_scope.add(soname)
                print(f'    "{soname}",')
        print("),")
    print()

    print("SONAMEs out of scope")
    print("====================")
    for soname in sorted(sonames_from_file - sonames_in_scope):
        print(soname)
    print()


if __name__ == "__main__":
    run(args=sys.argv[1:])
