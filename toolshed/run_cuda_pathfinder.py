# Copyright (c) 2025, NVIDIA CORPORATION & AFFILIATES. ALL RIGHTS RESERVED.
#
# SPDX-License-Identifier: Apache-2.0

import sys
import traceback

from cuda import pathfinder


def run(args):
    if args:
        libnames = args
    else:
        libnames = pathfinder.SUPPORTED_NVIDIA_LIBNAMES

    for libname in libnames:
        print(f"{libname=}")
        try:
            loaded_dl = pathfinder.load_nvidia_dynamic_lib(libname)
        except Exception:
            print(f"EXCEPTION for {libname=}:")
            traceback.print_exc(file=sys.stdout)
        else:
            print(f"    {loaded_dl.abs_path=!r}")
            print(f"    {loaded_dl.was_already_loaded_from_elsewhere=!r}")
        print()


if __name__ == "__main__":
    run(args=sys.argv[1:])
