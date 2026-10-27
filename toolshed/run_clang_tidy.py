# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

"""Run clang-tidy on the C++ sources with the compile flags the build uses.

The C++ under ``cuda_core/cuda/core/_cpp`` is compiled by setuptools (see
``cuda_core/build_hooks.py``), so there is no ``compile_commands.json``. The
include directories and C++ standard below mirror that build and must be kept
in step with it; toolshed/check_clang_tidy_flags.py verifies that in CI. The ``CUDA_CORE_*`` macros that ``rt/versions.hpp`` takes from
the build are omitted: it derives the CUDA major from ``cuda.h`` without them,
and they only gate ``#error`` checks.

The CUDA headers come from ``CUDA_PATH`` or ``CUDA_HOME`` (as in the build), or,
with ``--cuda-wheel``, from the installed ``nvidia-cuda-runtime`` and
``nvidia-cuda-nvrtc`` wheels, which ignores both variables so the result does
not depend on the machine. The CPython headers, which setuptools adds
implicitly, come from the running interpreter. The ``clang-tidy`` executable is
expected on ``PATH``.

Usage: run_clang_tidy.py [--cuda-wheel] [clang-tidy options] FILE [FILE ...]
"""

import argparse
import importlib.util
import os
import shutil
import subprocess
import sys
import sysconfig
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
CUDA_CORE_DIR = REPO_ROOT / "cuda_core"

# c++17: required by structured bindings and if constexpr in cuda/core/_cpp/.
CXX_STANDARD = 17

# Include directories of the build, relative to cuda_core/. The sources under
# _cpp/ include nothing from CCCL, so the submodule is not needed.
INCLUDE_DIRS = ("cuda/core", "cuda/core/_include", "cuda/core/_cpp")

# Headers are only analyzed through the .cpp files that include them, so a
# changed header alone does not trigger a run. These are the translation units.
SOURCE_SUFFIXES = {".cpp", ".cc", ".cxx"}


def _cuda_wheel_path() -> str:
    # The wheel installs the toolkit headers to site-packages/nvidia/cu<major>/include.
    spec = importlib.util.find_spec("nvidia")
    if spec is not None and spec.submodule_search_locations is not None:
        for base in spec.submodule_search_locations:
            for root in sorted(Path(base).glob("cu*")):
                if (root / "include" / "cuda.h").is_file():
                    return str(root)
    sys.exit("error: --cuda-wheel needs the nvidia-cuda-runtime package installed")


def _cuda_path(use_wheel: bool) -> str:
    if use_wheel:
        return _cuda_wheel_path()
    for var in ("CUDA_PATH", "CUDA_HOME"):
        root = os.environ.get(var)
        if root and (Path(root) / "include" / "cuda.h").is_file():
            return root
    sys.exit("error: set CUDA_PATH or CUDA_HOME to a CUDA Toolkit with include/cuda.h")


def _compile_flags(use_wheel: bool) -> list[str]:
    cuda_path = _cuda_path(use_wheel)
    flags = ["-x", "c++", f"-std=c++{CXX_STANDARD}"]
    flags += [f"-I{CUDA_CORE_DIR / d}" for d in INCLUDE_DIRS]
    # System includes keep third-party headers out of the diagnostics.
    flags.append(f"-isystem{Path(cuda_path) / 'include'}")
    flags.append(f"-isystem{sysconfig.get_paths()['include']}")
    return flags


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=__doc__.splitlines()[0],
        usage="%(prog)s [--cuda-wheel] [clang-tidy options] FILE [FILE ...]",
        epilog="Any other option is passed through to clang-tidy.",
    )
    parser.add_argument(
        "--cuda-wheel",
        action="store_true",
        help="take the CUDA headers from the nvidia-cuda-* wheels instead of CUDA_PATH/CUDA_HOME",
    )
    parser.add_argument("files", nargs="*", metavar="FILE", help="source files to analyze")
    return parser


def main(argv: list[str]) -> int:
    clang_tidy = shutil.which("clang-tidy")
    if clang_tidy is None:
        sys.exit("error: clang-tidy not found on PATH")

    args, options = _parser().parse_known_args(argv)
    files = [f for f in args.files if Path(f).suffix in SOURCE_SUFFIXES]
    if not files:
        return 0

    cmd = [clang_tidy, "--quiet", *options, *files, "--", *_compile_flags(args.cuda_wheel)]
    return subprocess.run(cmd, cwd=REPO_ROOT, check=False).returncode  # noqa: S603


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
