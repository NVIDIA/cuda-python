# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Shared builder for the cuda.bindings and cuda.core Cython test extensions."""

from __future__ import annotations

import argparse
import os
import shutil
import sys
from collections.abc import MutableMapping, Sequence
from pathlib import Path


def _bindings_source_root() -> Path:
    import cuda.bindings

    # cuda.bindings.__file__ -> .../<root>/cuda/bindings/__init__.py
    root = Path(cuda.bindings.__file__).resolve().parents[2]
    if not (root / "cuda" / "bindings").is_dir():
        raise RuntimeError(
            f"cuda.bindings source tree not found at {root}; pixi-build editable install layout may have changed."
        )
    return root


def _output_directory(script_dir: Path, value: str) -> Path:
    project_root = script_dir.parents[1]
    output_root = project_root / ".moon-out"
    requested = Path(value)
    output = Path(os.path.abspath(requested if requested.is_absolute() else project_root.parent / requested))
    if output_root not in output.parents:
        raise ValueError(f"output must be below {output_root}: {output}")

    current = output
    while current != project_root:
        if current.is_symlink():
            raise ValueError(f"output path must not traverse a symlink: {current}")
        current = current.parent

    if output.exists():
        if not output.is_dir():
            raise ValueError(f"refusing to replace non-directory output: {output}")
        shutil.rmtree(output)
    output.mkdir(parents=True)
    return output


def _set_compiler_include_paths(
    include_dirs: Sequence[Path],
    *,
    environ: MutableMapping[str, str] | None = None,
    platform_name: str | None = None,
) -> None:
    environment = os.environ if environ is None else environ
    platform_name = os.name if platform_name is None else platform_name
    if platform_name == "nt":
        flags = " ".join(f'/I"{path}"' for path in include_dirs)
        environment["CL"] = " ".join(part for part in (flags, environment.get("CL", "")) if part)
    else:
        paths = [str(path) for path in include_dirs]
        if existing := environment.get("CPLUS_INCLUDE_PATH"):
            paths.append(existing)
        environment["CPLUS_INCLUDE_PATH"] = ":".join(paths)


def _configure_compiler_includes(script_dir: Path, *, include_core_headers: bool) -> None:
    cuda_root = os.environ.get("CUDA_HOME") or os.environ.get("CUDA_PATH")
    if not cuda_root:
        raise RuntimeError("CUDA_HOME or CUDA_PATH must identify the CUDA Toolkit")

    include_dirs = []
    if include_core_headers:
        include_dirs.append(script_dir.parents[1] / "cuda" / "core" / "_include")
    include_dirs.append(Path(cuda_root) / "include")

    missing = [path for path in include_dirs if not path.is_dir()]
    if missing:
        raise RuntimeError(f"required include directory does not exist: {missing[0]}")
    _set_compiler_include_paths(include_dirs)


def build_cython_tests(
    *,
    script_file: str,
    distribution_name: str,
    include_core_headers: bool = False,
    nthreads: int | None = None,
) -> None:
    """Build all ``test_*.pyx`` siblings of *script_file*."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir")
    args = parser.parse_args()

    script_dir = Path(script_file).resolve().parent
    output = _output_directory(script_dir, args.output_dir) if args.output_dir else None
    _configure_compiler_includes(script_dir, include_core_headers=include_core_headers)

    # Use short sibling names. Appending an absolute checkout path under
    # build/temp can exceed Windows' path limit.
    os.chdir(script_dir)
    pyx_files = sorted(path.name for path in script_dir.glob("test_*.pyx"))
    if not pyx_files:
        raise SystemExit(f"no test_*.pyx files under {script_dir}")

    from Cython.Build import cythonize
    from setuptools import setup

    cython_options: dict[str, object] = {
        "language_level": 3,
        "include_path": [str(_bindings_source_root())],
        "compiler_directives": {"freethreading_compatible": True},
    }
    if nthreads is not None:
        cython_options["nthreads"] = nthreads

    # Cython otherwise writes generated C/C++ beside the .pyx inputs. Moon
    # builds keep all generated sources and compiler intermediates in outputs.
    if output is None:
        ext_modules = cythonize(pyx_files, **cython_options)
    else:
        cython_build = output / ".cython-build"
        ext_modules = cythonize(pyx_files, build_dir=str(cython_build), **cython_options)

    sys.argv = [sys.argv[0], "build_ext"]
    if output is None:
        sys.argv.append("--inplace")
    else:
        build_temp = output / ".build-temp"
        sys.argv.extend(["--build-lib", str(output), "--build-temp", str(build_temp)])
    setup(name=distribution_name, ext_modules=ext_modules)

    if output is None:
        return

    for intermediate in (build_temp, cython_build):
        if intermediate.exists():
            shutil.rmtree(intermediate)
    for source in pyx_files:
        matches = [
            path
            for pattern in (f"{Path(source).stem}*.so", f"{Path(source).stem}*.pyd", f"{Path(source).stem}*.dylib")
            for path in output.glob(pattern)
            if path.is_file()
        ]
        if len(matches) != 1:
            raise RuntimeError(f"expected one extension for {source} in {output}, found {len(matches)}")
