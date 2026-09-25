# SPDX-FileCopyrightText: Copyright (c) 2024-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

import contextlib
import os
from concurrent.futures import ThreadPoolExecutor
from distutils.ccompiler import CCompiler
from pathlib import Path

import build_hooks  # our build backend
from setuptools import setup
from setuptools.command.build_ext import build_ext as _build_ext
from setuptools.command.build_py import build_py as _build_py

nthreads = int(os.environ.get("CUDA_PYTHON_PARALLEL_LEVEL", os.cpu_count() // 2))
coverage_mode = bool(int(os.environ.get("CUDA_PYTHON_COVERAGE", "0")))
_ROOT_DIR = Path(__file__).resolve().parent
_AOTI_SHIM_DEF_FILE = _ROOT_DIR / "cuda" / "core" / "_include" / "aoti_shim.def"
_AOTI_SHIM_LIB_FILE = _ROOT_DIR / "build" / "aoti_shim.lib"
_TENSOR_BRIDGE_EXT_NAME = "cuda.core._tensor_bridge"


def _ensure_compiler_initialized(compiler, plat_name):
    initialize = getattr(compiler, "initialize", None)
    if callable(initialize) and not getattr(compiler, "initialized", False):
        if plat_name is None:
            initialize()
        else:
            initialize(plat_name)


def _build_aoti_shim_lib(compiler, plat_name):
    # Reuse setuptools' initialized MSVC compiler instead of rediscovering
    # lib.exe separately in the build backend.
    lib_exe = getattr(compiler, "lib", None)
    if not lib_exe:
        raise RuntimeError("MSVC compiler did not expose lib.exe after initialization.")

    _AOTI_SHIM_LIB_FILE.parent.mkdir(exist_ok=True)
    machine = {
        "win-amd64": "X64",
        "win-arm64": "ARM64",
    }.get(plat_name, "X64")
    compiler.spawn(
        [
            lib_exe,
            f"/DEF:{_AOTI_SHIM_DEF_FILE}",
            f"/OUT:{_AOTI_SHIM_LIB_FILE}",
            f"/MACHINE:{machine}",
        ]
    )
    return str(_AOTI_SHIM_LIB_FILE)


class build_ext(_build_ext):  # noqa: N801
    def finalize_options(self):
        super().finalize_options()
        # A cu13 .so in the source tree looks perfectly fresh to a cu12 build;
        # see build_hooks._check_build_major().
        if build_hooks.force_build_ext:
            self.force = True

    def _configure_windows_tensor_bridge(self):
        if os.name != "nt" or getattr(self.compiler, "compiler_type", None) != "msvc":
            return

        # _tensor_bridge imports AOTI symbols from torch_cpu.dll, which on
        # Windows requires a stub import library for the MSVC linker.
        for ext in self.extensions:
            if ext.name != _TENSOR_BRIDGE_EXT_NAME:
                continue

            _ensure_compiler_initialized(self.compiler, self.plat_name)
            shim_lib = _build_aoti_shim_lib(self.compiler, self.plat_name)
            link_args = list(ext.extra_link_args or [])
            if shim_lib not in link_args:
                ext.extra_link_args = [*link_args, shim_lib]
            return

        raise RuntimeError(f"Failed to find extension {_TENSOR_BRIDGE_EXT_NAME!r} for Windows build.")

    @contextlib.contextmanager
    def _parallel_source_compilation(self):
        """Compile the sources of every extension through one shared thread pool.

        setuptools runs extensions in parallel (self.parallel) but compiles the
        sources of one extension serially, so a multi-source extension such as
        cuda.core._rt (a dozen .cpp files) becomes the critical path. This
        mirrors CCompiler.compile() and fans its per-object _compile() calls out
        to a pool shared by all extensions, so at most `nthreads` compiler
        processes run at once. It applies only to compilers that still use
        CCompiler.compile(), which drives the per-object _compile() hook (the
        Unix family); MSVC overrides compile() wholesale and keeps the stock path.
        """
        compiler = self.compiler
        if nthreads <= 1 or type(compiler).compile is not CCompiler.compile:
            yield
            return
        stock_compile = compiler.compile
        with ThreadPoolExecutor(max_workers=nthreads) as pool:

            def compile(
                sources,
                output_dir=None,
                macros=None,
                include_dirs=None,
                debug=0,
                extra_preargs=None,
                extra_postargs=None,
                depends=None,
            ):
                macros, objects, extra_postargs, pp_opts, build = compiler._setup_compile(
                    output_dir, macros, include_dirs, sources, depends, extra_postargs
                )
                cc_args = compiler._get_cc_args(pp_opts, debug, extra_preargs)

                def compile_one(obj):
                    try:
                        src, ext = build[obj]
                    except KeyError:
                        return  # up to date
                    compiler._compile(obj, src, ext, cc_args, extra_postargs, pp_opts)

                list(pool.map(compile_one, objects))  # re-raises the first failure
                return objects

            compiler.compile = compile
            try:
                yield
            finally:
                compiler.compile = stock_compile

    def build_extensions(self):
        self.parallel = nthreads
        self._configure_windows_tensor_bridge()
        with self._parallel_source_compilation():
            super().build_extensions()
        build_hooks.record_build_major()


class build_py(_build_py):  # noqa: N801
    def finalize_options(self):
        super().finalize_options()
        if coverage_mode:
            self.package_data.setdefault("", [])
            self.package_data[""] += ["*.pxi", "*.pyx", "*.cpp"]


# Guarded so tests can import the command classes above. setuptools always
# runs this file as __main__, so real builds are unaffected.
if __name__ == "__main__":
    setup(
        ext_modules=build_hooks._extensions,
        cmdclass={
            "build_ext": build_ext,
            "build_py": build_py,
        },
        zip_safe=False,
    )
