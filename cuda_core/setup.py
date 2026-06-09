# SPDX-FileCopyrightText: Copyright (c) 2024-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

import os
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


def _build_aoti_shim_lib(compiler):
    # Reuse setuptools' initialized MSVC compiler instead of rediscovering
    # lib.exe separately in the build backend.
    lib_exe = getattr(compiler, "lib", None)
    if not lib_exe:
        raise RuntimeError("MSVC compiler did not expose lib.exe after initialization.")

    _AOTI_SHIM_LIB_FILE.parent.mkdir(exist_ok=True)
    compiler.spawn(
        [
            lib_exe,
            f"/DEF:{_AOTI_SHIM_DEF_FILE}",
            f"/OUT:{_AOTI_SHIM_LIB_FILE}",
            "/MACHINE:X64",
        ]
    )
    return str(_AOTI_SHIM_LIB_FILE)


class build_ext(_build_ext):  # noqa: N801
    def _configure_windows_tensor_bridge(self):
        if os.name != "nt" or getattr(self.compiler, "compiler_type", None) != "msvc":
            return

        # _tensor_bridge imports AOTI symbols from torch_cpu.dll, which on
        # Windows requires a stub import library for the MSVC linker.
        for ext in self.extensions:
            if ext.name != _TENSOR_BRIDGE_EXT_NAME:
                continue

            _ensure_compiler_initialized(self.compiler, self.plat_name)
            shim_lib = _build_aoti_shim_lib(self.compiler)
            link_args = list(ext.extra_link_args or [])
            if shim_lib not in link_args:
                ext.extra_link_args = [*link_args, shim_lib]
            return

        raise RuntimeError(f"Failed to find extension {_TENSOR_BRIDGE_EXT_NAME!r} for Windows build.")

    def build_extensions(self):
        self.parallel = nthreads
        self._configure_windows_tensor_bridge()
        super().build_extensions()


class build_py(_build_py):  # noqa: N801
    def finalize_options(self):
        super().finalize_options()
        if coverage_mode:
            self.package_data.setdefault("", [])
            self.package_data[""] += ["*.pxi", "*.pyx", "*.cpp"]


setup(
    ext_modules=build_hooks._extensions,
    cmdclass={
        "build_ext": build_ext,
        "build_py": build_py,
    },
    zip_safe=False,
)
