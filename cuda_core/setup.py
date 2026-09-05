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
_CUDART_SHIM_DEF_FILE = _ROOT_DIR / "build" / "cudart_shim.def"
_CUDART_SHIM_LIB_FILE = _ROOT_DIR / "build" / "cudart_shim.lib"
_TENSOR_MAP_CCCL_EXT_NAME = "cuda.core._tensor_map_cccl"


def _ensure_compiler_initialized(compiler, plat_name):
    initialize = getattr(compiler, "initialize", None)
    if callable(initialize) and not getattr(compiler, "initialized", False):
        if plat_name is None:
            initialize()
        else:
            initialize(plat_name)


def _build_stub_import_lib(compiler, plat_name, def_file, lib_file):
    # Reuse setuptools' initialized MSVC compiler instead of rediscovering
    # lib.exe separately in the build backend.
    lib_exe = getattr(compiler, "lib", None)
    if not lib_exe:
        raise RuntimeError("MSVC compiler did not expose lib.exe after initialization.")

    lib_file.parent.mkdir(exist_ok=True)
    machine = {
        "win-amd64": "X64",
        "win-arm64": "ARM64",
    }.get(plat_name, "X64")
    compiler.spawn(
        [
            lib_exe,
            f"/DEF:{def_file}",
            f"/OUT:{lib_file}",
            f"/MACHINE:{machine}",
        ]
    )
    return str(lib_file)


def _write_cudart_shim_def():
    """Emit the .def naming the cudart symbols _tensor_map_cccl resolves lazily.

    CCCL's exception formatting calls cudaGetErrorString, so the extension is
    left with that one unresolved cudart symbol. Unlike aoti_shim.def this file
    is generated, because the DLL name carries the CUDA major version.
    """
    cuda_major = build_hooks._determine_cuda_major_version()
    _CUDART_SHIM_DEF_FILE.parent.mkdir(exist_ok=True)
    _CUDART_SHIM_DEF_FILE.write_text(
        f"LIBRARY cudart64_{cuda_major}.dll\nEXPORTS\n    cudaGetErrorString\n", encoding="utf-8"
    )
    return _CUDART_SHIM_DEF_FILE


class build_ext(_build_ext):  # noqa: N801
    def finalize_options(self):
        super().finalize_options()
        # A cu13 .so in the source tree looks perfectly fresh to a cu12 build;
        # see build_hooks._check_build_major().
        if build_hooks.force_build_ext:
            self.force = True

    def _attach_stub_import_lib(self, ext_name, def_file, lib_file):
        for ext in self.extensions:
            if ext.name != ext_name:
                continue

            _ensure_compiler_initialized(self.compiler, self.plat_name)
            shim_lib = _build_stub_import_lib(self.compiler, self.plat_name, def_file, lib_file)
            link_args = list(ext.extra_link_args or [])
            if shim_lib not in link_args:
                ext.extra_link_args = [*link_args, shim_lib]
            return

        raise RuntimeError(f"Failed to find extension {ext_name!r} for Windows build.")

    def _configure_windows_stub_imports(self):
        if os.name != "nt" or getattr(self.compiler, "compiler_type", None) != "msvc":
            return

        # _tensor_bridge imports AOTI symbols from torch_cpu.dll and
        # _tensor_map_cccl imports cudaGetErrorString from cudart. Both resolve
        # at runtime from a DLL that is already loaded by the time the
        # extension is imported, but the MSVC linker still demands an import
        # library, so synthesize a minimal one from a .def file.
        self._attach_stub_import_lib(_TENSOR_BRIDGE_EXT_NAME, _AOTI_SHIM_DEF_FILE, _AOTI_SHIM_LIB_FILE)
        self._attach_stub_import_lib(_TENSOR_MAP_CCCL_EXT_NAME, _write_cudart_shim_def(), _CUDART_SHIM_LIB_FILE)

    def build_extensions(self):
        self.parallel = nthreads
        self._configure_windows_stub_imports()
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
