# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Source-level regression tests for FIPS-safe program-cache hashing.

These tests load the leaf program-cache modules directly from source with
small stubs, so they can run without importing the full built ``cuda.core``
package. Run with ``--noconftest`` when the compiled extensions are not
available:

    pytest cuda_core/tests/test_program_cache_fips.py --noconftest
"""

from __future__ import annotations

import importlib.util
import sys
import types
from pathlib import Path


def _load_program_cache_modules(monkeypatch):
    cuda_pkg = types.ModuleType("cuda")
    cuda_pkg.__path__ = []
    core_pkg = types.ModuleType("cuda.core")
    core_pkg.__path__ = []
    utils_pkg = types.ModuleType("cuda.core.utils")
    utils_pkg.__path__ = []
    cache_pkg = types.ModuleType("cuda.core.utils._program_cache")
    cache_pkg.__path__ = []
    utils_internal_pkg = types.ModuleType("cuda.core._utils")
    utils_internal_pkg.__path__ = []

    module_mod = types.ModuleType("cuda.core._module")

    class ObjectCode:
        pass

    module_mod.ObjectCode = ObjectCode

    program_mod = types.ModuleType("cuda.core._program")

    class ProgramOptions:
        def __init__(self, **kwargs):
            self.arch = kwargs.pop("arch", "sm_80")
            self.name = kwargs.pop("name", "default_program")
            for key, value in kwargs.items():
                setattr(self, key, value)

        def as_bytes(self, backend, target_type):
            return [
                f"backend={backend}".encode(),
                f"target_type={target_type}".encode(),
                f"arch={self.arch}".encode(),
                f"name={self.name}".encode(),
            ]

    program_mod.ProgramOptions = ProgramOptions

    cuda_utils_mod = types.ModuleType("cuda.core._utils.cuda_utils")
    cuda_utils_mod.driver = types.SimpleNamespace()
    cuda_utils_mod.handle_return = lambda result: result
    cuda_utils_mod.nvrtc = types.SimpleNamespace(nvrtcVersion=lambda: (13, 0))

    modules = {
        "cuda": cuda_pkg,
        "cuda.core": core_pkg,
        "cuda.core.utils": utils_pkg,
        "cuda.core.utils._program_cache": cache_pkg,
        "cuda.core._utils": utils_internal_pkg,
        "cuda.core._module": module_mod,
        "cuda.core._program": program_mod,
        "cuda.core._utils.cuda_utils": cuda_utils_mod,
    }
    for name, module in modules.items():
        monkeypatch.setitem(sys.modules, name, module)

    base = Path(__file__).parent.parent / "cuda" / "core" / "utils" / "_program_cache"

    def _load(name, filename):
        spec = importlib.util.spec_from_file_location(name, base / filename)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        monkeypatch.setitem(sys.modules, name, module)
        return module

    _load("cuda.core.utils._program_cache._abc", "_abc.py")
    keys_mod = _load("cuda.core.utils._program_cache._keys", "_keys.py")
    file_stream_mod = _load("cuda.core.utils._program_cache._file_stream", "_file_stream.py")
    return keys_mod, file_stream_mod, ProgramOptions


def test_make_program_cache_key_avoids_fips_blocked_blake2b(monkeypatch):
    import hashlib

    keys_mod, _file_stream_mod, ProgramOptions = _load_program_cache_modules(monkeypatch)

    def _blake2b_disabled(*args, **kwargs):
        raise ValueError("disabled for FIPS")

    monkeypatch.setattr(hashlib, "blake2b", _blake2b_disabled)

    key = keys_mod.make_program_cache_key(
        code="extern \"C\" __global__ void k() {}",
        code_type="c++",
        options=ProgramOptions(arch="sm_80"),
        target_type="cubin",
    )

    assert isinstance(key, bytes)
    assert len(key) == 32


def test_filestream_cache_path_hash_avoids_fips_blocked_blake2b(tmp_path, monkeypatch):
    import hashlib

    _keys_mod, file_stream_mod, _ProgramOptions = _load_program_cache_modules(monkeypatch)

    def _blake2b_disabled(*args, **kwargs):
        raise ValueError("disabled for FIPS")

    monkeypatch.setattr(hashlib, "blake2b", _blake2b_disabled)

    with file_stream_mod.FileStreamProgramCache(tmp_path / "fc") as cache:
        cache[b"my-key"] = b"payload"
        assert cache[b"my-key"] == b"payload"
