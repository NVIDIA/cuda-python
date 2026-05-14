# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Source-level tests for checkpoint helper validation.

These tests load ``cuda/core/checkpoint.py`` directly from source with small
stub modules, so they can run without importing the full built ``cuda.core``
package. Run with ``--noconftest`` in environments that do not have the CUDA
extensions available:

    pytest cuda_core/tests/test_checkpoint_helpers.py --noconftest
"""

from __future__ import annotations

import importlib.util
import sys
import types
from pathlib import Path

import pytest


def _load_checkpoint_module(monkeypatch):
    cuda_pkg = types.ModuleType("cuda")
    cuda_pkg.__path__ = []
    core_pkg = types.ModuleType("cuda.core")
    core_pkg.__path__ = []
    utils_pkg = types.ModuleType("cuda.core._utils")
    utils_pkg.__path__ = []
    bindings_pkg = types.ModuleType("cuda.bindings")
    bindings_pkg.__path__ = []

    cuda_utils = types.ModuleType("cuda.core._utils.cuda_utils")
    cuda_utils.handle_return = lambda result: result

    version_mod = types.ModuleType("cuda.core._utils.version")
    version_mod.binding_version = lambda: (13, 0, 2)
    version_mod.driver_version = lambda: (12, 8, 0)

    typing_mod = types.ModuleType("cuda.core.typing")
    typing_mod.ProcessStateType = str

    driver_mod = types.ModuleType("cuda.bindings.driver")

    class CUuuid:
        def __init__(self, value):
            self.value = value

    class CUcheckpointGpuPair:
        def __init__(self):
            self.oldUuid = None
            self.newUuid = None

    class CUcheckpointRestoreArgs:
        def __init__(self):
            self.gpuPairs = None
            self.gpuPairsCount = 0

    driver_mod.CUuuid = CUuuid
    driver_mod.CUcheckpointGpuPair = CUcheckpointGpuPair
    driver_mod.CUcheckpointRestoreArgs = CUcheckpointRestoreArgs

    modules = {
        "cuda": cuda_pkg,
        "cuda.core": core_pkg,
        "cuda.core._utils": utils_pkg,
        "cuda.core._utils.cuda_utils": cuda_utils,
        "cuda.core._utils.version": version_mod,
        "cuda.core.typing": typing_mod,
        "cuda.bindings": bindings_pkg,
        "cuda.bindings.driver": driver_mod,
    }
    for name, module in modules.items():
        monkeypatch.setitem(sys.modules, name, module)

    checkpoint_path = Path(__file__).parent.parent / "cuda" / "core" / "checkpoint.py"
    spec = importlib.util.spec_from_file_location("cuda.core._checkpoint_test", checkpoint_path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module, driver_mod


def test_make_restore_args_rejects_non_uuid_values(monkeypatch):
    checkpoint, driver = _load_checkpoint_module(monkeypatch)

    with pytest.raises(TypeError, match="GPU UUID values must be CUDA UUID objects or UUID strings"):
        checkpoint._make_restore_args(driver, {"01234567-89ab-cdef-0123-456789abcdef": object()})


@pytest.mark.parametrize(
    "bad_uuid",
    [
        pytest.param("not-hex-uuid-0000-0000-000000000000", id="non_hex"),
        pytest.param("01234567-89ab-cdef-0123-456789abcde", id="short"),
    ],
)
def test_make_restore_args_rejects_invalid_uuid_strings(monkeypatch, bad_uuid):
    checkpoint, driver = _load_checkpoint_module(monkeypatch)

    with pytest.raises(ValueError, match="GPU UUID string must be 32 hex characters"):
        checkpoint._make_restore_args(driver, {bad_uuid: "01234567-89ab-cdef-0123-456789abcdef"})


def test_make_restore_args_accepts_uuid_objects(monkeypatch):
    checkpoint, driver = _load_checkpoint_module(monkeypatch)

    old_uuid = driver.CUuuid(111)
    new_uuid = driver.CUuuid(222)
    args = checkpoint._make_restore_args(driver, {old_uuid: new_uuid})

    assert args.gpuPairsCount == 1
    assert args.gpuPairs[0].oldUuid is old_uuid
    assert args.gpuPairs[0].newUuid is new_uuid
