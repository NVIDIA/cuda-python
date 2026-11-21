# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""CPU checks that experimental variants exercise different driver behavior."""

import json
import runpy
import sys
from enum import IntEnum
from pathlib import Path
from types import ModuleType, SimpleNamespace

import pytest


class Status(IntEnum):
    CUDA_SUCCESS = 0
    CUDA_ERROR_OUT_OF_MEMORY = 2
    CUDA_ERROR_INVALID_CONTEXT = 201


class DriverModel:
    CUdevice = int
    CUstream = int
    cuuint64_t = int
    CUmemAllocationType = SimpleNamespace(CU_MEM_ALLOCATION_TYPE_PINNED=1)
    CUmemAllocationHandleType = SimpleNamespace(CU_MEM_HANDLE_TYPE_NONE=0)
    CUmemLocationType = SimpleNamespace(CU_MEM_LOCATION_TYPE_HOST=2, CU_MEM_LOCATION_TYPE_DEVICE=1)
    CUmemAccess_flags = SimpleNamespace(CU_MEM_ACCESS_FLAGS_PROT_READWRITE=3, CU_MEM_ACCESS_FLAGS_PROT_NONE=0)
    CUstream_flags = SimpleNamespace(CU_STREAM_NON_BLOCKING=1)
    CUmemPool_attribute = SimpleNamespace(
        CU_MEMPOOL_ATTR_RESERVED_MEM_HIGH=1,
        CU_MEMPOOL_ATTR_RESERVED_MEM_CURRENT=2,
        CU_MEMPOOL_ATTR_USED_MEM_HIGH=3,
        CU_MEMPOOL_ATTR_USED_MEM_CURRENT=4,
    )

    def __init__(self, fail_after_destroy=False):
        self.fail_after_destroy = fail_after_destroy
        self.destroyed = False
        self.poisoned = False
        self.high = 1048576
        self.streams = []
        self.revoked = 0
        self.reset_kinds = []
        self.completed = 0

    @staticmethod
    def CUmemPoolProps():
        return SimpleNamespace(location=SimpleNamespace())

    @staticmethod
    def CUmemAccessDesc():
        return SimpleNamespace(location=SimpleNamespace())

    def __getattr__(self, name):
        if not name.startswith("cu"):
            raise AttributeError(name)

        def call(*args):
            values = ()
            if name == "cuMemPoolCreate":
                self.destroyed = False
                self.high = 1048576
                values = (10,)
            elif name == "cuMemPoolDestroy":
                self.destroyed = True
            elif name == "cuMemPoolSetAccess":
                self.revoked += int(args[1][0].flags == 0)
            elif name == "cuMemAllocFromPoolAsync":
                self.streams.append(args[2])
                values = (100,)
            elif name == "cuMemPoolGetAttribute":
                values = (self.high if args[1] in (1, 3) else 0,)
            elif name == "cuMemPoolSetAttribute":
                self.reset_kinds.append(args[1])
                self.high = 0
            elif name == "cuCtxSynchronize" and self.destroyed:
                if self.fail_after_destroy:
                    self.poisoned = True
                    return (Status.CUDA_ERROR_OUT_OF_MEMORY,)
                self.completed += 1
            elif name == "cuDevicePrimaryCtxRetain":
                values = (123,)
            elif name == "cuCtxPopCurrent":
                if self.poisoned:
                    return (Status.CUDA_ERROR_INVALID_CONTEXT,)
                values = (123,)
            elif name == "cuStreamCreate":
                values = (42,)
            elif name == "cuDeviceGetCount":
                values = (1,)
            elif name == "cuMemGetInfo":
                values = (8 << 30, 16 << 30)
            elif name == "cuDriverGetVersion":
                values = (13020,)
            elif name == "cuDeviceGetName":
                values = (b"CPU model",)
            elif name not in (
                "cuInit",
                "cuCtxPushCurrent",
                "cuDevicePrimaryCtxRelease",
                "cuMemFreeAsync",
                "cuMemPoolTrimTo",
                "cuCtxSynchronize",
                "cuStreamDestroy",
            ):
                raise AssertionError(f"Unexpected diagnostic CUDA call: {name}")
            return (Status.CUDA_SUCCESS, *values)

        return call


def run_probe(monkeypatch, tmp_path, model, stream, revoke):
    cuda = ModuleType("cuda")
    bindings = ModuleType("cuda.bindings")
    bindings.driver = model
    cuda.bindings = bindings
    monkeypatch.setitem(sys.modules, "cuda", cuda)
    monkeypatch.setitem(sys.modules, "cuda.bindings", bindings)
    probe = Path(__file__).parents[1] / "pinned_pool_probe.py"
    argv = [
        str(probe),
        "--mode",
        "owned-capped",
        "--host-id",
        "-1",
        "--cycles",
        "2",
        "--watermark",
        "alternating",
        "--low-overhead",
        "--stream",
        stream,
        "--output-dir",
        str(tmp_path),
    ]
    if revoke:
        argv.append("--revoke-access")
    monkeypatch.setattr(sys, "argv", argv)
    monkeypatch.setattr("subprocess.run", lambda *_args, **_kwargs: SimpleNamespace(returncode=0, stdout="", stderr=""))
    with pytest.raises(SystemExit) as stopped:
        runpy.run_path(str(probe), run_name="__main__")
    return stopped.value.code, json.loads((tmp_path / "summary.json").read_text())


@pytest.mark.parametrize("stream,handle", [("legacy", 1), ("per-thread", 2), ("nonblocking", 42)])
@pytest.mark.parametrize("revoke", [False, True])
@pytest.mark.agent_authored(model="gpt-6")
def test_variants_exercise_selected_stream_access_and_both_watermarks(monkeypatch, tmp_path, stream, handle, revoke):
    model = DriverModel()
    code, report = run_probe(monkeypatch, tmp_path, model, stream, revoke)
    assert code == 0
    assert report["completed_cycles"] == model.completed == 2
    assert model.streams == [handle, handle]
    assert model.revoked == 2 * int(revoke)
    assert model.reset_kinds == [1, 3]
    assert report["stream"] == stream
    assert report["revoke_access"] == revoke


@pytest.mark.agent_authored(model="gpt-6")
def test_first_failure_survives_cleanup_and_progress_is_not_overwritten(monkeypatch, tmp_path):
    model = DriverModel(fail_after_destroy=True)
    code, report = run_probe(monkeypatch, tmp_path, model, "legacy", True)
    assert code == 1
    assert not report["required_path_completed"]
    assert report["completed_cycles"] == 0
    assert report["failure"]["api"] == "cuCtxSynchronize"
    assert report["failure"]["status"] == 2
    progress = json.loads((tmp_path / "progress.json").read_text())
    assert progress["stage"] == "owned_post_destroy_sync"
    assert progress["status"] == 2
    assert progress["iteration"] == 0
