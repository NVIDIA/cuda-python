# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Compile the real native probe against a CPU driver model to check evidence loss."""

import json
import os
import shutil
import subprocess
from collections import Counter
from pathlib import Path

import pytest

from ci.tools import diagnose_pinned_memory

# This is a model, not an ABI substitute for cuda.h. Real toolkit compilation
# happens in CDCI. Failures here exercise the probe's reporting and controls.
DRIVER_MODEL = r"""
#include <cstddef>
#include <cstdlib>
#include <chrono>
#include <thread>
using CUresult = int;
using CUcontext = void*;
using CUstream = void*;
using CUmemoryPool = void*;
using CUdeviceptr = unsigned long long;
using cuuint64_t = unsigned long long;
#define CUDA_VERSION 13000
#define CU_STREAM_LEGACY ((CUstream)1)
#define CU_STREAM_PER_THREAD ((CUstream)2)
constexpr int CUDA_SUCCESS = 0, CU_STREAM_NON_BLOCKING = 1;
constexpr int CU_MEM_ALLOCATION_TYPE_PINNED = 1, CU_MEM_HANDLE_TYPE_NONE = 0;
constexpr int CU_MEM_LOCATION_TYPE_HOST = 2, CU_MEM_LOCATION_TYPE_DEVICE = 1;
constexpr int CU_MEM_ACCESS_FLAGS_PROT_READWRITE = 3, CU_MEM_ACCESS_FLAGS_PROT_NONE = 0;
constexpr int CU_MEMPOOL_ATTR_RESERVED_MEM_HIGH = 1, CU_MEMPOOL_ATTR_RESERVED_MEM_CURRENT = 2;
constexpr int CU_MEMPOOL_ATTR_USED_MEM_HIGH = 3, CU_MEMPOOL_ATTR_USED_MEM_CURRENT = 4;
struct CUmemLocation { int type, id; };
struct CUmemPoolProps { int allocType, handleTypes; CUmemLocation location; size_t maxSize; };
struct CUmemAccessDesc { CUmemLocation location; int flags; };
static bool destroyed = false, poisoned = false;
static cuuint64_t watermark = 1 << 20;
CUresult cuInit(unsigned) { return 0; }
CUresult cuDriverGetVersion(int* version) { *version = 13000; return 0; }
CUresult cuDeviceGetName(char* name, int, int) { name[0] = 0; return 0; }
CUresult cuDevicePrimaryCtxRetain(CUcontext* ctx, int) { *ctx = (void*)1; return 0; }
CUresult cuDevicePrimaryCtxRelease(int) { return 0; }
CUresult cuCtxPushCurrent(CUcontext) { return 0; }
CUresult cuCtxPopCurrent(CUcontext* ctx) { *ctx = (void*)1; return 0; }
CUresult cuMemGetInfo(size_t* free, size_t* total) { *free = *total = 1; return poisoned ? 201 : 0; }
CUresult cuStreamCreate(CUstream* stream, unsigned flags) { *stream = (void*)42; return flags == 1 ? 0 : 1; }
CUresult cuStreamDestroy(CUstream) { return 0; }
CUresult cuDeviceGetCount(int* count) { *count = 1; return 0; }
CUresult cuMemPoolCreate(CUmemoryPool* pool, const CUmemPoolProps* props) {
    *pool = (void*)10; destroyed = false;
    return props->allocType == 1 && props->handleTypes == 0 && props->location.type == 2 &&
           props->location.id == -1 && props->maxSize == 2 << 20 ? 0 : 1;
}
CUresult cuMemPoolSetAccess(CUmemoryPool, const CUmemAccessDesc* desc, size_t count) {
    return desc->location.type == 1 && desc->location.id == 0 && count == 1 &&
           (desc->flags == 0 || desc->flags == 3) ? 0 : 1;
}
CUresult cuMemAllocFromPoolAsync(CUdeviceptr* ptr, size_t bytes, CUmemoryPool, CUstream stream) {
    *ptr = 100; watermark = 1 << 20;
    const char* expected = std::getenv("NATIVE_TEST_STREAM");
    return bytes == 1 << 20 && (!expected || stream == (CUstream)std::strtoull(expected, nullptr, 10)) ? 0 : 1;
}
CUresult cuMemFreeAsync(CUdeviceptr, CUstream) { return 0; }
CUresult cuMemPoolTrimTo(CUmemoryPool, size_t) { return 0; }
CUresult cuMemPoolGetAttribute(CUmemoryPool, int attr, void* value) {
    *static_cast<cuuint64_t*>(value) = attr == 1 || attr == 3 ? watermark : 0; return 0;
}
CUresult cuMemPoolSetAttribute(CUmemoryPool, int, void* value) {
    watermark = *static_cast<cuuint64_t*>(value); return watermark == 0 ? 0 : 1;
}
CUresult cuMemPoolDestroy(CUmemoryPool) { destroyed = true; return 0; }
CUresult cuCtxSynchronize() {
    const char* mode = std::getenv("NATIVE_TEST_FAILURE");
    if (destroyed && mode) {
        if (mode[0] == 'c') std::_Exit(86);
        if (mode[0] == 'h') std::this_thread::sleep_for(std::chrono::seconds(60));
        poisoned = true;
        return 2;
    }
    return 0;
}
"""


@pytest.fixture(scope="module")
def native_probe(tmp_path_factory):
    compiler = shutil.which("c++")
    if compiler is None or os.name == "nt":
        pytest.skip("CPU native-probe checks require a POSIX C++ compiler")
    build = tmp_path_factory.mktemp("native-probe")
    (build / "cuda.h").write_text(DRIVER_MODEL)
    executable = build / "pinned_pool_probe"
    source = Path(diagnose_pinned_memory.__file__).with_name("pinned_pool_probe.cpp")
    subprocess.run(  # noqa: S603 - local compiler, fixed source and test-owned paths.
        [compiler, "-std=c++17", "-O2", "-I", str(build), str(source), "-o", str(executable)],
        check=True,
        capture_output=True,
        timeout=60,
    )
    return executable


def run_native(native_probe, output, stream="legacy", revoke=False, variant="baseline", failure=None):
    name = f"native-{stream}-{'revoke' if revoke else 'keep-access'}"
    if variant != "baseline":
        name += f"-{variant}"
    case = next(
        case for case in diagnose_pinned_memory.build_cases(output, output, 1, native_probe) if case["name"] == name
    )
    case["timeout_seconds"] = 2
    env = {**os.environ, "NATIVE_TEST_STREAM": {"legacy": "1", "per-thread": "2", "nonblocking": "42"}[stream]}
    env.pop("NATIVE_TEST_FAILURE", None)
    if failure is not None:
        env["NATIVE_TEST_FAILURE"] = failure
    return diagnose_pinned_memory.run_case(case, output, env), Path(case["raw_output"])


@pytest.mark.parametrize("_name,stream,revoke,variant", diagnose_pinned_memory.NATIVE_CASES)
@pytest.mark.agent_authored(model="gpt-6")
def test_native_controls_execute_distinct_lifecycles(native_probe, tmp_path, _name, stream, revoke, variant):
    result, output = run_native(native_probe, tmp_path, stream, revoke, variant)
    assert not result["failed"], result
    assert result["report"]["completed_cycles"] == 2
    calls = [json.loads(line) for line in (output / "calls.jsonl").read_text().splitlines()]
    counts = Counter(call["api"] for call in calls)
    pools = 1 if variant == "reuse" else 2
    assert counts["cuMemPoolCreate"] == counts["cuMemPoolDestroy"] == pools
    assert counts["cuMemPoolSetAccess"] == pools * (1 + int(revoke))
    assert counts["cuMemPoolTrimTo"] == (0 if variant == "no-trim" else 2)
    assert counts["cuMemPoolSetAttribute"] == (0 if variant == "no-reset" else 2)
    assert counts["cuMemAllocFromPoolAsync"] == counts["cuMemFreeAsync"] == 2


@pytest.mark.parametrize("variant,completed", [("baseline", 0), ("reuse", 2)])
@pytest.mark.agent_authored(model="gpt-6")
def test_native_preserves_first_error_even_after_all_reuse_cycles(native_probe, tmp_path, variant, completed):
    result, output = run_native(native_probe, tmp_path, variant=variant, failure="oom")
    assert result["failed"]
    report = result["report"]
    assert not report["required_path_completed"]
    assert report["completed_cycles"] == completed
    assert report["failure"]["api"] == "cuCtxSynchronize"
    assert report["failure"]["stage"] == "owned_post_destroy_sync"
    assert report["failure"]["status"] == 2
    assert json.loads((output / "progress.json").read_text())["status"] == 2
    assert "cuMemGetInfo status=201" in Path(result["log_path"]).read_text()


@pytest.mark.parametrize("failure", ["crash", "hang"])
@pytest.mark.agent_authored(model="gpt-6")
def test_native_retains_inflight_api_after_crash_or_watchdog(native_probe, tmp_path, failure):
    result, output = run_native(native_probe, tmp_path, failure=failure)
    assert result["failed"]
    assert result["timed_out"] == (failure == "hang")
    progress = json.loads((output / "progress.json").read_text())
    assert progress["api"] == "cuCtxSynchronize"
    assert progress["stage"] == "owned_post_destroy_sync"
    assert progress["status"] == -1
    assert not progress["returned"]
    assert not (output / "summary.json").exists()
