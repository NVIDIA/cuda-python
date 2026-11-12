# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

"""Phase-logged raw-driver HOST-pool probe; deliberately does not import cuda.core."""

import argparse
import ctypes
import importlib.metadata
import json
import os
import platform
import subprocess
import sys
import time
import traceback
from pathlib import Path

parser = argparse.ArgumentParser()
parser.add_argument(
    "--mode", required=True, choices=("lookup-only", "default-allocate", "owned-capped", "warmed-owned-capped")
)
parser.add_argument("--output-dir", type=Path, required=True)
parser.add_argument("--host-id", type=int, default=0)
parser.add_argument("--lookup-api", choices=("current", "default"), default="current")
parser.add_argument("--watermark", choices=("none", "reserved", "used"), default="none")
parser.add_argument("--cycles", type=int, default=1)
parser.add_argument("--allow-warmup-allocation-oom", action="store_true")
args = parser.parse_args()
if not 1 <= args.cycles <= 100:
    parser.error("--cycles must be between 1 and 100")
if args.allow_warmup_allocation_oom and args.mode != "warmed-owned-capped":
    parser.error("--allow-warmup-allocation-oom requires warmed-owned-capped mode")
if args.watermark != "none" and args.mode not in ("owned-capped", "warmed-owned-capped"):
    parser.error("--watermark requires an owned-capped mode")
args.output_dir.mkdir(parents=True, exist_ok=True)
stem = f"raw-{args.mode}"
if args.watermark != "none":
    stem += f"-{args.watermark}"
if args.cycles != 1:
    stem += f"-cycles{args.cycles}"
jsonl_path = args.output_dir / f"{stem}.jsonl"
jsonl = jsonl_path.open("w", encoding="utf-8", buffering=1)
rows = []
allowed_warmup_ooms = []
iteration = -1
current_stage = "startup"
driver = None


def emit(event, **data):
    row = {
        "mode": args.mode,
        "watermark": args.watermark,
        "iteration": iteration,
        "stage": current_stage,
        "event": event,
        "monotonic_ns": time.monotonic_ns(),
        **data,
    }
    line = json.dumps(row, default=str, sort_keys=True)
    # Flush to the artifact before stdout: native aborts must preserve the last API boundary.
    jsonl.write(line + "\n")
    jsonl.flush()
    rows.append(row)
    print(line, flush=True)


def stage(name):
    global current_stage
    current_stage = name
    emit("stage_begin")


class CudaCallError(RuntimeError):
    def __init__(self, name, status, failed_stage):
        self.api = name
        self.status = int(status)
        self.status_name = status.name
        self.failed_stage = failed_stage
        super().__init__(f"{name} at {failed_stage}: {status.name} ({self.status})")


def raw_call(name, *values, required=True):
    emit("api_begin", api=name, required=required)
    try:
        result = getattr(driver, name)(*values)
    except Exception as exc:
        emit("api_exception", api=name, required=required, error_type=type(exc).__name__, error=str(exc))
        if required:
            raise
        return None
    status = int(result[0])
    emit("raw_status", api=name, required=required, status=status, status_name=result[0].name)
    if status and required:
        raise CudaCallError(name, result[0], current_stage)
    return result


def api(name, *values):
    return raw_call(name, *values)[1:]


def snapshot(phase):
    if os.name == "nt":
        # Public Win32 layout; 64-bit Windows is the target CI platform.
        # https://learn.microsoft.com/windows/win32/api/winnt/ns-winnt-memory_basic_information
        if ctypes.sizeof(ctypes.c_void_p) != 8:
            raise RuntimeError("The Windows memory probe requires a 64-bit Python")

        class MemoryBasicInformation(ctypes.Structure):
            _fields_ = [
                ("BaseAddress", ctypes.c_void_p),
                ("AllocationBase", ctypes.c_void_p),
                ("AllocationProtect", ctypes.c_uint32),
                ("PartitionId", ctypes.c_uint16),
                ("RegionSize", ctypes.c_size_t),
                ("State", ctypes.c_uint32),
                ("Protect", ctypes.c_uint32),
                ("Type", ctypes.c_uint32),
            ]

        kernel32 = ctypes.WinDLL("kernel32", use_last_error=True)
        query = kernel32.VirtualQuery
        query.argtypes = [ctypes.c_void_p, ctypes.POINTER(MemoryBasicInformation), ctypes.c_size_t]
        query.restype = ctypes.c_size_t
        address = 0
        reserved = 0
        committed = 0
        maps = []
        while True:
            mbi = MemoryBasicInformation()
            size = query(ctypes.c_void_p(address), ctypes.byref(mbi), ctypes.sizeof(mbi))
            if size == 0:
                error = ctypes.get_last_error()
                if error == 87:  # Above the process's highest accessible address.
                    break
                raise ctypes.WinError(error)
            base_address = mbi.BaseAddress or 0
            next_address = base_address + mbi.RegionSize
            if next_address <= address:
                raise RuntimeError("VirtualQuery did not advance")
            if mbi.State in (0x1000, 0x2000):  # MEM_COMMIT, MEM_RESERVE
                if mbi.State == 0x1000:
                    committed += mbi.RegionSize
                else:
                    reserved += mbi.RegionSize
                maps.append(
                    {
                        "bytes": mbi.RegionSize,
                        "range": f"{base_address:x}-{next_address:x}",
                        "state": "MEM_COMMIT" if mbi.State == 0x1000 else "MEM_RESERVE",
                    }
                )
            address = next_address
        status = {
            "VmSize": (reserved + committed) // 1024,
            "VirtualReserved": reserved // 1024,
            "VirtualCommitted": committed // 1024,
        }
        measurement = "Win32 VirtualQuery sum of MEM_RESERVE plus MEM_COMMIT; excludes MEM_FREE"
    else:
        wanted = ("VmSize:", "VmRSS:", "VmPin:", "VmLck:")
        status = {
            line.split(":", 1)[0]: int(line.split()[1])
            for line in Path("/proc/self/status").read_text().splitlines()
            if line.startswith(wanted)
        }
        maps = []
        for line in Path("/proc/self/maps").read_text().splitlines():
            fields = line.split(maxsplit=5)
            start, end = (int(x, 16) for x in fields[0].split("-"))
            maps.append(
                {
                    "bytes": end - start,
                    "range": fields[0],
                    "perms": fields[1],
                    "path": fields[5] if len(fields) > 5 else "",
                }
            )
        measurement = "Linux /proc/self/status VmSize and /proc/self/maps"
    emit(
        "snapshot",
        phase=phase,
        process_memory_kB=status,
        measurement=measurement,
        largest_mappings=sorted(maps, key=lambda r: r["bytes"], reverse=True)[:5],
    )


def os_snapshot(phase):
    try:
        snapshot(phase)
    except (OSError, RuntimeError) as exc:
        emit(
            "optional_metadata_error",
            component="process_virtual_memory",
            phase=phase,
            error_type=type(exc).__name__,
            error=str(exc),
        )


def host_ram(phase):
    try:
        if os.name == "nt":
            # https://learn.microsoft.com/windows/win32/api/sysinfoapi/ns-sysinfoapi-memorystatusex
            class MemoryStatusEx(ctypes.Structure):
                _fields_ = [("dwLength", ctypes.c_uint32), ("dwMemoryLoad", ctypes.c_uint32)] + [
                    (name, ctypes.c_uint64)
                    for name in (
                        "ullTotalPhys",
                        "ullAvailPhys",
                        "ullTotalPageFile",
                        "ullAvailPageFile",
                        "ullTotalVirtual",
                        "ullAvailVirtual",
                        "ullAvailExtendedVirtual",
                    )
                ]

            kernel32 = ctypes.WinDLL("kernel32", use_last_error=True)
            query = kernel32.GlobalMemoryStatusEx
            query.argtypes = [ctypes.POINTER(MemoryStatusEx)]
            query.restype = ctypes.c_int
            info = MemoryStatusEx()
            info.dwLength = ctypes.sizeof(info)
            if not query(ctypes.byref(info)):
                raise ctypes.WinError(ctypes.get_last_error())
            emit(
                "host_ram",
                phase=phase,
                source="GlobalMemoryStatusEx",
                fields={name: int(getattr(info, name)) for name, _ in info._fields_},
            )
        else:
            names = {"MemTotal", "MemAvailable", "MemFree", "CommitLimit", "Committed_AS", "SwapTotal", "SwapFree"}
            values = {}
            for line in Path("/proc/meminfo").read_text().splitlines():
                name, value = line.split(":", 1)
                if name in names:
                    values[name] = int(value.split()[0]) * 1024
            emit("host_ram", phase=phase, source="/proc/meminfo", fields_bytes=values)
    except (OSError, RuntimeError) as exc:
        emit(
            "optional_metadata_error", component="host_ram", phase=phase, error_type=type(exc).__name__, error=str(exc)
        )


def vram(phase):
    result = raw_call("cuMemGetInfo", required=False)
    if result is not None and int(result[0]) == 0:
        emit("vram", phase=phase, free_bytes=int(result[1]), total_bytes=int(result[2]))


def baseline_metadata():
    stage("baseline_metadata")
    result = raw_call("cuDriverGetVersion", required=False)
    max_cuda = int(result[1]) if result is not None and int(result[0]) == 0 else None
    result = raw_call("cuDeviceGetName", 256, driver.CUdevice(0), required=False)
    device_name = None
    if result is not None and int(result[0]) == 0:
        value = result[1]
        device_name = value.decode("utf-8", errors="replace").rstrip("\0") if isinstance(value, bytes) else str(value)
    try:
        bindings_version = importlib.metadata.version("cuda-bindings")
    except importlib.metadata.PackageNotFoundError:
        bindings_version = None
    gil = getattr(sys, "_is_gil_enabled", None)
    emit(
        "baseline_versions",
        python=sys.version,
        platform=platform.platform(),
        gil_enabled=gil() if gil is not None else None,
        device_name=device_name,
        driver_max_cuda_version=max_cuda,
        bindings_compiled_cuda_version=getattr(driver, "CUDA_VERSION", None),
        bindings_package_version=bindings_version,
    )
    vram("baseline")
    host_ram("baseline")
    os_snapshot("initialized_context_baseline")
    try:
        command = [
            "nvidia-smi",
            "--query-gpu=index,name,driver_version,memory.total,memory.free",
            "--format=csv,noheader,nounits",
        ]
        # Fixed nvidia-smi arguments; no shell or user-provided command.
        result = subprocess.run(command, text=True, capture_output=True, timeout=10)  # noqa: S603
        emit("nvidia_smi", returncode=result.returncode, stdout=result.stdout, stderr=result.stderr)
    except (OSError, subprocess.TimeoutExpired) as exc:
        emit("optional_metadata_error", component="nvidia_smi", error_type=type(exc).__name__, error=str(exc))


def attributes(pool, phase):
    names = (
        "MAX_POOL_SIZE",
        "ALLOCATION_TYPE",
        "LOCATION_TYPE",
        "LOCATION_ID",
        "RELEASE_THRESHOLD",
        "RESERVED_MEM_CURRENT",
        "RESERVED_MEM_HIGH",
        "USED_MEM_CURRENT",
        "USED_MEM_HIGH",
    )
    values = {}
    for name in names:
        attr = getattr(driver.CUmemPool_attribute, f"CU_MEMPOOL_ATTR_{name}", None)
        if attr is None:
            values[name] = {"status": None, "value": None, "rejected": "enum unavailable in bindings"}
            emit("optional_query_rejected", query=name, reason="enum unavailable in bindings")
            continue
        result = raw_call("cuMemPoolGetAttribute", pool, attr, required=False)
        if result is None:
            values[name] = {"status": None, "value": None, "rejected": "API exception"}
        else:
            status = int(result[0])
            values[name] = {
                "status": status,
                "status_name": result[0].name,
                "value": int(result[1]) if status == 0 else None,
            }
    emit("pool_attributes", phase=phase, pool_handle=int(pool), attributes=values)


def grant_access(pool):
    desc = driver.CUmemAccessDesc()
    desc.location.type = driver.CUmemLocationType.CU_MEM_LOCATION_TYPE_DEVICE
    desc.location.id = 0
    desc.flags = driver.CUmemAccess_flags.CU_MEM_ACCESS_FLAGS_PROT_READWRITE
    api("cuMemPoolSetAccess", pool, [desc], 1)


def default_pool_allocate(prefix, do_allocate):
    loc = driver.CUmemLocation()
    loc.type = driver.CUmemLocationType.CU_MEM_LOCATION_TYPE_HOST
    loc.id = args.host_id
    lookup_name = "cuMemGetMemPool" if args.lookup_api == "current" else "cuMemGetDefaultMemPool"
    stage(f"{prefix}_lookup")
    emit("lookup_parameters", api=lookup_name, host_id=args.host_id)
    (pool,) = api(lookup_name, loc, driver.CUmemAllocationType.CU_MEM_ALLOCATION_TYPE_PINNED)
    os_snapshot(f"{prefix}_after_lookup")
    attributes(pool, f"{prefix}_after_lookup")
    if do_allocate:
        stage(f"{prefix}_access_grant")
        grant_access(pool)
        os_snapshot(f"{prefix}_after_access_grant")
        stage(f"{prefix}_allocate_4096B")
        try:
            (ptr,) = api("cuMemAllocFromPoolAsync", 4096, pool, driver.CUstream(0))
        except CudaCallError as exc:
            if not (
                args.allow_warmup_allocation_oom
                and args.mode == "warmed-owned-capped"
                and prefix == "warmup_default"
                and exc.api == "cuMemAllocFromPoolAsync"
                and exc.status == 2
            ):
                raise
            observed = {
                "api": exc.api,
                "status": exc.status,
                "status_name": exc.status_name,
                "stage": exc.failed_stage,
                "error": str(exc),
            }
            allowed_warmup_ooms.append(observed)
            emit(
                "allowed_warmup_allocation_oom",
                preserved_error=observed,
                continuation="requires_successful_post_oom_sync_before_owned_cycles",
            )
            stage("warmup_default_post_oom_sync")
            api("cuCtxSynchronize")
            os_snapshot("warmup_default_after_oom_sync")
            attributes(pool, "warmup_default_after_oom_sync")
            emit(
                "warmup_continuation",
                outcome="allocation_oom_preserved",
                continuation="post_oom_sync_passed_proceed_to_owned_cycles",
            )
            return pool
        os_snapshot(f"{prefix}_after_allocate_4096B")
        attributes(pool, f"{prefix}_after_allocate_4096B")
        stage(f"{prefix}_free_enqueue")
        api("cuMemFreeAsync", ptr, driver.CUstream(0))
        # No diagnostic CUDA API is inserted between the free and its initial sync.
        stage(f"{prefix}_initial_sync")
        api("cuCtxSynchronize")
        os_snapshot(f"{prefix}_after_free_sync")
        attributes(pool, f"{prefix}_after_free_sync")
    return pool


def reset_watermark(pool):
    stage(f"owned_{args.watermark}_watermark_read_before")
    high_attr = getattr(driver.CUmemPool_attribute, f"CU_MEMPOOL_ATTR_{args.watermark.upper()}_MEM_HIGH")
    current_attr = getattr(driver.CUmemPool_attribute, f"CU_MEMPOOL_ATTR_{args.watermark.upper()}_MEM_CURRENT")
    (high,) = api("cuMemPoolGetAttribute", pool, high_attr)
    (current,) = api("cuMemPoolGetAttribute", pool, current_attr)
    high, current = int(high), int(current)
    emit("watermark_before", kind=args.watermark, high=int(high), current=int(current))
    if high < 1048576 or current >= 1048576:
        raise AssertionError(f"Unexpected pre-reset watermark: high={high}, current={current}")
    stage(f"owned_{args.watermark}_watermark_set_zero")
    api("cuMemPoolSetAttribute", pool, high_attr, driver.cuuint64_t(0))
    stage(f"owned_{args.watermark}_watermark_verify")
    (high,) = api("cuMemPoolGetAttribute", pool, high_attr)
    (current,) = api("cuMemPoolGetAttribute", pool, current_attr)
    high, current = int(high), int(current)
    emit("watermark_after", kind=args.watermark, high=int(high), current=int(current))
    if high != current:
        raise AssertionError(f"Reset watermark mismatch: high={high}, current={current}")


def owned_cycle():
    props = driver.CUmemPoolProps()
    props.allocType = driver.CUmemAllocationType.CU_MEM_ALLOCATION_TYPE_PINNED
    props.handleTypes = driver.CUmemAllocationHandleType.CU_MEM_HANDLE_TYPE_NONE
    props.location.type = driver.CUmemLocationType.CU_MEM_LOCATION_TYPE_HOST
    props.location.id = args.host_id
    props.maxSize = 2 * 1024 * 1024
    stage("owned_create")
    emit(
        "create_props",
        allocType=int(props.allocType),
        location_type=int(props.location.type),
        location_id=int(props.location.id),
        maxSize=int(props.maxSize),
    )
    (pool,) = api("cuMemPoolCreate", props)
    os_snapshot("owned_after_create")
    attributes(pool, "owned_after_create")
    stage("owned_access_grant")
    grant_access(pool)
    os_snapshot("owned_after_access_grant")
    stage("owned_allocate_1MiB")
    (ptr,) = api("cuMemAllocFromPoolAsync", 1048576, pool, driver.CUstream(0))
    os_snapshot("owned_after_allocate_1MiB")
    attributes(pool, "owned_after_allocate_1MiB")
    stage("owned_free_enqueue")
    api("cuMemFreeAsync", ptr, driver.CUstream(0))
    # Keep this adjacency: free -> first sync, with only flushed CPU log writes between.
    stage("owned_initial_sync")
    api("cuCtxSynchronize")
    os_snapshot("owned_after_free_sync")
    attributes(pool, "owned_after_free_sync")
    stage("owned_trim")
    api("cuMemPoolTrimTo", pool, 0)
    os_snapshot("owned_after_trim")
    if args.watermark != "none":
        reset_watermark(pool)
    stage("owned_destroy")
    api("cuMemPoolDestroy", pool)
    os_snapshot("owned_after_destroy")
    stage("owned_post_destroy_sync")
    api("cuCtxSynchronize")
    os_snapshot("owned_after_post_destroy_sync")


completed = 0
failure = None
context_current = False
primary_retained = False
emit(
    "configuration",
    platform=os.name,
    host_id=args.host_id,
    lookup_api=args.lookup_api,
    cycles=args.cycles,
    alloc_type="PINNED",
    stream=0,
    default_allocation_bytes=4096,
    owned_allocation_bytes=1048576,
    owned_requested_maxSize=2097152,
    handle_types=0,
    allow_warmup_allocation_oom=args.allow_warmup_allocation_oom,
    jsonl_path=str(jsonl_path),
)
try:
    stage("import_driver")
    from cuda.bindings import driver

    stage("initialize")
    api("cuInit", 0)
    (ctx,) = api("cuDevicePrimaryCtxRetain", driver.CUdevice(0))
    primary_retained = True
    api("cuCtxPushCurrent", ctx)
    context_current = True
    baseline_metadata()
    if args.mode == "warmed-owned-capped":
        default_pool_allocate("warmup_default", True)
        os_snapshot("warmed_owned_baseline")
    stage("baseline_context_pop")
    api("cuCtxPopCurrent")
    context_current = False
    for iteration in range(args.cycles):
        stage("iteration_context_push")
        api("cuCtxPushCurrent", ctx)
        context_current = True
        if args.mode in ("lookup-only", "default-allocate"):
            default_pool_allocate("default", args.mode == "default-allocate")
        else:
            owned_cycle()
        stage("iteration_context_pop")
        api("cuCtxPopCurrent")
        context_current = False
        os_snapshot("after_context_pop")
        completed += 1
        emit("iteration_complete", completed_cycles=completed)
    stage("primary_context_release")
    api("cuDevicePrimaryCtxRelease", driver.CUdevice(0))
    primary_retained = False
    os_snapshot("after_primary_context_release")
except BaseException as exc:
    failed_stage = current_stage
    failure = {
        "error_type": type(exc).__name__,
        "error": str(exc),
        "stage": failed_stage,
        "iteration": iteration,
        "traceback": traceback.format_exc(),
    }
    emit("failure", **failure)
    stage("failure_probes")
    # CPU-only OS snapshots are safe after a CUDA failure. One optional VRAM query
    # records rejection too; no synchronization/retry is used to clear the failure.
    host_ram("failure")
    os_snapshot("failure")
    if context_current:
        vram("failure")
finally:
    if context_current:
        stage("failure_context_pop")
        raw_call("cuCtxPopCurrent", required=False)
    if primary_retained:
        stage("failure_primary_context_release")
        raw_call("cuDevicePrimaryCtxRelease", driver.CUdevice(0), required=False)
    stage("finished")
    outcome = (
        "failed" if failure is not None else ("completed_with_allowed_warmup_oom" if allowed_warmup_ooms else "passed")
    )
    summary = {
        "requested_cycles": args.cycles,
        "completed_cycles": completed,
        "passed": failure is None and not allowed_warmup_ooms,
        "required_path_completed": failure is None,
        "outcome": outcome,
        "allowed_warmup_ooms": allowed_warmup_ooms,
        "failure": failure,
        "jsonl_path": str(jsonl_path),
    }
    emit("summary", **summary)
    (args.output_dir / f"{stem}.json").write_text(json.dumps(rows, indent=2, default=str) + "\n", encoding="utf-8")
    (args.output_dir / f"{stem}-summary.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    (args.output_dir / "summary.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    jsonl.close()
raise SystemExit(1 if failure else 0)
