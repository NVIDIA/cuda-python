# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

"""TEMPORARY diagnostic for the Linux-vs-Windows-TCC difference seen in PR #3035.

On Linux the driver rejects an executable update that replaces a captured
memcpy node's device operand with pinned host memory; Windows TCC accepts it.
The driver source says the rejection comes from a check that only runs when
the original operand was classified as a VA-range operand, which depends on
whether the pointer lookup finds the pool object behind a pool-backed buffer.
This test gathers the evidence on every CI row and FAILS ON PURPOSE so the
report lands in the job log. Do not merge.
"""

import ctypes
import sys

import pytest

from cuda.core import Buffer, LegacyPinnedMemoryResource, VirtualMemoryResource, VirtualMemoryResourceOptions
from cuda.core._utils.cuda_utils import CUDAError, driver, handle_return
from cuda.core._utils.version import driver_version
from cuda.core.graph import MemcpyNode

_ATTRS = [
    "MEMORY_TYPE",
    "MEMPOOL_HANDLE",
    "IS_MANAGED",
    "DEVICE_ORDINAL",
    "IS_LEGACY_CUDA_IPC_CAPABLE",
    "RANGE_START_ADDR",
    "RANGE_SIZE",
    "MAPPED",
    "ALLOWED_HANDLE_TYPES",
    "MAPPING_SIZE",
    "MAPPING_BASE_ADDR",
    "MEMORY_BLOCK_ID",
    "ACCESS_FLAGS",
    "BUFFER_ID",
    "CONTEXT",
]


def _attr(name, ptr):
    try:
        value = handle_return(
            driver.cuPointerGetAttribute(getattr(driver.CUpointer_attribute, f"CU_POINTER_ATTRIBUTE_{name}"), ptr)
        )
    except Exception as exc:  # diagnostic: record anything
        return f"ERR({type(exc).__name__}: {str(exc)[:60]})"
    try:
        return f"0x{int(value):x}" if name.endswith(("ADDR", "HANDLE", "CONTEXT")) else str(int(value))
    except Exception:  # diagnostic: record anything
        return repr(value)[:60]


def _attempt(call):
    try:
        call()
    except CUDAError as exc:
        return f"REJECTED {str(exc)[:160]}"
    except Exception as exc:  # diagnostic: record anything
        return f"RAISED {type(exc).__name__}: {str(exc)[:160]}"
    return "ACCEPTED"


def _read(buffer, stream):
    host = LegacyPinnedMemoryResource().allocate(buffer.size)
    buffer.copy_to(host, stream=stream)
    stream.sync()
    return list((ctypes.c_uint8 * buffer.size).from_address(int(host.handle)))


class _RawDevice:
    """A cuMemAlloc buffer wrapped as a cuda.core Buffer (classic allocation)."""

    def __init__(self, size):
        self.ptr = int(handle_return(driver.cuMemAlloc(size)))
        self.buffer = Buffer.from_handle(self.ptr, size)

    def close(self):
        self.buffer.close()
        handle_return(driver.cuMemFree(self.ptr))


def _allocators(device, stream):
    """Yield (kind, allocate, raw_holders) for each allocation API under test."""
    pool = device.memory_resource
    yield "pool", lambda n: pool.allocate(n, stream=stream), []

    raw = []

    def allocate_raw(n):
        holder = _RawDevice(n)
        raw.append(holder)
        return holder.buffer

    yield "cumemalloc", allocate_raw, raw
    if device.properties.virtual_memory_management_supported:
        vmm = VirtualMemoryResource(device, config=VirtualMemoryResourceOptions(handle_type=None))
        yield "vmm", lambda n: vmm.allocate(n), []


def _probe_kind(device, stream, kind, allocate, host_src, lines):
    size = 64
    src = allocate(size)
    dst = allocate(size)
    n = dst.size  # vmm rounds up
    src.fill(0x5A, stream=stream)
    dst.fill(0, stream=stream)
    stream.sync()
    lines.append(f"[{kind}] size={n}")
    for name, buf in (("src", src), ("dst", dst)):
        lines.append(f"  {name} attrs: " + ", ".join(f"{a}={_attr(a, int(buf.handle))}" for a in _ATTRS))

    builder = device.create_graph_builder().begin_building()
    dst.copy_from(src, stream=builder)
    builder.end_building()
    graph_def = builder.graph_definition
    node = next(x for x in graph_def.nodes() if isinstance(x, MemcpyNode))
    params = handle_return(driver.cuGraphMemcpyNodeGetParams(node.handle))
    lines.append(
        f"  captured: srcMemoryType={int(params.srcMemoryType)} dstMemoryType={int(params.dstMemoryType)}"
        f" srcDevice=0x{int(params.srcDevice):x} dstDevice=0x{int(params.dstDevice):x} width={params.WidthInBytes}"
    )
    before = graph_def.instantiate()
    before.launch(stream)
    lines.append(f"  baseline copy ok: {_read(dst, stream)[:4] == [0x5A] * 4}")

    # 1. Replace src with pinned host memory (the PR #3035 scenario).
    ctypes.memset(int(host_src.handle), 0xA5, 64)
    lines.append("  def-node update(src=pinned host): " + _attempt(lambda: node.update(src=host_src)))
    params = handle_return(driver.cuGraphMemcpyNodeGetParams(node.handle))
    lines.append(
        f"    recorded after: srcMemoryType={int(params.srcMemoryType)} dstMemoryType={int(params.dstMemoryType)}"
    )
    fresh = graph_def.instantiate()
    fresh.launch(stream)
    lines.append(f"    fresh instantiate copies host bytes: {_read(dst, stream)[:4] == [0xA5] * 4}")
    dst.fill(0, stream=stream)
    stream.sync()
    outcome = _attempt(lambda: before.update(graph_def))
    lines.append(f"    Graph.update() on earlier instantiation: {outcome}")
    if outcome == "ACCEPTED":
        before.launch(stream)
        lines.append(f"      launch after accepted update copies host bytes: {_read(dst, stream)[:4] == [0xA5] * 4}")
    other = graph_def.instantiate()
    lines.append(
        "    exec-view update(src=pinned host): " + _attempt(lambda: other[node].update(dst=dst, src=host_src, size=n))
    )

    # 2. Control: replace src with another buffer of the same kind.
    same = allocate(size)
    same.fill(0x3C, stream=stream)
    stream.sync()
    lines.append("  def-node update(src=same kind): " + _attempt(lambda: node.update(src=same)))
    before2 = before
    lines.append("    Graph.update() on earlier instantiation: " + _attempt(lambda: before2.update(graph_def)))

    # 3. Control: replace src with a classic cuMemAlloc buffer (different class on Linux).
    raw = _RawDevice(size)
    lines.append("  def-node update(src=cuMemAlloc buffer): " + _attempt(lambda: node.update(src=raw.buffer)))
    lines.append("    Graph.update() on earlier instantiation: " + _attempt(lambda: before2.update(graph_def)))
    raw.close()
    for b in (src, dst, same):
        b.close()


@pytest.mark.agent_authored(model="claude-fable-5-1")
def test_zz_tcc_memcpy_update_diag(init_cuda):
    device = init_cuda
    stream = device.create_stream()
    props = device.properties
    lines = [
        "=== TCC MEMCPY UPDATE DIAGNOSTIC ===",
        f"platform={sys.platform} device={device.name} driver={driver_version()}",
        f"tcc_driver={props.tcc_driver} unified_addressing={props.unified_addressing}"
        f" memory_pools_supported={props.memory_pools_supported}"
        f" vmm_supported={props.virtual_memory_management_supported}"
        f" concurrent_managed_access={props.concurrent_managed_access}",
    ]
    host_src = LegacyPinnedMemoryResource().allocate(64)
    lines.append("pinned host attrs: " + ", ".join(f"{a}={_attr(a, int(host_src.handle))}" for a in _ATTRS))
    raws = []
    for kind, allocate, raw in _allocators(device, stream):
        try:
            _probe_kind(device, stream, kind, allocate, host_src, lines)
        except Exception as exc:  # diagnostic: record anything
            lines.append(f"[{kind}] PROBE FAILED: {type(exc).__name__}: {str(exc)[:200]}")
        raws.extend(raw)
    for r in raws:
        r.close()
    lines.append("=== END DIAGNOSTIC ===")
    pytest.fail("\n".join(lines), pytrace=False)
