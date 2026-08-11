# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Force the driver's two large address-space reservations up front (issue #2381).

The driver keeps two pools per device that it never gives back, and each one
reserves a virtual address window of roughly twice the installed device memory
when it is first touched:

- the **default device mempool**, reserved by ``cuDeviceGetMemPool``
- the **graph memory pool**, reserved by ``cuGraphAddMemAllocNode`` at node
  creation time

Neither is capped by ``max_size`` and neither is released by
``cuDeviceGraphMemTrim``, ``cuGraphDestroy``, or anything else short of process
exit. On a large-memory GPU with a bounded per-process address space -- 357 GiB
per pool on a 179 GiB device -- the two together consume most of the budget, and
whether the *second* one finds a contiguous range depends on how fragmented the
space has become. That is what makes the full-suite failures intermittent, and
why they always begin at the first test to need whichever pool came second.

Taking both reservations at session start, back to back into a nearly empty
address space, removes test order and accumulated fragmentation from the
question. It does not reduce the footprint; it makes the outcome deterministic.

Measurement here goes through ``cuMemAddressReserve``, not the OS, so it behaves
the same on Windows and Linux.
"""

from __future__ import annotations

import os
import time

from cuda.bindings import driver

MIB = 1024 * 1024
GIB = 1024 * MIB

# cuMemAddressReserve wants a power-of-two alignment and a size that is a
# multiple of it. 2 MiB is the granularity the driver uses for pools.
VA_ALIGNMENT = 2 * MIB

# Each driver-managed pool reserves about this multiple of installed device
# memory. Used to express remaining headroom in units of "one more pool".
POOL_RESERVATION_MULTIPLE = 2

# The probe only has to answer "how much room is left relative to one pool", so
# it never asks for more than this many pools' worth. An absolute ceiling is the
# wrong shape: 64 TiB, the previous value, is meaningless on a machine with room
# to spare and hangs under WSL, where the ask crosses into the host GPU driver.
PROBE_CEILING_POOLS = 4
# Used when the device memory size cannot be read.
DEFAULT_PROBE_CEILING_BYTES = 64 * 1024 * MIB


def align_up(size: int) -> int:
    """Round to a multiple of the reservation alignment.

    Device memory sizes are not generally a multiple of it, and
    cuMemAddressReserve rejects a size that is not with
    CUDA_ERROR_INVALID_VALUE -- which would otherwise read as "no address space
    left" at every size probed.
    """
    return ((size + VA_ALIGNMENT - 1) // VA_ALIGNMENT) * VA_ALIGNMENT


def _reserve_and_release(size: int) -> bool:
    """True if the driver still grants a contiguous reservation of ``size``.

    Reserving costs address space but no memory, so this reads what the address
    space can satisfy without perturbing it. Raises if the release fails, since
    a leaked reservation would corrupt every later measurement.
    """
    err, ptr = driver.cuMemAddressReserve(align_up(size), VA_ALIGNMENT, 0, 0)
    if err != driver.CUresult.CUDA_SUCCESS:
        return False
    (err,) = driver.cuMemAddressFree(ptr, align_up(size))
    if err != driver.CUresult.CUDA_SUCCESS:
        raise RuntimeError(f"cuMemAddressFree({size:#x}) -> {err!r}; address space measurement is unreliable")
    return True


def pool_bytes_for(device_memory: int | None) -> int | None:
    """Address space one driver-managed pool reserves, measured at 2x device memory."""
    if device_memory is None:
        return None
    return align_up(POOL_RESERVATION_MULTIPLE * device_memory)


def probe_ceiling(device_memory: int | None) -> int:
    """Largest size the probe is willing to ask for.

    Expressed in pools rather than absolute bytes: past a few pools' worth the
    exact figure tells nobody anything, and asking for an enormous range is
    where the cost and the WSL hang live.

    Built by multiplying the *rounded* pool size rather than rounding a multiple
    of device memory, so that the ceiling is a whole number of pools. Rounding
    each independently leaves the ceiling a hair under, and the headroom count
    then reports one pool fewer than it should.
    """
    pool = pool_bytes_for(device_memory)
    if pool is None:
        return DEFAULT_PROBE_CEILING_BYTES
    return PROBE_CEILING_POOLS * pool


def largest_reservable(max_bytes: int, *, reserve=None, refine_steps: int = 4) -> int:
    """Largest contiguous reservation the driver still grants, up to ``max_bytes``.

    Returns ``max_bytes`` when even that is granted, so the result is a lower
    bound rather than the true maximum -- which is the point, see probe_ceiling.

    Halves down from ``max_bytes`` rather than doubling up from the granularity,
    because a *refused* reservation allocates nothing and returns immediately
    while releasing a granted one is expensive -- hundreds of milliseconds for a
    large range, and seconds on some configurations. Descending pays that cost
    once instead of once per rung.

    Halving alone only resolves to a power of two, which is too coarse to show
    what the reservations cost when the budget is much larger than they are, so
    ``refine_steps`` bisections then narrow the answer. Each bisection risks one
    more expensive release, hence the small default.

    ``reserve`` is injectable so the search can be tested without a GPU.
    """
    reserve = _reserve_and_release if reserve is None else reserve

    # Halving has to re-align: unlike the old power-of-two ceiling, a ceiling
    # derived from device memory goes odd after a couple of steps, and an
    # unaligned ask is refused outright -- which would read as no space left.
    ceiling = align_up(max_bytes)
    size = ceiling
    refused = None
    while size >= VA_ALIGNMENT and not reserve(size):
        size, refused = (size // 2 // VA_ALIGNMENT) * VA_ALIGNMENT, size
    if size < VA_ALIGNMENT:
        return 0
    if refused is None:
        return ceiling  # granted at the ceiling, so the answer is a lower bound

    low, high = size, refused
    for _ in range(refine_steps):
        middle = ((low + high) // 2 // VA_ALIGNMENT) * VA_ALIGNMENT
        if middle <= low or middle >= high:
            break
        if reserve(middle):
            low = middle
        else:
            high = middle
    return low


def vmm_supported(device_id: int = 0) -> bool:
    """True if this device exposes the virtual memory management APIs."""
    err, dev = driver.cuDeviceGet(device_id)
    if err != driver.CUresult.CUDA_SUCCESS:
        return False
    attribute = driver.CUdevice_attribute.CU_DEVICE_ATTRIBUTE_VIRTUAL_MEMORY_MANAGEMENT_SUPPORTED
    err, supported = driver.cuDeviceGetAttribute(attribute, dev)
    return err == driver.CUresult.CUDA_SUCCESS and bool(supported)


def format_bytes(value: int | None) -> str:
    if value is None:
        return "unknown"
    return f"{value / GIB:.2f} GiB"


class Reservation:
    """One driver-managed pool that has to be materialized."""

    def __init__(self, name: str, detail: str, materialize):
        self.name = name
        self.detail = detail
        self._materialize = materialize
        self.error: str | None = None

    def run(self) -> bool:
        """Materialize the pool. Returns True on success, recording any error."""
        try:
            self._materialize()
        except Exception as exc:  # surfaced to the user by build_failure_message
            self.error = f"{type(exc).__name__}: {exc}"
            return False
        return True

    @property
    def succeeded(self) -> bool:
        return self.error is None


def _materialize_default_mempool(device):
    """Touch the device's default memory pool.

    cuDeviceGetMemPool is the call that makes the reservation; the allocation
    only proves the pool is usable afterwards.
    """
    buffer = device.memory_resource.allocate(8, stream=device.default_stream)
    device.default_stream.sync()
    buffer.close(device.default_stream)
    device.default_stream.sync()


def _materialize_graph_mempool(device):
    """Add one graph memory-allocation node, which reserves the graph pool.

    The graph is discarded immediately: the address reservation it triggers
    outlives it, which is the whole point.
    """
    from cuda.core.graph import GraphDefinition

    definition = GraphDefinition()
    definition.allocate(1024)
    del definition


def _forced_failure():
    """Stand-in for a refused reservation, so the abort path can be exercised.

    Set CUDA_CORE_TEST_FORCE_RESERVATION_FAILURE=1 to check what this suite
    reports on a machine whose address space is too small, without needing one.
    """
    raise RuntimeError("CUDA_ERROR_OUT_OF_MEMORY: simulated refusal (CUDA_CORE_TEST_FORCE_RESERVATION_FAILURE is set)")


def reservations_for(device) -> list[Reservation]:
    if os.environ.get("CUDA_CORE_TEST_FORCE_RESERVATION_FAILURE", "0") not in ("0", ""):
        return [
            Reservation("default device mempool", "cuDeviceGetMemPool", _forced_failure),
            Reservation("graph memory pool", "cuGraphAddMemAllocNode", _forced_failure),
        ]
    return [
        Reservation(
            "default device mempool",
            "cuDeviceGetMemPool",
            lambda: _materialize_default_mempool(device),
        ),
        Reservation(
            "graph memory pool",
            "cuGraphAddMemAllocNode",
            lambda: _materialize_graph_mempool(device),
        ),
    ]


class ReservationReport:
    """What the early reservations cost, for the terminal."""

    def __init__(
        self,
        device_name,
        device_memory,
        before,
        after,
        reservations,
        measured,
        seconds=0.0,
        unsupported=False,
        ceiling=None,
    ):
        self.device_name = device_name
        self.device_memory = device_memory
        self.before = before
        self.after = after
        self.reservations = reservations
        self.measured = measured
        self.seconds = seconds
        self.unsupported = unsupported
        self.ceiling = ceiling

    def _measured_range(self, value) -> str:
        """Render a probe result, flagging it as a lower bound when capped."""
        if value is not None and self.ceiling is not None and value >= self.ceiling:
            return f">= {format_bytes(value)}"
        return format_bytes(value)

    @property
    def failed(self) -> list[Reservation]:
        return [item for item in self.reservations if not item.succeeded]

    @property
    def pool_reservation_bytes(self) -> int | None:
        return pool_bytes_for(self.device_memory)

    def lines(self) -> list[str]:
        out = [f"device 0: {self.device_name} ({format_bytes(self.device_memory)} device memory)"]
        if self.unsupported:
            out.append("device does not support memory pools; nothing to reserve")
            return out
        if self.measured:
            out.append(f"largest reservable range before: {self._measured_range(self.before)}")
        else:
            out.append("largest reservable range: not measured (no virtual memory management support)")

        for item in self.reservations:
            status = "reserved" if item.succeeded else f"FAILED: {item.error}"
            out.append(f"  {item.name:<24} {item.detail:<24} {status}")

        if self.measured:
            out.append(f"largest reservable range after:  {self._measured_range(self.after)}")
            pool_bytes = self.pool_reservation_bytes
            if pool_bytes and self.after is not None:
                capped = self.ceiling is not None and self.after >= self.ceiling
                count = f"{self.after // pool_bytes}{'+' if capped else ''}"
                out.append(
                    f"remaining headroom: {count} more pool-sized "
                    f"({format_bytes(pool_bytes)}) reservations  [{self.seconds:.1f}s measuring]"
                )
        return out


def build_failure_message(report: ReservationReport) -> str:
    """Explain, for a human, why this machine cannot run the suite."""
    pool_bytes = report.pool_reservation_bytes
    failed = ", ".join(item.name for item in report.failed)
    lines = [
        "",
        "cuda_core tests cannot run on this machine: the CUDA driver could not reserve",
        f"address space for {failed}.",
        "",
        f"  device 0                        {report.device_name}",
        f"  installed device memory         {format_bytes(report.device_memory)}",
        f"  needed per driver-managed pool  {format_bytes(pool_bytes)} of *virtual address space*",
        f"  largest range still available   {report._measured_range(report.after) if report.measured else 'unknown'}",
        "",
    ]
    for item in report.failed:
        lines.append(f"  {item.name} ({item.detail}): {item.error}")

    return "\n".join(lines)


def reserve_driver_pools(device, measure: bool = True) -> ReservationReport:
    """Materialize both driver-managed pools, measuring address space around them.

    Both pools require mempool support, so on a device without it there is
    nothing to reserve and nothing to pre-empt. Skip rather than fail: the tests
    that need pools skip themselves on such a device, and the rest still run.
    """
    device_memory = None
    err, _free, total = driver.cuMemGetInfo()
    if err == driver.CUresult.CUDA_SUCCESS:
        device_memory = int(total)

    if not device.properties.memory_pools_supported:
        return ReservationReport(device.name, device_memory, None, None, [], measured=False, unsupported=True)

    measured = measure and vmm_supported(device.device_id)
    ceiling = probe_ceiling(device_memory)
    started = time.perf_counter()
    before = largest_reservable(ceiling) if measured else None
    elapsed = time.perf_counter() - started

    reservations = reservations_for(device)
    for item in reservations:
        item.run()

    started = time.perf_counter()
    after = largest_reservable(ceiling) if measured else None
    elapsed += time.perf_counter() - started
    return ReservationReport(
        device.name, device_memory, before, after, reservations, measured, elapsed, ceiling=ceiling
    )
