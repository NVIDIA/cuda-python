# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""OOM reason checker for the cuda_core test suite (issue #2381).

``CUDA_ERROR_OUT_OF_MEMORY`` does not mean "the device ran out of memory": the
driver returns it whenever it could not obtain some resource, and one of
those resources is a purely host-side virtual-address (VA) reservation.
Creating -- or even just looking up -- a memory pool reserves a VA window
before any device memory is touched (observed default: about 2x installed
device memory). If that host reservation fails, the error looks identical to
genuine physical exhaustion even though ``cuMemGetInfo`` may report the
device is almost entirely free. This module runs a small, ordered sequence of
``cuda.bindings.driver`` calls to tell the two apart, and turns the raw
results into a verdict a reader does not have to reconstruct by hand.

Every probe uses only documented driver APIs against the current context and
device 0, so the same sequence runs unmodified on Linux, Windows (WDDM / TCC /
MCDM), and WSL. There is no OS-specific branch and no external process
(``nvidia-smi``, NVML): see the PR #2458 review for why the previous version's
``nvidia-smi -q`` dump was dropped.

Capture is latched once per session, on the first failing OOM. A failing run
can report ~190 of these, all descending from one earlier event, and
re-running the probes on each would add noticeable time and bury the log. The
report is written via ``terminalreporter`` rather than ``print()`` so it
survives the stdout redirection these runs use, and ``pytest_terminal_summary``
points at the artifact from the end of the run, where it is actually noticed.
"""

import os
import pathlib
import sys
import threading
from dataclasses import dataclass

from cuda.bindings import driver
from helpers.constants import POOL_SIZE

OOM_MARKER = "CUDA_ERROR_OUT_OF_MEMORY"
DEFAULT_FILENAME = "cuda_core_oom_diagnostics.txt"

GIB = 1 << 30
# Used only if cuMemGetAllocationGranularity is unavailable; see _allocation_granularity.
FALLBACK_ALIGNMENT = 2 * 1024 * 1024

_BANNER = "=" * 78

_LESSON = (
    "CUDA_ERROR_OUT_OF_MEMORY means the driver could not obtain some resource;\n"
    "it is not proof that device memory is exhausted. Creating -- or even just\n"
    "looking up -- a memory pool first reserves a host virtual-address window\n"
    "(observed default: about 2x installed device memory) before any device\n"
    "memory is touched. That reservation can fail while cuMemGetInfo still\n"
    "reports most of the device free. The probes below check host VA and\n"
    "physical device memory separately so the two are not confused. Note that\n"
    "the '2x device memory' figure is an observation from measurement and a\n"
    "driver source comment, not a documented guarantee -- it can differ across\n"
    "driver versions and platforms."
)


def _round_up(value, alignment):
    if alignment <= 0:
        return value
    remainder = value % alignment
    return value if remainder == 0 else value + (alignment - remainder)


def _call(fn, *args):
    """Invoke a driver API and split the result into ``(ok, error, values)``.

    Never raises: an exception here would replace the original test failure
    that this module exists to diagnose.
    """
    try:
        result = fn(*args)
    except Exception as exc:
        return False, repr(exc), ()
    err, *values = result if isinstance(result, tuple) else (result,)
    if err != driver.CUresult.CUDA_SUCCESS:
        return False, str(err), tuple(values)
    return True, None, tuple(values)


def _allocation_granularity(device_id):
    """Best-effort VMM alignment for the current device; falls back to 2 MiB.

    Using the driver's own recommended granularity keeps alignment portable
    instead of assuming every platform pads to 2 MiB.
    """
    prop = driver.CUmemAllocationProp()
    prop.type = driver.CUmemAllocationType.CU_MEM_ALLOCATION_TYPE_PINNED
    prop.location.type = driver.CUmemLocationType.CU_MEM_LOCATION_TYPE_DEVICE
    prop.location.id = device_id
    ok, _err, values = _call(
        driver.cuMemGetAllocationGranularity,
        prop,
        driver.CUmemAllocationGranularity_flags.CU_MEM_ALLOC_GRANULARITY_RECOMMENDED,
    )
    if ok and values and values[0]:
        return values[0]
    return FALLBACK_ALIGNMENT


def _reserve_and_free(size, alignment):
    """Reserve, then immediately free, a host VA window of ``size`` bytes.

    Returns ``(ok, error)``. Never leaves the reservation live: a probe that
    holds VA open would itself become the leak this module diagnoses.
    """
    aligned_size = _round_up(size, alignment)
    ok, err, values = _call(driver.cuMemAddressReserve, aligned_size, alignment, None, 0)
    if not ok:
        return False, err
    (ptr,) = values
    free_ok, free_err, _ = _call(driver.cuMemAddressFree, ptr, aligned_size)
    if not free_ok:
        # Report success (the reservation itself answered the question) but
        # note the leak rather than retrying the free from diagnostics code.
        return True, f"reserved but cuMemAddressFree reported {free_err}"
    return True, None


@dataclass
class ProbeSnapshot:
    """Raw results of one pass of the OOM reason checker.

    Fields default to ``None`` ("not probed") rather than ``False``, because a
    later step is skipped -- not failed -- once an earlier, cheaper step
    already answers the question. For example, a pool-sized VA reservation is
    never attempted once a single-granularity reservation has already failed.
    """

    has_context: bool = False
    context_error: str | None = None

    mem_free: int | None = None
    mem_total: int | None = None
    mem_get_info_error: str | None = None

    mempools_supported: bool | None = None
    vmm_supported: bool | None = None

    small_alloc_ok: bool | None = None
    small_alloc_error: str | None = None

    granularity: int | None = None
    small_va_ok: bool | None = None
    small_va_error: str | None = None

    pool_va_size: int | None = None
    pool_va_ok: bool | None = None
    pool_va_error: str | None = None

    get_mem_pool_ok: bool | None = None
    get_mem_pool_error: str | None = None
    get_default_mem_pool_ok: bool | None = None
    get_default_mem_pool_error: str | None = None

    capped_pool_create_ok: bool | None = None
    capped_pool_create_error: str | None = None


def probe_basics():
    """Run the cheap, side-effect-free prefix of the OOM reason checker.

    Covers steps 1-4 below: context, physical free/total memory, device
    attributes, and one small physical allocation. None of these touch host
    VA or memory pools, so this is safe to call from an ordinary (non-OOM)
    test as a live smoke check. Returns ``(snapshot, dev_or_None)``; ``dev``
    is threaded through to :func:`run_probe` so it does not have to re-derive
    it.

    See :func:`run_probe` for the full, numbered probe sequence.
    """
    snapshot = ProbeSnapshot()

    ctx_ok, ctx_err, ctx_values = _call(driver.cuCtxGetCurrent)
    # cuCtxGetCurrent succeeds even with no bound context: it returns a null
    # CUcontext, not None, so "no context" has to be checked via int(), not
    # an identity check against None.
    snapshot.has_context = bool(ctx_ok and ctx_values and ctx_values[0] is not None and int(ctx_values[0]) != 0)
    if not ctx_ok:
        snapshot.context_error = ctx_err
    if not snapshot.has_context:
        return snapshot, None

    mem_ok, mem_err, mem_values = _call(driver.cuMemGetInfo)
    if not mem_ok:
        snapshot.mem_get_info_error = mem_err
        return snapshot, None
    snapshot.mem_free, snapshot.mem_total = mem_values

    count_ok, _count_err, count_values = _call(driver.cuDeviceGetCount)
    if not count_ok or not count_values or count_values[0] < 1:
        return snapshot, None
    dev_ok, _dev_err, dev_values = _call(driver.cuDeviceGet, 0)
    if not dev_ok:
        return snapshot, None
    (dev,) = dev_values

    pools_ok, _e1, pools_values = _call(
        driver.cuDeviceGetAttribute, driver.CUdevice_attribute.CU_DEVICE_ATTRIBUTE_MEMORY_POOLS_SUPPORTED, dev
    )
    snapshot.mempools_supported = bool(pools_values[0]) if pools_ok else None

    vmm_ok, _e2, vmm_values = _call(
        driver.cuDeviceGetAttribute,
        driver.CUdevice_attribute.CU_DEVICE_ATTRIBUTE_VIRTUAL_MEMORY_MANAGEMENT_SUPPORTED,
        dev,
    )
    snapshot.vmm_supported = bool(vmm_values[0]) if vmm_ok else None

    alloc_ok, alloc_err, alloc_values = _call(driver.cuMemAlloc, 4096)
    snapshot.small_alloc_ok = alloc_ok
    if alloc_ok:
        (ptr,) = alloc_values
        _call(driver.cuMemFree, ptr)
    else:
        snapshot.small_alloc_error = alloc_err

    return snapshot, dev


def run_probe():
    """Run the OOM reason checker once and return its :class:`ProbeSnapshot`.

    Ordered from cheapest/most-general to most-specific so a probe that
    already answers the question skips the more expensive ones after it:

    1-4. :func:`probe_basics` -- context, ``cuMemGetInfo``, device
       attributes (are mempools / VMM even supported here?), and a small
       ``cuMemAlloc`` (the legacy physical allocator, not a pool).
    5. A single-granularity ``cuMemAddressReserve`` -- the cheapest possible
       host VA probe. If this fails, nothing bigger can succeed either.
    6. A pool-sized (~2x device memory) ``cuMemAddressReserve`` -- the same
       size a default or uncapped pool would need.
    7. ``cuDeviceGetMemPool`` / ``cuDeviceGetDefaultMemPool`` -- note that,
       unlike step 6, a *successful* call here is not undone: querying the
       default pool performs this reservation for real and for the rest of
       the process's life, whether or not this checker ever ran.
    8. A capped ``cuMemPoolCreate(maxSize=POOL_SIZE)`` -- distinguishes "no
       pool at all fits" from "only the ~2x default-sized window does not".
    """
    snapshot, dev = probe_basics()
    if dev is None:
        return snapshot
    device_id = int(dev)

    if not snapshot.vmm_supported:
        return snapshot

    granularity = _allocation_granularity(device_id)
    snapshot.granularity = granularity

    small_ok, small_err = _reserve_and_free(granularity, granularity)
    snapshot.small_va_ok = small_ok
    if not small_ok:
        snapshot.small_va_error = small_err
        return snapshot  # A bigger reservation cannot succeed if this one did not.

    if not snapshot.mempools_supported:
        return snapshot

    pool_window = _round_up(2 * snapshot.mem_total, granularity)
    snapshot.pool_va_size = pool_window
    pool_ok, pool_err = _reserve_and_free(pool_window, granularity)
    snapshot.pool_va_ok = pool_ok
    if not pool_ok:
        snapshot.pool_va_error = pool_err

    # Unlike the reserve/free probe above, a successful call here is real and
    # permanent: it is the same reservation cuda.core's DeviceMemoryResource
    # would trigger. It is deliberately still probed, because it is the exact
    # call that failed in the original #2381 logs.
    get_ok, get_err, _v1 = _call(driver.cuDeviceGetMemPool, dev)
    snapshot.get_mem_pool_ok = get_ok
    if not get_ok:
        snapshot.get_mem_pool_error = get_err

    default_ok, default_err, _v2 = _call(driver.cuDeviceGetDefaultMemPool, dev)
    snapshot.get_default_mem_pool_ok = default_ok
    if not default_ok:
        snapshot.get_default_mem_pool_error = default_err

    props = driver.CUmemPoolProps()
    props.allocType = driver.CUmemAllocationType.CU_MEM_ALLOCATION_TYPE_PINNED
    props.handleTypes = driver.CUmemAllocationHandleType.CU_MEM_HANDLE_TYPE_NONE
    props.location.type = driver.CUmemLocationType.CU_MEM_LOCATION_TYPE_DEVICE
    props.location.id = device_id
    props.maxSize = POOL_SIZE  # Never 0: an uncapped pool reserves another ~2x window.
    create_ok, create_err, create_values = _call(driver.cuMemPoolCreate, props)
    snapshot.capped_pool_create_ok = create_ok
    if create_ok:
        (pool,) = create_values
        _call(driver.cuMemPoolDestroy, pool)
    else:
        snapshot.capped_pool_create_error = create_err

    return snapshot


def classify(snapshot: ProbeSnapshot) -> str:
    """Turn a probe snapshot into a one-line, human-readable verdict.

    Pure function of the snapshot, so this can -- and should -- be tested
    without a GPU. Checks are ordered from "nothing could be probed" to
    "everything probed fine", each returning as soon as it names something
    specific enough to act on.
    """
    if not snapshot.has_context:
        return "no current CUDA context; cannot narrow down the reason"

    if snapshot.mem_get_info_error is not None:
        return f"cuMemGetInfo itself failed ({snapshot.mem_get_info_error}); the device may be unavailable"

    if snapshot.small_alloc_ok is False:
        return "likely physical device memory exhaustion: a small cuMemAlloc failed outright"

    low_free = (
        snapshot.mem_total is not None
        and snapshot.mem_free is not None
        and snapshot.mem_total > 0
        and (snapshot.mem_free / snapshot.mem_total) < 0.05
    )
    if low_free:
        return "likely physical device memory exhaustion: cuMemGetInfo reports under 5% free"

    if snapshot.small_va_ok is False:
        return "host virtual-address space is exhausted even for a single allocation-granularity window"

    if snapshot.mempools_supported is False:
        return "mempools are not supported on this device; the OOM is unrelated to memory pools"

    if snapshot.pool_va_ok is False:
        if snapshot.capped_pool_create_ok:
            return (
                "likely host VA exhaustion for the pool-sized window only: a capped "
                "memory pool (helpers.constants.POOL_SIZE) still creates fine, but a "
                "reservation the size of the observed default pool window does not"
            )
        return "likely host VA exhaustion: a pool-sized reservation failed while device memory is mostly free"

    if snapshot.get_mem_pool_ok is False or snapshot.get_default_mem_pool_ok is False:
        return (
            "default mempool materialization failed even though an equivalently sized "
            "standalone VA reservation just succeeded; the OOM may be from a resource "
            "other than host VA or physical device memory"
        )

    if snapshot.capped_pool_create_ok is False:
        return "a capped memory pool create failed even though larger probes above succeeded; inconclusive"

    return "inconclusive: all probes succeeded; the original failure may have been transient or another process holding resources"


def format_probe_log(snapshot: ProbeSnapshot) -> str:
    """Render the raw probe results, independent of the verdict."""
    lines = ["--- direct driver probe (bypasses cuda.core's error reporting) ---"]

    if not snapshot.has_context:
        suffix = f" <raised {snapshot.context_error}>" if snapshot.context_error else " (no context)"
        lines.append(f"cuCtxGetCurrent(){suffix}")
        return "\n".join(lines)
    lines.append("cuCtxGetCurrent() -> ok")

    if snapshot.mem_get_info_error is not None:
        lines.append(f"cuMemGetInfo() -> <failed: {snapshot.mem_get_info_error}>")
        return "\n".join(lines)
    free_frac = snapshot.mem_free / snapshot.mem_total if snapshot.mem_total else float("nan")
    lines.append(
        f"cuMemGetInfo() -> free={snapshot.mem_free / GIB:.2f} GiB, "
        f"total={snapshot.mem_total / GIB:.2f} GiB ({free_frac:.1%} free)"
    )
    lines.append(f"mempools supported: {snapshot.mempools_supported}")
    lines.append(f"VMM (cuMemAddressReserve) supported: {snapshot.vmm_supported}")
    lines.append(
        "small cuMemAlloc(4 KiB) -> "
        + ("ok, freed" if snapshot.small_alloc_ok else f"<failed: {snapshot.small_alloc_error}>")
    )

    if not snapshot.vmm_supported:
        lines.append("(VMM unsupported: skipping host VA reservation probes)")
        return "\n".join(lines)

    lines.append(f"allocation granularity: {snapshot.granularity} bytes")
    lines.append(
        f"cuMemAddressReserve({snapshot.granularity} bytes) -> "
        + ("ok, freed" if snapshot.small_va_ok else f"<failed: {snapshot.small_va_error}>")
    )
    if not snapshot.small_va_ok:
        return "\n".join(lines)

    if not snapshot.mempools_supported:
        lines.append("(mempools unsupported: skipping pool-related probes)")
        return "\n".join(lines)

    lines.append(
        f"cuMemAddressReserve({snapshot.pool_va_size} bytes, observed default pool window, "
        "not a documented guarantee) -> "
        + ("ok, freed" if snapshot.pool_va_ok else f"<failed: {snapshot.pool_va_error}>")
    )
    lines.append(
        "cuDeviceGetMemPool(dev 0) -> "
        + (
            "ok -- NOTE: this permanently reserves the window above, for the rest of "
            "this process, if it was not already reserved"
            if snapshot.get_mem_pool_ok
            else f"<failed: {snapshot.get_mem_pool_error}>"
        )
    )
    lines.append(
        "cuDeviceGetDefaultMemPool(dev 0) -> "
        + ("ok" if snapshot.get_default_mem_pool_ok else f"<failed: {snapshot.get_default_mem_pool_error}>")
    )
    lines.append(
        f"cuMemPoolCreate(maxSize={POOL_SIZE}) -> "
        + ("ok, destroyed" if snapshot.capped_pool_create_ok else f"<failed: {snapshot.capped_pool_create_error}>")
    )
    return "\n".join(lines)


class OomDiagnosticsRecorder:
    """Captures the OOM reason checker the first time a CUDA OOM is seen, and only then."""

    def __init__(self, filename=DEFAULT_FILENAME):
        self._filename = filename
        self._lock = threading.Lock()
        self._captured = False
        self._nodeid = None
        self._artifact_path = None
        self._artifact_written = False

    @property
    def captured(self):
        return self._captured

    @property
    def nodeid(self):
        """Node id of the test that triggered capture, or None."""
        return self._nodeid

    @property
    def artifact_path(self):
        """Where the report was written, or None if nothing was captured."""
        return self._artifact_path

    @property
    def artifact_written(self):
        return self._artifact_written

    @staticmethod
    def matches(exc_text):
        return OOM_MARKER in exc_text

    def build_report(self, nodeid, phase, exc_text, snapshot=None):
        if snapshot is None:
            snapshot = run_probe()
        verdict = classify(snapshot)
        return "\n".join(
            [
                _BANNER,
                "cuda_core OOM reason checker: first CUDA_ERROR_OUT_OF_MEMORY of this session",
                _BANNER,
                f"test:      {nodeid}",
                f"phase:     {phase}",
                f"pid:       {os.getpid()}",
                f"platform:  {sys.platform}",
                f"exception: {exc_text}",
                "",
                _LESSON,
                "",
                format_probe_log(snapshot),
                "",
                f"verdict: {verdict}",
                _BANNER,
            ]
        )

    def capture(self, nodeid, phase, exc_text, directory, snapshot=None):
        """Build and persist the report. Returns None if already captured.

        ``snapshot`` lets callers (mainly tests) skip the live driver probe;
        production callers leave it unset so :meth:`build_report` runs
        :func:`run_probe`.
        """
        with self._lock:
            if self._captured:
                return None
            self._captured = True

        report = self.build_report(nodeid, phase, exc_text, snapshot=snapshot)
        destination = pathlib.Path(directory) / self._filename
        self._nodeid = nodeid
        self._artifact_path = destination
        try:
            destination.write_text(report, encoding="utf-8")
            self._artifact_written = True
            return f"{report}\n(diagnostics also written to {destination})"
        except OSError as exc:
            return f"{report}\n(could not write {destination}: {exc!r})"


_default_recorder = OomDiagnosticsRecorder()


def record_if_oom(item, call, report, recorder=None, snapshot=None):
    """Capture diagnostics when ``report`` is the session's first CUDA OOM.

    ``recorder`` defaults to a module-level singleton so the conftest hook does
    not have to hold session state; tests pass their own to stay isolated.
    ``snapshot`` is likewise test-only; see :meth:`OomDiagnosticsRecorder.capture`.

    Returns the emitted text, or None when nothing was captured.
    """
    if recorder is None:
        recorder = _default_recorder

    if recorder.captured or not report.failed or call.excinfo is None:
        return None

    exc_text = str(call.excinfo.value)
    if not recorder.matches(exc_text):
        return None

    text = recorder.capture(item.nodeid, call.when, exc_text, item.config.rootpath, snapshot=snapshot)
    if text is None:
        return None

    # terminalreporter writes outside pytest's stdout capture, so this survives
    # into a redirected log; a bare print() would not.
    terminal_reporter = item.config.pluginmanager.get_plugin("terminalreporter")
    if terminal_reporter is not None:
        terminal_reporter.write_line("")
        terminal_reporter.write_line(text)
    return text


def report_terminal_summary(terminalreporter, recorder=None):
    """Point at the diagnostics artifact from pytest's end-of-run summary.

    The report itself is emitted beside the failing test, which in a real
    failing run is thousands of lines above the summary people actually read.

    Returns the emitted line, or None when nothing was captured.
    """
    if recorder is None:
        recorder = _default_recorder

    if not recorder.captured:
        return None

    verb = "written to" if recorder.artifact_written else "could NOT be written to"
    line = f"first CUDA OOM at {recorder.nodeid}; diagnostics {verb} {recorder.artifact_path}"
    terminalreporter.write_sep("=", "cuda_core OOM diagnostics", red=True)
    terminalreporter.write_line(line)
    return line
