# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

# Environment fingerprint taken around `import cuda.core`.
#
# docs/WINDOWS_COVERAGE_SEGFAULT.md section 19.1 reduced the nightly crash to a
# single import under coverage, and section 17.7 ranked what still differs
# between the NVIDIA runner and ours.  Nothing in that ranking has ever been
# measured on our side beyond the A100 box's `nvidia-smi`, so this script
# records the whole comparison surface in one pass:
#
#   * host: OS build, CPU, RAM, account, session id
#   * GPU: driver version and TCC/MCDM mode
#   * tracer: which coverage core is actually installed once tracing starts --
#     if it is not ctrace, `__Pyx_TraceLine` never calls back into Python and
#     every "we cannot reproduce it" run so far was testing the wrong thing
#   * modules: the loaded DLL map before and after the import, with base
#     addresses, which is the direct fingerprint for a fault that happens in
#     extension-module initialisation
#
# The module delta also answers a question the source could not: whether
# `import cuda.core` pulls in nvcuda.dll at all.  If it does not, the driver
# version and TCC/MCDM mode drop out of the suspect list and the repro matrix
# opens up to any Windows box.
#
# Run it the same way the minimal repro runs:
#
#   coverage run --rcfile=<repo>/.coveragerc env_fingerprint.py [out.txt]
#
# Exits 0 whether or not the import succeeds -- a crash is a native fault that
# kills the process outright, so any Python-level exception is reported and
# then swallowed, keeping the fingerprint itself readable.

from __future__ import annotations

import ctypes
import ctypes.wintypes as wintypes
import os
import platform
import struct
import subprocess
import sys
import threading
import time

_out_lines: list[str] = []


def emit(line: str = "") -> None:
    print(line, flush=True)
    _out_lines.append(line)


def section(title: str) -> None:
    emit()
    emit(f"===== {title} =====")


def safe(title: str, fn) -> None:
    """Run one collector.  A collector that fails must not lose the rest."""
    section(title)
    try:
        fn()
    except Exception as exc:  # diagnostics must never abort
        emit(f"<failed: {type(exc).__name__}: {exc}>")


# --------------------------------------------------------------------------
# Win32 helpers
# --------------------------------------------------------------------------

_kernel32 = ctypes.WinDLL("kernel32", use_last_error=True)
_psapi = ctypes.WinDLL("psapi", use_last_error=True)
_advapi32 = ctypes.WinDLL("advapi32", use_last_error=True)


class MODULEINFO(ctypes.Structure):
    _fields_ = [
        ("lpBaseOfDll", ctypes.c_void_p),
        ("SizeOfImage", wintypes.DWORD),
        ("EntryPoint", ctypes.c_void_p),
    ]


class MEMORYSTATUSEX(ctypes.Structure):
    _fields_ = [
        ("dwLength", wintypes.DWORD),
        ("dwMemoryLoad", wintypes.DWORD),
        ("ullTotalPhys", ctypes.c_ulonglong),
        ("ullAvailPhys", ctypes.c_ulonglong),
        ("ullTotalPageFile", ctypes.c_ulonglong),
        ("ullAvailPageFile", ctypes.c_ulonglong),
        ("ullTotalVirtual", ctypes.c_ulonglong),
        ("ullAvailVirtual", ctypes.c_ulonglong),
        ("ullAvailExtendedVirtual", ctypes.c_ulonglong),
    ]


_kernel32.GetCurrentProcess.restype = wintypes.HANDLE
_psapi.EnumProcessModules.argtypes = [
    wintypes.HANDLE,
    ctypes.POINTER(ctypes.c_void_p),
    wintypes.DWORD,
    ctypes.POINTER(wintypes.DWORD),
]
_psapi.GetModuleFileNameExW.argtypes = [
    wintypes.HANDLE,
    ctypes.c_void_p,
    wintypes.LPWSTR,
    wintypes.DWORD,
]
_psapi.GetModuleInformation.argtypes = [
    wintypes.HANDLE,
    ctypes.c_void_p,
    ctypes.POINTER(MODULEINFO),
    wintypes.DWORD,
]


def loaded_modules() -> dict[str, tuple[int, int]]:
    """path -> (base address, image size) for every DLL mapped right now."""
    handle = _kernel32.GetCurrentProcess()
    # Ask twice: the first call reports how much room the list actually needs.
    needed = wintypes.DWORD()
    arr = (ctypes.c_void_p * 1024)()
    if not _psapi.EnumProcessModules(handle, arr, ctypes.sizeof(arr), ctypes.byref(needed)):
        raise ctypes.WinError(ctypes.get_last_error())
    count = needed.value // ctypes.sizeof(ctypes.c_void_p)
    if count > len(arr):
        arr = (ctypes.c_void_p * count)()
        if not _psapi.EnumProcessModules(handle, arr, ctypes.sizeof(arr), ctypes.byref(needed)):
            raise ctypes.WinError(ctypes.get_last_error())
        count = needed.value // ctypes.sizeof(ctypes.c_void_p)

    out: dict[str, tuple[int, int]] = {}
    name = ctypes.create_unicode_buffer(32768)
    for i in range(count):
        hmod = arr[i]
        if not _psapi.GetModuleFileNameExW(handle, hmod, name, len(name)):
            continue
        info = MODULEINFO()
        if not _psapi.GetModuleInformation(handle, hmod, ctypes.byref(info), ctypes.sizeof(info)):
            continue
        out[name.value] = (info.lpBaseOfDll or 0, info.SizeOfImage)
    return out


def format_modules(mods: dict[str, tuple[int, int]]) -> list[str]:
    rows = sorted(mods.items(), key=lambda kv: kv[1][0])
    return [f"0x{base:016x}  {size:>9}  {path}" for path, (base, size) in rows]


# --------------------------------------------------------------------------
# Collectors
# --------------------------------------------------------------------------


def collect_host() -> None:
    emit(f"hostname        = {platform.node()}")
    emit(f"platform        = {platform.platform()}")
    emit(f"win32_ver       = {platform.win32_ver()}")
    emit(f"win32_edition   = {platform.win32_edition()}")
    emit(f"is_server_core  = {platform.win32_is_iot()!r} (win32_is_iot; edition string above is the real signal)")
    emit(f"machine         = {platform.machine()}")
    emit(f"processor       = {platform.processor()}")
    emit(f"cpu_count       = {os.cpu_count()}")

    status = MEMORYSTATUSEX()
    status.dwLength = ctypes.sizeof(MEMORYSTATUSEX)
    if _kernel32.GlobalMemoryStatusEx(ctypes.byref(status)):
        gb = 1024**3
        emit(f"ram_total_gb    = {status.ullTotalPhys / gb:.1f}")
        emit(f"ram_avail_gb    = {status.ullAvailPhys / gb:.1f}")

    user = ctypes.create_unicode_buffer(256)
    size = wintypes.DWORD(len(user))
    if _advapi32.GetUserNameW(user, ctypes.byref(size)):
        emit(f"account         = {user.value}")

    session = wintypes.DWORD()
    if _kernel32.ProcessIdToSessionId(os.getpid(), ctypes.byref(session)):
        # The nightly runs as a service; session 0 means no interactive desktop.
        emit(f"session_id      = {session.value}")

    emit(f"cwd             = {os.getcwd()}")


def collect_gpu() -> None:
    # -q carries the fields the plain table hides: the TCC/MCDM mode, and on
    # r600+ the split KMD / CUDA UMD versions (see section 17.6).
    proc = subprocess.run(
        ["nvidia-smi", "-q"],  # noqa: S607 - resolved via PATH on purpose
        capture_output=True,
        text=True,
        timeout=120,
        check=False,
    )
    if proc.returncode != 0:
        emit(f"<nvidia-smi -q exited {proc.returncode}>")
        emit(proc.stderr.strip()[:500])
        return
    wanted = (
        "Driver Version",
        "KMD Version",
        "CUDA Version",
        "CUDA UMD Version",
        "Product Name",
        "Compute Mode",
        "GPU UUID",
    )
    # "Current"/"Pending" appear under a dozen unrelated headings (temperature,
    # page retirement, PCIe width), so they are only taken while inside the
    # Driver Model block -- that pair is the TCC/MCDM/WDDM answer.
    in_driver_model = 0
    for line in proc.stdout.splitlines():
        stripped = line.strip()
        if stripped.startswith("Driver Model"):
            emit(line.rstrip())
            in_driver_model = 2
            continue
        if in_driver_model and (stripped.startswith(("Current", "Pending"))):
            emit(line.rstrip())
            in_driver_model -= 1
            continue
        in_driver_model = 0
        if any(key in line for key in wanted):
            emit(line.rstrip())


def pe_stack_reserve(path: str) -> tuple[int, int]:
    """(reserve, commit) for the main thread, from the executable's PE header.

    The main thread's stack is whatever the linker wrote here; only threads
    created later can pick their own size.  python.org builds link with
    /STACK:3000000 (2.86 MB); other redistributions need not, and a Cython
    linetrace build's module-init frames are large enough for the difference
    to decide whether `import cuda.core` survives.
    """
    with open(path, "rb") as fh:
        data = fh.read(1024)
    pe = struct.unpack_from("<I", data, 0x3C)[0]
    magic = struct.unpack_from("<H", data, pe + 24)[0]
    fmt = "<QQ" if magic == 0x20B else "<II"
    return struct.unpack_from(fmt, data, pe + 24 + 72)


def collect_python() -> None:
    emit(f"executable      = {sys.executable}")
    emit(f"version         = {sys.version}")
    emit(f"bits            = {ctypes.sizeof(ctypes.c_void_p) * 8}")
    try:
        reserve, commit = pe_stack_reserve(sys.executable)
        emit(f"main_stack_mb   = {reserve / 1024 / 1024:.2f}  (PE SizeOfStackReserve)")
        emit(f"main_stack_commit_kb = {commit / 1024:.0f}")
    except Exception as exc:
        emit(f"main_stack_mb   = <{type(exc).__name__}: {exc}>")
    current = threading.current_thread()
    emit(f"thread          = {current.name} (main={current is threading.main_thread()})")
    emit(f"thread_stack_mb = {threading.stack_size() / 1024 / 1024:.2f}  (0.00 = OS default)")
    for dist in ("coverage", "Cython", "pytest", "numpy"):
        try:
            from importlib.metadata import version as _version

            emit(f"{dist:<15} = {_version(dist)}")
        except Exception as exc:
            emit(f"{dist:<15} = <{type(exc).__name__}>")


def collect_wheel() -> None:
    """Prove the installed cuda.core is a linetrace build, without importing it.

    Necessary condition 1 (section 4) is the cov wheel, and a package name in
    the registry is not evidence of it -- a nocov set has been published under
    the cov name before.  Two physical markers settle it: setup.py's build_py
    only ships .pyx/.cpp alongside the .pyd when CUDA_PYTHON_COVERAGE=1, and
    the generated .cpp carries the compile flags in its Cython Metadata header.

    Uses find_spec so the parent namespace package is all that gets imported;
    `import cuda.core` is the thing that crashes, and it happens later.
    """
    import importlib.util

    spec = importlib.util.find_spec("cuda.core")
    if spec is None or not spec.submodule_search_locations:
        emit("<cuda.core not found on sys.path>")
        return
    pkg_dir = next(iter(spec.submodule_search_locations))
    emit(f"package_dir     = {pkg_dir}")

    for name in ("_stream.pyx", "_stream.cpp", "_resource_handles.pyx"):
        path = os.path.join(pkg_dir, name)
        emit(f"{name:<22} {'present' if os.path.exists(path) else 'MISSING'}")

    cpp = os.path.join(pkg_dir, "_stream.cpp")
    if os.path.exists(cpp):
        # The metadata block is the first ~30 lines; the flags live in it.
        with open(cpp, encoding="utf-8", errors="replace") as fh:
            head = [next(fh, "") for _ in range(40)]
        for line in head:
            if "CYTHON_TRACE" in line or "std:c++" in line or "USE_SYS_MONITORING" in line:
                emit(f"compile_arg     = {line.strip()}")
    emit(
        "linetrace_build = "
        + str(os.path.exists(os.path.join(pkg_dir, "_stream.pyx")))
        + "  (sources shipped == built with CUDA_PYTHON_COVERAGE=1)"
    )


def collect_tracer() -> None:
    """Which tracer is armed -- the whole crash hinges on it being ctrace.

    The cov wheels are built with CYTHON_USE_SYS_MONITORING=0, so the
    `__Pyx_TraceLine` hooks only fire through the legacy sys.settrace path.  If
    coverage quietly selected sysmon here, our runs never armed the trigger and
    every negative result on this runner means nothing.
    """
    emit(f"sys.gettrace()  = {sys.gettrace()!r}")
    emit(f"sys.monitoring in use = {getattr(sys, 'monitoring', None) is not None}")
    try:
        import coverage

        cov = coverage.Coverage.current()
        emit(f"coverage.current() = {cov!r}")
        if cov is not None:
            collector = getattr(cov, "_collector", None)
            emit(f"collector       = {collector!r}")
            for attr in ("tracer_name", "core_name"):
                fn = getattr(collector, attr, None)
                if callable(fn):
                    emit(f"{attr}()      = {fn()}")
                elif fn is not None:
                    emit(f"{attr}        = {fn}")
            emit(f"plugins         = {list(getattr(cov, '_plugins', []) or [])}")
    except Exception as exc:
        emit(f"<coverage introspection failed: {type(exc).__name__}: {exc}>")


def run(out_path: str | None) -> None:
    emit("cuda.core import fingerprint")
    emit(f"timestamp       = {time.strftime('%Y-%m-%d %H:%M:%S')}")
    emit(f"argv            = {sys.argv}")

    safe("host", collect_host)
    safe("gpu", collect_gpu)
    safe("python", collect_python)
    safe("wheel", collect_wheel)
    safe("tracer (before import)", collect_tracer)

    before = loaded_modules()
    section("modules before import")
    emit(f"count = {len(before)}")

    section("import cuda.core")
    started = time.perf_counter()
    failure = None
    try:
        import cuda.core

        emit(f"import OK, version = {cuda.core.__version__}")
    except BaseException as exc:  # report, do not propagate
        failure = exc
        emit(f"import FAILED: {type(exc).__name__}: {exc}")
    emit(f"elapsed_s = {time.perf_counter() - started:.3f}")

    after = loaded_modules()
    section("modules loaded by the import")
    new = {path: info for path, info in after.items() if path not in before}
    emit(f"count before = {len(before)}, after = {len(after)}, new = {len(new)}")
    for row in format_modules(new):
        emit(row)

    section("driver DLLs anywhere in the process")
    # F1: does the import touch the driver at all?  If nvcuda.dll never shows
    # up, driver version and TCC/MCDM cannot be the trigger.
    for needle in ("nvcuda", "nvml", "cudart", "nvrtc", "nvJitLink", "nvvm"):
        hits = [p for p in after if needle.lower() in os.path.basename(p).lower()]
        emit(f"{needle:<10} -> {hits if hits else '<not loaded>'}")

    safe("tracer (after import)", collect_tracer)

    section("full module map after import")
    for row in format_modules(after):
        emit(row)

    section("result")
    emit(f"import_failed = {failure is not None}")

    if out_path:
        with open(out_path, "w", encoding="utf-8") as fh:
            fh.write("\n".join(_out_lines) + "\n")
        print(f"fingerprint written to {out_path}", flush=True)


def main() -> int:
    argv = sys.argv[1:]
    stack_mb = 0.0
    if "--stack-mb" in argv:
        i = argv.index("--stack-mb")
        stack_mb = float(argv[i + 1])
        del argv[i : i + 2]
    out_path = argv[0] if argv else None

    if not stack_mb:
        run(out_path)
        return 0

    # The nightly's cuda.core step runs everything inside run_pytest_with_stack.py's
    # worker, so an import on the main thread is not the same measurement: on a
    # build whose PE stack reserve is small, it dies of stack exhaustion long
    # before reaching whatever the nightly hits.  Matching the wrapper's thread
    # size is what makes the two comparable.
    threading.stack_size(int(stack_mb * 1024 * 1024))
    worker = threading.Thread(target=run, args=(out_path,), name=f"stack-{stack_mb:g}mb")
    worker.start()
    worker.join()
    return 0


if __name__ == "__main__":
    sys.exit(main())
