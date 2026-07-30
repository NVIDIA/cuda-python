# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

"""Record the native side of a Windows fault, from inside the faulting process.

faulthandler names the exception and prints Python frames, but the fault we are
chasing happens inside an extension module's initialiser, so the interesting
frames are the ones it cannot see: `<cannot get C stack on this system>`.

Nothing that would normally fill that gap is available on the runner this has
to work on.  A postmortem debugger registered under AeDebug never fires,
because that Server Core image does not record user-process crashes with WER --
that is why the crash-dumps directory came back empty the last time.  cdb is
not installed: the SDK is there but without the Debugging Tools feature, so
`Windows Kits\\10\\Debuggers\\x64` holds the DLLs and no debugger executables.
Downloading procdump means an outbound request on a runner whose whole job is
to police outbound requests.  And there are no PDBs either way, since
build_hooks.py refuses debuggable builds on Windows.

What *is* always present is dbgcore.dll in System32, exporting
MiniDumpWriteDump.  So the process reports on itself: a vectored exception
handler catches the fault first-chance and writes, in order,

    the exception record      code, flags, faulting address, and for an access
                              violation the operation and target address
    the module map            base and size for every loaded module, so an
                              address becomes module + RVA
    the native return chain   RtlCaptureStackBackTrace, each frame resolved the
                              same way
    a minidump                for reading back with a real debugger later

then returns EXCEPTION_CONTINUE_SEARCH so the process still dies exactly as it
would have.  Symbols are absent, so the output is addresses; module + RVA is
enough to say which binary faulted, which is the question being asked.

Usage:

    import native_crash_handler
    native_crash_handler.install("crash-native.txt", dump_path="crash.dmp")
"""

from __future__ import annotations

import contextlib
import ctypes
import ctypes.wintypes as wintypes
import os

EXCEPTION_CONTINUE_SEARCH = 0

# Only genuinely fatal codes.  A process raises first-chance exceptions all the
# time -- 0xE06D7363 for every C++ throw, 0x406D1388 when a debugger names a
# thread -- and reporting those would bury the one that matters.
FATAL_CODES = {
    0xC0000005: "EXCEPTION_ACCESS_VIOLATION",
    0xC00000FD: "EXCEPTION_STACK_OVERFLOW",
    0xC0000374: "STATUS_HEAP_CORRUPTION",
    0xC0000409: "STATUS_STACK_BUFFER_OVERRUN",
    0xC000041D: "STATUS_FATAL_USER_CALLBACK_EXCEPTION",
    0xC000001D: "EXCEPTION_ILLEGAL_INSTRUCTION",
    0x80000003: "STATUS_BREAKPOINT",
}

# MiniDumpNormal plus the extras that make a stack walk possible.  Deliberately
# not MiniDumpWithFullMemory: python.exe with ~120 modules loaded dumps
# hundreds of megabytes that way, and the stacks are what we came for.
MINIDUMP_TYPE = 0x0000 | 0x0040 | 0x0100 | 0x1000

_kernel32 = ctypes.WinDLL("kernel32", use_last_error=True)
_psapi = ctypes.WinDLL("psapi", use_last_error=True)
_ntdll = ctypes.WinDLL("ntdll", use_last_error=True)


class EXCEPTION_RECORD(ctypes.Structure):  # noqa: N801 - mirrors the Win32 name
    pass


EXCEPTION_RECORD._fields_ = [
    ("ExceptionCode", wintypes.DWORD),
    ("ExceptionFlags", wintypes.DWORD),
    ("ExceptionRecord", ctypes.POINTER(EXCEPTION_RECORD)),
    ("ExceptionAddress", ctypes.c_void_p),
    ("NumberParameters", wintypes.DWORD),
    ("ExceptionInformation", ctypes.c_size_t * 15),
]


class EXCEPTION_POINTERS(ctypes.Structure):  # noqa: N801 - mirrors the Win32 name
    _fields_ = [
        ("ExceptionRecord", ctypes.POINTER(EXCEPTION_RECORD)),
        ("ContextRecord", ctypes.c_void_p),
    ]


class MODULEINFO(ctypes.Structure):
    _fields_ = [
        ("lpBaseOfDll", ctypes.c_void_p),
        ("SizeOfImage", wintypes.DWORD),
        ("EntryPoint", ctypes.c_void_p),
    ]


class MINIDUMP_EXCEPTION_INFORMATION(ctypes.Structure):  # noqa: N801 - mirrors the Win32 name
    _fields_ = [
        ("ThreadId", wintypes.DWORD),
        ("ExceptionPointers", ctypes.POINTER(EXCEPTION_POINTERS)),
        ("ClientPointers", wintypes.BOOL),
    ]


PVECTORED_EXCEPTION_HANDLER = ctypes.WINFUNCTYPE(ctypes.c_long, ctypes.POINTER(EXCEPTION_POINTERS))

# argtypes are not optional here.  Without them ctypes marshals a Python int
# as a C int, and every HANDLE and module base on win64 is wider than that:
# the truncated value comes back as a failed call, inside an exception handler
# where the failure has nowhere to surface.
_kernel32.GetCurrentProcess.restype = wintypes.HANDLE
_kernel32.AddVectoredExceptionHandler.argtypes = [wintypes.ULONG, PVECTORED_EXCEPTION_HANDLER]
_kernel32.AddVectoredExceptionHandler.restype = ctypes.c_void_p
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
_ntdll.RtlCaptureStackBackTrace.argtypes = [
    wintypes.ULONG,
    wintypes.ULONG,
    ctypes.POINTER(ctypes.c_void_p),
    ctypes.POINTER(wintypes.ULONG),
]
_ntdll.RtlCaptureStackBackTrace.restype = wintypes.USHORT

# Held at module scope on purpose.  ctypes does not keep the callback alive,
# and a collected handler is a crash of its own; the log handle must outlive
# install() for the same reason.
_handler_ref = None
_log = None
_dump_path = None

# A vectored handler sees first-chance exceptions, including ones somebody is
# about to handle -- ctypes turns an access violation into OSError exactly
# that way, which is how --selftest reports rather than dies.  Reporting only
# the first one would let a benign, handled fault earlier in the import
# consume the single slot and leave the real crash unrecorded, so a few are
# allowed through and each is numbered.
_MAX_REPORTS = 3
_reports = 0


def _modules() -> list[tuple[int, int, str]]:
    """(base, size, path), sorted by base, for every module mapped right now."""
    process = _kernel32.GetCurrentProcess()
    needed = wintypes.DWORD()
    arr = (ctypes.c_void_p * 1024)()
    if not _psapi.EnumProcessModules(process, arr, ctypes.sizeof(arr), ctypes.byref(needed)):
        return []
    count = min(needed.value // ctypes.sizeof(ctypes.c_void_p), len(arr))
    name = ctypes.create_unicode_buffer(32768)
    out = []
    for i in range(count):
        hmod = arr[i]
        if not _psapi.GetModuleFileNameExW(process, hmod, name, len(name)):
            continue
        info = MODULEINFO()
        if not _psapi.GetModuleInformation(process, hmod, ctypes.byref(info), ctypes.sizeof(info)):
            continue
        out.append((info.lpBaseOfDll or 0, info.SizeOfImage, name.value))
    out.sort()
    return out


def _resolve(address: int, modules: list[tuple[int, int, str]]) -> str:
    """address -> "module+0xRVA", the closest thing to a symbol available here."""
    for base, size, path in modules:
        if base <= address < base + size:
            return f"{os.path.basename(path)}+0x{address - base:x}"
    return "<unmapped>"


def _write_dump(exception_pointers, index: int) -> None:
    if not _dump_path:
        return
    # One file per report, so a second fault cannot overwrite the first.
    root, ext = os.path.splitext(_dump_path)
    dump_path = _dump_path if index == 1 else f"{root}.{index}{ext}"
    try:
        import msvcrt

        # MiniDumpWriteDump lives in dbgcore.dll, and lived in dbghelp.dll
        # before that; both are system DLLs, so at least one is present on
        # anything this could run on.  If neither loads, the text report above
        # already carries the answer -- the dump is the convenience, not the
        # evidence -- so this says so and returns instead of failing.
        dbgcore = None
        for name in ("dbgcore", "dbghelp"):
            try:
                dbgcore = ctypes.WinDLL(name, use_last_error=True)
                _log.write(f"dump provider: {name}.dll\n")
                break
            except OSError:
                continue
        if dbgcore is None:
            _log.write("no dbgcore.dll or dbghelp.dll; dump skipped\n")
            return
        dbgcore.MiniDumpWriteDump.argtypes = [
            wintypes.HANDLE,
            wintypes.DWORD,
            wintypes.HANDLE,
            wintypes.DWORD,
            ctypes.c_void_p,  # PMINIDUMP_EXCEPTION_INFORMATION, or NULL
            ctypes.c_void_p,
            ctypes.c_void_p,
        ]
        dbgcore.MiniDumpWriteDump.restype = wintypes.BOOL
        info = MINIDUMP_EXCEPTION_INFORMATION(
            ThreadId=_kernel32.GetCurrentThreadId(),
            ExceptionPointers=exception_pointers,
            ClientPointers=False,
        )
        # Attaching the exception record is worth a try -- it is what makes a
        # debugger open the dump on the faulting instruction -- but writing it
        # from inside a vectored handler has been seen to come back as
        # ERROR_NOACCESS (998).  The thread stacks are the reason for the dump
        # and they are present either way, and the exception itself is already
        # recorded above in text, so a refusal falls back rather than gives up.
        for exception_param in (ctypes.byref(info), None):
            with open(dump_path, "wb") as fh:
                ok = dbgcore.MiniDumpWriteDump(
                    _kernel32.GetCurrentProcess(),
                    _kernel32.GetCurrentProcessId(),
                    wintypes.HANDLE(msvcrt.get_osfhandle(fh.fileno())),
                    MINIDUMP_TYPE,
                    exception_param,
                    None,
                    None,
                )
            if ok:
                size = os.path.getsize(dump_path)
                kind = "with exception record" if exception_param is not None else "no exception record"
                _log.write(f"dump written: {_dump_path} ({size} bytes, {kind})\n")
                return
            _log.write(
                f"MiniDumpWriteDump failed (exception_param={exception_param is not None}), "
                f"GetLastError={ctypes.get_last_error() & 0xFFFFFFFF}\n"
            )
    except Exception as exc:  # a failed dump must not hide the report
        _log.write(f"dump failed: {type(exc).__name__}: {exc}\n")


def _on_exception(pointers):
    global _reports
    try:
        record = pointers[0].ExceptionRecord[0]
        code = record.ExceptionCode & 0xFFFFFFFF
        if code not in FATAL_CODES or _reports >= _MAX_REPORTS:
            return EXCEPTION_CONTINUE_SEARCH
        # Counted before doing any work: if the reporting itself faults, the
        # re-entry must fall through instead of recursing forever.
        _reports += 1

        # Header first, before anything that can fail: if module enumeration
        # or the dump goes wrong, the record of the fault itself survives.
        address = record.ExceptionAddress or 0
        _log.write(f"=== native fault #{_reports} ===\n")
        _log.write(f"code            = 0x{code:08X} ({FATAL_CODES[code]})\n")
        modules = _modules()
        _log.write(f"flags           = 0x{record.ExceptionFlags:08X}\n")
        _log.write(f"address         = 0x{address:016x}  {_resolve(address, modules)}\n")
        if code == 0xC0000005 and record.NumberParameters >= 2:
            op = {0: "read", 1: "write", 8: "DEP"}.get(record.ExceptionInformation[0], "?")
            target = record.ExceptionInformation[1]
            _log.write(f"operation       = {op} at 0x{target:016x}  {_resolve(target, modules)}\n")
        _log.write(f"thread          = {_kernel32.GetCurrentThreadId()}\n")

        frames = (ctypes.c_void_p * 62)()
        n = _ntdll.RtlCaptureStackBackTrace(0, 62, frames, None)
        _log.write(f"--- native backtrace ({n} frames, innermost first) ---\n")
        for i in range(n):
            frame = frames[i] or 0
            _log.write(f"  [{i:02d}] 0x{frame:016x}  {_resolve(frame, modules)}\n")

        _log.write(f"--- modules ({len(modules)}) ---\n")
        for base, size, path in modules:
            _log.write(f"  0x{base:016x}  {size:>9}  {path}\n")

        _write_dump(pointers, _reports)
        _log.write(f"=== end native fault #{_reports} ===\n")
    except BaseException as exc:  # never let reporting change the outcome
        # Reported, not swallowed: a silent handler is indistinguishable from
        # one that never ran, and that ambiguity costs a whole CI round.  The
        # log is what would have failed, so even saying so is allowed to fail.
        with contextlib.suppress(Exception):
            _log.write(f"handler error: {type(exc).__name__}: {exc}\n")
    # Always continue searching: the process must die the way it would have,
    # so the exit code the CI sees is unchanged.
    return EXCEPTION_CONTINUE_SEARCH


def install(log_path: str, dump_path: str | None = None) -> None:
    """Arm the handler.  Call once, as early as possible, from any thread."""
    global _handler_ref, _log, _dump_path

    # Line buffered: a hard fault takes the process down without flushing, so
    # every line has to reach the OS as it is written.
    _log = open(  # noqa: SIM115 - must outlive this function
        os.path.abspath(log_path), "w", buffering=1, encoding="utf-8", errors="backslashreplace"
    )
    _dump_path = os.path.abspath(dump_path) if dump_path else None
    _handler_ref = PVECTORED_EXCEPTION_HANDLER(_on_exception)
    if not _kernel32.AddVectoredExceptionHandler(1, _handler_ref):
        _log.write("AddVectoredExceptionHandler failed\n")
        return
    _log.write(f"handler armed, pid={os.getpid()}, dump={_dump_path or '<disabled>'}\n")
