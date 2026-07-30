# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

"""Import cuda.core under full instrumentation and record where it dies.

The crash reduces to a single import, on a thread the size of the one
run_pytest_with_stack.py creates.  Two records are taken around it, and they
answer different halves of the same question:

    the breadcrumb   the last .pyx line that executed, i.e. which source
                     statement was running -- written by our own trace
                     function, which also arms Cython's linetrace hooks the
                     way coverage does, so the fault still triggers
    the native log   which module faulted and at what offset, from
                     native_crash_handler

A run under coverage answers "does it still crash", and a run with the
breadcrumb instead of coverage answers "is coverage necessary, or does any
tracer do".  Only one trace function can be installed at a time, so those are
separate invocations rather than one.

    crash_probe.py --stack-mb 8 --native-log native.txt --dump crash.dmp
    crash_probe.py --stack-mb 8 --breadcrumb lines.txt --native-log native.txt
    crash_probe.py --selftest --native-log native.txt

--selftest writes to address zero on purpose.  It is how the whole chain gets
verified somewhere that does not crash on its own: if the handler, the module
map, the dump and the artifact upload all work there, then the one gated run
on the runner that does crash is not spent discovering a typo.
"""

from __future__ import annotations

import argparse
import ctypes
import importlib
import os
import sys
import threading
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import native_crash_handler


def install_breadcrumb(path: str, needle: str) -> None:
    """Log every executed line whose file matches `needle`, line buffered.

    Cython emits __Pyx_TraceLine for module-level statements too, which is the
    only reason this can see anything at all: the fault happens while a module
    body is still executing, before any function in it has been called.
    """
    log = open(  # noqa: SIM115 - must outlive this function
        os.path.abspath(path), "w", buffering=1, encoding="utf-8", errors="backslashreplace"
    )

    def tracer(frame, event, _arg):
        if event == "line":
            filename = frame.f_code.co_filename
            if needle in filename:
                log.write(f"{filename}:{frame.f_lineno}\n")
        return tracer

    # Only one trace function exists per thread, so arming this one under
    # `coverage run` would quietly switch coverage off and leave a run that
    # looks like it measured something.  Say so rather than let it pass.
    displaced = sys.gettrace()
    if displaced is not None:
        message = f"# WARNING: replaced an existing trace function: {displaced!r}"
        log.write(message + "\n")
        print(message, flush=True)

    log.write(f"# breadcrumb armed, filter={needle!r}\n")
    sys.settrace(tracer)


def selftest() -> None:
    """Deliberately fault, to prove the handler reports what it should."""
    print("selftest: writing to address 0", flush=True)
    ctypes.memset(0, 0, 1)
    print("selftest: still alive -- the write did NOT fault", flush=True)


def body(args) -> None:
    if args.breadcrumb:
        install_breadcrumb(args.breadcrumb, args.breadcrumb_filter)
    if args.selftest:
        selftest()
        return
    started = time.perf_counter()
    importlib.import_module(args.module)
    print(f"PROBE OK {args.module} in {time.perf_counter() - started:.2f}s", flush=True)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--module", default="cuda.core")
    parser.add_argument("--stack-mb", type=float, default=0.0)
    parser.add_argument("--native-log", default="crash-native.txt")
    parser.add_argument("--dump")
    parser.add_argument("--breadcrumb")
    parser.add_argument("--breadcrumb-filter", default="cuda")
    parser.add_argument("--selftest", action="store_true")
    args = parser.parse_args()

    # Process-wide, so arming it here covers the worker thread as well.
    native_crash_handler.install(args.native_log, dump_path=args.dump)

    if not args.stack_mb:
        body(args)
        return 0

    # sys.settrace only affects the thread that calls it, so the breadcrumb is
    # installed inside body(), on the thread that does the import.
    threading.stack_size(int(args.stack_mb * 1024 * 1024))
    worker = threading.Thread(target=body, args=(args,), name=f"stack-{args.stack_mb:g}mb")
    worker.start()
    worker.join()
    return 0


if __name__ == "__main__":
    sys.exit(main())
