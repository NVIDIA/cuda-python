# Copyright 2025 NVIDIA Corporation.  All rights reserved.
# SPDX-License-Identifier: LicenseRef-NVIDIA-SOFTWARE-LICENSE

import multiprocessing
import queue  # for Empty
import sys
import traceback
from dataclasses import dataclass
from io import StringIO


class Worker:
    def __init__(self, result_queue, func, args, kwargs):
        self.func = func
        self.args = args or ()
        self.kwargs = kwargs or {}
        self.result_queue = result_queue

    def __call__(self):
        # Capture stdout/stderr
        old_stdout = sys.stdout
        old_stderr = sys.stderr
        sys.stdout = StringIO()
        sys.stderr = StringIO()

        try:
            self.func(*self.args, **self.kwargs)
            returncode = 0
        except SystemExit as e:  # Handle sys.exit()
            returncode = e.code if isinstance(e.code, int) else 0
        except BaseException:
            traceback.print_exc()
            returncode = 1
        finally:
            # Collect outputs and restore streams
            stdout = sys.stdout.getvalue()
            stderr = sys.stderr.getvalue()
            sys.stdout = old_stdout
            sys.stderr = old_stderr
            try:  # noqa: SIM105
                self.result_queue.put((returncode, stdout, stderr))
            except Exception:  # nosec B110
                # If the queue is broken (e.g., parent gone), best effort logging
                pass


# Similar to https://docs.python.org/3/library/subprocess.html#subprocess.CompletedProcess
# (args, check_returncode() are intentionally not supported here.)
@dataclass
class CompletedProcess:
    returncode: int
    stdout: str
    stderr: str


def run_in_spawned_child_process(func, *, args=None, kwargs=None, timeout=None):
    """Run Python code in a spawned child process, capturing stdout/stderr/output."""
    ctx = multiprocessing.get_context("spawn")
    result_queue = ctx.Queue()
    process = ctx.Process(target=Worker(result_queue, func, args, kwargs))
    process.start()

    try:
        process.join(timeout)
        if process.is_alive():
            process.terminate()
            process.join()
            return CompletedProcess(
                returncode=-9,
                stdout="",
                stderr=f"Process timed out after {timeout} seconds and was terminated.",
            )

        try:
            returncode, stdout, stderr = result_queue.get(timeout=1.0)
        except (queue.Empty, EOFError):
            return CompletedProcess(
                returncode=-999,
                stdout="",
                stderr="Process exited or crashed before returning results.",
            )

        return CompletedProcess(
            returncode=returncode,
            stdout=stdout,
            stderr=stderr,
        )

    finally:
        try:
            result_queue.close()
            result_queue.join_thread()
        except Exception:  # nosec B110
            pass
        if process.is_alive():
            process.kill()
            process.join()
