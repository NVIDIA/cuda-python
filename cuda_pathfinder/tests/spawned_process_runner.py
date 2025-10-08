# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import multiprocessing
import queue  # for Empty
import sys
import traceback
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from io import StringIO
from typing import Any

PROCESS_KILLED = -9
PROCESS_NO_RESULT = -999


# Similar to https://docs.python.org/3/library/subprocess.html#subprocess.CompletedProcess
# (args, check_returncode() are intentionally not supported here.)
@dataclass
class CompletedProcess:
    returncode: int
    stdout: str
    stderr: str


class ChildProcessWrapper:
    def __init__(self, result_queue, target, args, kwargs):
        self.target = target
        self.args = () if args is None else args
        self.kwargs = {} if kwargs is None else kwargs
        self.result_queue = result_queue

    def __call__(self):
        # Capture stdout/stderr
        old_stdout = sys.stdout
        old_stderr = sys.stderr
        sys.stdout = StringIO()
        sys.stderr = StringIO()

        try:
            self.target(*self.args, **self.kwargs)
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
            except Exception:  # noqa: S110
                # If the queue is broken (e.g., parent gone), best effort logging
                pass


def run_in_spawned_child_process(
    target: Callable[..., None],
    *,
    args: Sequence[Any] | None = None,
    kwargs: dict[str, Any] | None = None,
    timeout: float | None = None,
    rethrow: bool = False,
) -> CompletedProcess:
    """Run `target` in a spawned child process, capturing stdout/stderr.

    The provided `target` must be defined at the top level of a module, and must
    be importable in the spawned child process. Lambdas, closures, or interactively
    defined functions (e.g., in Jupyter notebooks) will not work.

    If `rethrow=True` and the child process exits with a nonzero code,
    raises ChildProcessError with the captured stderr.
    """
    ctx = multiprocessing.get_context("spawn")
    result_queue = ctx.Queue()
    process = ctx.Process(target=ChildProcessWrapper(result_queue, target, args, kwargs))
    process.start()

    try:
        process.join(timeout)
        if process.is_alive():
            process.terminate()
            process.join()
            result = CompletedProcess(
                returncode=PROCESS_KILLED,
                stdout="",
                stderr=f"Process timed out after {timeout} seconds and was terminated.",
            )
        else:
            try:
                returncode, stdout, stderr = result_queue.get(timeout=1.0)
            except (queue.Empty, EOFError):
                result = CompletedProcess(
                    returncode=PROCESS_NO_RESULT,
                    stdout="",
                    stderr="Process exited or crashed before returning results.",
                )
            else:
                result = CompletedProcess(
                    returncode=returncode,
                    stdout=stdout,
                    stderr=stderr,
                )

        if rethrow and result.returncode != 0:
            raise ChildProcessError(
                f"Child process exited with code {result.returncode}.\n"
                "--- stderr-from-child-process ---\n"
                f"{result.stderr}"
                "<end-of-stderr-from-child-process>\n"
            )

        return result

    finally:
        try:
            result_queue.close()
            result_queue.join_thread()
        except Exception:  # noqa: S110
            pass
        if process.is_alive():
            process.kill()
            process.join()
