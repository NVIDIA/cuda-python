import multiprocessing
import queue  # for Empty
import subprocess  # nosec B404
import sys
import traceback
from io import StringIO


class Worker:
    def __init__(self, python_code, result_queue):
        self.python_code = python_code
        self.result_queue = result_queue

    def __call__(self):
        # Capture stdout/stderr
        old_stdout = sys.stdout
        old_stderr = sys.stderr
        sys.stdout = StringIO()
        sys.stderr = StringIO()

        try:
            exec(self.python_code, {"__name__": "__main__"})  # nosec B102
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


def run_python_code_safely(python_code, *, timeout=None):
    """Run Python code in a spawned subprocess, capturing stdout/stderr/output."""
    ctx = multiprocessing.get_context("spawn")
    result_queue = ctx.Queue()
    process = ctx.Process(target=Worker(python_code, result_queue))
    process.start()

    try:
        process.join(timeout)
        if process.is_alive():
            process.terminate()
            process.join()
            return subprocess.CompletedProcess(
                args=[sys.executable, "-c", python_code],
                returncode=-9,
                stdout="",
                stderr=f"Process timed out after {timeout} seconds and was terminated.",
            )

        try:
            returncode, stdout, stderr = result_queue.get(timeout=1.0)
        except (queue.Empty, EOFError):
            return subprocess.CompletedProcess(
                args=[sys.executable, "-c", python_code],
                returncode=-999,
                stdout="",
                stderr="Process exited or crashed before returning results.",
            )

        return subprocess.CompletedProcess(
            args=[sys.executable, "-c", python_code],
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
