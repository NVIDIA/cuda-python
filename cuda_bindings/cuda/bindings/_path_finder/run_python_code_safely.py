import multiprocessing
import subprocess  # nosec B404
import sys
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

        returncode = 0
        try:
            exec(self.python_code, {"__name__": "__main__"})  # nosec B102
        except SystemExit as e:  # Handle sys.exit()
            returncode = e.code if isinstance(e.code, int) else 0
        except Exception:  # Capture other exceptions
            import traceback

            traceback.print_exc()
            returncode = 1
        finally:
            # Collect outputs and restore streams
            stdout = sys.stdout.getvalue()
            stderr = sys.stderr.getvalue()
            sys.stdout = old_stdout
            sys.stderr = old_stderr
            self.result_queue.put((returncode, stdout, stderr))


def run_python_code_safely(python_code, *, timeout=None):
    """Replacement for subprocess.run that forces 'spawn' context"""
    ctx = multiprocessing.get_context("spawn")
    result_queue = ctx.Queue()
    process = ctx.Process(target=Worker(python_code, result_queue))
    process.start()

    try:
        # Wait with timeout support
        process.join(timeout)
        if process.is_alive():
            process.terminate()
            process.join()
            raise subprocess.TimeoutExpired([sys.executable, "-c", python_code], timeout)

        # Get results from queue
        if result_queue.empty():
            return subprocess.CompletedProcess(
                [sys.executable, "-c", python_code],
                returncode=-999,
                stdout="",
                stderr="Process failed to return results",
            )

        returncode, stdout, stderr = result_queue.get()
        return subprocess.CompletedProcess(
            [sys.executable, "-c", python_code], returncode=returncode, stdout=stdout, stderr=stderr
        )
    finally:
        # Cleanup if needed
        if process.is_alive():
            process.kill()
