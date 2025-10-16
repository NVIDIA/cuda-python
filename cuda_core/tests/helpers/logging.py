# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import time

class TimestampedLogger:
    """
    A logger that prefixes each output with a timestamp, containing the elapsed
    time since the logger was created.

    Example:

        import multiprocess as mp
        import time

        def main():
            log = TimestampedLogger(prefix="parent: ")
            log("begin")
            process = mp.Process(target=child_main, args=(log,))
            process.start()
            process.join()
            log("done")

        def child_main(log):
            log.prefix = " child: "
            log("begin")
            time.sleep(1)
            log("done")

        if __name__ == "__main__":
            main()

    Possible output:

        [     0.003 ms] parent: begin
        [   819.464 ms]  child: begin
        [  1819.666 ms]  child: done
        [  1882.954 ms] parent: done
    """

    def __init__(self, prefix=None, start_time=None, enabled=True):
        self.prefix = "" if prefix is None else prefix
        self.start_time = start_time if start_time is not None else time.time_ns()
        self.enabled = enabled

    def __call__(self, msg):
        if self.enabled:
            now = (time.time_ns() - self.start_time) * 1e-6
            print(f"[{now:>10.3f} ms] {self.prefix}{msg}")
