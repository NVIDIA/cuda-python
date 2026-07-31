#!/usr/bin/env python3

# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

"""Run pytest on a thread with a larger stack size.

Cython linetrace instrumentation under coverage on Windows can exceed the
default 1 MB thread stack.  This helper spawns a single worker thread with
a configurable stack (default 8 MB) so the rest of the CI workflow stays
readable.

Usage:
    python run_pytest_with_stack.py [--stack-mb N] [--cwd DIR] [pytest args ...]
"""

import argparse
import concurrent.futures
import os
import sys
import threading

import pytest


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--stack-mb",
        type=int,
        default=8,
        help="Thread stack size in megabytes (default: 8)",
    )
    parser.add_argument(
        "--cwd",
        default=None,
        help="Working directory for the test run",
    )
    args, pytest_args = parser.parse_known_args()

    if args.cwd:
        os.chdir(args.cwd)

    threading.stack_size(args.stack_mb * 1024 * 1024)

    with concurrent.futures.ThreadPoolExecutor(max_workers=1) as pool:
        code = pool.submit(pytest.main, pytest_args).result()

    sys.exit(code)


if __name__ == "__main__":
    main()
