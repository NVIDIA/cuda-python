# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Per-directory conftest for memory IPC tests.

Applies an outer-guard ``pytest.mark.timeout`` to every test in this directory.
Individual tests still drive their own per-process waits using
``child_timeout_sec()`` from ``helpers.child_processes``; this marker is the
final fallback so that no IPC test can wedge the CI runner for hours if
deadlock occurs.
"""

import pathlib

import pytest
from helpers.child_processes import child_timeout_sec

_HERE = pathlib.Path(__file__).parent.resolve()


def _outer_timeout_sec() -> int:
    # IPC tests spawn children that run concurrently, so expected wall-clock
    # is ~CHILD_TIMEOUT_SEC regardless of how many subsequent join/wait
    # timeouts the test chains together (each subsequent join returns
    # immediately once its child is already done). Exceeding that already
    # means something is genuinely stuck, at which point the outer guard
    # firing is the right outcome -- the per-test asserts wouldn't add
    # useful diagnostic value over "test exceeded its budget", and the
    # autouse track_child_processes() context manager still cleans up.
    return child_timeout_sec() + 30


def pytest_collection_modifyitems(config, items):
    marker = pytest.mark.timeout(_outer_timeout_sec())
    for item in items:
        try:
            item_path = pathlib.Path(str(item.fspath)).resolve()
        except OSError:
            continue
        if _HERE in item_path.parents:
            item.add_marker(marker)
