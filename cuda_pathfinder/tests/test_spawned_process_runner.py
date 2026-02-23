# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

# Note: This only covers what is not covered already in test_nvidia_dynamic_libs_load_lib.py

import pytest

from cuda.pathfinder._utils.spawned_process_runner import run_in_spawned_child_process


def child_crashes():
    raise RuntimeError("this is an intentional failure")


def test_rethrow_child_exception():
    with pytest.raises(ChildProcessError) as excinfo:
        run_in_spawned_child_process(child_crashes, rethrow=True)

    msg = str(excinfo.value)
    assert "Child process exited with code 1" in msg
    assert "this is an intentional failure" in msg
    assert "--- stderr-from-child-process ---" in msg
