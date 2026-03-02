# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import contextlib
import gc
import multiprocessing as mp
import platform

try:
    import psutil
except ImportError:
    HAVE_PSUTIL = False
else:
    HAVE_PSUTIL = True
import pytest

CHILD_TIMEOUT_SEC = 30
NBYTES = 64

USING_FDS = platform.system() == "Linux"
skip_if_unrunnable = pytest.mark.skipif(
    not USING_FDS or not HAVE_PSUTIL, reason="mempool allocation handle is not using fds or psutil is unavailable"
)


@pytest.mark.flaky(reruns=2)
@skip_if_unrunnable
def test_alloc_handle(ipc_memory_resource):
    """Check for fd leaks in get_allocation_handle."""
    mr = ipc_memory_resource
    with CheckFDLeaks():
        [mr.get_allocation_handle() for _ in range(10)]


def exec_success(obj, number=1):
    """Succesfully run a child process."""
    for _ in range(number):
        process = mp.Process(target=child_main, args=(obj,))
        process.start()
        process.join(timeout=CHILD_TIMEOUT_SEC)
        assert process.exitcode == 0


def child_main(obj, *args):
    pass


def exec_launch_failure(obj, number=1):
    """
    Unsuccesfully try to launch a child process. This fails when
    after the child starts.
    """
    for _ in range(number):
        process = mp.Process(target=child_main_bad, args=(obj,))
        process.start()
        process.join(timeout=CHILD_TIMEOUT_SEC)
        assert process.exitcode != 0


def child_main_bad():
    """Fails when passed arguments."""
    pass


def exec_reduce_failure(obj, number=1):
    """
    Unsuccesfully try to launch a child process. This fails before
    the child starts but after the resource-owning object is serialized.
    """
    for _ in range(number):
        fails_to_reduce = Irreducible()
        with contextlib.suppress(RuntimeError):
            mp.Process(target=child_main, args=(obj, fails_to_reduce)).start()


class Irreducible:
    """A class that cannot be serialized."""

    def __reduce__(self):
        raise RuntimeError("Irreducible")


@pytest.mark.flaky(reruns=2)
@skip_if_unrunnable
@pytest.mark.parametrize(
    "getobject",
    [
        lambda mr: mr.get_allocation_handle(),
        lambda mr: mr,
        lambda mr: mr.allocate(NBYTES),
        lambda mr: mr.allocate(NBYTES).get_ipc_descriptor(),
    ],
    ids=["alloc_handle", "mr", "buffer", "buffer_desc"],
)
@pytest.mark.parametrize("launcher", [exec_success, exec_launch_failure, exec_reduce_failure])
def test_pass_object(ipc_memory_resource, launcher, getobject):
    """Check for fd leaks when an object is sent as a subprocess argument."""
    mr = ipc_memory_resource
    with CheckFDLeaks():
        obj = getobject(mr)
        try:
            launcher(obj, number=2)
        finally:
            del obj


class CheckFDLeaks:
    """
    Context manager to check for file descriptor leaks.
    Ensures the number of open file descriptors is the same before and after the block.
    """

    def __init__(self):
        self.process = psutil.Process()

    def __enter__(self):
        prime()
        gc.collect()
        self.initial_fds = self.process.num_fds()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        if exc_type is not None:
            gc.collect()
            final_fds = self.process.num_fds()
            assert final_fds == self.initial_fds
        return False


prime_was_run = False


def prime():
    """Multiprocessing consumes a file descriptor on first launch."""
    assert mp.get_start_method() == "spawn"
    global prime_was_run
    if not prime_was_run:
        process = mp.Process()
        process.start()
        process.join(timeout=CHILD_TIMEOUT_SEC)
        assert process.exitcode == 0
        prime_was_run = True
