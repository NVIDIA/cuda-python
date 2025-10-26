# SPDX-FileCopyrightText: Copyright (c) 2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# SPDX-License-Identifier: LicenseRef-NVIDIA-SOFTWARE-LICENSE

import time

import pytest
from cuda.core.experimental import Device
from helpers import IS_WINDOWS, IS_WSL
from helpers.buffers import PatternGen, compare_equal_buffers, make_scratch_buffer
from helpers.latch import LatchKernel
from helpers.logging import TimestampedLogger

ENABLE_LOGGING = False  # Set True for test debugging and development
NBYTES = 64


def test_latchkernel():
    """Test LatchKernel."""
    log = TimestampedLogger(enabled=ENABLE_LOGGING)
    log("begin")
    device = Device()
    device.set_current()
    stream = device.create_stream()
    target = make_scratch_buffer(device, 0, NBYTES)
    zeros = make_scratch_buffer(device, 0, NBYTES)
    ones = make_scratch_buffer(device, 1, NBYTES)
    latch = LatchKernel(device)
    log("launching latch kernel")
    latch.launch(stream)
    log("launching copy (0->1) kernel")
    target.copy_from(ones, stream=stream)
    log("going to sleep")
    time.sleep(1)
    if not IS_WINDOWS and not IS_WSL:
        # On any sort of Windows system, checking the memory before stream
        # sync results in a page error.
        log("checking target == 0")
        assert compare_equal_buffers(target, zeros)
    log("releasing latch and syncing")
    latch.release()
    stream.sync()
    log("checking target == 1")
    assert compare_equal_buffers(target, ones)
    log("done")


def test_patterngen_seeds():
    """Test PatternGen with seed argument."""
    device = Device()
    device.set_current()
    buffer = make_scratch_buffer(device, 0, NBYTES)

    # All seeds are pairwise different.
    pgen = PatternGen(device, NBYTES)
    for i in range(256):
        pgen.fill_buffer(buffer, seed=i)
        pgen.verify_buffer(buffer, seed=i)
        for j in range(i + 1, 256):
            with pytest.raises(AssertionError):
                pgen.verify_buffer(buffer, seed=j)


def test_patterngen_values():
    """Test PatternGen with value argument, also compare_equal_buffers."""
    device = Device()
    device.set_current()
    ones = make_scratch_buffer(device, 1, NBYTES)
    twos = make_scratch_buffer(device, 2, NBYTES)
    assert compare_equal_buffers(ones, ones)
    assert not compare_equal_buffers(ones, twos)
    pgen = PatternGen(device, NBYTES)
    pgen.verify_buffer(ones, value=1)
    pgen.verify_buffer(twos, value=2)
