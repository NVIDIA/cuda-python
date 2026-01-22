# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

# ruff: noqa: E402

from .conftest import skip_if_nvml_unsupported

pytestmark = skip_if_nvml_unsupported

import helpers
import pytest
from cuda.core import system


@pytest.mark.skipif(helpers.IS_WSL or helpers.IS_WINDOWS, reason="System events not supported on WSL or Windows")
def test_register_events():
    # This is not the world's greatest test.  All of the events are pretty
    # infrequent and hard to simulate.  So all we do here is register an event,
    # wait with a timeout, and ensure that we get no event (since we didn't do
    # anything to trigger one).

    # Also, some hardware doesn't support any event types.

    events = system.register_events([system.SystemEventType.GPU_DRIVER_UNBIND])
    with pytest.raises(system.TimeoutError):
        events.wait(timeout_ms=500, buffer_size=1)
