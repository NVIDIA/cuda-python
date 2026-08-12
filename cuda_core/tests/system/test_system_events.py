# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0


from cuda_python_test_helpers.arch_check import skip_if_nvml_unsupported

pytestmark = skip_if_nvml_unsupported

import helpers
import pytest

from cuda.core import system
from cuda.core.system import typing

if system.CUDA_BINDINGS_NVML_IS_COMPATIBLE:
    from cuda.bindings import nvml
    from cuda.core.system._system_events import SystemEvent, SystemEvents, _pci_bus_id_from_gpu_id


@pytest.mark.agent_authored(model="claude-opus-4.7")
def test_system_events_wraps_event_data():
    # Use synthetic data because real bind/unbind events are difficult to
    # trigger reliably.
    event_data = nvml.SystemEventData_v1(2)
    event_data.event_type = nvml.SystemEventType.GPU_DRIVER_BIND
    event_data.gpu_id = [0x0000_0200, 0x0000_C100]

    events = SystemEvents(event_data)
    assert len(events) == 2

    event = events[0]
    assert isinstance(event, SystemEvent)
    assert event.event_type is typing.SystemEventType.BIND
    assert event.gpu_id == 0x0000_0200


@pytest.mark.agent_authored(model="claude-opus-4.7")
@pytest.mark.parametrize(
    ("gpu_id", "expected"),
    [
        (0x0000_0200, "00000000:02:00.0"),
        (0x0000_C100, "00000000:C1:00.0"),
        # The device occupies bits [7:0]; the function is always 0.
        (0x0001_0A0F, "00000001:0A:0F.0"),
        (0xFFFF_FFFF, "0000FFFF:FF:FF.0"),
    ],
)
def test_pci_bus_id_from_gpu_id(gpu_id, expected):
    assert _pci_bus_id_from_gpu_id(gpu_id) == expected


@pytest.mark.agent_authored(model="claude-opus-4.7")
def test_system_event_device_resolves_pci_bus_id():
    # Pack live PCI data as RM does, then resolve it through SystemEvent.device.
    if system.get_num_devices() == 0:
        pytest.skip("No GPUs available")

    for device in system.Device.get_all_devices():
        pci = device.pci_info
        if pci.domain > 0xFFFF:
            pytest.skip(f"PCI domain {pci.domain:#x} does not fit in a packed gpu_id")
        gpu_id = (pci.domain << 16) | (pci.bus << 8) | (pci.device & 0xFF)

        event_data = nvml.SystemEventData_v1(1)
        event_data.event_type = nvml.SystemEventType.GPU_DRIVER_BIND
        event_data.gpu_id = gpu_id
        event = SystemEvent(event_data)
        resolved_device = event.device

        assert resolved_device.pci_bus_id == device.pci_bus_id
        assert resolved_device.index == device.index


@pytest.mark.skipif(helpers.IS_WSL or helpers.IS_WINDOWS, reason="System events not supported on WSL or Windows")
def test_register_events():
    # This is not the world's greatest test.  All of the events are pretty
    # infrequent and hard to simulate.  So all we do here is register an event,
    # wait with a timeout, and ensure that we get no event (since we didn't do
    # anything to trigger one).

    # Also, some hardware doesn't support any event types.

    try:
        events = system.register_events([typing.SystemEventType.UNBIND])
    except system.UnknownError:
        pytest.skip("system events may only be registered once per process")

    with pytest.raises(system.TimeoutError):
        events.wait(timeout_ms=500, buffer_size=1)
