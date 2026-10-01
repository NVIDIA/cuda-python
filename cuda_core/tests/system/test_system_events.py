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


@pytest.mark.agent_authored(model="gpt-6")
def test_system_event_device_resolves_pci_bus_id(init_cuda):
    devices = list(system.Device.get_all_devices())
    if not devices:
        pytest.skip("No NVML devices available")

    for device in devices:
        try:
            original_pci = device.pci_info
        except system.NotSupportedError:
            # Orin supports PCI lookup but needs CUDA to supply the PCI bus ID.
            cuda_pci_bus_id = device.to_cuda_device().pci_bus_id
            domain_string, bus_string, device_function = cuda_pci_bus_id.split(":")
            device_string, function_string = device_function.split(".")
            domain = int(domain_string, 16)
            bus = int(bus_string, 16)
            pci_device = int(device_string, 16)
            assert int(function_string, 16) == 0
            expected_pci_bus_id = f"{domain:08X}:{bus:02X}:{pci_device:02X}.0"
        else:
            domain, bus, pci_device = original_pci.domain, original_pci.bus, original_pci.device
            expected_pci_bus_id = original_pci.bus_id

        if domain > 0xFFFF:
            pytest.skip(f"PCI domain {domain:#x} does not fit in a packed gpu_id")
        gpu_id = (domain << 16) | (bus << 8) | pci_device

        event_data = nvml.SystemEventData_v1(1)
        event_data.event_type = nvml.SystemEventType.GPU_DRIVER_BIND
        event_data.gpu_id = gpu_id
        event = SystemEvent(event_data)
        resolved_device = event.device

        assert resolved_device.uuid == device.uuid
        assert resolved_device.index == device.index

        # PCI lookup can work even when the PCI information query is unsupported.
        try:
            pci = resolved_device.pci_info
        except system.NotSupportedError:
            continue
        assert (pci.domain, pci.bus, pci.device) == (domain, bus, pci_device)
        assert pci.bus_id == expected_pci_bus_id
        assert resolved_device.pci_bus_id == expected_pci_bus_id


@pytest.mark.skipif(helpers.IS_WSL or helpers.IS_WINDOWS, reason="System events not supported on WSL or Windows")
@pytest.mark.agent_authored(model="gpt-6")
def test_register_events():
    # This is not the world's greatest test.  All of the events are pretty
    # infrequent and hard to simulate.  So all we do here is register an event,
    # wait with a timeout, and ensure that we get no event (since we didn't do
    # anything to trigger one).

    # Also, some hardware doesn't support any event types.

    try:
        events = system.register_events([typing.SystemEventType.UNBIND])
    except system.NotSupportedError:
        # The documented outcome when none of the requested events are supported.
        return
    except system.UnknownError:
        pytest.skip("system events may only be registered once per process")

    with pytest.raises(system.TimeoutError):
        events.wait(timeout_ms=500, buffer_size=1)
