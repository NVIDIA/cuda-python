# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

# ruff: noqa: E402

from .conftest import skip_if_nvml_unsupported, unsupported_before

pytestmark = skip_if_nvml_unsupported

import array
import multiprocessing
import os
import re
import warnings

import helpers
import pytest
from cuda.core import system

if system.CUDA_BINDINGS_NVML_IS_COMPATIBLE:
    from cuda.bindings import nvml
    from cuda.core.system import DeviceArch, _device


@pytest.fixture(autouse=True, scope="module")
def check_gpu_available():
    if not system.CUDA_BINDINGS_NVML_IS_COMPATIBLE or system.get_num_devices() == 0:
        pytest.skip("No GPUs available to run device tests", allow_module_level=True)


def test_devices_are_the_same_architecture():
    # The tests in this directory that use `unsupported_before` will generally
    # skip the entire test after the first device that isn't supported is found.
    # This means that if subsequent devices are of a different architecture,
    # they won't be tested properly.  This tests for the (hopefully rare) case
    # where a system has devices of different architectures and produces a warning.

    all_arches = set(device.arch for device in system.Device.get_all_devices())

    if len(all_arches) > 1:
        warnings.warn(  # noqa: B028
            f"System has devices of multiple architectures ({', '.join(x.name for x in all_arches)}). "
            f" Some tests may be skipped unexpectedly",
            UserWarning,
        )


def test_device_count():
    assert system.Device.get_device_count() == system.get_num_devices()


def test_to_cuda_device():
    from cuda.core import Device as CudaDevice

    for device in system.Device.get_all_devices():
        cuda_device = device.to_cuda_device()

        assert isinstance(cuda_device, CudaDevice)
        assert cuda_device.uuid == device.uuid

        # Technically, this test will only work with PCI devices, but are there
        # non-PCI devices we need to support?

        # CUDA only returns a 2-byte PCI bus ID domain, whereas NVML returns a
        # 4-byte domain
        assert cuda_device.pci_bus_id == device.pci_info.bus_id[4:]


def test_device_architecture():
    for device in system.Device.get_all_devices():
        device_arch = device.arch
        assert isinstance(device_arch, system.DeviceArch)


def test_device_bar1_memory():
    for device in system.Device.get_all_devices():
        with unsupported_before(device, DeviceArch.KEPLER):
            bar1_memory_info = device.bar1_memory_info
        free, total, used = (
            bar1_memory_info.free,
            bar1_memory_info.total,
            bar1_memory_info.used,
        )

        assert isinstance(bar1_memory_info, system.BAR1MemoryInfo)
        assert isinstance(free, int)
        assert isinstance(total, int)
        assert isinstance(used, int)

        assert free >= 0
        assert total >= 0
        assert used >= 0
        assert free + used == total


@pytest.mark.skipif(helpers.IS_WSL or helpers.IS_WINDOWS, reason="Device attributes not supported on WSL or Windows")
def test_device_cpu_affinity():
    for device in system.Device.get_all_devices():
        with unsupported_before(device, DeviceArch.KEPLER):
            affinity = device.get_cpu_affinity(system.AffinityScope.NODE)
        assert isinstance(affinity, list)
        os.sched_setaffinity(0, affinity)
        assert os.sched_getaffinity(0) == set(affinity)


@pytest.mark.skipif(helpers.IS_WSL or helpers.IS_WINDOWS, reason="Device attributes not supported on WSL or Windows")
def test_affinity():
    for device in system.Device.get_all_devices():
        for scope in (system.AffinityScope.NODE, system.AffinityScope.SOCKET):
            with unsupported_before(device, DeviceArch.KEPLER):
                affinity = device.get_cpu_affinity(scope)
            assert isinstance(affinity, list)

            affinity = device.get_memory_affinity(scope)
            assert isinstance(affinity, list)


def test_numa_node_id():
    for device in system.Device.get_all_devices():
        with unsupported_before(device, None):
            numa_node_id = device.numa_node_id
        assert isinstance(numa_node_id, int)
        assert numa_node_id >= -1


def test_device_cuda_compute_capability():
    for device in system.Device.get_all_devices():
        cuda_compute_capability = device.cuda_compute_capability
        assert isinstance(cuda_compute_capability, tuple)
        assert len(cuda_compute_capability) == 2
        assert all([isinstance(i, int) for i in cuda_compute_capability])
        assert 3 <= cuda_compute_capability[0] <= 99
        assert 0 <= cuda_compute_capability[1] <= 9


def test_device_memory():
    for device in system.Device.get_all_devices():
        memory_info = device.memory_info
        free, total, used, reserved = memory_info.free, memory_info.total, memory_info.used, memory_info.reserved

        assert isinstance(memory_info, system.MemoryInfo)
        assert isinstance(free, int)
        assert isinstance(total, int)
        assert isinstance(used, int)
        assert isinstance(reserved, int)

        assert free >= 0
        assert total >= 0
        assert used >= 0
        assert reserved >= 0
        assert free + used + reserved == total


def test_device_name():
    for device in system.Device.get_all_devices():
        name = device.name
        assert isinstance(name, str)
        assert len(name) > 0


def test_device_pci_info():
    for device in system.Device.get_all_devices():
        pci_info = device.pci_info
        assert isinstance(pci_info, system.PciInfo)

        assert isinstance(pci_info.bus_id, str)
        assert re.match("[a-f0-9]{8}:[a-f0-9]{2}:[a-f0-9]{2}.[a-f0-9]", pci_info.bus_id.lower())
        bus_id_domain = int(pci_info.bus_id.split(":")[0], 16)
        bus_id_bus = int(pci_info.bus_id.split(":")[1], 16)
        bus_id_device = int(pci_info.bus_id.split(":")[2][:2], 16)

        assert isinstance(pci_info.domain, int)
        assert 0x00 <= pci_info.domain <= 0xFFFFFFFF
        assert pci_info.domain == bus_id_domain

        assert isinstance(pci_info.bus, int)
        assert 0x00 <= pci_info.bus <= 0xFF
        assert pci_info.bus == bus_id_bus

        assert isinstance(pci_info.device, int)
        assert 0x00 <= pci_info.device <= 0xFF
        assert pci_info.device == bus_id_device

        assert isinstance(pci_info.vendor_id, int)
        assert 0x0000 <= pci_info.vendor_id <= 0xFFFF

        assert isinstance(pci_info.device_id, int)
        assert 0x0000 <= pci_info.device_id <= 0xFFFF

        assert isinstance(pci_info.subsystem_id, int)
        assert 0x00000000 <= pci_info.subsystem_id <= 0xFFFFFFFF

        assert isinstance(pci_info.base_class, int)
        assert 0x00 <= pci_info.base_class <= 0xFF

        assert isinstance(pci_info.sub_class, int)
        assert 0x00 <= pci_info.sub_class <= 0xFF

        assert isinstance(pci_info.get_max_pcie_link_generation(), int)
        assert 0 <= pci_info.get_max_pcie_link_generation() <= 0xFF

        assert isinstance(pci_info.get_gpu_max_pcie_link_generation(), int)
        assert 0 <= pci_info.get_gpu_max_pcie_link_generation() <= 0xFF

        assert isinstance(pci_info.get_max_pcie_link_width(), int)
        assert 0 <= pci_info.get_max_pcie_link_width() <= 0xFF

        assert isinstance(pci_info.get_current_pcie_link_generation(), int)
        assert 0 <= pci_info.get_current_pcie_link_generation() <= 0xFF

        assert isinstance(pci_info.get_current_pcie_link_width(), int)
        assert 0 <= pci_info.get_current_pcie_link_width() <= 0xFF

        assert isinstance(pci_info.get_pcie_throughput(system.PcieUtilCounter.PCIE_UTIL_TX_BYTES), int)

        assert isinstance(pci_info.get_pcie_replay_counter(), int)


def test_device_serial():
    for device in system.Device.get_all_devices():
        with unsupported_before(device, "HAS_INFOROM"):
            serial = device.serial
        assert isinstance(serial, str)
        assert len(serial) > 0


def test_device_uuid():
    for device in system.Device.get_all_devices():
        uuid = device.uuid
        assert isinstance(uuid, str)

        # Expands to GPU-8hex-4hex-4hex-4hex-12hex, where 8hex means 8 consecutive
        # hex characters, e.g.: "GPU-abcdef12-abcd-0123-4567-1234567890ab"


@pytest.mark.parametrize(
    "params",
    [
        {
            "input": [1152920405096267775, 0],
            "output": [i for i in range(20)] + [i + 40 for i in range(20)],
        },
        {
            "input": [17293823668613283840, 65535],
            "output": [i + 20 for i in range(20)] + [i + 60 for i in range(20)],
        },
        {"input": [18446744073709551615, 0], "output": [i for i in range(64)]},
        {"input": [0, 18446744073709551615], "output": [i + 64 for i in range(64)]},
    ],
)
def test_unpack_bitmask(params):
    assert _device._unpack_bitmask(array.array("Q", params["input"])) == params["output"]


def test_unpack_bitmask_single_value():
    with pytest.raises(TypeError):
        _device._unpack_bitmask(1)


@pytest.mark.skipif(helpers.IS_WSL or helpers.IS_WINDOWS, reason="Events not supported on WSL or Windows")
def test_register_events():
    # This is not the world's greatest test.  All of the events are pretty
    # infrequent and hard to simulate.  So all we do here is register an event,
    # wait with a timeout, and ensure that we get no event (since we didn't do
    # anything to trigger one).

    # Also, some hardware doesn't support any event types.

    for device in system.Device.get_all_devices():
        supported_events = device.get_supported_event_types()
        assert isinstance(supported_events, list)
        assert all(isinstance(ev, system.EventType) for ev in supported_events)

    for device in system.Device.get_all_devices():
        events = device.register_events([])
        with pytest.raises(system.TimeoutError):
            events.wait(timeout_ms=500)

    for device in system.Device.get_all_devices():
        events = device.register_events(0)
        with pytest.raises(system.TimeoutError):
            events.wait(timeout_ms=500)


def test_event_type_parsing():
    events = [system.EventType(1 << ev) for ev in _device._unpack_bitmask(array.array("Q", [3]))]
    assert events == [
        system.EventType.SINGLE_BIT_ECC_ERROR,
        system.EventType.DOUBLE_BIT_ECC_ERROR,
    ]


def test_device_brand():
    for device in system.Device.get_all_devices():
        brand = device.brand
        assert isinstance(brand, system.BrandType)
        assert isinstance(brand.name, str)
        assert isinstance(brand.value, int)


def test_device_pci_bus_id():
    for device in system.Device.get_all_devices():
        pci_bus_id = device.pci_info.bus_id
        assert isinstance(pci_bus_id, str)

        new_device = system.Device(pci_bus_id=device.pci_info.bus_id)
        assert new_device.index == device.index


@pytest.mark.skipif(helpers.IS_WSL or helpers.IS_WINDOWS, reason="Device attributes not supported on WSL or Windows")
def test_device_attributes():
    for device in system.Device.get_all_devices():
        # Docs say this should work on AMPERE or newer, but experimentally
        # that's not the case.
        with unsupported_before(device, None):
            attributes = device.attributes
        assert isinstance(attributes, system.DeviceAttributes)

        assert isinstance(attributes.multiprocessor_count, int)
        assert attributes.multiprocessor_count > 0

        assert isinstance(attributes.shared_copy_engine_count, int)
        assert isinstance(attributes.shared_decoder_count, int)
        assert isinstance(attributes.shared_encoder_count, int)
        assert isinstance(attributes.shared_jpeg_count, int)
        assert isinstance(attributes.shared_ofa_count, int)
        assert isinstance(attributes.gpu_instance_slice_count, int)
        assert isinstance(attributes.compute_instance_slice_count, int)
        assert isinstance(attributes.memory_size_mb, int)
        assert attributes.memory_size_mb > 0


def test_c2c_mode_enabled():
    for device in system.Device.get_all_devices():
        with unsupported_before(device, None):
            is_enabled = device.is_c2c_mode_enabled
        assert isinstance(is_enabled, bool)


@pytest.mark.skipif(helpers.IS_WSL or helpers.IS_WINDOWS, reason="Persistence mode not supported on WSL or Windows")
def test_persistence_mode_enabled():
    for device in system.Device.get_all_devices():
        is_enabled = device.persistence_mode_enabled
        assert isinstance(is_enabled, bool)
        try:
            device.persistence_mode_enabled = False
        except nvml.NoPermissionError as e:
            pytest.xfail(f"nvml.NoPermissionError: {e}")
        try:
            assert device.persistence_mode_enabled is False
        finally:
            device.persistence_mode_enabled = is_enabled


def test_field_values():
    for device in system.Device.get_all_devices():
        # TODO: Are there any fields that return double's?  It would be good to
        # test those.

        field_ids = [
            system.FieldId.DEV_TOTAL_ENERGY_CONSUMPTION,
            system.FieldId.DEV_PCIE_COUNT_TX_BYTES,
        ]
        field_values = device.get_field_values(field_ids)
        with unsupported_before(device, None):
            field_values.validate()

        with pytest.raises(TypeError):
            field_values["invalid_index"]

        assert isinstance(field_values, system.FieldValues)
        assert len(field_values) == len(field_ids)

        raw_values = field_values.get_all_values()
        assert all(x == y.value for x, y in zip(raw_values, field_values))

        for field_id, field_value in zip(field_ids, field_values):
            assert field_value.field_id == field_id
            assert type(field_value.value) is int
            assert field_value.latency_usec >= 0
            assert field_value.timestamp >= 0

        orig_timestamp = field_values[0].timestamp
        field_values = device.get_field_values(field_ids)
        assert field_values[0].timestamp >= orig_timestamp

        # Test only one element, because that's weirdly a special case
        field_ids = [
            system.FieldId.DEV_PCIE_REPLAY_COUNTER,
        ]
        field_values = device.get_field_values(field_ids)
        assert len(field_values) == 1
        field_values.validate()
        old_value = field_values[0].value

        # Test clear_field_values
        device.clear_field_values(field_ids)
        field_values = device.get_field_values(field_ids)
        field_values.validate()
        assert len(field_values) == 1
        assert field_values[0].value <= old_value


@pytest.mark.skipif(helpers.IS_WSL or helpers.IS_WINDOWS, reason="Device attributes not supported on WSL or Windows")
def test_get_all_devices_with_cpu_affinity():
    for i in range(multiprocessing.cpu_count()):
        for device in system.Device.get_all_devices_with_cpu_affinity(i):
            with unsupported_before(device, DeviceArch.KEPLER):
                affinity = device.get_cpu_affinity()
            assert isinstance(affinity, list)
            assert i in affinity


def test_index():
    for i, device in enumerate(system.Device.get_all_devices()):
        index = device.index
        assert isinstance(index, int)
        assert index == i


def test_module_id():
    for device in system.Device.get_all_devices():
        module_id = device.module_id
        assert isinstance(module_id, int)
        assert module_id >= 0


def test_addressing_mode():
    for device in system.Device.get_all_devices():
        # By docs, should be supported on TURING or newer, but experimentally,
        # is also unsupported on other hardware.
        with unsupported_before(device, None):
            addressing_mode = device.addressing_mode
        assert isinstance(addressing_mode, system.AddressingMode)


def test_display_mode():
    for device in system.Device.get_all_devices():
        display_mode = device.display_mode
        assert isinstance(display_mode, bool)

        display_active = device.display_active
        assert isinstance(display_active, bool)


def test_repair_status():
    for device in system.Device.get_all_devices():
        # By docs, should be supported on AMPERE or newer, but experimentally,
        # this seems to also work on some TURING systems.
        with unsupported_before(device, None):
            repair_status = device.repair_status
        assert isinstance(repair_status, system.RepairStatus)

        assert isinstance(repair_status.channel_repair_pending, bool)
        assert isinstance(repair_status.tpc_repair_pending, bool)


@pytest.mark.skipif(helpers.IS_WSL or helpers.IS_WINDOWS, reason="Device attributes not supported on WSL or Windows")
def test_get_topology_common_ancestor():
    # TODO: This is not a great test, and probably doesn't test much of anything
    # in practice on our CI.

    if system.Device.get_device_count() < 2:
        pytest.skip("Test requires at least 2 GPUs")
        return

    devices = list(system.Device.get_all_devices())

    ancestor = system.get_topology_common_ancestor(devices[0], devices[1])
    assert isinstance(ancestor, system.GpuTopologyLevel)


@pytest.mark.skipif(helpers.IS_WSL or helpers.IS_WINDOWS, reason="Device attributes not supported on WSL or Windows")
def test_get_p2p_status():
    # TODO: This is not a great test, and probably doesn't test much of anything
    # in practice on our CI.

    if system.Device.get_device_count() < 2:
        pytest.skip("Test requires at least 2 GPUs")
        return

    devices = list(system.Device.get_all_devices())

    status = system.get_p2p_status(devices[0], devices[1], system.GpuP2PCapsIndex.P2P_CAPS_INDEX_READ)
    assert isinstance(status, system.GpuP2PStatus)


@pytest.mark.skipif(helpers.IS_WSL or helpers.IS_WINDOWS, reason="Device attributes not supported on WSL or Windows")
def test_get_nearest_gpus():
    # TODO: This is not a great test, and probably doesn't test much of anything
    # in practice on our CI.

    for device in system.Device.get_all_devices():
        for near_device in device.get_topology_nearest_gpus(system.GpuTopologyLevel.TOPOLOGY_SINGLE):
            assert isinstance(near_device, system.Device)


@pytest.mark.skipif(helpers.IS_WSL or helpers.IS_WINDOWS, reason="Device attributes not supported on WSL or Windows")
def test_get_minor_number():
    for device in system.Device.get_all_devices():
        minor_number = device.minor_number
        assert isinstance(minor_number, int)
        assert minor_number >= 0


def test_get_inforom_version():
    for device in system.Device.get_all_devices():
        with unsupported_before(device, "HAS_INFOROM"):
            inforom = device.inforom

        inforom_image_version = inforom.image_version
        assert isinstance(inforom_image_version, str)
        assert len(inforom_image_version) > 0

        inforom_version = inforom.get_version(system.InforomObject.INFOROM_OEM)
        assert isinstance(inforom_version, str)
        assert len(inforom_version) > 0

        checksum = inforom.configuration_checksum
        assert isinstance(checksum, int)

        # TODO: This is untested locally.
        try:
            timestamp, duration_us = inforom.bbx_flush_time
        except (system.NotSupportedError, system.NotReadyError):
            pass
        else:
            assert isinstance(timestamp, int)
            assert timestamp > 0
            assert isinstance(duration_us, int)
            assert duration_us > 0

        with unsupported_before(device, "HAS_INFOROM"):
            board_part_number = inforom.board_part_number
        assert isinstance(board_part_number, str)
        assert len(board_part_number) > 0

        inforom.validate()


def test_auto_boosted_clocks_enabled():
    for device in system.Device.get_all_devices():
        # This API is supported on KEPLER and newer, but it also seems
        # unsupported elsewhere.
        with unsupported_before(device, None):
            current, default = device.get_auto_boosted_clocks_enabled()
        assert isinstance(current, bool)
        assert isinstance(default, bool)


def test_clock():
    for device in system.Device.get_all_devices():
        for clock_type in system.ClockType:
            clock = device.clock(clock_type)
            assert isinstance(clock, system.ClockInfo)

            # These are ordered from oldest API to newest API so we test as much
            # as we can on each hardware architecture.

            with unsupported_before(device, "FERMI"):
                pstate = device.performance_state

            min_, max_ = clock.get_min_max_clock_of_pstate_mhz(pstate)
            assert isinstance(min_, int)
            assert min_ >= 0
            assert isinstance(max_, int)
            assert max_ >= 0

            with unsupported_before(device, "FERMI"):
                max_mhz = clock.get_max_mhz()
            assert isinstance(max_mhz, int)
            assert max_mhz >= 0

            with unsupported_before(device, DeviceArch.KEPLER):
                current_mhz = clock.get_current_mhz()
            assert isinstance(current_mhz, int)
            assert current_mhz >= 0

            # Docs say this should work on PASCAL or newer, but experimentally,
            # is also unsupported on other hardware.
            with unsupported_before(device, DeviceArch.MAXWELL):
                try:
                    offsets = clock.get_offsets(pstate)
                except system.InvalidArgumentError:
                    pass
                else:
                    assert isinstance(offsets, system.ClockOffsets)
                    assert isinstance(offsets.clock_offset_mhz, int)
                    assert isinstance(offsets.max_offset_mhz, int)
                    assert isinstance(offsets.min_offset_mhz, int)

            # By docs, should be supported on PASCAL or newer, but experimentally,
            # is also unsupported on other hardware.
            with unsupported_before(device, None):
                max_customer_boost = clock.get_max_customer_boost_mhz()
            assert isinstance(max_customer_boost, int)
            assert max_customer_boost >= 0


def test_clock_event_reasons():
    for device in system.Device.get_all_devices():
        reasons = device.get_current_clock_event_reasons()
        assert all(isinstance(reason, system.ClocksEventReasons) for reason in reasons)

        reasons = device.get_supported_clock_event_reasons()
        assert all(isinstance(reason, system.ClocksEventReasons) for reason in reasons)


def test_fan():
    for device in system.Device.get_all_devices():
        # The fan APIs are only supported on discrete devices with fans,
        # but when they are not available `device.num_fans` returns 0.
        if device.num_fans == 0:
            pytest.skip("Device has no fans to test")

        for fan_idx in range(device.num_fans):
            fan_info = device.fan(fan_idx)
            assert isinstance(fan_info, system.FanInfo)

            speed = fan_info.speed
            assert isinstance(speed, int)
            assert 0 <= speed <= 200
            try:
                fan_info.speed = 50
            except nvml.NoPermissionError as e:
                pytest.xfail(f"nvml.NoPermissionError: {e}")
            try:
                fan_info.speed = speed

                speed_rpm = fan_info.speed_rpm
                assert isinstance(speed_rpm, int)
                assert speed_rpm >= 0

                target_speed = fan_info.target_speed
                assert isinstance(target_speed, int)
                assert speed <= target_speed * 2

                min_, max_ = fan_info.min_max_speed
                assert isinstance(min_, int)
                assert isinstance(max_, int)
                assert min_ <= max_

                control_policy = fan_info.control_policy
                assert isinstance(control_policy, system.FanControlPolicy)
            finally:
                fan_info.set_default_fan_speed()


def test_cooler():
    for device in system.Device.get_all_devices():
        # The cooler APIs are only supported on discrete devices with fans,
        # but when they are not available `device.num_fans` returns 0.
        if device.num_fans == 0:
            pytest.skip("Device has no coolers to test")

        with unsupported_before(device, DeviceArch.MAXWELL):
            cooler_info = device.cooler

        assert isinstance(cooler_info, system.CoolerInfo)

        signal_type = cooler_info.signal_type
        assert isinstance(signal_type, system.CoolerControl)

        target = cooler_info.target
        assert all(isinstance(t, system.CoolerTarget) for t in target)


def test_temperature():
    for device in system.Device.get_all_devices():
        temperature = device.temperature
        assert isinstance(temperature, system.Temperature)

        sensor = temperature.sensor()
        assert isinstance(sensor, int)
        assert sensor >= 0

        # By docs, should be supported on KEPLER or newer, but experimentally,
        # is also unsupported on other hardware.
        with unsupported_before(device, None):
            for threshold in list(system.TemperatureThresholds)[:-1]:
                t = temperature.threshold(threshold)
                assert isinstance(t, int)
                assert t >= 0

        with unsupported_before(device, None):
            margin = temperature.margin
        assert isinstance(margin, int)
        assert margin >= 0

        with unsupported_before(device, None):
            thermals = temperature.thermal_settings(system.ThermalTarget.ALL)
        assert isinstance(thermals, system.ThermalSettings)

        for i, sensor in enumerate(thermals):
            assert isinstance(sensor, system.ThermalSensor)
            assert isinstance(sensor.target, system.ThermalTarget)
            assert isinstance(sensor.controller, system.ThermalController)
            assert isinstance(sensor.default_min_temp, int)
            assert sensor.default_min_temp >= 0
            assert isinstance(sensor.default_max_temp, int)
            assert sensor.default_max_temp >= sensor.default_min_temp
            assert isinstance(sensor.current_temp, int)
            assert sensor.default_min_temp <= sensor.current_temp <= sensor.default_max_temp


def test_pstates():
    for device in system.Device.get_all_devices():
        pstate = device.performance_state
        assert isinstance(pstate, system.Pstates)

        pstates = device.get_supported_pstates()
        assert all(isinstance(p, system.Pstates) for p in pstates)

        dynamic_pstates_info = device.dynamic_pstates_info
        assert isinstance(dynamic_pstates_info, system.GpuDynamicPstatesInfo)

        assert len(dynamic_pstates_info) == nvml.MAX_GPU_UTILIZATIONS

        for utilization in dynamic_pstates_info:
            assert isinstance(utilization.is_present, bool)
            assert isinstance(utilization.percentage, int)
            assert isinstance(utilization.inc_threshold, int)
            assert isinstance(utilization.dec_threshold, int)
