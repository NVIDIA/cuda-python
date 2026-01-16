# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: LicenseRef-NVIDIA-SOFTWARE-LICENSE

from collections import namedtuple
from contextlib import contextmanager

import pytest
from cuda.bindings import _nvml as nvml


class NVMLInitializer:
    def __init__(self):
        pass

    def __enter__(self):
        nvml.init_v2()

    def __exit__(self, exception_type, exception, trace):
        nvml.shutdown()


@pytest.fixture
def nvml_init():
    with NVMLInitializer():
        yield


@pytest.fixture(scope="session", autouse=True)
def device_info():
    dev_count = None
    bus_id_to_board_details = {}

    with NVMLInitializer():
        dev_count = nvml.device_get_count_v2()

        # Store some details for each device now when we know NVML is in known state
        for i in range(dev_count):
            try:
                dev = nvml.device_get_handle_by_index_v2(i)
            except nvml.NoPermissionError:
                continue
            pci_info = nvml.device_get_pci_info_v3(dev)

            name = nvml.device_get_name(dev)
            # Get architecture name ex: Ampere, Kepler
            arch_id = nvml.device_get_architecture(dev)

            BoardCfg = namedtuple("BoardCfg", "name, ids_arr")
            board = BoardCfg(name, ids_arr=[(pci_info.pci_device_id, pci_info.pci_sub_system_id)])

            try:
                serial = nvml.device_get_serial(dev)
            except nvml.NvmlError:
                serial = None

            bus_id = pci_info.bus_id
            device_id = pci_info.device_
            uuid = nvml.device_get_uuid(dev)

            BoardDetails = namedtuple("BoardDetails", "name, board, arch_id, bus_id, device_id, serial")
            bus_id_to_board_details[uuid] = BoardDetails(name, board, arch_id, bus_id, device_id, serial)

    return bus_id_to_board_details


def get_devices(device_info):
    for uuid in list(device_info.keys()):
        try:
            yield nvml.device_get_handle_by_uuid(uuid)
        except nvml.NoPermissionError:
            continue  # ignore devices that can't be accessed


@pytest.fixture
def all_devices(device_info):
    with NVMLInitializer():
        yield sorted(list(set(get_devices(device_info))))


@pytest.fixture
def driver(nvml_init, request):
    driver_vsn = nvml.system_get_driver_version()
    # Return "major" version only
    return int(driver_vsn.split(".")[0])


@pytest.fixture
def ngpus(nvml_init):
    result = nvml.device_get_count_v2()
    assert result > 0
    return result


@pytest.fixture
def handles(ngpus):
    handles = [nvml.device_get_handle_by_index_v2(i) for i in range(ngpus)]
    assert len(handles) == ngpus
    return handles


@pytest.fixture
def nmigs(handles):
    return nvml.device_get_max_mig_device_count(handles[0])


@pytest.fixture
def mig_handles(nmigs):
    handles = [nvml.device_get_mig_device_handle_by_index(i) for i in range(nmigs)]
    assert len(handles) == nmigs
    return handles


@pytest.fixture
def serials(ngpus, handles):
    serials = [nvml.device_get_serial(handles[i]) for i in range(ngpus)]
    assert len(serials) == ngpus
    return serials


@pytest.fixture
def uuids(ngpus, handles):
    uuids = [nvml.device_get_uuid(handles[i]) for i in range(ngpus)]
    assert len(uuids) == ngpus
    return uuids


@pytest.fixture
def pci_info(ngpus, handles):
    pci_info = [nvml.device_get_pci_info_v3(handles[i]) for i in range(ngpus)]
    assert len(pci_info) == ngpus
    return pci_info


@contextmanager
def unsupported_before(device: int, expected_device_arch: nvml.DeviceArch | str | None):
    device_arch = nvml.device_get_architecture(device)

    if isinstance(expected_device_arch, nvml.DeviceArch):
        expected_device_arch_int = int(expected_device_arch)
    elif expected_device_arch == "FERMI":
        expected_device_arch_int = 1
    else:
        expected_device_arch_int = 0

    if expected_device_arch is None or expected_device_arch == "HAS_INFOROM" or device_arch == nvml.DeviceArch.UNKNOWN:
        # In this case, we don't /know/ if it will fail, but we are ok if it
        # does or does not.

        # TODO: There are APIs that are documented as supported only if the
        # device has an InfoROM, but I couldn't find a way to detect that.  For
        # now, they are just handled as "possibly failing".

        try:
            yield
        except nvml.NotSupportedError:
            pytest.skip(
                f"Unsupported call for device architecture {nvml.DeviceArch(device_arch).name} "
                f"on device '{nvml.device_get_name(device)}'"
            )
    elif int(device_arch) < expected_device_arch_int:
        # In this case, we /know/ if will fail, and we want to assert that it does.
        with pytest.raises(nvml.NotSupportedError):
            yield
        pytest.skip("Unsupported before {expected_device_arch.name}, got {nvml.device_get_name(device)}")
    else:
        # In this case, we /know/ it should work, and if it fails, the test should fail.
        yield
