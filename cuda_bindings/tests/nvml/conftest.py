# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0


import pytest
from cuda_python_test_helpers.arch_check import unsupported_before  # noqa: F401

from cuda.bindings import nvml


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


def get_devices():
    dev_count = nvml.device_get_count_v2()
    for i in range(dev_count):
        try:
            yield nvml.device_get_handle_by_index_v2(i)
        except nvml.NoPermissionError:
            continue  # ignore devices that can't be accessed


@pytest.fixture
def all_devices():
    with NVMLInitializer():
        yield sorted(set(get_devices()))


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
    handles = []
    with NVMLInitializer():
        dev_count = nvml.device_get_count_v2()

        for dev_idx in range(dev_count):
            try:
                dev = nvml.device_get_handle_by_index_v2(dev_idx)
            except nvml.NoPermissionError:
                continue
            for mig_idx in range(nmigs):
                try:
                    mig = nvml.device_get_mig_device_handle_by_index(dev, mig_idx)
                except nvml.NotFoundError:
                    # Not all MIG devices may be available
                    continue
                else:
                    handles.append(mig)
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
