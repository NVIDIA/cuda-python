# SPDX-FileCopyrightText: Copyright (c) 2024-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import pytest
from helpers.buffers import DummyDeviceMemoryResource, DummyUnifiedMemoryResource

from conftest import (
    create_managed_memory_resource_or_skip,
    skip_if_managed_memory_unsupported,
)
from cuda.core import Device, utils
from cuda.core._utils.cuda_utils import handle_return

try:
    from cuda.bindings import driver
except ImportError:
    from cuda import cuda as driver


_MANAGED_TEST_ALLOCATION_SIZE = 4096
_MEM_RANGE_ATTRIBUTE_VALUE_SIZE = 4
_READ_MOSTLY_ENABLED = 1
_HOST_LOCATION_ID = -1
_INVALID_HOST_DEVICE_ORDINAL = 0


def _get_mem_range_attr(buffer, attribute, data_size):
    # cuMemRangeGetAttribute returns a raw integer when data_size <= 4.
    return handle_return(driver.cuMemRangeGetAttribute(data_size, attribute, buffer.handle, buffer.size))


def _get_int_mem_range_attr(buffer, attribute):
    return _get_mem_range_attr(buffer, attribute, _MEM_RANGE_ATTRIBUTE_VALUE_SIZE)


def _skip_if_managed_allocation_unsupported(device):
    try:
        if not device.properties.managed_memory:
            pytest.skip("Device does not support managed memory operations")
    except AttributeError:
        pytest.skip("Managed-memory buffer operations require CUDA support")


def _skip_if_managed_location_ops_unsupported(device):
    _skip_if_managed_allocation_unsupported(device)
    try:
        if not device.properties.concurrent_managed_access:
            pytest.skip("Device does not support concurrent managed memory access")
    except AttributeError:
        pytest.skip("Managed-memory location operations require CUDA support")


def _skip_if_managed_discard_prefetch_unsupported(device):
    _skip_if_managed_location_ops_unsupported(device)
    if not hasattr(driver, "cuMemDiscardAndPrefetchBatchAsync"):
        pytest.skip("discard-prefetch requires cuda.bindings support")

    visible_devices = Device.get_all_devices()
    if not all(dev.properties.concurrent_managed_access for dev in visible_devices):
        pytest.skip("discard-prefetch requires concurrent managed access on all visible devices")


def test_managed_memory_prefetch_supports_managed_pool_allocations(init_cuda):
    device = Device()
    skip_if_managed_memory_unsupported(device)
    device.set_current()

    mr = create_managed_memory_resource_or_skip()
    buffer = mr.allocate(_MANAGED_TEST_ALLOCATION_SIZE)
    stream = device.create_stream()

    utils.prefetch(buffer, _HOST_LOCATION_ID, stream=stream)
    stream.sync()
    last_location = _get_int_mem_range_attr(
        buffer,
        driver.CUmem_range_attribute.CU_MEM_RANGE_ATTRIBUTE_LAST_PREFETCH_LOCATION,
    )
    assert last_location == _HOST_LOCATION_ID

    utils.prefetch(buffer, device, stream=stream)
    stream.sync()
    last_location = _get_int_mem_range_attr(
        buffer,
        driver.CUmem_range_attribute.CU_MEM_RANGE_ATTRIBUTE_LAST_PREFETCH_LOCATION,
    )
    assert last_location == device.device_id

    buffer.close()


def test_managed_memory_advise_supports_external_managed_allocations(init_cuda):
    from cuda.core.utils import Location

    device = Device()
    _skip_if_managed_location_ops_unsupported(device)
    device.set_current()

    buffer = DummyUnifiedMemoryResource(device).allocate(_MANAGED_TEST_ALLOCATION_SIZE)

    utils.advise(buffer, "set_read_mostly")
    assert (
        _get_int_mem_range_attr(
            buffer,
            driver.CUmem_range_attribute.CU_MEM_RANGE_ATTRIBUTE_READ_MOSTLY,
        )
        == _READ_MOSTLY_ENABLED
    )

    # cuda.bindings currently exposes the combined location attributes for
    # cuMemRangeGetAttribute, so use the legacy location query here.
    utils.advise(buffer, "set_preferred_location", Location.host())
    preferred_location = _get_int_mem_range_attr(
        buffer,
        driver.CUmem_range_attribute.CU_MEM_RANGE_ATTRIBUTE_PREFERRED_LOCATION,
    )
    assert preferred_location == _HOST_LOCATION_ID

    buffer.close()


def test_managed_memory_prefetch_supports_external_managed_allocations(init_cuda):
    device = Device()
    _skip_if_managed_location_ops_unsupported(device)
    device.set_current()

    buffer = DummyUnifiedMemoryResource(device).allocate(_MANAGED_TEST_ALLOCATION_SIZE)
    stream = device.create_stream()

    utils.prefetch(buffer, device, stream=stream)
    stream.sync()

    last_location = _get_int_mem_range_attr(
        buffer,
        driver.CUmem_range_attribute.CU_MEM_RANGE_ATTRIBUTE_LAST_PREFETCH_LOCATION,
    )
    assert last_location == device.device_id

    buffer.close()


def test_managed_memory_discard_prefetch_supports_managed_pool_allocations(init_cuda):
    device = Device()
    skip_if_managed_memory_unsupported(device)
    _skip_if_managed_discard_prefetch_unsupported(device)
    device.set_current()

    mr = create_managed_memory_resource_or_skip()
    buffer = mr.allocate(_MANAGED_TEST_ALLOCATION_SIZE)
    stream = device.create_stream()

    utils.prefetch(buffer, _HOST_LOCATION_ID, stream=stream)
    stream.sync()

    utils.discard_prefetch(buffer, device, stream=stream)
    stream.sync()

    last_location = _get_int_mem_range_attr(
        buffer,
        driver.CUmem_range_attribute.CU_MEM_RANGE_ATTRIBUTE_LAST_PREFETCH_LOCATION,
    )
    assert last_location == device.device_id

    buffer.close()


def test_managed_memory_discard_prefetch_supports_external_managed_allocations(init_cuda):
    device = Device()
    _skip_if_managed_discard_prefetch_unsupported(device)
    device.set_current()

    buffer = DummyUnifiedMemoryResource(device).allocate(_MANAGED_TEST_ALLOCATION_SIZE)
    stream = device.create_stream()

    utils.prefetch(buffer, _HOST_LOCATION_ID, stream=stream)
    stream.sync()

    utils.discard_prefetch(buffer, device, stream=stream)
    stream.sync()

    last_location = _get_int_mem_range_attr(
        buffer,
        driver.CUmem_range_attribute.CU_MEM_RANGE_ATTRIBUTE_LAST_PREFETCH_LOCATION,
    )
    assert last_location == device.device_id

    buffer.close()


def test_managed_memory_operations_reject_non_managed_allocations(init_cuda):
    device = Device()
    device.set_current()

    buffer = DummyDeviceMemoryResource(device).allocate(_MANAGED_TEST_ALLOCATION_SIZE)
    stream = device.create_stream()

    with pytest.raises(ValueError, match="managed-memory allocation"):
        utils.advise(buffer, "set_read_mostly")
    with pytest.raises(ValueError, match="managed-memory allocation"):
        utils.prefetch(buffer, device, stream=stream)
    with pytest.raises(ValueError, match="managed-memory allocation"):
        utils.discard_prefetch(buffer, device, stream=stream)

    buffer.close()


def test_managed_memory_operation_validation(init_cuda):
    device = Device()
    skip_if_managed_memory_unsupported(device)
    device.set_current()

    mr = create_managed_memory_resource_or_skip()
    buffer = mr.allocate(_MANAGED_TEST_ALLOCATION_SIZE)
    stream = device.create_stream()

    with pytest.raises(ValueError, match="location is required"):
        utils.prefetch(buffer, stream=stream)
    from cuda.core.utils import Location

    with pytest.raises(ValueError, match="does not support location_type='host_numa'"):
        utils.advise(buffer, "set_accessed_by", Location.host_numa(_INVALID_HOST_DEVICE_ORDINAL))

    buffer.close()


def test_managed_memory_advise_location_validation(init_cuda):
    """Verify doc-specified location constraints for each advice kind."""
    from cuda.core.utils import Location

    device = Device()
    _skip_if_managed_location_ops_unsupported(device)
    device.set_current()

    buffer = DummyUnifiedMemoryResource(device).allocate(_MANAGED_TEST_ALLOCATION_SIZE)

    # set_read_mostly works without a location (location is ignored)
    utils.advise(buffer, "set_read_mostly")

    # set_preferred_location requires a location; device ordinal works
    utils.advise(buffer, "set_preferred_location", device.device_id)

    # set_preferred_location with host location
    utils.advise(buffer, "set_preferred_location", Location.host())

    # set_accessed_by with host_numa raises ValueError (INVALID per CUDA docs)
    with pytest.raises(ValueError, match="does not support location_type='host_numa'"):
        utils.advise(buffer, "set_accessed_by", Location.host_numa(0))

    # set_accessed_by with host_numa_current also raises ValueError
    with pytest.raises(ValueError, match="does not support location_type='host_numa_current'"):
        utils.advise(buffer, "set_accessed_by", Location.host_numa_current())

    # Inferred location from int: -1 maps to host, 0 maps to device
    utils.advise(buffer, "set_preferred_location", -1)
    utils.advise(buffer, "set_preferred_location", 0)

    buffer.close()


def test_managed_memory_advise_accepts_enum_value(init_cuda):
    """advise() accepts CUmem_advise enum values directly, not just string aliases."""
    device = Device()
    _skip_if_managed_location_ops_unsupported(device)
    device.set_current()

    buffer = DummyUnifiedMemoryResource(device).allocate(_MANAGED_TEST_ALLOCATION_SIZE)

    advice_enum = driver.CUmem_advise.CU_MEM_ADVISE_SET_READ_MOSTLY
    utils.advise(buffer, advice_enum)

    assert (
        _get_int_mem_range_attr(
            buffer,
            driver.CUmem_range_attribute.CU_MEM_RANGE_ATTRIBUTE_READ_MOSTLY,
        )
        == _READ_MOSTLY_ENABLED
    )

    buffer.close()


def test_managed_memory_advise_invalid_advice_values(init_cuda):
    """advise() rejects invalid advice strings and wrong types."""
    device = Device()
    _skip_if_managed_allocation_unsupported(device)
    device.set_current()

    buffer = DummyUnifiedMemoryResource(device).allocate(_MANAGED_TEST_ALLOCATION_SIZE)

    with pytest.raises(ValueError, match="advice must be one of"):
        utils.advise(buffer, "not_a_real_advice")

    with pytest.raises(TypeError, match="advice must be"):
        utils.advise(buffer, 42)

    buffer.close()


class TestLocation:
    def test_device_constructor(self):
        from cuda.core.utils import Location

        loc = Location.device(0)
        assert loc.kind == "device"
        assert loc.id == 0

    def test_host_constructor(self):
        from cuda.core.utils import Location

        loc = Location.host()
        assert loc.kind == "host"
        assert loc.id is None

    def test_host_numa_constructor(self):
        from cuda.core.utils import Location

        loc = Location.host_numa(3)
        assert loc.kind == "host_numa"
        assert loc.id == 3

    def test_host_numa_current_constructor(self):
        from cuda.core.utils import Location

        loc = Location.host_numa_current()
        assert loc.kind == "host_numa_current"
        assert loc.id is None

    def test_frozen(self):
        import dataclasses

        from cuda.core.utils import Location

        loc = Location.device(0)
        with pytest.raises(dataclasses.FrozenInstanceError):
            loc.id = 1

    def test_invalid_device_id(self):
        from cuda.core.utils import Location

        with pytest.raises(ValueError, match="device id must be >= 0"):
            Location.device(-1)

    def test_invalid_kind(self):
        from cuda.core.utils import Location

        with pytest.raises(ValueError, match="kind must be one of"):
            Location(kind="not_a_kind", id=None)


class TestLocationCoerce:
    def test_passthrough(self):
        from cuda.core._memory._managed_location import _coerce_location
        from cuda.core.utils import Location

        loc = Location.device(0)
        assert _coerce_location(loc) is loc

    def test_int_device(self):
        from cuda.core._memory._managed_location import _coerce_location

        assert _coerce_location(0).kind == "device"
        assert _coerce_location(0).id == 0

    def test_int_minus_one_is_host(self):
        from cuda.core._memory._managed_location import _coerce_location

        assert _coerce_location(-1).kind == "host"

    def test_device_object(self, init_cuda):
        from cuda.core import Device
        from cuda.core._memory._managed_location import _coerce_location

        dev = Device()
        loc = _coerce_location(dev)
        assert loc.kind == "device"
        assert loc.id == dev.device_id

    def test_none_when_disallowed(self):
        from cuda.core._memory._managed_location import _coerce_location

        with pytest.raises(ValueError, match="location is required"):
            _coerce_location(None, allow_none=False)

    def test_none_when_allowed(self):
        from cuda.core._memory._managed_location import _coerce_location

        assert _coerce_location(None, allow_none=True) is None

    def test_bad_int(self):
        from cuda.core._memory._managed_location import _coerce_location

        with pytest.raises(ValueError, match="device ordinal"):
            _coerce_location(-2)

    def test_bad_type(self):
        from cuda.core._memory._managed_location import _coerce_location

        with pytest.raises(TypeError, match="Location, Device, int, or None"):
            _coerce_location("device")


class TestPrefetch:
    def test_single_with_location_host(self, init_cuda):
        from cuda.core.utils import Location, prefetch

        device = Device()
        skip_if_managed_memory_unsupported(device)
        device.set_current()
        mr = create_managed_memory_resource_or_skip()
        buf = mr.allocate(_MANAGED_TEST_ALLOCATION_SIZE)
        stream = device.create_stream()

        prefetch(buf, Location.host(), stream=stream)
        stream.sync()
        last = _get_int_mem_range_attr(
            buf,
            driver.CUmem_range_attribute.CU_MEM_RANGE_ATTRIBUTE_LAST_PREFETCH_LOCATION,
        )
        assert last == _HOST_LOCATION_ID
        buf.close()

    def test_batched_same_location(self, init_cuda):
        from cuda.core.utils import Location, prefetch

        device = Device()
        skip_if_managed_memory_unsupported(device)
        if not hasattr(driver, "cuMemPrefetchBatchAsync"):
            pytest.skip("cuMemPrefetchBatchAsync unavailable")
        device.set_current()
        mr = create_managed_memory_resource_or_skip()
        bufs = [mr.allocate(_MANAGED_TEST_ALLOCATION_SIZE) for _ in range(3)]
        stream = device.create_stream()

        prefetch(bufs, Location.device(device.device_id), stream=stream)
        stream.sync()

        for buf in bufs:
            last = _get_int_mem_range_attr(
                buf,
                driver.CUmem_range_attribute.CU_MEM_RANGE_ATTRIBUTE_LAST_PREFETCH_LOCATION,
            )
            assert last == device.device_id
            buf.close()

    def test_batched_per_buffer_location(self, init_cuda):
        from cuda.core.utils import Location, prefetch

        device = Device()
        skip_if_managed_memory_unsupported(device)
        if not hasattr(driver, "cuMemPrefetchBatchAsync"):
            pytest.skip("cuMemPrefetchBatchAsync unavailable")
        device.set_current()
        mr = create_managed_memory_resource_or_skip()
        bufs = [mr.allocate(_MANAGED_TEST_ALLOCATION_SIZE) for _ in range(2)]
        stream = device.create_stream()

        prefetch(bufs, [Location.host(), Location.device(device.device_id)], stream=stream)
        stream.sync()

        last0 = _get_int_mem_range_attr(
            bufs[0],
            driver.CUmem_range_attribute.CU_MEM_RANGE_ATTRIBUTE_LAST_PREFETCH_LOCATION,
        )
        last1 = _get_int_mem_range_attr(
            bufs[1],
            driver.CUmem_range_attribute.CU_MEM_RANGE_ATTRIBUTE_LAST_PREFETCH_LOCATION,
        )
        assert last0 == _HOST_LOCATION_ID
        assert last1 == device.device_id
        for buf in bufs:
            buf.close()

    def test_length_mismatch(self, init_cuda):
        from cuda.core.utils import Location, prefetch

        device = Device()
        skip_if_managed_memory_unsupported(device)
        device.set_current()
        mr = create_managed_memory_resource_or_skip()
        bufs = [mr.allocate(_MANAGED_TEST_ALLOCATION_SIZE) for _ in range(2)]
        stream = device.create_stream()

        with pytest.raises(ValueError, match="length"):
            prefetch(bufs, [Location.host()], stream=stream)
        for buf in bufs:
            buf.close()

    def test_rejects_non_managed(self, init_cuda):
        from cuda.core.utils import Location, prefetch

        device = Device()
        device.set_current()
        buf = DummyDeviceMemoryResource(device).allocate(_MANAGED_TEST_ALLOCATION_SIZE)
        stream = device.create_stream()
        with pytest.raises(ValueError, match="managed-memory"):
            prefetch(buf, Location.host(), stream=stream)
        buf.close()

    def test_location_required(self, init_cuda):
        from cuda.core.utils import prefetch

        device = Device()
        skip_if_managed_memory_unsupported(device)
        device.set_current()
        mr = create_managed_memory_resource_or_skip()
        buf = mr.allocate(_MANAGED_TEST_ALLOCATION_SIZE)
        stream = device.create_stream()
        with pytest.raises(ValueError, match="location is required"):
            prefetch(buf, None, stream=stream)
        buf.close()

    def test_options_must_be_none(self, init_cuda):
        from cuda.core.utils import Location, prefetch

        device = Device()
        skip_if_managed_memory_unsupported(device)
        device.set_current()
        mr = create_managed_memory_resource_or_skip()
        buf = mr.allocate(_MANAGED_TEST_ALLOCATION_SIZE)
        stream = device.create_stream()
        with pytest.raises(TypeError, match="must be an? .*Options instance or None"):
            prefetch(buf, Location.host(), options={}, stream=stream)
        buf.close()


class TestDiscard:
    def test_single_buffer(self, init_cuda):
        from cuda.core.utils import Location, discard, prefetch

        device = Device()
        skip_if_managed_memory_unsupported(device)
        if not hasattr(driver, "cuMemDiscardBatchAsync"):
            pytest.skip("cuMemDiscardBatchAsync unavailable")
        device.set_current()
        mr = create_managed_memory_resource_or_skip()
        buf = mr.allocate(_MANAGED_TEST_ALLOCATION_SIZE)
        stream = device.create_stream()
        prefetch(buf, Location.device(device.device_id), stream=stream)
        stream.sync()
        discard(buf, stream=stream)
        stream.sync()
        buf.close()

    def test_batched(self, init_cuda):
        from cuda.core.utils import Location, discard, prefetch

        device = Device()
        skip_if_managed_memory_unsupported(device)
        if not hasattr(driver, "cuMemDiscardBatchAsync"):
            pytest.skip("cuMemDiscardBatchAsync unavailable")
        device.set_current()
        mr = create_managed_memory_resource_or_skip()
        bufs = [mr.allocate(_MANAGED_TEST_ALLOCATION_SIZE) for _ in range(3)]
        stream = device.create_stream()
        prefetch(bufs, Location.device(device.device_id), stream=stream)
        stream.sync()
        discard(bufs, stream=stream)
        stream.sync()
        for buf in bufs:
            buf.close()

    def test_rejects_non_managed(self, init_cuda):
        from cuda.core.utils import discard

        device = Device()
        device.set_current()
        buf = DummyDeviceMemoryResource(device).allocate(_MANAGED_TEST_ALLOCATION_SIZE)
        stream = device.create_stream()
        with pytest.raises(ValueError, match="managed-memory"):
            discard(buf, stream=stream)
        buf.close()

    def test_options_must_be_none(self, init_cuda):
        from cuda.core.utils import discard

        device = Device()
        skip_if_managed_memory_unsupported(device)
        device.set_current()
        mr = create_managed_memory_resource_or_skip()
        buf = mr.allocate(_MANAGED_TEST_ALLOCATION_SIZE)
        stream = device.create_stream()
        with pytest.raises(TypeError, match="must be an? .*Options instance or None"):
            discard(buf, options={}, stream=stream)
        buf.close()


class TestDiscardPrefetch:
    def test_single_buffer(self, init_cuda):
        from cuda.core.utils import Location, discard_prefetch, prefetch

        device = Device()
        skip_if_managed_memory_unsupported(device)
        if not hasattr(driver, "cuMemDiscardAndPrefetchBatchAsync"):
            pytest.skip("cuMemDiscardAndPrefetchBatchAsync unavailable")
        device.set_current()
        mr = create_managed_memory_resource_or_skip()
        buf = mr.allocate(_MANAGED_TEST_ALLOCATION_SIZE)
        stream = device.create_stream()

        prefetch(buf, Location.host(), stream=stream)
        stream.sync()
        discard_prefetch(buf, Location.device(device.device_id), stream=stream)
        stream.sync()

        last = _get_int_mem_range_attr(
            buf,
            driver.CUmem_range_attribute.CU_MEM_RANGE_ATTRIBUTE_LAST_PREFETCH_LOCATION,
        )
        assert last == device.device_id
        buf.close()

    def test_batched_same_location(self, init_cuda):
        from cuda.core.utils import Location, discard_prefetch, prefetch

        device = Device()
        skip_if_managed_memory_unsupported(device)
        if not hasattr(driver, "cuMemDiscardAndPrefetchBatchAsync"):
            pytest.skip("cuMemDiscardAndPrefetchBatchAsync unavailable")
        device.set_current()
        mr = create_managed_memory_resource_or_skip()
        bufs = [mr.allocate(_MANAGED_TEST_ALLOCATION_SIZE) for _ in range(2)]
        stream = device.create_stream()
        prefetch(bufs, Location.host(), stream=stream)
        stream.sync()
        discard_prefetch(bufs, Location.device(device.device_id), stream=stream)
        stream.sync()
        for buf in bufs:
            last = _get_int_mem_range_attr(
                buf,
                driver.CUmem_range_attribute.CU_MEM_RANGE_ATTRIBUTE_LAST_PREFETCH_LOCATION,
            )
            assert last == device.device_id
            buf.close()

    def test_length_mismatch(self, init_cuda):
        from cuda.core.utils import Location, discard_prefetch

        device = Device()
        skip_if_managed_memory_unsupported(device)
        device.set_current()
        mr = create_managed_memory_resource_or_skip()
        bufs = [mr.allocate(_MANAGED_TEST_ALLOCATION_SIZE) for _ in range(2)]
        stream = device.create_stream()
        with pytest.raises(ValueError, match="length"):
            discard_prefetch(bufs, [Location.host()], stream=stream)
        for buf in bufs:
            buf.close()

    def test_rejects_non_managed(self, init_cuda):
        from cuda.core.utils import Location, discard_prefetch

        device = Device()
        device.set_current()
        buf = DummyDeviceMemoryResource(device).allocate(_MANAGED_TEST_ALLOCATION_SIZE)
        stream = device.create_stream()
        with pytest.raises(ValueError, match="managed-memory"):
            discard_prefetch(buf, Location.host(), stream=stream)
        buf.close()


class TestAdvise:
    def test_batched_same_advice(self, init_cuda):
        from cuda.core.utils import advise

        device = Device()
        _skip_if_managed_location_ops_unsupported(device)
        device.set_current()
        mr = DummyUnifiedMemoryResource(device)
        bufs = [mr.allocate(_MANAGED_TEST_ALLOCATION_SIZE) for _ in range(2)]
        advise(bufs, "set_read_mostly")
        # Query all attributes BEFORE closing any buffer. On CUDA 12, freeing
        # a managed allocation can clear read-mostly advice on neighboring
        # ranges; close-then-query in a single loop falsely flags the later
        # iterations as having lost the advice.
        results = [
            _get_int_mem_range_attr(
                buf,
                driver.CUmem_range_attribute.CU_MEM_RANGE_ATTRIBUTE_READ_MOSTLY,
            )
            for buf in bufs
        ]
        for buf in bufs:
            buf.close()
        for r in results:
            assert r == _READ_MOSTLY_ENABLED

    def test_batched_per_buffer_location(self, init_cuda):
        from cuda.core.utils import Location, advise

        device = Device()
        _skip_if_managed_location_ops_unsupported(device)
        device.set_current()
        mr = DummyUnifiedMemoryResource(device)
        bufs = [mr.allocate(_MANAGED_TEST_ALLOCATION_SIZE) for _ in range(2)]
        advise(
            bufs,
            "set_preferred_location",
            [Location.host(), Location.device(device.device_id)],
        )
        for buf in bufs:
            buf.close()

    def test_options_must_be_none(self, init_cuda):
        from cuda.core.utils import advise

        device = Device()
        _skip_if_managed_allocation_unsupported(device)
        device.set_current()
        buf = DummyUnifiedMemoryResource(device).allocate(_MANAGED_TEST_ALLOCATION_SIZE)
        with pytest.raises(TypeError, match="must be an? .*Options instance or None"):
            advise(buf, "set_read_mostly", options={})
        buf.close()
