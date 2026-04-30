# SPDX-FileCopyrightText: Copyright (c) 2024-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import pytest
from helpers.buffers import DummyDeviceMemoryResource, DummyUnifiedMemoryResource

from conftest import (
    create_managed_memory_resource_or_skip,
    skip_if_managed_memory_unsupported,
)
from cuda.core import Device, Host, ManagedBuffer, utils
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

    utils.advise(buffer, "set_preferred_location", Host())
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

    with pytest.raises(ValueError, match="does not support location_type='host_numa'"):
        utils.advise(buffer, "set_accessed_by", Host(numa_id=_INVALID_HOST_DEVICE_ORDINAL))

    buffer.close()


def test_managed_memory_advise_location_validation(init_cuda):
    """Verify doc-specified location constraints for each advice kind."""
    device = Device()
    _skip_if_managed_location_ops_unsupported(device)
    device.set_current()

    buffer = DummyUnifiedMemoryResource(device).allocate(_MANAGED_TEST_ALLOCATION_SIZE)

    # set_read_mostly works without a location (location is ignored)
    utils.advise(buffer, "set_read_mostly")

    # set_preferred_location requires a location; device ordinal works
    utils.advise(buffer, "set_preferred_location", device.device_id)

    # set_preferred_location with host location
    utils.advise(buffer, "set_preferred_location", Host())

    # set_accessed_by with host_numa raises ValueError (INVALID per CUDA docs)
    with pytest.raises(ValueError, match="does not support location_type='host_numa'"):
        utils.advise(buffer, "set_accessed_by", Host(numa_id=0))

    # set_accessed_by with host_numa_current also raises ValueError
    with pytest.raises(ValueError, match="does not support location_type='host_numa_current'"):
        utils.advise(buffer, "set_accessed_by", Host.numa_current())

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


class TestHost:
    def test_default(self):
        h = Host()
        assert h.numa_id is None
        assert h.is_numa_current is False

    def test_numa(self):
        h = Host(numa_id=3)
        assert h.numa_id == 3
        assert h.is_numa_current is False

    def test_numa_current(self):
        h = Host.numa_current()
        assert h.is_numa_current is True
        assert h.numa_id is None

    def test_invalid_numa_id(self):
        with pytest.raises(ValueError, match="numa_id must be a non-negative int"):
            Host(numa_id=-1)

    def test_numa_current_with_id_rejected(self):
        with pytest.raises(ValueError, match="numa_current"):
            Host(numa_id=0, is_numa_current=True)

    def test_frozen(self):
        import dataclasses

        h = Host(numa_id=2)
        with pytest.raises(dataclasses.FrozenInstanceError):
            h.numa_id = 3

    def test_eq_hash(self):
        # Frozen dataclass equality is structural.
        assert Host() == Host()
        assert Host(numa_id=1) == Host(numa_id=1)
        assert Host() != Host(numa_id=0)
        assert Host.numa_current() != Host()
        assert hash(Host(numa_id=1)) == hash(Host(numa_id=1))


class TestLocationCoerce:
    """The coerce helper is internal; verify Device/Host/int/None inputs."""

    def test_device_passthrough(self, init_cuda):
        from cuda.core._memory._managed_location import _coerce_location

        dev = Device()
        spec = _coerce_location(dev)
        assert spec.kind == "device"
        assert spec.id == dev.device_id

    def test_host_passthrough(self):
        from cuda.core._memory._managed_location import _coerce_location

        spec = _coerce_location(Host())
        assert spec.kind == "host"

    def test_host_numa_passthrough(self):
        from cuda.core._memory._managed_location import _coerce_location

        spec = _coerce_location(Host(numa_id=3))
        assert spec.kind == "host_numa"
        assert spec.id == 3

    def test_host_numa_current_passthrough(self):
        from cuda.core._memory._managed_location import _coerce_location

        spec = _coerce_location(Host.numa_current())
        assert spec.kind == "host_numa_current"

    def test_int_device(self):
        from cuda.core._memory._managed_location import _coerce_location

        spec = _coerce_location(0)
        assert spec.kind == "device"
        assert spec.id == 0

    def test_int_minus_one_is_host(self):
        from cuda.core._memory._managed_location import _coerce_location

        assert _coerce_location(-1).kind == "host"

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

        with pytest.raises(TypeError, match="Device, Host, int, or None"):
            _coerce_location("device")


class TestPrefetch:
    def test_single_with_host_location(self, init_cuda):
        from cuda.core.utils import prefetch

        device = Device()
        skip_if_managed_memory_unsupported(device)
        device.set_current()
        mr = create_managed_memory_resource_or_skip()
        buf = mr.allocate(_MANAGED_TEST_ALLOCATION_SIZE)
        stream = device.create_stream()

        prefetch(buf, Host(), stream=stream)
        stream.sync()
        last = _get_int_mem_range_attr(
            buf,
            driver.CUmem_range_attribute.CU_MEM_RANGE_ATTRIBUTE_LAST_PREFETCH_LOCATION,
        )
        assert last == _HOST_LOCATION_ID
        buf.close()

    def test_batched_same_location(self, init_cuda):
        from cuda.core.utils import prefetch

        device = Device()
        skip_if_managed_memory_unsupported(device)
        if not hasattr(driver, "cuMemPrefetchBatchAsync"):
            pytest.skip("cuMemPrefetchBatchAsync unavailable")
        device.set_current()
        mr = create_managed_memory_resource_or_skip()
        bufs = [mr.allocate(_MANAGED_TEST_ALLOCATION_SIZE) for _ in range(3)]
        stream = device.create_stream()

        prefetch(bufs, device, stream=stream)
        stream.sync()

        for buf in bufs:
            last = _get_int_mem_range_attr(
                buf,
                driver.CUmem_range_attribute.CU_MEM_RANGE_ATTRIBUTE_LAST_PREFETCH_LOCATION,
            )
            assert last == device.device_id
            buf.close()

    def test_batched_per_buffer_location(self, init_cuda):
        from cuda.core.utils import prefetch

        device = Device()
        skip_if_managed_memory_unsupported(device)
        if not hasattr(driver, "cuMemPrefetchBatchAsync"):
            pytest.skip("cuMemPrefetchBatchAsync unavailable")
        device.set_current()
        mr = create_managed_memory_resource_or_skip()
        bufs = [mr.allocate(_MANAGED_TEST_ALLOCATION_SIZE) for _ in range(2)]
        stream = device.create_stream()

        prefetch(bufs, [Host(), device], stream=stream)
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
        from cuda.core.utils import prefetch

        device = Device()
        skip_if_managed_memory_unsupported(device)
        device.set_current()
        mr = create_managed_memory_resource_or_skip()
        bufs = [mr.allocate(_MANAGED_TEST_ALLOCATION_SIZE) for _ in range(2)]
        stream = device.create_stream()

        with pytest.raises(ValueError, match="length"):
            prefetch(bufs, [Host()], stream=stream)
        for buf in bufs:
            buf.close()

    def test_rejects_non_managed(self, init_cuda):
        from cuda.core.utils import prefetch

        device = Device()
        device.set_current()
        buf = DummyDeviceMemoryResource(device).allocate(_MANAGED_TEST_ALLOCATION_SIZE)
        stream = device.create_stream()
        with pytest.raises(ValueError, match="managed-memory"):
            prefetch(buf, Host(), stream=stream)
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

    def test_options_must_be_options_dataclass_or_none(self, init_cuda):
        from cuda.core.utils import prefetch

        device = Device()
        skip_if_managed_memory_unsupported(device)
        device.set_current()
        mr = create_managed_memory_resource_or_skip()
        buf = mr.allocate(_MANAGED_TEST_ALLOCATION_SIZE)
        stream = device.create_stream()
        with pytest.raises(TypeError, match="must be an? .*Options instance or None"):
            prefetch(buf, Host(), options={}, stream=stream)
        buf.close()


class TestDiscard:
    def test_single_buffer(self, init_cuda):
        from cuda.core.utils import discard, prefetch

        device = Device()
        skip_if_managed_memory_unsupported(device)
        if not hasattr(driver, "cuMemDiscardBatchAsync"):
            pytest.skip("cuMemDiscardBatchAsync unavailable")
        device.set_current()
        mr = create_managed_memory_resource_or_skip()
        buf = mr.allocate(_MANAGED_TEST_ALLOCATION_SIZE)
        stream = device.create_stream()
        prefetch(buf, device, stream=stream)
        stream.sync()
        discard(buf, stream=stream)
        stream.sync()
        buf.close()

    def test_batched(self, init_cuda):
        from cuda.core.utils import discard, prefetch

        device = Device()
        skip_if_managed_memory_unsupported(device)
        if not hasattr(driver, "cuMemDiscardBatchAsync"):
            pytest.skip("cuMemDiscardBatchAsync unavailable")
        device.set_current()
        mr = create_managed_memory_resource_or_skip()
        bufs = [mr.allocate(_MANAGED_TEST_ALLOCATION_SIZE) for _ in range(3)]
        stream = device.create_stream()
        prefetch(bufs, device, stream=stream)
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

    def test_options_must_be_options_dataclass_or_none(self, init_cuda):
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
        from cuda.core.utils import discard_prefetch, prefetch

        device = Device()
        skip_if_managed_memory_unsupported(device)
        if not hasattr(driver, "cuMemDiscardAndPrefetchBatchAsync"):
            pytest.skip("cuMemDiscardAndPrefetchBatchAsync unavailable")
        device.set_current()
        mr = create_managed_memory_resource_or_skip()
        buf = mr.allocate(_MANAGED_TEST_ALLOCATION_SIZE)
        stream = device.create_stream()

        prefetch(buf, Host(), stream=stream)
        stream.sync()
        discard_prefetch(buf, device, stream=stream)
        stream.sync()

        last = _get_int_mem_range_attr(
            buf,
            driver.CUmem_range_attribute.CU_MEM_RANGE_ATTRIBUTE_LAST_PREFETCH_LOCATION,
        )
        assert last == device.device_id
        buf.close()

    def test_batched_same_location(self, init_cuda):
        from cuda.core.utils import discard_prefetch, prefetch

        device = Device()
        skip_if_managed_memory_unsupported(device)
        if not hasattr(driver, "cuMemDiscardAndPrefetchBatchAsync"):
            pytest.skip("cuMemDiscardAndPrefetchBatchAsync unavailable")
        device.set_current()
        mr = create_managed_memory_resource_or_skip()
        bufs = [mr.allocate(_MANAGED_TEST_ALLOCATION_SIZE) for _ in range(2)]
        stream = device.create_stream()
        prefetch(bufs, Host(), stream=stream)
        stream.sync()
        discard_prefetch(bufs, device, stream=stream)
        stream.sync()
        for buf in bufs:
            last = _get_int_mem_range_attr(
                buf,
                driver.CUmem_range_attribute.CU_MEM_RANGE_ATTRIBUTE_LAST_PREFETCH_LOCATION,
            )
            assert last == device.device_id
            buf.close()

    def test_length_mismatch(self, init_cuda):
        from cuda.core.utils import discard_prefetch

        device = Device()
        skip_if_managed_memory_unsupported(device)
        device.set_current()
        mr = create_managed_memory_resource_or_skip()
        bufs = [mr.allocate(_MANAGED_TEST_ALLOCATION_SIZE) for _ in range(2)]
        stream = device.create_stream()
        with pytest.raises(ValueError, match="length"):
            discard_prefetch(bufs, [Host()], stream=stream)
        for buf in bufs:
            buf.close()

    def test_rejects_non_managed(self, init_cuda):
        from cuda.core.utils import discard_prefetch

        device = Device()
        device.set_current()
        buf = DummyDeviceMemoryResource(device).allocate(_MANAGED_TEST_ALLOCATION_SIZE)
        stream = device.create_stream()
        with pytest.raises(ValueError, match="managed-memory"):
            discard_prefetch(buf, Host(), stream=stream)
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
        from cuda.core.utils import advise

        device = Device()
        _skip_if_managed_location_ops_unsupported(device)
        device.set_current()
        mr = DummyUnifiedMemoryResource(device)
        bufs = [mr.allocate(_MANAGED_TEST_ALLOCATION_SIZE) for _ in range(2)]
        advise(bufs, "set_preferred_location", [Host(), device])
        for buf in bufs:
            buf.close()

    def test_options_must_be_options_dataclass_or_none(self, init_cuda):
        from cuda.core.utils import advise

        device = Device()
        _skip_if_managed_allocation_unsupported(device)
        device.set_current()
        buf = DummyUnifiedMemoryResource(device).allocate(_MANAGED_TEST_ALLOCATION_SIZE)
        with pytest.raises(TypeError, match="must be an? .*Options instance or None"):
            advise(buf, "set_read_mostly", options={})
        buf.close()


class TestManagedBuffer:
    """Property-style API on ManagedBuffer subclass."""

    def test_allocate_returns_managed_buffer(self, init_cuda):
        device = Device()
        skip_if_managed_memory_unsupported(device)
        device.set_current()
        mr = create_managed_memory_resource_or_skip()
        buf = mr.allocate(_MANAGED_TEST_ALLOCATION_SIZE)
        try:
            assert isinstance(buf, ManagedBuffer)
        finally:
            buf.close()

    def test_from_handle(self, init_cuda):
        from cuda.core import Buffer

        device = Device()
        _skip_if_managed_allocation_unsupported(device)
        device.set_current()
        # Allocate an external managed pointer through the dummy MR, then
        # adopt it as a ManagedBuffer via from_handle.
        plain = DummyUnifiedMemoryResource(device).allocate(_MANAGED_TEST_ALLOCATION_SIZE)
        try:
            mbuf = ManagedBuffer.from_handle(plain.handle, plain.size, owner=plain)
            assert isinstance(mbuf, ManagedBuffer)
            assert isinstance(mbuf, Buffer)
            assert mbuf.size == plain.size
        finally:
            plain.close()

    def test_read_mostly_roundtrip(self, init_cuda):
        # cuMemAdvise is exercised against an external managed allocation
        # (cuMemAllocManaged); pool-allocated managed memory may decline
        # certain advice on some driver/device combos.
        device = Device()
        _skip_if_managed_location_ops_unsupported(device)
        device.set_current()
        plain = DummyUnifiedMemoryResource(device).allocate(_MANAGED_TEST_ALLOCATION_SIZE)
        try:
            buf = ManagedBuffer.from_handle(plain.handle, plain.size, owner=plain)
            assert buf.read_mostly is False
            buf.read_mostly = True
            assert buf.read_mostly is True
            buf.read_mostly = False
            assert buf.read_mostly is False
        finally:
            plain.close()

    def test_preferred_location_roundtrip(self, init_cuda):
        device = Device()
        _skip_if_managed_location_ops_unsupported(device)
        device.set_current()
        plain = DummyUnifiedMemoryResource(device).allocate(_MANAGED_TEST_ALLOCATION_SIZE)
        try:
            buf = ManagedBuffer.from_handle(plain.handle, plain.size, owner=plain)
            buf.preferred_location = device
            got = buf.preferred_location
            assert isinstance(got, Device)
            assert got.device_id == device.device_id

            buf.preferred_location = Host()
            assert buf.preferred_location == Host()

            buf.preferred_location = None
            assert buf.preferred_location is None
        finally:
            plain.close()

    def test_accessed_by_add_discard(self, init_cuda):
        device = Device()
        _skip_if_managed_location_ops_unsupported(device)
        device.set_current()
        plain = DummyUnifiedMemoryResource(device).allocate(_MANAGED_TEST_ALLOCATION_SIZE)
        try:
            buf = ManagedBuffer.from_handle(plain.handle, plain.size, owner=plain)
            assert device not in buf.accessed_by

            buf.accessed_by.add(device)
            assert device in buf.accessed_by

            buf.accessed_by.discard(device)
            assert device not in buf.accessed_by
        finally:
            plain.close()

    def test_accessed_by_set_assignment(self, init_cuda):
        device = Device()
        _skip_if_managed_location_ops_unsupported(device)
        device.set_current()
        plain = DummyUnifiedMemoryResource(device).allocate(_MANAGED_TEST_ALLOCATION_SIZE)
        try:
            buf = ManagedBuffer.from_handle(plain.handle, plain.size, owner=plain)
            buf.accessed_by = {device}
            assert device in buf.accessed_by

            buf.accessed_by = set()
            assert device not in buf.accessed_by
        finally:
            plain.close()

    def test_instance_prefetch(self, init_cuda):
        device = Device()
        skip_if_managed_memory_unsupported(device)
        device.set_current()
        mr = create_managed_memory_resource_or_skip()
        buf = mr.allocate(_MANAGED_TEST_ALLOCATION_SIZE)
        stream = device.create_stream()
        try:
            buf.prefetch(device, stream=stream)
            stream.sync()
            last = _get_int_mem_range_attr(
                buf,
                driver.CUmem_range_attribute.CU_MEM_RANGE_ATTRIBUTE_LAST_PREFETCH_LOCATION,
            )
            assert last == device.device_id
        finally:
            buf.close()

    def test_instance_discard(self, init_cuda):
        device = Device()
        skip_if_managed_memory_unsupported(device)
        if not hasattr(driver, "cuMemDiscardBatchAsync"):
            pytest.skip("cuMemDiscardBatchAsync unavailable")
        device.set_current()
        mr = create_managed_memory_resource_or_skip()
        buf = mr.allocate(_MANAGED_TEST_ALLOCATION_SIZE)
        stream = device.create_stream()
        try:
            buf.prefetch(device, stream=stream)
            stream.sync()
            buf.discard(stream=stream)
            stream.sync()
        finally:
            buf.close()

    def test_instance_discard_prefetch(self, init_cuda):
        device = Device()
        _skip_if_managed_discard_prefetch_unsupported(device)
        device.set_current()
        mr = create_managed_memory_resource_or_skip()
        buf = mr.allocate(_MANAGED_TEST_ALLOCATION_SIZE)
        stream = device.create_stream()
        try:
            buf.prefetch(Host(), stream=stream)
            stream.sync()
            buf.discard_prefetch(device, stream=stream)
            stream.sync()
            last = _get_int_mem_range_attr(
                buf,
                driver.CUmem_range_attribute.CU_MEM_RANGE_ATTRIBUTE_LAST_PREFETCH_LOCATION,
            )
            assert last == device.device_id
        finally:
            buf.close()
