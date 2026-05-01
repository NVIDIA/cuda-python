# SPDX-FileCopyrightText: Copyright (c) 2024-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import pytest
from helpers.buffers import DummyDeviceMemoryResource, DummyUnifiedMemoryResource

from conftest import (
    create_managed_memory_resource_or_skip,
    skip_if_managed_memory_unsupported,
)
from cuda.core import Device, Host, ManagedBuffer
from cuda.core._memory._managed_buffer import _get_int_attr

try:
    from cuda.bindings import driver
except ImportError:
    from cuda import cuda as driver


_MANAGED_TEST_ALLOCATION_SIZE = 4096
_READ_MOSTLY_ENABLED = 1
_HOST_LOCATION_ID = -1
_INVALID_HOST_DEVICE_ORDINAL = 0


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

    buffer.prefetch(Host(), stream=stream)
    stream.sync()
    last_location = _get_int_attr(
        buffer,
        driver.CUmem_range_attribute.CU_MEM_RANGE_ATTRIBUTE_LAST_PREFETCH_LOCATION,
    )
    assert last_location == _HOST_LOCATION_ID

    buffer.prefetch(device, stream=stream)
    stream.sync()
    last_location = _get_int_attr(
        buffer,
        driver.CUmem_range_attribute.CU_MEM_RANGE_ATTRIBUTE_LAST_PREFETCH_LOCATION,
    )
    assert last_location == device.device_id

    buffer.close()


def test_managed_memory_advise_supports_external_managed_allocations(init_cuda):
    device = Device()
    _skip_if_managed_location_ops_unsupported(device)
    device.set_current()

    plain = DummyUnifiedMemoryResource(device).allocate(_MANAGED_TEST_ALLOCATION_SIZE)
    buffer = ManagedBuffer.from_handle(plain.handle, plain.size, owner=plain)

    buffer.read_mostly = True
    assert (
        _get_int_attr(
            buffer,
            driver.CUmem_range_attribute.CU_MEM_RANGE_ATTRIBUTE_READ_MOSTLY,
        )
        == _READ_MOSTLY_ENABLED
    )

    buffer.preferred_location = Host()
    preferred_location = _get_int_attr(
        buffer,
        driver.CUmem_range_attribute.CU_MEM_RANGE_ATTRIBUTE_PREFERRED_LOCATION,
    )
    assert preferred_location == _HOST_LOCATION_ID

    plain.close()


def test_managed_memory_prefetch_supports_external_managed_allocations(init_cuda):
    device = Device()
    _skip_if_managed_location_ops_unsupported(device)
    device.set_current()

    plain = DummyUnifiedMemoryResource(device).allocate(_MANAGED_TEST_ALLOCATION_SIZE)
    buffer = ManagedBuffer.from_handle(plain.handle, plain.size, owner=plain)
    stream = device.create_stream()

    buffer.prefetch(device, stream=stream)
    stream.sync()

    last_location = _get_int_attr(
        buffer,
        driver.CUmem_range_attribute.CU_MEM_RANGE_ATTRIBUTE_LAST_PREFETCH_LOCATION,
    )
    assert last_location == device.device_id

    plain.close()


def test_managed_memory_discard_prefetch_supports_managed_pool_allocations(init_cuda):
    device = Device()
    skip_if_managed_memory_unsupported(device)
    _skip_if_managed_discard_prefetch_unsupported(device)
    device.set_current()

    mr = create_managed_memory_resource_or_skip()
    buffer = mr.allocate(_MANAGED_TEST_ALLOCATION_SIZE)
    stream = device.create_stream()

    buffer.prefetch(Host(), stream=stream)
    stream.sync()

    buffer.discard_prefetch(device, stream=stream)
    stream.sync()

    last_location = _get_int_attr(
        buffer,
        driver.CUmem_range_attribute.CU_MEM_RANGE_ATTRIBUTE_LAST_PREFETCH_LOCATION,
    )
    assert last_location == device.device_id

    buffer.close()


def test_managed_memory_discard_prefetch_supports_external_managed_allocations(init_cuda):
    device = Device()
    _skip_if_managed_discard_prefetch_unsupported(device)
    device.set_current()

    plain = DummyUnifiedMemoryResource(device).allocate(_MANAGED_TEST_ALLOCATION_SIZE)
    buffer = ManagedBuffer.from_handle(plain.handle, plain.size, owner=plain)
    stream = device.create_stream()

    buffer.prefetch(Host(), stream=stream)
    stream.sync()

    buffer.discard_prefetch(device, stream=stream)
    stream.sync()

    last_location = _get_int_attr(
        buffer,
        driver.CUmem_range_attribute.CU_MEM_RANGE_ATTRIBUTE_LAST_PREFETCH_LOCATION,
    )
    assert last_location == device.device_id

    plain.close()


def test_managed_memory_operations_reject_non_managed_allocations(init_cuda):
    """Wrapping a non-managed pointer in ManagedBuffer raises at op time."""
    device = Device()
    device.set_current()

    plain = DummyDeviceMemoryResource(device).allocate(_MANAGED_TEST_ALLOCATION_SIZE)
    # Wrapping a device-only pointer as ManagedBuffer is allowed at construction
    # (no driver query yet); the runtime managed-ness check fires at op time.
    buffer = ManagedBuffer.from_handle(plain.handle, plain.size, owner=plain)
    stream = device.create_stream()

    with pytest.raises(ValueError, match="managed-memory allocation"):
        buffer.read_mostly = True
    with pytest.raises(ValueError, match="managed-memory allocation"):
        buffer.prefetch(device, stream=stream)
    with pytest.raises(ValueError, match="managed-memory allocation"):
        buffer.discard_prefetch(device, stream=stream)

    plain.close()


def test_managed_memory_operation_validation(init_cuda):
    device = Device()
    skip_if_managed_memory_unsupported(device)
    device.set_current()

    mr = create_managed_memory_resource_or_skip()
    buffer = mr.allocate(_MANAGED_TEST_ALLOCATION_SIZE)
    stream = device.create_stream()

    with pytest.raises(ValueError, match="location is required"):
        buffer.prefetch(None, stream=stream)

    with pytest.raises(ValueError, match="does not support location_type='host_numa'"):
        buffer.accessed_by.add(Host(numa_id=_INVALID_HOST_DEVICE_ORDINAL))

    buffer.close()


def test_managed_memory_advise_location_validation(init_cuda):
    """Verify doc-specified location constraints for each advice kind."""
    device = Device()
    _skip_if_managed_location_ops_unsupported(device)
    device.set_current()

    plain = DummyUnifiedMemoryResource(device).allocate(_MANAGED_TEST_ALLOCATION_SIZE)
    buffer = ManagedBuffer.from_handle(plain.handle, plain.size, owner=plain)

    # read_mostly works without a location
    buffer.read_mostly = True

    # preferred_location accepts Device
    buffer.preferred_location = device

    # preferred_location accepts Host()
    buffer.preferred_location = Host()

    # accessed_by rejects host_numa
    with pytest.raises(ValueError, match="does not support location_type='host_numa'"):
        buffer.accessed_by.add(Host(numa_id=0))

    # accessed_by rejects host_numa_current
    with pytest.raises(ValueError, match="does not support location_type='host_numa_current'"):
        buffer.accessed_by.add(Host.numa_current())

    # Both Host and Device are accepted
    buffer.preferred_location = Host()
    buffer.preferred_location = Device(0)

    plain.close()


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

    def test_numa_current_only_via_classmethod(self):
        # is_numa_current is internal state, only settable via Host.numa_current()
        with pytest.raises(TypeError):
            Host(is_numa_current=True)  # type: ignore[call-arg]

    def test_immutable(self):
        h = Host(numa_id=2)
        with pytest.raises(AttributeError):
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

    def test_none_when_disallowed(self):
        from cuda.core._memory._managed_location import _coerce_location

        with pytest.raises(ValueError, match="location is required"):
            _coerce_location(None, allow_none=False)

    def test_none_when_allowed(self):
        from cuda.core._memory._managed_location import _coerce_location

        assert _coerce_location(None, allow_none=True) is None

    def test_int_rejected(self):
        from cuda.core._memory._managed_location import _coerce_location

        # int shorthand was removed in favor of explicit Device/Host
        with pytest.raises(TypeError, match="Device, Host, or None"):
            _coerce_location(0)

    def test_bad_type(self):
        from cuda.core._memory._managed_location import _coerce_location

        with pytest.raises(TypeError, match="Device, Host, or None"):
            _coerce_location("device")


class TestPrefetchBatch:
    """Tests for utils.prefetch_batch (batched-only free function)."""

    def test_same_location(self, init_cuda):
        from cuda.core.utils import prefetch_batch

        device = Device()
        skip_if_managed_memory_unsupported(device)
        device.set_current()
        mr = create_managed_memory_resource_or_skip()
        bufs = [mr.allocate(_MANAGED_TEST_ALLOCATION_SIZE) for _ in range(3)]
        stream = device.create_stream()

        prefetch_batch(bufs, device, stream=stream)
        stream.sync()

        for buf in bufs:
            last = _get_int_attr(
                buf,
                driver.CUmem_range_attribute.CU_MEM_RANGE_ATTRIBUTE_LAST_PREFETCH_LOCATION,
            )
            assert last == device.device_id
            buf.close()

    def test_per_buffer_location(self, init_cuda):
        from cuda.core.utils import prefetch_batch

        device = Device()
        skip_if_managed_memory_unsupported(device)
        device.set_current()
        mr = create_managed_memory_resource_or_skip()
        bufs = [mr.allocate(_MANAGED_TEST_ALLOCATION_SIZE) for _ in range(2)]
        stream = device.create_stream()

        prefetch_batch(bufs, [Host(), device], stream=stream)
        stream.sync()

        last0 = _get_int_attr(
            bufs[0],
            driver.CUmem_range_attribute.CU_MEM_RANGE_ATTRIBUTE_LAST_PREFETCH_LOCATION,
        )
        last1 = _get_int_attr(
            bufs[1],
            driver.CUmem_range_attribute.CU_MEM_RANGE_ATTRIBUTE_LAST_PREFETCH_LOCATION,
        )
        assert last0 == _HOST_LOCATION_ID
        assert last1 == device.device_id
        for buf in bufs:
            buf.close()

    def test_length_mismatch(self, init_cuda):
        from cuda.core.utils import prefetch_batch

        device = Device()
        skip_if_managed_memory_unsupported(device)
        device.set_current()
        mr = create_managed_memory_resource_or_skip()
        bufs = [mr.allocate(_MANAGED_TEST_ALLOCATION_SIZE) for _ in range(2)]
        stream = device.create_stream()

        with pytest.raises(ValueError, match="length"):
            prefetch_batch(bufs, [Host()], stream=stream)
        for buf in bufs:
            buf.close()

    def test_rejects_single_buffer(self, init_cuda):
        from cuda.core.utils import prefetch_batch

        device = Device()
        skip_if_managed_memory_unsupported(device)
        device.set_current()
        mr = create_managed_memory_resource_or_skip()
        buf = mr.allocate(_MANAGED_TEST_ALLOCATION_SIZE)
        stream = device.create_stream()
        with pytest.raises(TypeError, match="sequence of Buffers"):
            prefetch_batch(buf, Host(), stream=stream)
        buf.close()


class TestDiscardBatch:
    """Tests for utils.discard_batch (batched-only free function)."""

    def test_basic(self, init_cuda):
        from cuda.core.utils import discard_batch, prefetch_batch

        device = Device()
        skip_if_managed_memory_unsupported(device)
        if not hasattr(driver, "cuMemDiscardBatchAsync"):
            pytest.skip("cuMemDiscardBatchAsync unavailable")
        device.set_current()
        mr = create_managed_memory_resource_or_skip()
        bufs = [mr.allocate(_MANAGED_TEST_ALLOCATION_SIZE) for _ in range(3)]
        stream = device.create_stream()
        prefetch_batch(bufs, device, stream=stream)
        stream.sync()
        discard_batch(bufs, stream=stream)
        stream.sync()
        for buf in bufs:
            buf.close()

    def test_rejects_single_buffer(self, init_cuda):
        from cuda.core.utils import discard_batch

        device = Device()
        skip_if_managed_memory_unsupported(device)
        device.set_current()
        mr = create_managed_memory_resource_or_skip()
        buf = mr.allocate(_MANAGED_TEST_ALLOCATION_SIZE)
        stream = device.create_stream()
        with pytest.raises(TypeError, match="sequence of Buffers"):
            discard_batch(buf, stream=stream)
        buf.close()


class TestDiscardPrefetchBatch:
    """Tests for utils.discard_prefetch_batch (batched-only free function)."""

    def test_same_location(self, init_cuda):
        from cuda.core.utils import discard_prefetch_batch, prefetch_batch

        device = Device()
        skip_if_managed_memory_unsupported(device)
        if not hasattr(driver, "cuMemDiscardAndPrefetchBatchAsync"):
            pytest.skip("cuMemDiscardAndPrefetchBatchAsync unavailable")
        device.set_current()
        mr = create_managed_memory_resource_or_skip()
        bufs = [mr.allocate(_MANAGED_TEST_ALLOCATION_SIZE) for _ in range(2)]
        stream = device.create_stream()
        prefetch_batch(bufs, Host(), stream=stream)
        stream.sync()
        discard_prefetch_batch(bufs, device, stream=stream)
        stream.sync()
        for buf in bufs:
            last = _get_int_attr(
                buf,
                driver.CUmem_range_attribute.CU_MEM_RANGE_ATTRIBUTE_LAST_PREFETCH_LOCATION,
            )
            assert last == device.device_id
            buf.close()

    def test_length_mismatch(self, init_cuda):
        from cuda.core.utils import discard_prefetch_batch

        device = Device()
        skip_if_managed_memory_unsupported(device)
        device.set_current()
        mr = create_managed_memory_resource_or_skip()
        bufs = [mr.allocate(_MANAGED_TEST_ALLOCATION_SIZE) for _ in range(2)]
        stream = device.create_stream()
        with pytest.raises(ValueError, match="length"):
            discard_prefetch_batch(bufs, [Host()], stream=stream)
        for buf in bufs:
            buf.close()

    def test_rejects_single_buffer(self, init_cuda):
        from cuda.core.utils import discard_prefetch_batch

        device = Device()
        skip_if_managed_memory_unsupported(device)
        device.set_current()
        mr = create_managed_memory_resource_or_skip()
        buf = mr.allocate(_MANAGED_TEST_ALLOCATION_SIZE)
        stream = device.create_stream()
        with pytest.raises(TypeError, match="sequence of Buffers"):
            discard_prefetch_batch(buf, Host(), stream=stream)
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

    def test_preferred_location_roundtrip_host_numa(self, init_cuda):
        """Host(numa_id=N) round-trips correctly on CUDA 13 builds."""
        from cuda.core._utils.version import binding_version

        if binding_version() < (13, 0, 0):
            pytest.skip("Host(numa_id=N) round-trip requires CUDA 13 bindings")
        device = Device()
        _skip_if_managed_location_ops_unsupported(device)
        device.set_current()
        plain = DummyUnifiedMemoryResource(device).allocate(_MANAGED_TEST_ALLOCATION_SIZE)
        try:
            buf = ManagedBuffer.from_handle(plain.handle, plain.size, owner=plain)
            # An explicit NUMA id round-trips via the cu13 v2 attribute pair.
            # NUMA node 0 exists on every multi-NUMA system; on single-NUMA
            # systems the driver may collapse to HOST or reject — skip then.
            buf.preferred_location = Host(numa_id=0)
            got = buf.preferred_location
            if got is None or not (isinstance(got, Host) and got.numa_id == 0):
                pytest.skip("host_numa preferred_location not supported by this driver / hardware")
            assert got == Host(numa_id=0)
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

    def test_accessed_by_read_methods(self, init_cuda):
        """Cover __iter__, __len__, __eq__, __repr__ on AccessedBySet."""
        device = Device()
        _skip_if_managed_location_ops_unsupported(device)
        device.set_current()
        plain = DummyUnifiedMemoryResource(device).allocate(_MANAGED_TEST_ALLOCATION_SIZE)
        try:
            buf = ManagedBuffer.from_handle(plain.handle, plain.size, owner=plain)

            # Empty initially
            assert len(buf.accessed_by) == 0
            assert list(buf.accessed_by) == []
            assert buf.accessed_by == set()
            assert "AccessedBySet" in repr(buf.accessed_by)

            # After add
            buf.accessed_by.add(device)
            assert len(buf.accessed_by) == 1
            assert list(buf.accessed_by) == [device]
            assert buf.accessed_by == {device}
            assert buf.accessed_by != frozenset()

            # __eq__ vs another AccessedBySet on the same buffer
            assert buf.accessed_by == buf.accessed_by
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
            last = _get_int_attr(
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
            last = _get_int_attr(
                buf,
                driver.CUmem_range_attribute.CU_MEM_RANGE_ATTRIBUTE_LAST_PREFETCH_LOCATION,
            )
            assert last == device.device_id
        finally:
            buf.close()
