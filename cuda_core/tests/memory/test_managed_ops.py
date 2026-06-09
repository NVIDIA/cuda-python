# SPDX-FileCopyrightText: Copyright (c) 2024-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import pytest
from helpers.buffers import DummyDeviceMemoryResource, DummyUnifiedMemoryResource

from conftest import create_managed_memory_resource_or_skip
from cuda.bindings import driver
from cuda.core import Device, Host, ManagedBuffer
from cuda.core._memory._managed_buffer import _get_int_attr

_MANAGED_TEST_ALLOCATION_SIZE = 4096
_READ_MOSTLY_ENABLED = 1
_HOST_LOCATION_ID = -1
_INVALID_HOST_DEVICE_ORDINAL = 0


# TODO(#2109): replace with ``buf.last_prefetch_location`` once
# ``ManagedBuffer`` exposes mem-range attributes directly.
def _last_prefetch_location(buf):
    return _get_int_attr(buf, driver.CUmem_range_attribute.CU_MEM_RANGE_ATTRIBUTE_LAST_PREFETCH_LOCATION)


def _skip_if_raw_managed_alloc_unsupported(device):
    # Raw `cuMemAllocManaged` capability — distinct from conftest's
    # `skip_if_managed_memory_unsupported`, which gates `ManagedMemoryResource`
    # pool creation. Used by tests that exercise `DummyUnifiedMemoryResource`.
    try:
        if not device.properties.managed_memory:
            pytest.skip("Device does not support managed memory operations")
    except AttributeError:
        pytest.skip("Managed-memory buffer operations require CUDA support")


def _skip_if_managed_location_ops_unsupported(device):
    _skip_if_raw_managed_alloc_unsupported(device)
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


# Fixture set:
#   * location_ops_device / location_ops_mr — concurrent_managed_access tier;
#     covers advise/prefetch (every test in this file needs it).
#   * discard_prefetch_device — adds cuMemDiscardAndPrefetchBatchAsync +
#     multi-GPU concurrent-managed-access check.
#   * managed_buffer — parametrized over pool-allocated (ManagedMemoryResource)
#     vs external (DummyUnifiedMemoryResource + from_handle); used by
#     TestManagedBuffer so each method runs against both buffer sources.


@pytest.fixture
def location_ops_device(init_cuda):
    device = Device()
    _skip_if_managed_location_ops_unsupported(device)
    device.set_current()
    return device


@pytest.fixture
def location_ops_mr(location_ops_device):
    return create_managed_memory_resource_or_skip()


@pytest.fixture
def discard_prefetch_device(init_cuda):
    device = Device()
    _skip_if_managed_discard_prefetch_unsupported(device)
    device.set_current()
    return device


@pytest.fixture(params=["pool", "external"], ids=["pool", "external"])
def managed_buffer(request, location_ops_device, location_ops_mr):
    # Use for prefetch / discard / discard_prefetch tests — both sources
    # work uniformly. Do NOT use for cuMemAdvise (see external_managed_buffer).
    size = _MANAGED_TEST_ALLOCATION_SIZE
    if request.param == "pool":
        buf = location_ops_mr.allocate(size, stream=location_ops_device.default_stream)
        yield buf
        buf.close()
    else:
        plain = DummyUnifiedMemoryResource(location_ops_device).allocate(size)
        buf = ManagedBuffer.from_handle(plain.handle, plain.size, owner=plain)
        yield buf
        plain.close()


@pytest.fixture
def external_managed_buffer(location_ops_device):
    # Pool-allocated managed memory declines certain cuMemAdvise values
    # (CUDA_ERROR_NOT_SUPPORTED) on some driver/device combos, so all
    # advise tests exercise the external (cuMemAllocManaged + from_handle)
    # path. Prefetch / discard / discard_prefetch are unaffected — those
    # use the parametrized `managed_buffer` fixture above.
    plain = DummyUnifiedMemoryResource(location_ops_device).allocate(_MANAGED_TEST_ALLOCATION_SIZE)
    buf = ManagedBuffer.from_handle(plain.handle, plain.size, owner=plain)
    yield buf
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
        from cuda.core._utils.version import binding_version

        if binding_version() < (13, 0, 0):
            pytest.skip("Host(numa_id=N) requires CUDA 13 bindings")
        spec = _coerce_location(Host(numa_id=3))
        assert spec.kind == "host_numa"
        assert spec.id == 3

    def test_host_numa_current_passthrough(self):
        from cuda.core._memory._managed_location import _coerce_location
        from cuda.core._utils.version import binding_version

        if binding_version() < (13, 0, 0):
            pytest.skip("Host.numa_current() requires CUDA 13 bindings")
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

    def test_same_location(self, location_ops_device, location_ops_mr):
        from cuda.core.utils import prefetch_batch

        device = location_ops_device
        bufs = [location_ops_mr.allocate(_MANAGED_TEST_ALLOCATION_SIZE, stream=device.default_stream) for _ in range(3)]
        stream = device.create_stream()

        prefetch_batch(stream, bufs, device)
        stream.sync()

        for buf in bufs:
            last = _last_prefetch_location(buf)
            assert last == device.device_id
            buf.close()

    def test_per_buffer_location(self, location_ops_device, location_ops_mr):
        from cuda.core.utils import prefetch_batch

        device = location_ops_device
        bufs = [location_ops_mr.allocate(_MANAGED_TEST_ALLOCATION_SIZE, stream=device.default_stream) for _ in range(2)]
        stream = device.create_stream()

        prefetch_batch(stream, bufs, [Host(), device])
        stream.sync()

        last0 = _last_prefetch_location(bufs[0])
        last1 = _last_prefetch_location(bufs[1])
        assert last0 == _HOST_LOCATION_ID
        assert last1 == device.device_id
        for buf in bufs:
            buf.close()


class TestDiscardBatch:
    """Tests for utils.discard_batch (batched-only free function)."""

    def test_basic(self, location_ops_device, location_ops_mr):
        from cuda.core.utils import discard_batch, prefetch_batch

        if not hasattr(driver, "cuMemDiscardBatchAsync"):
            pytest.skip("cuMemDiscardBatchAsync unavailable")
        device = location_ops_device
        bufs = [location_ops_mr.allocate(_MANAGED_TEST_ALLOCATION_SIZE, stream=device.default_stream) for _ in range(3)]
        stream = device.create_stream()
        prefetch_batch(stream, bufs, device)
        stream.sync()
        discard_batch(stream, bufs)
        stream.sync()
        for buf in bufs:
            buf.close()


class TestDiscardPrefetchBatch:
    """Tests for utils.discard_prefetch_batch (batched-only free function)."""

    def test_same_location(self, location_ops_device, location_ops_mr):
        from cuda.core.utils import discard_prefetch_batch, prefetch_batch

        if not hasattr(driver, "cuMemDiscardAndPrefetchBatchAsync"):
            pytest.skip("cuMemDiscardAndPrefetchBatchAsync unavailable")
        device = location_ops_device
        bufs = [location_ops_mr.allocate(_MANAGED_TEST_ALLOCATION_SIZE, stream=device.default_stream) for _ in range(2)]
        stream = device.create_stream()
        prefetch_batch(stream, bufs, Host())
        stream.sync()
        discard_prefetch_batch(stream, bufs, device)
        stream.sync()
        for buf in bufs:
            last = _last_prefetch_location(buf)
            assert last == device.device_id
            buf.close()


# Module-level parametrized rejection tests — formerly per-class duplicates.


@pytest.mark.parametrize(
    "fn_name,needs_loc",
    [
        ("prefetch_batch", True),
        ("discard_batch", False),
        ("discard_prefetch_batch", True),
    ],
)
def test_batch_rejects_single_buffer(location_ops_device, location_ops_mr, fn_name, needs_loc):
    from cuda.core import utils

    fn = getattr(utils, fn_name)
    buf = location_ops_mr.allocate(_MANAGED_TEST_ALLOCATION_SIZE, stream=location_ops_device.default_stream)
    stream = location_ops_device.create_stream()
    args = (stream, buf, Host()) if needs_loc else (stream, buf)
    with pytest.raises(TypeError, match="sequence of Buffers"):
        fn(*args)
    buf.close()


# discard_batch takes no location sequence; only the prefetch variants validate length.
@pytest.mark.parametrize("fn_name", ["prefetch_batch", "discard_prefetch_batch"])
def test_batch_length_mismatch(location_ops_device, location_ops_mr, fn_name):
    from cuda.core import utils

    fn = getattr(utils, fn_name)
    bufs = [
        location_ops_mr.allocate(_MANAGED_TEST_ALLOCATION_SIZE, stream=location_ops_device.default_stream)
        for _ in range(2)
    ]
    stream = location_ops_device.create_stream()
    with pytest.raises(ValueError, match="length"):
        fn(stream, bufs, [Host()])
    for b in bufs:
        b.close()


class TestManagedBuffer:
    """Property-style API on ManagedBuffer subclass.

    Most tests consume the ``managed_buffer`` fixture and therefore run
    twice — once against a pool-allocated buffer
    (``ManagedMemoryResource``) and once against an external one
    (``DummyUnifiedMemoryResource`` + ``ManagedBuffer.from_handle``).
    """

    def test_allocate_returns_managed_buffer(self, location_ops_device, location_ops_mr):
        # Pool-only: asserts that ManagedMemoryResource.allocate returns
        # a ManagedBuffer subclass. Doesn't apply to from_handle (see
        # test_from_handle below).
        buf = location_ops_mr.allocate(_MANAGED_TEST_ALLOCATION_SIZE, stream=location_ops_device.default_stream)
        try:
            assert isinstance(buf, ManagedBuffer)
        finally:
            buf.close()

    def test_from_handle(self, init_cuda):
        from cuda.core import Buffer

        device = Device()
        _skip_if_raw_managed_alloc_unsupported(device)
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

    def test_read_mostly_roundtrip(self, external_managed_buffer):
        buf = external_managed_buffer
        assert buf.read_mostly is False
        buf.read_mostly = True
        assert buf.read_mostly is True
        buf.read_mostly = False
        assert buf.read_mostly is False

    def test_preferred_location_roundtrip(self, location_ops_device, external_managed_buffer):
        device = location_ops_device
        buf = external_managed_buffer
        buf.preferred_location = device
        got = buf.preferred_location
        assert isinstance(got, Device)
        assert got.device_id == device.device_id

        buf.preferred_location = Host()
        assert buf.preferred_location == Host()

        buf.preferred_location = None
        assert buf.preferred_location is None

    def test_preferred_location_roundtrip_host_numa(self, location_ops_device):
        """Host(numa_id=N) round-trips correctly on CUDA 13 builds."""
        from cuda.core._utils.version import binding_version

        if binding_version() < (13, 0, 0):
            pytest.skip("Host(numa_id=N) round-trip requires CUDA 13 bindings")
        plain = DummyUnifiedMemoryResource(location_ops_device).allocate(_MANAGED_TEST_ALLOCATION_SIZE)
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

    def test_accessed_by_add_discard(self, location_ops_device, external_managed_buffer):
        device = location_ops_device
        buf = external_managed_buffer
        assert device not in buf.accessed_by

        buf.accessed_by.add(device)
        assert device in buf.accessed_by

        buf.accessed_by.discard(device)
        assert device not in buf.accessed_by

    def test_accessed_by_mutable_set_interface(self, location_ops_device, external_managed_buffer):
        """Full MutableSet conformance pass on AccessedBySetProxy.

        Uses the shared helper introduced by NVIDIA/cuda-python#2018
        for peer-access. Both proxies share the same single-member
        contract (one valid insertable element on a single-GPU box),
        so the helper applies directly.
        """
        from helpers.collection_interface_testers import assert_single_member_mutable_set_interface

        # Host(numa_id=0) is rejected by set_accessed_by (host_numa
        # kind is not in DEVICE_HOST_ONLY), so it is guaranteed
        # never to enter the proxy — perfect non-member sentinel.
        assert_single_member_mutable_set_interface(
            external_managed_buffer.accessed_by,
            member=location_ops_device,
            non_member=Host(numa_id=0),
        )

    def test_accessed_by_set_assignment(self, location_ops_device, external_managed_buffer):
        device = location_ops_device
        buf = external_managed_buffer
        buf.accessed_by = {device}
        assert device in buf.accessed_by

        buf.accessed_by = set()
        assert device not in buf.accessed_by

    def test_instance_prefetch(self, location_ops_device, managed_buffer):
        device = location_ops_device
        buf = managed_buffer
        stream = device.create_stream()
        buf.prefetch(device, stream=stream)
        stream.sync()
        assert _last_prefetch_location(buf) == device.device_id

    def test_instance_discard(self, location_ops_device, managed_buffer):
        if not hasattr(driver, "cuMemDiscardBatchAsync"):
            pytest.skip("cuMemDiscardBatchAsync unavailable")
        device = location_ops_device
        buf = managed_buffer
        stream = device.create_stream()
        buf.prefetch(device, stream=stream)
        stream.sync()
        buf.discard(stream=stream)
        stream.sync()

    def test_instance_discard_prefetch(self, discard_prefetch_device):
        device = discard_prefetch_device
        mr = create_managed_memory_resource_or_skip()
        buf = mr.allocate(_MANAGED_TEST_ALLOCATION_SIZE, stream=device.default_stream)
        try:
            stream = device.create_stream()
            buf.prefetch(Host(), stream=stream)
            stream.sync()
            buf.discard_prefetch(device, stream=stream)
            stream.sync()
            assert _last_prefetch_location(buf) == device.device_id
        finally:
            buf.close()

    def test_operation_validation(self, managed_buffer):
        """Error paths: prefetch(None) and accessed_by host_numa rejection."""
        buf = managed_buffer
        stream = Device().create_stream()

        with pytest.raises(ValueError, match="location is required"):
            buf.prefetch(None, stream=stream)

        # CUDA 13: kind-allowed check fires (ValueError). CUDA 12: NUMA-host is
        # rejected at the boundary first (TypeError).
        with pytest.raises(
            (ValueError, TypeError),
            match=r"does not support location_type='host_numa'|cuda-bindings 13\.0\+",
        ):
            buf.accessed_by.add(Host(numa_id=_INVALID_HOST_DEVICE_ORDINAL))

    def test_advise_location_validation(self, location_ops_device, external_managed_buffer):
        """Doc-specified location constraints for each advice kind."""
        device = location_ops_device
        buf = external_managed_buffer

        # read_mostly works without a location
        buf.read_mostly = True

        # preferred_location accepts Device and Host
        buf.preferred_location = device
        buf.preferred_location = Host()

        # accessed_by rejects host_numa (CUDA 13: kind check; CUDA 12: boundary)
        with pytest.raises(
            (ValueError, TypeError),
            match=r"does not support location_type='host_numa'|cuda-bindings 13\.0\+",
        ):
            buf.accessed_by.add(Host(numa_id=0))

        # accessed_by rejects host_numa_current (same reasoning)
        with pytest.raises(
            (ValueError, TypeError),
            match=r"does not support location_type='host_numa_current'|cuda-bindings 13\.0\+",
        ):
            buf.accessed_by.add(Host.numa_current())
