# cuda.core test suite

Package-wide conventions live in `../AGENTS.md`; repository-wide ones in
`../../AGENTS.md`. This file covers conventions specific to the tests.

## Never create an uncapped memory pool

A memory pool created without `max_size` reserves a virtual address window
sized from device memory (roughly 1x device memory) regardless of what the
test actually allocates. The reservation is charged to the process address
space even though it is not backed by physical memory, and it is not returned
until the pool is destroyed *and* the stream-ordered frees of its outstanding
allocations retire. The whole suite shares one process and one device, so
these reservations accumulate across tests.

When a test needs its own pool, use the suite-wide cap:

```python
POOL_SIZE = 2097152  # 2 MiB

mr = DeviceMemoryResource(dev, DeviceMemoryResourceOptions(max_size=POOL_SIZE))
```

Use a larger value only if a test genuinely requires it.

### Passing no options is different from passing empty options

`DeviceMemoryResource(dev)` with no options does **not** create a pool. It
wraps the device's existing default mempool (`_mempool_owned` is false) and
costs no additional address space. Passing *any* options object creates a new
owned pool, and a new pool without `max_size` is uncapped:

```python
DeviceMemoryResource(dev)                                     # wraps default pool, free
DeviceMemoryResource(dev, DeviceMemoryResourceOptions())      # NEW uncapped pool, expensive
DeviceMemoryResource(dev, {"ipc_enabled": True})              # NEW uncapped pool, expensive
```

Do not add `max_size` to a call that currently passes no options: that
converts a free default-pool wrapper into a new pool and makes things worse.

### Managed pools are exempt

`cuMemPoolCreate` requires `CUmemPoolProps.maxSize` to be zero for managed
pools, so `ManagedMemoryResourceOptions` has no `max_size` option. Managed
pools cannot be right-sized and are not checked.

### Enforcement

`test_mempool_hygiene.py` statically scans this directory and fails on
`DeviceMemoryResourceOptions` / `PinnedMemoryResourceOptions` constructions
that omit `max_size`. When a call is deliberately exempt -- most often because
it sits inside `pytest.raises` and no pool is ever created -- annotate it:

```python
with pytest.raises(RuntimeError, match="IPC is not available"):
    # uncapped-pool-ok: raises before the pool is created
    DeviceMemoryResource(mempool_device, DeviceMemoryResourceOptions(ipc_enabled=True))
```

## Release resources at test boundaries

The `_init_cuda_context` fixture in `conftest.py` runs `gc.collect()` followed
by `cuCtxSynchronize()` before popping the context. Tests should not rely on
that as a substitute for cleaning up explicitly: prefer context managers for
resources whose lifetime fits a single scope, and keep pool lifetimes inside
the test that creates them.
