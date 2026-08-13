## Description

closes #2615

Pool-backed memory resources (`DeviceMemoryResource`, `PinnedMemoryResource`, `ManagedMemoryResource`) and `GraphMemoryResource` previously returned buffers whose C++ deleter called `cuMemFreeAsync` directly, bypassing Python `deallocate()` overrides.

This change routes teardown for **subclasses** of pool-backed and graph memory resources through MR-owned device pointer handles (the same path as `Buffer.from_handle(mr=...)`), recording the allocation stream at creation and invoking `MemoryResource.deallocate()`.

Built-in types (`DeviceMemoryResource`, `PinnedMemoryResource`, `ManagedMemoryResource`, `GraphMemoryResource`) keep the existing direct C++ deleter so stream-ordered frees still work without the GIL during interpreter shutdown.

## Checklist
- [x] New or existing tests cover these changes.
- [ ] The documentation is up to date with these changes.

## Test plan

- [ ] `pytest cuda_core/tests/test_memory.py -k "pool_backed_mr or dmr_deallocate_frees_pool_pointer or dmr_from_handle_deallocate"`
- [ ] CI source builds and GPU tests for `cuda.core`
