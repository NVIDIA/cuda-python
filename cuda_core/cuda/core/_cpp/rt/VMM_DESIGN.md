# VirtualMemoryResource on the handle layer

This document describes how `VirtualMemoryResource` uses the `_rt` handle layer that the
pool-backed resources also use. It complements [DESIGN.md](DESIGN.md), which describes the layer
itself.

## Summary

Each physical allocation, address reservation, and mapping has its own `std::shared_ptr` handle
with a deleter that knows the exact driver call to undo it. A buffer owns a list of mappings.
Teardown order follows from what holds what, and a failed multi-step operation unwinds by letting
its local handles die. The module is written in Cython, because the handles are `cdef` types.

## Driver behavior the design relies on

- **Reservation.** `cuMemAddressReserve(size, align, hint)` returns `(ptr, size)`.
  [`cuMemAddressFree`](https://docs.nvidia.com/cuda/cuda-driver-api/group__CUDA__VA.html)
  succeeds only for the exact `(ptr, size)` pair of one reservation. Reservations never overlap;
  growing a buffer in place yields two adjacent reservations, each of which is freed separately.
  A hint must be a multiple of `max(align, 2 MiB)`; `align == 0` means the default 2 MiB.
- **Physical allocation.** `cuMemCreate` returns a handle with one reference; `cuMemRelease` drops
  one; `cuMemRetainAllocationHandle` adds one. Mappings are counted separately. The memory is
  freed when references are zero and no mapping remains. Releasing while mapped is legal.
- **Mapping.** `cuMemMap(ptr, size, 0, handle)` maps the whole allocation at `ptr`; offset must be
  0 and size must equal the allocation's size. `cuMemSetAccess(ptr, size, descs, count)` grants
  access per mapped range and rejects `count == 0`. `cuMemUnmap(ptr, size)` may cover several
  whole mappings, never part of one. One allocation may be mapped at several addresses at once;
  each mapping has its own access state.
- **Coherence of aliases.** Two addresses that map one allocation reach the same physical pages.
  Writes through one are visible through the other in stream order and at kernel boundaries; the
  driver adds no synchronization of its own.
- **Context and synchronization.** No VMM entry point needs a current context. `cuMemUnmap` does
  not synchronize. `cuStreamSynchronize` is rejected on a capturing stream. It is also one of the
  calls the driver treats as unsafe while any capture is active: in the calling thread's default
  (global) capture mode it invalidates every global-mode capture in the process, and any
  non-relaxed capture the thread began, before it returns an error. Switching the thread to
  relaxed mode with `cuThreadExchangeStreamCaptureMode` disables that interaction; the stream's
  own capture state is unaffected by the mode. The VMM entry points do not have this
  interaction.

So a mapping depends on exactly one reservation and one allocation, mappings never depend on
other mappings, and reservations and allocations are independent of each other. A buffer is a
list of mappings.

## Handles

Three new `std::shared_ptr` aliases into boxes, following the conventions in `types.hpp`.
`CUmemGenericAllocationHandle` and `CUdeviceptr` are both `unsigned long long`, so the values are
wrapped in `TaggedHandle<T, N>` to keep the overload sets distinct.

| Handle | Box | Deleter | Depends on |
|---|---|---|---|
| `MemAllocationHandle` | `{handle, size, access descriptors}` | `pw_cuMemRelease(handle)` | nothing |
| `VaReservationHandle` | `{ptr, size}` | `pw_cuMemAddressFree(ptr, size)` | nothing |
| `VaMappingHandle` | `{ptr, size, h_alloc, h_reservation}` | `pw_cuMemUnmap(ptr, size)`, then the members release | allocation, reservation |

The allocation box carries the access descriptors it was created with. A mapping applies its
allocation's descriptors (and skips the call when there are none), so a chunk keeps its access
wherever it is mapped.

```
MemAllocationHandle create_mem_allocation_handle(size_t size, const CUmemAllocationProp& prop,
                                                 const CUmemAccessDesc* descs, size_t count);
VaReservationHandle create_va_reservation_handle(size_t size, size_t align, CUdeviceptr hint);
VaMappingHandle     create_va_mapping_handle(CUdeviceptr ptr, const MemAllocationHandle& h_alloc,
                                             const VaReservationHandle& h_res);
size_t mem_allocation_size(...) noexcept;  size_t va_reservation_size(...) noexcept;
```

Factories return the handle and put the status in thread-local `err`, as the other factories do;
an empty input handle sets `err` too, so an empty result always carries a status.
`create_va_mapping_handle` checks that the range lies inside the reservation, maps, and applies
access; if access fails it unmaps and returns empty. The eight VMM entry points, plus
`cuStreamSynchronize` and `cuStreamGetCaptureInfo`, join the `driver_api` pointer table.

### The range and the device pointer

```
struct VmmRange {                             // one per buffer base address
    std::vector<VaMappingHandle> mappings;    // ascending, contiguous; sum of sizes = range total
    std::mutex mu;                            // guards `streams`; nothing under it takes the GIL
    std::vector<DeallocationStream> streams;  // every stream a dying owner forwarded, deduplicated
};
using VmmRangeHandle = std::shared_ptr<VmmRange>;      // deleter: sync each stream, then destroy
DevicePtrHandle deviceptr_create_vmm(CUdeviceptr base, VmmRangeHandle range);
VmmRangeHandle  vmm_range(const DevicePtrHandle& h);   // empty for a non-VMM or closed handle
```

There is exactly one ownership chain: `Buffer._h_ptr` -> `DevicePtrBox` (holds the range) ->
`VmmRange` -> mappings -> reservations and allocations. The Cython buffer keeps no other
reference; grow operations call `Buffer_check_open` and then `vmm_range(buf._h_ptr)`.
`VirtualMemoryBuffer.close()` first refuses a capturing stream other than a default stream
(see "The resource") and then resets `_h_ptr`, as `Buffer.close()` does; the release itself is
the deleter chain below. A buffer's `size` is always a prefix of its range.

The `DevicePtrHandle` must own the memory because graph memcpy nodes retain `buf._h_ptr` as an
opaque owner; a non-owning handle would let a launched graph outlive its buffer. Any number of
owners may therefore exist at once: aliases from a grow, and graph attachments.

Deleters:

- `DevicePtrBox`, VMM flavor: release the GIL; append this box's recorded `DeallocationStream` to
  the range under `mu` (skipping an empty stream and duplicates); release `mu`; drop the range
  reference. It never blocks.
- `VmmRange`: release the GIL; unless the interpreter is finalizing, for each forwarded stream
  check the capture status and skip the stream with one report when a sync would disturb a
  capture (the stream is capturing, or it is the legacy stream while a blocking stream in its
  context is capturing), otherwise synchronize it under its bound context with the thread's
  capture mode switched to relaxed for the call, so a capture on an unrelated stream is not
  invalidated. Then, whether or not the syncs succeeded, destroy the mappings. Each mapping unmaps; the reservations free and the allocations release as
  their last references go. Every forwarded stream is synchronized because two aliases may have
  recorded different streams; synchronizing only the last one to die would unmap under work
  queued on the other. This is the first blocking deleter in the layer, and it may run inside
  the deferred-cleanup drain on the main thread, with the GIL released.

Allocations are shared by two ranges after a grow that moves the buffer. Shared ownership is
what makes that safe: the allocation is released exactly once, when its last mapping goes.

## The resource

- `VirtualMemoryResourceOptions` describes the allocations. `_check_config` rejects
  `location_type="host"` with a handle type other than `None`, which the driver rejects, and a
  request for GPUDirect RDMA on a device without support. `__init__` runs it after the VMM-support
  check; `modify_allocation` runs it on a per-call configuration, which must also name the
  resource's location. The resource reports `is_ipc_enabled = False`, which
  `Buffer.ipc_descriptor` reads.
- `cdef class VirtualMemoryBuffer(Buffer)` carries no extra state. It is created with
  `Buffer_from_deviceptr_handle(h_ptr, size, self, cls=VirtualMemoryBuffer)` and documented in
  `api.rst` like `ManagedBuffer`. It overrides `close(stream=None)` to reject a capturing stream
  other than a default stream, since VMM deallocation is synchronous and cannot be captured; a
  default stream is checked by the range deleter under its bound context, which reports and skips
  the sync instead of raising. `allocate(0)` returns one with no mapping.
- `allocate(size, *, stream=None)`:
  1. `size == 0` returns an empty buffer without a driver call, like the other resources.
  2. Build `CUmemAllocationProp` and the access descriptors from the options; query the
     granularity; align the size.
  3. Create the allocation, the reservation, and the mapping as locals; on any empty handle,
     `HANDLE_RETURN(get_last_error())`. The locals unwind everything.
  4. Build the range and the device pointer handle.
  5. Record the deallocation stream: the caller's stream if it is a real stream, otherwise the
     legacy default token bound to the device's primary context, as `_SynchronousMemoryResource`
     does. `allocate()` therefore never needs a current context. Host-located resources record no
     stream and close without a sync.
  6. Return a `VirtualMemoryBuffer` whose `size` is the aligned size.
- `modify_allocation(buf, new_size, config=None)`:
  - `Buffer_check_open(buf)`; `range = vmm_range(buf._h_ptr)`; an empty range means the buffer
    did not come from this resource: `TypeError`. `cfg = config or self.config` passes
    `_check_config`, governs the new chunk only and is not stored on the resource. Let `req = align_up(new_size)` and `total` be
    the range total.
  - `req <= buf.size`: return `buf`. The buffer already covers the request; `cfg` is not applied,
    and the access of memory that is already mapped never changes.
  - `buf.size < req <= total`: return a new `VirtualMemoryBuffer` over the same range with size
    `req`; no driver call. This serves a shorter alias asking for what the range already maps.
  - `req > total`, in place: probe `cuMemAddressReserve(req - total, align=0, hint=base+total)`.
    If the driver grants the hint, create the new allocation with `cfg`'s descriptors and its
    mapping as locals; `mappings.reserve(n+1)`; create a second `DevicePtrHandle` on the same
    range, copying the input's recorded deallocation stream; build the new buffer; `push_back`
    the mapping as the last, non-throwing step. If the driver grants another address, drop the
    reservation and move the buffer instead. If the probe fails, drain the status with
    `get_last_error()` and move the buffer.
  - `req > total`, moved: reserve `req` with `addr_align`; map every mapping in the range (shared
    allocation handles, their own descriptors) at `base_new + offset`; create and map the new
    allocation; build the new range (copying the input's recorded stream), handle, and buffer.
  - Both paths return a new buffer and leave the input open. See "Why `modify_allocation`
    returns a new buffer" below.
  - `modify_allocation` is not thread-safe with respect to two buffers that share a range; that
    synchronization is the caller's responsibility, as elsewhere in cuda.core.
- `deallocate(ptr, size, *, stream=None)` stays for the `MemoryResource` contract. It serves
  pointers wrapped with `Buffer.from_handle(ptr, size, mr=self)`: synchronize `stream` if given,
  `cuMemUnmap`, `cuMemAddressFree`. It handles one reservation, and the caller must have released
  its own `cuMemCreate` reference. It is not called for buffers from `allocate()`, whose ranges
  free themselves. A subclass override of `deallocate()` therefore does not run for them.

## Why `modify_allocation` returns a new buffer

The input buffer stays open and aliases the result. The chunks the input already mapped are
shared by both buffers and are freed when the last of the two closes; the chunk the grow added
belongs to the result. When the grow happens in place, the two buffers share one range at one
base, and the shorter one pins the whole range until it closes. Callers who are done with the
input close it.

The alternative, growing the input object in place so every holder sees the new size and
pointer, was rejected:

- In-place update reaches only holders of the Python object. Holders of the handle, such as
  graph memcpy nodes, DLPack capsules, and IPC descriptors, would keep the old mapping alive but
  see a different address than the buffer reports.
- An address change should be visible. When the buffer moves, an object that quietly changes
  address turns cached `int(buf.handle)` values into dangling pointers.
- `Buffer.__hash__` and `__eq__` include the size, so in-place growth changes the hash of a live
  object.
- Leaving the input open costs the caller one line and gives them a valid, shorter alias, which
  no in-place scheme can offer.

## Failure handling

- Rollback is RAII: locals die in reverse order, deleters run the `pw_*` wrappers, and failures
  become `CUDAWarning`.
- Every deleter releases the GIL first; the range deleter holds no C++ lock while it synchronizes
  or reports. A failed sync (lost context, capturing stream) is reported and the unmap proceeds;
  the driver needs no context for it, so nothing leaks.
- Empty handles always carry a status; the in-place probe drains its status before falling back.
- Factories that allocate are declared `except+` in `_rt.pxd`; deleters only destroy vectors.

## Invariants

Reservations and allocations are shared. After a move, one reservation holds every remapped
chunk, and the old and new ranges share their allocations. The invariants below hold under that
sharing. The rule that no deleter holds a C++ lock while it calls CUDA or Python is a property of
the whole layer; see [DESIGN.md](DESIGN.md).

1. A reservation is freed exactly once, with its original pointer and size.
2. A reservation is freed only after every mapping inside it is unmapped.
3. An allocation is released exactly once.
4. An allocation is released only after its last mapping in any range is unmapped.
5. A failed grow leaves the input unchanged and leaks nothing. Pointer, size, contents, access,
   and free memory are as before the call.
6. A grow leaves the input open. The input and the result see the same memory, whether the range
   grew in place or moved.
7. Every stream recorded by any owner of a range finishes before the range is unmapped, except as
   invariant 8 states.
8. A release never invalidates a graph capture. An explicit close on a capturing non-default
   stream raises. A release from garbage collection, or one ordered on a default stream that
   would disturb a capture, proceeds without ordering on that stream, warns once, and unmaps.
9. Graph nodes and aliases keep the range alive. The range dies with its last owner, in any close
   order.
10. A chunk's access is fixed when the chunk is created and travels with its allocation. Every
    mapping of the chunk applies the same descriptors. A grow's `config` governs only the new
    chunk and never changes mapped memory.
11. The mappings of a range are contiguous and ascending, and the range total is the sum of their
    sizes. A buffer's size is a multiple of the granularity and a prefix of its range. A size that
    cannot be rounded raises.
12. A range is findable by its base address only while it is alive. The registry entry is removed
    before the reservation is freed.
13. Buffers from `allocate()` free themselves and never call `deallocate()`. `deallocate()` serves
    only pointers wrapped with `Buffer.from_handle`.
14. No operation needs a current context. A default-stream token is bound to the resource's
    device context when it is recorded, and the release runs under that context. A host-located
    resource records no default stream, so its release is ordered only on a stream the caller
    passes.
15. Buffers alive at interpreter shutdown are freed without a warning or an error.
