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
access; if access fails it unmaps and returns empty. The seven VMM entry points, plus
`cuStreamSynchronize`, `cuStreamIsCapturing` and `cuThreadExchangeStreamCaptureMode`, join the
`driver_api` function table.

### The range and the device pointer

```
using VmmRange = std::vector<VaMappingHandle>;      // ascending, contiguous; sum of sizes = range total
using VmmRangeHandle = std::shared_ptr<const VmmRange>;   // immutable once built
VmmRangeHandle  create_vmm_range(const std::vector<VaMappingHandle>& mappings);
DevicePtrHandle deviceptr_create_vmm(CUdeviceptr base, const VmmRangeHandle& range);
VmmRangeHandle  vmm_range(const DevicePtrHandle& h);   // the range; only for handles from deviceptr_create_vmm
```

A range is a list of mapping handles and nothing else. It is built once and never changes. A
grow reads the input's range, copies the list, appends the new mapping (or, when the buffer
moves, replaces every entry with a mapping at the new address) and wraps the copy in a new range
for the result. The input's range is untouched. Mapping handles are shared between the two
ranges, so each mapping unmaps once, when the last range that holds it goes; a chunk that only
the result holds is unmapped as soon as the result closes.

Every buffer therefore owns exactly one range and records exactly one deallocation stream, and
two buffers never share mutable state. A full alias (a request the input already covers) shares
the input's range object, which is safe because it is immutable. This is what makes concurrent
grows of aliased buffers safe without a lock: they read a shared immutable list and write only
their own locals. Which grow extends in place and which one moves depends on what address space
the driver has free when each probe runs, exactly as for a single thread.

There is exactly one ownership chain: `Buffer._h_ptr` -> `DevicePtrBox` (holds the range) ->
`VmmRange` -> mappings -> reservations and allocations. The Cython buffer keeps no other
reference; grow operations call `Buffer_check_open`, copy `buf._h_ptr` into a local (to isolate
it from concurrent operations that might, e.g., call `close()`), and then call `vmm_range()` on the copy.
`VirtualMemoryBuffer.close()` first refuses a capturing stream other than a default stream
(see "The resource") and then resets `_h_ptr`, as `Buffer.close()` does; the release itself is
the deleter below. A buffer's `size` is always a prefix of its range.

The `DevicePtrHandle` must own the memory because graph memcpy nodes retain `buf._h_ptr` as an
opaque owner; a non-owning handle would let a launched graph outlive its buffer. Any number of
owners may therefore exist at once: aliases from a grow, and graph attachments.

The box behind a VMM handle is a `VmmDevicePtrBox`: a `DevicePtrBox` with a `VmmRangeHandle`
member and no virtual functions, so other boxes pay nothing. Every handle on a
`VirtualMemoryBuffer` comes from `deviceptr_create_vmm`, so `modify_allocation` checks the Python
class and `vmm_range(h)` downcasts the box without a tag, the way `deallocation_stream(h)` reads
the stream. A size-zero buffer has a VMM box over an empty range.

Deleter of a `VmmDevicePtrBox`: release the GIL; unless the
interpreter is finalizing or no stream was recorded, check the capture status of the recorded
stream and skip it with one report when a sync would disturb a capture (the stream is capturing,
or it is the legacy stream while a blocking stream in its context is capturing), otherwise
synchronize it under its bound context with the thread's capture mode switched to relaxed for
the call, so a capture on an unrelated stream is not invalidated. Then drop the range: each
mapping this buffer was the last to hold unmaps, and its reservation frees and its allocation
releases as their last references go. This is the same model as every other `Buffer`: the
recorded stream orders the release of this buffer's memory, and a caller who touches that memory
from another stream must order that work before the close. The deleter blocks until that work
completes. `cuMemUnmap` is not stream-ordered, so waiting is the only way to honor the order;
`_SynchronousMemoryResource` and `LegacyPinnedMemoryResource` wait the same way in
`deallocate()`, while pool-backed buffers never block because `cuMemFreeAsync` is stream-ordered.
The wait happens wherever the last reference goes: an explicit `close()`, a garbage collection,
or the deferred-cleanup drain on the main thread, with the GIL released. A caller who needs to
control when it happens closes the buffer explicitly or records an idle deallocation stream.
Emulating stream order with a host callback that hands the release to the deferred-cleanup
queue is planned as a follow-up for every synchronous release in cuda.core.

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
  - `req <= total`: return a new `VirtualMemoryBuffer` over the same range with size
    `max(req, buf.size)`; no driver call. The range already maps the request, so `cfg` is not
    applied and the access of memory that is already mapped never changes. The result is a full
    alias of the input when the input already covers the request; `buf` itself is never returned,
    so closing the result never closes `buf`.
  - `req > total`, in place: copy the input's mapping list; probe
    `cuMemAddressReserve(req - total, align=0, hint=base+total)`. If the driver grants the hint,
    create the new allocation with `cfg`'s descriptors and its mapping as locals, append the
    mapping to the copy, build a new range from it, and create the result's `DevicePtrHandle` on
    that range at the same base, copying the input's recorded deallocation stream. If the driver
    grants another address, drop the reservation and move the buffer instead. If the probe fails,
    drain the status with `get_last_error()` and move the buffer.
  - `req > total`, moved: reserve `req` with `addr_align`; map every mapping in the copied list
    (shared allocation handles, their own descriptors) at `base_new + offset`, replacing the copy's
    entries; create and map the new allocation; build the new range, handle (copying the input's
    recorded stream), and buffer.
  - Every path returns a new buffer and leaves the input open and its range unchanged. See "Why
    `modify_allocation` returns a new buffer" below.
  - Concurrent calls on buffers that alias one another are safe: each reads an immutable range
    and writes only its own locals. Which call extends in place and which one moves depends on
    what address space the driver has free when each probe runs, exactly as for a single thread.
    Closing a buffer while another thread passes it to `modify_allocation` is unsupported
    (concurrency.rst); the call copies the input handle first, so that misuse defers the release
    rather than crashing.
- `deallocate(ptr, size, *, stream=None)` stays for the `MemoryResource` contract. It serves
  pointers wrapped with `Buffer.from_handle(ptr, size, mr=self)`: synchronize `stream` if given,
  `cuMemUnmap`, `cuMemAddressFree`. It handles one reservation, and the caller must have released
  its own `cuMemCreate` reference. It is not called for buffers from `allocate()`, whose ranges
  free themselves. A subclass override of `deallocate()` therefore does not run for them.

## Sharing across processes

A `VirtualMemoryBuffer` is shared the way pool-backed buffers are, through
`Buffer.ipc_descriptor` and `Buffer.from_ipc_descriptor`, when the resource's
`handle_type` can travel between processes (`posix_fd` on Linux;
`is_ipc_enabled` reports it). The exporter calls `cuMemExportToShareableHandle`
once per mapping that covers the buffer's size, in address order, and the
descriptor carries the handles with each allocation's size. Each file
descriptor is owned by an `IPCAllocationHandle` and closed when the descriptor
goes; `multiprocessing` duplicates it into the receiving process. The sizes
travel because the OS handle does not expose them to the importer; a wrong size
fails the importer's `cuMemMap`.

The importer is a `VirtualMemoryResource` of the receiving process with the
same `handle_type`, for the device that owns the memory. `import_mem_allocation_handle`
calls `cuMemImportFromShareableHandle` and wraps the result in the same box and
deleter as `create_mem_allocation_handle`: the driver gives an imported handle
one reference and frees the memory when all references are released and no
mapping remains, so `cuMemRelease` is the right teardown for both. After each
import, `cuMemGetAllocationPropertiesFromHandle` must report the resource's
location (the same device, or the same host location type); a mismatch is a
`ValueError` before anything is mapped, because the access descriptors would
describe the wrong location. The access descriptors stored in the box are the
importer's, built from its options for its device and peers; the exporter's
access does not travel. The import then reserves `sum(sizes)` with the
importer's alignment, maps each allocation in order, and builds a range and a
`VirtualMemoryBuffer` exactly as `allocate()` does, so a grown buffer imports
as one contiguous range with the same byte layout, and the imported buffer
frees itself through the same deleters. A descriptor imported in the exporting
process yields an alias of the exporter's memory at a new address. A
`VirtualMemoryResource` pickles as (device, options), which is what lets a
`Buffer` pickle as (resource, descriptor).

File descriptors and memory. The driver copies each shared allocation into a
handle of its own during the import and keeps no reference to the file
descriptor, so the imported buffer does not keep the descriptor; it records
only that it was imported (`is_mapped`). The descriptor's file descriptors live
as long as the descriptor object, on the exporter (where the buffer caches it)
and in every process that unpickled a copy, and each open descriptor keeps the
physical memory allocated. A descriptor on a `Queue` pins the memory in the
sender until the receiver unpickles it.

Re-export. The driver exports only allocations created with the requested
handle type, and an imported allocation carries none, so an imported buffer,
and any buffer grown from one, cannot be exported again; `ipc_descriptor` says
so instead of surfacing the driver's INVALID_VALUE. A `modify_allocation`
config must keep the resource's `handle_type` for the same reason: every chunk
of a buffer must be exportable the same way.

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
7. The stream a buffer recorded finishes before that buffer's share of its range is released,
   except as invariant 8 states. A mapping that another buffer still holds stays mapped.
8. A release never invalidates a graph capture. An explicit close on a capturing non-default
   stream raises. A release from garbage collection, or one ordered on a default stream that
   would disturb a capture, proceeds without ordering on that stream, warns once, and unmaps.
9. Graph nodes and aliases keep the mappings alive. A mapping dies with the last range that holds
   it, in any close order. A range is immutable once built.
10. A chunk's access is fixed when the chunk is created and travels with its allocation. Every
    mapping of the chunk applies the same descriptors. A grow's `config` governs only the new
    chunk and never changes mapped memory.
11. The mappings of a range are contiguous and ascending, and the range total is the sum of their
    sizes. A buffer's size is a multiple of the granularity and a prefix of its range. A size that
    cannot be rounded raises.
12. Every handle on a `VirtualMemoryBuffer` points at a VMM box whose range is the buffer's
    mappings; a size-zero buffer's range is empty.
13. Buffers from `allocate()` free themselves and never call `deallocate()`. `deallocate()` serves
    only pointers wrapped with `Buffer.from_handle`.
14. No operation needs a current context. A default-stream token is bound to the resource's
    device context when it is recorded, and the release runs under that context. A host-located
    resource records no default stream, so its release is ordered only on a stream the caller
    passes.
15. Buffers alive at interpreter shutdown are freed without a warning or an error.
