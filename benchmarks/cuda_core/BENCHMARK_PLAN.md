# cuda.core Benchmark Plan

## Goal

Measure the **additional Python-side latency** that `cuda.core` adds on top
of `cuda.bindings` for every public API with a clear cuda.bindings
counterpart. Target is **< 1 µs** of extra overhead per call.

The baseline for this suite is not C++: it is the cuda.bindings Python
results file (`../cuda_bindings/results-python.json`). Benchmark IDs are
kept identical across suites so a direct diff is possible — see
`compare.py`.

## Phase 1 coverage

Benchmark IDs shipped in this PR (all map 1:1 to cuda.bindings):

| ID                                        | cuda.core surface                                  |
|-------------------------------------------|----------------------------------------------------|
| `ctx_device.ctx_get_current`              | `Device()`                                         |
| `ctx_device.ctx_set_current`              | `dev.set_current()`                                |
| `ctx_device.device_get`                   | `Device(0)`                                        |
| `ctx_device.device_get_attribute`         | `dev.properties.compute_capability_major` (cached) |
| `stream.stream_create_destroy`            | `dev.create_stream()` + `stream.close()`           |
| `stream.stream_synchronize`               | `stream.sync()`                                    |
| `event.event_create_destroy`              | `dev.create_event()` + `event.close()`             |
| `event.event_record`                      | `stream.record(event)`                             |
| `event.event_query`                       | `event.is_done`                                    |
| `event.event_synchronize`                 | `event.sync()`                                     |
| `memory.mem_alloc_free`                   | `dev.allocate(size)` + `buf.close()` (async pool!) |
| `memory.mem_alloc_async_free_async`       | `dev.allocate(size, stream)` + `buf.close(stream)` |
| `launch.launch_empty_kernel`              | `launch(stream, config, kernel)`                   |
| `launch.launch_small_kernel`              | `launch(..., ptr)`                                 |
| `launch.launch_16_args`                   | `launch(..., *16 ptrs)`                            |
| `launch.launch_256_args`                  | `launch(..., *256 ptrs)`                           |
| `launch.launch_512_args`                  | `launch(..., *512 ptrs)`                           |

## Intentionally not covered in Phase 1

- `ctx_device.ctx_get_device`, `ctx_device.device_primary_ctx_retain`:
  cuda.core abstracts CUDA contexts away — no direct counterpart.
- `enum.*`: cuda.core does not re-export cuda.bindings enums; those
  benches measure a cuda.bindings-specific cost.
- `stream.stream_query`: no public `Stream.query()` in cuda.core.
- `launch.launch_*_pre_packed`: pre-packing is a cuda.bindings-specific
  optimization of its tuple-of-args API; cuda.core's `ParamHolder`
  handles packing internally on every call.
- `launch.launch_512_bools / _ints / _doubles / _bytes / _longlongs`:
  non-pointer scalar arg variants — deferred to Phase 2.
- `launch.launch_2048b`: struct-by-value arg — requires a
  `TensorMapDescriptor`/ctypes path that is not yet settled in cuda.core.
- `memory.memcpy_htod / _dtoh / _dtod`: cuda.core's `Buffer.copy_to /
  copy_from` only go buffer-to-buffer through `cuMemcpyAsync`; pairing
  with cuda.bindings' synchronous `cuMemcpyDtoD` / `HtoD` / `DtoH` would
  be apples-to-oranges. Deferred until the comparable host-memory
  resource path is finalized.
- NVRTC / module benches: cuda.core's `Program` / `ObjectCode` pipeline
  is meaningfully different from raw NVRTC; deserves its own set of
  bench functions rather than reusing cuda.bindings IDs.

## Audit notes: known driver-call mismatches

The IDs above match cuda.bindings 1:1 at the *public API* level, but a few
measure a different underlying driver call. Readers of `compare.py` should
know which deltas are "pure cuda.core Python overhead" vs. a deliberate
different driver path:

- `ctx_device.ctx_get_current`: `Device()` reads a TLS-cached device
  object; cuda.bindings calls `cuCtxGetCurrent` every iteration. Expect
  cuda.core to be faster. Not apples-to-apples at the driver level;
  apples-to-apples at the user-facing "give me the current device" level.
- `ctx_device.device_get_attribute`: `DeviceProperties` caches the first
  lookup in a Python dict (`_get_cached_attribute`, `_device.pyx:75`).
  After the first iteration this is a dict hit, not a `cuDeviceGetAttribute`
  driver call. Expect cuda.core to be faster here too. A future
  paired bench can use an uncached attribute (e.g. `compute_mode`) to
  measure the wrapper overhead on the driver-call path.
- `stream.stream_create_destroy`: default `StreamOptions(nonblocking=True)`
  yields the same `CU_STREAM_NON_BLOCKING` flag as the cuda.bindings
  bench, but cuda.core additionally calls `cuCtxGetStreamPriorityRange`
  and builds a `StreamOptions` dataclass per create — real cuda.core
  overhead, fair to measure.
- `memory.mem_alloc_free`: **deliberate mismatch**. `dev.allocate(size)`
  with `stream=None` routes through `_MP_allocate` → `cuMemAllocFromPoolAsync`
  on the cached default stream (`_memory_pool.pyx:302`). cuda.bindings
  measures the synchronous `cuMemAlloc`. The bench captures the
  user-visible cost of `dev.allocate(size)`, which is what a cuda.core
  user actually pays; it does **not** isolate "Python wrapper overhead
  on top of `cuMemAlloc`" because cuda.core does not expose a sync
  `cuMemAlloc` path.
- `memory.mem_alloc_async_free_async`: same internal path as
  `mem_alloc_free` (both go through `cuMemAllocFromPoolAsync`); the
  only difference is whether `default_stream()` is fetched or a stream
  is passed in. Driver call matches cuda.bindings' `cuMemAllocAsync`
  semantically but uses the pool-backed variant.
- `launch.*`: cuda.core uses `cuLaunchKernelEx` (takes a
  `CUlaunchConfig` struct) and allocates a fresh `ParamHolder` +
  `LaunchConfig._to_native_launch_config()` per call. cuda.bindings
  uses `cuLaunchKernel` with pre-built arg tuples. The delta captures
  both the Python-side per-call work and the `Ex` vs non-`Ex` driver
  cost; this is real and expected cuda.core overhead.

## Next up (not in this PR)

1. Scalar launch variants (512 bools/ints/doubles/bytes/longlongs) so
   arg-packing overhead is covered beyond the pointer fast-path.
2. Buffer-based memcpy benchmarks once the host-memory resource path is
   stable in cuda.core.
3. NVRTC / `Program` / `ObjectCode` latency benches.
4. TMA (`TensorMapDescriptor`) benches when cuda.core's CCCL-backed
   helper is formalised.
