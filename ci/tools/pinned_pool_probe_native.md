# Native pinned-memory pool reproducer

`pinned_pool_probe.cpp` calls the CUDA Driver API directly. It does not load
Python, cuda.bindings, cuda.core, or the CUDA runtime. It requires CUDA 13.x
headers and a driver supporting HOST memory pools.

The Windows wheel build compiles it once on the CPU runner with MSVC and the
current mini toolkit, before switching to the maintenance toolkit. It publishes
`native-pinned-memory-win-64-<workflow SHA>` containing the executable, source,
instructions, and build log. This also runs when Core wheels are reused but
current-major Core tests are required.

The dedicated pinned-memory CDCI job downloads that artifact from the current
run; the GPU test runner needs no Visual Studio installation. The executable
matches the workflow revision and diagnostic tools, including when older wheels
are selected for testing. The downloaded files and per-case results are included
in the existing pinned-memory diagnostic artifact.

Each case runs in a fresh process with a 120-second watchdog. A failed build
still publishes its source and log; a missing executable fails the native cases
while the Python probes continue. CUDA errors, crashes, timeouts, and incomplete
reports also fail the diagnostic job.

## Build and run independently

From an x64 Visual Studio developer PowerShell with CUDA_PATH set:

```powershell
cl.exe /nologo /std:c++17 /EHsc /O2 /MT /I"$env:CUDA_PATH/include" pinned_pool_probe.cpp /Fe:pinned_pool_probe.exe /link "$env:CUDA_PATH/lib/x64/cuda.lib"
./pinned_pool_probe.exe --cycles 200 --stream legacy --output-dir baseline
```

The CDCI mini toolkit may put `cuda.lib` directly under `lib`; the workflow
checks both layouts. No nvcc or device compilation is needed.

For a Linux driver comparison (with a local CUDA toolkit):

```sh
c++ -std=c++17 -O2 -I"$CUDA_PATH/include" pinned_pool_probe.cpp -L"$CUDA_PATH/lib64/stubs" -lcuda -o pinned_pool_probe
./pinned_pool_probe --cycles 200 --stream legacy --output-dir baseline
```

The stub is only a link-time dependency; execution needs the real driver.

## Experiments

The baseline creates a 2 MiB capped HOST/PINNED pool, grants device 0 read/write
access, allocates and frees 1 MiB, synchronizes, trims to zero, reads/resets/verifies
an alternating reserved/used high watermark, destroys the pool, and synchronizes
again. It pushes/pops the primary context each cycle, matching the raw Python
probe. There are no kernels or application threads.

CDCI runs all six baseline combinations of `--stream legacy`, `per-thread`, or
`nonblocking`, with and without `--revoke-access`. Without that flag, access is
still **granted**; only revocation is omitted.

Three additional cases use the legacy stream and keep access:

| `--variant` | Difference from baseline |
| --- | --- |
| `no-trim` | Omit the explicit trim call; automatic driver release remains enabled. |
| `no-reset` | Omit watermark queries and reset. |
| `reuse` | Create/grant once, reuse the pool across cycles, then destroy/synchronize once at the end. |

The default variant is `baseline`. Use separate output directories/processes for
each experiment. Native-only cases can also be selected through
`diagnose_pinned_memory.py --native-executable PATH --case NAME`; see its help for
case names.

## Evidence

- `summary.json`: completed/requested cycles, configuration, header/driver API
  versions, last watermark values, and the first failing stage/API/status.
  Iteration numbers are zero-based. Reuse is only successful if final destruction
  and synchronization also succeed, even when every allocation cycle completed.
- `calls.jsonl`: the last 128 completed API calls, buffered in memory during the
  experiment to reduce logging overhead.
- `progress.json`: memory-mapped JSON recording entry/return around each CUDA
  call. This survives child crashes and watchdog termination; `returned: false`
  and status -1 identify a call entered but not returned. An interrupted write
  can leave a partial record. On ordinary failure it retains the first error.
- The case log includes baseline GPU memory, and Windows host RAM/commit/address
  space availability. After failure, a separate `cuMemGetInfo` call records
  whether the current context still accepts queries. The summary/history are
  saved before that query, in case it hangs too.

The first error ends the experiment. CUDA cleanup is deliberately not attempted
on failure, so secondary errors cannot replace it; process exit reclaims state.
Successful runs check stream/context cleanup. This is a diagnostic, not a proposed
production workaround. Passing an omission variant suggests a follow-up; it does
not by itself prove the omitted API is defective.
