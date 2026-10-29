# Sample: Grace Hopper Memory

This sample prints the CUDA NUMA and pageable-memory properties and the NVML
C2C status, then runs a kernel directly on a pageable NumPy array using the
host page tables. It compares reads from a host-NUMA pinned buffer, pageable
NumPy memory, device memory and managed memory after prefetch to either the
host NUMA node or the device. Every kernel result is checked.

The read kernel sums 1 GiB of `float32` values using a grid-stride loop, one
shared-memory reduction and one atomic operation per block. The reported rate
is bytes read divided by host-clock time around a synchronized kernel launch,
the median of ten runs. It is not a tuned peak-bandwidth benchmark. Pageable
and pinned reads receive one untimed warmup. Before **each** managed-memory
measurement, the sample prefetches again to the indicated location and waits
for the prefetch to finish; the host-prefetched kernel time can include
migration to the GPU.

## Requirements

- A Grace Hopper system with CUDA pageable access through host page tables,
  concurrent managed access and host NUMA memory pools. Systems without these
  properties waive with exit code 2 (configurable via
  `CUDA_PYTHON_SAMPLE_WAIVER_EXIT_CODE`).
- CUDA Toolkit 13.x, `cuda-python`, `cuda-core` 1.3.0 or newer, and NumPy
  2.2.5 or newer. The required cuda-core APIs are present on the current main
  branch; 1.3.0 may not yet be released.
- About 3 GiB of host memory, 1 GiB of device memory, and managed-memory
  capacity. Allocate these only on a system where the GPU is available to you.

Install from `requirements.txt` after the required cuda-core release is
available, or build the matching development version from the repository.

```bash
cd samples/cuda_core/graceHopperMemory
python graceHopperMemory.py
```

## Historical probe measurements

A separate prototype using the same kernel and input size was measured on one
GH200 480GB system (driver 595.58.03, CUDA 13.4.2, Python 3.12). Three runs,
each reporting the median of ten launches, gave the following ranges:

| Memory | Read rate (GB/s) |
| --- | ---: |
| Pinned, host NUMA node 0 | 339–345 |
| Pageable NumPy array | 224–225 |
| Device memory | 2386–2417 |
| Managed, first kernel after host prefetch | 5.1–5.4 |
| Managed, after device prefetch | 2380–2393 |

The host-prefetched managed result includes first-touch migration on that
system; it should not be interpreted as direct C2C bandwidth. The prototype
results are historical measurements, not results from this sample revision.
Rates depend on topology, system load, memory placement and kernel design.
