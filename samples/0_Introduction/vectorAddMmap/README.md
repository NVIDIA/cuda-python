# Sample: Vector Add via Virtual Memory Management (Python)

## Description

Vector add ``C = A + B``, but each device buffer is built by hand out of
per-device physical backings stitched into a single contiguous virtual
address range. This is the only sample in ``/samples`` that teaches the
CUDA Virtual Memory Management (VMM) API.

The core VMM flow:

```
cuMemAddressReserve       reserve a chunk of virtual address space
cuMemGetAllocationGranularity  query alignment requirements
cuMemCreate               create a physical backing on a specific device
cuMemMap                  map the backing into the reserved VA range
cuMemSetAccess            grant read/write access to the mapping device(s)
cuMemUnmap                unmap the VA range
cuMemAddressFree          release the VA range back to the OS
cuMemRelease              release the physical handle
```

The allocation is **striped across all peer-capable devices** that also
support VMM ("backing devices"). It's accessed by a single mapping device
(the current device) via ``cuMemSetAccess``. This is the pattern to reach
for when you want NUMA-aware placement across a multi-GPU machine, or when
you need explicit control over which device physically owns each byte.

Waives on macOS, on 32-bit / aarch64 / sbsa configurations that don't
support VMM, and on devices that don't report
``VIRTUAL_ADDRESS_MANAGEMENT_SUPPORTED``.

## What You'll Learn

- Reserving virtual address space with `cuMemAddressReserve`
- Querying granularity with `cuMemGetAllocationGranularity`
- Creating physical backings with `cuMemCreate` and releasing the handle
  after mapping
- Mapping backings into a VA range with `cuMemMap` (contiguous virtual,
  discontiguous physical)
- Configuring access via `cuMemAccessDesc` + `cuMemSetAccess`
- Tearing down VMM allocations cleanly: `cuMemUnmap` +
  `cuMemAddressFree`
- Detecting peer capability with `cuDeviceCanAccessPeer` for striping
  across devices

## Key Libraries

- [`cuda.bindings`](https://nvidia.github.io/cuda-python/cuda-bindings/latest/) - driver bindings
- `numpy` - host arrays

## Key APIs

### From `cuda.bindings.driver`

- `cuMemAddressReserve` / `cuMemAddressFree`
- `cuMemCreate` / `cuMemRelease`
- `cuMemMap` / `cuMemUnmap`
- `cuMemGetAllocationGranularity`
- `cuMemSetAccess`
- `CUmemAllocationProp`, `CUmemLocationType`, `CUmemAllocationType`
- `CUmemAccessDesc`, `CUmemAccess_flags`, `CUmemAllocationGranularity_flags`
- `cuDeviceCanAccessPeer`
- `CU_DEVICE_ATTRIBUTE_VIRTUAL_ADDRESS_MANAGEMENT_SUPPORTED`

## Requirements

### Hardware

- NVIDIA GPU that reports
  `CU_DEVICE_ATTRIBUTE_VIRTUAL_ADDRESS_MANAGEMENT_SUPPORTED = 1`
  (Pascal-and-newer discrete GPUs on Linux x86-64 and Windows)

### Software

- Linux x86-64 or Windows (not supported on macOS, ARMv7, aarch64, or sbsa)
- CUDA Toolkit 13.0 or newer
- Python 3.10 or newer
- `cuda-python` (>=13.0.0)
- `numpy`

## Installation

```bash
pip install -r requirements.txt
```

## How to Run

```bash
python vectorAddMmap.py
python vectorAddMmap.py --device=1
```

## Expected Output

```
Device 0 VIRTUAL ADDRESS MANAGEMENT SUPPORTED = 1.
Result = PASS (max error 0.000e+00 over 50000 elements, striped across 2 device(s))
Done
```

## Files

- `vectorAddMmap.py` - Python implementation using `cuda.bindings.driver`
- `README.md` - This file
- `requirements.txt` - Sample dependencies
- `../../Utilities/cuda_bindings_utils.py` - Shared bindings helpers (imported by this sample)

## See Also

- [CUDA Python Documentation](https://nvidia.github.io/cuda-python/)
- [`cuda.bindings` driver API](https://nvidia.github.io/cuda-python/cuda-bindings/latest/module/driver.html)
- [CUDA Driver API — Virtual Memory Management](https://docs.nvidia.com/cuda/cuda-driver-api/group__CUDA__VA.html)
