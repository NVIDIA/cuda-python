# Sample: Grace Hopper Memory

On Grace Hopper, NVLink-C2C makes CPU memory coherently accessible to the GPU,
and the GPU translates addresses through the host page tables. GPU code can
therefore work in place on pinned, managed and ordinary pageable host memory,
and the CPU sees the results without a copy.

This sample allocates one buffer from each `cuda.core` memory resource, plus a
plain NumPy array:

- `PinnedMemoryResource` on the GPU's nearest host NUMA node
- `ManagedMemoryResource`
- pageable memory from `np.empty()`
- `DeviceMemoryResource`, for comparison

It wraps each buffer as a CuPy array and uses standard CuPy operations, with no
custom kernels. For each host-accessible buffer the CPU writes the input, CuPy
doubles it and sums it on the GPU, and the CPU checks the result in the same
memory. Device memory needs an explicit copy in each direction.

`cp.from_dlpack()` imports the device and managed buffers. It does not import
pinned (`kDLCUDAHost`) or CPU (`kDLCPU`) memory, so the sample wraps those
pointers with `cupy.cuda.UnownedMemory`.

## Requirements

- A Grace Hopper system: the GPU must access pageable memory through the host
  page tables, support concurrent managed access and support host NUMA memory
  pools. Other systems waive with exit code 2 (configurable via
  `CUDA_PYTHON_SAMPLE_WAIVER_EXIT_CODE`).
- `cuda-core` 1.2.2 or newer with CUDA 13, CuPy for CUDA 13 and NumPy 2.2.5
  or newer.

```bash
cd samples/cuda_core/graceHopperMemory
pip install -r requirements.txt
python graceHopperMemory.py
```

## Example Output

On a GH200 480GB:

```text
Device: NVIDIA GH200 480GB, compute capability 9.0
  host_numa_id: 0
  pageable_memory_access_uses_host_page_tables: True
  concurrent_managed_access: True
  host_numa_memory_pools_supported: True
GPU work on host-resident memory:
  pinned, host NUMA node 0               sum=17163091968 verified on the CPU without a copy
  managed                                sum=17163091968 verified on the CPU without a copy
  pageable NumPy array                   sum=17163091968 verified on the CPU without a copy
For comparison, device memory needs explicit copies:
  device                                 verified after a device-to-host copy
Done
```
