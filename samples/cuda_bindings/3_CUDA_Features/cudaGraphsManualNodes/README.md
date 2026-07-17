# Sample: CUDA Graphs — Manual Node Construction vs Stream Capture (Python)

## Description

Builds the same two-stage reduction as a CUDA graph twice, back-to-back:

1. **Manual node construction** — `cudaGraphCreate` +
   `cudaGraphAddMemcpyNode` / `cudaGraphAddMemsetNode` /
   `cuGraphAddKernelNode`, wiring node dependencies by hand.
2. **Stream capture** — `cudaStreamBeginCapture` / `cudaStreamEndCapture`
   on three streams joined by events; the driver derives the same DAG
   from the actual launches.

Both paths produce a `cudaGraph_t`, are instantiated
(`cudaGraphInstantiate`), cloned (`cudaGraphClone`), and replayed several
times (`cudaGraphLaunch`).

The high-level counterpart in `/samples/cuda_core` is
[`samples/cuda_core/cudaGraphs/`](../../../cuda_core/cudaGraphs/), which teaches stream capture
at the `cuda.core` layer. This sample is the only place in `/samples/cuda_bindings`
that shows the **manual node-by-node construction** pattern — useful
when you're programmatically building a graph without a driving stream.

## What You'll Learn

- Constructing a CUDA graph manually, one node at a time
- Wiring `cudaGraphAddMemcpyNode` / `cudaGraphAddMemsetNode` /
  `cuGraphAddKernelNode` with explicit dependency lists
- Instantiating (`cudaGraphInstantiate`) and cloning (`cudaGraphClone`)
  a graph
- Replaying an executable graph on a stream (`cudaGraphLaunch`)
- Building the same graph implicitly via stream capture on multiple
  streams joined by events

## Key Libraries

- [`cuda.bindings`](https://nvidia.github.io/cuda-python/cuda-bindings/latest/) - driver + runtime bindings
- `numpy` - host / dtype sizes

## Key APIs

### From `cuda.bindings.runtime`

- `cudaGraphCreate`, `cudaGraphDestroy`
- `cudaGraphAddMemcpyNode`, `cudaGraphAddMemsetNode`
- `cudaGraphInstantiate`, `cudaGraphExecDestroy`
- `cudaGraphClone`, `cudaGraphGetNodes`, `cudaGraphLaunch`
- `cudaStreamBeginCapture` / `cudaStreamEndCapture`
- `cudaStreamCreate`, `cudaStreamWaitEvent`, `cudaStreamSynchronize`
- `cudaEventCreate`, `cudaEventRecord`
- `cudaMallocHost`, `cudaMalloc`, `cudaFreeHost`, `cudaFree`

### From `cuda.bindings.driver`

- `cuGraphAddKernelNode`
- `CUDA_KERNEL_NODE_PARAMS`
- `cuLaunchKernel`

## Requirements

### Hardware

- NVIDIA GPU with Compute Capability 5.0 or higher

### Software

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
python cudaGraphsManualNodes.py
python cudaGraphsManualNodes.py --device=1
```

## Expected Output

```
16777216 elements
threads per block  = 512
Graph Launch iterations = 3

Num of nodes in the graph created manually = 6
Cloned Graph Output..

Num of nodes in the graph created using stream capture API = 7
Cloned Graph Output..
Done
```

## Files

- `cudaGraphsManualNodes.py` - Python implementation using `cuda.bindings`
- `README.md` - This file
- `requirements.txt` - Sample dependencies
- `../../Utilities/cuda_bindings_utils.py` - Shared bindings helpers (imported by this sample)

## See Also

- [CUDA Python Documentation](https://nvidia.github.io/cuda-python/)
- [`cuda.bindings` runtime API](https://nvidia.github.io/cuda-python/cuda-bindings/latest/module/runtime.html)
- [`samples/cuda_core/cudaGraphs/`](../../../cuda_core/cudaGraphs/) - the high-level `cuda.core` equivalent (stream capture only)
- [CUDA C++ Programming Guide — CUDA Graphs](https://docs.nvidia.com/cuda/cuda-c-programming-guide/index.html#cuda-graphs)
