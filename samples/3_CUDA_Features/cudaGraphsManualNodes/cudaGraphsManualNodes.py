# Copyright (c) 2025, NVIDIA CORPORATION. All rights reserved.
#
# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions
# are met:
#  * Redistributions of source code must retain the above copyright
#    notice, this list of conditions and the following disclaimer.
#  * Redistributions in binary form must reproduce the above copyright
#    notice, this list of conditions and the following disclaimer in the
#    documentation and/or other materials provided with the distribution.
#  * Neither the name of NVIDIA CORPORATION nor the names of its
#    contributors may be used to endorse or promote products derived
#    from this software without specific prior written permission.
#
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS ``AS IS'' AND ANY
# EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
# IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR
# PURPOSE ARE DISCLAIMED.  IN NO EVENT SHALL THE COPYRIGHT OWNER OR
# CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL,
# EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO,
# PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR
# PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY
# OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
# (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
# OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.

# /// script
# dependencies = ["cuda-python>=13.0.0", "numpy>=1.24"]
# ///

"""
CUDA Graphs: manual node construction vs stream capture

Builds the same two-stage reduction as a CUDA graph twice:

  1. **Manual node construction** -- ``cudaGraphCreate`` + explicit
     ``cudaGraphAddMemcpyNode`` / ``cudaGraphAddMemsetNode`` /
     ``cuGraphAddKernelNode`` calls, wiring dependencies by hand.
  2. **Stream capture** -- ``cudaStreamBeginCapture`` /
     ``cudaStreamEndCapture`` on three streams joined by events; the
     driver derives the same DAG from the actual launches.

Both paths produce a ``cudaGraph_t``, are instantiated
(``cudaGraphInstantiate``), cloned (``cudaGraphClone``), and replayed
several times (``cudaGraphLaunch``).

The high-level counterpart in ``/samples`` is
[`samples/cudaGraphs/`](../../cudaGraphs/), which teaches stream capture at
the ``cuda.core`` layer. This sample is the *only* place that shows the
manual-node-construction pattern (useful when you're building a graph
programmatically without a driving stream).
"""

import ctypes
import random as rnd
import sys
from pathlib import Path

# Add samples/Utilities/ to the import path so we can use the shared bindings helpers.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent / "Utilities"))

try:
    import numpy as np
    from cuda_bindings_utils import KernelHelper, check_cuda_errors, find_cuda_device

    from cuda.bindings import driver as cuda
    from cuda.bindings import runtime as cudart
except ImportError as e:
    print(f"Error: Required package not found: {e}")
    print("Please install from requirements.txt:")
    print("  pip install -r requirements.txt")
    sys.exit(1)


THREADS_PER_BLOCK = 512
GRAPH_LAUNCH_ITERATIONS = 3


REDUCE_KERNEL_SOURCE = """\
#include <cooperative_groups.h>
#include <cuda_runtime.h>

namespace cg = cooperative_groups;

#define THREADS_PER_BLOCK 512
#define GRAPH_LAUNCH_ITERATIONS 3

extern "C"
__global__ void reduce(float *inputVec, double *outputVec, size_t inputSize,
                       size_t outputSize) {
    __shared__ double tmp[THREADS_PER_BLOCK];

    cg::thread_block cta = cg::this_thread_block();
    size_t globaltid = blockIdx.x * blockDim.x + threadIdx.x;

    double temp_sum = 0.0;
    for (int i = globaltid; i < inputSize; i += gridDim.x * blockDim.x) {
        temp_sum += (double)inputVec[i];
    }
    tmp[cta.thread_rank()] = temp_sum;

    cg::sync(cta);

    cg::thread_block_tile<32> tile32 = cg::tiled_partition<32>(cta);

    double beta = temp_sum;
    double temp;

    for (int i = tile32.size() / 2; i > 0; i >>= 1) {
        if (tile32.thread_rank() < i) {
            temp = tmp[cta.thread_rank() + i];
            beta += temp;
            tmp[cta.thread_rank()] = beta;
        }
        cg::sync(tile32);
    }
    cg::sync(cta);

    if (cta.thread_rank() == 0 && blockIdx.x < outputSize) {
        beta = 0.0;
        for (int i = 0; i < cta.size(); i += tile32.size()) {
            beta += tmp[i];
        }
        outputVec[blockIdx.x] = beta;
    }
}

extern "C"
__global__ void reduceFinal(double *inputVec, double *result,
                            size_t inputSize) {
    __shared__ double tmp[THREADS_PER_BLOCK];

    cg::thread_block cta = cg::this_thread_block();
    size_t globaltid = blockIdx.x * blockDim.x + threadIdx.x;

    double temp_sum = 0.0;
    for (int i = globaltid; i < inputSize; i += gridDim.x * blockDim.x) {
        temp_sum += (double)inputVec[i];
    }
    tmp[cta.thread_rank()] = temp_sum;

    cg::sync(cta);

    cg::thread_block_tile<32> tile32 = cg::tiled_partition<32>(cta);

    // do reduction in shared mem
    if ((blockDim.x >= 512) && (cta.thread_rank() < 256)) {
        tmp[cta.thread_rank()] = temp_sum = temp_sum + tmp[cta.thread_rank() + 256];
    }
    cg::sync(cta);

    if ((blockDim.x >= 256) && (cta.thread_rank() < 128)) {
        tmp[cta.thread_rank()] = temp_sum = temp_sum + tmp[cta.thread_rank() + 128];
    }
    cg::sync(cta);

    if ((blockDim.x >= 128) && (cta.thread_rank() < 64)) {
        tmp[cta.thread_rank()] = temp_sum = temp_sum + tmp[cta.thread_rank() + 64];
    }
    cg::sync(cta);

    if (cta.thread_rank() < 32) {
          if (blockDim.x >= 64) temp_sum += tmp[cta.thread_rank() + 32];
          for (int offset = tile32.size() / 2; offset > 0; offset /= 2) {
                temp_sum += tile32.shfl_down(temp_sum, offset);
          }
    }
    if (cta.thread_rank() == 0) result[0] = temp_sum;
}
"""


# Kernels are looked up at main() startup and captured here so the graph
# construction helpers can reference them.
_reduce = None
_reduce_final = None


def _init_input(a, size):
    ctypes.c_float.from_address(a)
    a_list = ctypes.pointer(ctypes.c_float.from_address(a))
    for i in range(size):
        a_list[i] = rnd.random()


def cuda_graphs_manual(input_vec_h, input_vec_d, output_vec_d, result_d, input_size, num_of_blocks):
    """Build the graph node by node with explicit dependency arrays."""
    result_h = ctypes.c_double(0.0)
    node_dependencies = []

    stream_for_graph = check_cuda_errors(cudart.cudaStreamCreate())

    memcpy_params = cudart.cudaMemcpy3DParms()
    memset_params = cudart.cudaMemsetParams()

    # H2D memcpy node: input_vec_h -> input_vec_d
    memcpy_params.srcArray = None
    memcpy_params.srcPos = cudart.make_cudaPos(0, 0, 0)
    memcpy_params.srcPtr = cudart.make_cudaPitchedPtr(
        input_vec_h, np.dtype(np.float32).itemsize * input_size, input_size, 1
    )
    memcpy_params.dstArray = None
    memcpy_params.dstPos = cudart.make_cudaPos(0, 0, 0)
    memcpy_params.dstPtr = cudart.make_cudaPitchedPtr(
        input_vec_d, np.dtype(np.float32).itemsize * input_size, input_size, 1
    )
    memcpy_params.extent = cudart.make_cudaExtent(np.dtype(np.float32).itemsize * input_size, 1, 1)
    memcpy_params.kind = cudart.cudaMemcpyKind.cudaMemcpyHostToDevice

    # Memset node: zero output_vec_d
    memset_params.dst = output_vec_d
    memset_params.value = 0
    memset_params.pitch = 0
    memset_params.elementSize = np.dtype(np.float32).itemsize  # max 4 bytes
    memset_params.width = num_of_blocks * 2
    memset_params.height = 1

    graph = check_cuda_errors(cudart.cudaGraphCreate(0))

    memcpy_node = check_cuda_errors(cudart.cudaGraphAddMemcpyNode(graph, None, 0, memcpy_params))
    memset_node = check_cuda_errors(cudart.cudaGraphAddMemsetNode(graph, None, 0, memset_params))
    node_dependencies.append(memset_node)
    node_dependencies.append(memcpy_node)

    # First reduce kernel node: input_vec_d -> output_vec_d
    kernel_node_params = cuda.CUDA_KERNEL_NODE_PARAMS()
    kernel_node_params.func = _reduce
    kernel_node_params.gridDimX = num_of_blocks
    kernel_node_params.gridDimY = kernel_node_params.gridDimZ = 1
    kernel_node_params.blockDimX = THREADS_PER_BLOCK
    kernel_node_params.blockDimY = kernel_node_params.blockDimZ = 1
    kernel_node_params.sharedMemBytes = 0
    kernel_node_params.kernelParams = (
        (input_vec_d, output_vec_d, input_size, num_of_blocks),
        (ctypes.c_void_p, ctypes.c_void_p, ctypes.c_size_t, ctypes.c_uint),
    )
    kernel_node = check_cuda_errors(
        cuda.cuGraphAddKernelNode(graph, node_dependencies, len(node_dependencies), kernel_node_params)
    )
    node_dependencies.clear()
    node_dependencies.append(kernel_node)

    # Zero result_d
    memset_params = cudart.cudaMemsetParams()
    memset_params.dst = result_d
    memset_params.value = 0
    memset_params.elementSize = np.dtype(np.float32).itemsize
    memset_params.width = 2
    memset_params.height = 1
    memset_node = check_cuda_errors(cudart.cudaGraphAddMemsetNode(graph, None, 0, memset_params))
    node_dependencies.append(memset_node)

    # Second (final) reduce kernel node: output_vec_d -> result_d
    kernel_node_params = cuda.CUDA_KERNEL_NODE_PARAMS()
    kernel_node_params.func = _reduce_final
    kernel_node_params.gridDimX = kernel_node_params.gridDimY = kernel_node_params.gridDimZ = 1
    kernel_node_params.blockDimX = THREADS_PER_BLOCK
    kernel_node_params.blockDimY = kernel_node_params.blockDimZ = 1
    kernel_node_params.sharedMemBytes = 0
    kernel_node_params.kernelParams = (
        (output_vec_d, result_d, num_of_blocks),
        (ctypes.c_void_p, ctypes.c_void_p, ctypes.c_uint),
    )
    kernel_node = check_cuda_errors(
        cuda.cuGraphAddKernelNode(graph, node_dependencies, len(node_dependencies), kernel_node_params)
    )
    node_dependencies.clear()
    node_dependencies.append(kernel_node)

    # D2H memcpy node: result_d -> result_h
    memcpy_params = cudart.cudaMemcpy3DParms()
    memcpy_params.srcArray = None
    memcpy_params.srcPos = cudart.make_cudaPos(0, 0, 0)
    memcpy_params.srcPtr = cudart.make_cudaPitchedPtr(result_d, np.dtype(np.float64).itemsize, 1, 1)
    memcpy_params.dstArray = None
    memcpy_params.dstPos = cudart.make_cudaPos(0, 0, 0)
    memcpy_params.dstPtr = cudart.make_cudaPitchedPtr(result_h, np.dtype(np.float64).itemsize, 1, 1)
    memcpy_params.extent = cudart.make_cudaExtent(np.dtype(np.float64).itemsize, 1, 1)
    memcpy_params.kind = cudart.cudaMemcpyKind.cudaMemcpyDeviceToHost
    memcpy_node = check_cuda_errors(
        cudart.cudaGraphAddMemcpyNode(graph, node_dependencies, len(node_dependencies), memcpy_params)
    )
    node_dependencies.clear()
    node_dependencies.append(memcpy_node)

    _, num_nodes = check_cuda_errors(cudart.cudaGraphGetNodes(graph))
    print(f"\nNum of nodes in the graph created manually = {num_nodes}")

    graph_exec = check_cuda_errors(cudart.cudaGraphInstantiate(graph, 0))
    cloned_graph = check_cuda_errors(cudart.cudaGraphClone(graph))
    cloned_graph_exec = check_cuda_errors(cudart.cudaGraphInstantiate(cloned_graph, 0))

    for _ in range(GRAPH_LAUNCH_ITERATIONS):
        check_cuda_errors(cudart.cudaGraphLaunch(graph_exec, stream_for_graph))
    check_cuda_errors(cudart.cudaStreamSynchronize(stream_for_graph))

    print("Cloned Graph Output..")
    for _ in range(GRAPH_LAUNCH_ITERATIONS):
        check_cuda_errors(cudart.cudaGraphLaunch(cloned_graph_exec, stream_for_graph))
    check_cuda_errors(cudart.cudaStreamSynchronize(stream_for_graph))

    check_cuda_errors(cudart.cudaGraphExecDestroy(graph_exec))
    check_cuda_errors(cudart.cudaGraphExecDestroy(cloned_graph_exec))
    check_cuda_errors(cudart.cudaGraphDestroy(graph))
    check_cuda_errors(cudart.cudaGraphDestroy(cloned_graph))
    check_cuda_errors(cudart.cudaStreamDestroy(stream_for_graph))


def cuda_graphs_using_stream_capture(input_vec_h, input_vec_d, output_vec_d, result_d, input_size, num_of_blocks):
    """Capture the same DAG by recording actual launches on three streams."""
    result_h = ctypes.c_double(0.0)

    stream1 = check_cuda_errors(cudart.cudaStreamCreate())
    stream2 = check_cuda_errors(cudart.cudaStreamCreate())
    stream3 = check_cuda_errors(cudart.cudaStreamCreate())
    stream_for_graph = check_cuda_errors(cudart.cudaStreamCreate())

    fork_event = check_cuda_errors(cudart.cudaEventCreate())
    memset_event1 = check_cuda_errors(cudart.cudaEventCreate())
    memset_event2 = check_cuda_errors(cudart.cudaEventCreate())

    check_cuda_errors(cudart.cudaStreamBeginCapture(stream1, cudart.cudaStreamCaptureMode.cudaStreamCaptureModeGlobal))

    # Fork stream1 into stream2 and stream3.
    check_cuda_errors(cudart.cudaEventRecord(fork_event, stream1))
    check_cuda_errors(cudart.cudaStreamWaitEvent(stream2, fork_event, 0))
    check_cuda_errors(cudart.cudaStreamWaitEvent(stream3, fork_event, 0))

    # H2D on stream1; two zeroing memsets in parallel on stream2 / stream3.
    check_cuda_errors(
        cudart.cudaMemcpyAsync(
            input_vec_d,
            input_vec_h,
            np.dtype(np.float32).itemsize * input_size,
            cudart.cudaMemcpyKind.cudaMemcpyDefault,
            stream1,
        )
    )
    check_cuda_errors(cudart.cudaMemsetAsync(output_vec_d, 0, np.dtype(np.float64).itemsize * num_of_blocks, stream2))
    check_cuda_errors(cudart.cudaEventRecord(memset_event1, stream2))
    check_cuda_errors(cudart.cudaMemsetAsync(result_d, 0, np.dtype(np.float64).itemsize, stream3))
    check_cuda_errors(cudart.cudaEventRecord(memset_event2, stream3))

    check_cuda_errors(cudart.cudaStreamWaitEvent(stream1, memset_event1, 0))

    # First reduce.
    kernel_args = (
        (input_vec_d, output_vec_d, input_size, num_of_blocks),
        (ctypes.c_void_p, ctypes.c_void_p, ctypes.c_size_t, ctypes.c_uint),
    )
    check_cuda_errors(
        cuda.cuLaunchKernel(_reduce, num_of_blocks, 1, 1, THREADS_PER_BLOCK, 1, 1, 0, stream1, kernel_args, 0)
    )

    check_cuda_errors(cudart.cudaStreamWaitEvent(stream1, memset_event2, 0))

    # Final reduce.
    kernel_args2 = (
        (output_vec_d, result_d, num_of_blocks),
        (ctypes.c_void_p, ctypes.c_void_p, ctypes.c_uint),
    )
    check_cuda_errors(cuda.cuLaunchKernel(_reduce_final, 1, 1, 1, THREADS_PER_BLOCK, 1, 1, 0, stream1, kernel_args2, 0))

    # D2H copy of the scalar result.
    check_cuda_errors(
        cudart.cudaMemcpyAsync(
            result_h,
            result_d,
            np.dtype(np.float64).itemsize,
            cudart.cudaMemcpyKind.cudaMemcpyDefault,
            stream1,
        )
    )

    graph = check_cuda_errors(cudart.cudaStreamEndCapture(stream1))
    _, num_nodes = check_cuda_errors(cudart.cudaGraphGetNodes(graph))
    print(f"\nNum of nodes in the graph created using stream capture API = {num_nodes}")

    graph_exec = check_cuda_errors(cudart.cudaGraphInstantiate(graph, 0))
    cloned_graph = check_cuda_errors(cudart.cudaGraphClone(graph))
    cloned_graph_exec = check_cuda_errors(cudart.cudaGraphInstantiate(cloned_graph, 0))

    for _ in range(GRAPH_LAUNCH_ITERATIONS):
        check_cuda_errors(cudart.cudaGraphLaunch(graph_exec, stream_for_graph))
    check_cuda_errors(cudart.cudaStreamSynchronize(stream_for_graph))

    print("Cloned Graph Output..")
    for _ in range(GRAPH_LAUNCH_ITERATIONS):
        check_cuda_errors(cudart.cudaGraphLaunch(cloned_graph_exec, stream_for_graph))
    check_cuda_errors(cudart.cudaStreamSynchronize(stream_for_graph))

    check_cuda_errors(cudart.cudaGraphExecDestroy(graph_exec))
    check_cuda_errors(cudart.cudaGraphExecDestroy(cloned_graph_exec))
    check_cuda_errors(cudart.cudaGraphDestroy(graph))
    check_cuda_errors(cudart.cudaGraphDestroy(cloned_graph))
    check_cuda_errors(cudart.cudaStreamDestroy(stream1))
    check_cuda_errors(cudart.cudaStreamDestroy(stream2))
    check_cuda_errors(cudart.cudaStreamDestroy(stream_for_graph))


def main():
    size = 1 << 24
    max_blocks = 512

    dev_id = find_cuda_device()

    global _reduce, _reduce_final
    kernel_helper = KernelHelper(REDUCE_KERNEL_SOURCE, dev_id)
    _reduce = kernel_helper.get_function(b"reduce")
    _reduce_final = kernel_helper.get_function(b"reduceFinal")

    print(f"{size} elements")
    print(f"threads per block  = {THREADS_PER_BLOCK}")
    print(f"Graph Launch iterations = {GRAPH_LAUNCH_ITERATIONS}")

    input_vec_h = check_cuda_errors(cudart.cudaMallocHost(size * np.dtype(np.float32).itemsize))
    input_vec_d = check_cuda_errors(cudart.cudaMalloc(size * np.dtype(np.float32).itemsize))
    output_vec_d = check_cuda_errors(cudart.cudaMalloc(max_blocks * np.dtype(np.float64).itemsize))
    result_d = check_cuda_errors(cudart.cudaMalloc(np.dtype(np.float64).itemsize))

    _init_input(input_vec_h, size)

    cuda_graphs_manual(input_vec_h, input_vec_d, output_vec_d, result_d, size, max_blocks)
    cuda_graphs_using_stream_capture(input_vec_h, input_vec_d, output_vec_d, result_d, size, max_blocks)

    check_cuda_errors(cudart.cudaFree(input_vec_d))
    check_cuda_errors(cudart.cudaFree(output_vec_d))
    check_cuda_errors(cudart.cudaFree(result_d))
    check_cuda_errors(cudart.cudaFreeHost(input_vec_h))

    print("Done")
    return 0


if __name__ == "__main__":
    sys.exit(main())
