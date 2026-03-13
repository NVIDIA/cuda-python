# Copyright 2021-2025 NVIDIA Corporation.  All rights reserved.
# SPDX-License-Identifier: LicenseRef-NVIDIA-SOFTWARE-LICENSE

import ctypes
import random as rnd

import numpy as np
from common import common
from common.helper_cuda import check_cuda_errors, find_cuda_device

from cuda.bindings import driver as cuda
from cuda.bindings import runtime as cudart

THREADS_PER_BLOCK = 512
GRAPH_LAUNCH_ITERATIONS = 3

simple_cuda_graphs = """\
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
          // Fetch final intermediate sum from 2nd warp
          if (blockDim.x >= 64) temp_sum += tmp[cta.thread_rank() + 32];
          // Reduce final warp using shuffle
          for (int offset = tile32.size() / 2; offset > 0; offset /= 2) {
                temp_sum += tile32.shfl_down(temp_sum, offset);
          }
    }
    // write result for this block to global mem
    if (cta.thread_rank() == 0) result[0] = temp_sum;
}
"""


def init_input(a, size):
    ctypes.c_float.from_address(a)
    a_list = ctypes.pointer(ctypes.c_float.from_address(a))
    for i in range(size):
        a_list[i] = rnd.random()


def cuda_graphs_manual(input_vec_h, input_vec_d, output_vec_d, result_d, input_size, num_of_blocks):
    result_h = ctypes.c_double(0.0)
    node_dependencies = []

    stream_for_graph = check_cuda_errors(cudart.cudaStreamCreate())

    kernel_node_params = cuda.CUDA_KERNEL_NODE_PARAMS()
    memcpy_params = cudart.cudaMemcpy3DParms()
    memset_params = cudart.cudaMemsetParams()

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

    memset_params.dst = output_vec_d
    memset_params.value = 0
    memset_params.pitch = 0
    memset_params.elementSize = np.dtype(np.float32).itemsize  # elementSize can be max 4 bytes
    memset_params.width = num_of_blocks * 2
    memset_params.height = 1

    graph = check_cuda_errors(cudart.cudaGraphCreate(0))

    memcpy_node = check_cuda_errors(cudart.cudaGraphAddMemcpyNode(graph, None, 0, memcpy_params))
    memset_node = check_cuda_errors(cudart.cudaGraphAddMemsetNode(graph, None, 0, memset_params))

    node_dependencies.append(memset_node)
    node_dependencies.append(memcpy_node)

    kernel_args = (
        (input_vec_d, output_vec_d, input_size, num_of_blocks),
        (ctypes.c_void_p, ctypes.c_void_p, ctypes.c_size_t, ctypes.c_uint),
    )

    kernel_node_params.func = _reduce
    kernel_node_params.gridDimX = num_of_blocks
    kernel_node_params.gridDimY = kernel_node_params.gridDimZ = 1
    kernel_node_params.blockDimX = THREADS_PER_BLOCK
    kernel_node_params.blockDimY = kernel_node_params.blockDimZ = 1
    kernel_node_params.sharedMemBytes = 0
    kernel_node_params.kernelParams = kernel_args
    # kernelNodeParams.extra = None

    kernel_node = check_cuda_errors(
        cuda.cuGraphAddKernelNode(graph, node_dependencies, len(node_dependencies), kernel_node_params)
    )

    node_dependencies.clear()
    node_dependencies.append(kernel_node)

    memset_params = cudart.cudaMemsetParams()
    memset_params.dst = result_d
    memset_params.value = 0
    memset_params.elementSize = np.dtype(np.float32).itemsize
    memset_params.width = 2
    memset_params.height = 1
    memset_node = check_cuda_errors(cudart.cudaGraphAddMemsetNode(graph, None, 0, memset_params))

    node_dependencies.append(memset_node)

    kernel_node_params = cuda.CUDA_KERNEL_NODE_PARAMS()
    kernel_node_params.func = _reduceFinal
    kernel_node_params.gridDimX = kernel_node_params.gridDimY = kernel_node_params.gridDimZ = 1
    kernel_node_params.blockDimX = THREADS_PER_BLOCK
    kernel_node_params.blockDimY = kernel_node_params.blockDimZ = 1
    kernel_node_params.sharedMemBytes = 0
    kernel_args2 = (
        (output_vec_d, result_d, num_of_blocks),
        (ctypes.c_void_p, ctypes.c_void_p, ctypes.c_uint),
    )
    kernel_node_params.kernelParams = kernel_args2
    # kernelNodeParams.extra = None

    kernel_node = check_cuda_errors(
        cuda.cuGraphAddKernelNode(graph, node_dependencies, len(node_dependencies), kernel_node_params)
    )

    node_dependencies.clear()
    node_dependencies.append(kernel_node)

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

    # WIP: Host nodes

    nodes, num_nodes = check_cuda_errors(cudart.cudaGraphGetNodes(graph))
    print(f"\nNum of nodes in the graph created manually = {num_nodes}")

    graph_exec = check_cuda_errors(cudart.cudaGraphInstantiate(graph, 0))

    cloned_graph = check_cuda_errors(cudart.cudaGraphClone(graph))
    cloned_graph_exec = check_cuda_errors(cudart.cudaGraphInstantiate(cloned_graph, 0))

    for _i in range(GRAPH_LAUNCH_ITERATIONS):
        check_cuda_errors(cudart.cudaGraphLaunch(graph_exec, stream_for_graph))

    check_cuda_errors(cudart.cudaStreamSynchronize(stream_for_graph))

    print("Cloned Graph Output..")
    for _i in range(GRAPH_LAUNCH_ITERATIONS):
        check_cuda_errors(cudart.cudaGraphLaunch(cloned_graph_exec, stream_for_graph))

    check_cuda_errors(cudart.cudaStreamSynchronize(stream_for_graph))

    check_cuda_errors(cudart.cudaGraphExecDestroy(graph_exec))
    check_cuda_errors(cudart.cudaGraphExecDestroy(cloned_graph_exec))
    check_cuda_errors(cudart.cudaGraphDestroy(graph))
    check_cuda_errors(cudart.cudaGraphDestroy(cloned_graph))
    check_cuda_errors(cudart.cudaStreamDestroy(stream_for_graph))


def cuda_graphs_using_stream_capture(input_vec_h, input_vec_d, output_vec_d, result_d, input_size, num_of_blocks):
    result_h = ctypes.c_double(0.0)

    stream1 = check_cuda_errors(cudart.cudaStreamCreate())
    stream2 = check_cuda_errors(cudart.cudaStreamCreate())
    stream3 = check_cuda_errors(cudart.cudaStreamCreate())
    stream_for_graph = check_cuda_errors(cudart.cudaStreamCreate())

    fork_stream_event = check_cuda_errors(cudart.cudaEventCreate())
    memset_event1 = check_cuda_errors(cudart.cudaEventCreate())
    memset_event2 = check_cuda_errors(cudart.cudaEventCreate())

    check_cuda_errors(cudart.cudaStreamBeginCapture(stream1, cudart.cudaStreamCaptureMode.cudaStreamCaptureModeGlobal))

    check_cuda_errors(cudart.cudaEventRecord(fork_stream_event, stream1))
    check_cuda_errors(cudart.cudaStreamWaitEvent(stream2, fork_stream_event, 0))
    check_cuda_errors(cudart.cudaStreamWaitEvent(stream3, fork_stream_event, 0))

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

    kernel_args = (
        (input_vec_d, output_vec_d, input_size, num_of_blocks),
        (ctypes.c_void_p, ctypes.c_void_p, ctypes.c_size_t, ctypes.c_uint),
    )
    check_cuda_errors(
        cuda.cuLaunchKernel(
            _reduce,
            num_of_blocks,
            1,
            1,
            THREADS_PER_BLOCK,
            1,
            1,
            0,
            stream1,
            kernel_args,
            0,
        )
    )

    check_cuda_errors(cudart.cudaStreamWaitEvent(stream1, memset_event2, 0))

    kernel_args2 = (
        (output_vec_d, result_d, num_of_blocks),
        (ctypes.c_void_p, ctypes.c_void_p, ctypes.c_uint),
    )
    check_cuda_errors(cuda.cuLaunchKernel(_reduceFinal, 1, 1, 1, THREADS_PER_BLOCK, 1, 1, 0, stream1, kernel_args2, 0))

    check_cuda_errors(
        cudart.cudaMemcpyAsync(
            result_h,
            result_d,
            np.dtype(np.float64).itemsize,
            cudart.cudaMemcpyKind.cudaMemcpyDefault,
            stream1,
        )
    )

    # WIP: Host nodes

    graph = check_cuda_errors(cudart.cudaStreamEndCapture(stream1))

    nodes, num_nodes = check_cuda_errors(cudart.cudaGraphGetNodes(graph))
    print(f"\nNum of nodes in the graph created using stream capture API = {num_nodes}")

    graph_exec = check_cuda_errors(cudart.cudaGraphInstantiate(graph, 0))

    cloned_graph = check_cuda_errors(cudart.cudaGraphClone(graph))
    cloned_graph_exec = check_cuda_errors(cudart.cudaGraphInstantiate(cloned_graph, 0))

    for _i in range(GRAPH_LAUNCH_ITERATIONS):
        check_cuda_errors(cudart.cudaGraphLaunch(graph_exec, stream_for_graph))

    check_cuda_errors(cudart.cudaStreamSynchronize(stream_for_graph))

    print("Cloned Graph Output..")
    for _i in range(GRAPH_LAUNCH_ITERATIONS):
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
    size = 1 << 24  # number of elements to reduce
    max_blocks = 512

    # This will pick the best possible CUDA capable device
    dev_id = find_cuda_device()

    global _reduce
    global _reduceFinal
    kernel_helper = common.KernelHelper(simple_cuda_graphs, dev_id)
    _reduce = kernel_helper.get_function(b"reduce")
    _reduceFinal = kernel_helper.get_function(b"reduceFinal")

    print(f"{size} elements")
    print(f"threads per block  = {THREADS_PER_BLOCK}")
    print(f"Graph Launch iterations = {GRAPH_LAUNCH_ITERATIONS}")

    input_vec_h = check_cuda_errors(cudart.cudaMallocHost(size * np.dtype(np.float32).itemsize))
    input_vec_d = check_cuda_errors(cudart.cudaMalloc(size * np.dtype(np.float32).itemsize))
    output_vec_d = check_cuda_errors(cudart.cudaMalloc(max_blocks * np.dtype(np.float64).itemsize))
    result_d = check_cuda_errors(cudart.cudaMalloc(np.dtype(np.float64).itemsize))

    init_input(input_vec_h, size)

    cuda_graphs_manual(input_vec_h, input_vec_d, output_vec_d, result_d, size, max_blocks)
    cuda_graphs_using_stream_capture(input_vec_h, input_vec_d, output_vec_d, result_d, size, max_blocks)

    check_cuda_errors(cudart.cudaFree(input_vec_d))
    check_cuda_errors(cudart.cudaFree(output_vec_d))
    check_cuda_errors(cudart.cudaFree(result_d))
    check_cuda_errors(cudart.cudaFreeHost(input_vec_h))


if __name__ == "__main__":
    main()
