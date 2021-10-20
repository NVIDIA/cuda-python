# Copyright 2021 NVIDIA Corporation.  All rights reserved.
#
# Please refer to the NVIDIA end user license agreement (EULA) associated
# with this source code for terms and conditions that govern your use of
# this software. Any use, reproduction, disclosure, or distribution of
# this software and related documentation outside the terms of the EULA
# is strictly prohibited.
import ctypes
import numpy as np
import random as rnd
from cuda import cuda, cudart
from examples.common import common
from examples.common.helper_cuda import checkCudaErrors, findCudaDevice

THREADS_PER_BLOCK = 512
GRAPH_LAUNCH_ITERATIONS = 3

simpleCudaGraphs = '''\
#include <cooperative_groups.h>
#include <cuda_runtime.h>

namespace cg = cooperative_groups;

#define THREADS_PER_BLOCK 512
#define GRAPH_LAUNCH_ITERATIONS 3

extern "C"
__global__ void reduce(float *inputVec, double *outputVec, size_t inputSize,
                       unsigned int outputSize) {
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
                            unsigned int inputSize) {
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
'''

def init_input(a, size):
    ctypes.c_float.from_address(a)
    a_list = ctypes.pointer(ctypes.c_float.from_address(a))
    for i in range(0, size):
        a_list[i] = rnd.random()

def cudaGraphsManual(inputVec_h, inputVec_d, outputVec_d, result_d, inputSize, numOfBlocks):
    result_h = ctypes.c_double(0.0)
    nodeDependencies = []

    streamForGraph = checkCudaErrors(cudart.cudaStreamCreateWithFlags(cudart.cudaStreamNonBlocking))

    kernelNodeParams = cuda.CUDA_KERNEL_NODE_PARAMS()
    memcpyParams = cudart.cudaMemcpy3DParms()
    memsetParams = cudart.cudaMemsetParams()

    memcpyParams.srcArray = None
    memcpyParams.srcPos = cudart.make_cudaPos(0, 0, 0)
    memcpyParams.srcPtr = cudart.make_cudaPitchedPtr(inputVec_h, np.dtype(np.float32).itemsize * inputSize, inputSize, 1)
    memcpyParams.dstArray = None
    memcpyParams.dstPos = cudart.make_cudaPos(0, 0, 0)
    memcpyParams.dstPtr = cudart.make_cudaPitchedPtr(inputVec_d, np.dtype(np.float32).itemsize * inputSize, inputSize, 1)
    memcpyParams.extent = cudart.make_cudaExtent(np.dtype(np.float32).itemsize * inputSize, 1, 1)
    memcpyParams.kind = cudart.cudaMemcpyKind.cudaMemcpyHostToDevice

    memsetParams.dst = outputVec_d
    memsetParams.value = 0
    memsetParams.pitch = 0
    memsetParams.elementSize = np.dtype(np.float32).itemsize # elementSize can be max 4 bytes
    memsetParams.width = numOfBlocks * 2
    memsetParams.height = 1

    graph = checkCudaErrors(cudart.cudaGraphCreate(0))

    memcpyNode = checkCudaErrors(cudart.cudaGraphAddMemcpyNode(graph, None, 0, memcpyParams))
    memsetNode = checkCudaErrors(cudart.cudaGraphAddMemsetNode(graph, None, 0, memsetParams))

    nodeDependencies.append(memsetNode)
    nodeDependencies.append(memcpyNode)

    kernelArgs = ((inputVec_d, outputVec_d, inputSize, numOfBlocks),
                  (ctypes.c_void_p, ctypes.c_void_p, ctypes.c_size_t, ctypes.c_uint))

    kernelNodeParams.func = _reduce
    kernelNodeParams.gridDimX = numOfBlocks
    kernelNodeParams.gridDimY = kernelNodeParams.gridDimZ = 1
    kernelNodeParams.blockDimX = THREADS_PER_BLOCK
    kernelNodeParams.blockDimY = kernelNodeParams.blockDimZ = 1
    kernelNodeParams.sharedMemBytes = 0
    kernelNodeParams.kernelParams = kernelArgs
    # kernelNodeParams.extra = None

    kernelNode = checkCudaErrors(cuda.cuGraphAddKernelNode(graph, nodeDependencies, len(nodeDependencies), kernelNodeParams))

    nodeDependencies.clear()
    nodeDependencies.append(kernelNode)

    memsetParams = cudart.cudaMemsetParams()
    memsetParams.dst = result_d
    memsetParams.value = 0
    memsetParams.elementSize = np.dtype(np.float32).itemsize
    memsetParams.width = 2
    memsetParams.height = 1
    memsetNode = checkCudaErrors(cudart.cudaGraphAddMemsetNode(graph, None, 0, memsetParams))

    nodeDependencies.append(memsetNode)

    kernelNodeParams = cuda.CUDA_KERNEL_NODE_PARAMS()
    kernelNodeParams.func = _reduceFinal
    kernelNodeParams.gridDimX = kernelNodeParams.gridDimY = kernelNodeParams.gridDimZ = 1
    kernelNodeParams.blockDimX = THREADS_PER_BLOCK
    kernelNodeParams.blockDimY = kernelNodeParams.blockDimZ = 1
    kernelNodeParams.sharedMemBytes = 0
    kernelArgs2 = ((outputVec_d, result_d, numOfBlocks),
                   (ctypes.c_void_p, ctypes.c_void_p, ctypes.c_uint))
    kernelNodeParams.kernelParams = kernelArgs2
    # kernelNodeParams.extra = None

    kernelNode = checkCudaErrors(cuda.cuGraphAddKernelNode(graph, nodeDependencies, len(nodeDependencies), kernelNodeParams))

    nodeDependencies.clear()
    nodeDependencies.append(kernelNode)

    memcpyParams = cudart.cudaMemcpy3DParms()

    memcpyParams.srcArray = None
    memcpyParams.srcPos = cudart.make_cudaPos(0, 0, 0)
    memcpyParams.srcPtr = cudart.make_cudaPitchedPtr(result_d, np.dtype(np.float64).itemsize, 1, 1)
    memcpyParams.dstArray = None
    memcpyParams.dstPos = cudart.make_cudaPos(0, 0, 0)
    memcpyParams.dstPtr = cudart.make_cudaPitchedPtr(result_h, np.dtype(np.float64).itemsize, 1, 1)
    memcpyParams.extent = cudart.make_cudaExtent(np.dtype(np.float64).itemsize, 1, 1)
    memcpyParams.kind = cudart.cudaMemcpyKind.cudaMemcpyDeviceToHost
    memcpyNode = checkCudaErrors(cudart.cudaGraphAddMemcpyNode(graph, nodeDependencies, len(nodeDependencies), memcpyParams))

    nodeDependencies.clear()
    nodeDependencies.append(memcpyNode)

    nodes, numNodes = checkCudaErrors(cudart.cudaGraphGetNodes(graph))
    print("\nNum of nodes in the graph created manually = {}".format(numNodes))

    graphExec, errorNode = checkCudaErrors(cudart.cudaGraphInstantiate(graph, b'', 0))

    clonedGraph = checkCudaErrors(cudart.cudaGraphClone(graph))
    clonedGraphExec, errorNode = checkCudaErrors(cudart.cudaGraphInstantiate(clonedGraph, b'', 0))

    for i in range(GRAPH_LAUNCH_ITERATIONS):
        checkCudaErrors(cudart.cudaGraphLaunch(graphExec, streamForGraph))
        checkCudaErrors(cudart.cudaStreamSynchronize(streamForGraph))
        print("[cudaGraphsManual] final reduced sum = {}".format(result_h.value))
        result_h.value = 0.0

    print("Cloned Graph Output..")
    for i in range(GRAPH_LAUNCH_ITERATIONS):
        checkCudaErrors(cudart.cudaGraphLaunch(clonedGraphExec, streamForGraph))
        checkCudaErrors(cudart.cudaStreamSynchronize(streamForGraph))
        print("[cudaGraphsManual] final reduced sum = {}".format(result_h.value))
        result_h.value = 0.0

    checkCudaErrors(cudart.cudaGraphExecDestroy(graphExec))
    checkCudaErrors(cudart.cudaGraphExecDestroy(clonedGraphExec))
    checkCudaErrors(cudart.cudaGraphDestroy(graph))
    checkCudaErrors(cudart.cudaGraphDestroy(clonedGraph))
    checkCudaErrors(cudart.cudaStreamDestroy(streamForGraph))

def cudaGraphsUsingStreamCapture(inputVec_h, inputVec_d, outputVec_d, result_d, inputSize, numOfBlocks):
    result_h = ctypes.c_double(0.0)

    stream1 = checkCudaErrors(cudart.cudaStreamCreateWithFlags(cudart.cudaStreamNonBlocking))
    stream2 = checkCudaErrors(cudart.cudaStreamCreateWithFlags(cudart.cudaStreamNonBlocking))
    stream3 = checkCudaErrors(cudart.cudaStreamCreateWithFlags(cudart.cudaStreamNonBlocking))
    streamForGraph = checkCudaErrors(cudart.cudaStreamCreateWithFlags(cudart.cudaStreamNonBlocking))

    forkStreamEvent = checkCudaErrors(cudart.cudaEventCreate())
    memsetEvent1 = checkCudaErrors(cudart.cudaEventCreate())
    memsetEvent2 = checkCudaErrors(cudart.cudaEventCreate())

    checkCudaErrors(cudart.cudaStreamBeginCapture(stream1, cudart.cudaStreamCaptureMode.cudaStreamCaptureModeGlobal))

    checkCudaErrors(cudart.cudaEventRecord(forkStreamEvent, stream1))
    checkCudaErrors(cudart.cudaStreamWaitEvent(stream2, forkStreamEvent, 0))
    checkCudaErrors(cudart.cudaStreamWaitEvent(stream3, forkStreamEvent, 0))

    checkCudaErrors(cudart.cudaMemcpyAsync(inputVec_d, inputVec_h,
                                           np.dtype(np.float32).itemsize * inputSize, cudart.cudaMemcpyKind.cudaMemcpyDefault,
                                           stream1))

    checkCudaErrors(cudart.cudaMemsetAsync(outputVec_d, 0, np.dtype(np.float64).itemsize * numOfBlocks, stream2))

    checkCudaErrors(cudart.cudaEventRecord(memsetEvent1, stream2))

    checkCudaErrors(cudart.cudaMemsetAsync(result_d, 0, np.dtype(np.float64).itemsize, stream3))
    checkCudaErrors(cudart.cudaEventRecord(memsetEvent2, stream3))

    checkCudaErrors(cudart.cudaStreamWaitEvent(stream1, memsetEvent1, 0))

    kernelArgs = ((inputVec_d, outputVec_d, inputSize, numOfBlocks),
                  (ctypes.c_void_p, ctypes.c_void_p, ctypes.c_size_t, ctypes.c_uint))
    checkCudaErrors(cuda.cuLaunchKernel(_reduce,
                                        numOfBlocks, 1, 1,
                                        THREADS_PER_BLOCK, 1, 1,
                                        0, stream1,
                                        kernelArgs, 0))

    checkCudaErrors(cudart.cudaStreamWaitEvent(stream1, memsetEvent2, 0))

    kernelArgs2 = ((outputVec_d, result_d, numOfBlocks),
                   (ctypes.c_void_p, ctypes.c_void_p, ctypes.c_uint))
    checkCudaErrors(cuda.cuLaunchKernel(_reduceFinal,
                                        1, 1, 1,
                                        THREADS_PER_BLOCK, 1, 1,
                                        0, stream1,
                                        kernelArgs2, 0))

    checkCudaErrors(cudart.cudaMemcpyAsync(result_h, result_d, np.dtype(np.float64).itemsize,
                                           cudart.cudaMemcpyKind.cudaMemcpyDefault, stream1))

    graph = checkCudaErrors(cudart.cudaStreamEndCapture(stream1))

    nodes, numNodes = checkCudaErrors(cudart.cudaGraphGetNodes(graph))
    print("\nNum of nodes in the graph created using stream capture API = {}".format(numNodes))

    graphExec, errorNode = checkCudaErrors(cudart.cudaGraphInstantiate(graph, b'', 0))

    clonedGraph = checkCudaErrors(cudart.cudaGraphClone(graph))
    clonedGraphExec, errorNode = checkCudaErrors(cudart.cudaGraphInstantiate(clonedGraph, b'', 0))

    for i in range(GRAPH_LAUNCH_ITERATIONS):
        checkCudaErrors(cudart.cudaGraphLaunch(graphExec, streamForGraph))
        checkCudaErrors(cudart.cudaStreamSynchronize(streamForGraph))
        print("[cudaGraphsUsingStreamCapture] final reduced sum = {}".format(result_h.value))
        result_h.value = 0.0

    print("Cloned Graph Output..")
    for i in range(GRAPH_LAUNCH_ITERATIONS):
        checkCudaErrors(cudart.cudaGraphLaunch(clonedGraphExec, streamForGraph))
        checkCudaErrors(cudart.cudaStreamSynchronize(streamForGraph))
        print("[cudaGraphsUsingStreamCapture] final reduced sum = {}".format(result_h.value))
        result_h.value = 0.0

    checkCudaErrors(cudart.cudaStreamSynchronize(streamForGraph))

    checkCudaErrors(cudart.cudaGraphExecDestroy(graphExec))
    checkCudaErrors(cudart.cudaGraphExecDestroy(clonedGraphExec))
    checkCudaErrors(cudart.cudaGraphDestroy(graph))
    checkCudaErrors(cudart.cudaGraphDestroy(clonedGraph))
    checkCudaErrors(cudart.cudaStreamDestroy(stream1))
    checkCudaErrors(cudart.cudaStreamDestroy(stream2))
    checkCudaErrors(cudart.cudaStreamDestroy(streamForGraph))

def main():
    size = 1 << 24 # number of elements to reduce
    maxBlocks = 512

    # This will pick the best possible CUDA capable device
    devID = findCudaDevice()

    global _reduce
    global _reduceFinal
    kernelHelper = common.KernelHelper(simpleCudaGraphs, devID)
    _reduce = kernelHelper.getFunction(b'reduce')
    _reduceFinal = kernelHelper.getFunction(b'reduceFinal')

    print("{} elements".format(size))
    print("threads per block  = {}".format(THREADS_PER_BLOCK))
    print("Graph Launch iterations = {}".format(GRAPH_LAUNCH_ITERATIONS))

    inputVec_h = checkCudaErrors(cudart.cudaMallocHost(size * np.dtype(np.float32).itemsize))
    inputVec_d = checkCudaErrors(cudart.cudaMalloc(size * np.dtype(np.float32).itemsize))
    outputVec_d = checkCudaErrors(cudart.cudaMalloc(maxBlocks * np.dtype(np.float64).itemsize))
    result_d = checkCudaErrors(cudart.cudaMalloc(np.dtype(np.float64).itemsize))

    init_input(inputVec_h, size)

    cudaGraphsManual(inputVec_h, inputVec_d, outputVec_d, result_d, size, maxBlocks)
    cudaGraphsUsingStreamCapture(inputVec_h, inputVec_d, outputVec_d, result_d, size, maxBlocks)

    checkCudaErrors(cudart.cudaFree(inputVec_d))
    checkCudaErrors(cudart.cudaFree(outputVec_d))
    checkCudaErrors(cudart.cudaFree(result_d))
    checkCudaErrors(cudart.cudaFreeHost(inputVec_h))

if __name__ == "__main__":
    main()
