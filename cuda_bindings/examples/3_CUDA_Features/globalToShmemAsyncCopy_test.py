# Copyright 2021-2025 NVIDIA Corporation.  All rights reserved.
# SPDX-License-Identifier: LicenseRef-NVIDIA-SOFTWARE-LICENSE

import ctypes
import math
import platform
import sys
from enum import Enum

import numpy as np
from common import common
from common.helper_cuda import checkCudaErrors, findCudaDevice
from common.helper_string import checkCmdLineFlag, getCmdLineArgumentInt
from cuda.bindings import driver as cuda
from cuda.bindings import runtime as cudart

blockSize = 16


class kernels(Enum):
    AsyncCopyMultiStageLargeChunk = 0
    AsyncCopyLargeChunk = 1
    AsyncCopyLargeChunkAWBarrier = 2
    AsyncCopyMultiStageSharedState = 3
    AsyncCopyMultiStage = 4
    AsyncCopySingleStage = 5
    Naive = 6
    NaiveLargeChunk = 7


kernelNames = [
    "AsyncCopyMultiStageLargeChunk",
    "AsyncCopyLargeChunk",
    "AsyncCopyLargeChunkAWBarrier",
    "AsyncCopyMultiStageSharedState",
    "AsyncCopyMultiStage",
    "AsyncCopySingleStage",
    "Naive",
    "NaiveLargeChunk",
]

globalToShmemAsyncCopy = """\
#line __LINE__
#if __CUDA_ARCH__ >= 700
#include <cuda/barrier>
#endif
#include <cooperative_groups.h>
#include <cooperative_groups/reduce.h>
#include <cuda/pipeline>
namespace cg = cooperative_groups;

#define BLOCK_SIZE 16
#define BLOCK_SIZE_X 16

// Multi Stage memcpy_async pipeline with large chunk copy
extern "C"
__global__ void MatrixMulAsyncCopyMultiStageLargeChunk(float* __restrict__ C,
                                                       const float* __restrict__ A,
                                                       const float* __restrict__ B, int wA,
                                                       int wB) {
    // Requires BLOCK_SIZE % 4 == 0

    // Multi-stage pipeline version
    constexpr size_t maxPipelineStages = 4;

    // Declaration of the shared memory array As used to
    // store the sub-matrix of A for each stage
    __shared__ alignas(alignof(float4)) float As[maxPipelineStages][BLOCK_SIZE][BLOCK_SIZE];

    // Declaration of the shared memory array Bs used to
    // store the sub-matrix of B for each stage
    __shared__ alignas(alignof(float4)) float Bs[maxPipelineStages][BLOCK_SIZE][BLOCK_SIZE];

    float Csub = 0.0;

    // Index of the first sub-matrix of A processed by the block
    const int aBegin = wA * (BLOCK_SIZE) * blockIdx.y;

    // Index of the last sub-matrix of A processed by the block
    const int aEnd   = aBegin + wA - 1;

    // Step size used to iterate through the sub-matrices of A
    int aStep  = BLOCK_SIZE;

    // Index of the first sub-matrix of B processed by the block
    const int bBegin = BLOCK_SIZE * blockIdx.x;

    // Step size used to iterate through the sub-matrices of B
    int bStep  = BLOCK_SIZE * wB;

    const int t4x = threadIdx.x * 4;
    const auto shape4 = cuda::aligned_size_t<alignof(float4)>(sizeof(float4));

    cuda::pipeline<cuda::thread_scope_thread> pipe = cuda::make_pipeline();

    // Loop over all the sub-matrices of A and B
    // required to compute the block sub-matrix
    for (int a = aBegin, b = bBegin, i = 0, aStage = aBegin, bStage = bBegin, iStage = 0; a <= aEnd; a += aStep, b += bStep, ++i ) {
        // Load the matrices from device memory to shared memory; each thread loads
        // one element of each matrix
        for ( ; aStage <= a + aStep * maxPipelineStages ; aStage += aStep, bStage += bStep, ++iStage )
        {
            pipe.producer_acquire();
            if ( aStage <= aEnd && t4x < BLOCK_SIZE )
            {
                // Rotating buffer
                const int j = iStage % maxPipelineStages;
                cuda::memcpy_async(&As[j][threadIdx.y][t4x], &A[aStage + wA * threadIdx.y + t4x], shape4, pipe);
                cuda::memcpy_async(&Bs[j][threadIdx.y][t4x], &B[aStage + wA * threadIdx.y + t4x], shape4, pipe);
            }
            pipe.producer_commit();
        }

        pipe.consumer_wait();
        // Synchronize to make sure the matrices are loaded
        __syncthreads();

        // Rotating buffer
        const int j = i % maxPipelineStages;

        // Multiply the two matrices together;
        // each thread computes one element
        // of the block sub-matrix
        #pragma unroll
        for (int k = 0; k < BLOCK_SIZE; ++k) {
            Csub += As[j][threadIdx.y][k] * Bs[j][k][threadIdx.x];
        }
        pipe.consumer_release();

        // Don't have to synchronize because maxPipelineStages is greater than one
        // therefore next iteration is loading to a different buffer.
    }

    // Write the block sub-matrix to device memory;
    // each thread writes four element
    int c = wB * BLOCK_SIZE * blockIdx.y + BLOCK_SIZE * blockIdx.x;
    C[c + wB * threadIdx.y + threadIdx.x] = Csub;
}

// Single Stage memcpy_async pipeline with Large copy chunk (float4)
extern "C"
__global__ void MatrixMulAsyncCopyLargeChunk(float* __restrict__ C,
                                                        const float* __restrict__ A,
                                                        const float* __restrict__ B, int wA,
                                                        int wB) {
    // Requires BLOCK_SIZE % 4 == 0

    // Declaration of the shared memory array As used to
    // store the sub-matrix of A
    __shared__ alignas(alignof(float4)) float As[BLOCK_SIZE][BLOCK_SIZE];

    // Declaration of the shared memory array Bs used to
    // store the sub-matrix of B
    __shared__ alignas(alignof(float4)) float Bs[BLOCK_SIZE][BLOCK_SIZE];

    // Index of the first sub-matrix of A processed by the block
    int aBegin = wA * BLOCK_SIZE * blockIdx.y;

    // Index of the last sub-matrix of A processed by the block
    int aEnd   = aBegin + wA - 1;

    // Step size used to iterate through the sub-matrices of A
    int aStep  = BLOCK_SIZE;

    // Index of the first sub-matrix of B processed by the block
    int bBegin = BLOCK_SIZE * blockIdx.x;

    // Step size used to iterate through the sub-matrices of B
    int bStep  = BLOCK_SIZE * wB;

    // Single-stage pipeline version
    float Csub = 0.0;

    const int t4x = threadIdx.x * 4;
    const auto shape4 = cuda::aligned_size_t<alignof(float4)>(sizeof(float4));
    cuda::pipeline<cuda::thread_scope_thread> pipe = cuda::make_pipeline();

    // Loop over all the sub-matrices of A and B
    // required to compute the block sub-matrix
    for (int a = aBegin, b = bBegin; a <= aEnd; a += aStep, b += bStep) {
        // Load the matrices from device memory to shared memory;
        // a subset of threads loads a contiguous chunk of elements.

        // Previously, per-thread:
        // As[ty][tx] = A[a + wA * ty + tx];
        // Bs[ty][tx] = B[b + wB * ty + tx];

        // Now, one fourth of the threads load four elements of each matrix
        if ( t4x < BLOCK_SIZE ) {

            pipe.producer_acquire();

            cuda::memcpy_async(&As[threadIdx.y][t4x], &A[a + wA * threadIdx.y + t4x], shape4, pipe);
            cuda::memcpy_async(&Bs[threadIdx.y][t4x], &B[a + wA * threadIdx.y + t4x], shape4, pipe);

            pipe.producer_commit();
            pipe.consumer_wait();
        }

        // Synchronize to make sure the matrices are loaded
        __syncthreads();

        // Multiply the two matrices together;
        // each thread computes one element
        // of the block sub-matrix
#pragma unroll
        for (int k = 0; k < BLOCK_SIZE; ++k) {
            Csub += As[threadIdx.y][k] * Bs[k][threadIdx.x];
        }

        pipe.consumer_release();

        // Synchronize to make sure that the preceding
        // computation is done before overwriting the
        // shared memory sub-matrix buffers As and Bs in the next iteration.
        __syncthreads();
    }

    // Write the block sub-matrix to device memory;
    // each thread writes four element
    int c = wB * BLOCK_SIZE * blockIdx.y + BLOCK_SIZE * blockIdx.x;
    C[c + wB * threadIdx.y + threadIdx.x] = Csub;
}

// Single Stage memcpy_async pipeline with Large copy chunk (float4) using arrive-wait barrier
extern "C"
__global__ void MatrixMulAsyncCopyLargeChunkAWBarrier(float* __restrict__ C,
                                                      const float* __restrict__ A,
                                                      const float* __restrict__ B, int wA,
                                                      int wB) {
#if __CUDA_ARCH__ >= 700
#pragma diag_suppress static_var_with_dynamic_init
    // Requires BLOCK_SIZE % 4 == 0

    __shared__ cuda::barrier<cuda::thread_scope_block> bar;

    // Declaration of the shared memory array As used to
    // store the sub-matrix of A
    __shared__  alignas(alignof(float4)) float As[BLOCK_SIZE][BLOCK_SIZE];

    // Declaration of the shared memory array Bs used to
    // store the sub-matrix of B
    __shared__ alignas(alignof(float4)) float Bs[BLOCK_SIZE][BLOCK_SIZE];

    if (threadIdx.x == 0) {
        init(&bar, blockDim.x*blockDim.y);
    }
    __syncthreads();

    // Index of the first sub-matrix of A processed by the block
    int aBegin = wA * BLOCK_SIZE * blockIdx.y;

    // Index of the last sub-matrix of A processed by the block
    int aEnd   = aBegin + wA - 1;

    // Step size used to iterate through the sub-matrices of A
    int aStep  = BLOCK_SIZE;

    // Index of the first sub-matrix of B processed by the block
    int bBegin = BLOCK_SIZE * blockIdx.x;

    // Step size used to iterate through the sub-matrices of B
    int bStep  = BLOCK_SIZE * wB;

    float Csub = 0.0;

    const int t4x = threadIdx.x * 4;

    // Loop over all the sub-matrices of A and B
    // required to compute the block sub-matrix
    for (int a = aBegin, b = bBegin; a <= aEnd; a += aStep, b += bStep) {
        // Load the matrices from device memory to shared memory;
        // a subset of threads loads a contiguous chunk of elements.

        // Now, one fourth of the threads load four elements of each matrix
        if ( t4x < BLOCK_SIZE ) {
            float4 * const A4s = reinterpret_cast<float4*>(& As[threadIdx.y][t4x]);
            float4 * const B4s = reinterpret_cast<float4*>(& Bs[threadIdx.y][t4x]);
            const float4 * const A4  = reinterpret_cast<const float4*>(& A[a + wA * threadIdx.y + t4x]);
            const float4 * const B4  = reinterpret_cast<const float4*>(& B[a + wA * threadIdx.y + t4x]);

            cuda::memcpy_async(A4s, A4, sizeof(float4), bar);
            cuda::memcpy_async(B4s, B4, sizeof(float4), bar);
         }

        // Synchronize to make sure the matrices are loaded
        bar.arrive_and_wait();

        // Multiply the two matrices together;
        // each thread computes one element
        // of the block sub-matrix
#pragma unroll
        for (int k = 0; k < BLOCK_SIZE; ++k) {
            Csub += As[threadIdx.y][k] * Bs[k][threadIdx.x];
        }

        // Synchronize to make sure that the preceding
        // computation is done before overwriting the
        // shared memory sub-matrix buffers As and Bs in the next iteration.
        bar.arrive_and_wait();
    }

    // Write the block sub-matrix to device memory;
    // each thread writes four element
    int c = wB * BLOCK_SIZE * blockIdx.y + BLOCK_SIZE * blockIdx.x;
    C[c + wB * threadIdx.y + threadIdx.x] = Csub;
#endif
}

// Single Stage memcpy_async pipeline with float copy
extern "C"
 __global__ void MatrixMulAsyncCopySingleStage(float *C, const float *A,
                                                        const float *B, int wA,
                                                        int wB) {

    // Declaration of the shared memory array As used to
    // store the sub-matrix of A
    __shared__ float As[BLOCK_SIZE][BLOCK_SIZE];

    // Declaration of the shared memory array Bs used to
    // store the sub-matrix of B
    __shared__ float Bs[BLOCK_SIZE][BLOCK_SIZE];

    // Index of the first sub-matrix of A processed by the block
    int aBegin = wA * BLOCK_SIZE * blockIdx.y;

    // Index of the last sub-matrix of A processed by the block
    int aEnd   = aBegin + wA - 1;

    // Step size used to iterate through the sub-matrices of A
    int aStep  = BLOCK_SIZE;

    // Index of the first sub-matrix of B processed by the block
    int bBegin = BLOCK_SIZE * blockIdx.x;

    // Step size used to iterate through the sub-matrices of B
    int bStep  = BLOCK_SIZE * wB;

    // Single-stage pipeline version
    float Csub = 0.0;

    cuda::pipeline<cuda::thread_scope_thread> pipe = cuda::make_pipeline();
    const auto shape1 = cuda::aligned_size_t<alignof(float)>(sizeof(float));


    // Loop over all the sub-matrices of A and B
    // required to compute the block sub-matrix
    for (int a = aBegin, b = bBegin; a <= aEnd; a += aStep, b += bStep) {
        // Load the matrices from device memory to shared memory; each thread loads
        // one element of each matrix
        {
            pipe.producer_acquire();

            cuda::memcpy_async(&As[threadIdx.y][threadIdx.x], &A[a + wA * threadIdx.y + threadIdx.x], shape1, pipe);
            cuda::memcpy_async(&Bs[threadIdx.y][threadIdx.x], &B[b + wB * threadIdx.y + threadIdx.x], shape1, pipe);

            pipe.producer_commit();
        }

        pipe.consumer_wait();
        // Synchronize to make sure the matrices are loaded
        __syncthreads();

        // Multiply the two matrices together;
        // each thread computes one element
        // of the block sub-matrix
#pragma unroll
        for (int k = 0; k < BLOCK_SIZE; ++k) {
            Csub += As[threadIdx.y][k] * Bs[k][threadIdx.x];
        }

        // Synchronize to make sure that the preceding
        // computation is done before overwriting the
        // shared memory sub-matrix buffers As and Bs in the next iteration.
        __syncthreads();
    }

    // Write the block sub-matrix to device memory;
    // each thread writes four element
    int c = wB * BLOCK_SIZE * blockIdx.y + BLOCK_SIZE * blockIdx.x;
    C[c + wB * threadIdx.y + threadIdx.x] = Csub;
}

// Multi Stage memcpy_async thread_scope_thread pipeline with single-element async-copy
extern "C"
__global__ void MatrixMulAsyncCopyMultiStage(float* __restrict__ C,
                                                        const float* __restrict__ A,
                                                        const float* __restrict__ B, int wA,
                                                        int wB) {
    // Multi-stage pipeline version
    constexpr size_t maxPipelineStages = 4;

    // Declaration of the shared memory array As used to
    // store the sub-matrix of A for each stage
    __shared__ float As[maxPipelineStages][BLOCK_SIZE][BLOCK_SIZE];

    // Declaration of the shared memory array Bs used to
    // store the sub-matrix of B for each stage
    __shared__ float Bs[maxPipelineStages][BLOCK_SIZE][BLOCK_SIZE];

    float Csub = 0.0;

    // Index of the first sub-matrix of A processed by the block
    const int aBegin = wA * BLOCK_SIZE * blockIdx.y;

    // Index of the last sub-matrix of A processed by the block
    const int aEnd   = aBegin + wA - 1;

    // Step size used to iterate through the sub-matrices of A
    int aStep  = BLOCK_SIZE;

    // Index of the first sub-matrix of B processed by the block
    const int bBegin = BLOCK_SIZE * blockIdx.x;

    // Step size used to iterate through the sub-matrices of B
    int bStep  = BLOCK_SIZE * wB;

    cuda::pipeline<cuda::thread_scope_thread> pipe = cuda::make_pipeline();
    const auto shape1 = cuda::aligned_size_t<alignof(float)>(sizeof(float));

    // Loop over all the sub-matrices of A and B
    // required to compute the block sub-matrix
    for (int a = aBegin, b = bBegin, i = 0, aStage = aBegin, bStage = bBegin, iStage = 0; a <= aEnd; a += aStep, b += bStep, ++i ) {
        // Load the matrices from device memory to shared memory; each thread loads
        // one element of each matrix

        for ( ; aStage <= a + aStep * maxPipelineStages ; aStage += aStep, bStage += bStep, ++iStage )
        {
            if ( aStage <= aEnd )
            {
                // Rotating buffer
                const int j = iStage % maxPipelineStages;

                pipe.producer_acquire();

                cuda::memcpy_async(&As[j][threadIdx.y][threadIdx.x], &A[aStage + wA * threadIdx.y + threadIdx.x], shape1, pipe);
                cuda::memcpy_async(&Bs[j][threadIdx.y][threadIdx.x], &B[bStage + wB * threadIdx.y + threadIdx.x], shape1, pipe);

                pipe.producer_commit();
            }
        }
        pipe.consumer_wait();

        // Synchronize to make sure the matrices are loaded
        __syncthreads();

        const int j = i % maxPipelineStages;

        // Multiply the two matrices together;
        // each thread computes one element
        // of the block sub-matrix
        for (int k = 0; k < BLOCK_SIZE; ++k) {
            Csub += As[j][threadIdx.y][k] * Bs[j][k][threadIdx.x];
        }

        pipe.consumer_release();
        // Don't have to synchronize because maxPipelineStages is greater than one
        // therefore next iteration is loading to a different buffer.
    }

    // Write the block sub-matrix to device memory;
    // each thread writes four element
    int c = wB * BLOCK_SIZE * blockIdx.y + BLOCK_SIZE * blockIdx.x;
    C[c + wB * threadIdx.y + threadIdx.x] = Csub;
}

// Multi Stage shared state memcpy_async pipeline thread_scope_block
// with parititioned producer & consumer, here we've 1 warp as producer
// group which issues memcpy_async operations and rest all warps are part of
// consumer group which perform gemm computation on the loaded matrices by producer.
extern "C"
__global__ void MatrixMulAsyncCopyMultiStageSharedState(float* __restrict__ C,
                                                        const float* __restrict__ A,
                                                        const float* __restrict__ B, int wA,
                                                        int wB) {
    // Multi-stage pipeline version
    constexpr size_t maxPipelineStages = 4;

    // Declaration of the shared memory array As used to
    // store the sub-matrix of A for each stage
    __shared__ float As[maxPipelineStages][BLOCK_SIZE_X][BLOCK_SIZE_X];

    // Declaration of the shared memory array Bs used to
    // store the sub-matrix of B for each stage
    __shared__ float Bs[maxPipelineStages][BLOCK_SIZE_X][BLOCK_SIZE_X];

    float Csub = 0.0;

    // Index of the first sub-matrix of A processed by the block
    const int aBegin = wA * BLOCK_SIZE_X * blockIdx.y;

    // Index of the last sub-matrix of A processed by the block
    const int aEnd = aBegin + wA - 1;

    // Step size used to iterate through the sub-matrices of A
    constexpr int aStep  = BLOCK_SIZE_X;

    // Index of the first sub-matrix of B processed by the block
    const int bBegin = BLOCK_SIZE_X * blockIdx.x;

    // Step size used to iterate through the sub-matrices of B
    int bStep  = BLOCK_SIZE_X * wB;

    auto cta = cg::this_thread_block();

    const auto shape1 = cuda::aligned_size_t<alignof(float)>(sizeof(float));
    __shared__ cuda::pipeline_shared_state<cuda::thread_scope_block, maxPipelineStages> shared_state;
    constexpr int consumer_row_count =  BLOCK_SIZE_X;

    const auto thread_role = (cta.thread_index().y < consumer_row_count)
                                ? cuda::pipeline_role::consumer
                                : cuda::pipeline_role::producer;
    auto pipe = cuda::make_pipeline(cta, &shared_state, thread_role);

    // Loop over all the sub-matrices of A and B
    // required to compute the block sub-matrix
    for (int a = aBegin, b = bBegin, i = 0, aStage = aBegin, bStage = bBegin, iStage = 0;
                                                a <= aEnd; a += aStep, b += bStep, ++i) {
        if (threadIdx.y >= consumer_row_count) {
            // this is a whole producer warp because threadIdx.y >= 16 where 16 == consumer_row_count,
            // which loads the matrices from device memory to shared memory;
            for (; aStage <= a + aStep * maxPipelineStages; aStage += aStep, bStage += bStep, ++iStage) {
                if (aStage <= aEnd) {
                    // Rotating buffer
                    const int j = iStage % maxPipelineStages;
                    const int strideRows = (blockDim.y - consumer_row_count);
                    pipe.producer_acquire();
                    for (int rowId = threadIdx.y - consumer_row_count; rowId < BLOCK_SIZE_X; rowId += strideRows) {
                        cuda::memcpy_async(&As[j][rowId][threadIdx.x],
                                            &A[aStage + wA * rowId + threadIdx.x], shape1, pipe);
                        cuda::memcpy_async(&Bs[j][rowId][threadIdx.x],
                                            &B[bStage + wB * rowId + threadIdx.x], shape1, pipe);
                    }
                    pipe.producer_commit();
                }
            }
        }
        else {
            // this is a whole set of consumer group because threadIdx.y < consumer_row_count where consumer_row_count == 16,
            // which computes gemm operation on matrices loaded in shared memory by producer warp.
            const int j = i % maxPipelineStages;
            // Synchronize consumer group to make sure the matrices are loaded by producer group.
            pipe.consumer_wait();
            // Multiply the two matrices together;
            // each thread computes one element
            // of the block sub-matrix
            #pragma unroll
            for (int k = 0; k < BLOCK_SIZE_X; ++k) {
                Csub += As[j][threadIdx.y][k] * Bs[j][k][threadIdx.x];
            }
            pipe.consumer_release();
        }
    }

    // Write the block sub-matrix to device memory;
    // each thread writes four element
    if (threadIdx.y < consumer_row_count)
    {
        const int c = wB * BLOCK_SIZE_X * blockIdx.y + BLOCK_SIZE_X * blockIdx.x;
        C[c + wB * threadIdx.y + threadIdx.x] = Csub;
    }
}

/**
 * Matrix multiplication (CUDA Kernel) on the device: C = A * B
 * wA is A's width and wB is B's width
 */
 extern "C"
 __global__ void MatrixMulNaive(float *C, float *A,
                                                        float *B, int wA,
                                                        int wB) {
    // Declaration of the shared memory array As used to
    // store the sub-matrix of A
    __shared__ float As[BLOCK_SIZE][BLOCK_SIZE];

    // Declaration of the shared memory array Bs used to
    // store the sub-matrix of B
    __shared__ float Bs[BLOCK_SIZE][BLOCK_SIZE];

    // Index of the first sub-matrix of A processed by the block
    int aBegin = wA * BLOCK_SIZE * blockIdx.y;

    // Index of the last sub-matrix of A processed by the block
    int aEnd   = aBegin + wA - 1;

    // Step size used to iterate through the sub-matrices of A
    int aStep  = BLOCK_SIZE;

    // Index of the first sub-matrix of B processed by the block
    int bBegin = BLOCK_SIZE * blockIdx.x;

    // Step size used to iterate through the sub-matrices of B
    int bStep  = BLOCK_SIZE * wB;

    // Csub is used to store the element of the block sub-matrix
    // that is computed by the thread
    float Csub = 0;

    // Loop over all the sub-matrices of A and B
    // required to compute the block sub-matrix
    for (int a = aBegin, b = bBegin;
            a <= aEnd;
            a += aStep, b += bStep) {

        // Load the matrices from device memory
        // to shared memory; each thread loads
        // one element of each matrix
        As[threadIdx.y][threadIdx.x] = A[a + wA * threadIdx.y + threadIdx.x];
        Bs[threadIdx.y][threadIdx.x] = B[b + wB * threadIdx.y + threadIdx.x];

        // Synchronize to make sure the matrices are loaded
        __syncthreads();

        // Multiply the two matrices together;
        // each thread computes one element
        // of the block sub-matrix
#pragma unroll
        for (int k = 0; k < BLOCK_SIZE; ++k) {
            Csub += As[threadIdx.y][k] * Bs[k][threadIdx.x];
        }

        // Synchronize to make sure that the preceding
        // computation is done before loading two new
        // sub-matrices of A and B in the next iteration
        __syncthreads();
    }

    // Write the block sub-matrix to device memory;
    // each thread writes one element
    int c = wB * BLOCK_SIZE * blockIdx.y + BLOCK_SIZE * blockIdx.x;
    C[c + wB * threadIdx.y + threadIdx.x] = Csub;
}

extern "C"
__global__ void MatrixMulNaiveLargeChunk(float *C, float *A,
                                                        float *B, int wA,
                                                        int wB) {
    // Declaration of the shared memory array As used to
    // store the sub-matrix of A
    __shared__ alignas(alignof(float4)) float As[BLOCK_SIZE][BLOCK_SIZE];

    // Declaration of the shared memory array Bs used to
    // store the sub-matrix of B
    __shared__ alignas(alignof(float4)) float Bs[BLOCK_SIZE][BLOCK_SIZE];

    int t4x = threadIdx.x * 4 ;

    // Index of the first sub-matrix of A processed by the block
    int aBegin = wA * BLOCK_SIZE * blockIdx.y;

    // Index of the last sub-matrix of A processed by the block
    int aEnd   = aBegin + wA - 1;

    // Step size used to iterate through the sub-matrices of A
    int aStep  = BLOCK_SIZE;

    // Index of the first sub-matrix of B processed by the block
    int bBegin = BLOCK_SIZE * blockIdx.x;

    // Step size used to iterate through the sub-matrices of B
    int bStep  = BLOCK_SIZE * wB;

    // Csub is used to store the element of the block sub-matrix
    // that is computed by the thread
    float Csub = 0;

    // Loop over all the sub-matrices of A and B
    // required to compute the block sub-matrix
    for (int a = aBegin, b = bBegin;
            a <= aEnd;
            a += aStep, b += bStep) {

        // Load the matrices from device memory
        // to shared memory;

        // One fourth of the threads load four elements of each matrix
        if ( t4x < BLOCK_SIZE ) {
            float4 * const A4s = reinterpret_cast<float4*>(& As[threadIdx.y][t4x]);
            float4 * const B4s = reinterpret_cast<float4*>(& Bs[threadIdx.y][t4x]);
            const float4 * const A4 = reinterpret_cast<float4*>(& A[a + wA * threadIdx.y + t4x]);
            const float4 * const B4 = reinterpret_cast<float4*>(& B[a + wA * threadIdx.y + t4x]);
            *A4s = *A4 ;
            *B4s = *B4 ;
        }

        // Synchronize to make sure the matrices are loaded
        __syncthreads();

        // Multiply the two matrices together;
        // each thread computes one element
        // of the block sub-matrix
#pragma unroll
        for (int k = 0; k < BLOCK_SIZE; ++k) {
            Csub += As[threadIdx.y][k] * Bs[k][threadIdx.x];
        }

        // Synchronize to make sure that the preceding
        // computation is done before loading two new
        // sub-matrices of A and B in the next iteration
        __syncthreads();
    }

    // Write the block sub-matrix to device memory;
    // each thread writes one element
    int c = wB * BLOCK_SIZE * blockIdx.y + BLOCK_SIZE * blockIdx.x;
    C[c + wB * threadIdx.y + threadIdx.x] = Csub;
}
"""


def ConstantInit(data, size, val):
    p_data = (ctypes.c_float * size).from_address(data)
    for i in range(size):
        p_data[i] = val


#
# Run matrix multiplication using CUDA
#
def MatrixMultiply(dimsA, dimsB, kernel_number):
    # Allocate host memory for matricies A and B
    size_A = dimsA.x * dimsA.y
    mem_size_A = np.dtype(np.float32).itemsize * size_A
    h_A = checkCudaErrors(cudart.cudaMallocHost(mem_size_A))
    size_B = dimsB.x * dimsB.y
    mem_size_B = np.dtype(np.float32).itemsize * size_B
    h_B = checkCudaErrors(cudart.cudaMallocHost(mem_size_B))

    # Initialize host memory
    valB = 2.10
    ConstantInit(h_A, size_A, 1.0)
    ConstantInit(h_B, size_B, valB)

    # Allocate Device Memory

    # Allocate host matrix C
    dimsC = cudart.dim3()
    dimsC.x = dimsB.x
    dimsC.y = dimsA.y
    dimsC.z = 1
    mem_size_C = dimsC.x * dimsC.y * np.dtype(np.float32).itemsize
    h_C = checkCudaErrors(cudart.cudaMallocHost(mem_size_C))

    if h_C == 0:
        print("Failed to allocate host matri C!")
        exit(-1)

    d_A = checkCudaErrors(cudart.cudaMalloc(mem_size_A))
    d_B = checkCudaErrors(cudart.cudaMalloc(mem_size_B))
    d_C = checkCudaErrors(cudart.cudaMalloc(mem_size_C))
    # Allocate CUDA events that we'll use for timing
    start = checkCudaErrors(cudart.cudaEventCreate())
    stop = checkCudaErrors(cudart.cudaEventCreate())

    stream = checkCudaErrors(cudart.cudaStreamCreateWithFlags(cudart.cudaStreamNonBlocking))

    # Copy host memory to device
    checkCudaErrors(cudart.cudaMemcpyAsync(d_A, h_A, mem_size_A, cudart.cudaMemcpyKind.cudaMemcpyHostToDevice, stream))
    checkCudaErrors(cudart.cudaMemcpyAsync(d_B, h_B, mem_size_B, cudart.cudaMemcpyKind.cudaMemcpyHostToDevice, stream))
    checkCudaErrors(cudart.cudaMemsetAsync(d_C, 0, mem_size_C, stream))

    # Setup execution parameters
    threads = cudart.dim3()
    threads.x = threads.y = blockSize
    threads.z = 1
    grid = cudart.dim3()
    grid.x = dimsB.x / threads.x
    grid.y = dimsA.y / threads.y
    grid.z = 1

    # Here the block size is 16x18, where first 16 rows are consumer thread group
    # and last 2 rows (1 warp) is producer thread group
    threadsSharedStateKernel = cudart.dim3()
    threadsSharedStateKernel.x = blockSize
    threadsSharedStateKernel.y = blockSize + 2
    threadsSharedStateKernel.z = 1
    gridSharedStateKernel = cudart.dim3()
    gridSharedStateKernel.x = dimsB.x / threadsSharedStateKernel.x
    gridSharedStateKernel.y = dimsA.y / threadsSharedStateKernel.x

    print(f"Running kernel = {kernel_number} - {kernelNames[kernel_number.value]}")
    # Create and start timer
    print("Computing result using CUDA Kernel...")

    # Performs warmup operation using matrixMul CUDA kernel
    kernelArguments = (
        (d_C, d_A, d_B, dimsA.x, dimsB.x),
        (ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p, ctypes.c_int, ctypes.c_int),
    )
    if kernel_number == kernels.AsyncCopyMultiStageLargeChunk:
        checkCudaErrors(
            cuda.cuLaunchKernel(
                _MatrixMulAsyncCopyMultiStageLargeChunk,
                grid.x,
                grid.y,
                grid.z,  # grid dim
                threads.x,
                threads.y,
                threads.z,  # block dim
                0,  # shared mem
                stream,  # stream
                kernelArguments,
                0,
            )
        )  # arguments
    elif kernel_number == kernels.AsyncCopyLargeChunk:
        checkCudaErrors(
            cuda.cuLaunchKernel(
                _MatrixMulAsyncCopyLargeChunk,
                grid.x,
                grid.y,
                grid.z,  # grid dim
                threads.x,
                threads.y,
                threads.z,  # block dim
                0,  # shared mem
                stream,  # stream
                kernelArguments,
                0,
            )
        )  # arguments
    elif kernel_number == kernels.AsyncCopyLargeChunkAWBarrier:
        checkCudaErrors(
            cuda.cuLaunchKernel(
                _MatrixMulAsyncCopyLargeChunkAWBarrier,
                grid.x,
                grid.y,
                grid.z,  # grid dim
                threads.x,
                threads.y,
                threads.z,  # block dim
                0,  # shared mem
                stream,  # stream
                kernelArguments,
                0,
            )
        )  # arguments
    elif kernel_number == kernels.AsyncCopyMultiStageSharedState:
        checkCudaErrors(
            cuda.cuLaunchKernel(
                _MatrixMulAsyncCopyMultiStageSharedState,
                gridSharedStateKernel.x,
                gridSharedStateKernel.y,
                gridSharedStateKernel.z,  # grid dim
                threadsSharedStateKernel.x,
                threadsSharedStateKernel.y,
                threadsSharedStateKernel.z,  # block dim
                0,  # shared mem
                stream,  # stream
                kernelArguments,
                0,
            )
        )  # arguments
    elif kernel_number == kernels.AsyncCopyMultiStage:
        checkCudaErrors(
            cuda.cuLaunchKernel(
                _MatrixMulAsyncCopyMultiStage,
                grid.x,
                grid.y,
                grid.z,  # grid dim
                threads.x,
                threads.y,
                threads.z,  # block dim
                0,  # shared mem
                stream,  # stream
                kernelArguments,
                0,
            )
        )  # arguments
    elif kernel_number == kernels.AsyncCopySingleStage:
        checkCudaErrors(
            cuda.cuLaunchKernel(
                _MatrixMulAsyncCopySingleStage,
                grid.x,
                grid.y,
                grid.z,  # grid dim
                threads.x,
                threads.y,
                threads.z,  # block dim
                0,  # shared mem
                stream,  # stream
                kernelArguments,
                0,
            )
        )  # arguments
    elif kernel_number == kernels.Naive:
        checkCudaErrors(
            cuda.cuLaunchKernel(
                _MatrixMulNaive,
                grid.x,
                grid.y,
                grid.z,  # grid dim
                threads.x,
                threads.y,
                threads.z,  # block dim
                0,  # shared mem
                stream,  # stream
                kernelArguments,
                0,
            )
        )  # arguments
    elif kernel_number == kernels.NaiveLargeChunk:
        checkCudaErrors(
            cuda.cuLaunchKernel(
                _MatrixMulNaiveLargeChunk,
                grid.x,
                grid.y,
                grid.z,  # grid dim
                threads.x,
                threads.y,
                threads.z,  # block dim
                0,  # shared mem
                stream,  # stream
                kernelArguments,
                0,
            )
        )  # arguments

    print("done")
    checkCudaErrors(cudart.cudaStreamSynchronize(stream))

    # Execute the kernel
    nIter = 100

    # Record the start event
    checkCudaErrors(cudart.cudaEventRecord(start, stream))

    if kernel_number == kernels.AsyncCopyMultiStageLargeChunk:
        checkCudaErrors(
            cuda.cuLaunchKernel(
                _MatrixMulAsyncCopyMultiStageLargeChunk,
                grid.x,
                grid.y,
                grid.z,  # grid dim
                threads.x,
                threads.y,
                threads.z,  # block dim
                0,  # shared mem
                stream,  # stream
                kernelArguments,
                0,
            )
        )  # arguments
    elif kernel_number == kernels.AsyncCopyLargeChunk:
        checkCudaErrors(
            cuda.cuLaunchKernel(
                _MatrixMulAsyncCopyLargeChunk,
                grid.x,
                grid.y,
                grid.z,  # grid dim
                threads.x,
                threads.y,
                threads.z,  # block dim
                0,  # shared mem
                stream,  # stream
                kernelArguments,
                0,
            )
        )  # arguments
    elif kernel_number == kernels.AsyncCopyLargeChunkAWBarrier:
        checkCudaErrors(
            cuda.cuLaunchKernel(
                _MatrixMulAsyncCopyLargeChunkAWBarrier,
                grid.x,
                grid.y,
                grid.z,  # grid dim
                threads.x,
                threads.y,
                threads.z,  # block dim
                0,  # shared mem
                stream,  # stream
                kernelArguments,
                0,
            )
        )  # arguments
    elif kernel_number == kernels.AsyncCopyMultiStageSharedState:
        checkCudaErrors(
            cuda.cuLaunchKernel(
                _MatrixMulAsyncCopyMultiStageSharedState,
                gridSharedStateKernel.x,
                gridSharedStateKernel.y,
                gridSharedStateKernel.z,  # grid dim
                threadsSharedStateKernel.x,
                threadsSharedStateKernel.y,
                threadsSharedStateKernel.z,  # block dim
                0,  # shared mem
                stream,  # stream
                kernelArguments,
                0,
            )
        )  # arguments
    elif kernel_number == kernels.AsyncCopyMultiStage:
        checkCudaErrors(
            cuda.cuLaunchKernel(
                _MatrixMulAsyncCopyMultiStage,
                grid.x,
                grid.y,
                grid.z,  # grid dim
                threads.x,
                threads.y,
                threads.z,  # block dim
                0,  # shared mem
                stream,  # stream
                kernelArguments,
                0,
            )
        )  # arguments
    elif kernel_number == kernels.AsyncCopySingleStage:
        checkCudaErrors(
            cuda.cuLaunchKernel(
                _MatrixMulAsyncCopySingleStage,
                grid.x,
                grid.y,
                grid.z,  # grid dim
                threads.x,
                threads.y,
                threads.z,  # block dim
                0,  # shared mem
                stream,  # stream
                kernelArguments,
                0,
            )
        )  # arguments
    elif kernel_number == kernels.Naive:
        checkCudaErrors(
            cuda.cuLaunchKernel(
                _MatrixMulNaive,
                grid.x,
                grid.y,
                grid.z,  # grid dim
                threads.x,
                threads.y,
                threads.z,  # block dim
                0,  # shared mem
                stream,  # stream
                kernelArguments,
                0,
            )
        )  # arguments
    elif kernel_number == kernels.NaiveLargeChunk:
        checkCudaErrors(
            cuda.cuLaunchKernel(
                _MatrixMulNaiveLargeChunk,
                grid.x,
                grid.y,
                grid.z,  # grid dim
                threads.x,
                threads.y,
                threads.z,  # block dim
                0,  # shared mem
                stream,  # stream
                kernelArguments,
                0,
            )
        )  # arguments

    # Record the stop event
    checkCudaErrors(cudart.cudaEventRecord(stop, stream))

    # Wait for the stop event to complete
    checkCudaErrors(cudart.cudaEventSynchronize(stop))

    msecTotal = checkCudaErrors(cudart.cudaEventElapsedTime(start, stop))

    # Compute and print the performance
    msecPerMatrixMul = msecTotal / nIter
    flopsPerMatrixMul = 2.0 * dimsA.x * dimsA.y * dimsB.x
    gigaFlops = (flopsPerMatrixMul * 1.0e-9) / (msecPerMatrixMul / 1000.0)

    print(
        f"Performance= {gigaFlops:.2f} GFlop/s, Time= {msecPerMatrixMul:.2f} msec, Size= {flopsPerMatrixMul:.0f} Ops, WorkgroupSize= {threads.x * threads.y} threads/block"
    )

    # Copy result from device to host
    checkCudaErrors(cudart.cudaMemcpyAsync(h_C, d_C, mem_size_C, cudart.cudaMemcpyKind.cudaMemcpyDeviceToHost, stream))
    checkCudaErrors(cudart.cudaStreamSynchronize(stream))

    print("Checking computed result for correctness: ")
    correct = True

    # test relative error by the formula
    # |<x, y>_cpu - <x,y>_gpu|/<|x|, |y|>  < eps
    eps = 1.0e-6

    h_C_local = (ctypes.c_float * (dimsC.x * dimsC.y)).from_address(h_C)
    for i in range(dimsC.x * dimsC.y):
        abs_err = math.fabs(h_C_local[i] - (dimsA.x * valB))
        dot_length = dimsA.x
        abs_val = math.fabs(h_C_local[i])
        rel_err = abs_err / abs_val / dot_length

        if rel_err > eps:
            print(f"Error! Matrix[{i:.5f}]={h_C_local[i]:.8f} ref={dimsA.x * valB:.8f} err term is > {rel_err}")
            correct = False

    print("Result = PASS" if correct else "Result = FAIL")

    # Clean up memory
    checkCudaErrors(cudart.cudaFreeHost(h_A))
    checkCudaErrors(cudart.cudaFreeHost(h_B))
    checkCudaErrors(cudart.cudaFreeHost(h_C))
    checkCudaErrors(cudart.cudaFree(d_A))
    checkCudaErrors(cudart.cudaFree(d_B))
    checkCudaErrors(cudart.cudaFree(d_C))
    checkCudaErrors(cudart.cudaEventDestroy(start))
    checkCudaErrors(cudart.cudaEventDestroy(stop))
    print(
        "\nNOTE: The CUDA Samples are not meant for performance "
        "measurements. Results may vary when GPU Boost is enabled."
    )
    if correct:
        return 0
    return -1


def main():
    common.pytest_skipif_cuda_include_not_found()
    common.pytest_skipif_compute_capability_too_low(findCudaDevice(), (7, 0))

    print("[globalToShmemAsyncCopy] - Starting...")

    if platform.machine() == "qnx":
        print("globalToShmemAsyncCopy is not supported on QNX - waiving sample")
        return

    version = checkCudaErrors(cuda.cuDriverGetVersion())
    if version < 11010:
        print("CUDA Toolkit 11.1 or greater is required")
        return

    if checkCmdLineFlag("help") or checkCmdLineFlag("?"):
        print("Usage device=n (n >= 0 for deviceID)")
        print("      wA=WidthA hA=HeightA (Width x Height of Matrix A)")
        print("      wB=WidthB hB=HeightB (Width x Height of Matrix B)")
        print("      kernel=kernel_number (0 - AsyncCopyMultiStageLargeChunk; 1 - AsyncCopyLargeChunk)")
        print("                            (2 - AsyncCopyLargeChunkAWBarrier; 3 - AsyncCopyMultiStageSharedState)")
        print(
            "                            (4 - AsyncCopyMultiStage; 5 - AsyncCopySingleStage; 6 - Naive without memcpy_async)"
        )
        print("                            (7 - NaiveLargeChunk without memcpy_async)")
        print("  Note: Outer matrix dimensions of A & B matrices must be equal.")
        return

    # This will pick the best possible CUDA capable device, otherwise
    # override the device ID based on input provided at the command line
    devID = findCudaDevice()

    matrixBlock = 32
    dimsA = cudart.dim3()
    dimsA.x = dimsA.y = 10 * 4 * matrixBlock
    dimsA.z = 1
    dimsB = cudart.dim3()
    dimsB.x = dimsB.y = 10 * 4 * matrixBlock
    dimsB.z = 1

    # width of Matrix A
    if checkCmdLineFlag("wA="):
        dimsA.x = int(getCmdLineArgumentInt("wA="))

    # height of Matrix A
    if checkCmdLineFlag("hA="):
        dimsA.y = int(getCmdLineArgumentInt("hA="))

    # width of Matrix B
    if checkCmdLineFlag("wB="):
        dimsB.x = int(getCmdLineArgumentInt("wB="))

    # height of Matrix B
    if checkCmdLineFlag("hB="):
        dimsB.y = int(getCmdLineArgumentInt("hB="))

    if dimsA.x != dimsB.y:
        print(f"Error: outer matrix dimensions must be equal. ({dimsA.x} != {dimsB.y})")
        sys.exit(-1)

    selected_kernel = kernels.AsyncCopyMultiStageLargeChunk

    # kernel to run - default (AsyncCopyMultiStageLargeChunk == 0)
    if checkCmdLineFlag("kernel="):
        kernel_number = int(getCmdLineArgumentInt("kernel="))
        if kernel_number < 8:
            selected_kernel = kernels(kernel_number)
        else:
            print("Error: kernel number should be between 0 to 7, you have entered %d".format())
            sys.exit(-1)

    major = checkCudaErrors(
        cudart.cudaDeviceGetAttribute(cudart.cudaDeviceAttr.cudaDevAttrComputeCapabilityMajor, devID)
    )
    if major < 7:
        print("globalToShmemAsyncCopy requires SM 7.0 or higher.  Exiting...")
        return

    print(f"MatrixA({dimsA.x},{dimsA.y}), MatrixB({dimsB.x},{dimsB.y})")

    global _MatrixMulAsyncCopyMultiStageLargeChunk
    global _MatrixMulAsyncCopyLargeChunk
    global _MatrixMulAsyncCopyLargeChunkAWBarrier
    global _MatrixMulAsyncCopyMultiStageSharedState
    global _MatrixMulAsyncCopyMultiStage
    global _MatrixMulAsyncCopySingleStage
    global _MatrixMulNaive
    global _MatrixMulNaiveLargeChunk
    kernelHelper = common.KernelHelper(globalToShmemAsyncCopy, devID)
    _MatrixMulAsyncCopyMultiStageLargeChunk = kernelHelper.getFunction(b"MatrixMulAsyncCopyMultiStageLargeChunk")
    _MatrixMulAsyncCopyLargeChunk = kernelHelper.getFunction(b"MatrixMulAsyncCopyLargeChunk")
    _MatrixMulAsyncCopyLargeChunkAWBarrier = kernelHelper.getFunction(b"MatrixMulAsyncCopyLargeChunkAWBarrier")
    _MatrixMulAsyncCopyMultiStageSharedState = kernelHelper.getFunction(b"MatrixMulAsyncCopyMultiStageSharedState")
    _MatrixMulAsyncCopyMultiStage = kernelHelper.getFunction(b"MatrixMulAsyncCopyMultiStage")
    _MatrixMulAsyncCopySingleStage = kernelHelper.getFunction(b"MatrixMulAsyncCopySingleStage")
    _MatrixMulNaive = kernelHelper.getFunction(b"MatrixMulNaive")
    _MatrixMulNaiveLargeChunk = kernelHelper.getFunction(b"MatrixMulNaiveLargeChunk")

    matrix_result = MatrixMultiply(dimsA, dimsB, selected_kernel)

    if matrix_result != 0:
        sys.exit(-1)


if __name__ == "__main__":
    main()
