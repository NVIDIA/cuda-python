# Copyright 2021-2025 NVIDIA Corporation.  All rights reserved.
# SPDX-License-Identifier: LicenseRef-NVIDIA-SOFTWARE-LICENSE

import ctypes
import math
import platform
import sys
from enum import Enum

import numpy as np
from common import common
from common.helper_cuda import check_cuda_errors, find_cuda_device
from common.helper_string import check_cmd_line_flag, get_cmd_line_argument_int

from cuda.bindings import driver as cuda
from cuda.bindings import runtime as cudart

block_size = 16


class Kernels(Enum):
    AsyncCopyMultiStageLargeChunk = 0
    AsyncCopyLargeChunk = 1
    AsyncCopyLargeChunkAWBarrier = 2
    AsyncCopyMultiStageSharedState = 3
    AsyncCopyMultiStage = 4
    AsyncCopySingleStage = 5
    Naive = 6
    NaiveLargeChunk = 7


kernel_names = [
    "AsyncCopyMultiStageLargeChunk",
    "AsyncCopyLargeChunk",
    "AsyncCopyLargeChunkAWBarrier",
    "AsyncCopyMultiStageSharedState",
    "AsyncCopyMultiStage",
    "AsyncCopySingleStage",
    "Naive",
    "NaiveLargeChunk",
]

global_to_shmem_async_copy = """\
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


def constant_init(data, size, val):
    p_data = (ctypes.c_float * size).from_address(data)
    for i in range(size):
        p_data[i] = val


#
# Run matrix multiplication using CUDA
#
def matrix_multiply(dims_a, dims_b, kernel_number):
    # Allocate host memory for matricies A and B
    size_a = dims_a.x * dims_a.y
    mem_size_a = np.dtype(np.float32).itemsize * size_a
    h_a = check_cuda_errors(cudart.cudaMallocHost(mem_size_a))
    size_b = dims_b.x * dims_b.y
    mem_size_b = np.dtype(np.float32).itemsize * size_b
    h_b = check_cuda_errors(cudart.cudaMallocHost(mem_size_b))

    # Initialize host memory
    val_b = 2.10
    constant_init(h_a, size_a, 1.0)
    constant_init(h_b, size_b, val_b)

    # Allocate Device Memory

    # Allocate host matrix C
    dims_c = cudart.dim3()
    dims_c.x = dims_b.x
    dims_c.y = dims_a.y
    dims_c.z = 1
    mem_size_c = dims_c.x * dims_c.y * np.dtype(np.float32).itemsize
    h_c = check_cuda_errors(cudart.cudaMallocHost(mem_size_c))

    if h_c == 0:
        print("Failed to allocate host matrix C!", file=sys.stderr)
        sys.exit(1)

    d_a = check_cuda_errors(cudart.cudaMalloc(mem_size_a))
    d_b = check_cuda_errors(cudart.cudaMalloc(mem_size_b))
    d_c = check_cuda_errors(cudart.cudaMalloc(mem_size_c))
    # Allocate CUDA events that we'll use for timing
    start = check_cuda_errors(cudart.cudaEventCreate())
    stop = check_cuda_errors(cudart.cudaEventCreate())

    stream = check_cuda_errors(cudart.cudaStreamCreateWithFlags(cudart.cudaStreamNonBlocking))

    # Copy host memory to device
    check_cuda_errors(
        cudart.cudaMemcpyAsync(d_a, h_a, mem_size_a, cudart.cudaMemcpyKind.cudaMemcpyHostToDevice, stream)
    )
    check_cuda_errors(
        cudart.cudaMemcpyAsync(d_b, h_b, mem_size_b, cudart.cudaMemcpyKind.cudaMemcpyHostToDevice, stream)
    )
    check_cuda_errors(cudart.cudaMemsetAsync(d_c, 0, mem_size_c, stream))

    # Setup execution parameters
    threads = cudart.dim3()
    threads.x = threads.y = block_size
    threads.z = 1
    grid = cudart.dim3()
    grid.x = dims_b.x / threads.x
    grid.y = dims_a.y / threads.y
    grid.z = 1

    # Here the block size is 16x18, where first 16 rows are consumer thread group
    # and last 2 rows (1 warp) is producer thread group
    threads_shared_state_kernel = cudart.dim3()
    threads_shared_state_kernel.x = block_size
    threads_shared_state_kernel.y = block_size + 2
    threads_shared_state_kernel.z = 1
    grid_shared_state_kernel = cudart.dim3()
    grid_shared_state_kernel.x = dims_b.x / threads_shared_state_kernel.x
    grid_shared_state_kernel.y = dims_a.y / threads_shared_state_kernel.x

    print(f"Running kernel = {kernel_number} - {kernel_names[kernel_number.value]}")
    # Create and start timer
    print("Computing result using CUDA Kernel...")

    # Performs warmup operation using matrixMul CUDA kernel
    kernel_arguments = (
        (d_c, d_a, d_b, dims_a.x, dims_b.x),
        (ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p, ctypes.c_int, ctypes.c_int),
    )
    if kernel_number == Kernels.AsyncCopyMultiStageLargeChunk:
        check_cuda_errors(
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
                kernel_arguments,
                0,
            )
        )  # arguments
    elif kernel_number == Kernels.AsyncCopyLargeChunk:
        check_cuda_errors(
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
                kernel_arguments,
                0,
            )
        )  # arguments
    elif kernel_number == Kernels.AsyncCopyLargeChunkAWBarrier:
        check_cuda_errors(
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
                kernel_arguments,
                0,
            )
        )  # arguments
    elif kernel_number == Kernels.AsyncCopyMultiStageSharedState:
        check_cuda_errors(
            cuda.cuLaunchKernel(
                _MatrixMulAsyncCopyMultiStageSharedState,
                grid_shared_state_kernel.x,
                grid_shared_state_kernel.y,
                grid_shared_state_kernel.z,  # grid dim
                threads_shared_state_kernel.x,
                threads_shared_state_kernel.y,
                threads_shared_state_kernel.z,  # block dim
                0,  # shared mem
                stream,  # stream
                kernel_arguments,
                0,
            )
        )  # arguments
    elif kernel_number == Kernels.AsyncCopyMultiStage:
        check_cuda_errors(
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
                kernel_arguments,
                0,
            )
        )  # arguments
    elif kernel_number == Kernels.AsyncCopySingleStage:
        check_cuda_errors(
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
                kernel_arguments,
                0,
            )
        )  # arguments
    elif kernel_number == Kernels.Naive:
        check_cuda_errors(
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
                kernel_arguments,
                0,
            )
        )  # arguments
    elif kernel_number == Kernels.NaiveLargeChunk:
        check_cuda_errors(
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
                kernel_arguments,
                0,
            )
        )  # arguments

    check_cuda_errors(cudart.cudaStreamSynchronize(stream))

    # Execute the kernel
    n_iter = 100

    # Record the start event
    check_cuda_errors(cudart.cudaEventRecord(start, stream))

    if kernel_number == Kernels.AsyncCopyMultiStageLargeChunk:
        check_cuda_errors(
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
                kernel_arguments,
                0,
            )
        )  # arguments
    elif kernel_number == Kernels.AsyncCopyLargeChunk:
        check_cuda_errors(
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
                kernel_arguments,
                0,
            )
        )  # arguments
    elif kernel_number == Kernels.AsyncCopyLargeChunkAWBarrier:
        check_cuda_errors(
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
                kernel_arguments,
                0,
            )
        )  # arguments
    elif kernel_number == Kernels.AsyncCopyMultiStageSharedState:
        check_cuda_errors(
            cuda.cuLaunchKernel(
                _MatrixMulAsyncCopyMultiStageSharedState,
                grid_shared_state_kernel.x,
                grid_shared_state_kernel.y,
                grid_shared_state_kernel.z,  # grid dim
                threads_shared_state_kernel.x,
                threads_shared_state_kernel.y,
                threads_shared_state_kernel.z,  # block dim
                0,  # shared mem
                stream,  # stream
                kernel_arguments,
                0,
            )
        )  # arguments
    elif kernel_number == Kernels.AsyncCopyMultiStage:
        check_cuda_errors(
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
                kernel_arguments,
                0,
            )
        )  # arguments
    elif kernel_number == Kernels.AsyncCopySingleStage:
        check_cuda_errors(
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
                kernel_arguments,
                0,
            )
        )  # arguments
    elif kernel_number == Kernels.Naive:
        check_cuda_errors(
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
                kernel_arguments,
                0,
            )
        )  # arguments
    elif kernel_number == Kernels.NaiveLargeChunk:
        check_cuda_errors(
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
                kernel_arguments,
                0,
            )
        )  # arguments

    # Record the stop event
    check_cuda_errors(cudart.cudaEventRecord(stop, stream))

    # Wait for the stop event to complete
    check_cuda_errors(cudart.cudaEventSynchronize(stop))

    msec_total = check_cuda_errors(cudart.cudaEventElapsedTime(start, stop))

    # Compute and print the performance
    msec_per_matrix_mul = msec_total / n_iter
    flops_per_matrix_mul = 2.0 * dims_a.x * dims_a.y * dims_b.x
    giga_flops = (flops_per_matrix_mul * 1.0e-9) / (msec_per_matrix_mul / 1000.0)

    print(
        f"Performance= {giga_flops:.2f} GFlop/s, Time= {msec_per_matrix_mul:.2f} msec, Size= {flops_per_matrix_mul:.0f} Ops, WorkgroupSize= {threads.x * threads.y} threads/block"
    )

    # Copy result from device to host
    check_cuda_errors(
        cudart.cudaMemcpyAsync(h_c, d_c, mem_size_c, cudart.cudaMemcpyKind.cudaMemcpyDeviceToHost, stream)
    )
    check_cuda_errors(cudart.cudaStreamSynchronize(stream))

    correct = True

    # test relative error by the formula
    # |<x, y>_cpu - <x,y>_gpu|/<|x|, |y|>  < eps
    eps = 1.0e-6

    h_c_local = (ctypes.c_float * (dims_c.x * dims_c.y)).from_address(h_c)
    for i in range(dims_c.x * dims_c.y):
        abs_err = math.fabs(h_c_local[i] - (dims_a.x * val_b))
        dot_length = dims_a.x
        abs_val = math.fabs(h_c_local[i])
        rel_err = abs_err / abs_val / dot_length

        if rel_err > eps:
            print(
                f"Error! Matrix[{i:.5f}]={h_c_local[i]:.8f} ref={dims_a.x * val_b:.8f} err term is > {rel_err}",
                file=sys.stderr,
            )
            correct = False

    if not correct:
        print("Result = FAIL", file=sys.stderr)

    # Clean up memory
    check_cuda_errors(cudart.cudaFreeHost(h_a))
    check_cuda_errors(cudart.cudaFreeHost(h_b))
    check_cuda_errors(cudart.cudaFreeHost(h_c))
    check_cuda_errors(cudart.cudaFree(d_a))
    check_cuda_errors(cudart.cudaFree(d_b))
    check_cuda_errors(cudart.cudaFree(d_c))
    check_cuda_errors(cudart.cudaEventDestroy(start))
    check_cuda_errors(cudart.cudaEventDestroy(stop))
    print(
        "\nNOTE: The CUDA Samples are not meant for performance "
        "measurements. Results may vary when GPU Boost is enabled."
    )
    if correct:
        return 0
    return -1


def main():
    import pytest

    common.pytest_skipif_compute_capability_too_low(find_cuda_device(), (7, 0))

    if platform.machine() == "qnx":
        pytest.skip("globalToShmemAsyncCopy is not supported on QNX")

    version = check_cuda_errors(cuda.cuDriverGetVersion())
    if version < 11010:
        pytest.skip("CUDA Toolkit 11.1 or greater is required")

    if check_cmd_line_flag("help") or check_cmd_line_flag("?"):
        print("Usage device=n (n >= 0 for deviceID)", file=sys.stderr)
        print("      wA=WidthA hA=HeightA (Width x Height of Matrix A)", file=sys.stderr)
        print("      wB=WidthB hB=HeightB (Width x Height of Matrix B)", file=sys.stderr)
        print(
            "      kernel=kernel_number (0 - AsyncCopyMultiStageLargeChunk; 1 - AsyncCopyLargeChunk)", file=sys.stderr
        )
        print(
            "                            (2 - AsyncCopyLargeChunkAWBarrier; 3 - AsyncCopyMultiStageSharedState)",
            file=sys.stderr,
        )
        print(
            "                            (4 - AsyncCopyMultiStage; 5 - AsyncCopySingleStage; 6 - Naive without memcpy_async)",
            file=sys.stderr,
        )
        print("                            (7 - NaiveLargeChunk without memcpy_async)", file=sys.stderr)
        print("  Note: Outer matrix dimensions of A & B matrices must be equal.", file=sys.stderr)
        sys.exit(1)

    # This will pick the best possible CUDA capable device, otherwise
    # override the device ID based on input provided at the command line
    dev_id = find_cuda_device()

    matrix_block = 32
    dims_a = cudart.dim3()
    dims_a.x = dims_a.y = 10 * 4 * matrix_block
    dims_a.z = 1
    dims_b = cudart.dim3()
    dims_b.x = dims_b.y = 10 * 4 * matrix_block
    dims_b.z = 1

    # width of Matrix A
    if check_cmd_line_flag("wA="):
        dims_a.x = int(get_cmd_line_argument_int("wA="))

    # height of Matrix A
    if check_cmd_line_flag("hA="):
        dims_a.y = int(get_cmd_line_argument_int("hA="))

    # width of Matrix B
    if check_cmd_line_flag("wB="):
        dims_b.x = int(get_cmd_line_argument_int("wB="))

    # height of Matrix B
    if check_cmd_line_flag("hB="):
        dims_b.y = int(get_cmd_line_argument_int("hB="))

    if dims_a.x != dims_b.y:
        print(f"Error: outer matrix dimensions must be equal. ({dims_a.x} != {dims_b.y})", file=sys.stderr)
        sys.exit(1)

    selected_kernel = Kernels.AsyncCopyMultiStageLargeChunk

    # kernel to run - default (AsyncCopyMultiStageLargeChunk == 0)
    if check_cmd_line_flag("kernel="):
        kernel_number = int(get_cmd_line_argument_int("kernel="))
        if kernel_number < 8:
            selected_kernel = Kernels(kernel_number)
        else:
            print("Error: kernel number should be between 0 to 7", file=sys.stderr)
            sys.exit(1)

    major = check_cuda_errors(
        cudart.cudaDeviceGetAttribute(cudart.cudaDeviceAttr.cudaDevAttrComputeCapabilityMajor, dev_id)
    )
    if major < 7:
        pytest.skip("globalToShmemAsyncCopy requires SM 7.0 or higher.")

    print(f"MatrixA({dims_a.x},{dims_a.y}), MatrixB({dims_b.x},{dims_b.y})")

    global _MatrixMulAsyncCopyMultiStageLargeChunk
    global _MatrixMulAsyncCopyLargeChunk
    global _MatrixMulAsyncCopyLargeChunkAWBarrier
    global _MatrixMulAsyncCopyMultiStageSharedState
    global _MatrixMulAsyncCopyMultiStage
    global _MatrixMulAsyncCopySingleStage
    global _MatrixMulNaive
    global _MatrixMulNaiveLargeChunk
    kernel_helper = common.KernelHelper(global_to_shmem_async_copy, dev_id)
    _MatrixMulAsyncCopyMultiStageLargeChunk = kernel_helper.get_function(b"MatrixMulAsyncCopyMultiStageLargeChunk")
    _MatrixMulAsyncCopyLargeChunk = kernel_helper.get_function(b"MatrixMulAsyncCopyLargeChunk")
    _MatrixMulAsyncCopyLargeChunkAWBarrier = kernel_helper.get_function(b"MatrixMulAsyncCopyLargeChunkAWBarrier")
    _MatrixMulAsyncCopyMultiStageSharedState = kernel_helper.get_function(b"MatrixMulAsyncCopyMultiStageSharedState")
    _MatrixMulAsyncCopyMultiStage = kernel_helper.get_function(b"MatrixMulAsyncCopyMultiStage")
    _MatrixMulAsyncCopySingleStage = kernel_helper.get_function(b"MatrixMulAsyncCopySingleStage")
    _MatrixMulNaive = kernel_helper.get_function(b"MatrixMulNaive")
    _MatrixMulNaiveLargeChunk = kernel_helper.get_function(b"MatrixMulNaiveLargeChunk")

    matrix_result = matrix_multiply(dims_a, dims_b, selected_kernel)

    if matrix_result != 0:
        sys.exit(1)


if __name__ == "__main__":
    main()
