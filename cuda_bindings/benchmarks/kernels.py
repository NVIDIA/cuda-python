# Copyright 2021-2024 NVIDIA Corporation.  All rights reserved.
#
# Please refer to the NVIDIA end user license agreement (EULA) associated
# with this source code for terms and conditions that govern your use of
# this software. Any use, reproduction, disclosure, or distribution of
# this software and related documentation outside the terms of the EULA
# is strictly prohibited.
kernel_string = '''\
#define ITEM_PARAM(x, T) T x
#define REP1(x, T)   , ITEM_PARAM(x, T)	
#define REP2(x, T)   REP1(x##0, T)   REP1(x##1, T)
#define REP4(x, T)   REP2(x##0, T)   REP2(x##1, T)
#define REP8(x, T)   REP4(x##0, T)   REP4(x##1, T)
#define REP16(x, T)  REP8(x##0, T)   REP8(x##1, T)
#define REP32(x, T)  REP16(x##0, T)  REP16(x##1, T)
#define REP64(x, T)  REP32(x##0, T)  REP32(x##1, T)
#define REP128(x, T) REP64(x##0, T)  REP64(x##1, T)
#define REP256(x, T) REP128(x##0, T) REP128(x##1, T)

template<size_t maxBytes>
struct KernelFunctionParam
{
   unsigned char p[maxBytes];
};

extern "C" __global__ void small_kernel(float *f)
{
   *f = 0.0f;
}

extern "C" __global__ void empty_kernel()
{
   return;
}

extern "C" __global__
void small_kernel_512_args(
	ITEM_PARAM(F, int*)
	REP1(A, int*)
	REP2(A, int*)
	REP4(A, int*)
	REP8(A, int*)
	REP16(A, int*)
	REP32(A, int*)
	REP64(A, int*)
	REP128(A, int*)
	REP256(A, int*))
{
    *F = 0;
}

extern "C" __global__
void small_kernel_512_bools(
	ITEM_PARAM(F, bool)
	REP1(A, bool)
	REP2(A, bool)
	REP4(A, bool)
	REP8(A, bool)
	REP16(A, bool)
	REP32(A, bool)
	REP64(A, bool)
	REP128(A, bool)
	REP256(A, bool))
{
    return;
}

extern "C" __global__
void small_kernel_512_ints(
	ITEM_PARAM(F, int)
	REP1(A, int)
	REP2(A, int)
	REP4(A, int)
	REP8(A, int)
	REP16(A, int)
	REP32(A, int)
	REP64(A, int)
	REP128(A, int)
	REP256(A, int))
{
    return;
}

extern "C" __global__
void small_kernel_512_doubles(
	ITEM_PARAM(F, double)
	REP1(A, double)
	REP2(A, double)
	REP4(A, double)
	REP8(A, double)
	REP16(A, double)
	REP32(A, double)
	REP64(A, double)
	REP128(A, double)
	REP256(A, double))
{
    return;
}

extern "C" __global__
void small_kernel_512_chars(
	ITEM_PARAM(F, char)
	REP1(A, char)
	REP2(A, char)
	REP4(A, char)
	REP8(A, char)
	REP16(A, char)
	REP32(A, char)
	REP64(A, char)
	REP128(A, char)
	REP256(A, char))
{
    return;
}

extern "C" __global__
void small_kernel_512_longlongs(
	ITEM_PARAM(F, long long)
	REP1(A, long long)
	REP2(A, long long)
	REP4(A, long long)
	REP8(A, long long)
	REP16(A, long long)
	REP32(A, long long)
	REP64(A, long long)
	REP128(A, long long)
	REP256(A, long long))
{
    return;
}

extern "C" __global__
void small_kernel_256_args(
	ITEM_PARAM(F, int*)
	REP1(A, int*)
	REP2(A, int*)
	REP4(A, int*)
	REP8(A, int*)
	REP16(A, int*)
	REP32(A, int*)
	REP64(A, int*)
	REP128(A, int*))
{
    *F = 0;
}

extern "C" __global__
void small_kernel_16_args(
	ITEM_PARAM(F, int*)
	REP1(A, int*)
	REP2(A, int*)
	REP4(A, int*)
	REP8(A, int*))
{
    *F = 0;
}

extern "C" __global__ void small_kernel_2048B(KernelFunctionParam<2048> param)
{
    // Do not touch param to prevent compiler from copying
    // the whole structure from const bank to lmem.
}
'''
