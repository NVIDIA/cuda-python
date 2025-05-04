# Copyright 2021-2025 NVIDIA Corporation.  All rights reserved.
#
# Please refer to the NVIDIA end user license agreement (EULA) associated
# with this source code for terms and conditions that govern your use of
# this software. Any use, reproduction, disclosure, or distribution of
# this software and related documentation outside the terms of the EULA
# is strictly prohibited.
import cython
from cuda.bindings.cyruntime cimport *
from libc.string cimport memset
cimport cuda.bindings._bindings.cydriver as cydriver


cdef int case_desc(const cudaChannelFormatDesc* d, int x, int y, int z, int w, int f) except ?cudaErrorCallRequiresNewerDriver nogil:
    return d[0].x == x and d[0].y == y and d[0].z == z and d[0].w == w and d[0].f == f


cdef cudaError_t getDescInfo(const cudaChannelFormatDesc* d, int *numberOfChannels, cydriver.CUarray_format *format) except ?cudaErrorCallRequiresNewerDriver nogil:
    # Check validity
    if d[0].f in (cudaChannelFormatKind.cudaChannelFormatKindSigned,
                  cudaChannelFormatKind.cudaChannelFormatKindUnsigned):
        if (d[0].x != 8) and (d[0].x != 16) and (d[0].x != 32):
            return cudaErrorInvalidChannelDescriptor
    elif d[0].f in (cudaChannelFormatKind.cudaChannelFormatKindFloat,):
        if (d[0].x != 16) and (d[0].x != 32):
            return cudaErrorInvalidChannelDescriptor
    elif d[0].f in (cudaChannelFormatKind.cudaChannelFormatKindNV12,):
        if (d[0].x != 8) or (d[0].y != 8) or (d[0].z != 8) or (d[0].w != 0):
            return cudaErrorInvalidChannelDescriptor
    elif d[0].f in (cudaChannelFormatKind.cudaChannelFormatKindSignedNormalized8X1,
                    cudaChannelFormatKind.cudaChannelFormatKindSignedNormalized8X2,
                    cudaChannelFormatKind.cudaChannelFormatKindSignedNormalized8X4,
                    cudaChannelFormatKind.cudaChannelFormatKindUnsignedNormalized8X1,
                    cudaChannelFormatKind.cudaChannelFormatKindUnsignedNormalized8X2,
                    cudaChannelFormatKind.cudaChannelFormatKindUnsignedNormalized8X4,):
        if (d[0].x != 8):
            return cudaErrorInvalidChannelDescriptor
    elif d[0].f in (cudaChannelFormatKind.cudaChannelFormatKindSignedNormalized16X1,
                    cudaChannelFormatKind.cudaChannelFormatKindSignedNormalized16X2,
                    cudaChannelFormatKind.cudaChannelFormatKindSignedNormalized16X4,
                    cudaChannelFormatKind.cudaChannelFormatKindUnsignedNormalized16X1,
                    cudaChannelFormatKind.cudaChannelFormatKindUnsignedNormalized16X2,
                    cudaChannelFormatKind.cudaChannelFormatKindUnsignedNormalized16X4,):
        if (d[0].x != 16):
            return cudaErrorInvalidChannelDescriptor
    elif d[0].f in (cudaChannelFormatKind.cudaChannelFormatKindUnsignedBlockCompressed1,
                    cudaChannelFormatKind.cudaChannelFormatKindUnsignedBlockCompressed1SRGB,
                    cudaChannelFormatKind.cudaChannelFormatKindUnsignedBlockCompressed2,
                    cudaChannelFormatKind.cudaChannelFormatKindUnsignedBlockCompressed2SRGB,
                    cudaChannelFormatKind.cudaChannelFormatKindUnsignedBlockCompressed3,
                    cudaChannelFormatKind.cudaChannelFormatKindUnsignedBlockCompressed3SRGB,
                    cudaChannelFormatKind.cudaChannelFormatKindUnsignedBlockCompressed4,
                    cudaChannelFormatKind.cudaChannelFormatKindSignedBlockCompressed4,
                    cudaChannelFormatKind.cudaChannelFormatKindUnsignedBlockCompressed5,
                    cudaChannelFormatKind.cudaChannelFormatKindSignedBlockCompressed5,
                    cudaChannelFormatKind.cudaChannelFormatKindUnsignedBlockCompressed7,
                    cudaChannelFormatKind.cudaChannelFormatKindUnsignedBlockCompressed7SRGB,):
        if (d[0].x != 8):
            return cudaErrorInvalidChannelDescriptor
    elif d[0].f in (cudaChannelFormatKind.cudaChannelFormatKindUnsignedBlockCompressed6H,
                    cudaChannelFormatKind.cudaChannelFormatKindSignedBlockCompressed6H,):
        if (d[0].x != 16) or (d[0].y != 16) or (d[0].z != 16) or (d[0].w != 0):
            return cudaErrorInvalidChannelDescriptor
    elif d[0].f in (cudaChannelFormatKind.cudaChannelFormatKindUnsignedNormalized1010102,):
        if (d[0].x != 10) or (d[0].y != 10) or (d[0].z != 10) or (d[0].w != 2):
            return cudaErrorInvalidChannelDescriptor
    else:
        return cudaErrorInvalidChannelDescriptor

    # If Y is non-zero, it must match X
    # If Z is non-zero, it must match Y
    # If W is non-zero, it must match Z
    if (((d[0].y != 0) and (d[0].y != d[0].x)) or
        ((d[0].z != 0) and (d[0].z != d[0].y)) or
        ((d[0].w != 0) and (d[0].w != d[0].z))):
        if d[0].f != cudaChannelFormatKind.cudaChannelFormatKindUnsignedNormalized1010102:
            return cudaErrorInvalidChannelDescriptor
    if case_desc(d, 8, 0, 0, 0, cudaChannelFormatKind.cudaChannelFormatKindSigned):
        numberOfChannels[0] = 1
        format[0] = cydriver.CUarray_format_enum.CU_AD_FORMAT_SIGNED_INT8
    elif case_desc(d, 8, 8, 0, 0, cudaChannelFormatKind.cudaChannelFormatKindSigned):
        numberOfChannels[0] = 2
        format[0] = cydriver.CUarray_format_enum.CU_AD_FORMAT_SIGNED_INT8
    elif case_desc(d, 8, 8, 8, 0, cudaChannelFormatKind.cudaChannelFormatKindSigned):
        numberOfChannels[0] = 3
        format[0] = cydriver.CUarray_format_enum.CU_AD_FORMAT_SIGNED_INT8
    elif case_desc(d, 8, 8, 8, 8, cudaChannelFormatKind.cudaChannelFormatKindSigned):
        numberOfChannels[0] = 4
        format[0] = cydriver.CUarray_format_enum.CU_AD_FORMAT_SIGNED_INT8
    elif case_desc(d, 8, 0, 0, 0, cudaChannelFormatKind.cudaChannelFormatKindUnsigned):
        numberOfChannels[0] = 1
        format[0] = cydriver.CUarray_format_enum.CU_AD_FORMAT_UNSIGNED_INT8
    elif case_desc(d, 8, 8, 0, 0, cudaChannelFormatKind.cudaChannelFormatKindUnsigned):
        numberOfChannels[0] = 2
        format[0] = cydriver.CUarray_format_enum.CU_AD_FORMAT_UNSIGNED_INT8
    elif case_desc(d, 8, 8, 8, 0, cudaChannelFormatKind.cudaChannelFormatKindUnsigned):
        numberOfChannels[0] = 3
        format[0] = cydriver.CUarray_format_enum.CU_AD_FORMAT_UNSIGNED_INT8
    elif case_desc(d, 8, 8, 8, 8, cudaChannelFormatKind.cudaChannelFormatKindUnsigned):
        numberOfChannels[0] = 4
        format[0] = cydriver.CUarray_format_enum.CU_AD_FORMAT_UNSIGNED_INT8
    elif case_desc(d, 16, 0, 0, 0, cudaChannelFormatKind.cudaChannelFormatKindSigned):
        numberOfChannels[0] = 1
        format[0] = cydriver.CUarray_format_enum.CU_AD_FORMAT_SIGNED_INT16
    elif case_desc(d, 16, 16, 0, 0, cudaChannelFormatKind.cudaChannelFormatKindSigned):
        numberOfChannels[0] = 2
        format[0] = cydriver.CUarray_format_enum.CU_AD_FORMAT_SIGNED_INT16
    elif case_desc(d, 16, 16, 16, 0, cudaChannelFormatKind.cudaChannelFormatKindSigned):
        numberOfChannels[0] = 3
        format[0] = cydriver.CUarray_format_enum.CU_AD_FORMAT_SIGNED_INT16
    elif case_desc(d, 16, 16, 16, 16, cudaChannelFormatKind.cudaChannelFormatKindSigned):
        numberOfChannels[0] = 4
        format[0] = cydriver.CUarray_format_enum.CU_AD_FORMAT_SIGNED_INT16
    elif case_desc(d, 16, 0, 0, 0, cudaChannelFormatKind.cudaChannelFormatKindUnsigned):
        numberOfChannels[0] = 1
        format[0] = cydriver.CUarray_format_enum.CU_AD_FORMAT_UNSIGNED_INT16
    elif case_desc(d, 16, 16, 0, 0, cudaChannelFormatKind.cudaChannelFormatKindUnsigned):
        numberOfChannels[0] = 2
        format[0] = cydriver.CUarray_format_enum.CU_AD_FORMAT_UNSIGNED_INT16
    elif case_desc(d, 16, 16, 16, 0, cudaChannelFormatKind.cudaChannelFormatKindUnsigned):
        numberOfChannels[0] = 3
        format[0] = cydriver.CUarray_format_enum.CU_AD_FORMAT_UNSIGNED_INT16
    elif case_desc(d, 16, 16, 16, 16, cudaChannelFormatKind.cudaChannelFormatKindUnsigned):
        numberOfChannels[0] = 4
        format[0] = cydriver.CUarray_format_enum.CU_AD_FORMAT_UNSIGNED_INT16
    elif case_desc(d, 32, 0, 0, 0, cudaChannelFormatKind.cudaChannelFormatKindSigned):
        numberOfChannels[0] = 1
        format[0] = cydriver.CUarray_format_enum.CU_AD_FORMAT_SIGNED_INT32
    elif case_desc(d, 32, 32, 0, 0, cudaChannelFormatKind.cudaChannelFormatKindSigned):
        numberOfChannels[0] = 2
        format[0] = cydriver.CUarray_format_enum.CU_AD_FORMAT_SIGNED_INT32
    elif case_desc(d, 32, 32, 32, 0, cudaChannelFormatKind.cudaChannelFormatKindSigned):
        numberOfChannels[0] = 3
        format[0] = cydriver.CUarray_format_enum.CU_AD_FORMAT_SIGNED_INT32
    elif case_desc(d, 32, 32, 32, 32, cudaChannelFormatKind.cudaChannelFormatKindSigned):
        numberOfChannels[0] = 4
        format[0] = cydriver.CUarray_format_enum.CU_AD_FORMAT_SIGNED_INT32
    elif case_desc(d, 32, 0, 0, 0, cudaChannelFormatKind.cudaChannelFormatKindUnsigned):
        numberOfChannels[0] = 1
        format[0] = cydriver.CUarray_format_enum.CU_AD_FORMAT_UNSIGNED_INT32
    elif case_desc(d, 32, 32, 0, 0, cudaChannelFormatKind.cudaChannelFormatKindUnsigned):
        numberOfChannels[0] = 2
        format[0] = cydriver.CUarray_format_enum.CU_AD_FORMAT_UNSIGNED_INT32
    elif case_desc(d, 32, 32, 32, 0, cudaChannelFormatKind.cudaChannelFormatKindUnsigned):
        numberOfChannels[0] = 3
        format[0] = cydriver.CUarray_format_enum.CU_AD_FORMAT_UNSIGNED_INT32
    elif case_desc(d, 32, 32, 32, 32, cudaChannelFormatKind.cudaChannelFormatKindUnsigned):
        numberOfChannels[0] = 4
        format[0] = cydriver.CUarray_format_enum.CU_AD_FORMAT_UNSIGNED_INT32
    elif case_desc(d, 16, 0, 0, 0, cudaChannelFormatKind.cudaChannelFormatKindFloat):
        numberOfChannels[0] = 1
        format[0] = cydriver.CUarray_format_enum.CU_AD_FORMAT_HALF
    elif case_desc(d, 16, 16, 0, 0, cudaChannelFormatKind.cudaChannelFormatKindFloat):
        numberOfChannels[0] = 2
        format[0] = cydriver.CUarray_format_enum.CU_AD_FORMAT_HALF
    elif case_desc(d, 16, 16, 16, 0, cudaChannelFormatKind.cudaChannelFormatKindFloat):
        numberOfChannels[0] = 3
        format[0] = cydriver.CUarray_format_enum.CU_AD_FORMAT_HALF
    elif case_desc(d, 16, 16, 16, 16, cudaChannelFormatKind.cudaChannelFormatKindFloat):
        numberOfChannels[0] = 4
        format[0] = cydriver.CUarray_format_enum.CU_AD_FORMAT_HALF
    elif case_desc(d, 32, 0, 0, 0, cudaChannelFormatKind.cudaChannelFormatKindFloat):
        numberOfChannels[0] = 1
        format[0] = cydriver.CUarray_format_enum.CU_AD_FORMAT_FLOAT
    elif case_desc(d, 32, 32, 0, 0, cudaChannelFormatKind.cudaChannelFormatKindFloat):
        numberOfChannels[0] = 2
        format[0] = cydriver.CUarray_format_enum.CU_AD_FORMAT_FLOAT
    elif case_desc(d, 32, 32, 32, 0, cudaChannelFormatKind.cudaChannelFormatKindFloat):
        numberOfChannels[0] = 3
        format[0] = cydriver.CUarray_format_enum.CU_AD_FORMAT_FLOAT
    elif case_desc(d, 32, 32, 32, 32, cudaChannelFormatKind.cudaChannelFormatKindFloat):
        numberOfChannels[0] = 4
        format[0] = cydriver.CUarray_format_enum.CU_AD_FORMAT_FLOAT
    elif case_desc(d, 8, 8, 8, 0, cudaChannelFormatKind.cudaChannelFormatKindNV12):
        numberOfChannels[0] = 3
        format[0] = cydriver.CUarray_format_enum.CU_AD_FORMAT_NV12
    elif case_desc(d, 8, 8, 8, 8, cudaChannelFormatKind.cudaChannelFormatKindUnsignedBlockCompressed1):
        numberOfChannels[0] = 4
        format[0] = cydriver.CUarray_format_enum.CU_AD_FORMAT_BC1_UNORM
    elif case_desc(d, 8, 8, 8, 8, cudaChannelFormatKind.cudaChannelFormatKindUnsignedBlockCompressed1SRGB):
        numberOfChannels[0] = 4
        format[0] = cydriver.CUarray_format_enum.CU_AD_FORMAT_BC1_UNORM_SRGB
    elif case_desc(d, 8, 8, 8, 8, cudaChannelFormatKind.cudaChannelFormatKindUnsignedBlockCompressed2):
        numberOfChannels[0] = 4
        format[0] = cydriver.CUarray_format_enum.CU_AD_FORMAT_BC2_UNORM
    elif case_desc(d, 8, 8, 8, 8, cudaChannelFormatKind.cudaChannelFormatKindUnsignedBlockCompressed2SRGB):
        numberOfChannels[0] = 4
        format[0] = cydriver.CUarray_format_enum.CU_AD_FORMAT_BC2_UNORM_SRGB
    elif case_desc(d, 8, 8, 8, 8, cudaChannelFormatKind.cudaChannelFormatKindUnsignedBlockCompressed3):
        numberOfChannels[0] = 4
        format[0] = cydriver.CUarray_format_enum.CU_AD_FORMAT_BC3_UNORM
    elif case_desc(d, 8, 8, 8, 8, cudaChannelFormatKind.cudaChannelFormatKindUnsignedBlockCompressed3SRGB):
        numberOfChannels[0] = 4
        format[0] = cydriver.CUarray_format_enum.CU_AD_FORMAT_BC3_UNORM_SRGB
    elif case_desc(d, 8, 0, 0, 0, cudaChannelFormatKind.cudaChannelFormatKindUnsignedBlockCompressed4):
        numberOfChannels[0] = 1
        format[0] = cydriver.CUarray_format_enum.CU_AD_FORMAT_BC4_UNORM
    elif case_desc(d, 8, 0, 0, 0, cudaChannelFormatKind.cudaChannelFormatKindSignedBlockCompressed4):
        numberOfChannels[0] = 1
        format[0] = cydriver.CUarray_format_enum.CU_AD_FORMAT_BC4_SNORM
    elif case_desc(d, 8, 8, 0, 0, cudaChannelFormatKind.cudaChannelFormatKindUnsignedBlockCompressed5):
        numberOfChannels[0] = 2
        format[0] = cydriver.CUarray_format_enum.CU_AD_FORMAT_BC5_UNORM
    elif case_desc(d, 8, 8, 0, 0, cudaChannelFormatKind.cudaChannelFormatKindSignedBlockCompressed5):
        numberOfChannels[0] = 2
        format[0] = cydriver.CUarray_format_enum.CU_AD_FORMAT_BC5_SNORM
    elif case_desc(d, 16, 16, 16, 0, cudaChannelFormatKind.cudaChannelFormatKindUnsignedBlockCompressed6H):
        numberOfChannels[0] = 3
        format[0] = cydriver.CUarray_format_enum.CU_AD_FORMAT_BC6H_UF16
    elif case_desc(d, 16, 16, 16, 0, cudaChannelFormatKind.cudaChannelFormatKindSignedBlockCompressed6H):
        numberOfChannels[0] = 3
        format[0] = cydriver.CUarray_format_enum.CU_AD_FORMAT_BC6H_SF16
    elif case_desc(d, 8, 8, 8, 8, cudaChannelFormatKind.cudaChannelFormatKindUnsignedBlockCompressed7):
        numberOfChannels[0] = 4
        format[0] = cydriver.CUarray_format_enum.CU_AD_FORMAT_BC7_UNORM
    elif case_desc(d, 8, 8, 8, 8, cudaChannelFormatKind.cudaChannelFormatKindUnsignedBlockCompressed7SRGB):
        numberOfChannels[0] = 4
        format[0] = cydriver.CUarray_format_enum.CU_AD_FORMAT_BC7_UNORM_SRGB
    elif case_desc(d, 10, 10, 10, 2, cudaChannelFormatKind.cudaChannelFormatKindUnsignedNormalized1010102):
        numberOfChannels[0] = 4
        format[0] = cydriver.CUarray_format_enum.CU_AD_FORMAT_UNORM_INT_101010_2
    else:
        return cudaErrorInvalidChannelDescriptor

    if d[0].f in (cudaChannelFormatKind.cudaChannelFormatKindNV12,
                  cudaChannelFormatKind.cudaChannelFormatKindUnsignedBlockCompressed6H,
                  cudaChannelFormatKind.cudaChannelFormatKindSignedBlockCompressed6H,):
        if numberOfChannels[0] != 3:
            return cudaErrorInvalidChannelDescriptor
    else:
        if (numberOfChannels[0] != 1) and (numberOfChannels[0] != 2) and (numberOfChannels[0] != 4):
            return cudaErrorInvalidChannelDescriptor
    return cudaSuccess


cdef cudaError_t getChannelFormatDescFromDriverDesc(cudaChannelFormatDesc* pRuntimeDesc, size_t* pDepth, size_t* pHeight, size_t* pWidth, const cydriver.CUDA_ARRAY3D_DESCRIPTOR_v2* pDriverDesc) except ?cudaErrorCallRequiresNewerDriver nogil:

    cdef int channel_size = 0
    if pDriverDesc[0].Format == cydriver.CU_AD_FORMAT_UNSIGNED_INT8:
        pRuntimeDesc[0].f = cudaChannelFormatKind.cudaChannelFormatKindUnsigned
        channel_size = 8
    elif pDriverDesc[0].Format == cydriver.CU_AD_FORMAT_UNSIGNED_INT16:
        pRuntimeDesc[0].f = cudaChannelFormatKind.cudaChannelFormatKindUnsigned
        channel_size = 16
    elif pDriverDesc[0].Format == cydriver.CU_AD_FORMAT_UNSIGNED_INT32:
        pRuntimeDesc[0].f = cudaChannelFormatKind.cudaChannelFormatKindUnsigned
        channel_size = 32
    elif pDriverDesc[0].Format == cydriver.CU_AD_FORMAT_SIGNED_INT8:
        pRuntimeDesc[0].f = cudaChannelFormatKind.cudaChannelFormatKindSigned
        channel_size = 8
    elif pDriverDesc[0].Format == cydriver.CU_AD_FORMAT_SIGNED_INT16:
        pRuntimeDesc[0].f = cudaChannelFormatKind.cudaChannelFormatKindSigned
        channel_size = 16
    elif pDriverDesc[0].Format == cydriver.CU_AD_FORMAT_SIGNED_INT32:
        pRuntimeDesc[0].f = cudaChannelFormatKind.cudaChannelFormatKindSigned
        channel_size = 32
    elif pDriverDesc[0].Format == cydriver.CU_AD_FORMAT_HALF:
        pRuntimeDesc[0].f = cudaChannelFormatKind.cudaChannelFormatKindFloat
        channel_size = 16
    elif pDriverDesc[0].Format == cydriver.CU_AD_FORMAT_FLOAT:
        pRuntimeDesc[0].f = cudaChannelFormatKind.cudaChannelFormatKindFloat
        channel_size = 32
    elif pDriverDesc[0].Format == cydriver.CU_AD_FORMAT_NV12:
        pRuntimeDesc[0].f = cudaChannelFormatKind.cudaChannelFormatKindNV12
        channel_size = 8
    elif pDriverDesc[0].Format == cydriver.CU_AD_FORMAT_UNORM_INT8X1:
        pRuntimeDesc[0].f = cudaChannelFormatKind.cudaChannelFormatKindUnsignedNormalized8X1
        channel_size = 8
    elif pDriverDesc[0].Format == cydriver.CU_AD_FORMAT_UNORM_INT8X2:
        pRuntimeDesc[0].f = cudaChannelFormatKind.cudaChannelFormatKindUnsignedNormalized8X2
        channel_size = 8
    elif pDriverDesc[0].Format == cydriver.CU_AD_FORMAT_UNORM_INT8X4:
        pRuntimeDesc[0].f = cudaChannelFormatKind.cudaChannelFormatKindUnsignedNormalized8X4
        channel_size = 8
    elif pDriverDesc[0].Format == cydriver.CU_AD_FORMAT_SNORM_INT8X1:
        pRuntimeDesc[0].f = cudaChannelFormatKind.cudaChannelFormatKindSignedNormalized8X1
        channel_size = 8
    elif pDriverDesc[0].Format == cydriver.CU_AD_FORMAT_SNORM_INT8X2:
        pRuntimeDesc[0].f = cudaChannelFormatKind.cudaChannelFormatKindSignedNormalized8X2
        channel_size = 8
    elif pDriverDesc[0].Format == cydriver.CU_AD_FORMAT_SNORM_INT8X4:
        pRuntimeDesc[0].f = cudaChannelFormatKind.cudaChannelFormatKindSignedNormalized8X4
        channel_size = 8
    elif pDriverDesc[0].Format == cydriver.CU_AD_FORMAT_UNORM_INT16X1:
        pRuntimeDesc[0].f = cudaChannelFormatKind.cudaChannelFormatKindUnsignedNormalized16X1
        channel_size = 16
    elif pDriverDesc[0].Format == cydriver.CU_AD_FORMAT_UNORM_INT16X2:
        pRuntimeDesc[0].f = cudaChannelFormatKind.cudaChannelFormatKindUnsignedNormalized16X2
        channel_size = 16
    elif pDriverDesc[0].Format == cydriver.CU_AD_FORMAT_UNORM_INT16X4:
        pRuntimeDesc[0].f = cudaChannelFormatKind.cudaChannelFormatKindUnsignedNormalized16X4
        channel_size = 16
    elif pDriverDesc[0].Format == cydriver.CU_AD_FORMAT_SNORM_INT16X1:
        pRuntimeDesc[0].f = cudaChannelFormatKind.cudaChannelFormatKindSignedNormalized16X1
        channel_size = 16
    elif pDriverDesc[0].Format == cydriver.CU_AD_FORMAT_SNORM_INT16X2:
        pRuntimeDesc[0].f = cudaChannelFormatKind.cudaChannelFormatKindSignedNormalized16X2
        channel_size = 16
    elif pDriverDesc[0].Format == cydriver.CU_AD_FORMAT_SNORM_INT16X4:
        pRuntimeDesc[0].f = cudaChannelFormatKind.cudaChannelFormatKindSignedNormalized16X4
        channel_size = 16
    elif pDriverDesc[0].Format == cydriver.CU_AD_FORMAT_BC1_UNORM:
        pRuntimeDesc[0].f = cudaChannelFormatKind.cudaChannelFormatKindUnsignedBlockCompressed1
        channel_size = 8
    elif pDriverDesc[0].Format == cydriver.CU_AD_FORMAT_BC1_UNORM_SRGB:
        pRuntimeDesc[0].f = cudaChannelFormatKind.cudaChannelFormatKindUnsignedBlockCompressed1SRGB
        channel_size = 8
    elif pDriverDesc[0].Format == cydriver.CU_AD_FORMAT_BC2_UNORM:
        pRuntimeDesc[0].f = cudaChannelFormatKind.cudaChannelFormatKindUnsignedBlockCompressed2
        channel_size = 8
    elif pDriverDesc[0].Format == cydriver.CU_AD_FORMAT_BC2_UNORM_SRGB:
        pRuntimeDesc[0].f = cudaChannelFormatKind.cudaChannelFormatKindUnsignedBlockCompressed2SRGB
        channel_size = 8
    elif pDriverDesc[0].Format == cydriver.CU_AD_FORMAT_BC3_UNORM:
        pRuntimeDesc[0].f = cudaChannelFormatKind.cudaChannelFormatKindUnsignedBlockCompressed3
        channel_size = 8
    elif pDriverDesc[0].Format == cydriver.CU_AD_FORMAT_BC3_UNORM_SRGB:
        pRuntimeDesc[0].f = cudaChannelFormatKind.cudaChannelFormatKindUnsignedBlockCompressed3SRGB
        channel_size = 8
    elif pDriverDesc[0].Format == cydriver.CU_AD_FORMAT_BC4_UNORM:
        pRuntimeDesc[0].f = cudaChannelFormatKind.cudaChannelFormatKindUnsignedBlockCompressed4
        channel_size = 8
    elif pDriverDesc[0].Format == cydriver.CU_AD_FORMAT_BC4_SNORM:
        pRuntimeDesc[0].f = cudaChannelFormatKind.cudaChannelFormatKindSignedBlockCompressed4
        channel_size = 8
    elif pDriverDesc[0].Format == cydriver.CU_AD_FORMAT_BC5_UNORM:
        pRuntimeDesc[0].f = cudaChannelFormatKind.cudaChannelFormatKindUnsignedBlockCompressed5
        channel_size = 8
    elif pDriverDesc[0].Format == cydriver.CU_AD_FORMAT_BC5_SNORM:
        pRuntimeDesc[0].f = cudaChannelFormatKind.cudaChannelFormatKindSignedBlockCompressed5
        channel_size = 8
    elif pDriverDesc[0].Format == cydriver.CU_AD_FORMAT_BC6H_UF16:
        pRuntimeDesc[0].f = cudaChannelFormatKind.cudaChannelFormatKindUnsignedBlockCompressed6H
        channel_size = 16
    elif pDriverDesc[0].Format == cydriver.CU_AD_FORMAT_BC6H_SF16:
        pRuntimeDesc[0].f = cudaChannelFormatKind.cudaChannelFormatKindSignedBlockCompressed6H
        channel_size = 16
    elif pDriverDesc[0].Format == cydriver.CU_AD_FORMAT_BC7_UNORM:
        pRuntimeDesc[0].f = cudaChannelFormatKind.cudaChannelFormatKindUnsignedBlockCompressed7
        channel_size = 8
    elif pDriverDesc[0].Format == cydriver.CU_AD_FORMAT_BC7_UNORM_SRGB:
        pRuntimeDesc[0].f = cudaChannelFormatKind.cudaChannelFormatKindUnsignedBlockCompressed7SRGB
        channel_size = 8
    elif pDriverDesc[0].Format == cydriver.CU_AD_FORMAT_UNORM_INT_101010_2:
        pRuntimeDesc[0].f = cudaChannelFormatKind.cudaChannelFormatKindUnsignedNormalized1010102
    else:
        return cudaErrorInvalidChannelDescriptor

    # populate bits per channel
    pRuntimeDesc[0].x = 0
    pRuntimeDesc[0].y = 0
    pRuntimeDesc[0].z = 0
    pRuntimeDesc[0].w = 0

    if pDriverDesc[0].Format == cydriver.CU_AD_FORMAT_UNORM_INT_101010_2 and pDriverDesc[0].NumChannels == 4:
        pRuntimeDesc[0].w = 2
        pRuntimeDesc[0].z = 10
        pRuntimeDesc[0].y = 10
        pRuntimeDesc[0].x = 10
    else:
        if pDriverDesc[0].NumChannels >= 4:
            pRuntimeDesc[0].w = channel_size
        if pDriverDesc[0].NumChannels >= 3:
            pRuntimeDesc[0].z = channel_size
        if pDriverDesc[0].NumChannels >= 2:
            pRuntimeDesc[0].y = channel_size
        if pDriverDesc[0].NumChannels >= 1:
            pRuntimeDesc[0].x = channel_size

    if pDriverDesc[0].NumChannels not in (4, 3, 2, 1):
        return cudaErrorInvalidChannelDescriptor

    # populate dimensions
    if pDepth != NULL:
        pDepth[0]  = pDriverDesc[0].Depth
    if pHeight != NULL:
        pHeight[0] = pDriverDesc[0].Height
    if pWidth != NULL:
        pWidth[0]  = pDriverDesc[0].Width
    return cudaSuccess


cdef cudaError_t getDriverEglFrame(cydriver.CUeglFrame *cuEglFrame, cudaEglFrame eglFrame) except ?cudaErrorCallRequiresNewerDriver nogil:
    cdef cudaError_t err = cudaSuccess
    cdef unsigned int i = 0

    err = getDescInfo(&eglFrame.planeDesc[0].channelDesc, <int*>&cuEglFrame[0].numChannels, &cuEglFrame[0].cuFormat)
    if err != cudaSuccess:
        return err
    for i in range(eglFrame.planeCount):
        if eglFrame.frameType == cudaEglFrameTypeArray:
            cuEglFrame[0].frame.pArray[i] = <cydriver.CUarray>eglFrame.frame.pArray[i]
        else:
            cuEglFrame[0].frame.pPitch[i] = eglFrame.frame.pPitch[i].ptr
    cuEglFrame[0].width = eglFrame.planeDesc[0].width
    cuEglFrame[0].height = eglFrame.planeDesc[0].height
    cuEglFrame[0].depth = eglFrame.planeDesc[0].depth
    cuEglFrame[0].pitch = eglFrame.planeDesc[0].pitch
    cuEglFrame[0].planeCount = eglFrame.planeCount
    if eglFrame.eglColorFormat == cudaEglColorFormatYUV420Planar:
        cuEglFrame[0].eglColorFormat = cydriver.CUeglColorFormat_enum.CU_EGL_COLOR_FORMAT_YUV420_PLANAR
    elif eglFrame.eglColorFormat == cudaEglColorFormatYUV420SemiPlanar:
        cuEglFrame[0].eglColorFormat = cydriver.CUeglColorFormat_enum.CU_EGL_COLOR_FORMAT_YUV420_SEMIPLANAR
    elif eglFrame.eglColorFormat == cudaEglColorFormatYUV422Planar:
        cuEglFrame[0].eglColorFormat = cydriver.CUeglColorFormat_enum.CU_EGL_COLOR_FORMAT_YUV422_PLANAR
    elif eglFrame.eglColorFormat == cudaEglColorFormatYUV422SemiPlanar:
        cuEglFrame[0].eglColorFormat = cydriver.CUeglColorFormat_enum.CU_EGL_COLOR_FORMAT_YUV422_SEMIPLANAR
    elif eglFrame.eglColorFormat == cudaEglColorFormatYUV444Planar:
        cuEglFrame[0].eglColorFormat = cydriver.CUeglColorFormat_enum.CU_EGL_COLOR_FORMAT_YUV444_PLANAR
    elif eglFrame.eglColorFormat == cudaEglColorFormatYUV444SemiPlanar:
        cuEglFrame[0].eglColorFormat = cydriver.CUeglColorFormat_enum.CU_EGL_COLOR_FORMAT_YUV444_SEMIPLANAR
    elif eglFrame.eglColorFormat == cudaEglColorFormatYUYV422:
        cuEglFrame[0].eglColorFormat = cydriver.CUeglColorFormat_enum.CU_EGL_COLOR_FORMAT_YUYV_422
    elif eglFrame.eglColorFormat == cudaEglColorFormatUYVY422:
        cuEglFrame[0].eglColorFormat = cydriver.CUeglColorFormat_enum.CU_EGL_COLOR_FORMAT_UYVY_422
    elif eglFrame.eglColorFormat == cudaEglColorFormatUYVY709:
        cuEglFrame[0].eglColorFormat = cydriver.CUeglColorFormat_enum.CU_EGL_COLOR_FORMAT_UYVY_709
    elif eglFrame.eglColorFormat == cudaEglColorFormatUYVY709_ER:
        cuEglFrame[0].eglColorFormat = cydriver.CUeglColorFormat_enum.CU_EGL_COLOR_FORMAT_UYVY_709_ER
    elif eglFrame.eglColorFormat == cudaEglColorFormatUYVY2020:
        cuEglFrame[0].eglColorFormat = cydriver.CUeglColorFormat_enum.CU_EGL_COLOR_FORMAT_UYVY_2020
    elif eglFrame.eglColorFormat == cudaEglColorFormatARGB:
        cuEglFrame[0].eglColorFormat = cydriver.CUeglColorFormat_enum.CU_EGL_COLOR_FORMAT_ARGB
    elif eglFrame.eglColorFormat == cudaEglColorFormatRGBA:
        cuEglFrame[0].eglColorFormat = cydriver.CUeglColorFormat_enum.CU_EGL_COLOR_FORMAT_RGBA
    elif eglFrame.eglColorFormat == cudaEglColorFormatABGR:
        cuEglFrame[0].eglColorFormat = cydriver.CUeglColorFormat_enum.CU_EGL_COLOR_FORMAT_ABGR
    elif eglFrame.eglColorFormat == cudaEglColorFormatBGRA:
        cuEglFrame[0].eglColorFormat = cydriver.CUeglColorFormat_enum.CU_EGL_COLOR_FORMAT_BGRA
    elif eglFrame.eglColorFormat == cudaEglColorFormatL:
        cuEglFrame[0].eglColorFormat = cydriver.CUeglColorFormat_enum.CU_EGL_COLOR_FORMAT_L
    elif eglFrame.eglColorFormat == cudaEglColorFormatR:
        cuEglFrame[0].eglColorFormat = cydriver.CUeglColorFormat_enum.CU_EGL_COLOR_FORMAT_R
    elif eglFrame.eglColorFormat == cudaEglColorFormatA:
        cuEglFrame[0].eglColorFormat = cydriver.CUeglColorFormat_enum.CU_EGL_COLOR_FORMAT_A
    elif eglFrame.eglColorFormat == cudaEglColorFormatRG:
        cuEglFrame[0].eglColorFormat = cydriver.CUeglColorFormat_enum.CU_EGL_COLOR_FORMAT_RG
    elif eglFrame.eglColorFormat == cudaEglColorFormatAYUV:
        cuEglFrame[0].eglColorFormat = cydriver.CUeglColorFormat_enum.CU_EGL_COLOR_FORMAT_AYUV
    elif eglFrame.eglColorFormat == cudaEglColorFormatYVU444SemiPlanar:
        cuEglFrame[0].eglColorFormat = cydriver.CUeglColorFormat_enum.CU_EGL_COLOR_FORMAT_YVU444_SEMIPLANAR
    elif eglFrame.eglColorFormat == cudaEglColorFormatYVU422SemiPlanar:
        cuEglFrame[0].eglColorFormat = cydriver.CUeglColorFormat_enum.CU_EGL_COLOR_FORMAT_YVU422_SEMIPLANAR
    elif eglFrame.eglColorFormat == cudaEglColorFormatYVU420SemiPlanar:
        cuEglFrame[0].eglColorFormat = cydriver.CUeglColorFormat_enum.CU_EGL_COLOR_FORMAT_YVU420_SEMIPLANAR
    elif eglFrame.eglColorFormat == cudaEglColorFormatY10V10U10_444SemiPlanar:
        cuEglFrame[0].eglColorFormat = cydriver.CUeglColorFormat_enum.CU_EGL_COLOR_FORMAT_Y10V10U10_444_SEMIPLANAR
    elif eglFrame.eglColorFormat == cudaEglColorFormatY10V10U10_420SemiPlanar:
        cuEglFrame[0].eglColorFormat = cydriver.CUeglColorFormat_enum.CU_EGL_COLOR_FORMAT_Y10V10U10_420_SEMIPLANAR
    elif eglFrame.eglColorFormat == cudaEglColorFormatY12V12U12_444SemiPlanar:
        cuEglFrame[0].eglColorFormat = cydriver.CUeglColorFormat_enum.CU_EGL_COLOR_FORMAT_Y12V12U12_444_SEMIPLANAR
    elif eglFrame.eglColorFormat == cudaEglColorFormatY12V12U12_420SemiPlanar:
        cuEglFrame[0].eglColorFormat = cydriver.CUeglColorFormat_enum.CU_EGL_COLOR_FORMAT_Y12V12U12_420_SEMIPLANAR
    elif eglFrame.eglColorFormat == cudaEglColorFormatVYUY_ER:
        cuEglFrame[0].eglColorFormat = cydriver.CUeglColorFormat_enum.CU_EGL_COLOR_FORMAT_VYUY_ER
    elif eglFrame.eglColorFormat == cudaEglColorFormatUYVY_ER:
        cuEglFrame[0].eglColorFormat = cydriver.CUeglColorFormat_enum.CU_EGL_COLOR_FORMAT_UYVY_ER
    elif eglFrame.eglColorFormat == cudaEglColorFormatYUYV_ER:
        cuEglFrame[0].eglColorFormat = cydriver.CUeglColorFormat_enum.CU_EGL_COLOR_FORMAT_YUYV_ER
    elif eglFrame.eglColorFormat == cudaEglColorFormatYVYU_ER:
        cuEglFrame[0].eglColorFormat = cydriver.CUeglColorFormat_enum.CU_EGL_COLOR_FORMAT_YVYU_ER
    elif eglFrame.eglColorFormat == cudaEglColorFormatYUVA_ER:
        cuEglFrame[0].eglColorFormat = cydriver.CUeglColorFormat_enum.CU_EGL_COLOR_FORMAT_YUVA_ER
    elif eglFrame.eglColorFormat == cudaEglColorFormatAYUV_ER:
        cuEglFrame[0].eglColorFormat = cydriver.CUeglColorFormat_enum.CU_EGL_COLOR_FORMAT_AYUV_ER
    elif eglFrame.eglColorFormat == cudaEglColorFormatYUV444Planar_ER:
        cuEglFrame[0].eglColorFormat = cydriver.CUeglColorFormat_enum.CU_EGL_COLOR_FORMAT_YUV444_PLANAR_ER
    elif eglFrame.eglColorFormat == cudaEglColorFormatYUV422Planar_ER:
        cuEglFrame[0].eglColorFormat = cydriver.CUeglColorFormat_enum.CU_EGL_COLOR_FORMAT_YUV422_PLANAR_ER
    elif eglFrame.eglColorFormat == cudaEglColorFormatYUV420Planar_ER:
        cuEglFrame[0].eglColorFormat = cydriver.CUeglColorFormat_enum.CU_EGL_COLOR_FORMAT_YUV420_PLANAR_ER
    elif eglFrame.eglColorFormat == cudaEglColorFormatYUV444SemiPlanar_ER:
        cuEglFrame[0].eglColorFormat = cydriver.CUeglColorFormat_enum.CU_EGL_COLOR_FORMAT_YUV444_SEMIPLANAR_ER
    elif eglFrame.eglColorFormat == cudaEglColorFormatYUV422SemiPlanar_ER:
        cuEglFrame[0].eglColorFormat = cydriver.CUeglColorFormat_enum.CU_EGL_COLOR_FORMAT_YUV422_SEMIPLANAR_ER
    elif eglFrame.eglColorFormat == cudaEglColorFormatYUV420SemiPlanar_ER:
        cuEglFrame[0].eglColorFormat = cydriver.CUeglColorFormat_enum.CU_EGL_COLOR_FORMAT_YUV420_SEMIPLANAR_ER
    elif eglFrame.eglColorFormat == cudaEglColorFormatYVU444Planar_ER:
        cuEglFrame[0].eglColorFormat = cydriver.CUeglColorFormat_enum.CU_EGL_COLOR_FORMAT_YVU444_PLANAR_ER
    elif eglFrame.eglColorFormat == cudaEglColorFormatYVU422Planar_ER:
        cuEglFrame[0].eglColorFormat = cydriver.CUeglColorFormat_enum.CU_EGL_COLOR_FORMAT_YVU422_PLANAR_ER
    elif eglFrame.eglColorFormat == cudaEglColorFormatYVU420Planar_ER:
        cuEglFrame[0].eglColorFormat = cydriver.CUeglColorFormat_enum.CU_EGL_COLOR_FORMAT_YVU420_PLANAR_ER
    elif eglFrame.eglColorFormat == cudaEglColorFormatYVU444SemiPlanar_ER:
        cuEglFrame[0].eglColorFormat = cydriver.CUeglColorFormat_enum.CU_EGL_COLOR_FORMAT_YVU444_SEMIPLANAR_ER
    elif eglFrame.eglColorFormat == cudaEglColorFormatYVU422SemiPlanar_ER:
        cuEglFrame[0].eglColorFormat = cydriver.CUeglColorFormat_enum.CU_EGL_COLOR_FORMAT_YVU422_SEMIPLANAR_ER
    elif eglFrame.eglColorFormat == cudaEglColorFormatYVU420SemiPlanar_ER:
        cuEglFrame[0].eglColorFormat = cydriver.CUeglColorFormat_enum.CU_EGL_COLOR_FORMAT_YVU420_SEMIPLANAR_ER
    elif eglFrame.eglColorFormat == cudaEglColorFormatBayerRGGB:
        cuEglFrame[0].eglColorFormat = cydriver.CUeglColorFormat_enum.CU_EGL_COLOR_FORMAT_BAYER_RGGB
    elif eglFrame.eglColorFormat == cudaEglColorFormatBayerBGGR:
        cuEglFrame[0].eglColorFormat = cydriver.CUeglColorFormat_enum.CU_EGL_COLOR_FORMAT_BAYER_BGGR
    elif eglFrame.eglColorFormat == cudaEglColorFormatBayerGRBG:
        cuEglFrame[0].eglColorFormat = cydriver.CUeglColorFormat_enum.CU_EGL_COLOR_FORMAT_BAYER_GRBG
    elif eglFrame.eglColorFormat == cudaEglColorFormatBayerGBRG:
        cuEglFrame[0].eglColorFormat = cydriver.CUeglColorFormat_enum.CU_EGL_COLOR_FORMAT_BAYER_GBRG
    elif eglFrame.eglColorFormat == cudaEglColorFormatBayer10RGGB:
        cuEglFrame[0].eglColorFormat = cydriver.CUeglColorFormat_enum.CU_EGL_COLOR_FORMAT_BAYER10_RGGB
    elif eglFrame.eglColorFormat == cudaEglColorFormatBayer10BGGR:
        cuEglFrame[0].eglColorFormat = cydriver.CUeglColorFormat_enum.CU_EGL_COLOR_FORMAT_BAYER10_BGGR
    elif eglFrame.eglColorFormat == cudaEglColorFormatBayer10GRBG:
        cuEglFrame[0].eglColorFormat = cydriver.CUeglColorFormat_enum.CU_EGL_COLOR_FORMAT_BAYER10_GRBG
    elif eglFrame.eglColorFormat == cudaEglColorFormatBayer10GBRG:
        cuEglFrame[0].eglColorFormat = cydriver.CUeglColorFormat_enum.CU_EGL_COLOR_FORMAT_BAYER10_GBRG
    elif eglFrame.eglColorFormat == cudaEglColorFormatBayer12RGGB:
        cuEglFrame[0].eglColorFormat = cydriver.CUeglColorFormat_enum.CU_EGL_COLOR_FORMAT_BAYER12_RGGB
    elif eglFrame.eglColorFormat == cudaEglColorFormatBayer12BGGR:
        cuEglFrame[0].eglColorFormat = cydriver.CUeglColorFormat_enum.CU_EGL_COLOR_FORMAT_BAYER12_BGGR
    elif eglFrame.eglColorFormat == cudaEglColorFormatBayer12GRBG:
        cuEglFrame[0].eglColorFormat = cydriver.CUeglColorFormat_enum.CU_EGL_COLOR_FORMAT_BAYER12_GRBG
    elif eglFrame.eglColorFormat == cudaEglColorFormatBayer12GBRG:
        cuEglFrame[0].eglColorFormat = cydriver.CUeglColorFormat_enum.CU_EGL_COLOR_FORMAT_BAYER12_GBRG
    elif eglFrame.eglColorFormat == cudaEglColorFormatBayer14RGGB:
        cuEglFrame[0].eglColorFormat = cydriver.CUeglColorFormat_enum.CU_EGL_COLOR_FORMAT_BAYER14_RGGB
    elif eglFrame.eglColorFormat == cudaEglColorFormatBayer14BGGR:
        cuEglFrame[0].eglColorFormat = cydriver.CUeglColorFormat_enum.CU_EGL_COLOR_FORMAT_BAYER14_BGGR
    elif eglFrame.eglColorFormat == cudaEglColorFormatBayer14GRBG:
        cuEglFrame[0].eglColorFormat = cydriver.CUeglColorFormat_enum.CU_EGL_COLOR_FORMAT_BAYER14_GRBG
    elif eglFrame.eglColorFormat == cudaEglColorFormatBayer14GBRG:
        cuEglFrame[0].eglColorFormat = cydriver.CUeglColorFormat_enum.CU_EGL_COLOR_FORMAT_BAYER14_GBRG
    elif eglFrame.eglColorFormat == cudaEglColorFormatBayer20RGGB:
        cuEglFrame[0].eglColorFormat = cydriver.CUeglColorFormat_enum.CU_EGL_COLOR_FORMAT_BAYER20_RGGB
    elif eglFrame.eglColorFormat == cudaEglColorFormatBayer20BGGR:
        cuEglFrame[0].eglColorFormat = cydriver.CUeglColorFormat_enum.CU_EGL_COLOR_FORMAT_BAYER20_BGGR
    elif eglFrame.eglColorFormat == cudaEglColorFormatBayer20GRBG:
        cuEglFrame[0].eglColorFormat = cydriver.CUeglColorFormat_enum.CU_EGL_COLOR_FORMAT_BAYER20_GRBG
    elif eglFrame.eglColorFormat == cudaEglColorFormatBayer20GBRG:
        cuEglFrame[0].eglColorFormat = cydriver.CUeglColorFormat_enum.CU_EGL_COLOR_FORMAT_BAYER20_GBRG
    elif eglFrame.eglColorFormat == cudaEglColorFormatBayerIspRGGB:
        cuEglFrame[0].eglColorFormat = cydriver.CUeglColorFormat_enum.CU_EGL_COLOR_FORMAT_BAYER_ISP_RGGB
    elif eglFrame.eglColorFormat == cudaEglColorFormatBayerIspBGGR:
        cuEglFrame[0].eglColorFormat = cydriver.CUeglColorFormat_enum.CU_EGL_COLOR_FORMAT_BAYER_ISP_BGGR
    elif eglFrame.eglColorFormat == cudaEglColorFormatBayerIspGRBG:
        cuEglFrame[0].eglColorFormat = cydriver.CUeglColorFormat_enum.CU_EGL_COLOR_FORMAT_BAYER_ISP_GRBG
    elif eglFrame.eglColorFormat == cudaEglColorFormatBayerIspGBRG:
        cuEglFrame[0].eglColorFormat = cydriver.CUeglColorFormat_enum.CU_EGL_COLOR_FORMAT_BAYER_ISP_GBRG
    elif eglFrame.eglColorFormat == cudaEglColorFormatYVU444Planar:
        cuEglFrame[0].eglColorFormat = cydriver.CUeglColorFormat_enum.CU_EGL_COLOR_FORMAT_YVU444_PLANAR
    elif eglFrame.eglColorFormat == cudaEglColorFormatYVU422Planar:
        cuEglFrame[0].eglColorFormat = cydriver.CUeglColorFormat_enum.CU_EGL_COLOR_FORMAT_YVU422_PLANAR
    elif eglFrame.eglColorFormat == cudaEglColorFormatYVU420Planar:
        cuEglFrame[0].eglColorFormat = cydriver.CUeglColorFormat_enum.CU_EGL_COLOR_FORMAT_YVU420_PLANAR
    elif eglFrame.eglColorFormat == cudaEglColorFormatBayerBCCR:
        cuEglFrame[0].eglColorFormat = cydriver.CUeglColorFormat_enum.CU_EGL_COLOR_FORMAT_BAYER_BCCR
    elif eglFrame.eglColorFormat == cudaEglColorFormatBayerRCCB:
        cuEglFrame[0].eglColorFormat = cydriver.CUeglColorFormat_enum.CU_EGL_COLOR_FORMAT_BAYER_RCCB
    elif eglFrame.eglColorFormat == cudaEglColorFormatBayerCRBC:
        cuEglFrame[0].eglColorFormat = cydriver.CUeglColorFormat_enum.CU_EGL_COLOR_FORMAT_BAYER_CRBC
    elif eglFrame.eglColorFormat == cudaEglColorFormatBayerCBRC:
        cuEglFrame[0].eglColorFormat = cydriver.CUeglColorFormat_enum.CU_EGL_COLOR_FORMAT_BAYER_CBRC
    elif eglFrame.eglColorFormat == cudaEglColorFormatBayer10CCCC:
        cuEglFrame[0].eglColorFormat = cydriver.CUeglColorFormat_enum.CU_EGL_COLOR_FORMAT_BAYER10_CCCC
    elif eglFrame.eglColorFormat == cudaEglColorFormatBayer12BCCR:
        cuEglFrame[0].eglColorFormat = cydriver.CUeglColorFormat_enum.CU_EGL_COLOR_FORMAT_BAYER12_BCCR
    elif eglFrame.eglColorFormat == cudaEglColorFormatBayer12RCCB:
        cuEglFrame[0].eglColorFormat = cydriver.CUeglColorFormat_enum.CU_EGL_COLOR_FORMAT_BAYER12_RCCB
    elif eglFrame.eglColorFormat == cudaEglColorFormatBayer12CRBC:
        cuEglFrame[0].eglColorFormat = cydriver.CUeglColorFormat_enum.CU_EGL_COLOR_FORMAT_BAYER12_CRBC
    elif eglFrame.eglColorFormat == cudaEglColorFormatBayer12CBRC:
        cuEglFrame[0].eglColorFormat = cydriver.CUeglColorFormat_enum.CU_EGL_COLOR_FORMAT_BAYER12_CBRC
    elif eglFrame.eglColorFormat == cudaEglColorFormatBayer12CCCC:
        cuEglFrame[0].eglColorFormat = cydriver.CUeglColorFormat_enum.CU_EGL_COLOR_FORMAT_BAYER12_CCCC
    elif eglFrame.eglColorFormat == cudaEglColorFormatY:
        cuEglFrame[0].eglColorFormat = cydriver.CUeglColorFormat_enum.CU_EGL_COLOR_FORMAT_Y
    elif eglFrame.eglColorFormat == cudaEglColorFormatYUV420SemiPlanar_2020:
        cuEglFrame[0].eglColorFormat = cydriver.CUeglColorFormat_enum.CU_EGL_COLOR_FORMAT_YUV420_SEMIPLANAR_2020
    elif eglFrame.eglColorFormat == cudaEglColorFormatYVU420SemiPlanar_2020:
        cuEglFrame[0].eglColorFormat = cydriver.CUeglColorFormat_enum.CU_EGL_COLOR_FORMAT_YVU420_SEMIPLANAR_2020
    elif eglFrame.eglColorFormat == cudaEglColorFormatYUV420Planar_2020:
        cuEglFrame[0].eglColorFormat = cydriver.CUeglColorFormat_enum.CU_EGL_COLOR_FORMAT_YUV420_PLANAR_2020
    elif eglFrame.eglColorFormat == cudaEglColorFormatYVU420Planar_2020:
        cuEglFrame[0].eglColorFormat = cydriver.CUeglColorFormat_enum.CU_EGL_COLOR_FORMAT_YVU420_PLANAR_2020
    elif eglFrame.eglColorFormat == cudaEglColorFormatYUV420SemiPlanar_709:
        cuEglFrame[0].eglColorFormat = cydriver.CUeglColorFormat_enum.CU_EGL_COLOR_FORMAT_YUV420_SEMIPLANAR_709
    elif eglFrame.eglColorFormat == cudaEglColorFormatYVU420SemiPlanar_709:
        cuEglFrame[0].eglColorFormat = cydriver.CUeglColorFormat_enum.CU_EGL_COLOR_FORMAT_YVU420_SEMIPLANAR_709
    elif eglFrame.eglColorFormat == cudaEglColorFormatYUV420Planar_709:
        cuEglFrame[0].eglColorFormat = cydriver.CUeglColorFormat_enum.CU_EGL_COLOR_FORMAT_YUV420_PLANAR_709
    elif eglFrame.eglColorFormat == cudaEglColorFormatYVU420Planar_709:
        cuEglFrame[0].eglColorFormat = cydriver.CUeglColorFormat_enum.CU_EGL_COLOR_FORMAT_YVU420_PLANAR_709
    elif eglFrame.eglColorFormat == cudaEglColorFormatY10V10U10_420SemiPlanar_709:
        cuEglFrame[0].eglColorFormat = cydriver.CUeglColorFormat_enum.CU_EGL_COLOR_FORMAT_Y10V10U10_420_SEMIPLANAR_709
    elif eglFrame.eglColorFormat == cudaEglColorFormatY10V10U10_420SemiPlanar_2020:
        cuEglFrame[0].eglColorFormat = cydriver.CUeglColorFormat_enum.CU_EGL_COLOR_FORMAT_Y10V10U10_420_SEMIPLANAR_2020
    elif eglFrame.eglColorFormat == cudaEglColorFormatY10V10U10_422SemiPlanar_2020:
        cuEglFrame[0].eglColorFormat = cydriver.CUeglColorFormat_enum.CU_EGL_COLOR_FORMAT_Y10V10U10_422_SEMIPLANAR_2020
    elif eglFrame.eglColorFormat == cudaEglColorFormatY10V10U10_422SemiPlanar:
        cuEglFrame[0].eglColorFormat = cydriver.CUeglColorFormat_enum.CU_EGL_COLOR_FORMAT_Y10V10U10_422_SEMIPLANAR
    elif eglFrame.eglColorFormat == cudaEglColorFormatY10V10U10_422SemiPlanar_709:
        cuEglFrame[0].eglColorFormat = cydriver.CUeglColorFormat_enum.CU_EGL_COLOR_FORMAT_Y10V10U10_422_SEMIPLANAR_709
    elif eglFrame.eglColorFormat == cudaEglColorFormatY_ER:
        cuEglFrame[0].eglColorFormat = cydriver.CUeglColorFormat_enum.CU_EGL_COLOR_FORMAT_Y_ER
    elif eglFrame.eglColorFormat == cudaEglColorFormatY_709_ER:
        cuEglFrame[0].eglColorFormat = cydriver.CUeglColorFormat_enum.CU_EGL_COLOR_FORMAT_Y_709_ER
    elif eglFrame.eglColorFormat == cudaEglColorFormatY10_ER:
        cuEglFrame[0].eglColorFormat = cydriver.CUeglColorFormat_enum.CU_EGL_COLOR_FORMAT_Y10_ER
    elif eglFrame.eglColorFormat == cudaEglColorFormatY10_709_ER:
        cuEglFrame[0].eglColorFormat = cydriver.CUeglColorFormat_enum.CU_EGL_COLOR_FORMAT_Y10_709_ER
    elif eglFrame.eglColorFormat == cudaEglColorFormatY12_ER:
        cuEglFrame[0].eglColorFormat = cydriver.CUeglColorFormat_enum.CU_EGL_COLOR_FORMAT_Y12_ER
    elif eglFrame.eglColorFormat == cudaEglColorFormatY12_709_ER:
        cuEglFrame[0].eglColorFormat = cydriver.CUeglColorFormat_enum.CU_EGL_COLOR_FORMAT_Y12_709_ER
    elif eglFrame.eglColorFormat == cudaEglColorFormatYUVA:
        cuEglFrame[0].eglColorFormat = cydriver.CUeglColorFormat_enum.CU_EGL_COLOR_FORMAT_YUVA
    elif eglFrame.eglColorFormat == cudaEglColorFormatYVYU:
        cuEglFrame[0].eglColorFormat = cydriver.CUeglColorFormat_enum.CU_EGL_COLOR_FORMAT_YVYU
    elif eglFrame.eglColorFormat == cudaEglColorFormatVYUY:
        cuEglFrame[0].eglColorFormat = cydriver.CUeglColorFormat_enum.CU_EGL_COLOR_FORMAT_VYUY
    elif eglFrame.eglColorFormat == cudaEglColorFormatY10V10U10_420SemiPlanar_ER:
        cuEglFrame[0].eglColorFormat = cydriver.CUeglColorFormat_enum.CU_EGL_COLOR_FORMAT_Y10V10U10_420_SEMIPLANAR_ER
    elif eglFrame.eglColorFormat == cudaEglColorFormatY10V10U10_420SemiPlanar_709_ER:
        cuEglFrame[0].eglColorFormat = cydriver.CUeglColorFormat_enum.CU_EGL_COLOR_FORMAT_Y10V10U10_420_SEMIPLANAR_709_ER
    elif eglFrame.eglColorFormat == cudaEglColorFormatY10V10U10_444SemiPlanar_ER:
        cuEglFrame[0].eglColorFormat =  cydriver.CUeglColorFormat_enum.CU_EGL_COLOR_FORMAT_Y10V10U10_444_SEMIPLANAR_ER
    elif eglFrame.eglColorFormat == cudaEglColorFormatY10V10U10_444SemiPlanar_709_ER:
        cuEglFrame[0].eglColorFormat = cydriver.CUeglColorFormat_enum.CU_EGL_COLOR_FORMAT_Y10V10U10_444_SEMIPLANAR_709_ER
    elif eglFrame.eglColorFormat == cudaEglColorFormatY12V12U12_420SemiPlanar_ER:
        cuEglFrame[0].eglColorFormat = cydriver.CUeglColorFormat_enum.CU_EGL_COLOR_FORMAT_Y12V12U12_420_SEMIPLANAR_ER
    elif eglFrame.eglColorFormat == cudaEglColorFormatY12V12U12_420SemiPlanar_709_ER:
        cuEglFrame[0].eglColorFormat = cydriver.CUeglColorFormat_enum.CU_EGL_COLOR_FORMAT_Y12V12U12_420_SEMIPLANAR_709_ER
    elif eglFrame.eglColorFormat == cudaEglColorFormatY12V12U12_444SemiPlanar_ER:
        cuEglFrame[0].eglColorFormat = cydriver.CUeglColorFormat_enum.CU_EGL_COLOR_FORMAT_Y12V12U12_444_SEMIPLANAR_ER
    elif eglFrame.eglColorFormat == cudaEglColorFormatY12V12U12_444SemiPlanar_709_ER:
        cuEglFrame[0].eglColorFormat = cydriver.CUeglColorFormat_enum.CU_EGL_COLOR_FORMAT_Y12V12U12_444_SEMIPLANAR_709_ER
    else:
        return cudaErrorInvalidValue
    if eglFrame.frameType == cudaEglFrameTypeArray:
        cuEglFrame[0].frameType = cydriver.CUeglFrameType_enum.CU_EGL_FRAME_TYPE_ARRAY
    elif eglFrame.frameType == cudaEglFrameTypePitch:
        cuEglFrame[0].frameType = cydriver.CUeglFrameType_enum.CU_EGL_FRAME_TYPE_PITCH
    else:
        return cudaErrorInvalidValue


@cython.show_performance_hints(False)
cdef cudaError_t getRuntimeEglFrame(cudaEglFrame *eglFrame, cydriver.CUeglFrame cueglFrame) except ?cudaErrorCallRequiresNewerDriver nogil:
    cdef cudaError_t err = cudaSuccess
    cdef unsigned int i
    cdef cydriver.CUDA_ARRAY3D_DESCRIPTOR_v2 ad
    cdef cudaPitchedPtr pPtr
    memset(eglFrame, 0, sizeof(eglFrame[0]))
    memset(&ad, 0, sizeof(ad))
    for i in range(cueglFrame.planeCount):
        ad.Depth = cueglFrame.depth
        ad.Flags = 0
        ad.Format = cueglFrame.cuFormat
        ad.Height = cueglFrame.height
        ad.NumChannels = cueglFrame.numChannels
        ad.Width = cueglFrame.width

        err = getChannelFormatDescFromDriverDesc(&eglFrame[0].planeDesc[i].channelDesc, NULL, NULL, NULL, &ad)
        if err != cudaSuccess:
            return err

        eglFrame[0].planeDesc[i].depth = cueglFrame.depth
        eglFrame[0].planeDesc[i].numChannels = cueglFrame.numChannels
        if i == 0:
            eglFrame[0].planeDesc[i].width = cueglFrame.width
            eglFrame[0].planeDesc[i].height = cueglFrame.height
            eglFrame[0].planeDesc[i].pitch = cueglFrame.pitch
        elif (cueglFrame.eglColorFormat == cydriver.CUeglColorFormat_enum.CU_EGL_COLOR_FORMAT_YUV420_PLANAR or
              cueglFrame.eglColorFormat == cydriver.CUeglColorFormat_enum.CU_EGL_COLOR_FORMAT_YUV420_PLANAR_ER or
              cueglFrame.eglColorFormat == cydriver.CUeglColorFormat_enum.CU_EGL_COLOR_FORMAT_YVU420_PLANAR or
              cueglFrame.eglColorFormat == cydriver.CUeglColorFormat_enum.CU_EGL_COLOR_FORMAT_YVU420_PLANAR_ER or
              cueglFrame.eglColorFormat == cydriver.CUeglColorFormat_enum.CU_EGL_COLOR_FORMAT_YUV420_PLANAR_2020 or
              cueglFrame.eglColorFormat == cydriver.CUeglColorFormat_enum.CU_EGL_COLOR_FORMAT_YVU420_PLANAR_2020 or
              cueglFrame.eglColorFormat == cydriver.CUeglColorFormat_enum.CU_EGL_COLOR_FORMAT_YUV420_PLANAR_709 or
              cueglFrame.eglColorFormat == cydriver.CUeglColorFormat_enum.CU_EGL_COLOR_FORMAT_YVU420_PLANAR_709):
            eglFrame[0].planeDesc[i].width = <unsigned int>(cueglFrame.width / 2)
            eglFrame[0].planeDesc[i].height = <unsigned int>(cueglFrame.height / 2)
            eglFrame[0].planeDesc[i].pitch = <unsigned int>(cueglFrame.pitch / 2)
        elif (cueglFrame.eglColorFormat == cydriver.CUeglColorFormat_enum.CU_EGL_COLOR_FORMAT_YUV420_SEMIPLANAR or
              cueglFrame.eglColorFormat == cydriver.CUeglColorFormat_enum.CU_EGL_COLOR_FORMAT_YUV420_SEMIPLANAR_ER or
              cueglFrame.eglColorFormat == cydriver.CUeglColorFormat_enum.CU_EGL_COLOR_FORMAT_YVU420_SEMIPLANAR or
              cueglFrame.eglColorFormat == cydriver.CUeglColorFormat_enum.CU_EGL_COLOR_FORMAT_YVU420_SEMIPLANAR_ER or
              cueglFrame.eglColorFormat == cydriver.CUeglColorFormat_enum.CU_EGL_COLOR_FORMAT_Y10V10U10_420_SEMIPLANAR or
              cueglFrame.eglColorFormat == cydriver.CUeglColorFormat_enum.CU_EGL_COLOR_FORMAT_Y12V12U12_420_SEMIPLANAR or
              cueglFrame.eglColorFormat == cydriver.CUeglColorFormat_enum.CU_EGL_COLOR_FORMAT_YUV420_SEMIPLANAR_2020 or
              cueglFrame.eglColorFormat == cydriver.CUeglColorFormat_enum.CU_EGL_COLOR_FORMAT_YVU420_SEMIPLANAR_2020 or
              cueglFrame.eglColorFormat == cydriver.CUeglColorFormat_enum.CU_EGL_COLOR_FORMAT_YUV420_SEMIPLANAR_709 or
              cueglFrame.eglColorFormat == cydriver.CUeglColorFormat_enum.CU_EGL_COLOR_FORMAT_YVU420_SEMIPLANAR_709 or 
              cueglFrame.eglColorFormat == cydriver.CUeglColorFormat_enum.CU_EGL_COLOR_FORMAT_Y10V10U10_420_SEMIPLANAR_709 or
              cueglFrame.eglColorFormat == cydriver.CUeglColorFormat_enum.CU_EGL_COLOR_FORMAT_Y10V10U10_420_SEMIPLANAR_2020 or
              cueglFrame.eglColorFormat == cydriver.CUeglColorFormat_enum.CU_EGL_COLOR_FORMAT_Y10V10U10_420_SEMIPLANAR_ER or
              cueglFrame.eglColorFormat == cydriver.CUeglColorFormat_enum.CU_EGL_COLOR_FORMAT_Y10V10U10_420_SEMIPLANAR_709_ER or
              cueglFrame.eglColorFormat == cydriver.CUeglColorFormat_enum.CU_EGL_COLOR_FORMAT_Y12V12U12_420_SEMIPLANAR_ER or
              cueglFrame.eglColorFormat == cydriver.CUeglColorFormat_enum.CU_EGL_COLOR_FORMAT_Y12V12U12_420_SEMIPLANAR_709_ER):
            eglFrame[0].planeDesc[i].width = <unsigned int>(cueglFrame.width / 2)
            eglFrame[0].planeDesc[i].height = <unsigned int>(cueglFrame.height / 2)
            eglFrame[0].planeDesc[i].pitch = <unsigned int>(cueglFrame.pitch / 2)
            eglFrame[0].planeDesc[1].channelDesc.y = 8
            if (cueglFrame.eglColorFormat == cydriver.CUeglColorFormat_enum.CU_EGL_COLOR_FORMAT_Y10V10U10_420_SEMIPLANAR or
                cueglFrame.eglColorFormat == cydriver.CUeglColorFormat_enum.CU_EGL_COLOR_FORMAT_Y12V12U12_420_SEMIPLANAR or
                cueglFrame.eglColorFormat == cydriver.CUeglColorFormat_enum.CU_EGL_COLOR_FORMAT_Y10V10U10_420_SEMIPLANAR_709 or
                cueglFrame.eglColorFormat == cydriver.CUeglColorFormat_enum.CU_EGL_COLOR_FORMAT_Y10V10U10_420_SEMIPLANAR_2020 or
                cueglFrame.eglColorFormat == cydriver.CUeglColorFormat_enum.CU_EGL_COLOR_FORMAT_Y10V10U10_420_SEMIPLANAR_ER or
                cueglFrame.eglColorFormat == cydriver.CUeglColorFormat_enum.CU_EGL_COLOR_FORMAT_Y10V10U10_420_SEMIPLANAR_709_ER or
                cueglFrame.eglColorFormat == cydriver.CUeglColorFormat_enum.CU_EGL_COLOR_FORMAT_Y12V12U12_420_SEMIPLANAR_ER or
                cueglFrame.eglColorFormat == cydriver.CUeglColorFormat_enum.CU_EGL_COLOR_FORMAT_Y12V12U12_420_SEMIPLANAR_709_ER):
                eglFrame[0].planeDesc[1].channelDesc.y = 16
        elif (cueglFrame.eglColorFormat == cydriver.CUeglColorFormat_enum.CU_EGL_COLOR_FORMAT_YUV422_PLANAR or
              cueglFrame.eglColorFormat == cydriver.CUeglColorFormat_enum.CU_EGL_COLOR_FORMAT_YUV422_PLANAR_ER or
              cueglFrame.eglColorFormat == cydriver.CUeglColorFormat_enum.CU_EGL_COLOR_FORMAT_YVU422_PLANAR or
              cueglFrame.eglColorFormat == cydriver.CUeglColorFormat_enum.CU_EGL_COLOR_FORMAT_YVU422_PLANAR_ER):
            eglFrame[0].planeDesc[i].height = cueglFrame.height
            eglFrame[0].planeDesc[i].width = <unsigned int>(cueglFrame.width / 2)
            eglFrame[0].planeDesc[i].pitch = <unsigned int>(cueglFrame.pitch / 2)
        elif (cueglFrame.eglColorFormat == cydriver.CUeglColorFormat_enum.CU_EGL_COLOR_FORMAT_YUV422_SEMIPLANAR or
              cueglFrame.eglColorFormat == cydriver.CUeglColorFormat_enum.CU_EGL_COLOR_FORMAT_YUV422_SEMIPLANAR_ER or
              cueglFrame.eglColorFormat == cydriver.CUeglColorFormat_enum.CU_EGL_COLOR_FORMAT_YVU422_SEMIPLANAR or
              cueglFrame.eglColorFormat == cydriver.CUeglColorFormat_enum.CU_EGL_COLOR_FORMAT_YVU422_SEMIPLANAR_ER or
              cueglFrame.eglColorFormat == cydriver.CUeglColorFormat_enum.CU_EGL_COLOR_FORMAT_Y10V10U10_422_SEMIPLANAR_2020 or
              cueglFrame.eglColorFormat == cydriver.CUeglColorFormat_enum.CU_EGL_COLOR_FORMAT_Y10V10U10_422_SEMIPLANAR or
              cueglFrame.eglColorFormat == cydriver.CUeglColorFormat_enum.CU_EGL_COLOR_FORMAT_Y10V10U10_422_SEMIPLANAR_709):
            eglFrame[0].planeDesc[i].width = <unsigned int>(cueglFrame.width / 2)
            eglFrame[0].planeDesc[i].height = cueglFrame.height
            eglFrame[0].planeDesc[i].pitch = <unsigned int>(cueglFrame.pitch / 2)
            eglFrame[0].planeDesc[1].channelDesc.y = 8
            if (cueglFrame.eglColorFormat == cydriver.CUeglColorFormat_enum.CU_EGL_COLOR_FORMAT_Y10V10U10_422_SEMIPLANAR_2020 or
                cueglFrame.eglColorFormat == cydriver.CUeglColorFormat_enum.CU_EGL_COLOR_FORMAT_Y10V10U10_422_SEMIPLANAR or
                cueglFrame.eglColorFormat == cydriver.CUeglColorFormat_enum.CU_EGL_COLOR_FORMAT_Y10V10U10_422_SEMIPLANAR_709):
                eglFrame[0].planeDesc[1].channelDesc.y = 16
        elif (cueglFrame.eglColorFormat == cydriver.CUeglColorFormat_enum.CU_EGL_COLOR_FORMAT_YUV444_PLANAR or
              cueglFrame.eglColorFormat == cydriver.CUeglColorFormat_enum.CU_EGL_COLOR_FORMAT_YUV444_PLANAR_ER or
              cueglFrame.eglColorFormat == cydriver.CUeglColorFormat_enum.CU_EGL_COLOR_FORMAT_YVU444_PLANAR or
              cueglFrame.eglColorFormat == cydriver.CUeglColorFormat_enum.CU_EGL_COLOR_FORMAT_YVU444_PLANAR_ER):
            eglFrame[0].planeDesc[i].height = cueglFrame.height
            eglFrame[0].planeDesc[i].width = cueglFrame.width
            eglFrame[0].planeDesc[i].pitch = cueglFrame.pitch
        elif (cueglFrame.eglColorFormat == cydriver.CUeglColorFormat_enum.CU_EGL_COLOR_FORMAT_YUV444_SEMIPLANAR or
              cueglFrame.eglColorFormat == cydriver.CUeglColorFormat_enum.CU_EGL_COLOR_FORMAT_YUV444_SEMIPLANAR_ER or
              cueglFrame.eglColorFormat == cydriver.CUeglColorFormat_enum.CU_EGL_COLOR_FORMAT_YVU444_SEMIPLANAR or
              cueglFrame.eglColorFormat == cydriver.CUeglColorFormat_enum.CU_EGL_COLOR_FORMAT_YVU444_SEMIPLANAR_ER or
              cueglFrame.eglColorFormat == cydriver.CUeglColorFormat_enum.CU_EGL_COLOR_FORMAT_Y10V10U10_444_SEMIPLANAR or
              cueglFrame.eglColorFormat == cydriver.CUeglColorFormat_enum.CU_EGL_COLOR_FORMAT_Y12V12U12_444_SEMIPLANAR or
              cueglFrame.eglColorFormat == cydriver.CUeglColorFormat_enum.CU_EGL_COLOR_FORMAT_Y10V10U10_444_SEMIPLANAR_ER or
              cueglFrame.eglColorFormat == cydriver.CUeglColorFormat_enum.CU_EGL_COLOR_FORMAT_Y10V10U10_444_SEMIPLANAR_709_ER or
              cueglFrame.eglColorFormat == cydriver.CUeglColorFormat_enum.CU_EGL_COLOR_FORMAT_Y12V12U12_444_SEMIPLANAR_ER or
              cueglFrame.eglColorFormat == cydriver.CUeglColorFormat_enum.CU_EGL_COLOR_FORMAT_Y12V12U12_444_SEMIPLANAR_709_ER):
            eglFrame[0].planeDesc[i].height = cueglFrame.height
            eglFrame[0].planeDesc[i].width = cueglFrame.width
            eglFrame[0].planeDesc[i].pitch = cueglFrame.pitch
            eglFrame[0].planeDesc[1].channelDesc.y = 8
            if (cueglFrame.eglColorFormat == cydriver.CUeglColorFormat_enum.CU_EGL_COLOR_FORMAT_Y10V10U10_444_SEMIPLANAR or
                cueglFrame.eglColorFormat == cydriver.CUeglColorFormat_enum.CU_EGL_COLOR_FORMAT_Y12V12U12_444_SEMIPLANAR or
                cueglFrame.eglColorFormat == cydriver.CUeglColorFormat_enum.CU_EGL_COLOR_FORMAT_Y10V10U10_444_SEMIPLANAR_ER or
                cueglFrame.eglColorFormat == cydriver.CUeglColorFormat_enum.CU_EGL_COLOR_FORMAT_Y10V10U10_444_SEMIPLANAR_709_ER or
                cueglFrame.eglColorFormat == cydriver.CUeglColorFormat_enum.CU_EGL_COLOR_FORMAT_Y12V12U12_444_SEMIPLANAR_ER or
                cueglFrame.eglColorFormat == cydriver.CUeglColorFormat_enum.CU_EGL_COLOR_FORMAT_Y12V12U12_444_SEMIPLANAR_709_ER):
                eglFrame[0].planeDesc[1].channelDesc.y = 16
        if cueglFrame.frameType == cydriver.CUeglFrameType_enum.CU_EGL_FRAME_TYPE_ARRAY:
            eglFrame[0].frame.pArray[i] = <cudaArray_t>cueglFrame.frame.pArray[i]
        else:
            pPtr = make_cudaPitchedPtr(cueglFrame.frame.pPitch[i], eglFrame[0].planeDesc[i].pitch,
                    eglFrame[0].planeDesc[i].width, eglFrame[0].planeDesc[i].height)
            eglFrame[0].frame.pPitch[i] = pPtr

    eglFrame[0].planeCount = cueglFrame.planeCount
    if cueglFrame.eglColorFormat == cydriver.CUeglColorFormat_enum.CU_EGL_COLOR_FORMAT_YUV420_PLANAR:
        eglFrame[0].eglColorFormat = cudaEglColorFormatYUV420Planar
    elif cueglFrame.eglColorFormat == cydriver.CUeglColorFormat_enum.CU_EGL_COLOR_FORMAT_YUV420_SEMIPLANAR:
        eglFrame[0].eglColorFormat = cudaEglColorFormatYUV420SemiPlanar
    elif cueglFrame.eglColorFormat == cydriver.CUeglColorFormat_enum.CU_EGL_COLOR_FORMAT_YUV422_PLANAR:
        eglFrame[0].eglColorFormat = cudaEglColorFormatYUV422Planar
    elif cueglFrame.eglColorFormat == cydriver.CUeglColorFormat_enum.CU_EGL_COLOR_FORMAT_YUV422_SEMIPLANAR:
        eglFrame[0].eglColorFormat = cudaEglColorFormatYUV422SemiPlanar
    elif cueglFrame.eglColorFormat == cydriver.CUeglColorFormat_enum.CU_EGL_COLOR_FORMAT_YUV444_PLANAR:
        eglFrame[0].eglColorFormat = cudaEglColorFormatYUV444Planar
    elif cueglFrame.eglColorFormat == cydriver.CUeglColorFormat_enum.CU_EGL_COLOR_FORMAT_YUV444_SEMIPLANAR:
        eglFrame[0].eglColorFormat = cudaEglColorFormatYUV444SemiPlanar
    elif cueglFrame.eglColorFormat == cydriver.CUeglColorFormat_enum.CU_EGL_COLOR_FORMAT_YUYV_422:
        eglFrame[0].eglColorFormat = cudaEglColorFormatYUYV422
    elif cueglFrame.eglColorFormat == cydriver.CUeglColorFormat_enum.CU_EGL_COLOR_FORMAT_UYVY_422:
        eglFrame[0].eglColorFormat = cudaEglColorFormatUYVY422
    elif cueglFrame.eglColorFormat == cydriver.CUeglColorFormat_enum.CU_EGL_COLOR_FORMAT_UYVY_709:
        eglFrame[0].eglColorFormat = cudaEglColorFormatUYVY709
    elif cueglFrame.eglColorFormat == cydriver.CUeglColorFormat_enum.CU_EGL_COLOR_FORMAT_UYVY_709_ER:
        eglFrame[0].eglColorFormat = cudaEglColorFormatUYVY709_ER
    elif cueglFrame.eglColorFormat == cydriver.CUeglColorFormat_enum.CU_EGL_COLOR_FORMAT_UYVY_2020:
        eglFrame[0].eglColorFormat = cudaEglColorFormatUYVY2020
    elif cueglFrame.eglColorFormat == cydriver.CUeglColorFormat_enum.CU_EGL_COLOR_FORMAT_ARGB:
        eglFrame[0].eglColorFormat = cudaEglColorFormatARGB
    elif cueglFrame.eglColorFormat == cydriver.CUeglColorFormat_enum.CU_EGL_COLOR_FORMAT_RGBA:
        eglFrame[0].eglColorFormat = cudaEglColorFormatRGBA
    elif cueglFrame.eglColorFormat == cydriver.CUeglColorFormat_enum.CU_EGL_COLOR_FORMAT_ABGR:
        eglFrame[0].eglColorFormat = cudaEglColorFormatABGR
    elif cueglFrame.eglColorFormat == cydriver.CUeglColorFormat_enum.CU_EGL_COLOR_FORMAT_BGRA:
        eglFrame[0].eglColorFormat = cudaEglColorFormatBGRA
    elif cueglFrame.eglColorFormat == cydriver.CUeglColorFormat_enum.CU_EGL_COLOR_FORMAT_L:
        eglFrame[0].eglColorFormat = cudaEglColorFormatL
    elif cueglFrame.eglColorFormat == cydriver.CUeglColorFormat_enum.CU_EGL_COLOR_FORMAT_R:
        eglFrame[0].eglColorFormat = cudaEglColorFormatR
    elif cueglFrame.eglColorFormat == cydriver.CUeglColorFormat_enum.CU_EGL_COLOR_FORMAT_A:
        eglFrame[0].eglColorFormat = cudaEglColorFormatA
    elif cueglFrame.eglColorFormat == cydriver.CUeglColorFormat_enum.CU_EGL_COLOR_FORMAT_RG:
        eglFrame[0].eglColorFormat = cudaEglColorFormatRG
    elif cueglFrame.eglColorFormat == cydriver.CUeglColorFormat_enum.CU_EGL_COLOR_FORMAT_AYUV:
        eglFrame[0].eglColorFormat = cudaEglColorFormatAYUV
    elif cueglFrame.eglColorFormat == cydriver.CUeglColorFormat_enum.CU_EGL_COLOR_FORMAT_YVU444_SEMIPLANAR:
        eglFrame[0].eglColorFormat = cudaEglColorFormatYVU444SemiPlanar
    elif cueglFrame.eglColorFormat == cydriver.CUeglColorFormat_enum.CU_EGL_COLOR_FORMAT_YVU422_SEMIPLANAR:
        eglFrame[0].eglColorFormat = cudaEglColorFormatYVU422SemiPlanar
    elif cueglFrame.eglColorFormat == cydriver.CUeglColorFormat_enum.CU_EGL_COLOR_FORMAT_YVU420_SEMIPLANAR:
        eglFrame[0].eglColorFormat = cudaEglColorFormatYVU420SemiPlanar
    elif cueglFrame.eglColorFormat == cydriver.CUeglColorFormat_enum.CU_EGL_COLOR_FORMAT_Y10V10U10_444_SEMIPLANAR:
        eglFrame[0].eglColorFormat = cudaEglColorFormatY10V10U10_444SemiPlanar
    elif cueglFrame.eglColorFormat == cydriver.CUeglColorFormat_enum.CU_EGL_COLOR_FORMAT_Y10V10U10_420_SEMIPLANAR:
        eglFrame[0].eglColorFormat = cudaEglColorFormatY10V10U10_420SemiPlanar
    elif cueglFrame.eglColorFormat == cydriver.CUeglColorFormat_enum.CU_EGL_COLOR_FORMAT_Y12V12U12_444_SEMIPLANAR:
        eglFrame[0].eglColorFormat = cudaEglColorFormatY12V12U12_444SemiPlanar
    elif cueglFrame.eglColorFormat == cydriver.CUeglColorFormat_enum.CU_EGL_COLOR_FORMAT_Y12V12U12_420_SEMIPLANAR:
        eglFrame[0].eglColorFormat = cudaEglColorFormatY12V12U12_420SemiPlanar
    elif cueglFrame.eglColorFormat == cydriver.CUeglColorFormat_enum.CU_EGL_COLOR_FORMAT_VYUY_ER:
        eglFrame[0].eglColorFormat = cudaEglColorFormatVYUY_ER
    elif cueglFrame.eglColorFormat == cydriver.CUeglColorFormat_enum.CU_EGL_COLOR_FORMAT_UYVY_ER:
        eglFrame[0].eglColorFormat = cudaEglColorFormatUYVY_ER
    elif cueglFrame.eglColorFormat == cydriver.CUeglColorFormat_enum.CU_EGL_COLOR_FORMAT_YUYV_ER:
        eglFrame[0].eglColorFormat = cudaEglColorFormatYUYV_ER
    elif cueglFrame.eglColorFormat == cydriver.CUeglColorFormat_enum.CU_EGL_COLOR_FORMAT_YVYU_ER:
        eglFrame[0].eglColorFormat = cudaEglColorFormatYVYU_ER
    elif cueglFrame.eglColorFormat == cydriver.CUeglColorFormat_enum.CU_EGL_COLOR_FORMAT_YUVA_ER:
        eglFrame[0].eglColorFormat = cudaEglColorFormatYUVA_ER
    elif cueglFrame.eglColorFormat == cydriver.CUeglColorFormat_enum.CU_EGL_COLOR_FORMAT_AYUV_ER:
        eglFrame[0].eglColorFormat = cudaEglColorFormatAYUV_ER
    elif cueglFrame.eglColorFormat == cydriver.CUeglColorFormat_enum.CU_EGL_COLOR_FORMAT_YUV444_PLANAR_ER:
        eglFrame[0].eglColorFormat = cudaEglColorFormatYUV444Planar_ER
    elif cueglFrame.eglColorFormat == cydriver.CUeglColorFormat_enum.CU_EGL_COLOR_FORMAT_YUV422_PLANAR_ER:
        eglFrame[0].eglColorFormat = cudaEglColorFormatYUV422Planar_ER
    elif cueglFrame.eglColorFormat == cydriver.CUeglColorFormat_enum.CU_EGL_COLOR_FORMAT_YUV420_PLANAR_ER:
        eglFrame[0].eglColorFormat = cudaEglColorFormatYUV420Planar_ER
    elif cueglFrame.eglColorFormat == cydriver.CUeglColorFormat_enum.CU_EGL_COLOR_FORMAT_YUV444_SEMIPLANAR_ER:
        eglFrame[0].eglColorFormat = cudaEglColorFormatYUV444SemiPlanar_ER
    elif cueglFrame.eglColorFormat == cydriver.CUeglColorFormat_enum.CU_EGL_COLOR_FORMAT_YUV422_SEMIPLANAR_ER:
        eglFrame[0].eglColorFormat = cudaEglColorFormatYUV422SemiPlanar_ER
    elif cueglFrame.eglColorFormat == cydriver.CUeglColorFormat_enum.CU_EGL_COLOR_FORMAT_YUV420_SEMIPLANAR_ER:
        eglFrame[0].eglColorFormat = cudaEglColorFormatYUV420SemiPlanar_ER
    elif cueglFrame.eglColorFormat == cydriver.CUeglColorFormat_enum.CU_EGL_COLOR_FORMAT_YVU444_PLANAR_ER:
        eglFrame[0].eglColorFormat = cudaEglColorFormatYVU444Planar_ER
    elif cueglFrame.eglColorFormat == cydriver.CUeglColorFormat_enum.CU_EGL_COLOR_FORMAT_YVU422_PLANAR_ER:
        eglFrame[0].eglColorFormat = cudaEglColorFormatYVU422Planar_ER
    elif cueglFrame.eglColorFormat == cydriver.CUeglColorFormat_enum.CU_EGL_COLOR_FORMAT_YVU420_PLANAR_ER:
        eglFrame[0].eglColorFormat = cudaEglColorFormatYVU420Planar_ER
    elif cueglFrame.eglColorFormat == cydriver.CUeglColorFormat_enum.CU_EGL_COLOR_FORMAT_YVU444_SEMIPLANAR_ER:
        eglFrame[0].eglColorFormat = cudaEglColorFormatYVU444SemiPlanar_ER
    elif cueglFrame.eglColorFormat == cydriver.CUeglColorFormat_enum.CU_EGL_COLOR_FORMAT_YVU422_SEMIPLANAR_ER:
        eglFrame[0].eglColorFormat = cudaEglColorFormatYVU422SemiPlanar_ER
    elif cueglFrame.eglColorFormat == cydriver.CUeglColorFormat_enum.CU_EGL_COLOR_FORMAT_YVU420_SEMIPLANAR_ER:
        eglFrame[0].eglColorFormat = cudaEglColorFormatYVU420SemiPlanar_ER
    elif cueglFrame.eglColorFormat == cydriver.CUeglColorFormat_enum.CU_EGL_COLOR_FORMAT_BAYER_RGGB:
        eglFrame[0].eglColorFormat = cudaEglColorFormatBayerRGGB
    elif cueglFrame.eglColorFormat == cydriver.CUeglColorFormat_enum.CU_EGL_COLOR_FORMAT_BAYER_BGGR:
        eglFrame[0].eglColorFormat = cudaEglColorFormatBayerBGGR
    elif cueglFrame.eglColorFormat == cydriver.CUeglColorFormat_enum.CU_EGL_COLOR_FORMAT_BAYER_GRBG:
        eglFrame[0].eglColorFormat = cudaEglColorFormatBayerGRBG
    elif cueglFrame.eglColorFormat == cydriver.CUeglColorFormat_enum.CU_EGL_COLOR_FORMAT_BAYER_GBRG:
        eglFrame[0].eglColorFormat = cudaEglColorFormatBayerGBRG
    elif cueglFrame.eglColorFormat == cydriver.CUeglColorFormat_enum.CU_EGL_COLOR_FORMAT_BAYER10_RGGB:
        eglFrame[0].eglColorFormat = cudaEglColorFormatBayer10RGGB
    elif cueglFrame.eglColorFormat == cydriver.CUeglColorFormat_enum.CU_EGL_COLOR_FORMAT_BAYER10_BGGR:
        eglFrame[0].eglColorFormat = cudaEglColorFormatBayer10BGGR
    elif cueglFrame.eglColorFormat == cydriver.CUeglColorFormat_enum.CU_EGL_COLOR_FORMAT_BAYER10_GRBG:
        eglFrame[0].eglColorFormat = cudaEglColorFormatBayer10GRBG
    elif cueglFrame.eglColorFormat == cydriver.CUeglColorFormat_enum.CU_EGL_COLOR_FORMAT_BAYER10_GBRG:
        eglFrame[0].eglColorFormat = cudaEglColorFormatBayer10GBRG
    elif cueglFrame.eglColorFormat == cydriver.CUeglColorFormat_enum.CU_EGL_COLOR_FORMAT_BAYER12_RGGB:
        eglFrame[0].eglColorFormat = cudaEglColorFormatBayer12RGGB
    elif cueglFrame.eglColorFormat == cydriver.CUeglColorFormat_enum.CU_EGL_COLOR_FORMAT_BAYER12_BGGR:
        eglFrame[0].eglColorFormat = cudaEglColorFormatBayer12BGGR
    elif cueglFrame.eglColorFormat == cydriver.CUeglColorFormat_enum.CU_EGL_COLOR_FORMAT_BAYER12_GRBG:
        eglFrame[0].eglColorFormat = cudaEglColorFormatBayer12GRBG
    elif cueglFrame.eglColorFormat == cydriver.CUeglColorFormat_enum.CU_EGL_COLOR_FORMAT_BAYER12_GBRG:
        eglFrame[0].eglColorFormat = cudaEglColorFormatBayer12GBRG
    elif cueglFrame.eglColorFormat == cydriver.CUeglColorFormat_enum.CU_EGL_COLOR_FORMAT_BAYER14_RGGB:
        eglFrame[0].eglColorFormat = cudaEglColorFormatBayer14RGGB
    elif cueglFrame.eglColorFormat == cydriver.CUeglColorFormat_enum.CU_EGL_COLOR_FORMAT_BAYER14_BGGR:
        eglFrame[0].eglColorFormat = cudaEglColorFormatBayer14BGGR
    elif cueglFrame.eglColorFormat == cydriver.CUeglColorFormat_enum.CU_EGL_COLOR_FORMAT_BAYER14_GRBG:
        eglFrame[0].eglColorFormat = cudaEglColorFormatBayer14GRBG
    elif cueglFrame.eglColorFormat == cydriver.CUeglColorFormat_enum.CU_EGL_COLOR_FORMAT_BAYER14_GBRG:
        eglFrame[0].eglColorFormat = cudaEglColorFormatBayer14GBRG
    elif cueglFrame.eglColorFormat == cydriver.CUeglColorFormat_enum.CU_EGL_COLOR_FORMAT_BAYER20_RGGB:
        eglFrame[0].eglColorFormat = cudaEglColorFormatBayer20RGGB
    elif cueglFrame.eglColorFormat == cydriver.CUeglColorFormat_enum.CU_EGL_COLOR_FORMAT_BAYER20_BGGR:
        eglFrame[0].eglColorFormat = cudaEglColorFormatBayer20BGGR
    elif cueglFrame.eglColorFormat == cydriver.CUeglColorFormat_enum.CU_EGL_COLOR_FORMAT_BAYER20_GRBG:
        eglFrame[0].eglColorFormat = cudaEglColorFormatBayer20GRBG
    elif cueglFrame.eglColorFormat == cydriver.CUeglColorFormat_enum.CU_EGL_COLOR_FORMAT_BAYER20_GBRG:
        eglFrame[0].eglColorFormat = cudaEglColorFormatBayer20GBRG
    elif cueglFrame.eglColorFormat == cydriver.CUeglColorFormat_enum.CU_EGL_COLOR_FORMAT_BAYER_ISP_RGGB:
        eglFrame[0].eglColorFormat = cudaEglColorFormatBayerIspRGGB
    elif cueglFrame.eglColorFormat == cydriver.CUeglColorFormat_enum.CU_EGL_COLOR_FORMAT_BAYER_ISP_BGGR:
        eglFrame[0].eglColorFormat = cudaEglColorFormatBayerIspBGGR
    elif cueglFrame.eglColorFormat == cydriver.CUeglColorFormat_enum.CU_EGL_COLOR_FORMAT_BAYER_ISP_GRBG:
        eglFrame[0].eglColorFormat = cudaEglColorFormatBayerIspGRBG
    elif cueglFrame.eglColorFormat == cydriver.CUeglColorFormat_enum.CU_EGL_COLOR_FORMAT_BAYER_ISP_GBRG:
        eglFrame[0].eglColorFormat = cudaEglColorFormatBayerIspGBRG
    elif cueglFrame.eglColorFormat == cydriver.CUeglColorFormat_enum.CU_EGL_COLOR_FORMAT_YVU444_PLANAR:
        eglFrame[0].eglColorFormat = cudaEglColorFormatYVU444Planar
    elif cueglFrame.eglColorFormat == cydriver.CUeglColorFormat_enum.CU_EGL_COLOR_FORMAT_YVU422_PLANAR:
        eglFrame[0].eglColorFormat = cudaEglColorFormatYVU422Planar
    elif cueglFrame.eglColorFormat == cydriver.CUeglColorFormat_enum.CU_EGL_COLOR_FORMAT_YVU420_PLANAR:
        eglFrame[0].eglColorFormat = cudaEglColorFormatYVU420Planar
    elif cueglFrame.eglColorFormat == cydriver.CUeglColorFormat_enum.CU_EGL_COLOR_FORMAT_BAYER_BCCR:
        eglFrame[0].eglColorFormat = cudaEglColorFormatBayerBCCR
    elif cueglFrame.eglColorFormat == cydriver.CUeglColorFormat_enum.CU_EGL_COLOR_FORMAT_BAYER_RCCB:
        eglFrame[0].eglColorFormat = cudaEglColorFormatBayerRCCB
    elif cueglFrame.eglColorFormat == cydriver.CUeglColorFormat_enum.CU_EGL_COLOR_FORMAT_BAYER_CRBC:
        eglFrame[0].eglColorFormat = cudaEglColorFormatBayerCRBC
    elif cueglFrame.eglColorFormat == cydriver.CUeglColorFormat_enum.CU_EGL_COLOR_FORMAT_BAYER_CBRC:
        eglFrame[0].eglColorFormat = cudaEglColorFormatBayerCBRC
    elif cueglFrame.eglColorFormat == cydriver.CUeglColorFormat_enum.CU_EGL_COLOR_FORMAT_BAYER10_CCCC:
        eglFrame[0].eglColorFormat = cudaEglColorFormatBayer10CCCC
    elif cueglFrame.eglColorFormat == cydriver.CUeglColorFormat_enum.CU_EGL_COLOR_FORMAT_BAYER12_BCCR:
        eglFrame[0].eglColorFormat = cudaEglColorFormatBayer12BCCR
    elif cueglFrame.eglColorFormat == cydriver.CUeglColorFormat_enum.CU_EGL_COLOR_FORMAT_BAYER12_RCCB:
        eglFrame[0].eglColorFormat = cudaEglColorFormatBayer12RCCB
    elif cueglFrame.eglColorFormat == cydriver.CUeglColorFormat_enum.CU_EGL_COLOR_FORMAT_BAYER12_CRBC:
        eglFrame[0].eglColorFormat = cudaEglColorFormatBayer12CRBC
    elif cueglFrame.eglColorFormat == cydriver.CUeglColorFormat_enum.CU_EGL_COLOR_FORMAT_BAYER12_CBRC:
        eglFrame[0].eglColorFormat = cudaEglColorFormatBayer12CBRC
    elif cueglFrame.eglColorFormat == cydriver.CUeglColorFormat_enum.CU_EGL_COLOR_FORMAT_BAYER12_CCCC:
        eglFrame[0].eglColorFormat = cudaEglColorFormatBayer12CCCC
    elif cueglFrame.eglColorFormat == cydriver.CUeglColorFormat_enum.CU_EGL_COLOR_FORMAT_Y:
        eglFrame[0].eglColorFormat = cudaEglColorFormatY
    elif cueglFrame.eglColorFormat == cydriver.CUeglColorFormat_enum.CU_EGL_COLOR_FORMAT_YUV420_SEMIPLANAR_2020:
        eglFrame[0].eglColorFormat = cudaEglColorFormatYUV420SemiPlanar_2020
    elif cueglFrame.eglColorFormat == cydriver.CUeglColorFormat_enum.CU_EGL_COLOR_FORMAT_YVU420_SEMIPLANAR_2020:
        eglFrame[0].eglColorFormat = cudaEglColorFormatYVU420SemiPlanar_2020
    elif cueglFrame.eglColorFormat == cydriver.CUeglColorFormat_enum.CU_EGL_COLOR_FORMAT_YUV420_PLANAR_2020:
        eglFrame[0].eglColorFormat = cudaEglColorFormatYUV420Planar_2020
    elif cueglFrame.eglColorFormat == cydriver.CUeglColorFormat_enum.CU_EGL_COLOR_FORMAT_YVU420_PLANAR_2020:
        eglFrame[0].eglColorFormat = cudaEglColorFormatYVU420Planar_2020
    elif cueglFrame.eglColorFormat == cydriver.CUeglColorFormat_enum.CU_EGL_COLOR_FORMAT_YUV420_SEMIPLANAR_709:
        eglFrame[0].eglColorFormat = cudaEglColorFormatYUV420SemiPlanar_709
    elif cueglFrame.eglColorFormat == cydriver.CUeglColorFormat_enum.CU_EGL_COLOR_FORMAT_YVU420_SEMIPLANAR_709:
        eglFrame[0].eglColorFormat = cudaEglColorFormatYVU420SemiPlanar_709
    elif cueglFrame.eglColorFormat == cydriver.CUeglColorFormat_enum.CU_EGL_COLOR_FORMAT_YUV420_PLANAR_709:
        eglFrame[0].eglColorFormat = cudaEglColorFormatYUV420Planar_709
    elif cueglFrame.eglColorFormat == cydriver.CUeglColorFormat_enum.CU_EGL_COLOR_FORMAT_YVU420_PLANAR_709:
        eglFrame[0].eglColorFormat = cudaEglColorFormatYVU420Planar_709
    elif cueglFrame.eglColorFormat == cydriver.CUeglColorFormat_enum.CU_EGL_COLOR_FORMAT_Y10V10U10_420_SEMIPLANAR_709:
        eglFrame[0].eglColorFormat = cudaEglColorFormatY10V10U10_420SemiPlanar_709
    elif cueglFrame.eglColorFormat == cydriver.CUeglColorFormat_enum.CU_EGL_COLOR_FORMAT_Y10V10U10_420_SEMIPLANAR_2020:
        eglFrame[0].eglColorFormat = cudaEglColorFormatY10V10U10_420SemiPlanar_2020
    elif cueglFrame.eglColorFormat == cydriver.CUeglColorFormat_enum.CU_EGL_COLOR_FORMAT_Y10V10U10_422_SEMIPLANAR_2020:
        eglFrame[0].eglColorFormat = cudaEglColorFormatY10V10U10_422SemiPlanar_2020
    elif cueglFrame.eglColorFormat == cydriver.CUeglColorFormat_enum.CU_EGL_COLOR_FORMAT_Y10V10U10_422_SEMIPLANAR:
        eglFrame[0].eglColorFormat = cudaEglColorFormatY10V10U10_422SemiPlanar
    elif cueglFrame.eglColorFormat == cydriver.CUeglColorFormat_enum.CU_EGL_COLOR_FORMAT_Y10V10U10_422_SEMIPLANAR_709:
        eglFrame[0].eglColorFormat = cudaEglColorFormatY10V10U10_422SemiPlanar_709
    elif cueglFrame.eglColorFormat == cydriver.CUeglColorFormat_enum.CU_EGL_COLOR_FORMAT_Y_ER:
        eglFrame[0].eglColorFormat = cudaEglColorFormatY_ER
    elif cueglFrame.eglColorFormat == cydriver.CUeglColorFormat_enum.CU_EGL_COLOR_FORMAT_Y_709_ER:
        eglFrame[0].eglColorFormat = cudaEglColorFormatY_709_ER
    elif cueglFrame.eglColorFormat == cydriver.CUeglColorFormat_enum.CU_EGL_COLOR_FORMAT_Y10_ER:
        eglFrame[0].eglColorFormat = cudaEglColorFormatY10_ER
    elif cueglFrame.eglColorFormat == cydriver.CUeglColorFormat_enum.CU_EGL_COLOR_FORMAT_Y10_709_ER:
        eglFrame[0].eglColorFormat = cudaEglColorFormatY10_709_ER
    elif cueglFrame.eglColorFormat == cydriver.CUeglColorFormat_enum.CU_EGL_COLOR_FORMAT_Y12_ER:
        eglFrame[0].eglColorFormat = cudaEglColorFormatY12_ER
    elif cueglFrame.eglColorFormat == cydriver.CUeglColorFormat_enum.CU_EGL_COLOR_FORMAT_Y12_709_ER:
        eglFrame[0].eglColorFormat = cudaEglColorFormatY12_709_ER
    elif cueglFrame.eglColorFormat == cydriver.CUeglColorFormat_enum.CU_EGL_COLOR_FORMAT_YUVA:
        eglFrame[0].eglColorFormat = cudaEglColorFormatYUVA
    elif cueglFrame.eglColorFormat == cydriver.CUeglColorFormat_enum.CU_EGL_COLOR_FORMAT_YVYU:
        eglFrame[0].eglColorFormat = cudaEglColorFormatYVYU
    elif cueglFrame.eglColorFormat == cydriver.CUeglColorFormat_enum.CU_EGL_COLOR_FORMAT_VYUY:
        eglFrame[0].eglColorFormat = cudaEglColorFormatVYUY
    elif cueglFrame.eglColorFormat == cydriver.CUeglColorFormat_enum.CU_EGL_COLOR_FORMAT_Y10V10U10_420_SEMIPLANAR_ER:
        eglFrame[0].eglColorFormat = cudaEglColorFormatY10V10U10_420SemiPlanar_ER
    elif cueglFrame.eglColorFormat == cydriver.CUeglColorFormat_enum.CU_EGL_COLOR_FORMAT_Y10V10U10_420_SEMIPLANAR_709_ER:
        eglFrame[0].eglColorFormat = cudaEglColorFormatY10V10U10_420SemiPlanar_709_ER
    elif cueglFrame.eglColorFormat == cydriver.CUeglColorFormat_enum.CU_EGL_COLOR_FORMAT_Y10V10U10_444_SEMIPLANAR_ER:
        eglFrame[0].eglColorFormat = cudaEglColorFormatY10V10U10_444SemiPlanar_ER
    elif cueglFrame.eglColorFormat == cydriver.CUeglColorFormat_enum.CU_EGL_COLOR_FORMAT_Y10V10U10_444_SEMIPLANAR_709_ER:
        eglFrame[0].eglColorFormat = cudaEglColorFormatY10V10U10_444SemiPlanar_709_ER
    elif cueglFrame.eglColorFormat == cydriver.CUeglColorFormat_enum.CU_EGL_COLOR_FORMAT_Y12V12U12_420_SEMIPLANAR_ER:
        eglFrame[0].eglColorFormat = cudaEglColorFormatY12V12U12_420SemiPlanar_ER
    elif cueglFrame.eglColorFormat == cydriver.CUeglColorFormat_enum.CU_EGL_COLOR_FORMAT_Y12V12U12_420_SEMIPLANAR_709_ER:
        eglFrame[0].eglColorFormat = cudaEglColorFormatY12V12U12_420SemiPlanar_709_ER
    elif cueglFrame.eglColorFormat == cydriver.CUeglColorFormat_enum.CU_EGL_COLOR_FORMAT_Y12V12U12_444_SEMIPLANAR_ER:
        eglFrame[0].eglColorFormat = cudaEglColorFormatY12V12U12_444SemiPlanar_ER
    elif cueglFrame.eglColorFormat == cydriver.CUeglColorFormat_enum.CU_EGL_COLOR_FORMAT_Y12V12U12_444_SEMIPLANAR_709_ER:
        eglFrame[0].eglColorFormat = cudaEglColorFormatY12V12U12_444SemiPlanar_709_ER
    else:
        return cudaErrorInvalidValue
    if cueglFrame.frameType == cydriver.CUeglFrameType_enum.CU_EGL_FRAME_TYPE_ARRAY:
        eglFrame[0].frameType = cudaEglFrameTypeArray
    elif cueglFrame.frameType == cydriver.CUeglFrameType_enum.CU_EGL_FRAME_TYPE_PITCH:
        eglFrame[0].frameType = cudaEglFrameTypePitch
    else:
        return cudaErrorInvalidValue
