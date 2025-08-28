# SPDX-FileCopyrightText: Copyright (c) 2021-2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: LicenseRef-NVIDIA-SOFTWARE-LICENSE

# These graphics API are the reimplemented version of what's supported by CUDA Runtime.
# Issue https://github.com/NVIDIA/cuda-python/issues/488 will remove them by letting us
# use call into the static library directly.

# This file is included from cuda/bindings/_bindings/cyruntime.pyx.in but kept in a
# separate file to keep it separated from the auto-generated code there.

# Prior to https://github.com/NVIDIA/cuda-python/pull/914, this was two
# independent modules (c.b._lib.cyruntime.cyruntime and
# c.b._lib.cyruntime.utils), but was merged into one.

from libc.string cimport memset
cimport cuda.bindings.cydriver as cydriver

cdef cudaError_t _cudaEGLStreamProducerPresentFrame(cyruntime.cudaEglStreamConnection* conn, cyruntime.cudaEglFrame eglframe, cudaStream_t* pStream) except ?cudaErrorCallRequiresNewerDriver nogil:
    cdef cudaError_t err = cudaSuccess
    # cudaFree(0) is a NOP operations that initializes the context state
    err = cudaFree(<void*>0)
    if err != cudaSuccess:
        return err
    cdef cydriver.CUeglFrame cueglFrame
    err = getDriverEglFrame(&cueglFrame, eglframe)
    if err != cudaSuccess:
        return err
    # err = <cudaError_t>cydriver._cuEGLStreamProducerPresentFrame(<cydriver.CUeglStreamConnection*>conn, cueglFrame, pStream)
    return err

cdef cudaError_t _cudaEGLStreamProducerReturnFrame(cyruntime.cudaEglStreamConnection* conn, cyruntime.cudaEglFrame* eglframe, cudaStream_t* pStream) except ?cudaErrorCallRequiresNewerDriver nogil:
    cdef cudaError_t err = cudaSuccess
    # cudaFree(0) is a NOP operations that initializes the context state
    err = cudaFree(<void*>0)
    if err != cudaSuccess:
        return err
    if eglframe == NULL:
        err = cudaErrorInvalidResourceHandle
        return err
    cdef cydriver.CUeglFrame cueglFrame
    # err = <cudaError_t>cydriver._cuEGLStreamProducerReturnFrame(<cydriver.CUeglStreamConnection*>conn, &cueglFrame, pStream)
    if err != cudaSuccess:
        return err
    err = getRuntimeEglFrame(eglframe, cueglFrame)
    return err

cdef cudaError_t _cudaGraphicsResourceGetMappedEglFrame(cyruntime.cudaEglFrame* eglFrame, cudaGraphicsResource_t resource, unsigned int index, unsigned int mipLevel) except ?cudaErrorCallRequiresNewerDriver nogil:
    cdef cudaError_t err = cudaSuccess
    # cudaFree(0) is a NOP operations that initializes the context state
    err = cudaFree(<void*>0)
    if err != cudaSuccess:
        return err
    cdef cydriver.CUeglFrame cueglFrame
    memset(&cueglFrame, 0, sizeof(cueglFrame))
    # err = <cudaError_t>cydriver._cuGraphicsResourceGetMappedEglFrame(&cueglFrame, <cydriver.CUgraphicsResource>resource, index, mipLevel)
    if err != cudaSuccess:
        return err
    err = getRuntimeEglFrame(eglFrame, cueglFrame)
    return err

cdef cudaError_t _cudaVDPAUSetVDPAUDevice(int device, cyruntime.VdpDevice vdpDevice, cyruntime.VdpGetProcAddress* vdpGetProcAddress) except ?cudaErrorCallRequiresNewerDriver nogil:
    return cudaErrorNotSupported

cdef cudaError_t _cudaVDPAUGetDevice(int* device, cyruntime.VdpDevice vdpDevice, cyruntime.VdpGetProcAddress* vdpGetProcAddress) except ?cudaErrorCallRequiresNewerDriver nogil:
    cdef cudaError_t err = cudaSuccess
    # cudaFree(0) is a NOP operations that initializes the context state
    err = cudaFree(<void*>0)
    if err != cudaSuccess:
        return err
    # err = <cudaError_t>cydriver._cuVDPAUGetDevice(<cydriver.CUdevice*>device, vdpDevice, vdpGetProcAddress)
    return err

cdef cudaError_t _cudaGraphicsVDPAURegisterVideoSurface(cudaGraphicsResource** resource, cyruntime.VdpVideoSurface vdpSurface, unsigned int flags) except ?cudaErrorCallRequiresNewerDriver nogil:
    cdef cudaError_t err = cudaSuccess
    # cudaFree(0) is a NOP operations that initializes the context state
    err = cudaFree(<void*>0)
    if err != cudaSuccess:
        return err
    # err = <cudaError_t>cydriver._cuGraphicsVDPAURegisterVideoSurface(<cydriver.CUgraphicsResource*>resource, vdpSurface, flags)
    return err

cdef cudaError_t _cudaGraphicsVDPAURegisterOutputSurface(cudaGraphicsResource** resource, cyruntime.VdpOutputSurface vdpSurface, unsigned int flags) except ?cudaErrorCallRequiresNewerDriver nogil:
    cdef cudaError_t err = cudaSuccess
    # cudaFree(0) is a NOP operations that initializes the context state
    err = cudaFree(<void*>0)
    if err != cudaSuccess:
        return err
    # err = <cudaError_t>cydriver._cuGraphicsVDPAURegisterOutputSurface(<cydriver.CUgraphicsResource*>resource, vdpSurface, flags)
    return err

cdef cudaError_t _cudaGLGetDevices(unsigned int* pCudaDeviceCount, int* pCudaDevices, unsigned int cudaDeviceCount, cyruntime.cudaGLDeviceList deviceList) except ?cudaErrorCallRequiresNewerDriver nogil:
    cdef cudaError_t err = cudaSuccess
    # cudaFree(0) is a NOP operations that initializes the context state
    err = cudaFree(<void*>0)
    if err != cudaSuccess:
        return err
    # err = <cudaError_t>cydriver._cuGLGetDevices_v2(pCudaDeviceCount, <cydriver.CUdevice*>pCudaDevices, cudaDeviceCount, <cydriver.CUGLDeviceList>deviceList)
    return err

cdef cudaError_t _cudaGraphicsGLRegisterImage(cudaGraphicsResource** resource, cyruntime.GLuint image, cyruntime.GLenum target, unsigned int flags) except ?cudaErrorCallRequiresNewerDriver nogil:
    cdef cudaError_t err = cudaSuccess
    # cudaFree(0) is a NOP operations that initializes the context state
    err = cudaFree(<void*>0)
    if err != cudaSuccess:
        return err
    # err = <cudaError_t>cydriver._cuGraphicsGLRegisterImage(<cydriver.CUgraphicsResource*>resource, image, target, flags)
    return err

cdef cudaError_t _cudaGraphicsGLRegisterBuffer(cudaGraphicsResource** resource, cyruntime.GLuint buffer, unsigned int flags) except ?cudaErrorCallRequiresNewerDriver nogil:
    cdef cudaError_t err = cudaSuccess
    # cudaFree(0) is a NOP operations that initializes the context state
    err = cudaFree(<void*>0)
    if err != cudaSuccess:
        return err
    # err = <cudaError_t>cydriver._cuGraphicsGLRegisterBuffer(<cydriver.CUgraphicsResource*>resource, buffer, flags)
    return err

cdef cudaError_t _cudaGraphicsEGLRegisterImage(cudaGraphicsResource_t* pCudaResource, cyruntime.EGLImageKHR image, unsigned int flags) except ?cudaErrorCallRequiresNewerDriver nogil:
    cdef cudaError_t err = cudaSuccess
    # cudaFree(0) is a NOP operations that initializes the context state
    err = cudaFree(<void*>0)
    if err != cudaSuccess:
        return err
    # err = <cudaError_t>cydriver._cuGraphicsEGLRegisterImage(<cydriver.CUgraphicsResource*>pCudaResource, image, flags)
    return err

cdef cudaError_t _cudaEGLStreamConsumerConnect(cyruntime.cudaEglStreamConnection* conn, cyruntime.EGLStreamKHR eglStream) except ?cudaErrorCallRequiresNewerDriver nogil:
    cdef cudaError_t err = cudaSuccess
    # cudaFree(0) is a NOP operations that initializes the context state
    err = cudaFree(<void*>0)
    if err != cudaSuccess:
        return err
    # err = <cudaError_t>cydriver._cuEGLStreamConsumerConnect(<cydriver.CUeglStreamConnection*>conn, eglStream)
    return err

cdef cudaError_t _cudaEGLStreamConsumerConnectWithFlags(cyruntime.cudaEglStreamConnection* conn, cyruntime.EGLStreamKHR eglStream, unsigned int flags) except ?cudaErrorCallRequiresNewerDriver nogil:
    cdef cudaError_t err = cudaSuccess
    # cudaFree(0) is a NOP operations that initializes the context state
    err = cudaFree(<void*>0)
    if err != cudaSuccess:
        return err
    # err = <cudaError_t>cydriver._cuEGLStreamConsumerConnectWithFlags(<cydriver.CUeglStreamConnection*>conn, eglStream, flags)
    return err

cdef cudaError_t _cudaEGLStreamConsumerDisconnect(cyruntime.cudaEglStreamConnection* conn) except ?cudaErrorCallRequiresNewerDriver nogil:
    cdef cudaError_t err = cudaSuccess
    # cudaFree(0) is a NOP operations that initializes the context state
    err = cudaFree(<void*>0)
    if err != cudaSuccess:
        return err
    # err = <cudaError_t>cydriver._cuEGLStreamConsumerDisconnect(<cydriver.CUeglStreamConnection*>conn)
    return err

cdef cudaError_t _cudaEGLStreamConsumerAcquireFrame(cyruntime.cudaEglStreamConnection* conn, cudaGraphicsResource_t* pCudaResource, cudaStream_t* pStream, unsigned int timeout) except ?cudaErrorCallRequiresNewerDriver nogil:
    cdef cudaError_t err = cudaSuccess
    # cudaFree(0) is a NOP operations that initializes the context state
    err = cudaFree(<void*>0)
    if err != cudaSuccess:
        return err
    # err = <cudaError_t>cydriver._cuEGLStreamConsumerAcquireFrame(<cydriver.CUeglStreamConnection*>conn, <cydriver.CUgraphicsResource*>pCudaResource, <cydriver.CUstream*>pStream, timeout)
    return err

cdef cudaError_t _cudaEGLStreamConsumerReleaseFrame(cyruntime.cudaEglStreamConnection* conn, cudaGraphicsResource_t pCudaResource, cudaStream_t* pStream) except ?cudaErrorCallRequiresNewerDriver nogil:
    cdef cudaError_t err = cudaSuccess
    # cudaFree(0) is a NOP operations that initializes the context state
    err = cudaFree(<void*>0)
    if err != cudaSuccess:
        return err
    # err = <cudaError_t>cydriver._cuEGLStreamConsumerReleaseFrame(<cydriver.CUeglStreamConnection*>conn, <cydriver.CUgraphicsResource>pCudaResource, <cydriver.CUstream*>pStream)
    return err

cdef cudaError_t _cudaEGLStreamProducerConnect(cyruntime.cudaEglStreamConnection* conn, cyruntime.EGLStreamKHR eglStream, cyruntime.EGLint width, cyruntime.EGLint height) except ?cudaErrorCallRequiresNewerDriver nogil:
    cdef cudaError_t err = cudaSuccess
    # cudaFree(0) is a NOP operations that initializes the context state
    err = cudaFree(<void*>0)
    if err != cudaSuccess:
        return err
    # err = <cudaError_t>cydriver._cuEGLStreamProducerConnect(<cydriver.CUeglStreamConnection*>conn, eglStream, width, height)
    return err

cdef cudaError_t _cudaEGLStreamProducerDisconnect(cyruntime.cudaEglStreamConnection* conn) except ?cudaErrorCallRequiresNewerDriver nogil:
    cdef cudaError_t err = cudaSuccess
    # cudaFree(0) is a NOP operations that initializes the context state
    err = cudaFree(<void*>0)
    if err != cudaSuccess:
        return err
    # err = <cudaError_t>cydriver._cuEGLStreamProducerDisconnect(<cydriver.CUeglStreamConnection*>conn)
    return err

cdef cudaError_t _cudaEventCreateFromEGLSync(cudaEvent_t* phEvent, cyruntime.EGLSyncKHR eglSync, unsigned int flags) except ?cudaErrorCallRequiresNewerDriver nogil:
    cdef cudaError_t err = cudaSuccess
    # cudaFree(0) is a NOP operations that initializes the context state
    err = cudaFree(<void*>0)
    if err != cudaSuccess:
        return err
    # err = <cudaError_t>cydriver._cuEventCreateFromEGLSync(<cydriver.CUevent*>phEvent, eglSync, flags)
    return err

## utility functions

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

cdef cudaError_t getDriverEglFrame(cydriver.CUeglFrame *cuEglFrame, cyruntime.cudaEglFrame eglFrame) except ?cudaErrorCallRequiresNewerDriver nogil:
    cdef cudaError_t err = cudaSuccess
    cdef unsigned int i = 0

    err = getDescInfo(&eglFrame.planeDesc[0].channelDesc, <int*>&cuEglFrame[0].numChannels, &cuEglFrame[0].cuFormat)
    if err != cudaSuccess:
        return err
    for i in range(eglFrame.planeCount):
        if eglFrame.frameType == cyruntime.cudaEglFrameTypeArray:
            cuEglFrame[0].frame.pArray[i] = <cydriver.CUarray>eglFrame.frame.pArray[i]
        else:
            cuEglFrame[0].frame.pPitch[i] = eglFrame.frame.pPitch[i].ptr
    cuEglFrame[0].width = eglFrame.planeDesc[0].width
    cuEglFrame[0].height = eglFrame.planeDesc[0].height
    cuEglFrame[0].depth = eglFrame.planeDesc[0].depth
    cuEglFrame[0].pitch = eglFrame.planeDesc[0].pitch
    cuEglFrame[0].planeCount = eglFrame.planeCount
    if eglFrame.eglColorFormat == cyruntime.cudaEglColorFormatYUV420Planar:
        cuEglFrame[0].eglColorFormat = cydriver.CUeglColorFormat_enum.CU_EGL_COLOR_FORMAT_YUV420_PLANAR
    elif eglFrame.eglColorFormat == cyruntime.cudaEglColorFormatYUV420SemiPlanar:
        cuEglFrame[0].eglColorFormat = cydriver.CUeglColorFormat_enum.CU_EGL_COLOR_FORMAT_YUV420_SEMIPLANAR
    elif eglFrame.eglColorFormat == cyruntime.cudaEglColorFormatYUV422Planar:
        cuEglFrame[0].eglColorFormat = cydriver.CUeglColorFormat_enum.CU_EGL_COLOR_FORMAT_YUV422_PLANAR
    elif eglFrame.eglColorFormat == cyruntime.cudaEglColorFormatYUV422SemiPlanar:
        cuEglFrame[0].eglColorFormat = cydriver.CUeglColorFormat_enum.CU_EGL_COLOR_FORMAT_YUV422_SEMIPLANAR
    elif eglFrame.eglColorFormat == cyruntime.cudaEglColorFormatYUV444Planar:
        cuEglFrame[0].eglColorFormat = cydriver.CUeglColorFormat_enum.CU_EGL_COLOR_FORMAT_YUV444_PLANAR
    elif eglFrame.eglColorFormat == cyruntime.cudaEglColorFormatYUV444SemiPlanar:
        cuEglFrame[0].eglColorFormat = cydriver.CUeglColorFormat_enum.CU_EGL_COLOR_FORMAT_YUV444_SEMIPLANAR
    elif eglFrame.eglColorFormat == cyruntime.cudaEglColorFormatYUYV422:
        cuEglFrame[0].eglColorFormat = cydriver.CUeglColorFormat_enum.CU_EGL_COLOR_FORMAT_YUYV_422
    elif eglFrame.eglColorFormat == cyruntime.cudaEglColorFormatUYVY422:
        cuEglFrame[0].eglColorFormat = cydriver.CUeglColorFormat_enum.CU_EGL_COLOR_FORMAT_UYVY_422
    elif eglFrame.eglColorFormat == cyruntime.cudaEglColorFormatUYVY709:
        cuEglFrame[0].eglColorFormat = cydriver.CUeglColorFormat_enum.CU_EGL_COLOR_FORMAT_UYVY_709
    elif eglFrame.eglColorFormat == cyruntime.cudaEglColorFormatUYVY709_ER:
        cuEglFrame[0].eglColorFormat = cydriver.CUeglColorFormat_enum.CU_EGL_COLOR_FORMAT_UYVY_709_ER
    elif eglFrame.eglColorFormat == cyruntime.cudaEglColorFormatUYVY2020:
        cuEglFrame[0].eglColorFormat = cydriver.CUeglColorFormat_enum.CU_EGL_COLOR_FORMAT_UYVY_2020
    elif eglFrame.eglColorFormat == cyruntime.cudaEglColorFormatARGB:
        cuEglFrame[0].eglColorFormat = cydriver.CUeglColorFormat_enum.CU_EGL_COLOR_FORMAT_ARGB
    elif eglFrame.eglColorFormat == cyruntime.cudaEglColorFormatRGBA:
        cuEglFrame[0].eglColorFormat = cydriver.CUeglColorFormat_enum.CU_EGL_COLOR_FORMAT_RGBA
    elif eglFrame.eglColorFormat == cyruntime.cudaEglColorFormatABGR:
        cuEglFrame[0].eglColorFormat = cydriver.CUeglColorFormat_enum.CU_EGL_COLOR_FORMAT_ABGR
    elif eglFrame.eglColorFormat == cyruntime.cudaEglColorFormatBGRA:
        cuEglFrame[0].eglColorFormat = cydriver.CUeglColorFormat_enum.CU_EGL_COLOR_FORMAT_BGRA
    elif eglFrame.eglColorFormat == cyruntime.cudaEglColorFormatL:
        cuEglFrame[0].eglColorFormat = cydriver.CUeglColorFormat_enum.CU_EGL_COLOR_FORMAT_L
    elif eglFrame.eglColorFormat == cyruntime.cudaEglColorFormatR:
        cuEglFrame[0].eglColorFormat = cydriver.CUeglColorFormat_enum.CU_EGL_COLOR_FORMAT_R
    elif eglFrame.eglColorFormat == cyruntime.cudaEglColorFormatA:
        cuEglFrame[0].eglColorFormat = cydriver.CUeglColorFormat_enum.CU_EGL_COLOR_FORMAT_A
    elif eglFrame.eglColorFormat == cyruntime.cudaEglColorFormatRG:
        cuEglFrame[0].eglColorFormat = cydriver.CUeglColorFormat_enum.CU_EGL_COLOR_FORMAT_RG
    elif eglFrame.eglColorFormat == cyruntime.cudaEglColorFormatAYUV:
        cuEglFrame[0].eglColorFormat = cydriver.CUeglColorFormat_enum.CU_EGL_COLOR_FORMAT_AYUV
    elif eglFrame.eglColorFormat == cyruntime.cudaEglColorFormatYVU444SemiPlanar:
        cuEglFrame[0].eglColorFormat = cydriver.CUeglColorFormat_enum.CU_EGL_COLOR_FORMAT_YVU444_SEMIPLANAR
    elif eglFrame.eglColorFormat == cyruntime.cudaEglColorFormatYVU422SemiPlanar:
        cuEglFrame[0].eglColorFormat = cydriver.CUeglColorFormat_enum.CU_EGL_COLOR_FORMAT_YVU422_SEMIPLANAR
    elif eglFrame.eglColorFormat == cyruntime.cudaEglColorFormatYVU420SemiPlanar:
        cuEglFrame[0].eglColorFormat = cydriver.CUeglColorFormat_enum.CU_EGL_COLOR_FORMAT_YVU420_SEMIPLANAR
    elif eglFrame.eglColorFormat == cyruntime.cudaEglColorFormatY10V10U10_444SemiPlanar:
        cuEglFrame[0].eglColorFormat = cydriver.CUeglColorFormat_enum.CU_EGL_COLOR_FORMAT_Y10V10U10_444_SEMIPLANAR
    elif eglFrame.eglColorFormat == cyruntime.cudaEglColorFormatY10V10U10_420SemiPlanar:
        cuEglFrame[0].eglColorFormat = cydriver.CUeglColorFormat_enum.CU_EGL_COLOR_FORMAT_Y10V10U10_420_SEMIPLANAR
    elif eglFrame.eglColorFormat == cyruntime.cudaEglColorFormatY12V12U12_444SemiPlanar:
        cuEglFrame[0].eglColorFormat = cydriver.CUeglColorFormat_enum.CU_EGL_COLOR_FORMAT_Y12V12U12_444_SEMIPLANAR
    elif eglFrame.eglColorFormat == cyruntime.cudaEglColorFormatY12V12U12_420SemiPlanar:
        cuEglFrame[0].eglColorFormat = cydriver.CUeglColorFormat_enum.CU_EGL_COLOR_FORMAT_Y12V12U12_420_SEMIPLANAR
    elif eglFrame.eglColorFormat == cyruntime.cudaEglColorFormatVYUY_ER:
        cuEglFrame[0].eglColorFormat = cydriver.CUeglColorFormat_enum.CU_EGL_COLOR_FORMAT_VYUY_ER
    elif eglFrame.eglColorFormat == cyruntime.cudaEglColorFormatUYVY_ER:
        cuEglFrame[0].eglColorFormat = cydriver.CUeglColorFormat_enum.CU_EGL_COLOR_FORMAT_UYVY_ER
    elif eglFrame.eglColorFormat == cyruntime.cudaEglColorFormatYUYV_ER:
        cuEglFrame[0].eglColorFormat = cydriver.CUeglColorFormat_enum.CU_EGL_COLOR_FORMAT_YUYV_ER
    elif eglFrame.eglColorFormat == cyruntime.cudaEglColorFormatYVYU_ER:
        cuEglFrame[0].eglColorFormat = cydriver.CUeglColorFormat_enum.CU_EGL_COLOR_FORMAT_YVYU_ER
    elif eglFrame.eglColorFormat == cyruntime.cudaEglColorFormatYUVA_ER:
        cuEglFrame[0].eglColorFormat = cydriver.CUeglColorFormat_enum.CU_EGL_COLOR_FORMAT_YUVA_ER
    elif eglFrame.eglColorFormat == cyruntime.cudaEglColorFormatAYUV_ER:
        cuEglFrame[0].eglColorFormat = cydriver.CUeglColorFormat_enum.CU_EGL_COLOR_FORMAT_AYUV_ER
    elif eglFrame.eglColorFormat == cyruntime.cudaEglColorFormatYUV444Planar_ER:
        cuEglFrame[0].eglColorFormat = cydriver.CUeglColorFormat_enum.CU_EGL_COLOR_FORMAT_YUV444_PLANAR_ER
    elif eglFrame.eglColorFormat == cyruntime.cudaEglColorFormatYUV422Planar_ER:
        cuEglFrame[0].eglColorFormat = cydriver.CUeglColorFormat_enum.CU_EGL_COLOR_FORMAT_YUV422_PLANAR_ER
    elif eglFrame.eglColorFormat == cyruntime.cudaEglColorFormatYUV420Planar_ER:
        cuEglFrame[0].eglColorFormat = cydriver.CUeglColorFormat_enum.CU_EGL_COLOR_FORMAT_YUV420_PLANAR_ER
    elif eglFrame.eglColorFormat == cyruntime.cudaEglColorFormatYUV444SemiPlanar_ER:
        cuEglFrame[0].eglColorFormat = cydriver.CUeglColorFormat_enum.CU_EGL_COLOR_FORMAT_YUV444_SEMIPLANAR_ER
    elif eglFrame.eglColorFormat == cyruntime.cudaEglColorFormatYUV422SemiPlanar_ER:
        cuEglFrame[0].eglColorFormat = cydriver.CUeglColorFormat_enum.CU_EGL_COLOR_FORMAT_YUV422_SEMIPLANAR_ER
    elif eglFrame.eglColorFormat == cyruntime.cudaEglColorFormatYUV420SemiPlanar_ER:
        cuEglFrame[0].eglColorFormat = cydriver.CUeglColorFormat_enum.CU_EGL_COLOR_FORMAT_YUV420_SEMIPLANAR_ER
    elif eglFrame.eglColorFormat == cyruntime.cudaEglColorFormatYVU444Planar_ER:
        cuEglFrame[0].eglColorFormat = cydriver.CUeglColorFormat_enum.CU_EGL_COLOR_FORMAT_YVU444_PLANAR_ER
    elif eglFrame.eglColorFormat == cyruntime.cudaEglColorFormatYVU422Planar_ER:
        cuEglFrame[0].eglColorFormat = cydriver.CUeglColorFormat_enum.CU_EGL_COLOR_FORMAT_YVU422_PLANAR_ER
    elif eglFrame.eglColorFormat == cyruntime.cudaEglColorFormatYVU420Planar_ER:
        cuEglFrame[0].eglColorFormat = cydriver.CUeglColorFormat_enum.CU_EGL_COLOR_FORMAT_YVU420_PLANAR_ER
    elif eglFrame.eglColorFormat == cyruntime.cudaEglColorFormatYVU444SemiPlanar_ER:
        cuEglFrame[0].eglColorFormat = cydriver.CUeglColorFormat_enum.CU_EGL_COLOR_FORMAT_YVU444_SEMIPLANAR_ER
    elif eglFrame.eglColorFormat == cyruntime.cudaEglColorFormatYVU422SemiPlanar_ER:
        cuEglFrame[0].eglColorFormat = cydriver.CUeglColorFormat_enum.CU_EGL_COLOR_FORMAT_YVU422_SEMIPLANAR_ER
    elif eglFrame.eglColorFormat == cyruntime.cudaEglColorFormatYVU420SemiPlanar_ER:
        cuEglFrame[0].eglColorFormat = cydriver.CUeglColorFormat_enum.CU_EGL_COLOR_FORMAT_YVU420_SEMIPLANAR_ER
    elif eglFrame.eglColorFormat == cyruntime.cudaEglColorFormatBayerRGGB:
        cuEglFrame[0].eglColorFormat = cydriver.CUeglColorFormat_enum.CU_EGL_COLOR_FORMAT_BAYER_RGGB
    elif eglFrame.eglColorFormat == cyruntime.cudaEglColorFormatBayerBGGR:
        cuEglFrame[0].eglColorFormat = cydriver.CUeglColorFormat_enum.CU_EGL_COLOR_FORMAT_BAYER_BGGR
    elif eglFrame.eglColorFormat == cyruntime.cudaEglColorFormatBayerGRBG:
        cuEglFrame[0].eglColorFormat = cydriver.CUeglColorFormat_enum.CU_EGL_COLOR_FORMAT_BAYER_GRBG
    elif eglFrame.eglColorFormat == cyruntime.cudaEglColorFormatBayerGBRG:
        cuEglFrame[0].eglColorFormat = cydriver.CUeglColorFormat_enum.CU_EGL_COLOR_FORMAT_BAYER_GBRG
    elif eglFrame.eglColorFormat == cyruntime.cudaEglColorFormatBayer10RGGB:
        cuEglFrame[0].eglColorFormat = cydriver.CUeglColorFormat_enum.CU_EGL_COLOR_FORMAT_BAYER10_RGGB
    elif eglFrame.eglColorFormat == cyruntime.cudaEglColorFormatBayer10BGGR:
        cuEglFrame[0].eglColorFormat = cydriver.CUeglColorFormat_enum.CU_EGL_COLOR_FORMAT_BAYER10_BGGR
    elif eglFrame.eglColorFormat == cyruntime.cudaEglColorFormatBayer10GRBG:
        cuEglFrame[0].eglColorFormat = cydriver.CUeglColorFormat_enum.CU_EGL_COLOR_FORMAT_BAYER10_GRBG
    elif eglFrame.eglColorFormat == cyruntime.cudaEglColorFormatBayer10GBRG:
        cuEglFrame[0].eglColorFormat = cydriver.CUeglColorFormat_enum.CU_EGL_COLOR_FORMAT_BAYER10_GBRG
    elif eglFrame.eglColorFormat == cyruntime.cudaEglColorFormatBayer12RGGB:
        cuEglFrame[0].eglColorFormat = cydriver.CUeglColorFormat_enum.CU_EGL_COLOR_FORMAT_BAYER12_RGGB
    elif eglFrame.eglColorFormat == cyruntime.cudaEglColorFormatBayer12BGGR:
        cuEglFrame[0].eglColorFormat = cydriver.CUeglColorFormat_enum.CU_EGL_COLOR_FORMAT_BAYER12_BGGR
    elif eglFrame.eglColorFormat == cyruntime.cudaEglColorFormatBayer12GRBG:
        cuEglFrame[0].eglColorFormat = cydriver.CUeglColorFormat_enum.CU_EGL_COLOR_FORMAT_BAYER12_GRBG
    elif eglFrame.eglColorFormat == cyruntime.cudaEglColorFormatBayer12GBRG:
        cuEglFrame[0].eglColorFormat = cydriver.CUeglColorFormat_enum.CU_EGL_COLOR_FORMAT_BAYER12_GBRG
    elif eglFrame.eglColorFormat == cyruntime.cudaEglColorFormatBayer14RGGB:
        cuEglFrame[0].eglColorFormat = cydriver.CUeglColorFormat_enum.CU_EGL_COLOR_FORMAT_BAYER14_RGGB
    elif eglFrame.eglColorFormat == cyruntime.cudaEglColorFormatBayer14BGGR:
        cuEglFrame[0].eglColorFormat = cydriver.CUeglColorFormat_enum.CU_EGL_COLOR_FORMAT_BAYER14_BGGR
    elif eglFrame.eglColorFormat == cyruntime.cudaEglColorFormatBayer14GRBG:
        cuEglFrame[0].eglColorFormat = cydriver.CUeglColorFormat_enum.CU_EGL_COLOR_FORMAT_BAYER14_GRBG
    elif eglFrame.eglColorFormat == cyruntime.cudaEglColorFormatBayer14GBRG:
        cuEglFrame[0].eglColorFormat = cydriver.CUeglColorFormat_enum.CU_EGL_COLOR_FORMAT_BAYER14_GBRG
    elif eglFrame.eglColorFormat == cyruntime.cudaEglColorFormatBayer20RGGB:
        cuEglFrame[0].eglColorFormat = cydriver.CUeglColorFormat_enum.CU_EGL_COLOR_FORMAT_BAYER20_RGGB
    elif eglFrame.eglColorFormat == cyruntime.cudaEglColorFormatBayer20BGGR:
        cuEglFrame[0].eglColorFormat = cydriver.CUeglColorFormat_enum.CU_EGL_COLOR_FORMAT_BAYER20_BGGR
    elif eglFrame.eglColorFormat == cyruntime.cudaEglColorFormatBayer20GRBG:
        cuEglFrame[0].eglColorFormat = cydriver.CUeglColorFormat_enum.CU_EGL_COLOR_FORMAT_BAYER20_GRBG
    elif eglFrame.eglColorFormat == cyruntime.cudaEglColorFormatBayer20GBRG:
        cuEglFrame[0].eglColorFormat = cydriver.CUeglColorFormat_enum.CU_EGL_COLOR_FORMAT_BAYER20_GBRG
    elif eglFrame.eglColorFormat == cyruntime.cudaEglColorFormatBayerIspRGGB:
        cuEglFrame[0].eglColorFormat = cydriver.CUeglColorFormat_enum.CU_EGL_COLOR_FORMAT_BAYER_ISP_RGGB
    elif eglFrame.eglColorFormat == cyruntime.cudaEglColorFormatBayerIspBGGR:
        cuEglFrame[0].eglColorFormat = cydriver.CUeglColorFormat_enum.CU_EGL_COLOR_FORMAT_BAYER_ISP_BGGR
    elif eglFrame.eglColorFormat == cyruntime.cudaEglColorFormatBayerIspGRBG:
        cuEglFrame[0].eglColorFormat = cydriver.CUeglColorFormat_enum.CU_EGL_COLOR_FORMAT_BAYER_ISP_GRBG
    elif eglFrame.eglColorFormat == cyruntime.cudaEglColorFormatBayerIspGBRG:
        cuEglFrame[0].eglColorFormat = cydriver.CUeglColorFormat_enum.CU_EGL_COLOR_FORMAT_BAYER_ISP_GBRG
    elif eglFrame.eglColorFormat == cyruntime.cudaEglColorFormatYVU444Planar:
        cuEglFrame[0].eglColorFormat = cydriver.CUeglColorFormat_enum.CU_EGL_COLOR_FORMAT_YVU444_PLANAR
    elif eglFrame.eglColorFormat == cyruntime.cudaEglColorFormatYVU422Planar:
        cuEglFrame[0].eglColorFormat = cydriver.CUeglColorFormat_enum.CU_EGL_COLOR_FORMAT_YVU422_PLANAR
    elif eglFrame.eglColorFormat == cyruntime.cudaEglColorFormatYVU420Planar:
        cuEglFrame[0].eglColorFormat = cydriver.CUeglColorFormat_enum.CU_EGL_COLOR_FORMAT_YVU420_PLANAR
    elif eglFrame.eglColorFormat == cyruntime.cudaEglColorFormatBayerBCCR:
        cuEglFrame[0].eglColorFormat = cydriver.CUeglColorFormat_enum.CU_EGL_COLOR_FORMAT_BAYER_BCCR
    elif eglFrame.eglColorFormat == cyruntime.cudaEglColorFormatBayerRCCB:
        cuEglFrame[0].eglColorFormat = cydriver.CUeglColorFormat_enum.CU_EGL_COLOR_FORMAT_BAYER_RCCB
    elif eglFrame.eglColorFormat == cyruntime.cudaEglColorFormatBayerCRBC:
        cuEglFrame[0].eglColorFormat = cydriver.CUeglColorFormat_enum.CU_EGL_COLOR_FORMAT_BAYER_CRBC
    elif eglFrame.eglColorFormat == cyruntime.cudaEglColorFormatBayerCBRC:
        cuEglFrame[0].eglColorFormat = cydriver.CUeglColorFormat_enum.CU_EGL_COLOR_FORMAT_BAYER_CBRC
    elif eglFrame.eglColorFormat == cyruntime.cudaEglColorFormatBayer10CCCC:
        cuEglFrame[0].eglColorFormat = cydriver.CUeglColorFormat_enum.CU_EGL_COLOR_FORMAT_BAYER10_CCCC
    elif eglFrame.eglColorFormat == cyruntime.cudaEglColorFormatBayer12BCCR:
        cuEglFrame[0].eglColorFormat = cydriver.CUeglColorFormat_enum.CU_EGL_COLOR_FORMAT_BAYER12_BCCR
    elif eglFrame.eglColorFormat == cyruntime.cudaEglColorFormatBayer12RCCB:
        cuEglFrame[0].eglColorFormat = cydriver.CUeglColorFormat_enum.CU_EGL_COLOR_FORMAT_BAYER12_RCCB
    elif eglFrame.eglColorFormat == cyruntime.cudaEglColorFormatBayer12CRBC:
        cuEglFrame[0].eglColorFormat = cydriver.CUeglColorFormat_enum.CU_EGL_COLOR_FORMAT_BAYER12_CRBC
    elif eglFrame.eglColorFormat == cyruntime.cudaEglColorFormatBayer12CBRC:
        cuEglFrame[0].eglColorFormat = cydriver.CUeglColorFormat_enum.CU_EGL_COLOR_FORMAT_BAYER12_CBRC
    elif eglFrame.eglColorFormat == cyruntime.cudaEglColorFormatBayer12CCCC:
        cuEglFrame[0].eglColorFormat = cydriver.CUeglColorFormat_enum.CU_EGL_COLOR_FORMAT_BAYER12_CCCC
    elif eglFrame.eglColorFormat == cyruntime.cudaEglColorFormatY:
        cuEglFrame[0].eglColorFormat = cydriver.CUeglColorFormat_enum.CU_EGL_COLOR_FORMAT_Y
    elif eglFrame.eglColorFormat == cyruntime.cudaEglColorFormatYUV420SemiPlanar_2020:
        cuEglFrame[0].eglColorFormat = cydriver.CUeglColorFormat_enum.CU_EGL_COLOR_FORMAT_YUV420_SEMIPLANAR_2020
    elif eglFrame.eglColorFormat == cyruntime.cudaEglColorFormatYVU420SemiPlanar_2020:
        cuEglFrame[0].eglColorFormat = cydriver.CUeglColorFormat_enum.CU_EGL_COLOR_FORMAT_YVU420_SEMIPLANAR_2020
    elif eglFrame.eglColorFormat == cyruntime.cudaEglColorFormatYUV420Planar_2020:
        cuEglFrame[0].eglColorFormat = cydriver.CUeglColorFormat_enum.CU_EGL_COLOR_FORMAT_YUV420_PLANAR_2020
    elif eglFrame.eglColorFormat == cyruntime.cudaEglColorFormatYVU420Planar_2020:
        cuEglFrame[0].eglColorFormat = cydriver.CUeglColorFormat_enum.CU_EGL_COLOR_FORMAT_YVU420_PLANAR_2020
    elif eglFrame.eglColorFormat == cyruntime.cudaEglColorFormatYUV420SemiPlanar_709:
        cuEglFrame[0].eglColorFormat = cydriver.CUeglColorFormat_enum.CU_EGL_COLOR_FORMAT_YUV420_SEMIPLANAR_709
    elif eglFrame.eglColorFormat == cyruntime.cudaEglColorFormatYVU420SemiPlanar_709:
        cuEglFrame[0].eglColorFormat = cydriver.CUeglColorFormat_enum.CU_EGL_COLOR_FORMAT_YVU420_SEMIPLANAR_709
    elif eglFrame.eglColorFormat == cyruntime.cudaEglColorFormatYUV420Planar_709:
        cuEglFrame[0].eglColorFormat = cydriver.CUeglColorFormat_enum.CU_EGL_COLOR_FORMAT_YUV420_PLANAR_709
    elif eglFrame.eglColorFormat == cyruntime.cudaEglColorFormatYVU420Planar_709:
        cuEglFrame[0].eglColorFormat = cydriver.CUeglColorFormat_enum.CU_EGL_COLOR_FORMAT_YVU420_PLANAR_709
    elif eglFrame.eglColorFormat == cyruntime.cudaEglColorFormatY10V10U10_420SemiPlanar_709:
        cuEglFrame[0].eglColorFormat = cydriver.CUeglColorFormat_enum.CU_EGL_COLOR_FORMAT_Y10V10U10_420_SEMIPLANAR_709
    elif eglFrame.eglColorFormat == cyruntime.cudaEglColorFormatY10V10U10_420SemiPlanar_2020:
        cuEglFrame[0].eglColorFormat = cydriver.CUeglColorFormat_enum.CU_EGL_COLOR_FORMAT_Y10V10U10_420_SEMIPLANAR_2020
    elif eglFrame.eglColorFormat == cyruntime.cudaEglColorFormatY10V10U10_422SemiPlanar_2020:
        cuEglFrame[0].eglColorFormat = cydriver.CUeglColorFormat_enum.CU_EGL_COLOR_FORMAT_Y10V10U10_422_SEMIPLANAR_2020
    elif eglFrame.eglColorFormat == cyruntime.cudaEglColorFormatY10V10U10_422SemiPlanar:
        cuEglFrame[0].eglColorFormat = cydriver.CUeglColorFormat_enum.CU_EGL_COLOR_FORMAT_Y10V10U10_422_SEMIPLANAR
    elif eglFrame.eglColorFormat == cyruntime.cudaEglColorFormatY10V10U10_422SemiPlanar_709:
        cuEglFrame[0].eglColorFormat = cydriver.CUeglColorFormat_enum.CU_EGL_COLOR_FORMAT_Y10V10U10_422_SEMIPLANAR_709
    elif eglFrame.eglColorFormat == cyruntime.cudaEglColorFormatY_ER:
        cuEglFrame[0].eglColorFormat = cydriver.CUeglColorFormat_enum.CU_EGL_COLOR_FORMAT_Y_ER
    elif eglFrame.eglColorFormat == cyruntime.cudaEglColorFormatY_709_ER:
        cuEglFrame[0].eglColorFormat = cydriver.CUeglColorFormat_enum.CU_EGL_COLOR_FORMAT_Y_709_ER
    elif eglFrame.eglColorFormat == cyruntime.cudaEglColorFormatY10_ER:
        cuEglFrame[0].eglColorFormat = cydriver.CUeglColorFormat_enum.CU_EGL_COLOR_FORMAT_Y10_ER
    elif eglFrame.eglColorFormat == cyruntime.cudaEglColorFormatY10_709_ER:
        cuEglFrame[0].eglColorFormat = cydriver.CUeglColorFormat_enum.CU_EGL_COLOR_FORMAT_Y10_709_ER
    elif eglFrame.eglColorFormat == cyruntime.cudaEglColorFormatY12_ER:
        cuEglFrame[0].eglColorFormat = cydriver.CUeglColorFormat_enum.CU_EGL_COLOR_FORMAT_Y12_ER
    elif eglFrame.eglColorFormat == cyruntime.cudaEglColorFormatY12_709_ER:
        cuEglFrame[0].eglColorFormat = cydriver.CUeglColorFormat_enum.CU_EGL_COLOR_FORMAT_Y12_709_ER
    elif eglFrame.eglColorFormat == cyruntime.cudaEglColorFormatYUVA:
        cuEglFrame[0].eglColorFormat = cydriver.CUeglColorFormat_enum.CU_EGL_COLOR_FORMAT_YUVA
    elif eglFrame.eglColorFormat == cyruntime.cudaEglColorFormatYVYU:
        cuEglFrame[0].eglColorFormat = cydriver.CUeglColorFormat_enum.CU_EGL_COLOR_FORMAT_YVYU
    elif eglFrame.eglColorFormat == cyruntime.cudaEglColorFormatVYUY:
        cuEglFrame[0].eglColorFormat = cydriver.CUeglColorFormat_enum.CU_EGL_COLOR_FORMAT_VYUY
    elif eglFrame.eglColorFormat == cyruntime.cudaEglColorFormatY10V10U10_420SemiPlanar_ER:
        cuEglFrame[0].eglColorFormat = cydriver.CUeglColorFormat_enum.CU_EGL_COLOR_FORMAT_Y10V10U10_420_SEMIPLANAR_ER
    elif eglFrame.eglColorFormat == cyruntime.cudaEglColorFormatY10V10U10_420SemiPlanar_709_ER:
        cuEglFrame[0].eglColorFormat = cydriver.CUeglColorFormat_enum.CU_EGL_COLOR_FORMAT_Y10V10U10_420_SEMIPLANAR_709_ER
    elif eglFrame.eglColorFormat == cyruntime.cudaEglColorFormatY10V10U10_444SemiPlanar_ER:
        cuEglFrame[0].eglColorFormat =  cydriver.CUeglColorFormat_enum.CU_EGL_COLOR_FORMAT_Y10V10U10_444_SEMIPLANAR_ER
    elif eglFrame.eglColorFormat == cyruntime.cudaEglColorFormatY10V10U10_444SemiPlanar_709_ER:
        cuEglFrame[0].eglColorFormat = cydriver.CUeglColorFormat_enum.CU_EGL_COLOR_FORMAT_Y10V10U10_444_SEMIPLANAR_709_ER
    elif eglFrame.eglColorFormat == cyruntime.cudaEglColorFormatY12V12U12_420SemiPlanar_ER:
        cuEglFrame[0].eglColorFormat = cydriver.CUeglColorFormat_enum.CU_EGL_COLOR_FORMAT_Y12V12U12_420_SEMIPLANAR_ER
    elif eglFrame.eglColorFormat == cyruntime.cudaEglColorFormatY12V12U12_420SemiPlanar_709_ER:
        cuEglFrame[0].eglColorFormat = cydriver.CUeglColorFormat_enum.CU_EGL_COLOR_FORMAT_Y12V12U12_420_SEMIPLANAR_709_ER
    elif eglFrame.eglColorFormat == cyruntime.cudaEglColorFormatY12V12U12_444SemiPlanar_ER:
        cuEglFrame[0].eglColorFormat = cydriver.CUeglColorFormat_enum.CU_EGL_COLOR_FORMAT_Y12V12U12_444_SEMIPLANAR_ER
    elif eglFrame.eglColorFormat == cyruntime.cudaEglColorFormatY12V12U12_444SemiPlanar_709_ER:
        cuEglFrame[0].eglColorFormat = cydriver.CUeglColorFormat_enum.CU_EGL_COLOR_FORMAT_Y12V12U12_444_SEMIPLANAR_709_ER
    else:
        return cudaErrorInvalidValue
    if eglFrame.frameType == cyruntime.cudaEglFrameTypeArray:
        cuEglFrame[0].frameType = cydriver.CUeglFrameType_enum.CU_EGL_FRAME_TYPE_ARRAY
    elif eglFrame.frameType == cyruntime.cudaEglFrameTypePitch:
        cuEglFrame[0].frameType = cydriver.CUeglFrameType_enum.CU_EGL_FRAME_TYPE_PITCH
    else:
        return cudaErrorInvalidValue

@cython.show_performance_hints(False)
cdef cudaError_t getRuntimeEglFrame(cyruntime.cudaEglFrame *eglFrame, cydriver.CUeglFrame cueglFrame) except ?cudaErrorCallRequiresNewerDriver nogil:
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
        eglFrame[0].eglColorFormat = cyruntime.cudaEglColorFormatYUV420Planar
    elif cueglFrame.eglColorFormat == cydriver.CUeglColorFormat_enum.CU_EGL_COLOR_FORMAT_YUV420_SEMIPLANAR:
        eglFrame[0].eglColorFormat = cyruntime.cudaEglColorFormatYUV420SemiPlanar
    elif cueglFrame.eglColorFormat == cydriver.CUeglColorFormat_enum.CU_EGL_COLOR_FORMAT_YUV422_PLANAR:
        eglFrame[0].eglColorFormat = cyruntime.cudaEglColorFormatYUV422Planar
    elif cueglFrame.eglColorFormat == cydriver.CUeglColorFormat_enum.CU_EGL_COLOR_FORMAT_YUV422_SEMIPLANAR:
        eglFrame[0].eglColorFormat = cyruntime.cudaEglColorFormatYUV422SemiPlanar
    elif cueglFrame.eglColorFormat == cydriver.CUeglColorFormat_enum.CU_EGL_COLOR_FORMAT_YUV444_PLANAR:
        eglFrame[0].eglColorFormat = cyruntime.cudaEglColorFormatYUV444Planar
    elif cueglFrame.eglColorFormat == cydriver.CUeglColorFormat_enum.CU_EGL_COLOR_FORMAT_YUV444_SEMIPLANAR:
        eglFrame[0].eglColorFormat = cyruntime.cudaEglColorFormatYUV444SemiPlanar
    elif cueglFrame.eglColorFormat == cydriver.CUeglColorFormat_enum.CU_EGL_COLOR_FORMAT_YUYV_422:
        eglFrame[0].eglColorFormat = cyruntime.cudaEglColorFormatYUYV422
    elif cueglFrame.eglColorFormat == cydriver.CUeglColorFormat_enum.CU_EGL_COLOR_FORMAT_UYVY_422:
        eglFrame[0].eglColorFormat = cyruntime.cudaEglColorFormatUYVY422
    elif cueglFrame.eglColorFormat == cydriver.CUeglColorFormat_enum.CU_EGL_COLOR_FORMAT_UYVY_709:
        eglFrame[0].eglColorFormat = cyruntime.cudaEglColorFormatUYVY709
    elif cueglFrame.eglColorFormat == cydriver.CUeglColorFormat_enum.CU_EGL_COLOR_FORMAT_UYVY_709_ER:
        eglFrame[0].eglColorFormat = cyruntime.cudaEglColorFormatUYVY709_ER
    elif cueglFrame.eglColorFormat == cydriver.CUeglColorFormat_enum.CU_EGL_COLOR_FORMAT_UYVY_2020:
        eglFrame[0].eglColorFormat = cyruntime.cudaEglColorFormatUYVY2020
    elif cueglFrame.eglColorFormat == cydriver.CUeglColorFormat_enum.CU_EGL_COLOR_FORMAT_ARGB:
        eglFrame[0].eglColorFormat = cyruntime.cudaEglColorFormatARGB
    elif cueglFrame.eglColorFormat == cydriver.CUeglColorFormat_enum.CU_EGL_COLOR_FORMAT_RGBA:
        eglFrame[0].eglColorFormat = cyruntime.cudaEglColorFormatRGBA
    elif cueglFrame.eglColorFormat == cydriver.CUeglColorFormat_enum.CU_EGL_COLOR_FORMAT_ABGR:
        eglFrame[0].eglColorFormat = cyruntime.cudaEglColorFormatABGR
    elif cueglFrame.eglColorFormat == cydriver.CUeglColorFormat_enum.CU_EGL_COLOR_FORMAT_BGRA:
        eglFrame[0].eglColorFormat = cyruntime.cudaEglColorFormatBGRA
    elif cueglFrame.eglColorFormat == cydriver.CUeglColorFormat_enum.CU_EGL_COLOR_FORMAT_L:
        eglFrame[0].eglColorFormat = cyruntime.cudaEglColorFormatL
    elif cueglFrame.eglColorFormat == cydriver.CUeglColorFormat_enum.CU_EGL_COLOR_FORMAT_R:
        eglFrame[0].eglColorFormat = cyruntime.cudaEglColorFormatR
    elif cueglFrame.eglColorFormat == cydriver.CUeglColorFormat_enum.CU_EGL_COLOR_FORMAT_A:
        eglFrame[0].eglColorFormat = cyruntime.cudaEglColorFormatA
    elif cueglFrame.eglColorFormat == cydriver.CUeglColorFormat_enum.CU_EGL_COLOR_FORMAT_RG:
        eglFrame[0].eglColorFormat = cyruntime.cudaEglColorFormatRG
    elif cueglFrame.eglColorFormat == cydriver.CUeglColorFormat_enum.CU_EGL_COLOR_FORMAT_AYUV:
        eglFrame[0].eglColorFormat = cyruntime.cudaEglColorFormatAYUV
    elif cueglFrame.eglColorFormat == cydriver.CUeglColorFormat_enum.CU_EGL_COLOR_FORMAT_YVU444_SEMIPLANAR:
        eglFrame[0].eglColorFormat = cyruntime.cudaEglColorFormatYVU444SemiPlanar
    elif cueglFrame.eglColorFormat == cydriver.CUeglColorFormat_enum.CU_EGL_COLOR_FORMAT_YVU422_SEMIPLANAR:
        eglFrame[0].eglColorFormat = cyruntime.cudaEglColorFormatYVU422SemiPlanar
    elif cueglFrame.eglColorFormat == cydriver.CUeglColorFormat_enum.CU_EGL_COLOR_FORMAT_YVU420_SEMIPLANAR:
        eglFrame[0].eglColorFormat = cyruntime.cudaEglColorFormatYVU420SemiPlanar
    elif cueglFrame.eglColorFormat == cydriver.CUeglColorFormat_enum.CU_EGL_COLOR_FORMAT_Y10V10U10_444_SEMIPLANAR:
        eglFrame[0].eglColorFormat = cyruntime.cudaEglColorFormatY10V10U10_444SemiPlanar
    elif cueglFrame.eglColorFormat == cydriver.CUeglColorFormat_enum.CU_EGL_COLOR_FORMAT_Y10V10U10_420_SEMIPLANAR:
        eglFrame[0].eglColorFormat = cyruntime.cudaEglColorFormatY10V10U10_420SemiPlanar
    elif cueglFrame.eglColorFormat == cydriver.CUeglColorFormat_enum.CU_EGL_COLOR_FORMAT_Y12V12U12_444_SEMIPLANAR:
        eglFrame[0].eglColorFormat = cyruntime.cudaEglColorFormatY12V12U12_444SemiPlanar
    elif cueglFrame.eglColorFormat == cydriver.CUeglColorFormat_enum.CU_EGL_COLOR_FORMAT_Y12V12U12_420_SEMIPLANAR:
        eglFrame[0].eglColorFormat = cyruntime.cudaEglColorFormatY12V12U12_420SemiPlanar
    elif cueglFrame.eglColorFormat == cydriver.CUeglColorFormat_enum.CU_EGL_COLOR_FORMAT_VYUY_ER:
        eglFrame[0].eglColorFormat = cyruntime.cudaEglColorFormatVYUY_ER
    elif cueglFrame.eglColorFormat == cydriver.CUeglColorFormat_enum.CU_EGL_COLOR_FORMAT_UYVY_ER:
        eglFrame[0].eglColorFormat = cyruntime.cudaEglColorFormatUYVY_ER
    elif cueglFrame.eglColorFormat == cydriver.CUeglColorFormat_enum.CU_EGL_COLOR_FORMAT_YUYV_ER:
        eglFrame[0].eglColorFormat = cyruntime.cudaEglColorFormatYUYV_ER
    elif cueglFrame.eglColorFormat == cydriver.CUeglColorFormat_enum.CU_EGL_COLOR_FORMAT_YVYU_ER:
        eglFrame[0].eglColorFormat = cyruntime.cudaEglColorFormatYVYU_ER
    elif cueglFrame.eglColorFormat == cydriver.CUeglColorFormat_enum.CU_EGL_COLOR_FORMAT_YUVA_ER:
        eglFrame[0].eglColorFormat = cyruntime.cudaEglColorFormatYUVA_ER
    elif cueglFrame.eglColorFormat == cydriver.CUeglColorFormat_enum.CU_EGL_COLOR_FORMAT_AYUV_ER:
        eglFrame[0].eglColorFormat = cyruntime.cudaEglColorFormatAYUV_ER
    elif cueglFrame.eglColorFormat == cydriver.CUeglColorFormat_enum.CU_EGL_COLOR_FORMAT_YUV444_PLANAR_ER:
        eglFrame[0].eglColorFormat = cyruntime.cudaEglColorFormatYUV444Planar_ER
    elif cueglFrame.eglColorFormat == cydriver.CUeglColorFormat_enum.CU_EGL_COLOR_FORMAT_YUV422_PLANAR_ER:
        eglFrame[0].eglColorFormat = cyruntime.cudaEglColorFormatYUV422Planar_ER
    elif cueglFrame.eglColorFormat == cydriver.CUeglColorFormat_enum.CU_EGL_COLOR_FORMAT_YUV420_PLANAR_ER:
        eglFrame[0].eglColorFormat = cyruntime.cudaEglColorFormatYUV420Planar_ER
    elif cueglFrame.eglColorFormat == cydriver.CUeglColorFormat_enum.CU_EGL_COLOR_FORMAT_YUV444_SEMIPLANAR_ER:
        eglFrame[0].eglColorFormat = cyruntime.cudaEglColorFormatYUV444SemiPlanar_ER
    elif cueglFrame.eglColorFormat == cydriver.CUeglColorFormat_enum.CU_EGL_COLOR_FORMAT_YUV422_SEMIPLANAR_ER:
        eglFrame[0].eglColorFormat = cyruntime.cudaEglColorFormatYUV422SemiPlanar_ER
    elif cueglFrame.eglColorFormat == cydriver.CUeglColorFormat_enum.CU_EGL_COLOR_FORMAT_YUV420_SEMIPLANAR_ER:
        eglFrame[0].eglColorFormat = cyruntime.cudaEglColorFormatYUV420SemiPlanar_ER
    elif cueglFrame.eglColorFormat == cydriver.CUeglColorFormat_enum.CU_EGL_COLOR_FORMAT_YVU444_PLANAR_ER:
        eglFrame[0].eglColorFormat = cyruntime.cudaEglColorFormatYVU444Planar_ER
    elif cueglFrame.eglColorFormat == cydriver.CUeglColorFormat_enum.CU_EGL_COLOR_FORMAT_YVU422_PLANAR_ER:
        eglFrame[0].eglColorFormat = cyruntime.cudaEglColorFormatYVU422Planar_ER
    elif cueglFrame.eglColorFormat == cydriver.CUeglColorFormat_enum.CU_EGL_COLOR_FORMAT_YVU420_PLANAR_ER:
        eglFrame[0].eglColorFormat = cyruntime.cudaEglColorFormatYVU420Planar_ER
    elif cueglFrame.eglColorFormat == cydriver.CUeglColorFormat_enum.CU_EGL_COLOR_FORMAT_YVU444_SEMIPLANAR_ER:
        eglFrame[0].eglColorFormat = cyruntime.cudaEglColorFormatYVU444SemiPlanar_ER
    elif cueglFrame.eglColorFormat == cydriver.CUeglColorFormat_enum.CU_EGL_COLOR_FORMAT_YVU422_SEMIPLANAR_ER:
        eglFrame[0].eglColorFormat = cyruntime.cudaEglColorFormatYVU422SemiPlanar_ER
    elif cueglFrame.eglColorFormat == cydriver.CUeglColorFormat_enum.CU_EGL_COLOR_FORMAT_YVU420_SEMIPLANAR_ER:
        eglFrame[0].eglColorFormat = cyruntime.cudaEglColorFormatYVU420SemiPlanar_ER
    elif cueglFrame.eglColorFormat == cydriver.CUeglColorFormat_enum.CU_EGL_COLOR_FORMAT_BAYER_RGGB:
        eglFrame[0].eglColorFormat = cyruntime.cudaEglColorFormatBayerRGGB
    elif cueglFrame.eglColorFormat == cydriver.CUeglColorFormat_enum.CU_EGL_COLOR_FORMAT_BAYER_BGGR:
        eglFrame[0].eglColorFormat = cyruntime.cudaEglColorFormatBayerBGGR
    elif cueglFrame.eglColorFormat == cydriver.CUeglColorFormat_enum.CU_EGL_COLOR_FORMAT_BAYER_GRBG:
        eglFrame[0].eglColorFormat = cyruntime.cudaEglColorFormatBayerGRBG
    elif cueglFrame.eglColorFormat == cydriver.CUeglColorFormat_enum.CU_EGL_COLOR_FORMAT_BAYER_GBRG:
        eglFrame[0].eglColorFormat = cyruntime.cudaEglColorFormatBayerGBRG
    elif cueglFrame.eglColorFormat == cydriver.CUeglColorFormat_enum.CU_EGL_COLOR_FORMAT_BAYER10_RGGB:
        eglFrame[0].eglColorFormat = cyruntime.cudaEglColorFormatBayer10RGGB
    elif cueglFrame.eglColorFormat == cydriver.CUeglColorFormat_enum.CU_EGL_COLOR_FORMAT_BAYER10_BGGR:
        eglFrame[0].eglColorFormat = cyruntime.cudaEglColorFormatBayer10BGGR
    elif cueglFrame.eglColorFormat == cydriver.CUeglColorFormat_enum.CU_EGL_COLOR_FORMAT_BAYER10_GRBG:
        eglFrame[0].eglColorFormat = cyruntime.cudaEglColorFormatBayer10GRBG
    elif cueglFrame.eglColorFormat == cydriver.CUeglColorFormat_enum.CU_EGL_COLOR_FORMAT_BAYER10_GBRG:
        eglFrame[0].eglColorFormat = cyruntime.cudaEglColorFormatBayer10GBRG
    elif cueglFrame.eglColorFormat == cydriver.CUeglColorFormat_enum.CU_EGL_COLOR_FORMAT_BAYER12_RGGB:
        eglFrame[0].eglColorFormat = cyruntime.cudaEglColorFormatBayer12RGGB
    elif cueglFrame.eglColorFormat == cydriver.CUeglColorFormat_enum.CU_EGL_COLOR_FORMAT_BAYER12_BGGR:
        eglFrame[0].eglColorFormat = cyruntime.cudaEglColorFormatBayer12BGGR
    elif cueglFrame.eglColorFormat == cydriver.CUeglColorFormat_enum.CU_EGL_COLOR_FORMAT_BAYER12_GRBG:
        eglFrame[0].eglColorFormat = cyruntime.cudaEglColorFormatBayer12GRBG
    elif cueglFrame.eglColorFormat == cydriver.CUeglColorFormat_enum.CU_EGL_COLOR_FORMAT_BAYER12_GBRG:
        eglFrame[0].eglColorFormat = cyruntime.cudaEglColorFormatBayer12GBRG
    elif cueglFrame.eglColorFormat == cydriver.CUeglColorFormat_enum.CU_EGL_COLOR_FORMAT_BAYER14_RGGB:
        eglFrame[0].eglColorFormat = cyruntime.cudaEglColorFormatBayer14RGGB
    elif cueglFrame.eglColorFormat == cydriver.CUeglColorFormat_enum.CU_EGL_COLOR_FORMAT_BAYER14_BGGR:
        eglFrame[0].eglColorFormat = cyruntime.cudaEglColorFormatBayer14BGGR
    elif cueglFrame.eglColorFormat == cydriver.CUeglColorFormat_enum.CU_EGL_COLOR_FORMAT_BAYER14_GRBG:
        eglFrame[0].eglColorFormat = cyruntime.cudaEglColorFormatBayer14GRBG
    elif cueglFrame.eglColorFormat == cydriver.CUeglColorFormat_enum.CU_EGL_COLOR_FORMAT_BAYER14_GBRG:
        eglFrame[0].eglColorFormat = cyruntime.cudaEglColorFormatBayer14GBRG
    elif cueglFrame.eglColorFormat == cydriver.CUeglColorFormat_enum.CU_EGL_COLOR_FORMAT_BAYER20_RGGB:
        eglFrame[0].eglColorFormat = cyruntime.cudaEglColorFormatBayer20RGGB
    elif cueglFrame.eglColorFormat == cydriver.CUeglColorFormat_enum.CU_EGL_COLOR_FORMAT_BAYER20_BGGR:
        eglFrame[0].eglColorFormat = cyruntime.cudaEglColorFormatBayer20BGGR
    elif cueglFrame.eglColorFormat == cydriver.CUeglColorFormat_enum.CU_EGL_COLOR_FORMAT_BAYER20_GRBG:
        eglFrame[0].eglColorFormat = cyruntime.cudaEglColorFormatBayer20GRBG
    elif cueglFrame.eglColorFormat == cydriver.CUeglColorFormat_enum.CU_EGL_COLOR_FORMAT_BAYER20_GBRG:
        eglFrame[0].eglColorFormat = cyruntime.cudaEglColorFormatBayer20GBRG
    elif cueglFrame.eglColorFormat == cydriver.CUeglColorFormat_enum.CU_EGL_COLOR_FORMAT_BAYER_ISP_RGGB:
        eglFrame[0].eglColorFormat = cyruntime.cudaEglColorFormatBayerIspRGGB
    elif cueglFrame.eglColorFormat == cydriver.CUeglColorFormat_enum.CU_EGL_COLOR_FORMAT_BAYER_ISP_BGGR:
        eglFrame[0].eglColorFormat = cyruntime.cudaEglColorFormatBayerIspBGGR
    elif cueglFrame.eglColorFormat == cydriver.CUeglColorFormat_enum.CU_EGL_COLOR_FORMAT_BAYER_ISP_GRBG:
        eglFrame[0].eglColorFormat = cyruntime.cudaEglColorFormatBayerIspGRBG
    elif cueglFrame.eglColorFormat == cydriver.CUeglColorFormat_enum.CU_EGL_COLOR_FORMAT_BAYER_ISP_GBRG:
        eglFrame[0].eglColorFormat = cyruntime.cudaEglColorFormatBayerIspGBRG
    elif cueglFrame.eglColorFormat == cydriver.CUeglColorFormat_enum.CU_EGL_COLOR_FORMAT_YVU444_PLANAR:
        eglFrame[0].eglColorFormat = cyruntime.cudaEglColorFormatYVU444Planar
    elif cueglFrame.eglColorFormat == cydriver.CUeglColorFormat_enum.CU_EGL_COLOR_FORMAT_YVU422_PLANAR:
        eglFrame[0].eglColorFormat = cyruntime.cudaEglColorFormatYVU422Planar
    elif cueglFrame.eglColorFormat == cydriver.CUeglColorFormat_enum.CU_EGL_COLOR_FORMAT_YVU420_PLANAR:
        eglFrame[0].eglColorFormat = cyruntime.cudaEglColorFormatYVU420Planar
    elif cueglFrame.eglColorFormat == cydriver.CUeglColorFormat_enum.CU_EGL_COLOR_FORMAT_BAYER_BCCR:
        eglFrame[0].eglColorFormat = cyruntime.cudaEglColorFormatBayerBCCR
    elif cueglFrame.eglColorFormat == cydriver.CUeglColorFormat_enum.CU_EGL_COLOR_FORMAT_BAYER_RCCB:
        eglFrame[0].eglColorFormat = cyruntime.cudaEglColorFormatBayerRCCB
    elif cueglFrame.eglColorFormat == cydriver.CUeglColorFormat_enum.CU_EGL_COLOR_FORMAT_BAYER_CRBC:
        eglFrame[0].eglColorFormat = cyruntime.cudaEglColorFormatBayerCRBC
    elif cueglFrame.eglColorFormat == cydriver.CUeglColorFormat_enum.CU_EGL_COLOR_FORMAT_BAYER_CBRC:
        eglFrame[0].eglColorFormat = cyruntime.cudaEglColorFormatBayerCBRC
    elif cueglFrame.eglColorFormat == cydriver.CUeglColorFormat_enum.CU_EGL_COLOR_FORMAT_BAYER10_CCCC:
        eglFrame[0].eglColorFormat = cyruntime.cudaEglColorFormatBayer10CCCC
    elif cueglFrame.eglColorFormat == cydriver.CUeglColorFormat_enum.CU_EGL_COLOR_FORMAT_BAYER12_BCCR:
        eglFrame[0].eglColorFormat = cyruntime.cudaEglColorFormatBayer12BCCR
    elif cueglFrame.eglColorFormat == cydriver.CUeglColorFormat_enum.CU_EGL_COLOR_FORMAT_BAYER12_RCCB:
        eglFrame[0].eglColorFormat = cyruntime.cudaEglColorFormatBayer12RCCB
    elif cueglFrame.eglColorFormat == cydriver.CUeglColorFormat_enum.CU_EGL_COLOR_FORMAT_BAYER12_CRBC:
        eglFrame[0].eglColorFormat = cyruntime.cudaEglColorFormatBayer12CRBC
    elif cueglFrame.eglColorFormat == cydriver.CUeglColorFormat_enum.CU_EGL_COLOR_FORMAT_BAYER12_CBRC:
        eglFrame[0].eglColorFormat = cyruntime.cudaEglColorFormatBayer12CBRC
    elif cueglFrame.eglColorFormat == cydriver.CUeglColorFormat_enum.CU_EGL_COLOR_FORMAT_BAYER12_CCCC:
        eglFrame[0].eglColorFormat = cyruntime.cudaEglColorFormatBayer12CCCC
    elif cueglFrame.eglColorFormat == cydriver.CUeglColorFormat_enum.CU_EGL_COLOR_FORMAT_Y:
        eglFrame[0].eglColorFormat = cyruntime.cudaEglColorFormatY
    elif cueglFrame.eglColorFormat == cydriver.CUeglColorFormat_enum.CU_EGL_COLOR_FORMAT_YUV420_SEMIPLANAR_2020:
        eglFrame[0].eglColorFormat = cyruntime.cudaEglColorFormatYUV420SemiPlanar_2020
    elif cueglFrame.eglColorFormat == cydriver.CUeglColorFormat_enum.CU_EGL_COLOR_FORMAT_YVU420_SEMIPLANAR_2020:
        eglFrame[0].eglColorFormat = cyruntime.cudaEglColorFormatYVU420SemiPlanar_2020
    elif cueglFrame.eglColorFormat == cydriver.CUeglColorFormat_enum.CU_EGL_COLOR_FORMAT_YUV420_PLANAR_2020:
        eglFrame[0].eglColorFormat = cyruntime.cudaEglColorFormatYUV420Planar_2020
    elif cueglFrame.eglColorFormat == cydriver.CUeglColorFormat_enum.CU_EGL_COLOR_FORMAT_YVU420_PLANAR_2020:
        eglFrame[0].eglColorFormat = cyruntime.cudaEglColorFormatYVU420Planar_2020
    elif cueglFrame.eglColorFormat == cydriver.CUeglColorFormat_enum.CU_EGL_COLOR_FORMAT_YUV420_SEMIPLANAR_709:
        eglFrame[0].eglColorFormat = cyruntime.cudaEglColorFormatYUV420SemiPlanar_709
    elif cueglFrame.eglColorFormat == cydriver.CUeglColorFormat_enum.CU_EGL_COLOR_FORMAT_YVU420_SEMIPLANAR_709:
        eglFrame[0].eglColorFormat = cyruntime.cudaEglColorFormatYVU420SemiPlanar_709
    elif cueglFrame.eglColorFormat == cydriver.CUeglColorFormat_enum.CU_EGL_COLOR_FORMAT_YUV420_PLANAR_709:
        eglFrame[0].eglColorFormat = cyruntime.cudaEglColorFormatYUV420Planar_709
    elif cueglFrame.eglColorFormat == cydriver.CUeglColorFormat_enum.CU_EGL_COLOR_FORMAT_YVU420_PLANAR_709:
        eglFrame[0].eglColorFormat = cyruntime.cudaEglColorFormatYVU420Planar_709
    elif cueglFrame.eglColorFormat == cydriver.CUeglColorFormat_enum.CU_EGL_COLOR_FORMAT_Y10V10U10_420_SEMIPLANAR_709:
        eglFrame[0].eglColorFormat = cyruntime.cudaEglColorFormatY10V10U10_420SemiPlanar_709
    elif cueglFrame.eglColorFormat == cydriver.CUeglColorFormat_enum.CU_EGL_COLOR_FORMAT_Y10V10U10_420_SEMIPLANAR_2020:
        eglFrame[0].eglColorFormat = cyruntime.cudaEglColorFormatY10V10U10_420SemiPlanar_2020
    elif cueglFrame.eglColorFormat == cydriver.CUeglColorFormat_enum.CU_EGL_COLOR_FORMAT_Y10V10U10_422_SEMIPLANAR_2020:
        eglFrame[0].eglColorFormat = cyruntime.cudaEglColorFormatY10V10U10_422SemiPlanar_2020
    elif cueglFrame.eglColorFormat == cydriver.CUeglColorFormat_enum.CU_EGL_COLOR_FORMAT_Y10V10U10_422_SEMIPLANAR:
        eglFrame[0].eglColorFormat = cyruntime.cudaEglColorFormatY10V10U10_422SemiPlanar
    elif cueglFrame.eglColorFormat == cydriver.CUeglColorFormat_enum.CU_EGL_COLOR_FORMAT_Y10V10U10_422_SEMIPLANAR_709:
        eglFrame[0].eglColorFormat = cyruntime.cudaEglColorFormatY10V10U10_422SemiPlanar_709
    elif cueglFrame.eglColorFormat == cydriver.CUeglColorFormat_enum.CU_EGL_COLOR_FORMAT_Y_ER:
        eglFrame[0].eglColorFormat = cyruntime.cudaEglColorFormatY_ER
    elif cueglFrame.eglColorFormat == cydriver.CUeglColorFormat_enum.CU_EGL_COLOR_FORMAT_Y_709_ER:
        eglFrame[0].eglColorFormat = cyruntime.cudaEglColorFormatY_709_ER
    elif cueglFrame.eglColorFormat == cydriver.CUeglColorFormat_enum.CU_EGL_COLOR_FORMAT_Y10_ER:
        eglFrame[0].eglColorFormat = cyruntime.cudaEglColorFormatY10_ER
    elif cueglFrame.eglColorFormat == cydriver.CUeglColorFormat_enum.CU_EGL_COLOR_FORMAT_Y10_709_ER:
        eglFrame[0].eglColorFormat = cyruntime.cudaEglColorFormatY10_709_ER
    elif cueglFrame.eglColorFormat == cydriver.CUeglColorFormat_enum.CU_EGL_COLOR_FORMAT_Y12_ER:
        eglFrame[0].eglColorFormat = cyruntime.cudaEglColorFormatY12_ER
    elif cueglFrame.eglColorFormat == cydriver.CUeglColorFormat_enum.CU_EGL_COLOR_FORMAT_Y12_709_ER:
        eglFrame[0].eglColorFormat = cyruntime.cudaEglColorFormatY12_709_ER
    elif cueglFrame.eglColorFormat == cydriver.CUeglColorFormat_enum.CU_EGL_COLOR_FORMAT_YUVA:
        eglFrame[0].eglColorFormat = cyruntime.cudaEglColorFormatYUVA
    elif cueglFrame.eglColorFormat == cydriver.CUeglColorFormat_enum.CU_EGL_COLOR_FORMAT_YVYU:
        eglFrame[0].eglColorFormat = cyruntime.cudaEglColorFormatYVYU
    elif cueglFrame.eglColorFormat == cydriver.CUeglColorFormat_enum.CU_EGL_COLOR_FORMAT_VYUY:
        eglFrame[0].eglColorFormat = cyruntime.cudaEglColorFormatVYUY
    elif cueglFrame.eglColorFormat == cydriver.CUeglColorFormat_enum.CU_EGL_COLOR_FORMAT_Y10V10U10_420_SEMIPLANAR_ER:
        eglFrame[0].eglColorFormat = cyruntime.cudaEglColorFormatY10V10U10_420SemiPlanar_ER
    elif cueglFrame.eglColorFormat == cydriver.CUeglColorFormat_enum.CU_EGL_COLOR_FORMAT_Y10V10U10_420_SEMIPLANAR_709_ER:
        eglFrame[0].eglColorFormat = cyruntime.cudaEglColorFormatY10V10U10_420SemiPlanar_709_ER
    elif cueglFrame.eglColorFormat == cydriver.CUeglColorFormat_enum.CU_EGL_COLOR_FORMAT_Y10V10U10_444_SEMIPLANAR_ER:
        eglFrame[0].eglColorFormat = cyruntime.cudaEglColorFormatY10V10U10_444SemiPlanar_ER
    elif cueglFrame.eglColorFormat == cydriver.CUeglColorFormat_enum.CU_EGL_COLOR_FORMAT_Y10V10U10_444_SEMIPLANAR_709_ER:
        eglFrame[0].eglColorFormat = cyruntime.cudaEglColorFormatY10V10U10_444SemiPlanar_709_ER
    elif cueglFrame.eglColorFormat == cydriver.CUeglColorFormat_enum.CU_EGL_COLOR_FORMAT_Y12V12U12_420_SEMIPLANAR_ER:
        eglFrame[0].eglColorFormat = cyruntime.cudaEglColorFormatY12V12U12_420SemiPlanar_ER
    elif cueglFrame.eglColorFormat == cydriver.CUeglColorFormat_enum.CU_EGL_COLOR_FORMAT_Y12V12U12_420_SEMIPLANAR_709_ER:
        eglFrame[0].eglColorFormat = cyruntime.cudaEglColorFormatY12V12U12_420SemiPlanar_709_ER
    elif cueglFrame.eglColorFormat == cydriver.CUeglColorFormat_enum.CU_EGL_COLOR_FORMAT_Y12V12U12_444_SEMIPLANAR_ER:
        eglFrame[0].eglColorFormat = cyruntime.cudaEglColorFormatY12V12U12_444SemiPlanar_ER
    elif cueglFrame.eglColorFormat == cydriver.CUeglColorFormat_enum.CU_EGL_COLOR_FORMAT_Y12V12U12_444_SEMIPLANAR_709_ER:
        eglFrame[0].eglColorFormat = cyruntime.cudaEglColorFormatY12V12U12_444SemiPlanar_709_ER
    else:
        return cudaErrorInvalidValue
    if cueglFrame.frameType == cydriver.CUeglFrameType_enum.CU_EGL_FRAME_TYPE_ARRAY:
        eglFrame[0].frameType = cyruntime.cudaEglFrameTypeArray
    elif cueglFrame.frameType == cydriver.CUeglFrameType_enum.CU_EGL_FRAME_TYPE_PITCH:
        eglFrame[0].frameType = cyruntime.cudaEglFrameTypePitch
    else:
        return cudaErrorInvalidValue
