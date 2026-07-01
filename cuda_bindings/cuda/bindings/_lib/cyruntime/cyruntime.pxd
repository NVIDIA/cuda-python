# SPDX-FileCopyrightText: Copyright (c) 2021-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: LicenseRef-NVIDIA-SOFTWARE-LICENSE

cimport cuda.bindings._internal.driver as _cydriver

# These graphics API are the reimplemented version of what's supported by CUDA Runtime.
# Issue https://github.com/NVIDIA/cuda-python/issues/488 will remove them by letting us
# use call into the static library directly.
#
# This is an ABI breaking change which can only happen in a major version bump.

# This file is included from cuda/bindings/_bindings/cyruntime.pxd.in but kept in a
# separate file to keep it separated from the auto-generated code there.

# Prior to https://github.com/NVIDIA/cuda-python/pull/914, this was two
# independent modules (c.b._lib.cyruntime.cyruntime and
# c.b._lib.cyruntime.utils), but was merged into one.

cdef cudaError_t _cudaEGLStreamProducerPresentFrame(cudaEglStreamConnection* conn, cudaEglFrame eglframe, cudaStream_t* pStream) except ?cudaErrorCallRequiresNewerDriver nogil
cdef cudaError_t _cudaEGLStreamProducerReturnFrame(cudaEglStreamConnection* conn, cudaEglFrame* eglframe, cudaStream_t* pStream) except ?cudaErrorCallRequiresNewerDriver nogil
cdef cudaError_t _cudaGraphicsResourceGetMappedEglFrame(cudaEglFrame* eglFrame, cudaGraphicsResource_t resource, unsigned int index, unsigned int mipLevel) except ?cudaErrorCallRequiresNewerDriver nogil
cdef cudaError_t _cudaVDPAUSetVDPAUDevice(int device, VdpDevice vdpDevice, VdpGetProcAddress* vdpGetProcAddress) except ?cudaErrorCallRequiresNewerDriver nogil
cdef cudaError_t _cudaVDPAUGetDevice(int* device, VdpDevice vdpDevice, VdpGetProcAddress* vdpGetProcAddress) except ?cudaErrorCallRequiresNewerDriver nogil
cdef cudaError_t _cudaGraphicsVDPAURegisterVideoSurface(cudaGraphicsResource** resource, VdpVideoSurface vdpSurface, unsigned int flags) except ?cudaErrorCallRequiresNewerDriver nogil
cdef cudaError_t _cudaGraphicsVDPAURegisterOutputSurface(cudaGraphicsResource** resource, VdpOutputSurface vdpSurface, unsigned int flags) except ?cudaErrorCallRequiresNewerDriver nogil
cdef cudaError_t _cudaGLGetDevices(unsigned int* pCudaDeviceCount, int* pCudaDevices, unsigned int cudaDeviceCount, cudaGLDeviceList deviceList) except ?cudaErrorCallRequiresNewerDriver nogil
cdef cudaError_t _cudaGraphicsGLRegisterImage(cudaGraphicsResource** resource, GLuint image, GLenum target, unsigned int flags) except ?cudaErrorCallRequiresNewerDriver nogil
cdef cudaError_t _cudaGraphicsGLRegisterBuffer(cudaGraphicsResource** resource, GLuint buffer, unsigned int flags) except ?cudaErrorCallRequiresNewerDriver nogil
cdef cudaError_t _cudaGraphicsEGLRegisterImage(cudaGraphicsResource_t* pCudaResource, EGLImageKHR image, unsigned int flags) except ?cudaErrorCallRequiresNewerDriver nogil
cdef cudaError_t _cudaEGLStreamConsumerConnect(cudaEglStreamConnection* conn, EGLStreamKHR eglStream) except ?cudaErrorCallRequiresNewerDriver nogil
cdef cudaError_t _cudaEGLStreamConsumerConnectWithFlags(cudaEglStreamConnection* conn, EGLStreamKHR eglStream, unsigned int flags) except ?cudaErrorCallRequiresNewerDriver nogil
cdef cudaError_t _cudaEGLStreamConsumerDisconnect(cudaEglStreamConnection* conn) except ?cudaErrorCallRequiresNewerDriver nogil
cdef cudaError_t _cudaEGLStreamConsumerAcquireFrame(cudaEglStreamConnection* conn, cudaGraphicsResource_t* pCudaResource, cudaStream_t* pStream, unsigned int timeout) except ?cudaErrorCallRequiresNewerDriver nogil
cdef cudaError_t _cudaEGLStreamConsumerReleaseFrame(cudaEglStreamConnection* conn, cudaGraphicsResource_t pCudaResource, cudaStream_t* pStream) except ?cudaErrorCallRequiresNewerDriver nogil
cdef cudaError_t _cudaEGLStreamProducerConnect(cudaEglStreamConnection* conn, EGLStreamKHR eglStream, EGLint width, EGLint height) except ?cudaErrorCallRequiresNewerDriver nogil
cdef cudaError_t _cudaEGLStreamProducerDisconnect(cudaEglStreamConnection* conn) except ?cudaErrorCallRequiresNewerDriver nogil
cdef cudaError_t _cudaEventCreateFromEGLSync(cudaEvent_t* phEvent, EGLSyncKHR eglSync, unsigned int flags) except ?cudaErrorCallRequiresNewerDriver nogil

# utility functions

cdef cudaError_t getDriverEglFrame(_cydriver.CUeglFrame *cuEglFrame, cudaEglFrame eglFrame) except ?cudaErrorCallRequiresNewerDriver nogil
cdef cudaError_t getRuntimeEglFrame(cudaEglFrame *eglFrame, _cydriver.CUeglFrame cueglFrame) except ?cudaErrorCallRequiresNewerDriver nogil
