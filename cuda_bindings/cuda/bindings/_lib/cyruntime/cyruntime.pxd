# SPDX-FileCopyrightText: Copyright (c) 2021-2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: LicenseRef-NVIDIA-SOFTWARE-LICENSE

cimport cuda.bindings.cyruntime as cyruntime
cimport cuda.bindings._bindings.cydriver as _cydriver

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

cdef cudaError_t _cudaEGLStreamProducerPresentFrame(cyruntime.cudaEglStreamConnection* conn, cyruntime.cudaEglFrame eglframe, cudaStream_t* pStream) except ?cudaErrorCallRequiresNewerDriver nogil
cdef cudaError_t _cudaEGLStreamProducerReturnFrame(cyruntime.cudaEglStreamConnection* conn, cyruntime.cudaEglFrame* eglframe, cudaStream_t* pStream) except ?cudaErrorCallRequiresNewerDriver nogil
cdef cudaError_t _cudaGraphicsResourceGetMappedEglFrame(cyruntime.cudaEglFrame* eglFrame, cudaGraphicsResource_t resource, unsigned int index, unsigned int mipLevel) except ?cudaErrorCallRequiresNewerDriver nogil
cdef cudaError_t _cudaVDPAUSetVDPAUDevice(int device, cyruntime.VdpDevice vdpDevice, cyruntime.VdpGetProcAddress* vdpGetProcAddress) except ?cudaErrorCallRequiresNewerDriver nogil
cdef cudaError_t _cudaVDPAUGetDevice(int* device, cyruntime.VdpDevice vdpDevice, cyruntime.VdpGetProcAddress* vdpGetProcAddress) except ?cudaErrorCallRequiresNewerDriver nogil
cdef cudaError_t _cudaGraphicsVDPAURegisterVideoSurface(cudaGraphicsResource** resource, cyruntime.VdpVideoSurface vdpSurface, unsigned int flags) except ?cudaErrorCallRequiresNewerDriver nogil
cdef cudaError_t _cudaGraphicsVDPAURegisterOutputSurface(cudaGraphicsResource** resource, cyruntime.VdpOutputSurface vdpSurface, unsigned int flags) except ?cudaErrorCallRequiresNewerDriver nogil
cdef cudaError_t _cudaGLGetDevices(unsigned int* pCudaDeviceCount, int* pCudaDevices, unsigned int cudaDeviceCount, cyruntime.cudaGLDeviceList deviceList) except ?cudaErrorCallRequiresNewerDriver nogil
cdef cudaError_t _cudaGraphicsGLRegisterImage(cudaGraphicsResource** resource, cyruntime.GLuint image, cyruntime.GLenum target, unsigned int flags) except ?cudaErrorCallRequiresNewerDriver nogil
cdef cudaError_t _cudaGraphicsGLRegisterBuffer(cudaGraphicsResource** resource, cyruntime.GLuint buffer, unsigned int flags) except ?cudaErrorCallRequiresNewerDriver nogil
cdef cudaError_t _cudaGraphicsEGLRegisterImage(cudaGraphicsResource_t* pCudaResource, cyruntime.EGLImageKHR image, unsigned int flags) except ?cudaErrorCallRequiresNewerDriver nogil
cdef cudaError_t _cudaEGLStreamConsumerConnect(cyruntime.cudaEglStreamConnection* conn, cyruntime.EGLStreamKHR eglStream) except ?cudaErrorCallRequiresNewerDriver nogil
cdef cudaError_t _cudaEGLStreamConsumerConnectWithFlags(cyruntime.cudaEglStreamConnection* conn, cyruntime.EGLStreamKHR eglStream, unsigned int flags) except ?cudaErrorCallRequiresNewerDriver nogil
cdef cudaError_t _cudaEGLStreamConsumerDisconnect(cyruntime.cudaEglStreamConnection* conn) except ?cudaErrorCallRequiresNewerDriver nogil
cdef cudaError_t _cudaEGLStreamConsumerAcquireFrame(cyruntime.cudaEglStreamConnection* conn, cudaGraphicsResource_t* pCudaResource, cudaStream_t* pStream, unsigned int timeout) except ?cudaErrorCallRequiresNewerDriver nogil
cdef cudaError_t _cudaEGLStreamConsumerReleaseFrame(cyruntime.cudaEglStreamConnection* conn, cudaGraphicsResource_t pCudaResource, cudaStream_t* pStream) except ?cudaErrorCallRequiresNewerDriver nogil
cdef cudaError_t _cudaEGLStreamProducerConnect(cyruntime.cudaEglStreamConnection* conn, cyruntime.EGLStreamKHR eglStream, cyruntime.EGLint width, cyruntime.EGLint height) except ?cudaErrorCallRequiresNewerDriver nogil
cdef cudaError_t _cudaEGLStreamProducerDisconnect(cyruntime.cudaEglStreamConnection* conn) except ?cudaErrorCallRequiresNewerDriver nogil
cdef cudaError_t _cudaEventCreateFromEGLSync(cudaEvent_t* phEvent, cyruntime.EGLSyncKHR eglSync, unsigned int flags) except ?cudaErrorCallRequiresNewerDriver nogil

# utility functions

cdef cudaError_t getDriverEglFrame(_cydriver.CUeglFrame *cuEglFrame, cyruntime.cudaEglFrame eglFrame) except ?cudaErrorCallRequiresNewerDriver nogil
cdef cudaError_t getRuntimeEglFrame(cyruntime.cudaEglFrame *eglFrame, _cydriver.CUeglFrame cueglFrame) except ?cudaErrorCallRequiresNewerDriver nogil
