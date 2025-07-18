# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# SPDX-License-Identifier: LicenseRef-NVIDIA-SOFTWARE-LICENSE
#
# This code was automatically generated with version 12.9.0. Do not modify it directly.

from libc.time cimport time_t
from libcpp cimport bool as cpp_bool
from posix.types cimport off_t

cimport cuda.bindings.cydriver
from cuda.bindings.cydriver cimport CUresult


###############################################################################
# Types (structs, enums, ...)
###############################################################################

# TODO: switch to "from libc.time cimport timespec" once we can use recent
# Cython to build
cdef extern from "<time.h>":
    cdef struct timespec:
        time_t tv_sec
        long   tv_nsec
cdef extern from "<sys/socket.h>":
    cdef struct sockaddr:
        unsigned short sa_family
        char sa_data[14]
    ctypedef sockaddr sockaddr_t


cdef extern from '<cufile.h>':
    # enums
    ctypedef enum CUfileOpError:
        CU_FILE_SUCCESS
        CU_FILE_DRIVER_NOT_INITIALIZED
        CU_FILE_DRIVER_INVALID_PROPS
        CU_FILE_DRIVER_UNSUPPORTED_LIMIT
        CU_FILE_DRIVER_VERSION_MISMATCH
        CU_FILE_DRIVER_VERSION_READ_ERROR
        CU_FILE_DRIVER_CLOSING
        CU_FILE_PLATFORM_NOT_SUPPORTED
        CU_FILE_IO_NOT_SUPPORTED
        CU_FILE_DEVICE_NOT_SUPPORTED
        CU_FILE_NVFS_DRIVER_ERROR
        CU_FILE_CUDA_DRIVER_ERROR
        CU_FILE_CUDA_POINTER_INVALID
        CU_FILE_CUDA_MEMORY_TYPE_INVALID
        CU_FILE_CUDA_POINTER_RANGE_ERROR
        CU_FILE_CUDA_CONTEXT_MISMATCH
        CU_FILE_INVALID_MAPPING_SIZE
        CU_FILE_INVALID_MAPPING_RANGE
        CU_FILE_INVALID_FILE_TYPE
        CU_FILE_INVALID_FILE_OPEN_FLAG
        CU_FILE_DIO_NOT_SET
        CU_FILE_INVALID_VALUE
        CU_FILE_MEMORY_ALREADY_REGISTERED
        CU_FILE_MEMORY_NOT_REGISTERED
        CU_FILE_PERMISSION_DENIED
        CU_FILE_DRIVER_ALREADY_OPEN
        CU_FILE_HANDLE_NOT_REGISTERED
        CU_FILE_HANDLE_ALREADY_REGISTERED
        CU_FILE_DEVICE_NOT_FOUND
        CU_FILE_INTERNAL_ERROR
        CU_FILE_GETNEWFD_FAILED
        CU_FILE_NVFS_SETUP_ERROR
        CU_FILE_IO_DISABLED
        CU_FILE_BATCH_SUBMIT_FAILED
        CU_FILE_GPU_MEMORY_PINNING_FAILED
        CU_FILE_BATCH_FULL
        CU_FILE_ASYNC_NOT_SUPPORTED
        CU_FILE_IO_MAX_ERROR

    ctypedef enum CUfileDriverStatusFlags_t:
        CU_FILE_LUSTRE_SUPPORTED
        CU_FILE_WEKAFS_SUPPORTED
        CU_FILE_NFS_SUPPORTED
        CU_FILE_GPFS_SUPPORTED
        CU_FILE_NVME_SUPPORTED
        CU_FILE_NVMEOF_SUPPORTED
        CU_FILE_SCSI_SUPPORTED
        CU_FILE_SCALEFLUX_CSD_SUPPORTED
        CU_FILE_NVMESH_SUPPORTED
        CU_FILE_BEEGFS_SUPPORTED
        CU_FILE_NVME_P2P_SUPPORTED
        CU_FILE_SCATEFS_SUPPORTED

    ctypedef enum CUfileDriverControlFlags_t:
        CU_FILE_USE_POLL_MODE
        CU_FILE_ALLOW_COMPAT_MODE

    ctypedef enum CUfileFeatureFlags_t:
        CU_FILE_DYN_ROUTING_SUPPORTED
        CU_FILE_BATCH_IO_SUPPORTED
        CU_FILE_STREAMS_SUPPORTED
        CU_FILE_PARALLEL_IO_SUPPORTED

    ctypedef enum CUfileFileHandleType:
        CU_FILE_HANDLE_TYPE_OPAQUE_FD
        CU_FILE_HANDLE_TYPE_OPAQUE_WIN32
        CU_FILE_HANDLE_TYPE_USERSPACE_FS

    ctypedef enum CUfileOpcode_t:
        CUFILE_READ
        CUFILE_WRITE

    ctypedef enum CUfileStatus_t:
        CUFILE_WAITING
        CUFILE_PENDING
        CUFILE_INVALID
        CUFILE_CANCELED
        CUFILE_COMPLETE
        CUFILE_TIMEOUT
        CUFILE_FAILED

    ctypedef enum CUfileBatchMode_t:
        CUFILE_BATCH

    ctypedef enum CUFileSizeTConfigParameter_t:
        CUFILE_PARAM_PROFILE_STATS
        CUFILE_PARAM_EXECUTION_MAX_IO_QUEUE_DEPTH
        CUFILE_PARAM_EXECUTION_MAX_IO_THREADS
        CUFILE_PARAM_EXECUTION_MIN_IO_THRESHOLD_SIZE_KB
        CUFILE_PARAM_EXECUTION_MAX_REQUEST_PARALLELISM
        CUFILE_PARAM_PROPERTIES_MAX_DIRECT_IO_SIZE_KB
        CUFILE_PARAM_PROPERTIES_MAX_DEVICE_CACHE_SIZE_KB
        CUFILE_PARAM_PROPERTIES_PER_BUFFER_CACHE_SIZE_KB
        CUFILE_PARAM_PROPERTIES_MAX_DEVICE_PINNED_MEM_SIZE_KB
        CUFILE_PARAM_PROPERTIES_IO_BATCHSIZE
        CUFILE_PARAM_POLLTHRESHOLD_SIZE_KB
        CUFILE_PARAM_PROPERTIES_BATCH_IO_TIMEOUT_MS

    ctypedef enum CUFileBoolConfigParameter_t:
        CUFILE_PARAM_PROPERTIES_USE_POLL_MODE
        CUFILE_PARAM_PROPERTIES_ALLOW_COMPAT_MODE
        CUFILE_PARAM_FORCE_COMPAT_MODE
        CUFILE_PARAM_FS_MISC_API_CHECK_AGGRESSIVE
        CUFILE_PARAM_EXECUTION_PARALLEL_IO
        CUFILE_PARAM_PROFILE_NVTX
        CUFILE_PARAM_PROPERTIES_ALLOW_SYSTEM_MEMORY
        CUFILE_PARAM_USE_PCIP2PDMA
        CUFILE_PARAM_PREFER_IO_URING
        CUFILE_PARAM_FORCE_ODIRECT_MODE
        CUFILE_PARAM_SKIP_TOPOLOGY_DETECTION
        CUFILE_PARAM_STREAM_MEMOPS_BYPASS

    ctypedef enum CUFileStringConfigParameter_t:
        CUFILE_PARAM_LOGGING_LEVEL
        CUFILE_PARAM_ENV_LOGFILE_PATH
        CUFILE_PARAM_LOG_DIR

    # types
    ctypedef void* CUfileHandle_t 'CUfileHandle_t'
    ctypedef void* CUfileBatchHandle_t 'CUfileBatchHandle_t'
    ctypedef struct CUfileError_t 'CUfileError_t':
        CUfileOpError err
        CUresult cu_err
    cdef struct _anon_pod0 '_anon_pod0':
        unsigned int major_version
        unsigned int minor_version
        size_t poll_thresh_size
        size_t max_direct_io_size
        unsigned int dstatusflags
        unsigned int dcontrolflags
    ctypedef struct cufileRDMAInfo_t 'cufileRDMAInfo_t':
        int version
        int desc_len
        char* desc_str
    ctypedef struct CUfileFSOps_t 'CUfileFSOps_t':
        char* (*fs_type)(void*)
        int (*getRDMADeviceList)(void*, sockaddr_t**)
        int (*getRDMADevicePriority)(void*, char*, size_t, loff_t, sockaddr_t*)
        ssize_t (*read)(void*, char*, size_t, loff_t, cufileRDMAInfo_t*)
        ssize_t (*write)(void*, const char*, size_t, loff_t, cufileRDMAInfo_t*)
    cdef union _anon_pod1 '_anon_pod1':
        int fd
        void* handle
    cdef struct _anon_pod3 '_anon_pod3':
        void* devPtr_base
        off_t file_offset
        off_t devPtr_offset
        size_t size
    ctypedef struct CUfileIOEvents_t 'CUfileIOEvents_t':
        void* cookie
        CUfileStatus_t status
        size_t ret
    ctypedef struct CUfileDrvProps_t 'CUfileDrvProps_t':
        _anon_pod0 nvfs
        unsigned int fflags
        unsigned int max_device_cache_size
        unsigned int per_buffer_cache_size
        unsigned int max_device_pinned_mem_size
        unsigned int max_batch_io_size
        unsigned int max_batch_io_timeout_msecs
    ctypedef struct CUfileDescr_t 'CUfileDescr_t':
        CUfileFileHandleType type
        _anon_pod1 handle
        CUfileFSOps_t* fs_ops
    cdef union _anon_pod2 '_anon_pod2':
        _anon_pod3 batch
    ctypedef struct CUfileIOParams_t 'CUfileIOParams_t':
        CUfileBatchMode_t mode
        _anon_pod2 u
        CUfileHandle_t fh
        CUfileOpcode_t opcode
        void* cookie


cdef extern from *:
    """
    // This is the missing piece we need to supply to help Cython & C++ compilers.
    inline bool operator==(const CUfileError_t& lhs, const CUfileError_t& rhs) {
        return (lhs.err == rhs.err) && (lhs.cu_err == rhs.cu_err);
    }
    static CUfileError_t CUFILE_LOADING_ERROR{(CUfileOpError)-1, (CUresult)-1};
    """
    const CUfileError_t CUFILE_LOADING_ERROR
    ctypedef void* CUstream "CUstream"

    const char* cufileop_status_error(CUfileOpError)


###############################################################################
# Functions
###############################################################################

cdef CUfileError_t cuFileHandleRegister(CUfileHandle_t* fh, CUfileDescr_t* descr) except?<CUfileError_t>CUFILE_LOADING_ERROR nogil
cdef void cuFileHandleDeregister(CUfileHandle_t fh) except* nogil
cdef CUfileError_t cuFileBufRegister(const void* bufPtr_base, size_t length, int flags) except?<CUfileError_t>CUFILE_LOADING_ERROR nogil
cdef CUfileError_t cuFileBufDeregister(const void* bufPtr_base) except?<CUfileError_t>CUFILE_LOADING_ERROR nogil
cdef ssize_t cuFileRead(CUfileHandle_t fh, void* bufPtr_base, size_t size, off_t file_offset, off_t bufPtr_offset) except* nogil
cdef ssize_t cuFileWrite(CUfileHandle_t fh, const void* bufPtr_base, size_t size, off_t file_offset, off_t bufPtr_offset) except* nogil
cdef CUfileError_t cuFileDriverOpen() except?<CUfileError_t>CUFILE_LOADING_ERROR nogil
cdef CUfileError_t cuFileDriverClose_v2() except?<CUfileError_t>CUFILE_LOADING_ERROR nogil
cdef long cuFileUseCount() except* nogil
cdef CUfileError_t cuFileDriverGetProperties(CUfileDrvProps_t* props) except?<CUfileError_t>CUFILE_LOADING_ERROR nogil
cdef CUfileError_t cuFileDriverSetPollMode(cpp_bool poll, size_t poll_threshold_size) except?<CUfileError_t>CUFILE_LOADING_ERROR nogil
cdef CUfileError_t cuFileDriverSetMaxDirectIOSize(size_t max_direct_io_size) except?<CUfileError_t>CUFILE_LOADING_ERROR nogil
cdef CUfileError_t cuFileDriverSetMaxCacheSize(size_t max_cache_size) except?<CUfileError_t>CUFILE_LOADING_ERROR nogil
cdef CUfileError_t cuFileDriverSetMaxPinnedMemSize(size_t max_pinned_size) except?<CUfileError_t>CUFILE_LOADING_ERROR nogil
cdef CUfileError_t cuFileBatchIOSetUp(CUfileBatchHandle_t* batch_idp, unsigned nr) except?<CUfileError_t>CUFILE_LOADING_ERROR nogil
cdef CUfileError_t cuFileBatchIOSubmit(CUfileBatchHandle_t batch_idp, unsigned nr, CUfileIOParams_t* iocbp, unsigned int flags) except?<CUfileError_t>CUFILE_LOADING_ERROR nogil
cdef CUfileError_t cuFileBatchIOGetStatus(CUfileBatchHandle_t batch_idp, unsigned min_nr, unsigned* nr, CUfileIOEvents_t* iocbp, timespec* timeout) except?<CUfileError_t>CUFILE_LOADING_ERROR nogil
cdef CUfileError_t cuFileBatchIOCancel(CUfileBatchHandle_t batch_idp) except?<CUfileError_t>CUFILE_LOADING_ERROR nogil
cdef void cuFileBatchIODestroy(CUfileBatchHandle_t batch_idp) except* nogil
cdef CUfileError_t cuFileReadAsync(CUfileHandle_t fh, void* bufPtr_base, size_t* size_p, off_t* file_offset_p, off_t* bufPtr_offset_p, ssize_t* bytes_read_p, CUstream stream) except?<CUfileError_t>CUFILE_LOADING_ERROR nogil
cdef CUfileError_t cuFileWriteAsync(CUfileHandle_t fh, void* bufPtr_base, size_t* size_p, off_t* file_offset_p, off_t* bufPtr_offset_p, ssize_t* bytes_written_p, CUstream stream) except?<CUfileError_t>CUFILE_LOADING_ERROR nogil
cdef CUfileError_t cuFileStreamRegister(CUstream stream, unsigned flags) except?<CUfileError_t>CUFILE_LOADING_ERROR nogil
cdef CUfileError_t cuFileStreamDeregister(CUstream stream) except?<CUfileError_t>CUFILE_LOADING_ERROR nogil
cdef CUfileError_t cuFileGetVersion(int* version) except?<CUfileError_t>CUFILE_LOADING_ERROR nogil
cdef CUfileError_t cuFileGetParameterSizeT(CUFileSizeTConfigParameter_t param, size_t* value) except?<CUfileError_t>CUFILE_LOADING_ERROR nogil
cdef CUfileError_t cuFileGetParameterBool(CUFileBoolConfigParameter_t param, cpp_bool* value) except?<CUfileError_t>CUFILE_LOADING_ERROR nogil
cdef CUfileError_t cuFileGetParameterString(CUFileStringConfigParameter_t param, char* desc_str, int len) except?<CUfileError_t>CUFILE_LOADING_ERROR nogil
cdef CUfileError_t cuFileSetParameterSizeT(CUFileSizeTConfigParameter_t param, size_t value) except?<CUfileError_t>CUFILE_LOADING_ERROR nogil
cdef CUfileError_t cuFileSetParameterBool(CUFileBoolConfigParameter_t param, cpp_bool value) except?<CUfileError_t>CUFILE_LOADING_ERROR nogil
cdef CUfileError_t cuFileSetParameterString(CUFileStringConfigParameter_t param, const char* desc_str) except?<CUfileError_t>CUFILE_LOADING_ERROR nogil
