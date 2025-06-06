# Copyright (c) 2025, NVIDIA CORPORATION & AFFILIATES. ALL RIGHTS RESERVED.
#
# SPDX-License-Identifier: LicenseRef-NVIDIA-SOFTWARE-LICENSE
#
# This code was automatically generated with version 12.9.0. Do not modify it directly.

cimport cython  # NOQA

from ._internal.utils cimport (get_buffer_pointer, get_nested_resource_ptr,
                               nested_resource)

from enum import IntEnum as _IntEnum
import cython

###############################################################################
# Enum
###############################################################################

class OpError(_IntEnum):
    """See `CUfileOpError`."""
    CU_FILE_SUCCESS = CU_FILE_SUCCESS
    CU_FILE_DRIVER_NOT_INITIALIZED = CU_FILE_DRIVER_NOT_INITIALIZED
    CU_FILE_DRIVER_INVALID_PROPS = CU_FILE_DRIVER_INVALID_PROPS
    CU_FILE_DRIVER_UNSUPPORTED_LIMIT = CU_FILE_DRIVER_UNSUPPORTED_LIMIT
    CU_FILE_DRIVER_VERSION_MISMATCH = CU_FILE_DRIVER_VERSION_MISMATCH
    CU_FILE_DRIVER_VERSION_READ_ERROR = CU_FILE_DRIVER_VERSION_READ_ERROR
    CU_FILE_DRIVER_CLOSING = CU_FILE_DRIVER_CLOSING
    CU_FILE_PLATFORM_NOT_SUPPORTED = CU_FILE_PLATFORM_NOT_SUPPORTED
    CU_FILE_IO_NOT_SUPPORTED = CU_FILE_IO_NOT_SUPPORTED
    CU_FILE_DEVICE_NOT_SUPPORTED = CU_FILE_DEVICE_NOT_SUPPORTED
    CU_FILE_NVFS_DRIVER_ERROR = CU_FILE_NVFS_DRIVER_ERROR
    CU_FILE_CUDA_DRIVER_ERROR = CU_FILE_CUDA_DRIVER_ERROR
    CU_FILE_CUDA_POINTER_INVALID = CU_FILE_CUDA_POINTER_INVALID
    CU_FILE_CUDA_MEMORY_TYPE_INVALID = CU_FILE_CUDA_MEMORY_TYPE_INVALID
    CU_FILE_CUDA_POINTER_RANGE_ERROR = CU_FILE_CUDA_POINTER_RANGE_ERROR
    CU_FILE_CUDA_CONTEXT_MISMATCH = CU_FILE_CUDA_CONTEXT_MISMATCH
    CU_FILE_INVALID_MAPPING_SIZE = CU_FILE_INVALID_MAPPING_SIZE
    CU_FILE_INVALID_MAPPING_RANGE = CU_FILE_INVALID_MAPPING_RANGE
    CU_FILE_INVALID_FILE_TYPE = CU_FILE_INVALID_FILE_TYPE
    CU_FILE_INVALID_FILE_OPEN_FLAG = CU_FILE_INVALID_FILE_OPEN_FLAG
    CU_FILE_DIO_NOT_SET = CU_FILE_DIO_NOT_SET
    CU_FILE_INVALID_VALUE = CU_FILE_INVALID_VALUE
    CU_FILE_MEMORY_ALREADY_REGISTERED = CU_FILE_MEMORY_ALREADY_REGISTERED
    CU_FILE_MEMORY_NOT_REGISTERED = CU_FILE_MEMORY_NOT_REGISTERED
    CU_FILE_PERMISSION_DENIED = CU_FILE_PERMISSION_DENIED
    CU_FILE_DRIVER_ALREADY_OPEN = CU_FILE_DRIVER_ALREADY_OPEN
    CU_FILE_HANDLE_NOT_REGISTERED = CU_FILE_HANDLE_NOT_REGISTERED
    CU_FILE_HANDLE_ALREADY_REGISTERED = CU_FILE_HANDLE_ALREADY_REGISTERED
    CU_FILE_DEVICE_NOT_FOUND = CU_FILE_DEVICE_NOT_FOUND
    CU_FILE_INTERNAL_ERROR = CU_FILE_INTERNAL_ERROR
    CU_FILE_GETNEWFD_FAILED = CU_FILE_GETNEWFD_FAILED
    CU_FILE_NVFS_SETUP_ERROR = CU_FILE_NVFS_SETUP_ERROR
    CU_FILE_IO_DISABLED = CU_FILE_IO_DISABLED
    CU_FILE_BATCH_SUBMIT_FAILED = CU_FILE_BATCH_SUBMIT_FAILED
    CU_FILE_GPU_MEMORY_PINNING_FAILED = CU_FILE_GPU_MEMORY_PINNING_FAILED
    CU_FILE_BATCH_FULL = CU_FILE_BATCH_FULL
    CU_FILE_ASYNC_NOT_SUPPORTED = CU_FILE_ASYNC_NOT_SUPPORTED
    CU_FILE_IO_MAX_ERROR = CU_FILE_IO_MAX_ERROR

class DriverStatusFlags(_IntEnum):
    """See `CUfileDriverStatusFlags_t`."""
    CU_FILE_LUSTRE_SUPPORTED = CU_FILE_LUSTRE_SUPPORTED
    CU_FILE_WEKAFS_SUPPORTED = CU_FILE_WEKAFS_SUPPORTED
    CU_FILE_NFS_SUPPORTED = CU_FILE_NFS_SUPPORTED
    CU_FILE_GPFS_SUPPORTED = CU_FILE_GPFS_SUPPORTED
    CU_FILE_NVME_SUPPORTED = CU_FILE_NVME_SUPPORTED
    CU_FILE_NVMEOF_SUPPORTED = CU_FILE_NVMEOF_SUPPORTED
    CU_FILE_SCSI_SUPPORTED = CU_FILE_SCSI_SUPPORTED
    CU_FILE_SCALEFLUX_CSD_SUPPORTED = CU_FILE_SCALEFLUX_CSD_SUPPORTED
    CU_FILE_NVMESH_SUPPORTED = CU_FILE_NVMESH_SUPPORTED
    CU_FILE_BEEGFS_SUPPORTED = CU_FILE_BEEGFS_SUPPORTED
    CU_FILE_NVME_P2P_SUPPORTED = CU_FILE_NVME_P2P_SUPPORTED
    CU_FILE_SCATEFS_SUPPORTED = CU_FILE_SCATEFS_SUPPORTED

class DriverControlFlags(_IntEnum):
    """See `CUfileDriverControlFlags_t`."""
    CU_FILE_USE_POLL_MODE = CU_FILE_USE_POLL_MODE
    CU_FILE_ALLOW_COMPAT_MODE = CU_FILE_ALLOW_COMPAT_MODE

class FeatureFlags(_IntEnum):
    """See `CUfileFeatureFlags_t`."""
    CU_FILE_DYN_ROUTING_SUPPORTED = CU_FILE_DYN_ROUTING_SUPPORTED
    CU_FILE_BATCH_IO_SUPPORTED = CU_FILE_BATCH_IO_SUPPORTED
    CU_FILE_STREAMS_SUPPORTED = CU_FILE_STREAMS_SUPPORTED
    CU_FILE_PARALLEL_IO_SUPPORTED = CU_FILE_PARALLEL_IO_SUPPORTED

class FileHandleType(_IntEnum):
    """See `CUfileFileHandleType_t`."""
    CU_OPAQUE_FD = CU_FILE_HANDLE_TYPE_OPAQUE_FD
    CU_OPAQUE_WIN32 = CU_FILE_HANDLE_TYPE_OPAQUE_WIN32
    CU_USERSPACE_FS = CU_FILE_HANDLE_TYPE_USERSPACE_FS

class Opcode(_IntEnum):
    """See `CUfileOpcode_t`."""
    READ = CUFILE_READ
    WRITE = CUFILE_WRITE

class Status(_IntEnum):
    """See `CUfileStatus_t`."""
    WAITING = CUFILE_WAITING
    PENDING = CUFILE_PENDING
    INVALID = CUFILE_INVALID
    CANCELED = CUFILE_CANCELED
    COMPLETE = CUFILE_COMPLETE
    TIMEOUT = CUFILE_TIMEOUT
    FAILED = CUFILE_FAILED

class BatchMode(_IntEnum):
    """See `CUfileBatchMode_t`."""
    BATCH = CUFILE_BATCH

class SizeTConfigParameter(_IntEnum):
    """See `CUFileSizeTConfigParameter_t`."""
    PARAM_PROFILE_STATS = CUFILE_PARAM_PROFILE_STATS
    PARAM_EXECUTION_MAX_IO_QUEUE_DEPTH = CUFILE_PARAM_EXECUTION_MAX_IO_QUEUE_DEPTH
    PARAM_EXECUTION_MAX_IO_THREADS = CUFILE_PARAM_EXECUTION_MAX_IO_THREADS
    PARAM_EXECUTION_MIN_IO_THRESHOLD_SIZE_KB = CUFILE_PARAM_EXECUTION_MIN_IO_THRESHOLD_SIZE_KB
    PARAM_EXECUTION_MAX_REQUEST_PARALLELISM = CUFILE_PARAM_EXECUTION_MAX_REQUEST_PARALLELISM
    PARAM_PROPERTIES_MAX_DIRECT_IO_SIZE_KB = CUFILE_PARAM_PROPERTIES_MAX_DIRECT_IO_SIZE_KB
    PARAM_PROPERTIES_MAX_DEVICE_CACHE_SIZE_KB = CUFILE_PARAM_PROPERTIES_MAX_DEVICE_CACHE_SIZE_KB
    PARAM_PROPERTIES_PER_BUFFER_CACHE_SIZE_KB = CUFILE_PARAM_PROPERTIES_PER_BUFFER_CACHE_SIZE_KB
    PARAM_PROPERTIES_MAX_DEVICE_PINNED_MEM_SIZE_KB = CUFILE_PARAM_PROPERTIES_MAX_DEVICE_PINNED_MEM_SIZE_KB
    PARAM_PROPERTIES_IO_BATCHSIZE = CUFILE_PARAM_PROPERTIES_IO_BATCHSIZE
    PARAM_POLLTHRESHOLD_SIZE_KB = CUFILE_PARAM_POLLTHRESHOLD_SIZE_KB
    PARAM_PROPERTIES_BATCH_IO_TIMEOUT_MS = CUFILE_PARAM_PROPERTIES_BATCH_IO_TIMEOUT_MS

class BoolConfigParameter(_IntEnum):
    """See `CUFileBoolConfigParameter_t`."""
    PARAM_PROPERTIES_USE_POLL_MODE = CUFILE_PARAM_PROPERTIES_USE_POLL_MODE
    PARAM_PROPERTIES_ALLOW_COMPAT_MODE = CUFILE_PARAM_PROPERTIES_ALLOW_COMPAT_MODE
    PARAM_FORCE_COMPAT_MODE = CUFILE_PARAM_FORCE_COMPAT_MODE
    PARAM_FS_MISC_API_CHECK_AGGRESSIVE = CUFILE_PARAM_FS_MISC_API_CHECK_AGGRESSIVE
    PARAM_EXECUTION_PARALLEL_IO = CUFILE_PARAM_EXECUTION_PARALLEL_IO
    PARAM_PROFILE_NVTX = CUFILE_PARAM_PROFILE_NVTX
    PARAM_PROPERTIES_ALLOW_SYSTEM_MEMORY = CUFILE_PARAM_PROPERTIES_ALLOW_SYSTEM_MEMORY
    PARAM_USE_PCIP2PDMA = CUFILE_PARAM_USE_PCIP2PDMA
    PARAM_PREFER_IO_URING = CUFILE_PARAM_PREFER_IO_URING
    PARAM_FORCE_ODIRECT_MODE = CUFILE_PARAM_FORCE_ODIRECT_MODE
    PARAM_SKIP_TOPOLOGY_DETECTION = CUFILE_PARAM_SKIP_TOPOLOGY_DETECTION
    PARAM_STREAM_MEMOPS_BYPASS = CUFILE_PARAM_STREAM_MEMOPS_BYPASS

class StringConfigParameter(_IntEnum):
    """See `CUFileStringConfigParameter_t`."""
    PARAM_LOGGING_LEVEL = CUFILE_PARAM_LOGGING_LEVEL
    PARAM_ENV_LOGFILE_PATH = CUFILE_PARAM_ENV_LOGFILE_PATH
    PARAM_LOG_DIR = CUFILE_PARAM_LOG_DIR


###############################################################################
# Error handling
###############################################################################

class cuFileError(Exception):

    def __init__(self, status, cu_err):
        self.status = status
        self.cuda_error = cu_err
        s = Result(status)
        cdef str err = f"{s.name} ({s.value}); CUDA status: {cu_err}"
        super(cuFileError, self).__init__(err)

    def __reduce__(self):
        return (type(self), (self.status, self.cuda_error))


@cython.profile(False)
cdef int check_status(CUfileError_t status) except 1 nogil:
    if status.err != 0 or status.cu_err != 0:
        with gil:
            raise cuFileError(status.err, status.cu_err)
    return 0


###############################################################################
# Wrapper functions
###############################################################################


cpdef handle_register(intptr_t fh, intptr_t descr):
    """cuFileHandleRegister is required, and performs extra checking that is memoized to provide increased performance on later cuFile operations.

    Args:
        fh (intptr_t): ``CUfileHandle_t`` opaque file handle for IO operations.
        descr (intptr_t): ``CUfileDescr_t`` file descriptor (OS agnostic).

    .. seealso:: `cuFileHandleRegister`
    """
    with nogil:
        status = cuFileHandleRegister(<Handle*>fh, <CUfileDescr_t*>descr)
    check_status(status)


cpdef void handle_deregister(intptr_t fh) except*:
    """releases a registered filehandle from cuFile.

    Args:
        fh (intptr_t): ``CUfileHandle_t`` file handle.

    .. seealso:: `cuFileHandleDeregister`
    """
    cuFileHandleDeregister(<Handle>fh)


cpdef buf_register(intptr_t buf_ptr_base, size_t length, int flags):
    """register an existing cudaMalloced memory with cuFile to pin for GPUDirect Storage access or register host allocated memory with cuFile.

    Args:
        buf_ptr_base (intptr_t): buffer pointer allocated.
        length (size_t): size of memory region from the above specified bufPtr.
        flags (int): CU_FILE_RDMA_REGISTER.

    .. seealso:: `cuFileBufRegister`
    """
    with nogil:
        status = cuFileBufRegister(<const void*>buf_ptr_base, length, flags)
    check_status(status)


cpdef buf_deregister(intptr_t buf_ptr_base):
    """deregister an already registered device or host memory from cuFile.

    Args:
        buf_ptr_base (intptr_t): buffer pointer to deregister.

    .. seealso:: `cuFileBufDeregister`
    """
    with nogil:
        status = cuFileBufDeregister(<const void*>buf_ptr_base)
    check_status(status)


cpdef read(intptr_t fh, intptr_t buf_ptr_base, size_t size, off_t file_offset, off_t buf_ptr_offset):
    """read data from a registered file handle to a specified device or host memory.

    Args:
        fh (intptr_t): ``CUfileHandle_t`` opaque file handle.
        buf_ptr_base (intptr_t): base address of buffer in device or host memory.
        size (size_t): size bytes to read.
        file_offset (off_t): file-offset from begining of the file.
        buf_ptr_offset (off_t): offset relative to the buf_ptr_base pointer to read into.

    .. seealso:: `cuFileRead`
    """
    with nogil:
        status = cuFileRead(<Handle>fh, <void*>buf_ptr_base, size, file_offset, buf_ptr_offset)
    check_status(status)


cpdef write(intptr_t fh, intptr_t buf_ptr_base, size_t size, off_t file_offset, off_t buf_ptr_offset):
    """write data from a specified device or host memory to a registered file handle.

    Args:
        fh (intptr_t): ``CUfileHandle_t`` opaque file handle.
        buf_ptr_base (intptr_t): base address of buffer in device or host memory.
        size (size_t): size bytes to write.
        file_offset (off_t): file-offset from begining of the file.
        buf_ptr_offset (off_t): offset relative to the buf_ptr_base pointer to write from.

    .. seealso:: `cuFileWrite`
    """
    with nogil:
        status = cuFileWrite(<Handle>fh, <const void*>buf_ptr_base, size, file_offset, buf_ptr_offset)
    check_status(status)


cpdef driver_open():
    """Initialize the cuFile library and open the nvidia-fs driver.

    .. seealso:: `cuFileDriverOpen`
    """
    with nogil:
        status = cuFileDriverOpen()
    check_status(status)


cpdef use_count():
    """returns use count of cufile drivers at that moment by the process.

    .. seealso:: `cuFileUseCount`
    """
    with nogil:
        status = cuFileUseCount()
    check_status(status)


cpdef driver_get_properties(intptr_t props):
    """Gets the Driver session properties.

    Args:
        props (intptr_t): to set.

    .. seealso:: `cuFileDriverGetProperties`
    """
    with nogil:
        status = cuFileDriverGetProperties(<CUfileDrvProps_t*>props)
    check_status(status)


cpdef driver_set_poll_mode(bool poll, size_t poll_threshold_size):
    """Sets whether the Read/Write APIs use polling to do IO operations.

    Args:
        poll (bool): boolean to indicate whether to use poll mode or not.
        poll_threshold_size (size_t): max IO size to use for POLLING mode in KB.

    .. seealso:: `cuFileDriverSetPollMode`
    """
    with nogil:
        status = cuFileDriverSetPollMode(poll, poll_threshold_size)
    check_status(status)


cpdef driver_set_max_direct_io_size(size_t max_direct_io_size):
    """Control parameter to set max IO size(KB) used by the library to talk to nvidia-fs driver.

    Args:
        max_direct_io_size (size_t): maximum allowed direct io size in KB.

    .. seealso:: `cuFileDriverSetMaxDirectIOSize`
    """
    with nogil:
        status = cuFileDriverSetMaxDirectIOSize(max_direct_io_size)
    check_status(status)


cpdef driver_set_max_cache_size(size_t max_cache_size):
    """Control parameter to set maximum GPU memory reserved per device by the library for internal buffering.

    Args:
        max_cache_size (size_t): The maximum GPU buffer space per device used for internal use in KB.

    .. seealso:: `cuFileDriverSetMaxCacheSize`
    """
    with nogil:
        status = cuFileDriverSetMaxCacheSize(max_cache_size)
    check_status(status)


cpdef driver_set_max_pinned_mem_size(size_t max_pinned_size):
    """Sets maximum buffer space that is pinned in KB for use by ``cuFileBufRegister``.

    Args:
        max_pinned_size (size_t): maximum buffer space that is pinned in KB.

    .. seealso:: `cuFileDriverSetMaxPinnedMemSize`
    """
    with nogil:
        status = cuFileDriverSetMaxPinnedMemSize(max_pinned_size)
    check_status(status)


cpdef batch_io_set_up(intptr_t batch_idp, unsigned nr):
    with nogil:
        status = cuFileBatchIOSetUp(<BatchHandle*>batch_idp, nr)
    check_status(status)


cpdef batch_io_submit(intptr_t batch_idp, unsigned nr, intptr_t iocbp, unsigned int flags):
    with nogil:
        status = cuFileBatchIOSubmit(<BatchHandle>batch_idp, nr, <CUfileIOParams_t*>iocbp, flags)
    check_status(status)


cpdef batch_io_get_status(intptr_t batch_idp, unsigned min_nr, intptr_t nr, intptr_t iocbp, intptr_t timeout):
    with nogil:
        status = cuFileBatchIOGetStatus(<BatchHandle>batch_idp, min_nr, <unsigned*>nr, <CUfileIOEvents_t*>iocbp, <timespec*>timeout)
    check_status(status)


cpdef batch_io_cancel(intptr_t batch_idp):
    with nogil:
        status = cuFileBatchIOCancel(<BatchHandle>batch_idp)
    check_status(status)


cpdef void batch_io_destroy(intptr_t batch_idp) except*:
    cuFileBatchIODestroy(<BatchHandle>batch_idp)


cpdef read_async(intptr_t fh, intptr_t buf_ptr_base, intptr_t size_p, intptr_t file_offset_p, intptr_t buf_ptr_offset_p, intptr_t bytes_read_p, intptr_t stream):
    with nogil:
        status = cuFileReadAsync(<Handle>fh, <void*>buf_ptr_base, <size_t*>size_p, <off_t*>file_offset_p, <off_t*>buf_ptr_offset_p, <ssize_t*>bytes_read_p, <void*>stream)
    check_status(status)


cpdef write_async(intptr_t fh, intptr_t buf_ptr_base, intptr_t size_p, intptr_t file_offset_p, intptr_t buf_ptr_offset_p, intptr_t bytes_written_p, intptr_t stream):
    with nogil:
        status = cuFileWriteAsync(<Handle>fh, <void*>buf_ptr_base, <size_t*>size_p, <off_t*>file_offset_p, <off_t*>buf_ptr_offset_p, <ssize_t*>bytes_written_p, <void*>stream)
    check_status(status)


cpdef stream_register(intptr_t stream, unsigned flags):
    with nogil:
        status = cuFileStreamRegister(<void*>stream, flags)
    check_status(status)


cpdef stream_deregister(intptr_t stream):
    with nogil:
        status = cuFileStreamDeregister(<void*>stream)
    check_status(status)


cpdef get_version(intptr_t version):
    with nogil:
        status = cuFileGetVersion(<int*>version)
    check_status(status)


cpdef get_parameter_size_t(int param, intptr_t value):
    with nogil:
        status = cuFileGetParameterSizeT(<_SizeTConfigParameter>param, <size_t*>value)
    check_status(status)


cpdef get_parameter_bool(int param, intptr_t value):
    with nogil:
        status = cuFileGetParameterBool(<_BoolConfigParameter>param, <bool*>value)
    check_status(status)


cpdef get_parameter_string(int param, intptr_t desc_str, int len):
    with nogil:
        status = cuFileGetParameterString(<_StringConfigParameter>param, <char*>desc_str, len)
    check_status(status)


cpdef set_parameter_size_t(int param, size_t value):
    with nogil:
        status = cuFileSetParameterSizeT(<_SizeTConfigParameter>param, value)
    check_status(status)


cpdef set_parameter_bool(int param, bool value):
    with nogil:
        status = cuFileSetParameterBool(<_BoolConfigParameter>param, value)
    check_status(status)


cpdef set_parameter_string(int param, intptr_t desc_str):
    with nogil:
        status = cuFileSetParameterString(<_StringConfigParameter>param, <const char*>desc_str)
    check_status(status)
