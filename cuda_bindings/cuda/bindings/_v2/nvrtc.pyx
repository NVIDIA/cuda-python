# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0
#
# This code was automatically generated across versions from 12.9.0 to 13.3.0. Do not modify it directly.
# CYTHON-BINDINGS-GENERATED-DO-NOT-MODIFY-THIS-FILE: format=1; content-sha256=e5360fc057cdd7b8e28b4d502821443d190e589b200dce1a234494ad6e9abf93


# <<<< PREAMBLE CONTENT >>>>

cimport cpython as _cyb_cpython
cimport cpython.buffer as _cyb_cpython_buffer
from cython cimport view as _cyb_view
from libc.stdlib cimport (
    calloc as _cyb_calloc,
    free as _cyb_free,
    malloc as _cyb_malloc,
)
from libc.string cimport (
    memcmp as _cyb_memcmp,
    memcpy as _cyb_memcpy,
)

from cuda.bindings._internal._fast_enum import FastEnum as _cyb_FastEnum

import numpy as _numpy

cdef _cyb___getbuffer(object self, _cyb_cpython.Py_buffer *buffer, void *ptr, int size, bint readonly):
    buffer.buf = <char *>ptr
    buffer.format = 'b'
    buffer.internal = NULL
    buffer.itemsize = 1
    buffer.len = size
    buffer.ndim = 1
    buffer.obj = self
    buffer.readonly = readonly
    buffer.shape = &buffer.len
    buffer.strides = &buffer.itemsize
    buffer.suboffsets = NULL

cdef _cyb_from_buffer(buffer, size, lowpp_type):
    cdef _cyb_cpython.Py_buffer view
    if _cyb_cpython.PyObject_GetBuffer(buffer, &view, _cyb_cpython_buffer.PyBUF_SIMPLE) != 0:
        raise TypeError("buffer argument does not support the buffer protocol")
    try:
        if view.itemsize != 1:
            raise ValueError("buffer itemsize must be 1 byte")
        if view.len != size:
            raise ValueError(f"buffer length must be {size} bytes")
        return lowpp_type.from_ptr(<intptr_t><void *>view.buf, not view.readonly, buffer)
    finally:
        _cyb_cpython.PyBuffer_Release(&view)

cdef _cyb_from_data(data, dtype_name, expected_dtype, lowpp_type):
    # _numpy.recarray is a subclass of _numpy.ndarray, so implicitly handled here.
    if isinstance(data, lowpp_type):
        return data
    if not isinstance(data, _numpy.ndarray):
        raise TypeError("data argument must be a NumPy ndarray")
    if data.size != 1:
        raise ValueError("data array must have a size of 1")
    if data.dtype != expected_dtype:
        raise ValueError(f"data array must be of dtype {dtype_name}")
    return lowpp_type.from_ptr(data.ctypes.data, not data.flags.writeable, data)


# <<<< END OF PREAMBLE CONTENT >>>>

cimport cython  # NOQA
from libcpp.vector cimport vector

from cuda.bindings._internal._fast_enum import FastEnum as _FastEnum


###############################################################################
# Enum
###############################################################################

class Result(_cyb_FastEnum):
    """
    The enumerated type `nvrtcResult` defines API call result codes. NVRTC
    API functions return `nvrtcResult` to indicate the call result.

    See `nvrtcResult`.
    """
    SUCCESS = NVRTC_SUCCESS
    ERROR_OUT_OF_MEMORY = NVRTC_ERROR_OUT_OF_MEMORY
    ERROR_PROGRAM_CREATION_FAILURE = NVRTC_ERROR_PROGRAM_CREATION_FAILURE
    ERROR_INVALID_INPUT = NVRTC_ERROR_INVALID_INPUT
    ERROR_INVALID_PROGRAM = NVRTC_ERROR_INVALID_PROGRAM
    ERROR_INVALID_OPTION = NVRTC_ERROR_INVALID_OPTION
    ERROR_COMPILATION = NVRTC_ERROR_COMPILATION
    ERROR_BUILTIN_OPERATION_FAILURE = NVRTC_ERROR_BUILTIN_OPERATION_FAILURE
    ERROR_NO_NAME_EXPRESSIONS_AFTER_COMPILATION = NVRTC_ERROR_NO_NAME_EXPRESSIONS_AFTER_COMPILATION
    ERROR_NO_LOWERED_NAMES_BEFORE_COMPILATION = NVRTC_ERROR_NO_LOWERED_NAMES_BEFORE_COMPILATION
    ERROR_NAME_EXPRESSION_NOT_VALID = NVRTC_ERROR_NAME_EXPRESSION_NOT_VALID
    ERROR_INTERNAL_ERROR = NVRTC_ERROR_INTERNAL_ERROR
    ERROR_TIME_FILE_WRITE_FAILED = NVRTC_ERROR_TIME_FILE_WRITE_FAILED
    ERROR_NO_PCH_CREATE_ATTEMPTED = NVRTC_ERROR_NO_PCH_CREATE_ATTEMPTED
    ERROR_PCH_CREATE_HEAP_EXHAUSTED = NVRTC_ERROR_PCH_CREATE_HEAP_EXHAUSTED
    ERROR_PCH_CREATE = NVRTC_ERROR_PCH_CREATE
    ERROR_CANCELLED = NVRTC_ERROR_CANCELLED
    ERROR_TIME_TRACE_FILE_WRITE_FAILED = NVRTC_ERROR_TIME_TRACE_FILE_WRITE_FAILED
    ERROR_BUSY = NVRTC_ERROR_BUSY


class InstallHeadersFlag(_FastEnum):
    """Flags for :func:`install_bundled_headers`."""
    SKIP_IF_EXISTS = (
        0x0,
        "Skip installation if version marker exists and version matches. "
        "This is the default behavior when flags=0."
    )
    FORCE_OVERWRITE = (
        0x1,
        "Clear existing directory contents before installation. "
        "Guarantees consistency by removing any existing files first."
    )
    NO_WAIT = (
        0x2,
        "Return NVRTC_ERROR_BUSY immediately if installation is in progress "
        "by another process, instead of waiting for the lock. "
        "Can be combined with FORCE_OVERWRITE using bitwise OR. "
        "Do not wait for installation to complete."
    )


###############################################################################
# Error handling
###############################################################################


class NvrtcError(Exception):
    def __init__(self, status):
        self.status = status
        s = get_error_string(status)
        super(NvrtcError, self).__init__(s)

    def __reduce__(self):
        return (type(self), (self.status,))

class OutOfMemoryError(NvrtcError):
    pass
class ProgramCreationFailureError(NvrtcError):
    pass
class InvalidInputError(NvrtcError):
    pass
class InvalidProgramError(NvrtcError):
    pass
class InvalidOptionError(NvrtcError):
    pass
class CompilationError(NvrtcError):
    pass
class BuiltinOperationFailureError(NvrtcError):
    pass
class NoNameExpressionsAfterCompilationError(NvrtcError):
    pass
class NoLoweredNamesBeforeCompilationError(NvrtcError):
    pass
class NameExpressionNotValidError(NvrtcError):
    pass
class InternalErrorError(NvrtcError):
    pass
class TimeFileWriteFailedError(NvrtcError):
    pass
class NoPchCreateAttemptedError(NvrtcError):
    pass
class PchCreateHeapExhaustedError(NvrtcError):
    pass
class PchCreateError(NvrtcError):
    pass
class CancelledError(NvrtcError):
    pass
class TimeTraceFileWriteFailedError(NvrtcError):
    pass
class BusyError(NvrtcError):
    pass
cdef object _nvrtc_error_factory(int status):
    cdef object pystatus = status
    if status == 1:
        return OutOfMemoryError(pystatus)
    elif status == 2:
        return ProgramCreationFailureError(pystatus)
    elif status == 3:
        return InvalidInputError(pystatus)
    elif status == 4:
        return InvalidProgramError(pystatus)
    elif status == 5:
        return InvalidOptionError(pystatus)
    elif status == 6:
        return CompilationError(pystatus)
    elif status == 7:
        return BuiltinOperationFailureError(pystatus)
    elif status == 8:
        return NoNameExpressionsAfterCompilationError(pystatus)
    elif status == 9:
        return NoLoweredNamesBeforeCompilationError(pystatus)
    elif status == 10:
        return NameExpressionNotValidError(pystatus)
    elif status == 11:
        return InternalErrorError(pystatus)
    elif status == 12:
        return TimeFileWriteFailedError(pystatus)
    elif status == 13:
        return NoPchCreateAttemptedError(pystatus)
    elif status == 14:
        return PchCreateHeapExhaustedError(pystatus)
    elif status == 15:
        return PchCreateError(pystatus)
    elif status == 16:
        return CancelledError(pystatus)
    elif status == 17:
        return TimeTraceFileWriteFailedError(pystatus)
    elif status == 18:
        return BusyError(pystatus)
    return NvrtcError(status)

InternalError = InternalErrorError


@cython.profile(False)
cdef int check_status(int status) except 1 nogil:
    if status != 0:
        with gil:
            raise _nvrtc_error_factory(status)
    return status


###############################################################################
# POD definitions
###############################################################################

cdef _get_bundled_headers_info_dtype_offsets():
    cdef nvrtcBundledHeadersInfo pod
    return _numpy.dtype({
        'names': ['available', 'compressed_size', 'uncompressed_size', 'cuda_version_major', 'cuda_version_minor', 'num_files'],
        'formats': [_numpy.int32, _numpy.uint64, _numpy.uint64, _numpy.int32, _numpy.int32, _numpy.uint32],
        'offsets': [
            (<intptr_t>&(pod.available)) - (<intptr_t>&pod),
            (<intptr_t>&(pod.compressedSize)) - (<intptr_t>&pod),
            (<intptr_t>&(pod.uncompressedSize)) - (<intptr_t>&pod),
            (<intptr_t>&(pod.cudaVersionMajor)) - (<intptr_t>&pod),
            (<intptr_t>&(pod.cudaVersionMinor)) - (<intptr_t>&pod),
            (<intptr_t>&(pod.numFiles)) - (<intptr_t>&pod),
        ],
        'itemsize': sizeof(nvrtcBundledHeadersInfo),
    })

bundled_headers_info_dtype = _get_bundled_headers_info_dtype_offsets()

cdef class BundledHeadersInfo:
    """Empty-initialize an instance of `nvrtcBundledHeadersInfo`.


    .. seealso:: `nvrtcBundledHeadersInfo`
    """
    cdef:
        nvrtcBundledHeadersInfo *_ptr
        object _owner
        bint _owned
        bint _readonly

    def __init__(self):
        self._ptr = <nvrtcBundledHeadersInfo *>_cyb_calloc(1, sizeof(nvrtcBundledHeadersInfo))
        if self._ptr == NULL:
            raise MemoryError("Error allocating BundledHeadersInfo")
        self._owner = None
        self._owned = True
        self._readonly = False

    def __dealloc__(self):
        cdef nvrtcBundledHeadersInfo *ptr
        if self._owned and self._ptr != NULL:
            ptr = self._ptr
            self._ptr = NULL
            _cyb_free(ptr)

    def __repr__(self):
        return f"<{__name__}.BundledHeadersInfo object at {hex(id(self))}>"

    @property
    def ptr(self):
        """Get the pointer address to the data as Python :class:`int`."""
        return <intptr_t>(self._ptr)

    cdef intptr_t _get_ptr(self):
        return <intptr_t>(self._ptr)

    def __int__(self):
        return <intptr_t>(self._ptr)

    def __eq__(self, other):
        cdef BundledHeadersInfo other_
        if not isinstance(other, BundledHeadersInfo):
            return False
        other_ = other
        return (_cyb_memcmp(<void *><intptr_t>(self._ptr), <void *><intptr_t>(other_._ptr), sizeof(nvrtcBundledHeadersInfo)) == 0)

    def __getbuffer__(self, _cyb_cpython.Py_buffer *buffer, int flags):
        _cyb___getbuffer(self, buffer, <void *>self._ptr, sizeof(nvrtcBundledHeadersInfo), self._readonly)

    def __releasebuffer__(self, Py_buffer *buffer):
        pass

    def __setitem__(self, key, val):
        if key == 0 and isinstance(val, _numpy.ndarray):
            self._ptr = <nvrtcBundledHeadersInfo *>_cyb_malloc(sizeof(nvrtcBundledHeadersInfo))
            if self._ptr == NULL:
                raise MemoryError("Error allocating BundledHeadersInfo")
            _cyb_memcpy(<void*>self._ptr, <void*><intptr_t>val.ctypes.data, sizeof(nvrtcBundledHeadersInfo))
            self._owner = None
            self._owned = True
            self._readonly = not val.flags.writeable
        else:
            setattr(self, key, val)

    @property
    def available(self):
        """int: Non-zero if bundled headers are available"""
        return self._ptr[0].available

    @available.setter
    def available(self, val):
        if self._readonly:
            raise ValueError("This BundledHeadersInfo instance is read-only")
        self._ptr[0].available = val

    @property
    def compressed_size(self):
        """int: Size of compressed archive in bytes"""
        return self._ptr[0].compressedSize

    @compressed_size.setter
    def compressed_size(self, val):
        if self._readonly:
            raise ValueError("This BundledHeadersInfo instance is read-only")
        self._ptr[0].compressedSize = val

    @property
    def uncompressed_size(self):
        """int: Estimated size when extracted in bytes"""
        return self._ptr[0].uncompressedSize

    @uncompressed_size.setter
    def uncompressed_size(self, val):
        if self._readonly:
            raise ValueError("This BundledHeadersInfo instance is read-only")
        self._ptr[0].uncompressedSize = val

    @property
    def cuda_version_major(self):
        """int: CUDA major version of bundled headers"""
        return self._ptr[0].cudaVersionMajor

    @cuda_version_major.setter
    def cuda_version_major(self, val):
        if self._readonly:
            raise ValueError("This BundledHeadersInfo instance is read-only")
        self._ptr[0].cudaVersionMajor = val

    @property
    def cuda_version_minor(self):
        """int: CUDA minor version of bundled headers"""
        return self._ptr[0].cudaVersionMinor

    @cuda_version_minor.setter
    def cuda_version_minor(self, val):
        if self._readonly:
            raise ValueError("This BundledHeadersInfo instance is read-only")
        self._ptr[0].cudaVersionMinor = val

    @property
    def num_files(self):
        """int: Number of header files in the bundle"""
        return self._ptr[0].numFiles

    @num_files.setter
    def num_files(self, val):
        if self._readonly:
            raise ValueError("This BundledHeadersInfo instance is read-only")
        self._ptr[0].numFiles = val

    @staticmethod
    def from_buffer(buffer):
        """Create an BundledHeadersInfo instance with the memory from the given buffer."""
        return _cyb_from_buffer(buffer, sizeof(nvrtcBundledHeadersInfo), BundledHeadersInfo)

    @staticmethod
    def from_data(data):
        """Create an BundledHeadersInfo instance wrapping the given NumPy array.

        Args:
            data (_numpy.ndarray): a single-element array of dtype `bundled_headers_info_dtype` holding the data.
        """
        return _cyb_from_data(data, "bundled_headers_info_dtype", bundled_headers_info_dtype, BundledHeadersInfo)

    @staticmethod
    def from_ptr(intptr_t ptr, bint readonly=False, object owner=None):
        """Create an BundledHeadersInfo instance wrapping the given pointer.

        Args:
            ptr (intptr_t): pointer address as Python :class:`int` to the data.
            owner (object): The Python object that owns the pointer. If not provided, data will be copied.
            readonly (bool): whether the data is read-only (to the user). default is `False`.
        """
        if ptr == 0:
            raise ValueError("ptr must not be null (0)")
        cdef BundledHeadersInfo obj = BundledHeadersInfo.__new__(BundledHeadersInfo)
        if owner is None:
            obj._ptr = <nvrtcBundledHeadersInfo *>_cyb_malloc(sizeof(nvrtcBundledHeadersInfo))
            if obj._ptr == NULL:
                raise MemoryError("Error allocating BundledHeadersInfo")
            _cyb_memcpy(<void*>(obj._ptr), <void*>ptr, sizeof(nvrtcBundledHeadersInfo))
            obj._owner = None
            obj._owned = True
        else:
            obj._ptr = <nvrtcBundledHeadersInfo *>ptr
            obj._owner = owner
            obj._owned = False
        obj._readonly = readonly
        return obj


###############################################################################
# Wrapper functions
###############################################################################

cpdef intptr_t create_program(bytes src, name, headers=None, include_names=None) except? 0:
    """nvrtcCreateProgram creates an instance of nvrtcProgram with the given input parameters.

    Args:
        src (bytes): CUDA program source.
        name (bytes | None): CUDA program name. ``None`` or ``""`` causes ``"default_program"``
            to be used.
        headers (list[bytes] | None): Sources of the headers. ``None`` is treated as an empty
            list (no headers).
        include_names (list[bytes] | None): Name of each header by which it can be included in
            the CUDA program source. Must have the same length as *headers*.

    Returns:
        intptr_t: Opaque handle to the created program. Pass to :func:`destroy_program` when done.

    .. seealso:: `nvrtcCreateProgram`
    """
    if headers is None:
        headers = []
    if include_names is None:
        include_names = []
    if len(headers) != len(include_names):
        raise ValueError(
            f"headers and include_names must have the same length "
            f"({len(headers)} != {len(include_names)})"
        )
    cdef int num_headers = len(headers)
    cdef Program prog
    cdef const char* c_src = src
    cdef bytes _name = b"" if name is None else name
    cdef const char* c_name = _name
    cdef vector[const char*] cy_headers = headers
    cdef vector[const char*] cy_include_names = include_names
    cdef const char** hdr_data = NULL
    cdef const char** inc_data = NULL
    if num_headers:
        hdr_data = cy_headers.data()
        inc_data = cy_include_names.data()
    with nogil:
        __status__ = nvrtcCreateProgram(&prog, c_src, c_name, num_headers, hdr_data, inc_data)
    check_status(__status__)
    return <intptr_t>prog


cpdef compile_program(intptr_t prog, options=None):
    """nvrtcCompileProgram compiles the given program.

    It supports compile options listed in Supported Compile Options.

    Args:
        prog (intptr_t): CUDA Runtime Compilation program.
        options (list[bytes] | None): Compiler options as a list of byte strings.
            May be ``None`` or an empty list for no options.

    .. seealso:: `nvrtcCompileProgram`
    """
    if options is None:
        options = []
    cdef int num_options = len(options)
    cdef vector[const char*] cy_options = options
    cdef const char** opt_data = NULL
    if num_options:
        opt_data = cy_options.data()
    with nogil:
        __status__ = nvrtcCompileProgram(<Program>prog, num_options, opt_data)
    check_status(__status__)


cpdef set_flow_callback(intptr_t prog, intptr_t callback, intptr_t payload):
    """nvrtcSetFlowCallback registers a callback that the compiler invokes during :func:`compile_program`.

    The callback signature must be ``int callback(void *param1, void *param2)``.
    The compiler passes *payload* as *param1* and ``NULL`` as *param2* (reserved).
    Return 1 to cancel compilation, 0 to continue; the callback must return
    consistently, be thread-safe, and must not call any NVRTC/libnvvm/PTX APIs.

    Pass *callback* as a raw C function-pointer integer (e.g. via ``ctypes.cast``).
    Pass 0 for *callback* to clear a previously registered callback.

    Args:
        prog (intptr_t): CUDA Runtime Compilation program.
        callback (intptr_t): C function pointer ``int (*)(void*, void*)`` cast to an integer,
            or 0 to clear.
        payload (intptr_t): Opaque pointer passed to the callback as its first argument.

    .. seealso:: `nvrtcSetFlowCallback`
    """
    with nogil:
        __status__ = nvrtcSetFlowCallback(<Program>prog, <void*>callback, <void*>payload)
    check_status(__status__)


cpdef bytes get_lowered_name(intptr_t prog, bytes name_expression):
    """nvrtcGetLoweredName extracts the lowered (mangled) name for a ``__global__`` function or ``__device__``/``__constant__`` variable.

    The memory containing the name is released when the program is destroyed by
    :func:`destroy_program`. The identical *name_expression* must have been previously
    provided to :func:`add_name_expression`.

    Args:
        prog (intptr_t): CUDA Runtime Compilation program.
        name_expression (bytes): Constant expression denoting the address of a
            ``__global__`` function or ``__device__``/``__constant__`` variable.

    Returns:
        bytes: C string containing the lowered (mangled) name, or ``None``.

    .. seealso:: `nvrtcGetLoweredName`
    """
    cdef const char* c_name_expression = name_expression
    cdef const char* lowered_name = NULL
    with nogil:
        __status__ = nvrtcGetLoweredName(<Program>prog, c_name_expression, &lowered_name)
    check_status(__status__)
    return <bytes>lowered_name if lowered_name != NULL else None


cpdef install_bundled_headers(bytes install_path, unsigned int flags):
    """nvrtcInstallBundledHeaders extracts CUDA headers bundled with NVRTC to a specified directory.

    NVRTC bundles a set of CUDA Toolkit headers and CCCL within libnvrtc-builtins.
    After extraction, compile kernels by passing ``-I<installPath>`` and
    ``-I<installPath>/cccl`` to :func:`compile_program`. A version marker file
    (``.nvrtc_headers_version``) is created to track the installed version.
    The function is thread-safe and process-safe; concurrent calls are serialized
    using file locking.

    Args:
        install_path (bytes): Path where headers should be extracted (UTF-8 encoded).
            The directory is created if it does not exist.
        flags (unsigned int): Bitwise OR of :class:`InstallHeadersFlag` values (or 0
            for the default ``SKIP_IF_EXISTS`` behaviour).

    Returns:
        bytes | None: Detailed error message on failure, or ``None`` on success.

    .. seealso:: `nvrtcInstallBundledHeaders`
    """
    cdef const char* c_install_path = install_path
    cdef const char* error_log = NULL
    with nogil:
        __status__ = nvrtcInstallBundledHeaders(c_install_path, flags, &error_log)
    check_status(__status__)
    return <bytes>error_log if error_log != NULL else None


cpdef tuple get_bundled_headers_info():
    """nvrtcGetBundledHeadersInfo queries information about the bundled headers without extracting them.

    Allows users to determine if bundled headers are available and get size estimates
    before calling :func:`install_bundled_headers`.

    Returns:
        tuple[BundledHeadersInfo, bytes | None]: Header information struct and an optional
        detailed error message (``None`` on success).

    .. seealso:: `nvrtcGetBundledHeadersInfo`
    """
    cdef BundledHeadersInfo info = BundledHeadersInfo()
    cdef nvrtcBundledHeadersInfo* c_info = <nvrtcBundledHeadersInfo*><intptr_t>(info._get_ptr())
    cdef const char* error_log = NULL
    with nogil:
        __status__ = nvrtcGetBundledHeadersInfo(c_info, &error_log)
    check_status(__status__)
    return info, (<bytes>error_log if error_log != NULL else None)


cpdef remove_bundled_headers(bytes install_path):
    """nvrtcRemoveBundledHeaders removes previously installed bundled headers.

    Recursively removes all files and subdirectories within the installation
    directory to help manage disk space.

    .. note:: This removes ALL contents of the specified directory, not just files
       installed by NVRTC. Use with caution.

    Args:
        install_path (bytes): Path where headers were previously installed; must be
            the same path used with :func:`install_bundled_headers`.

    Returns:
        bytes | None: Detailed error message on failure, or ``None`` on success.

    .. seealso:: `nvrtcRemoveBundledHeaders`
    """
    cdef const char* c_install_path = install_path
    cdef const char* error_log = NULL
    with nogil:
        __status__ = nvrtcRemoveBundledHeaders(c_install_path, &error_log)
    check_status(__status__)
    return <bytes>error_log if error_log != NULL else None


cpdef str get_error_string(int result):
    """nvrtcGetErrorString is a helper function that returns a string describing the given ``nvrtcResult`` code, e.g., NVRTC_SUCCESS to ``"NVRTC_SUCCESS"``. For unrecognized enumeration values, it returns ``"NVRTC_ERROR unknown"``.

    Args:
        result (Result): CUDA Runtime Compilation API result code.

    .. seealso:: `nvrtcGetErrorString`
    """
    cdef const char *_output_cstr_
    cdef bytes _output_
    with nogil:
        _output_cstr_ = nvrtcGetErrorString(<_Result>result)
    _output_ = _output_cstr_
    return _output_.decode()


cpdef tuple version():
    """nvrtcVersion sets the output parameters ``major`` and ``minor`` with the CUDA Runtime Compilation version number.

    Returns:
        A 2-tuple containing:

        - int: CUDA Runtime Compilation major version number.
        - int: CUDA Runtime Compilation minor version number.

    .. seealso:: `nvrtcVersion`
    """
    cdef int major
    cdef int minor
    with nogil:
        __status__ = nvrtcVersion(&major, &minor)
    check_status(__status__)
    return (major, minor)


cpdef int get_num_supported_archs() except? -1:
    """nvrtcGetNumSupportedArchs sets the output parameter ``num_archs`` with the number of architectures supported by NVRTC. This can then be used to pass an array to ``nvrtcGetSupportedArchs`` to get the supported architectures.

    Returns:
        int: number of supported architectures.

    .. seealso:: `nvrtcGetNumSupportedArchs`
    """
    cdef int num_archs
    with nogil:
        __status__ = nvrtcGetNumSupportedArchs(&num_archs)
    check_status(__status__)
    return num_archs


cpdef object get_supported_archs():
    """nvrtcGetSupportedArchs populates the array passed via the output parameter ``supported_archs`` with the architectures supported by NVRTC. The array is sorted in the ascending order. The size of the array to be passed can be determined using ``nvrtcGetNumSupportedArchs``.

    Returns:
        int: sorted array of supported architectures.

    .. seealso:: `nvrtcGetSupportedArchs`
    """
    cdef int numArchs
    with nogil:
        __status__ = nvrtcGetNumSupportedArchs(&numArchs)
    check_status(__status__)
    if numArchs == 0:
        return _cyb_view.array(shape=(1,), itemsize=sizeof(int), format="i", mode="c")[:0]
    cdef _cyb_view.array supported_archs = _cyb_view.array(shape=(numArchs,), itemsize=sizeof(int), format="i", mode="c")
    cdef int *supported_archs_ptr = <int *>(supported_archs.data)
    with nogil:
        __status__ = nvrtcGetSupportedArchs(supported_archs_ptr)
    check_status(__status__)
    return supported_archs


cpdef destroy_program(intptr_t prog):
    """nvrtcDestroyProgram destroys the given program.

    Args:
        prog (intptr_t): CUDA Runtime Compilation program.

    .. seealso:: `nvrtcDestroyProgram`
    """
    cdef Program _prog_ = <Program>prog
    with nogil:
        __status__ = nvrtcDestroyProgram(&_prog_)
    check_status(__status__)


cpdef size_t get_ptx_size(intptr_t prog) except? 0:
    """nvrtcGetPTXSize sets the value of ``ptx_size_ret`` with the size of the PTX generated by the previous compilation of ``prog`` (including the trailing ``NULL``).

    Args:
        prog (intptr_t): CUDA Runtime Compilation program.

    Returns:
        size_t: Size of the generated PTX (including the trailing
            ``NULL``).

    .. seealso:: `nvrtcGetPTXSize`
    """
    cdef size_t ptx_size_ret
    with nogil:
        __status__ = nvrtcGetPTXSize(<Program>prog, &ptx_size_ret)
    check_status(__status__)
    return ptx_size_ret


cpdef bytes get_ptx(intptr_t prog):
    """nvrtcGetPTX stores the PTX generated by the previous compilation of ``prog`` in the memory pointed by ``ptx``.

    Args:
        prog (intptr_t): CUDA Runtime Compilation program.

    Returns:
        char: Compiled result.

    .. seealso:: `nvrtcGetPTX`
    """
    cdef size_t ptxSizeRet
    with nogil:
        __status__ = nvrtcGetPTXSize(<Program>prog, &ptxSizeRet)
    check_status(__status__)
    if ptxSizeRet == 0:
        return b""
    cdef bytes _ptx_ = bytes(ptxSizeRet)
    cdef char* ptx = _ptx_
    with nogil:
        __status__ = nvrtcGetPTX(<Program>prog, ptx)
    check_status(__status__)
    return _ptx_


cpdef size_t get_cubin_size(intptr_t prog) except? 0:
    """nvrtcGetCUBINSize sets the value of ``cubin_size_ret`` with the size of the cubin generated by the previous compilation of ``prog``. The value of cubin_size_ret is set to 0 if the value specified to ``-arch`` is a virtual architecture instead of an actual architecture.

    Args:
        prog (intptr_t): CUDA Runtime Compilation program.

    Returns:
        size_t: Size of the generated cubin.

    .. seealso:: `nvrtcGetCUBINSize`
    """
    cdef size_t cubin_size_ret
    with nogil:
        __status__ = nvrtcGetCUBINSize(<Program>prog, &cubin_size_ret)
    check_status(__status__)
    return cubin_size_ret


cpdef bytes get_cubin(intptr_t prog):
    """nvrtcGetCUBIN stores the cubin generated by the previous compilation of ``prog`` in the memory pointed by ``cubin``. No cubin is available if the value specified to ``-arch`` is a virtual architecture instead of an actual architecture. The cubin does not contain code for the Tile functions (``__tile__`` / ``__tile_global__``) or variables (``__tile__``); use :func:`get_tile_ir` to extract the cuda_tile IR generated for Tile code.

    Args:
        prog (intptr_t): CUDA Runtime Compilation program.

    Returns:
        char: Compiled and assembled result.

    .. seealso:: `nvrtcGetCUBIN`
    """
    cdef size_t cubinSizeRet
    with nogil:
        __status__ = nvrtcGetCUBINSize(<Program>prog, &cubinSizeRet)
    check_status(__status__)
    if cubinSizeRet == 0:
        return b""
    cdef bytes _cubin_ = bytes(cubinSizeRet)
    cdef char* cubin = _cubin_
    with nogil:
        __status__ = nvrtcGetCUBIN(<Program>prog, cubin)
    check_status(__status__)
    return _cubin_


cpdef size_t get_ltoir_size(intptr_t prog) except? 0:
    """nvrtcGetLTOIRSize sets the value of ``ltoir_size_ret`` with the size of the LTO IR generated by the previous compilation of ``prog``. The value of ltoir_size_ret is set to 0 if the program was not compiled with ``-dlto``.

    Args:
        prog (intptr_t): CUDA Runtime Compilation program.

    Returns:
        size_t: Size of the generated LTO IR.

    .. seealso:: `nvrtcGetLTOIRSize`
    """
    cdef size_t ltoir_size_ret
    with nogil:
        __status__ = nvrtcGetLTOIRSize(<Program>prog, &ltoir_size_ret)
    check_status(__status__)
    return ltoir_size_ret


cpdef bytes get_ltoir(intptr_t prog):
    """nvrtcGetltoir stores the LTO IR generated by the previous compilation of ``prog`` in the memory pointed by ``ltoir``. No LTO IR is available if the program was compiled without ``-dlto``.

    Args:
        prog (intptr_t): CUDA Runtime Compilation program.

    Returns:
        char: Compiled result.

    .. seealso:: `nvrtcGetLTOIR`
    """
    cdef size_t LTOIRSizeRet
    with nogil:
        __status__ = nvrtcGetLTOIRSize(<Program>prog, &LTOIRSizeRet)
    check_status(__status__)
    if LTOIRSizeRet == 0:
        return b""
    cdef bytes _ltoir_ = bytes(LTOIRSizeRet)
    cdef char* ltoir = _ltoir_
    with nogil:
        __status__ = nvrtcGetLTOIR(<Program>prog, ltoir)
    check_status(__status__)
    return _ltoir_


cpdef size_t get_optix_ir_size(intptr_t prog) except? 0:
    """nvrtcGetOptiXIRSize sets the value of ``optixir_size_ret`` with the size of the OptiX IR generated by the previous compilation of ``prog``. The value of nvrtcGetOptiXIRSize is set to 0 if the program was compiled with options incompatible with OptiX IR generation.

    Args:
        prog (intptr_t): CUDA Runtime Compilation program.

    Returns:
        size_t: Size of the generated LTO IR.

    .. seealso:: `nvrtcGetOptiXIRSize`
    """
    cdef size_t optixir_size_ret
    with nogil:
        __status__ = nvrtcGetOptiXIRSize(<Program>prog, &optixir_size_ret)
    check_status(__status__)
    return optixir_size_ret


cpdef bytes get_optix_ir(intptr_t prog):
    """nvrtcGetOptiXIR stores the OptiX IR generated by the previous compilation of ``prog`` in the memory pointed by ``optixir``. No OptiX IR is available if the program was compiled with options incompatible with OptiX IR generation.

    Args:
        prog (intptr_t): CUDA Runtime Compilation program.

    Returns:
        char: Optix IR Compiled result.

    .. seealso:: `nvrtcGetOptiXIR`
    """
    cdef size_t optixirSizeRet
    with nogil:
        __status__ = nvrtcGetOptiXIRSize(<Program>prog, &optixirSizeRet)
    check_status(__status__)
    if optixirSizeRet == 0:
        return b""
    cdef bytes _optixir_ = bytes(optixirSizeRet)
    cdef char* optixir = _optixir_
    with nogil:
        __status__ = nvrtcGetOptiXIR(<Program>prog, optixir)
    check_status(__status__)
    return _optixir_


cpdef size_t get_program_log_size(intptr_t prog) except? 0:
    """nvrtcGetProgramLogSize sets ``log_size_ret`` with the size of the log generated by the previous compilation of ``prog`` (including the trailing ``NULL``).

    Args:
        prog (intptr_t): CUDA Runtime Compilation program.

    Returns:
        size_t: Size of the compilation log (including the trailing
            ``NULL``).

    .. seealso:: `nvrtcGetProgramLogSize`
    """
    cdef size_t log_size_ret
    with nogil:
        __status__ = nvrtcGetProgramLogSize(<Program>prog, &log_size_ret)
    check_status(__status__)
    return log_size_ret


cpdef bytes get_program_log(intptr_t prog):
    """nvrtcGetProgramLog stores the log generated by the previous compilation of ``prog`` in the memory pointed by ``log``.

    Args:
        prog (intptr_t): CUDA Runtime Compilation program.

    Returns:
        char: Compilation log.

    .. seealso:: `nvrtcGetProgramLog`
    """
    cdef size_t logSizeRet
    with nogil:
        __status__ = nvrtcGetProgramLogSize(<Program>prog, &logSizeRet)
    check_status(__status__)
    if logSizeRet == 0:
        return b""
    cdef bytes _log_ = bytes(logSizeRet)
    cdef char* log = _log_
    with nogil:
        __status__ = nvrtcGetProgramLog(<Program>prog, log)
    check_status(__status__)
    return _log_


cpdef add_name_expression(intptr_t prog, name_expression):
    """nvrtcAddNameExpression notes the given name expression denoting the address of a global function or device/__constant__ variable.

    Args:
        prog (intptr_t): CUDA Runtime Compilation program.
        name_expression (str): constant expression denoting the
            address of a global function or device/__constant__
            variable.

    .. seealso:: `nvrtcAddNameExpression`
    """
    if not isinstance(name_expression, str):
        raise TypeError("name_expression must be a Python str")
    cdef bytes _temp_name_expression_ = (<str>name_expression).encode()
    cdef char* _name_expression_ = _temp_name_expression_
    with nogil:
        __status__ = nvrtcAddNameExpression(<Program>prog, <const char* const>_name_expression_)
    check_status(__status__)


cpdef size_t get_pch_heap_size() except? 0:
    """retrieve the current size of the PCH Heap.

    Returns:
        size_t: pointer to location where the size of the PCH Heap
            will be stored.

    .. seealso:: `nvrtcGetPCHHeapSize`
    """
    cdef size_t ret
    with nogil:
        __status__ = nvrtcGetPCHHeapSize(&ret)
    check_status(__status__)
    return ret


cpdef set_pch_heap_size(size_t size):
    """set the size of the PCH Heap.

    Args:
        size (size_t): requested size of the PCH Heap, in bytes.

    .. seealso:: `nvrtcSetPCHHeapSize`
    """
    with nogil:
        __status__ = nvrtcSetPCHHeapSize(size)
    check_status(__status__)


cpdef int get_pch_create_status(intptr_t prog) except? -1:
    """returns the PCH creation status.

    Args:
        prog (intptr_t): CUDA Runtime Compilation program.

    .. seealso:: `nvrtcGetPCHCreateStatus`
    """
    cdef int ret
    with nogil:
        ret = <int>nvrtcGetPCHCreateStatus(<Program>prog)
    return ret


cpdef size_t get_pch_heap_size_required(intptr_t prog) except? 0:
    """retrieve the required size of the PCH heap required to compile the given program.

    Args:
        prog (intptr_t): CUDA Runtime Compilation program.

    Returns:
        size_t: pointer to location where the required size of the PCH
            Heap will be stored.

    .. seealso:: `nvrtcGetPCHHeapSizeRequired`
    """
    cdef size_t size
    with nogil:
        __status__ = nvrtcGetPCHHeapSizeRequired(<Program>prog, &size)
    check_status(__status__)
    return size


cpdef size_t get_tile_ir_size(intptr_t prog) except? 0:
    """nvrtcGetTileIRSize sets the value of ``tile_ir_size_ret`` with the size of the cuda_tile IR generated by the previous compilation of ``prog``.

    Args:
        prog (intptr_t): CUDA Runtime Compilation program.

    Returns:
        size_t: Size of the generated cuda_tile IR.

    .. seealso:: `nvrtcGetTileIRSize`
    """
    cdef size_t tile_ir_size_ret
    with nogil:
        __status__ = nvrtcGetTileIRSize(<Program>prog, &tile_ir_size_ret)
    check_status(__status__)
    return tile_ir_size_ret


cpdef bytes get_tile_ir(intptr_t prog):
    """nvrtcGettile_ir stores the cuda_tile IR generated by the previous compilation of ``prog`` in the memory pointed by ``tile_ir``.

    Args:
        prog (intptr_t): CUDA Runtime Compilation program.

    Returns:
        char: Generated cuda_tile IR.

    .. seealso:: `nvrtcGetTileIR`
    """
    cdef size_t TileIRSizeRet
    with nogil:
        __status__ = nvrtcGetTileIRSize(<Program>prog, &TileIRSizeRet)
    check_status(__status__)
    if TileIRSizeRet == 0:
        return b""
    cdef bytes _tile_ir_ = bytes(TileIRSizeRet)
    cdef char* tile_ir = _tile_ir_
    with nogil:
        __status__ = nvrtcGetTileIR(<Program>prog, tile_ir)
    check_status(__status__)
    return _tile_ir_


del _cyb_FastEnum
