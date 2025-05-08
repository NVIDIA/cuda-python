# Copyright (c) 2025, NVIDIA CORPORATION & AFFILIATES. ALL RIGHTS RESERVED.
#
# SPDX-License-Identifier: LicenseRef-NVIDIA-SOFTWARE-LICENSE

# To regenerate the dictionary below, navigate to:
#     https://docs.nvidia.com/cuda/cuda-driver-api/group__CUDA__TYPES.html#group__CUDA__TYPES
# (Chrome was used before, but probably it works with other browsers, too.)
# Search for:
#     enum CUresult
# With the mouse, select the entire region with the enum definitions:
#     CUDA_SUCCESS = 0
#     ...
#     CUDA_ERROR_UNKNOWN = 999
#         This indicates that an unknown internal error has occurred.
# Paste into a file, e.g. raw.txt
# python ../../../../../toolshed/reformat_cuda_enums_from_web_as_py.py raw.txt > raw.py
# ruff format raw.py
# Copy raw.py into this file (discarding the `DATA = {`, `}` lines).
# Also update the CUDA Toolkit version number below.
# Done.

# CUDA Toolkit v12.9.0
DRIVER_CU_RESULT_EXPLANATIONS = {
    0: (
        "The API call returned with no errors. In the case of query calls, this also means that the operation"
        " being queried is complete (see cuEventQuery() and cuStreamQuery())."
    ),
    1: (
        "This indicates that one or more of the parameters passed to the API call is not within an acceptable"
        " range of values."
    ),
    2: (
        "The API call failed because it was unable to allocate enough memory or other resources to perform "
        "the requested operation."
    ),
    3: (
        "This indicates that the CUDA driver has not been initialized with cuInit() or that initialization has failed."
    ),
    4: "This indicates that the CUDA driver is in the process of shutting down.",
    5: (
        "This indicates profiler is not initialized for this run. This can happen when the application is "
        "running with external profiling tools like visual profiler."
    ),
    6: (
        "This error return is deprecated as of CUDA 5.0. It is no longer an error to attempt to "
        "enable/disable the profiling via cuProfilerStart or cuProfilerStop without initialization."
    ),
    7: (
        "This error return is deprecated as of CUDA 5.0. It is no longer an error to call cuProfilerStart() "
        "when profiling is already enabled."
    ),
    8: (
        "This error return is deprecated as of CUDA 5.0. It is no longer an error to call cuProfilerStop() "
        "when profiling is already disabled."
    ),
    34: (
        "This indicates that the CUDA driver that the application has loaded is a stub library. Applications "
        "that run with the stub rather than a real driver loaded will result in CUDA API returning this "
        "error."
    ),
    46: (
        "This indicates that requested CUDA device is unavailable at the current time. Devices are often "
        "unavailable due to use of CU_COMPUTEMODE_EXCLUSIVE_PROCESS or CU_COMPUTEMODE_PROHIBITED."
    ),
    100: "This indicates that no CUDA-capable devices were detected by the installed CUDA driver.",
    101: (
        "This indicates that the device ordinal supplied by the user does not correspond to a valid CUDA "
        "device or that the action requested is invalid for the specified device."
    ),
    102: "This error indicates that the Grid license is not applied.",
    200: ("This indicates that the device kernel image is invalid. This can also indicate an invalid CUDA module."),
    201: (
        "This most frequently indicates that there is no context bound to the current thread. This can also "
        "be returned if the context passed to an API call is not a valid handle (such as a context that has "
        "had cuCtxDestroy() invoked on it). This can also be returned if a user mixes different API versions "
        "(i.e. 3010 context with 3020 API calls). See cuCtxGetApiVersion() for more details. This can also be"
        " returned if the green context passed to an API call was not converted to a CUcontext using "
        "cuCtxFromGreenCtx API."
    ),
    202: (
        "This error return is deprecated as of CUDA 3.2. It is no longer an error to attempt to push the "
        "active context via cuCtxPushCurrent(). This indicated that the context being supplied as a parameter"
        " to the API call was already the active context."
    ),
    205: "This indicates that a map or register operation has failed.",
    206: "This indicates that an unmap or unregister operation has failed.",
    207: "This indicates that the specified array is currently mapped and thus cannot be destroyed.",
    208: "This indicates that the resource is already mapped.",
    209: (
        "This indicates that there is no kernel image available that is suitable for the device. This can "
        "occur when a user specifies code generation options for a particular CUDA source file that do not "
        "include the corresponding device configuration."
    ),
    210: "This indicates that a resource has already been acquired.",
    211: "This indicates that a resource is not mapped.",
    212: "This indicates that a mapped resource is not available for access as an array.",
    213: "This indicates that a mapped resource is not available for access as a pointer.",
    214: "This indicates that an uncorrectable ECC error was detected during execution.",
    215: "This indicates that the CUlimit passed to the API call is not supported by the active device.",
    216: (
        "This indicates that the CUcontext passed to the API call can only be bound to a single CPU thread at"
        " a time but is already bound to a CPU thread."
    ),
    217: "This indicates that peer access is not supported across the given devices.",
    218: "This indicates that a PTX JIT compilation failed.",
    219: "This indicates an error with OpenGL or DirectX context.",
    220: "This indicates that an uncorrectable NVLink error was detected during the execution.",
    221: "This indicates that the PTX JIT compiler library was not found.",
    222: "This indicates that the provided PTX was compiled with an unsupported toolchain.",
    223: "This indicates that the PTX JIT compilation was disabled.",
    224: ("This indicates that the CUexecAffinityType passed to the API call is not supported by the active device."),
    225: (
        "This indicates that the code to be compiled by the PTX JIT contains unsupported call to cudaDeviceSynchronize."
    ),
    226: (
        "This indicates that an exception occurred on the device that is now contained by the GPU's error "
        "containment capability. Common causes are - a. Certain types of invalid accesses of peer GPU memory "
        "over nvlink b. Certain classes of hardware errors This leaves the process in an inconsistent state "
        "and any further CUDA work will return the same error. To continue using CUDA, the process must be "
        "terminated and relaunched."
    ),
    300: (
        "This indicates that the device kernel source is invalid. This includes compilation/linker errors "
        "encountered in device code or user error."
    ),
    301: "This indicates that the file specified was not found.",
    302: "This indicates that a link to a shared object failed to resolve.",
    303: "This indicates that initialization of a shared object failed.",
    304: "This indicates that an OS call failed.",
    400: (
        "This indicates that a resource handle passed to the API call was not valid. Resource handles are "
        "opaque types like CUstream and CUevent."
    ),
    401: (
        "This indicates that a resource required by the API call is not in a valid state to perform the "
        "requested operation."
    ),
    402: (
        "This indicates an attempt was made to introspect an object in a way that would discard semantically "
        "important information. This is either due to the object using funtionality newer than the API "
        "version used to introspect it or omission of optional return arguments."
    ),
    500: (
        "This indicates that a named symbol was not found. Examples of symbols are global/constant variable "
        "names, driver function names, texture names, and surface names."
    ),
    600: (
        "This indicates that asynchronous operations issued previously have not completed yet. This result is"
        " not actually an error, but must be indicated differently than CUDA_SUCCESS (which indicates "
        "completion). Calls that may return this value include cuEventQuery() and cuStreamQuery()."
    ),
    700: (
        "While executing a kernel, the device encountered a load or store instruction on an invalid memory "
        "address. This leaves the process in an inconsistent state and any further CUDA work will return the "
        "same error. To continue using CUDA, the process must be terminated and relaunched."
    ),
    701: (
        "This indicates that a launch did not occur because it did not have appropriate resources. This error"
        " usually indicates that the user has attempted to pass too many arguments to the device kernel, or "
        "the kernel launch specifies too many threads for the kernel's register count. Passing arguments of "
        "the wrong size (i.e. a 64-bit pointer when a 32-bit int is expected) is equivalent to passing too "
        "many arguments and can also result in this error."
    ),
    702: (
        "This indicates that the device kernel took too long to execute. This can only occur if timeouts are "
        "enabled - see the device attribute CU_DEVICE_ATTRIBUTE_KERNEL_EXEC_TIMEOUT for more information. "
        "This leaves the process in an inconsistent state and any further CUDA work will return the same "
        "error. To continue using CUDA, the process must be terminated and relaunched."
    ),
    703: "This error indicates a kernel launch that uses an incompatible texturing mode.",
    704: (
        "This error indicates that a call to cuCtxEnablePeerAccess() is trying to re-enable peer access to a "
        "context which has already had peer access to it enabled."
    ),
    705: (
        "This error indicates that cuCtxDisablePeerAccess() is trying to disable peer access which has not "
        "been enabled yet via cuCtxEnablePeerAccess()."
    ),
    708: "This error indicates that the primary context for the specified device has already been initialized.",
    709: (
        "This error indicates that the context current to the calling thread has been destroyed using "
        "cuCtxDestroy, or is a primary context which has not yet been initialized."
    ),
    710: (
        "A device-side assert triggered during kernel execution. The context cannot be used anymore, and must"
        " be destroyed. All existing device memory allocations from this context are invalid and must be "
        "reconstructed if the program is to continue using CUDA."
    ),
    711: (
        "This error indicates that the hardware resources required to enable peer access have been exhausted "
        "for one or more of the devices passed to cuCtxEnablePeerAccess()."
    ),
    712: ("This error indicates that the memory range passed to cuMemHostRegister() has already been registered."),
    713: (
        "This error indicates that the pointer passed to cuMemHostUnregister() does not correspond to any "
        "currently registered memory region."
    ),
    714: (
        "While executing a kernel, the device encountered a stack error. This can be due to stack corruption "
        "or exceeding the stack size limit. This leaves the process in an inconsistent state and any further "
        "CUDA work will return the same error. To continue using CUDA, the process must be terminated and "
        "relaunched."
    ),
    715: (
        "While executing a kernel, the device encountered an illegal instruction. This leaves the process in "
        "an inconsistent state and any further CUDA work will return the same error. To continue using CUDA, "
        "the process must be terminated and relaunched."
    ),
    716: (
        "While executing a kernel, the device encountered a load or store instruction on a memory address "
        "which is not aligned. This leaves the process in an inconsistent state and any further CUDA work "
        "will return the same error. To continue using CUDA, the process must be terminated and relaunched."
    ),
    717: (
        "While executing a kernel, the device encountered an instruction which can only operate on memory "
        "locations in certain address spaces (global, shared, or local), but was supplied a memory address "
        "not belonging to an allowed address space. This leaves the process in an inconsistent state and any "
        "further CUDA work will return the same error. To continue using CUDA, the process must be terminated"
        " and relaunched."
    ),
    718: (
        "While executing a kernel, the device program counter wrapped its address space. This leaves the "
        "process in an inconsistent state and any further CUDA work will return the same error. To continue "
        "using CUDA, the process must be terminated and relaunched."
    ),
    719: (
        "An exception occurred on the device while executing a kernel. Common causes include dereferencing an"
        " invalid device pointer and accessing out of bounds shared memory. Less common cases can be system "
        "specific - more information about these cases can be found in the system specific user guide. This "
        "leaves the process in an inconsistent state and any further CUDA work will return the same error. To"
        " continue using CUDA, the process must be terminated and relaunched."
    ),
    720: (
        "This error indicates that the number of blocks launched per grid for a kernel that was launched via "
        "either cuLaunchCooperativeKernel or cuLaunchCooperativeKernelMultiDevice exceeds the maximum number "
        "of blocks as allowed by cuOccupancyMaxActiveBlocksPerMultiprocessor or "
        "cuOccupancyMaxActiveBlocksPerMultiprocessorWithFlags times the number of multiprocessors as "
        "specified by the device attribute CU_DEVICE_ATTRIBUTE_MULTIPROCESSOR_COUNT."
    ),
    721: (
        "An exception occurred on the device while exiting a kernel using tensor memory: the tensor memory "
        "was not completely deallocated. This leaves the process in an inconsistent state and any further "
        "CUDA work will return the same error. To continue using CUDA, the process must be terminated and "
        "relaunched."
    ),
    800: "This error indicates that the attempted operation is not permitted.",
    801: "This error indicates that the attempted operation is not supported on the current system or device.",
    802: (
        "This error indicates that the system is not yet ready to start any CUDA work. To continue using "
        "CUDA, verify the system configuration is in a valid state and all required driver daemons are "
        "actively running. More information about this error can be found in the system specific user guide."
    ),
    803: (
        "This error indicates that there is a mismatch between the versions of the display driver and the "
        "CUDA driver. Refer to the compatibility documentation for supported versions."
    ),
    804: (
        "This error indicates that the system was upgraded to run with forward compatibility but the visible "
        "hardware detected by CUDA does not support this configuration. Refer to the compatibility "
        "documentation for the supported hardware matrix or ensure that only supported hardware is visible "
        "during initialization via the CUDA_VISIBLE_DEVICES environment variable."
    ),
    805: ("This error indicates that the MPS client failed to connect to the MPS control daemon or the MPS server."),
    806: ("This error indicates that the remote procedural call between the MPS server and the MPS client failed."),
    807: (
        "This error indicates that the MPS server is not ready to accept new MPS client requests. This error "
        "can be returned when the MPS server is in the process of recovering from a fatal failure."
    ),
    808: "This error indicates that the hardware resources required to create MPS client have been exhausted.",
    809: (
        "This error indicates the the hardware resources required to support device connections have been exhausted."
    ),
    810: (
        "This error indicates that the MPS client has been terminated by the server. To continue using CUDA, "
        "the process must be terminated and relaunched."
    ),
    811: (
        "This error indicates that the module is using CUDA Dynamic Parallelism, but the current "
        "configuration, like MPS, does not support it."
    ),
    812: (
        "This error indicates that a module contains an unsupported interaction between different versions of"
        " CUDA Dynamic Parallelism."
    ),
    900: "This error indicates that the operation is not permitted when the stream is capturing.",
    901: (
        "This error indicates that the current capture sequence on the stream has been invalidated due to a "
        "previous error."
    ),
    902: (
        "This error indicates that the operation would have resulted in a merge of two independent capture sequences."
    ),
    903: "This error indicates that the capture was not initiated in this stream.",
    904: ("This error indicates that the capture sequence contains a fork that was not joined to the primary stream."),
    905: (
        "This error indicates that a dependency would have been created which crosses the capture sequence "
        "boundary. Only implicit in-stream ordering dependencies are allowed to cross the boundary."
    ),
    906: ("This error indicates a disallowed implicit dependency on a current capture sequence from cudaStreamLegacy."),
    907: (
        "This error indicates that the operation is not permitted on an event which was last recorded in a "
        "capturing stream."
    ),
    908: (
        "A stream capture sequence not initiated with the CU_STREAM_CAPTURE_MODE_RELAXED argument to "
        "cuStreamBeginCapture was passed to cuStreamEndCapture in a different thread."
    ),
    909: "This error indicates that the timeout specified for the wait operation has lapsed.",
    910: (
        "This error indicates that the graph update was not performed because it included changes which "
        "violated constraints specific to instantiated graph update."
    ),
    911: (
        "This indicates that an async error has occurred in a device outside of CUDA. If CUDA was waiting for"
        " an external device's signal before consuming shared data, the external device signaled an error "
        "indicating that the data is not valid for consumption. This leaves the process in an inconsistent "
        "state and any further CUDA work will return the same error. To continue using CUDA, the process must"
        " be terminated and relaunched."
    ),
    912: "Indicates a kernel launch error due to cluster misconfiguration.",
    913: "Indiciates a function handle is not loaded when calling an API that requires a loaded function.",
    914: "This error indicates one or more resources passed in are not valid resource types for the operation.",
    915: "This error indicates one or more resources are insufficient or non-applicable for the operation.",
    916: "This error indicates that an error happened during the key rotation sequence.",
    999: "This indicates that an unknown internal error has occurred.",
}
