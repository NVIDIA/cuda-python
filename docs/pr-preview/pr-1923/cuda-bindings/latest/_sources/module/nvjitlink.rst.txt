.. SPDX-FileCopyrightText: Copyright (c) 2024-2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
.. SPDX-License-Identifier: LicenseRef-NVIDIA-SOFTWARE-LICENSE

.. default-role:: cpp:any

nvjitlink
=========

Note
----

The nvjitlink bindings are not supported on nvJitLink installations <12.3. Ensure the installed CUDA toolkit's nvJitLink version is >=12.3.  

Functions
---------

NvJitLink defines the following functions for linking code objects and querying the info and error logs.

.. autofunction:: cuda.bindings.nvjitlink.create
.. autofunction:: cuda.bindings.nvjitlink.destroy
.. autofunction:: cuda.bindings.nvjitlink.add_data
.. autofunction:: cuda.bindings.nvjitlink.add_file
.. autofunction:: cuda.bindings.nvjitlink.complete
.. autofunction:: cuda.bindings.nvjitlink.get_linked_cubin_size
.. autofunction:: cuda.bindings.nvjitlink.get_linked_cubin
.. autofunction:: cuda.bindings.nvjitlink.get_linked_ptx_size
.. autofunction:: cuda.bindings.nvjitlink.get_linked_ptx
.. autofunction:: cuda.bindings.nvjitlink.get_error_log_size
.. autofunction:: cuda.bindings.nvjitlink.get_error_log
.. autofunction:: cuda.bindings.nvjitlink.get_info_log_size
.. autofunction:: cuda.bindings.nvjitlink.get_info_log
.. autofunction:: cuda.bindings.nvjitlink.version

Types
---------
.. autoclass:: cuda.bindings.nvjitlink.Result

    .. autoattribute:: cuda.bindings.nvjitlink.Result.SUCCESS


    .. autoattribute:: cuda.bindings.nvjitlink.Result.ERROR_UNRECOGNIZED_OPTION


    .. autoattribute:: cuda.bindings.nvjitlink.Result.ERROR_MISSING_ARCH


    .. autoattribute:: cuda.bindings.nvjitlink.Result.ERROR_INVALID_INPUT


    .. autoattribute:: cuda.bindings.nvjitlink.Result.ERROR_PTX_COMPILE


    .. autoattribute:: cuda.bindings.nvjitlink.Result.ERROR_NVVM_COMPILE


    .. autoattribute:: cuda.bindings.nvjitlink.Result.ERROR_INTERNAL


    .. autoattribute:: cuda.bindings.nvjitlink.Result.ERROR_THREADPOOL


    .. autoattribute:: cuda.bindings.nvjitlink.Result.ERROR_UNRECOGNIZED_INPUT


    .. autoattribute:: cuda.bindings.nvjitlink.Result.ERROR_FINALIZE


.. autoclass:: cuda.bindings.nvjitlink.InputType

    .. autoattribute:: cuda.bindings.nvjitlink.InputType.NONE


    .. autoattribute:: cuda.bindings.nvjitlink.InputType.CUBIN


    .. autoattribute:: cuda.bindings.nvjitlink.InputType.PTX


    .. autoattribute:: cuda.bindings.nvjitlink.InputType.LTOIR


    .. autoattribute:: cuda.bindings.nvjitlink.InputType.FATBIN


    .. autoattribute:: cuda.bindings.nvjitlink.InputType.OBJECT


    .. autoattribute:: cuda.bindings.nvjitlink.InputType.LIBRARY


    .. autoattribute:: cuda.bindings.nvjitlink.InputType.INDEX


    .. autoattribute:: cuda.bindings.nvjitlink.InputType.ANY
