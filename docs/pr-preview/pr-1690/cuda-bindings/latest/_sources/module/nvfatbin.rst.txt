.. SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
.. SPDX-License-Identifier: LicenseRef-NVIDIA-SOFTWARE-LICENSE

.. default-role:: cpp:any

nvfatbin
========

Note
----

The nvfatbin bindings are not supported on nvFatbin installations <12.4. Ensure the installed CUDA toolkit's nvFatbin version is >=12.4.

The Tile IR API (:func:`cuda.bindings.nvfatbin.add_tile_ir`) is only available in CUDA 13.1+.

Functions
---------

NvFatbin defines the following functions for creating and populating fatbinaries.

.. autofunction:: cuda.bindings.nvfatbin.create
.. autofunction:: cuda.bindings.nvfatbin.destroy
.. autofunction:: cuda.bindings.nvfatbin.add_ptx
.. autofunction:: cuda.bindings.nvfatbin.add_cubin
.. autofunction:: cuda.bindings.nvfatbin.add_ltoir
.. autofunction:: cuda.bindings.nvfatbin.add_reloc
.. autofunction:: cuda.bindings.nvfatbin.add_tile_ir
.. autofunction:: cuda.bindings.nvfatbin.size
.. autofunction:: cuda.bindings.nvfatbin.get
.. autofunction:: cuda.bindings.nvfatbin.get_error_string
.. autofunction:: cuda.bindings.nvfatbin.version

Types
---------
.. autoclass:: cuda.bindings.nvfatbin.Result

    .. autoattribute:: cuda.bindings.nvfatbin.Result.SUCCESS


    .. autoattribute:: cuda.bindings.nvfatbin.Result.ERROR_INTERNAL


    .. autoattribute:: cuda.bindings.nvfatbin.Result.ERROR_ELF_ARCH_MISMATCH


    .. autoattribute:: cuda.bindings.nvfatbin.Result.ERROR_ELF_SIZE_MISMATCH


    .. autoattribute:: cuda.bindings.nvfatbin.Result.ERROR_MISSING_PTX_VERSION


    .. autoattribute:: cuda.bindings.nvfatbin.Result.ERROR_NULL_POINTER


    .. autoattribute:: cuda.bindings.nvfatbin.Result.ERROR_COMPRESSION_FAILED


    .. autoattribute:: cuda.bindings.nvfatbin.Result.ERROR_COMPRESSED_SIZE_EXCEEDED


    .. autoattribute:: cuda.bindings.nvfatbin.Result.ERROR_UNRECOGNIZED_OPTION


    .. autoattribute:: cuda.bindings.nvfatbin.Result.ERROR_INVALID_ARCH


    .. autoattribute:: cuda.bindings.nvfatbin.Result.ERROR_INVALID_NVVM


    .. autoattribute:: cuda.bindings.nvfatbin.Result.ERROR_EMPTY_INPUT


    .. autoattribute:: cuda.bindings.nvfatbin.Result.ERROR_MISSING_PTX_ARCH


    .. autoattribute:: cuda.bindings.nvfatbin.Result.ERROR_PTX_ARCH_MISMATCH


    .. autoattribute:: cuda.bindings.nvfatbin.Result.ERROR_MISSING_FATBIN


    .. autoattribute:: cuda.bindings.nvfatbin.Result.ERROR_INVALID_INDEX


    .. autoattribute:: cuda.bindings.nvfatbin.Result.ERROR_IDENTIFIER_REUSE


    .. autoattribute:: cuda.bindings.nvfatbin.Result.ERROR_INTERNAL_PTX_OPTION

