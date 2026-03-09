.. SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
.. SPDX-License-Identifier: LicenseRef-NVIDIA-SOFTWARE-LICENSE

.. default-role:: cpp:any

nvvm
====

The ``cuda.bindings.nvvm`` Python module wraps the
`libNVVM C API <https://docs.nvidia.com/cuda/libnvvm-api/>`_.

Functions
---------

.. autofunction:: cuda.bindings.nvvm.version
.. autofunction:: cuda.bindings.nvvm.ir_version
.. autofunction:: cuda.bindings.nvvm.create_program
.. autofunction:: cuda.bindings.nvvm.add_module_to_program
.. autofunction:: cuda.bindings.nvvm.lazy_add_module_to_program
.. autofunction:: cuda.bindings.nvvm.compile_program
.. autofunction:: cuda.bindings.nvvm.verify_program
.. autofunction:: cuda.bindings.nvvm.get_compiled_result_size
.. autofunction:: cuda.bindings.nvvm.get_compiled_result
.. autofunction:: cuda.bindings.nvvm.get_program_log_size
.. autofunction:: cuda.bindings.nvvm.get_program_log

Types
-----

..
   The empty lines below are important!

.. autoclass:: cuda.bindings.nvvm.Result

    .. autoattribute:: cuda.bindings.nvvm.Result.SUCCESS

    .. autoattribute:: cuda.bindings.nvvm.Result.ERROR_OUT_OF_MEMORY

    .. autoattribute:: cuda.bindings.nvvm.Result.ERROR_PROGRAM_CREATION_FAILURE

    .. autoattribute:: cuda.bindings.nvvm.Result.ERROR_IR_VERSION_MISMATCH

    .. autoattribute:: cuda.bindings.nvvm.Result.ERROR_INVALID_INPUT

    .. autoattribute:: cuda.bindings.nvvm.Result.ERROR_INVALID_PROGRAM

    .. autoattribute:: cuda.bindings.nvvm.Result.ERROR_INVALID_IR

    .. autoattribute:: cuda.bindings.nvvm.Result.ERROR_INVALID_OPTION

    .. autoattribute:: cuda.bindings.nvvm.Result.ERROR_NO_MODULE_IN_PROGRAM

    .. autoattribute:: cuda.bindings.nvvm.Result.ERROR_COMPILATION

    .. autoattribute:: cuda.bindings.nvvm.Result.ERROR_CANCELLED
