.. SPDX-FileCopyrightText: Copyright (c) 2021-2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
.. SPDX-License-Identifier: LicenseRef-NVIDIA-SOFTWARE-LICENSE

-----
nvrtc
-----

Error Handling
--------------

NVRTC defines the following enumeration type and function for API call error handling.

.. autoclass:: cuda.bindings.nvrtc.nvrtcResult

    .. autoattribute:: cuda.bindings.nvrtc.nvrtcResult.NVRTC_SUCCESS


    .. autoattribute:: cuda.bindings.nvrtc.nvrtcResult.NVRTC_ERROR_OUT_OF_MEMORY


    .. autoattribute:: cuda.bindings.nvrtc.nvrtcResult.NVRTC_ERROR_PROGRAM_CREATION_FAILURE


    .. autoattribute:: cuda.bindings.nvrtc.nvrtcResult.NVRTC_ERROR_INVALID_INPUT


    .. autoattribute:: cuda.bindings.nvrtc.nvrtcResult.NVRTC_ERROR_INVALID_PROGRAM


    .. autoattribute:: cuda.bindings.nvrtc.nvrtcResult.NVRTC_ERROR_INVALID_OPTION


    .. autoattribute:: cuda.bindings.nvrtc.nvrtcResult.NVRTC_ERROR_COMPILATION


    .. autoattribute:: cuda.bindings.nvrtc.nvrtcResult.NVRTC_ERROR_BUILTIN_OPERATION_FAILURE


    .. autoattribute:: cuda.bindings.nvrtc.nvrtcResult.NVRTC_ERROR_NO_NAME_EXPRESSIONS_AFTER_COMPILATION


    .. autoattribute:: cuda.bindings.nvrtc.nvrtcResult.NVRTC_ERROR_NO_LOWERED_NAMES_BEFORE_COMPILATION


    .. autoattribute:: cuda.bindings.nvrtc.nvrtcResult.NVRTC_ERROR_NAME_EXPRESSION_NOT_VALID


    .. autoattribute:: cuda.bindings.nvrtc.nvrtcResult.NVRTC_ERROR_INTERNAL_ERROR


    .. autoattribute:: cuda.bindings.nvrtc.nvrtcResult.NVRTC_ERROR_TIME_FILE_WRITE_FAILED


    .. autoattribute:: cuda.bindings.nvrtc.nvrtcResult.NVRTC_ERROR_NO_PCH_CREATE_ATTEMPTED


    .. autoattribute:: cuda.bindings.nvrtc.nvrtcResult.NVRTC_ERROR_PCH_CREATE_HEAP_EXHAUSTED


    .. autoattribute:: cuda.bindings.nvrtc.nvrtcResult.NVRTC_ERROR_PCH_CREATE


    .. autoattribute:: cuda.bindings.nvrtc.nvrtcResult.NVRTC_ERROR_CANCELLED


    .. autoattribute:: cuda.bindings.nvrtc.nvrtcResult.NVRTC_ERROR_TIME_TRACE_FILE_WRITE_FAILED

.. autofunction:: cuda.bindings.nvrtc.nvrtcGetErrorString

General Information Query
-------------------------

NVRTC defines the following function for general information query.

.. autofunction:: cuda.bindings.nvrtc.nvrtcVersion
.. autofunction:: cuda.bindings.nvrtc.nvrtcGetNumSupportedArchs
.. autofunction:: cuda.bindings.nvrtc.nvrtcGetSupportedArchs

Compilation
-----------

NVRTC defines the following type and functions for actual compilation.

.. autoclass:: cuda.bindings.nvrtc.nvrtcProgram
.. autofunction:: cuda.bindings.nvrtc.nvrtcCreateProgram
.. autofunction:: cuda.bindings.nvrtc.nvrtcDestroyProgram
.. autofunction:: cuda.bindings.nvrtc.nvrtcCompileProgram
.. autofunction:: cuda.bindings.nvrtc.nvrtcGetPTXSize
.. autofunction:: cuda.bindings.nvrtc.nvrtcGetPTX
.. autofunction:: cuda.bindings.nvrtc.nvrtcGetCUBINSize
.. autofunction:: cuda.bindings.nvrtc.nvrtcGetCUBIN
.. autofunction:: cuda.bindings.nvrtc.nvrtcGetLTOIRSize
.. autofunction:: cuda.bindings.nvrtc.nvrtcGetLTOIR
.. autofunction:: cuda.bindings.nvrtc.nvrtcGetOptiXIRSize
.. autofunction:: cuda.bindings.nvrtc.nvrtcGetOptiXIR
.. autofunction:: cuda.bindings.nvrtc.nvrtcGetProgramLogSize
.. autofunction:: cuda.bindings.nvrtc.nvrtcGetProgramLog
.. autofunction:: cuda.bindings.nvrtc.nvrtcAddNameExpression
.. autofunction:: cuda.bindings.nvrtc.nvrtcGetLoweredName
.. autofunction:: cuda.bindings.nvrtc.nvrtcSetFlowCallback

Precompiled header (PCH) (CUDA 12.8+)
-------------------------------------

NVRTC defines the following function related to PCH. Also see PCH related flags passed to nvrtcCompileProgram.

.. autofunction:: cuda.bindings.nvrtc.nvrtcGetPCHHeapSize
.. autofunction:: cuda.bindings.nvrtc.nvrtcSetPCHHeapSize
.. autofunction:: cuda.bindings.nvrtc.nvrtcGetPCHCreateStatus
.. autofunction:: cuda.bindings.nvrtc.nvrtcGetPCHHeapSizeRequired

Supported Compile Options
-------------------------

NVRTC supports the compile options below. Option names with two preceding dashs (``--``\ ) are long option names and option names with one preceding dash (``-``\ ) are short option names. Short option names can be used instead of long option names. When a compile option takes an argument, an assignment operator (``=``\ ) is used to separate the compile option argument from the compile option name, e.g., ``"--gpu-architecture=compute_100"``\ . Alternatively, the compile option name and the argument can be specified in separate strings without an assignment operator, .e.g, ``"--gpu-architecture"``\  ``"compute_100"``\ . Single-character short option names, such as ``-D``\ , ``-U``\ , and ``-I``\ , do not require an assignment operator, and the compile option name and the argument can be present in the same string with or without spaces between them. For instance, ``"-D=<def>"``\ , ``"-D<def>"``\ , and ``"-D <def>"``\  are all supported.



The valid compiler options are:





- Compilation targets





  - ``--gpu-architecture=<arch>``\  (``-arch``\ )

Specify the name of the class of GPU architectures for which the input must be compiled.











- Separate compilation / whole-program compilation





  - ``--device-c``\  (``-dc``\ )

Generate relocatable code that can be linked with other relocatable device code. It is equivalent to ``--relocatable-device-code=true``\ .







  - ``--device-w``\  (``-dw``\ )

Generate non-relocatable code. It is equivalent to ``--relocatable-device-code=false``\ .







  - ``--relocatable-device-code={true|false}``\  (``-rdc``\ )

Enable (disable) the generation of relocatable device code.







  - ``--extensible-whole-program``\  (``-ewp``\ )

Do extensible whole program compilation of device code.









- Debugging support





  - ``--device-debug``\  (``-G``\ )

Generate debug information. If ``--dopt``\  is not specified, then turns off all optimizations.







  - ``--generate-line-info``\  (``-lineinfo``\ )

Generate line-number information.









- Code generation





  - ``--dopt``\  ``on``\  (``-dopt``\ )







  - ``--dopt=on``\  

Enable device code optimization. When specified along with ``-G``\ , enables limited debug information generation for optimized device code (currently, only line number information). When ``-G``\  is not specified, ``-dopt=on``\  is implicit.







  - ``--Ofast-compile={0|min|mid|max}``\  (``-Ofc``\ )

Specify the fast-compile level for device code, which controls the tradeoff between compilation speed and runtime performance by disabling certain optimizations at varying levels.







  - ``--ptxas-options``\  <options> (``-Xptxas``\ )







  - ``--ptxas-options=<options>``\  

Specify options directly to ptxas, the PTX optimizing assembler.







  - ``--maxrregcount=<N>``\  (``-maxrregcount``\ )

Specify the maximum amount of registers that GPU functions can use. Until a function-specific limit, a higher value will generally increase the performance of individual GPU threads that execute this function. However, because thread registers are allocated from a global register pool on each GPU, a higher value of this option will also reduce the maximum thread block size, thereby reducing the amount of thread parallelism. Hence, a good maxrregcount value is the result of a trade-off. If this option is not specified, then no maximum is assumed. Value less than the minimum registers required by ABI will be bumped up by the compiler to ABI minimum limit.







  - ``--ftz={true|false}``\  (``-ftz``\ )

When performing single-precision floating-point operations, flush denormal values to zero or preserve denormal values.

``--use_fast_math``\  implies ``--ftz=true``\ .







  - ``--prec-sqrt={true|false}``\  (``-prec-sqrt``\ )

For single-precision floating-point square root, use IEEE round-to-nearest mode or use a faster approximation. ``--use_fast_math``\  implies ``--prec-sqrt=false``\ .







  - ``--prec-div={true|false}``\  (``-prec-div``\ ) For single-precision floating-point division and reciprocals, use IEEE round-to-nearest mode or use a faster approximation. ``--use_fast_math``\  implies ``--prec-div=false``\ .





    - Default: ``true``\  









  - ``--fmad={true|false}``\  (``-fmad``\ )

Enables (disables) the contraction of floating-point multiplies and adds/subtracts into floating-point multiply-add operations (FMAD, FFMA, or DFMA). ``--use_fast_math``\  implies ``--fmad=true``\ .







  - ``--use_fast_math``\  (``-use_fast_math``\ )

Make use of fast math operations. ``--use_fast_math``\  implies ``--ftz=true``\  ``--prec-div=false``\  ``--prec-sqrt=false``\  ``--fmad=true``\ .







  - ``--extra-device-vectorization``\  (``-extra-device-vectorization``\ )

Enables more aggressive device code vectorization in the NVVM optimizer.







  - ``--modify-stack-limit={true|false}``\  (``-modify-stack-limit``\ )

On Linux, during compilation, use ``setrlimit()``\  to increase stack size to maximum allowed. The limit is reset to the previous value at the end of compilation. Note: ``setrlimit()``\  changes the value for the entire process.







  - ``--dlink-time-opt``\  (``-dlto``\ )

Generate intermediate code for later link-time optimization. It implies ``-rdc=true``\ . Note: when this option is used the ``nvrtcGetLTOIR``\  API should be used, as PTX or Cubin will not be generated.







  - ``--gen-opt-lto``\  (``-gen-opt-lto``\ )

Run the optimizer passes before generating the LTO IR.







  - ``--optix-ir``\  (``-optix-ir``\ )

Generate OptiX IR. The Optix IR is only intended for consumption by OptiX through appropriate APIs. This feature is not supported with link-time-optimization (``-dlto``\ ).

Note: when this option is used the nvrtcGetOptiX API should be used, as PTX or Cubin will not be generated.







  - ``--jump-table-density=``\ [0-101] (``-jtd``\ )

Specify the case density percentage in switch statements, and use it as a minimal threshold to determine whether jump table(brx.idx instruction) will be used to implement a switch statement. Default value is 101. The percentage ranges from 0 to 101 inclusively.







  - ``--device-stack-protector={true|false}``\  (``-device-stack-protector``\ )

Enable (disable) the generation of stack canaries in device code.







  - ``--no-cache``\  (``-no-cache``\ )

Disable the use of cache for both ptx and cubin code generation.







  - ``--frandom-seed``\  (``-frandom-seed``\ )

The user specified random seed will be used to replace random numbers used in generating symbol names and variable names. The option can be used to generate deterministically identical ptx and object files. If the input value is a valid number (decimal, octal, or hex), it will be used directly as the random seed. Otherwise, the CRC value of the passed string will be used instead.









- Preprocessing





  - ``--define-macro=<def>``\  (``-D``\ )

``<def>``\  can be either ``<name>``\  or ``<name=definitions>``\ .







  - ``--undefine-macro=<def>``\  (``-U``\ )

Cancel any previous definition of ``<def>``\ .







  - ``--include-path=<dir>``\  (``-I``\ )

Add the directory ``<dir>``\  to the list of directories to be searched for headers. These paths are searched after the list of headers given to nvrtcCreateProgram.







  - ``--pre-include=<header>``\  (``-include``\ )

Preinclude ``<header>``\  during preprocessing.







  - ``--no-source-include``\  (``-no-source-include``\ )

The preprocessor by default adds the directory of each input sources to the include path. This option disables this feature and only considers the path specified explicitly.









- Language Dialect





  - ``--std={c++03|c++11|c++14|c++17|c++20}``\  (``-std``\ )

Set language dialect to C++03, C++11, C++14, C++17 or C++20







  - ``--builtin-move-forward={true|false}``\  (``-builtin-move-forward``\ )

Provide builtin definitions of ``std::move``\  and ``std::forward``\ , when C++11 or later language dialect is selected.







  - ``--builtin-initializer-list={true|false}``\  (``-builtin-initializer-list``\ )

Provide builtin definitions of ``std::initializer_list``\  class and member functions when C++11 or later language dialect is selected.









- Precompiled header support (CUDA 12.8+)





  - ``--pch``\  (``-pch``\ )

Enable automatic PCH processing.







  - ``--create-pch=<file-name>``\  (``-create-pch``\ )

Create a PCH file.







  - ``--use-pch=<file-name>``\  (``-use-pch``\ )

Use the specified PCH file.







  - ``--pch-dir=<directory-name>``\  (``-pch-dir``\ )

When using automatic PCH (``-pch``\ ), look for and create PCH files in the specified directory. When using explicit PCH (``-create-pch``\  or ``-use-pch``\ ), the directory name is prefixed before the specified file name, unless the file name is an absolute path name.







  - ``--pch-verbose={true|false}``\  (``-pch-verbose``\ )

In automatic PCH mode, for each PCH file that could not be used in current compilation, print the reason in the compilation log.







  - ``--pch-messages={true|false}``\  (``-pch-messages``\ )

Print a message in the compilation log, if a PCH file was created or used in the current compilation.







  - ``--instantiate-templates-in-pch={true|false}``\  (``-instantiate-templates-in-pch``\ )

Enable or disable instantiatiation of templates before PCH creation. Instantiating templates may increase the size of the PCH file, while reducing the compilation cost when using the PCH file (since some template instantiations can be skipped).









- Misc.





  - ``--disable-warnings``\  (``-w``\ )

Inhibit all warning messages.







  - ``--restrict``\  (``-restrict``\ )

Programmer assertion that all kernel pointer parameters are restrict pointers.







  - ``--device-as-default-execution-space``\  (``-default-device``\ )

Treat entities with no execution space annotation as ``device``\  entities.







  - ``--device-int128``\  (``-device-int128``\ )

Allow the ``__int128``\  type in device code. Also causes the macro ``CUDACC_RTC_INT128``\  to be defined.







  - ``--device-float128``\  (``-device-float128``\ )

Allow the ``__float128``\  and ``_Float128``\  types in device code. Also causes the macro ``D__CUDACC_RTC_FLOAT128__``\  to be defined.







  - ``--optimization-info=<kind>``\  (``-opt-info``\ )

Provide optimization reports for the specified kind of optimization. The following kind tags are supported:







  - ``--display-error-number``\  (``-err-no``\ )

Display diagnostic number for warning messages. (Default)







  - ``--no-display-error-number``\  (``-no-err-no``\ )

Disables the display of a diagnostic number for warning messages.







  - ``--diag-error=<error-number>``\ ,... (``-diag-error``\ )

Emit error for specified diagnostic message number(s). Message numbers can be separated by comma.







  - ``--diag-suppress=<error-number>``\ ,... (``-diag-suppress``\ )

Suppress specified diagnostic message number(s). Message numbers can be separated by comma.







  - ``--diag-warn=<error-number>``\ ,... (``-diag-warn``\ )

Emit warning for specified diagnostic message number(s). Message numbers can be separated by comma.







  - ``--brief-diagnostics={true|false}``\  (``-brief-diag``\ )

This option disables or enables showing source line and column info in a diagnostic. The ``--brief-diagnostics=true``\  will not show the source line and column info.







  - ``--time=<file-name>``\  (``-time``\ )

Generate a comma separated value table with the time taken by each compilation phase, and append it at the end of the file given as the option argument. If the file does not exist, the column headings are generated in the first row of the table. If the file name is '-', the timing data is written to the compilation log.







  - ``--split-compile=<number-of-threads>``\  (``-split-compile=<number-of-threads>``\ )

Perform compiler optimizations in parallel. Split compilation attempts to reduce compile time by enabling the compiler to run certain optimization passes concurrently. This option accepts a numerical value that specifies the maximum number of threads the compiler can use. One can also allow the compiler to use the maximum threads available on the system by setting ``--split-compile=0``\ . Setting ``--split-compile=1``\  will cause this option to be ignored.







  - ``--fdevice-syntax-only``\  (``-fdevice-syntax-only``\ )

Ends device compilation after front-end syntax checking. This option does not generate valid device code.







  - ``--minimal``\  (``-minimal``\ )

Omit certain language features to reduce compile time for small programs. In particular, the following are omitted:







  - ``--device-stack-protector``\  (``-device-stack-protector``\ )

Enable stack canaries in device code. Stack canaries make it more difficult to exploit certain types of memory safety bugs involving stack-local variables. The compiler uses heuristics to assess the risk of such a bug in each function. Only those functions which are deemed high-risk make use of a stack canary.







  - ``--fdevice-time-trace=<file-name>``\  (``-fdevice-time-trace=<file-name>``\ ) Enables the time profiler, outputting a JSON file based on given <file-name>. Results can be analyzed on chrome://tracing for a flamegraph visualization.

