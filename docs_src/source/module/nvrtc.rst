-----
nvrtc
-----

Error Handling
--------------

NVRTC defines the following enumeration type and function for API call error handling.

.. autoclass:: cuda.nvrtc.nvrtcResult

    .. autoattribute:: cuda.nvrtc.nvrtcResult.NVRTC_SUCCESS


    .. autoattribute:: cuda.nvrtc.nvrtcResult.NVRTC_ERROR_OUT_OF_MEMORY


    .. autoattribute:: cuda.nvrtc.nvrtcResult.NVRTC_ERROR_PROGRAM_CREATION_FAILURE


    .. autoattribute:: cuda.nvrtc.nvrtcResult.NVRTC_ERROR_INVALID_INPUT


    .. autoattribute:: cuda.nvrtc.nvrtcResult.NVRTC_ERROR_INVALID_PROGRAM


    .. autoattribute:: cuda.nvrtc.nvrtcResult.NVRTC_ERROR_INVALID_OPTION


    .. autoattribute:: cuda.nvrtc.nvrtcResult.NVRTC_ERROR_COMPILATION


    .. autoattribute:: cuda.nvrtc.nvrtcResult.NVRTC_ERROR_BUILTIN_OPERATION_FAILURE


    .. autoattribute:: cuda.nvrtc.nvrtcResult.NVRTC_ERROR_NO_NAME_EXPRESSIONS_AFTER_COMPILATION


    .. autoattribute:: cuda.nvrtc.nvrtcResult.NVRTC_ERROR_NO_LOWERED_NAMES_BEFORE_COMPILATION


    .. autoattribute:: cuda.nvrtc.nvrtcResult.NVRTC_ERROR_NAME_EXPRESSION_NOT_VALID


    .. autoattribute:: cuda.nvrtc.nvrtcResult.NVRTC_ERROR_INTERNAL_ERROR

.. autofunction:: cuda.nvrtc.nvrtcGetErrorString

General Information Query
-------------------------

NVRTC defines the following function for general information query.

.. autofunction:: cuda.nvrtc.nvrtcVersion
.. autofunction:: cuda.nvrtc.nvrtcGetNumSupportedArchs
.. autofunction:: cuda.nvrtc.nvrtcGetSupportedArchs

Compilation
-----------

NVRTC defines the following type and functions for actual compilation.

.. autoclass:: cuda.nvrtc.nvrtcProgram
.. autofunction:: cuda.nvrtc.nvrtcCreateProgram
.. autofunction:: cuda.nvrtc.nvrtcDestroyProgram
.. autofunction:: cuda.nvrtc.nvrtcCompileProgram
.. autofunction:: cuda.nvrtc.nvrtcGetPTXSize
.. autofunction:: cuda.nvrtc.nvrtcGetPTX
.. autofunction:: cuda.nvrtc.nvrtcGetCUBINSize
.. autofunction:: cuda.nvrtc.nvrtcGetCUBIN
.. autofunction:: cuda.nvrtc.nvrtcGetNVVMSize
.. autofunction:: cuda.nvrtc.nvrtcGetNVVM
.. autofunction:: cuda.nvrtc.nvrtcGetProgramLogSize
.. autofunction:: cuda.nvrtc.nvrtcGetProgramLog
.. autofunction:: cuda.nvrtc.nvrtcAddNameExpression
.. autofunction:: cuda.nvrtc.nvrtcGetLoweredName

Supported Compile Options
-------------------------

NVRTC supports the compile options below. Option names with two preceding dashs (``--``\ ) are long option names and option names with one preceding dash (``-``\ ) are short option names. Short option names can be used instead of long option names. When a compile option takes an argument, an assignment operator (``=``\ ) is used to separate the compile option argument from the compile option name, e.g., ``"--gpu-architecture=compute_60"``\ . Alternatively, the compile option name and the argument can be specified in separate strings without an assignment operator, .e.g, ``"--gpu-architecture"``\  ``"compute_60"``\ . Single-character short option names, such as ``-D``\ , ``-U``\ , and ``-I``\ , do not require an assignment operator, and the compile option name and the argument can be present in the same string with or without spaces between them. For instance, ``"-D=<def>"``\ , ``"-D<def>"``\ , and ``"-D <def>"``\  are all supported.



The valid compiler options are:





- Compilation targets





  - ``--gpu-architecture=<arch>``\  (``-arch``\ )



    Specify the name of the class of GPU architectures for which the input must be compiled.







    - Valid ``<arch>``\ s:





      - ``compute_35``\  







      - ``compute_37``\  







      - ``compute_50``\  







      - ``compute_52``\  







      - ``compute_53``\  







      - ``compute_60``\  







      - ``compute_61``\  







      - ``compute_62``\  







      - ``compute_70``\  







      - ``compute_72``\  







      - ``compute_75``\  







      - ``compute_80``\  







      - ``sm_35``\  







      - ``sm_37``\  







      - ``sm_50``\  







      - ``sm_52``\  







      - ``sm_53``\  







      - ``sm_60``\  







      - ``sm_61``\  







      - ``sm_62``\  







      - ``sm_70``\  







      - ``sm_72``\  







      - ``sm_75``\  







      - ``sm_80``\  









    - Default: ``compute_52``\  











- Separate compilation / whole-program compilation





  - ``--device-c``\  (``-dc``\ )



    Generate relocatable code that can be linked with other relocatable device code. It is equivalent to --relocatable-device-code=true.







  - ``--device-w``\  (``-dw``\ )



    Generate non-relocatable code. It is equivalent to ``--relocatable-device-code=false``\ .







  - ``--relocatable-device-code={true|false}``\  (``-rdc``\ )



    Enable (disable) the generation of relocatable device code.





    - Default: ``false``\  









  - ``--extensible-whole-program``\  (``-ewp``\ )



    Do extensible whole program compilation of device code.





    - Default: ``false``\  











- Debugging support





  - ``--device-debug``\  (``-G``\ )



    Generate debug information. If --dopt is not specified, then turns off all optimizations.







  - ``--generate-line-info``\  (``-lineinfo``\ )



    Generate line-number information.









- Code generation





  - ``--dopt``\  on (``-dopt``\ )









  - ``--dopt=on``\  



    Enable device code optimization. When specified along with '-G', enables limited debug information generation for optimized device code. When '-G' is not specified, '-dopt=on' is implicit.







  - ``--ptxas-options``\  <options> (``-Xptxas``\ )









  - ``--ptxas-options=<options>``\  



    Specify options directly to ptxas, the PTX optimizing assembler.







  - ``--maxrregcount=<N>``\  (``-maxrregcount``\ )



    Specify the maximum amount of registers that GPU functions can use. Until a function-specific limit, a higher value will generally increase the performance of individual GPU threads that execute this function. However, because thread registers are allocated from a global register pool on each GPU, a higher value of this option will also reduce the maximum thread block size, thereby reducing the amount of thread parallelism. Hence, a good maxrregcount value is the result of a trade-off. If this option is not specified, then no maximum is assumed. Value less than the minimum registers required by ABI will be bumped up by the compiler to ABI minimum limit.







  - ``--ftz={true|false}``\  (``-ftz``\ )



    When performing single-precision floating-point operations, flush denormal values to zero or preserve denormal values. ``--use_fast_math``\  implies ``--ftz=true``\ .





    - Default: ``false``\  









  - ``--prec-sqrt={true|false}``\  (``-prec-sqrt``\ )



    For single-precision floating-point square root, use IEEE round-to-nearest mode or use a faster approximation. ``--use_fast_math``\  implies ``--prec-sqrt=false``\ .





    - Default: ``true``\  









  - ``--prec-div={true|false}``\  (``-prec-div``\ )



    For single-precision floating-point division and reciprocals, use IEEE round-to-nearest mode or use a faster approximation. ``--use_fast_math``\  implies ``--prec-div=false``\ .





    - Default: ``true``\  









  - ``--fmad={true|false}``\  (``-fmad``\ )



    Enables (disables) the contraction of floating-point multiplies and adds/subtracts into floating-point multiply-add operations (FMAD, FFMA, or DFMA). ``--use_fast_math``\  implies ``--fmad=true``\ .





    - Default: ``true``\  









  - ``--use_fast_math``\  (``-use_fast_math``\ )



    Make use of fast math operations. ``--use_fast_math``\  implies ``--ftz=true``\  ``--prec-div=false``\  ``--prec-sqrt=false``\  ``--fmad=true``\ .







  - ``--extra-device-vectorization``\  (``-extra-device-vectorization``\ )



    Enables more aggressive device code vectorization in the NVVM optimizer.







  - ``--modify-stack-limit={true|false}``\  (``-modify-stack-limit``\ )



    On Linux, during compilation, use ``setrlimit()``\  to increase stack size to maximum allowed. The limit is reset to the previous value at the end of compilation. Note: ``setrlimit()``\  changes the value for the entire process.





    - Default: ``true``\  









  - ``--dlink-time-opt``\  (``-dlto``\ )



    Generate intermediate code for later link-time optimization. It implies ``-rdc=true``\ . 



    Note: when this is used the nvvmGetNVVM API should be used, as PTX or Cubin will not be generated.









- Preprocessing





  - ``--define-macro=<def>``\  (``-D``\ )



    ``<def>``\  can be either ``<name>``\  or ``<name=definitions>``\ .





    - ``<name>``\  



      Predefine ``<name>``\  as a macro with definition ``1``\ .







    - ``<name>=<definition>``\  



      The contents of ``<definition>``\  are tokenized and preprocessed as if they appeared during translation phase three in a ``#define``\  directive. In particular, the definition will be truncated by embedded new line characters.









  - ``--undefine-macro=<def>``\  (``-U``\ )



    Cancel any previous definition of ``<def>``\ .







  - ``--include-path=<dir>``\  (``-I``\ )



    Add the directory ``<dir>``\  to the list of directories to be searched for headers. These paths are searched after the list of headers given to nvrtcCreateProgram.







  - ``--pre-include=<header>``\  (``-include``\ )



    Preinclude ``<header>``\  during preprocessing.







  - ``--no-source-include``\  (``-no-source-include``\ ) The preprocessor by default adds the directory of each input sources to the include path. This option disables this feature and only considers the path specified explicitly.









- Language Dialect





  - ``--std={c++03|c++11|c++14|c++17|c++20}``\  (``-std={c++11|c++14|c++17|c++20}``\ )



    Set language dialect to C++03, C++11, C++14, C++17 or C++20







  - ``--builtin-move-forward={true|false}``\  (``-builtin-move-forward``\ )



    Provide builtin definitions of ``std::move``\  and ``std::forward``\ , when C++11 language dialect is selected.





    - Default: ``true``\  









  - ``--builtin-initializer-list={true|false}``\  (``-builtin-initializer-list``\ )



    Provide builtin definitions of ``std::initializer_list``\  class and member functions when C++11 language dialect is selected.





    - Default: ``true``\  











- Misc.





  - ``--disable-warnings``\  (``-w``\ )



    Inhibit all warning messages.







  - ``--restrict``\  (``-restrict``\ )



    Programmer assertion that all kernel pointer parameters are restrict pointers.







  - ``--device-as-default-execution-space``\  (``-default-device``\ )



    Treat entities with no execution space annotation as ``device``\  entities.







  - ``--device-int128``\  (``-device-int128``\ )



    Allow the ``__int128``\  type in device code. Also causes the macro ``CUDACC_RTC_INT128``\  to be defined.







  - ``--optimization-info=<kind>``\  (``-opt-info``\ )



    Provide optimization reports for the specified kind of optimization. The following kind tags are supported:





    - ``inline``\  : emit a remark when a function is inlined.









  - ``--version-ident={true|false}``\  (``-dQ``\ )



    Embed used compiler's version info into generated PTX/CUBIN





    - Default: ``false``\  









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

