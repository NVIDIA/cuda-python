import warnings as _warnings

from cuda.bindings.nvrtc import *


cdef extern from *:
    """
    #ifdef _MSC_VER
    #pragma message ( "The cuda.nvrtc module is deprecated and will be removed in a future release, " \
                      "please switch to use the cuda.bindings.nvrtc module instead." )
    #else
    #warning The cuda.nvrtc module is deprecated and will be removed in a future release, \
             please switch to use the cuda.bindings.nvrtc module instead.
    #endif
    """


_warnings.warn("The cuda.nvrtc module is deprecated and will be removed in a future release, "
               "please switch to use the cuda.bindings.nvrtc module instead.", FutureWarning, stacklevel=2)
