import warnings as _warnings

from cuda.bindings.runtime import *


cdef extern from *:
    """
    #ifdef _MSC_VER
    #pragma message ( "The cuda.cudart module is deprecated and will be removed in a future release, " \
                      "please switch to use the cuda.bindings.runtime module instead." )
    #else
    #warning The cuda.cudart module is deprecated and will be removed in a future release, \
             please switch to use the cuda.bindings.runtime module instead.
    #endif
    """


_warnings.warn("The cuda.cudart module is deprecated and will be removed in a future release, "
               "please switch to use the cuda.bindings.runtime module instead.", FutureWarning, stacklevel=2)
