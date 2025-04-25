import warnings as _warnings

from cuda.bindings.driver import *


cdef extern from *:
    """
    #ifdef _MSC_VER
    #pragma message ( "The cuda.cuda module is deprecated and will be removed in a future release, " \
                      "please switch to use the cuda.bindings.driver module instead." )
    #else
    #warning The cuda.cuda module is deprecated and will be removed in a future release, \
             please switch to use the cuda.bindings.driver module instead.
    #endif
    """


_warnings.warn("The cuda.cuda module is deprecated and will be removed in a future release, "
               "please switch to use the cuda.bindings.driver module instead.", FutureWarning, stacklevel=2)
