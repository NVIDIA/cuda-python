import warnings as _warnings

from cuda.bindings.runtime import *


cdef extern from *:
    """
    #pragma message ( "The cuda.cudart module is deprecated and will be removed in a future release, " \
                      "please switch to use the cuda.bindings.runtime module instead." )
    """


_warnings.warn("The cuda.cudart module is deprecated and will be removed in a future release, "
               "please switch to use the cuda.bindings.runtime module instead.", DeprecationWarning, stacklevel=2)
