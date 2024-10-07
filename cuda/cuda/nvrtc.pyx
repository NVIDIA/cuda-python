import warnings as _warnings

from cuda.bindings.nvrtc import *


cdef extern from *:
    """
    #pragma message ( "The cuda.nvrtc module is deprecated and will be removed in a future release, " \
                      "please switch to use the cuda.bindings.nvrtc module instead." )
    """


_warnings.warn("The cuda.nvrtc module is deprecated and will be removed in a future release, "
               "please switch to use the cuda.bindings.nvrtc module instead.", DeprecationWarning, stacklevel=2)
