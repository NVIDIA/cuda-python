from cuda.bindings.cynvrtc cimport *

cdef extern from *:
    """
    #pragma message ( "The cuda.cnvrtc module is deprecated and will be removed in a future release, " \
                      "please switch to use the cuda.bindings.cynvrtc module instead." )
    """
