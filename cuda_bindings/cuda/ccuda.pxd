from cuda.bindings.cydriver cimport *

cdef extern from *:
    """
    #ifdef _MSC_VER
    #pragma message ( "The cuda.ccuda module is deprecated and will be removed in a future release, " \
                      "please switch to use the cuda.bindings.cydriver module instead." )
    #else
    #warning The cuda.ccuda module is deprecated and will be removed in a future release, \
             please switch to use the cuda.bindings.cydriver module instead.
    #endif
    """
