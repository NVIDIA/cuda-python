from cuda.bindings.cydriver cimport *

cdef extern from *:
    """
    #pragma message ( "The cuda.ccuda module is deprecated and will be removed in a future release, " \
                      "please switch to use the cuda.bindings.cydriver module instead." )
    """


from cuda.bindings import cydriver
__pyx_capi__ = cydriver.__pyx_capi__
del cydriver
