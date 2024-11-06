from cuda.bindings.cyruntime cimport *

cdef extern from *:
    """
    #pragma message ( "The cuda.ccudart module is deprecated and will be removed in a future release, " \
                      "please switch to use the cuda.bindings.cyruntime module instead." )
    """


from cuda.bindings import cyruntime
__pyx_capi__ = cyruntime.__pyx_capi__
del cyruntime
