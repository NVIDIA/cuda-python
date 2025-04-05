from cuda.bindings.cyruntime cimport *
from cuda.bindings import cyruntime
__pyx_capi__ = cyruntime.__pyx_capi__
del cyruntime
