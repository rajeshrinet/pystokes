"""Some utilities codes which do not fit anywhere else,
but are essential in simulations of active colloids,
and plotting flow and phoretic fields
"""

import  numpy as np
cimport numpy as np
import matplotlib.pyplot as plt
cimport cython
from libc.math cimport sqrt, pow, log
from cython.parallel import prange

@cython.wraparound(False)
@cython.boundscheck(False)
@cython.cdivision(True)
@cython.nonecheck(False)
cdef class SlipModes:
    cdef int N
    cpdef V2s(self, double [:] S, double [:] p, double S0=?)


cpdef irreducibleTensors(l, p, Y0=?)
