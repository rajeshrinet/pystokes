cimport cython
from libc.math cimport sqrt
from cython.parallel import prange
import numpy as np
cimport numpy as np
cdef double PI = 3.14159265359

@cython.wraparound(False)
@cython.boundscheck(False)
@cython.cdivision(True)
@cython.nonecheck(False)
cdef class PD:
    cdef double a, eta, gammaT, gammaR, mu, muv, mur
    cdef int N 
    cdef readonly np.ndarray Mobility

    cpdef frictionTT(self, double depsilon, double [:] v, double [:] r)
               
   
    cpdef frictionTR(self, double depsilon, double [:] v, double [:] o, double [:] r)
    
    
    cpdef frictionT2s(self, double depsilon, double [:] V1s, double [:] S, double [:] r)


    cpdef frictionT3t(self, double depsilon, double [:] V1s, double [:] D, double [:] r)


    ## Angular velocities


    cpdef frictionRT(self, double depsilon, double [:] v, double [:] o, double [:] r)

               
    cpdef frictionRR(self, double depsilon, double [:] o, double [:] r)