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
cdef class Rbm:
    cdef double a, eta, L, mu, muv, mur
    cdef int N 
    cdef readonly np.ndarray Mobility

    cpdef mobilityTT(self, double [:] v,  double [:] r, double [:] F)
               
   
    cpdef mobilityTR(self,    double [:] v,  double [:] r, double [:] T)
    
    
    cpdef propulsionT2s(self, double [:] v,  double [:] r, double [:] S)


    ## Angular velocities


    cpdef mobilityRT(self, double [:] o,  double [:] r, double [:] F)

               
    cpdef mobilityRR(   self, double [:] o,  double [:] r, double [:] T)

    
    cpdef propulsionR2s(self, double [:] o,  double [:] r, double [:] S)