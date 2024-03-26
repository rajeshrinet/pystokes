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
    cdef double a, eta, mu, muv, mur
    cdef int N 
    cdef readonly np.ndarray Mobility

    cpdef mobilityTT(self, double [:] v,  double [:] r, double [:] F)
               
   
    cpdef mobilityTR(self,    double [:] v,  double [:] r, double [:] T)
    
    
    cpdef propulsionT2s(self, double [:] v,  double [:] r, double [:] S)


    cpdef propulsionT3t(self, double [:] v,  double [:] r, double [:] D)


    cpdef propulsionT3a(self,   double [:] v,  double [:] r, double [:] V)


    cpdef propulsionT3s(self,   double [:] v,  double [:] r, double [:] G)


    cpdef propulsionT4a(self,   double [:] v,  double [:] r, double [:] M)


    ## Angular velocities


    cpdef mobilityRT(self, double [:] o,  double [:] r, double [:] F)

               
    cpdef mobilityRR(   self, double [:] o,  double [:] r, double [:] T)

    
    cpdef propulsionR2s(self, double [:] o,  double [:] r, double [:] S)
    
    
    cpdef propulsionR3a(  self, double [:] o,  double [:] r, double [:] V)


    cpdef propulsionR3s(  self, double [:] o,  double [:] r, double [:] G)


    cpdef propulsionR4a(  self, double [:] o,  double [:] r, double [:] M) 


    cpdef noiseTT(self, double [:] v, double [:] r)


    cpdef noiseRR(self, double [:] o, double [:] r)



@cython.wraparound(False)
@cython.boundscheck(False)
@cython.cdivision(True)
@cython.nonecheck(False)
cdef class Flow:
    cdef double a, eta
    cdef int N
    cdef int Nt

    cpdef flowField1s(self, double [:] vv, double [:] rt, double [:] r, double [:] F, double maskR=?)
               
    cpdef flowField2a(   self, double [:] vv, double [:] rt, double [:] r, double [:] T, double maskR=?)

    cpdef flowField2s(self, double [:] vv, double [:] rt, double [:] r, double [:] S, double maskR=?)


    cpdef flowField3t(self, double [:] vv, double [:] rt, double [:] r, double [:] D, double maskR=?)


    cpdef flowField3s(self,   double [:] vv, double [:] rt, double [:] r, double [:] G, double maskR=?)


    cpdef flowField3a(self,   double [:] vv, double [:] rt, double [:] r, double [:] V, double maskR=?)


    cpdef flowField4a(self,   double [:] vv, double [:] rt, double [:] r, double [:] M, double maskR=?)



