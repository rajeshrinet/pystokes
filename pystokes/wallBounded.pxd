cimport cython
from libc.math cimport sqrt, exp, pow, erfc, sin, cos
from cython.parallel import prange
import numpy as np
cimport numpy as np
cdef double PI = 3.14159265359

@cython.wraparound(False)
@cython.boundscheck(False)
@cython.cdivision(True)
@cython.nonecheck(False)
cdef class Rbm:

    cdef readonly int Nx, Ny, Nz, Np
    cdef readonly np.ndarray Mobility
    cdef readonly double Lx, Ly, Lz, a, facx, facy, facz, eta, mu


    cpdef mobilityTT(self, double [:] v, double [:] r, double [:] F)


    cpdef mobilityTR(   self, double [:] v, double [:] r, double [:] T)


    cpdef propulsionT2s(self, double [:] v, double [:] r, double [:] S)


    cpdef propulsionT3t(self, double [:] v, double [:] r, double [:] D)



    cpdef mobilityRT(self, double [:] o, double [:] r, double [:] F)


    cpdef mobilityRR(self, double [:] o, double [:] r, double [:] T)
    
    cpdef propulsionR2s(self, double [:] o, double [:] r, double [:] S)


    cpdef propulsionR3a(self, double [:] o, double [:] r, double [:] M)


    cpdef propulsionR4a(self, double [:] o, double [:] r, double [:] M)


    cpdef noiseTT(self, double [:] v, double [:] r)


    cpdef noiseRR(self, double [:] o, double [:] r)


## Flow at given points
@cython.boundscheck(False)
@cython.cdivision(True)
@cython.nonecheck(False)
@cython.wraparound(False)
cdef class Flow:
    cdef readonly double a, eta
    cdef readonly int Nt, Np


    cpdef flowField1s(self, double [:] vv, double [:] rt, double [:] r, double [:] F)

    cpdef flowField2a(  self, double [:] vv, double [:] rt, double [:] r, double [:] T)

    cpdef flowField2s(self, double [:] vv, double [:] rt, double [:] r, double [:] S)

    cpdef flowField3t(self, double [:] vv, double [:] rt, double [:] r, double [:] D)
    
