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
cdef class PD:

    cdef readonly int Nx, Ny, Nz, N
    cdef readonly np.ndarray Mobility
    cdef readonly double Lx, Ly, Lz, a, facx, facy, facz, eta, gammaT, gammaR, mu, muv, mur


    cpdef frictionTT(self, double depsilon, double [:] v, double [:] r)


    cpdef frictionTR(self, double depsilon, double [:] v, double [:] o, double [:] r)


    cpdef frictionT2s(self, double depsilon, double [:] V1s, double [:] S, double [:] r)


    cpdef frictionT3t(self, double depsilon, double [:] V1s, double [:] D, double [:] r)



    cpdef frictionRT(self, double depsilon, double [:]v, double [:] o, double [:] r)


    cpdef frictionRR(self, double depsilon, double [:] o, double [:] r)