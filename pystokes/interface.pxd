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

    cdef readonly int Nx, Ny, Nz, N
    cdef readonly np.ndarray fkx, fky, fkz, vkx, vky, vkz, fk0, fx0, Mobility
    cdef readonly double Lx, Ly, Lz, a, facx, facy, facz, eta, mu, muv, mur


    cpdef mobilityTT(self, double [:] v, double [:] r, double [:] F, double ll=?)


    cpdef mobilityTR(   self, double [:] v, double [:] r, double [:] T, double ll=?)


    cpdef propulsionT2s(self, double [:] v, double [:] r, double [:] S, double ll=?)


    cpdef propulsionT3t(self, double [:] v, double [:] r, double [:] D, double ll=?)



    cpdef mobilityRT(self, double [:] o, double [:] r, double [:] F, double ll=?)


    cpdef mobilityRR(self, double [:] o, double [:] r, double [:] T, double ll=?)
    
    
    cpdef propulsionR2s(self, double [:] o, double [:] r, double [:] S, double ll=?)
    
    
    cpdef propulsionR3t(self, double [:] o, double [:] r, double [:] D, double ll=?)


    cpdef noiseTT_old(self, double [:] v, double [:] r)

    cpdef noiseRR_old(self, double [:] o, double [:] r)
    
    
    cpdef noiseTT(self, double [:] v, double [:] r, double ll=?)
    
    cpdef noiseTR(self, double [:] v, double [:] r, double ll=?)
    
    cpdef noiseRT(self, double [:] o, double [:] r, double ll=?)
    
    cpdef noiseRR(self, double [:] o, double [:] r, double ll=?)



@cython.wraparound(False)
@cython.boundscheck(False)
@cython.cdivision(True)
@cython.nonecheck(False)
cdef class Flow:
    cdef double a, eta
    cdef int N
    cdef int Nt

    cpdef flowField1s(self, double [:] vv, double [:] rt, double [:] r, double [:] F)

    cpdef flowField2a(  self, double [:] vv, double [:] rt, double [:] r, double [:] T)

    cpdef flowField2s(self, double [:] vv, double [:] rt, double [:] r, double [:] S)

    cpdef flowField3t(self, double [:] vv, double [:] rt, double [:] r, double [:] D)
    
