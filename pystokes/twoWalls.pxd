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
    cdef readonly np.ndarray fkx, fky, fkz, vkx, vky, vkz, fk0, fx0
    cdef readonly double Lx, Ly, Lz, b, facx, facy, facz, eta

    cpdef mobilityTT(self, double [:] v, double [:] r, double [:] F, double H)
    
    
    cpdef propulsionT2s(self, double [:] v, double [:] r, double [:] V2s, double H)
    
    
    cpdef propulsionT3t(self, double [:] v, double [:] r, double [:] V3t, double H)

    

    
@cython.wraparound(False)
@cython.boundscheck(False)
@cython.cdivision(True)
@cython.nonecheck(False)
cdef class Flow:
    cdef double b, eta
    cdef int N
    cdef int Nt

    cpdef flowField1s(self, double [:] vv, double [:] rt, double [:] r, double [:] F, double H)
    

    cpdef flowField2s(self, double [:] vv, double [:] rt, double [:] r, double [:] V2s, double H)
    

    cpdef flowField3t(self, double [:] vv, double [:] rt, double [:] r, double [:] V3t, double H)
    