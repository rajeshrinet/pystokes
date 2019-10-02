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
cdef class Phoresis:

    cdef readonly int Nx, Ny, Nz, Np
    cdef readonly np.ndarray Mobility
    cdef readonly double Lx, Ly, Lz, a, facx, facy, facz, D

    cpdef elastance00(self, double [:] C0, double [:] r, double [:] J0)
    cpdef elastance10(self, double [:] C1, double [:] r, double [:] J0)
    cpdef elastance11(self, double [:] C1, double [:] r, double [:] J1)




@cython.wraparound(False)
@cython.boundscheck(False)
@cython.cdivision(True)
@cython.nonecheck(False)
cdef class Field:

    cdef readonly int Nx, Ny, Nz, Np, Nt
    cdef readonly np.ndarray Mobility
    cdef readonly double Lx, Ly, Lz, a, facx, facy, facz, D


    cpdef phoreticField0(self, double [:] c, double [:] rt, double [:] r, double [:] J0)
    cpdef phoreticField1(self, double [:] c, double [:] rt, double [:] r, double [:] J1)
    cpdef phoreticField2(self, double [:] c, double [:] rt, double [:] r, double [:] J2)
