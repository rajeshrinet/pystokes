cimport cython
from libc.math cimport sqrt, exp, pow, erfc, sin, cos
from cython.parallel import prange
import numpy as np
cimport numpy as np
cdef double PI = 3.14159265359

@cython.wraparound(False)
@cython.boundscheck(False)
@cython.nonecheck(False)
cdef class Flow:
       
    cdef readonly int Nx, Ny, Np
    cdef readonly np.ndarray fx, fy, fkx, fky, vkx, vky, fx0, fk0, xm , ym , kxm, kym 
    cdef readonly double Lx, Ly, a, facx, facy

    cpdef stokesletV(self, np.ndarray v, double [:] r, double [:] F, double sigma=?, int NN=?)
    

    cpdef stressletV(self, np.ndarray v, double [:] r, double [:] S, double sigma=?, int NN=?)
    
    
    cpdef potDipoleV(self, np.ndarray v, double [:] r, double [:] D, double sigma=?, int NN=?)
    
    
    cpdef septletV(self, np.ndarray v, double [:] r, double [:] D, double sigma=?, int NN=?)


    cpdef solve(self, np.ndarray v, np.ndarray f)
    
    cdef fourierFk(self, double sigma, double scale)
    
    cpdef interpolate(self, double [:] V, double [:] r, np.ndarray vv, double sigma=?, int NN=?)
