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
cdef class Flow:
       
    cdef readonly int Nx, Ny, Nz, Np
    cdef readonly np.ndarray fkx, fky, fkz, vkx, vky, vkz, fk0, fx0
    cdef readonly double Lx, Ly, Lz, a, facx, facy, facz

    cpdef stokesletV(self, np.ndarray v, double [:] r, double [:] F, double sigma=?, int NN=?)
    
    cpdef rotletV(self, np.ndarray v, double [:] r, double [:] T, double sigma=?, int NN=?)
    
    cpdef stressletV(self, np.ndarray v, double [:] r, double [:] S, double sigma=?, int NN=?)
    
    cpdef potDipoleV(self, np.ndarray v, double [:] r, double [:] D, double sigma=?, int NN=?)
    
    cpdef septletV(self, np.ndarray v, double [:] r, double [:] G, double sigma=?, int NN=?)
    
    cpdef vortletV(self, np.ndarray v, double [:] r, double [:] V, double sigma=?, int NN=?)
    
    cpdef spinletV(self, np.ndarray v, double [:] r, double [:] M, double sigma=?, int NN=?)
    

    cpdef solve(self, np.ndarray v, np.ndarray f)
    
    
    cdef fourierFk(self, double sigma, int NN=?)

    cpdef interpolate(self, double [:] V, double [:] r, np.ndarray vv, double sigma=?, int NN=?)
