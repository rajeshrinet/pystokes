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
cdef class Mima3D:
       
    cdef readonly int Nx, Ny, Nz, Np
    cdef readonly np.ndarray fkx, fky, fkz, vkx, vky, vkz, fk0, fx0
    cdef readonly double Lx, Ly, Lz, a, facx, facy, facz

    cpdef flowField1s(self, np.ndarray v, double [:] r, double [:] F, double sigma=?, int NN=?)
    
    cpdef flowField2a(self, np.ndarray v, double [:] r, double [:] T, double sigma=?, int NN=?)
    
    cpdef flowField2s(self, np.ndarray v, double [:] r, double [:] S, double sigma=?, int NN=?)
    
    cpdef flowField3t(self, np.ndarray v, double [:] r, double [:] D, double sigma=?, int NN=?)
    
    cpdef flowField3s(self, np.ndarray v, double [:] r, double [:] G, double sigma=?, int NN=?)
    
    cpdef flowField3a(self, np.ndarray v, double [:] r, double [:] V, double sigma=?, int NN=?)
    
    cpdef flowField4a(self, np.ndarray v, double [:] r, double [:] M, double sigma=?, int NN=?)
    

    cpdef solve(self, np.ndarray v, np.ndarray f)
    
    
    cdef fourierFk(self, double sigma, int NN=?)

    cpdef interpolate(self, double [:] V, double [:] r, np.ndarray vv, double sigma=?, int NN=?)



@cython.wraparound(False)
@cython.boundscheck(False)
@cython.cdivision(True)
@cython.nonecheck(False)
cdef class Mima2D:
    cdef readonly int Nx, Ny, Np
    cdef readonly np.ndarray fx, fy, fkx, fky, vkx, vky, fx0, fk0, xm , ym , kxm, kym 
    cdef readonly double Lx, Ly, a, facx, facy

    cpdef flowField1s(self, np.ndarray v, double [:] r, double [:] F, double sigma=?, int NN=?)
    

    cpdef flowField2s(self, np.ndarray v, double [:] r, double [:] S, double sigma=?, int NN=?)
    
    
    cpdef flowField3t(self, np.ndarray v, double [:] r, double [:] D, double sigma=?, int NN=?)
    
    
    cpdef flowField3s(self, np.ndarray v, double [:] r, double [:] D, double sigma=?, int NN=?)


    cpdef solve(self, np.ndarray v, np.ndarray f)
    
    cdef fourierFk(self, double sigma, double scale)
    
    cpdef interpolate(self, double [:] V, double [:] r, np.ndarray vv, double sigma=?, int NN=?)

       
