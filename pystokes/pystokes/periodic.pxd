cimport cython
from cython.parallel import prange


@cython.wraparound(False)
@cython.boundscheck(False)
@cython.cdivision(True)
@cython.nonecheck(False)
cdef class Rbm:
    cdef int Np
    cdef double a, eta, L

    cpdef stokesletV(self, double [:] v, double [:] r, double [:] F, int Nb=?, int Nm=?)
    
    
    cpdef rotletV(   self, double [:] v, double [:] r, double [:] T, int Nb=?, int Nm=?)
    
    
    cpdef stressletV(self, double [:] v, double [:] r, double [:] S, int Nb=?, int Nm=?)


    cpdef potDipoleV(self, double [:] v, double [:] r, double [:] D, int Nb=?, int Nm=?)


    cpdef septletV(  self, double [:] v, double [:] r, double [:] G, int Nb=?, int Nm=?)

    
    cpdef vortletV(  self, double [:] v, double [:] r, double [:] V, int Nb=?, int Nm=?)


    cpdef spinletV(  self, double [:] v, double [:] r, double [:] M, int Nb=?, int Nm=?)


    ## Angular velocities


    cpdef stokesletO(self, double [:] o, double [:] r, double [:] F, int Nb=?, int Nm=?)


    cpdef rotletO(   self, double [:] o, double [:] r, double [:] T, int Nb=?, int Nm=?)

    
    cpdef stressletO(self, double [:] o, double [:] r, double [:] S, int Nb=?, int Nm=?)


    cpdef vortletO(  self, double [:] o, double [:] r, double [:] V, int Nb=?, int Nm=?)

    cpdef septletO(  self, double [:] o, double [:] r, double [:] G, int Nb=?, int Nm=?)


    cpdef spinletO(  self, double [:] o, double [:] r, double [:] M, int Nb=?, int Nm=?)



@cython.wraparound(False)
@cython.boundscheck(False)
@cython.cdivision(True)
@cython.nonecheck(False)
cdef class Flow:
    cdef double a
    cdef int Np, Nt
    cdef double L, eta

    cpdef stokesletV(self, double [:] v, double [:] rt, double [:] r, double [:] F, int Nb=?, int Nm=?)
    
    cpdef stressletV(self, double [:] v, double [:] rt, double [:] r, double [:] F, int Nb=?, int Nm=?)
    
    cpdef potDipoleV(self, double [:] v, double [:] rt, double [:] r, double [:] F, int Nb=?, int Nm=?)
