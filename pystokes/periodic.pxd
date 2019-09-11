cimport cython
from cython.parallel import prange


@cython.wraparound(False)
@cython.boundscheck(False)
@cython.cdivision(True)
@cython.nonecheck(False)
cdef class Rbm:
    cdef int Np
    cdef double a, eta, L

    cpdef mobilityTT(self, double [:] v, double [:] r, double [:] F, int Nb=?, int Nm=?)
    
    
    cpdef mobilityTR(self, double [:] v, double [:] r, double [:] T, int Nb=?, int Nm=?)
    
    
    cpdef propulsionT2s(self, double [:] v, double [:] r, double [:] S, int Nb=?, int Nm=?)


    cpdef propulsionT3t(self, double [:] v, double [:] r, double [:] D, int Nb=?, int Nm=?)


    cpdef propulsionT3s(  self, double [:] v, double [:] r, double [:] G, int Nb=?, int Nm=?)

    
    cpdef propulsionT3a(  self, double [:] v, double [:] r, double [:] V, int Nb=?, int Nm=?)


    cpdef propulsionT4a(  self, double [:] v, double [:] r, double [:] M, int Nb=?, int Nm=?)


    ## Angular velocities


    cpdef mobilityRT(self, double [:] o, double [:] r, double [:] F, int Nb=?, int Nm=?)


    cpdef mobilityRR(self, double [:] o, double [:] r, double [:] T, int Nb=?, int Nm=?)

    
    cpdef propulsionR2s(self, double [:] o, double [:] r, double [:] S, int Nb=?, int Nm=?)


    cpdef propulsionR3a(self, double [:] o, double [:] r, double [:] V, int Nb=?, int Nm=?)

    cpdef propulsionR3s(self, double [:] o, double [:] r, double [:] G, int Nb=?, int Nm=?)


    cpdef propulsionR4a(self, double [:] o, double [:] r, double [:] M, int Nb=?, int Nm=?)



@cython.wraparound(False)
@cython.boundscheck(False)
@cython.cdivision(True)
@cython.nonecheck(False)
cdef class Flow:
    cdef double a
    cdef int Np, Nt
    cdef double L, eta

    cpdef flowField1s(self, double [:] v, double [:] rt, double [:] r, double [:] F, int Nb=?, int Nm=?)
    
    cpdef flowField2s(self, double [:] v, double [:] rt, double [:] r, double [:] F, int Nb=?, int Nm=?)
    
    cpdef flowField3t(self, double [:] v, double [:] rt, double [:] r, double [:] F, int Nb=?, int Nm=?)
