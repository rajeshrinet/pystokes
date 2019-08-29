cimport cython
from libc.math cimport sqrt
from cython.parallel import prange


@cython.wraparound(False)
@cython.boundscheck(False)
@cython.cdivision(True)
@cython.nonecheck(False)
cdef class Rbm:
    cdef double a, eta
    cdef int Np

    cpdef stokesletV(self, double [:] v,  double [:] r, double [:] F)
               
   
    cpdef rotletV(self,    double [:] v,  double [:] r, double [:] T)
    
    
    cpdef stressletV(self, double [:] v,  double [:] r, double [:] S)


    cpdef potDipoleV(self, double [:] v,  double [:] r, double [:] D)


    cpdef vortletV(self,   double [:] v,  double [:] r, double [:] V)


    cpdef septletV(self,   double [:] v,  double [:] r, double [:] G)


    cpdef spinletV(self,   double [:] v,  double [:] r, double [:] M)


    ## Angular velocities


    cpdef stokesletO(self, double [:] o,  double [:] r, double [:] F)

               
    cpdef rotletO(   self, double [:] o,  double [:] r, double [:] T)

    
    cpdef stressletO(self, double [:] o,  double [:] r, double [:] S)
    
    
    cpdef vortletO(  self, double [:] o,  double [:] r, double [:] V)


    cpdef septletO(  self, double [:] o,  double [:] r, double [:] G)


    cpdef spinletO(  self, double [:] o,  double [:] r, double [:] M)

@cython.wraparound(False)
@cython.boundscheck(False)
@cython.cdivision(True)
@cython.nonecheck(False)
cdef class Flow:
    cdef double a, eta
    cdef int Np
    cdef int Nt

    cpdef stokesletV(self, double [:] vv, double [:] rt, double [:] r, double [:] F)
               
    cpdef rotletV(   self, double [:] vv, double [:] rt, double [:] r, double [:] T)

    cpdef stressletV(self, double [:] vv, double [:] rt, double [:] r, double [:] S)


    cpdef potDipoleV(self, double [:] vv, double [:] rt, double [:] r, double [:] D)


    cpdef septletV(self,   double [:] vv, double [:] rt, double [:] r, double [:] G)


    cpdef vortletV(self,   double [:] vv, double [:] rt, double [:] r, double [:] V)


    cpdef spinletV(self,   double [:] vv, double [:] rt, double [:] r, double [:] M)



