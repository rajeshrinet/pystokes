from __future__ import division
cimport cython
from libc.math cimport sqrt, pow, exp
from cython.parallel import prange

@cython.wraparound(False)
@cython.boundscheck(False)
@cython.cdivision(True)
@cython.nonecheck(False)
cdef class Forces:
    cdef int Np 

    cpdef lennardJones(self, double [:] F, double [:] r, double ljeps=?, double ljrmin=?)
   
    
    cpdef lennardJonesWall(self, double [:] F, double [:] r, double ljeps=?, double ljrmin=?, double wlje=?, double wljr=?)


    cpdef harmonicRepulsionPPPW(self, double [:] F, double [:] r, double partE=?, double partR=?, double wallE=?, double wallR=?)

    
    cpdef harmonicConfinement(self, double [:] F, double [:] r, double cn)
           

    cpdef opticalConfinement(self, double [:] F, double [:] r, double [:] r0, double [:] k)


    cpdef sedimentation(self, double [:] F, double g)


