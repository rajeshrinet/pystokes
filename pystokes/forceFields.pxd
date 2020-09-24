cimport cython
from libc.math cimport sqrt, pow, exp
from cython.parallel import prange

@cython.wraparound(False)
@cython.boundscheck(False)
@cython.cdivision(True)
@cython.nonecheck(False)
cdef class Forces:
    cdef int Np

    cpdef lennardJones(self, double [:] F, double [:] r, double lje=?, double ljr=?)


    cpdef lennardJonesWall(self, double [:] F, double [:] r, double lje=?, double ljr=?, double wlje=?, double wljr=?)

    cpdef softSpringWall(self, double [:] F, double [:] r, double pk=?, double prmin=?, double prmax=?,double wlje=?,double wljr=?)
    
    cpdef staticHarmonic(self, double [:] F, double [:] r, double [:] rS, double pk=?, double prmin=?, double prmax=?,double a=?)
    
    cpdef staticlennardJones(self, double [:] F, double [:] r, double [:] rS, double lje=?, double ljr=?, double a=?)
    
    cpdef softSpringLJWall(self, double [:] F, double [:] r, double pk=?, double prmin=?, double prmax=?,
                         double lje = ?, double ljr = ?, double wlje=?, double wljr=?)


    cpdef harmonicRepulsionPPPW(self, double [:] F, double [:] r, double partE=?, double partR=?, double wallE=?, double wallR=?)


    cpdef lennardJonesXWall(self, double [:] F, double [:] r, double wlje=?, double wljr=?)


    cpdef harmonicConfinement(self, double [:] F, double [:] r, double cn)


    cpdef opticalConfinement(self, double [:] F, double [:] r, double [:] r0, double [:] k)


    cpdef sedimentation(self, double [:] F, double g)


    cpdef membraneConfinement(self, double [:] F, double [:] r, double cn, double r0)


    cpdef membraneBound(self, double [:] F, double [:] r, double cn, double r0)


    cpdef spring(self, double [:] F, double [:] r, double bondLength, double springModulus)


    cpdef multipolymers(self, int Nf, double [:] F, double [:] r, double bondLength, double springModulus, double bendModulus, double twistModulus)


    cpdef multiRingpolymers(self, int Nf, double [:] F, double [:] r, double bondLength, double springModulus, double bendModulus, double twistModulus)


    cpdef membraneSurface(self, int Nmx, int Nmy, double [:] F, double [:] r, double bondLength, double springModulus, double bendModulus )




@cython.wraparound(False)
@cython.boundscheck(False)
@cython.cdivision(True)
@cython.nonecheck(False)
cdef class Torques:
    cdef int Np

    cpdef bottomHeaviness(self, double [:] T, double [:] p, double bh=?)

