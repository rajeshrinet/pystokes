cimport cython
from libc.math cimport sqrt, pow
from cython.parallel import prange
import numpy as np
cimport numpy as np
cimport pystokes.unbounded
cimport pyforces.forceFields

@cython.wraparound(False)
@cython.boundscheck(False)
@cython.cdivision(True)
@cython.nonecheck(False)
cdef class Rbm:
    cdef:
        pystokes.unbounded.Rbm uRbm
        pyforces.forceFields.Forces ff
        readonly int Np, dim
        readonly double a, mu, ljrmin, ljeps, sForm,  tau, lmda1, lmda2, lmda3
        readonly np.ndarray Position, Orientation, Velocity, AngularVelocity, Force, drpdt, rp0, k

    cpdef initialise(self, np.ndarray rp0)
    
    cdef assignPosition(self, np.ndarray Position)
    
    cdef extForce(self, double [:] F, t)

    cdef rhs(self, rp, t)
