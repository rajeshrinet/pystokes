cimport cython
from libc.math cimport sqrt, pow
from cython.parallel import prange
import numpy as np
cimport numpy as np
cimport pystokes.unbounded

@cython.wraparound(False)
@cython.boundscheck(False)
@cython.cdivision(True)
@cython.nonecheck(False)
cdef class trap:
    cdef:
        pystokes.unbounded.Rbm uRbm
        readonly int Np, dim
        readonly double a, eta, vs, S0, D0, mu, k , ljrmin, ljeps, pos2
        readonly np.ndarray Position, Orientation, Velocity, AngularVelocity, Force, Stresslet, PotDipole, drpdt, rp0, trapCentre

    cdef calcForce(self, double [:] F, double [:] r)
    
    cdef rhs(self, rp)
