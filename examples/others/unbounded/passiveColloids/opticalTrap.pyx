cimport cython
from scipy.signal import square
import odespy
from libc.math cimport sqrt, pow
from cython.parallel import prange
import numpy as np
cimport numpy as np
from scipy.io import savemat
cdef double PI = 3.14159265359

cimport pystokes.unbounded
cimport pyforces.forceFields


DTYPE   = np.float
ctypedef np.float_t DTYPE_t

@cython.wraparound(False)
@cython.boundscheck(False)
@cython.cdivision(True)
@cython.nonecheck(False)
cdef class Rbm:
    def __init__(self, a, Np, ljeps, ljrmin, sForm, tau, lmda1, lmda2, lmda3):
        self.a       = a 
        self.Np      = Np
        self.ljrmin  = ljrmin
        self.ljeps   = ljeps
        self.tau     = tau
        self.sForm   = sForm
        self.lmda1   = lmda1
        self.lmda2   = lmda2
        self.lmda3   = lmda3
        self.eta     = 1.0

        #class instantiations
        self.ff   = pyforces.forceFields.Forces(self.Np)
        self.uRbm = pystokes.unbounded.Rbm(self.a, self.Np, self.eta)
        
        # Memory allocations
        self.Position        = np.empty( 3*self.Np, dtype=DTYPE)
        self.Orientation     = np.empty( 3*self.Np, dtype=DTYPE)
        self.Force           = np.empty( 3*self.Np, dtype=DTYPE)
        self.Velocity        = np.empty( 3*self.Np, dtype=DTYPE)
        self.AngularVelocity = np.empty( 3*self.Np, dtype=DTYPE)
        self.rp0             = np.empty( 3*self.Np, dtype=DTYPE)
        self.drpdt           = np.empty( 3*self.Np, dtype=DTYPE)
        self.k               = np.empty( self.Np, dtype=DTYPE)


    cpdef initialise(self, np.ndarray rp0):
        self.rp0 = rp0
        self.k[0]   = self.lmda1
        self.k[1]   = self.lmda2
        return 
    
    cdef assignPosition(self, np.ndarray position):
        self.Position = position
        return

    cdef extForce(self, double [:] F, t):
        if self.sForm==1:
            F[0] += self.lmda3*square(2*np.pi*t/self.tau) 
            #F[3] += self.lmda3*square(2*np.pi*t/self.tau) 
        elif self.sForm==2:
            F[0] += self.lmda3*np.sin(2*np.pi*t/self.tau) 
            #F[3] += self.lmda3*np.sin(2*np.pi*t/self.tau) 

    cdef rhs(self, rp, t):
        cdef: 
            int Np = self.Np, i 
            double mu = self.mu, a=self.a      
            double [:] v  = self.Velocity        
            double [:] F  = self.Force        
            double [:] k  = self.k        
            double [:] r0 = self.rp0        
            double [:] X  = self.drpdt        
    
        for i in prange(3*Np, nogil =True):
            F[i] = 0.0
            v[i] = 0.0
        
        r = rp[0:3*Np]
        self.assignPosition(r)
        
        self.ff.opticalConfinement(F, r, r0, k)
        self.extForce(F, t)
        self.ff.lennardJones(F, r, self.ljeps, self.ljrmin)
        
        self.uRbm.stokesletV(v, r, F)
        
        for i in prange(0, 3*Np, 3, nogil=True):
            '''Velocity, \dot{r} = \mu F + HI, for a sphere in Stokes flow'''
            X[i]   = v[i  ]
            X[i+1] = v[i+1]
            X[i+2] = v[i+2]
        return


    def simulate(self, Tf, Npts, filename):
        def rhs0(rp, t):
            self.rhs(rp, t)
            return self.drpdt
            
        # integrate the resulting equation using odespy
        T, N = Tf, Npts;  time_points = np.linspace(0, T, N+1);  ## intervals at which output is returned by integrator. 
        solver = odespy.Vode(rhs0, method = 'bdf', atol=1E-7, rtol=1E-6, order=5, nsteps=10**6)
        solver.set_initial_condition(self.rp0)
        u, t = solver.solve(time_points)
        
        savemat('%s.mat'%(filename), {'X':u, 't':t, 'Np': self.Np, 'tau':self.tau, 'lmda1':self.lmda1, 'lmda2':self.lmda2, 'lmda3':self.lmda3})
        return 
