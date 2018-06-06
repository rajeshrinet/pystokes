cimport cython
from scipy.io import savemat
from libc.math cimport sqrt, pow
from cython.parallel import prange
import numpy as np
cimport numpy as np
cimport pystokes.unbounded
DTYPE   = np.float
ctypedef np.float_t DTYPE_t

@cython.wraparound(False)
@cython.boundscheck(False)
@cython.cdivision(True)
@cython.nonecheck(False)
cdef class trap:
    def __init__(self, a, Np, vs, eta, dim, S0, D0, k, ljeps, ljrmin):
        self.a      = a  
        self.Np     = Np
        self.vs     = vs
        self.eta    = eta
        self.dim    = dim
        self.S0     = S0
        self.D0     = D0
        self.mu     = 1.0/(6*np.pi*self.eta*self.a)  
        self.k      = k 
        self.ljrmin = ljrmin
        self.ljeps  = ljeps

        self.uRbm = pystokes.unbounded.Rbm(self.a, self.Np, self.eta)
        
        # Memory allocations
        self.trapCentre      = np.zeros( 3*self.Np, dtype=DTYPE)
        self.Position        = np.zeros( 3*self.Np, dtype=DTYPE)
        self.Orientation     = np.zeros( 3*self.Np, dtype=DTYPE)
        self.Force           = np.zeros( 3*self.Np, dtype=DTYPE)
        self.Velocity        = np.zeros( 3*self.Np, dtype=DTYPE)
        self.AngularVelocity = np.zeros( 3*self.Np, dtype=DTYPE)
        self.rp0             = np.zeros( 6*self.Np, dtype=DTYPE)
        self.drpdt           = np.zeros( 6*self.Np, dtype=DTYPE)
        self.Stresslet       = np.zeros( 5*self.Np, dtype=DTYPE)
        self.PotDipole       = np.zeros( 3*self.Np, dtype=DTYPE)


    def initialise(self, initialConfig, trapCentre):
        self.rp0 = initialConfig
        self.trapCentre = trapCentre
        return
    
    
    cdef calcForce(self, double [:] F, double [:] r):
        cdef: 
            int Np = self.Np
            int i, j, xx = 2*Np
            double dx, dy, dz, dr, idr, idr2, idr3, Fdotdr, cn=self.k
            double ljx, ljy, ljz, ljfac, rminbyr, rmin = self.ljrmin, ljeps = self.ljeps
            double [:] trapCentre = self.trapCentre
        
        for i in prange(Np, nogil=True):
            ljx=0;  ljy=0;  ljz=0;
            for j in range(Np):
                if i != j:
                    dx = r[i]   - r[j]
                    dy = r[i+Np] - r[j+Np]
                    dz = r[i+xx] - r[j+xx] 
                    dr = sqrt( dx*dx + dy*dy + dz*dz )
                    if dr < rmin:
                        idr=1/dr;  idr2 =idr*idr ;
                        rminbyr  = rmin*idr 
                        ljfac  = ljeps*( rminbyr**12 - rminbyr**6 )* idr2
                        ljx += ljfac*dx
                        ljy += ljfac*dy
                        ljz += ljfac*dz
            F[i]   += ljx - cn*(r[i   ]-trapCentre[i   ]) 
            F[i+Np]+= ljy - cn*(r[i+Np]-trapCentre[i+Np]) 
            F[i+xx]+= ljz - cn*(r[i+xx]-trapCentre[i+xx]) 
        return 
    
    
    cdef rhs(self, rp):
        cdef: 
            int Np = self.Np, i, xx=2*Np, xx1=3*Np, xx2=4*Np, xx3=5*Np, xx4=6*Np
            double vs = self.vs, mu = self.mu
            double vs1=0.05*vs, S0=self.S0, D0=self.D0
            double mu1 = 0.75*mu
            double [:] r = rp[0:3*Np]        
            double [:] p = rp[3*Np:6*Np]       
            double [:] v = self.Velocity        
            double [:] o = self.AngularVelocity        
            double [:] F = self.Force        
            double [:] S = self.Stresslet        
            double [:] D = self.PotDipole        
            double [:] X = self.drpdt        
        
        for i in prange(Np, nogil =True):
            F[i]    = 0.0
            F[i+Np] = 0.0
            F[i+xx] = 0.0
            v[i]    = 0.0
            v[i+Np] = 0.0
            v[i+xx] = 0.0
            o[i]    = 0.0
            o[i+Np] = 0.0
            o[i+xx] = 0.0

            S[i]      = S0*(p[i]*p[i] -(1.0/3))
            S[i+ Np]  = S0*(p[i + Np]*p[i + Np] -(1.0/3))
            S[i+ xx]  = S0*(p[i]*p[i + Np])
            S[i+ xx1] = S0*(p[i]*p[i + xx])
            S[i+ xx2] = S0*(p[i + Np]*p[i + xx])
            
            D[i]     = D0*(p[i])
            D[i+ Np] = D0*(p[i + Np])
            D[i+ xx] = D0*(p[i + xx])

        self.calcForce(F, r)
        self.uRbm.stokesletV(v, r, F)
        self.uRbm.stokesletO(o, r, F)
        self.uRbm.stressletV(v, r, S)
        self.uRbm.stressletO(o, r, S)
        self.uRbm.potDipoleV(v, r, D)

        for i in prange(Np, nogil=True):
            '''Velocity, \dot{r} = vs p + \mu F + HI, for a sphere in Stokes flow'''
            X[i]    = vs*p[i   ] + v[i   ]
            X[i+Np] = vs*p[i+Np] + v[i+Np]
            X[i+xx] = vs*p[i+xx] + v[i+xx]
            #
            #'''Orientational Velocity, \dot{p} = \omega \times  p, for a sphere in Stokes flow'''
            X[i+3*Np] = o[i+Np]*p[i+xx] - o[i+xx]*p[i+Np]    
            X[i+4*Np] = o[i+xx]*p[i   ] - o[i   ]*p[i+xx]  
            X[i+5*Np] = o[i   ]*p[i+Np] - o[i+Np]*p[i   ]       
        return


    def simulate(self, Tf, Npts):
        T, N=Tf, Npts 
        dt = N/T
        print dt, 1.0*T/N
        
        X = np.zeros( (N+1, 6*self.Np), dtype=DTYPE)
        X[0, :] = self.rp0

        for i in range(N):
            self.rhs(X[i, :])
            X[i+1, :] =  X[i, :] + self.drpdt

        savemat('Np=%s_vs=%4.4f_K=%4.4f.mat'%(self.Np, self.vs, self.k), {'trapCentre':self.trapCentre, 'X':X, 't':dt, 'Np':self.Np,'k':self.k, 'vs':self.vs, 'S0':self.S0,})
        return
