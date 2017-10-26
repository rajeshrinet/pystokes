cimport cython
from scipy.io import savemat
import odespy
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
    def __init__(self, a_, Np_, vs_, eta_, dim_, S0_, k_, ljeps_, ljrmin_):
        self.a      = a_  
        self.Np     = Np_
        self.vs     = vs_
        self.eta    = eta_
        self.dim    = dim_
        self.S0     = S0_
        self.mu     = 1.0/(6*np.pi*self.eta*self.a)  
        self.k      = k_ 
        self.ljrmin = ljrmin_
        self.ljeps  = ljeps_

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
            double vs1=0.05*vs, S0=self.S0      
            double mu1 = 0.75*mu
            double [:] r = rp[0:3*Np]        
            double [:] p = rp[3*Np:6*Np]       
            double [:] v = self.Velocity        
            double [:] o = self.AngularVelocity        
            double [:] F = self.Force        
            double [:] S = self.Stresslet        
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

        self.calcForce(F, r)
        self.uRbm.stokesletV(v, r, F)
        self.uRbm.stokesletO(o, r, F)
        #self.uRbm.stressletV(v, r, S)
        #self.uRbm.stressletO(o, r, S)

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


    def simulate(self, Tf, Npts, plotDynamics='no'):
        def rhs0(rp, t):
            self.rhs(rp)
            return self.drpdt
            
        # integrate the resulting equation using odespy
        T, N = Tf, Npts;  time_points = np.linspace(0, T, N+1);  ## intervals at which output is returned by integrator. 
        solver = odespy.Vode(rhs0, method = 'bdf', atol=1E-7, rtol=1E-6, order=5, nsteps=10**6)
        solver.set_initial_condition(self.rp0)
        u, t = solver.solve(time_points)
        savemat('Np=%s_vs=%4.4f_K=%4.4f_trapA=%4.4f.mat'%(self.Np, self.vs, self.k, np.abs(self.trapCentre[np.sqrt(self.Np)]-self.trapCentre[0])), {'trapCentre':self.trapCentre, 'X':u, 't':t, 'Np':self.Np,'k':self.k, 'vs':self.vs, 'S0':self.S0,})
        
        if plotDynamics=='x-t':
            import matplotlib.pyplot as plt
            ## plotting business
            ii = 0; xx = u[:, 0];   xx1 = u[:, 1];
            yy = u[:, 2];   yy1 = u[:, 3];
            plt.figure(figsize=(15, 8), dpi=80, facecolor='w', edgecolor='k')
            plt.plot(t[ii:],  xx[ii:], color="#a60628", label='particle 1', linewidth=2);
            plt.plot(t[ii:], xx1[ii:], color="#348abd", label='particle 2', linewidth=2);
            plt.plot(t[ii:],  yy[ii:], color="red");
            plt.plot(t[ii:], yy1[ii:], color="blue");
            plt.legend(loc='lower left', fontsize=20);
            plt.title("dynamics of active particles in a harmonic potential", fontsize=20);
            plt.show()
        
        elif plotDynamics=='snapshots':
            import matplotlib.pyplot as plt
            fig = plt.figure(figsize=(15, 8), dpi=80, facecolor='w', edgecolor='k')
            
            for tt in t[::4]: 
                x1 = u[tt, 0];    x2 = u[tt, 1];
                y1 = u[tt, 2];    y2 = u[tt, 3];
                px1 = u[tt, 6];   px2 = u[tt, 7];
                py1 = u[tt, 8];   py2 = u[tt, 9]
            
        #        plt.plot(x1, y1, 'o', color='#a60628')
        #        plt.plot(x2, y2, 'o', color='#348abd')
                plt.quiver(x1, y1, px1, py1, scale=60, color="#a60628")
                plt.quiver(x2, y2, px2, py2, scale=60, color="#348abd")
            
                plt.title('frame=%s'%tt)
                #plt.savefig('time= %04d.png'%(tt)) 
                plt.pause(0.001)
            plt.show()
        return
