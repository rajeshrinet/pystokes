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
    def __init__(self, a_, Np_, vs_, eta_, dim_, S0_, D0_, k_, ljeps_, ljrmin_):
        self.a      = a_  
        self.Np     = Np_
        self.vs     = vs_
        self.eta    = eta_
        self.dim    = dim_
        self.S0     = S0_
        self.D0     = D0_
        self.mu     = 1.0/(6*np.pi*self.eta*self.a)  
        self.k      = k_ 
        self.ljrmin = ljrmin_
        self.ljeps  = ljeps_
        
        self.uRbm = pystokes.unbounded.Rbm(self.a, self.Np, self.eta)
        
        # Memory allocations
        self.Position        = np.zeros( 3*self.Np, dtype=DTYPE)
        self.Orientation     = np.zeros( 3*self.Np, dtype=DTYPE)
        self.Force           = np.zeros( 3*self.Np, dtype=DTYPE)
        self.Velocity        = np.zeros( 3*self.Np, dtype=DTYPE)
        self.AngularVelocity = np.zeros( 3*self.Np, dtype=DTYPE)
        self.rp0             = np.zeros( 6*self.Np, dtype=DTYPE)
        self.drpdt           = np.zeros( 6*self.Np, dtype=DTYPE)
        self.Stresslet       = np.zeros( 5*self.Np, dtype=DTYPE)
        self.PotDipole       = np.zeros( 3*self.Np, dtype=DTYPE)


    def initialise(self, surfaceDist='cube'):
        ''' this module sets up the initial configuration'''
        Np = self.Np;
        ss = np.sqrt(1.0/2) 
        if Np==2: 
            self.rp0[0], self.rp0[2], self.rp0[4]  = -40, 40, 0   # particle 1 Position
            self.rp0[1], self.rp0[3], self.rp0[5]  = 40 , 40, 0   # particle 2 Position
            
            self.rp0[6], self.rp0[8], self.rp0[10] = ss, -ss, 0   # Orientation
            self.rp0[7], self.rp0[9], self.rp0[11] = -ss, ss, 0   # Orientation
        else:
            if surfaceDist=='sphere':
                rr = (np.pi*self.vs*self.a)/self.k
                gAngle = np.pi*(3-np.sqrt(5))

                theta = gAngle*np.arange(Np)
                z = np.linspace(1 - 1.0/Np, 1.0/Np-1, Np) 
                radius = np.sqrt(1-z*z)
                
                self.rp0[0:Np]      = rr*radius*np.cos(theta)    
                self.rp0[Np:2*Np]   = rr*radius*np.sin(theta)    
                self.rp0[2*Np:3*Np] = rr*z   
                ## Orientation of all the particles
                self.rp0[3*Np:4*Np] = radius*np.cos(theta)    
                self.rp0[4*Np:5*Np] = radius*np.sin(theta)    
                self.rp0[5*Np:6*Np] = z   

            else:
                Np1d = int(np.round( (Np)**(1.0/3) ))
                Np= Np1d*Np1d*Np1d 
                rr = np.pi/(Np1d*self.k)
                for i in range(Np1d):
                    for j in range(Np1d):
                        for k in range(Np1d):
                            ii                = i*Np1d**2 + j*Np1d + k
                            self.rp0[ii]      = rr*(-Np1d + 2*i)                  
                            self.rp0[ii+Np]   = rr*(-Np1d + 2+ 2*j)               
                            self.rp0[ii+2*Np] = rr*(-Np1d + 2+ 2*k)               
                ## Orientation of all the particles
                self.rp0[3*Np:4*Np] = (1-2*np.random.random((Np)))
                self.rp0[4*Np:5*Np] = (1-2*np.random.random((Np)))
                self.rp0[5*Np:6*Np] = 0*(1-2*np.random.random((Np)))
                modp                = np.sqrt(self.rp0[3*Np:4*Np]*self.rp0[3*Np:4*Np] + self.rp0[4*Np:5*Np]*self.rp0[4*Np:5*Np] + self.rp0[5*Np:6*Np]*self.rp0[5*Np:6*Np])

                self.rp0[3*Np:4*Np] = self.rp0[3*Np:4*Np]/modp
                self.rp0[4*Np:5*Np] = self.rp0[4*Np:5*Np]/modp
                self.rp0[5*Np:6*Np] = self.rp0[5*Np:6*Np]/modp


            
            ## to plot the initial condition
            ##import matplotlib.pyplot as plt
            ##from mpl_toolkits.mplot3d import Axes3D
            ##fig = plt.figure(num=None, figsize=(15, 8), dpi=80, facecolor='w', edgecolor='k')
            ##x = self.rp0[0:Np] 
            ##y = self.rp0[Np:2*Np] 
            ##z = self.rp0[2*Np:3*Np] 
            ##ax = fig.add_subplot(111, projection='3d')
            ##scatCollection = ax.scatter(x, y, z, s=1, cmap=plt.cm.spectral )
            ##plt.show()
        return 

    
    
    cdef calcForce(self, double [:] F, double [:] r):
        cdef int Np = self.Np
        cdef int i, j, xx = 2*Np
        cdef double dx, dy, dz, dr, idr, idr2, idr3, Fdotdr, cn=self.k
        cdef double ljx, ljy, ljz, ljfac, rminbyr, rmin = self.ljrmin, ljeps = self.ljeps
        
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
            F[i]   += ljx-cn*r[i   ] 
            F[i+Np]+= ljy-cn*r[i+Np] 
            F[i+xx]+= ljz-cn*r[i+xx] 
        return 
    
    
    cdef rhs(self, rp):
        cdef: 
            int Np = self.Np, i, xx=2*Np, xx1=3*Np, xx2=4*Np, xx3=5*Np
            double vs = self.vs, S0=self.S0, D0=self.D0
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
        self.uRbm.stressletV(v, r, S)
        self.uRbm.potDipoleV(v, r, D)

        self.uRbm.stokesletO(o, r, F)
        self.uRbm.stressletO(o, r, S)
        
        for i in prange(Np, nogil=True):
            X[i]    = v[i   ] + vs*p[i   ] 
            X[i+Np] = v[i+Np] + vs*p[i+Np] 
            X[i+xx] = v[i+xx] + vs*p[i+xx] 
            
            X[i+xx1] = o[i+Np]*p[i+xx] - o[i+xx]*p[i+Np]    
            X[i+xx2] = o[i+xx]*p[i   ] - o[i   ]*p[i+xx]  
            X[i+xx3] = o[i   ]*p[i+Np] - o[i+Np]*p[i   ]       
        return


    def simulate(self, Tf, Npts):
        def rhs0(rp, t):
            self.rhs(rp)
            return self.drpdt
            
        # integrate the resulting equation using odespy
        T, N = Tf, Npts;  time_points = np.linspace(0, T, N+1);  ## intervals at which output is returned by integrator. 
        solver = odespy.Vode(rhs0, method = 'bdf', atol=1E-7, rtol=1E-6, order=5, nsteps=10**6)
        solver.set_initial_condition(self.rp0)
        u, t = solver.solve(time_points)
        savemat('Np=%s_vs=%4.4f_K=%4.4f_s_0=%4.4f.mat'%(self.Np, self.vs, self.k, self.S0), {'X':u, 't':t, 'Np':self.Np,'k':self.k, 'vs':self.vs, 'S0':self.S0,})
        
