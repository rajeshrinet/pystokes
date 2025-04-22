cimport cython
from libc.math cimport sqrt
from cython.parallel import prange
cdef double PI = 3.14159265359
cdef double sqrt8 = 2.82842712475
cdef double sqrt2 = 1.41421356237
import numpy as np
cimport numpy as np


@cython.wraparound(False)
@cython.boundscheck(False)
@cython.cdivision(True)
@cython.nonecheck(False)
cdef class Rbm:
    """
    Rigid body motion (RBM) - velocity and angular velocity
    
    Methods in this class update velocities or angular velocities 
    using the inputs of -  arrays of positions, velocity or angular velocity, 
    along with an array of forces or torques or a slip mode

    The array of velocity or angular velocities is then update by each method. 
    
    ...

    ----------
    radius: float
        Radius of the particles (a).    
    particles: int
        Number of particles (N)
    viscosity: float 
        Viscosity of the fluid (eta)

   """

    def __init__(self, radius=1, particles=1, viscosity=1.0):
        self.b   = radius
        self.N  = particles
        self.eta = viscosity
        self.mu  = 1.0/(6*PI*self.eta*self.b)
        self.muv = 1.0/(8*PI*self.eta)
        self.mur = 1.0/(8*PI*self.eta*self.b**3)

        self.Mobility = np.zeros( (3*self.N, 3*self.N), dtype=np.float64)


    cpdef mobilityTT(self, double [:] v, double [:] r, double [:] F):
        """
        Compute velocity due to body forces using :math:`v=\mu^{TT}\cdot F` 
        ...

        Parameters
        ----------
        v: np.array
            An array of velocities
            An array of size 3*N,
        r: np.array
            An array of positions
            An array of size 3*N,
        F: np.array
            An array of forces
            An array of size 3*N,
    
        Examples
        --------
        An example of the RBM 

        >>> import pystokes, numpy as np, matplotlib.pyplot as plt
        >>> # particle radius, self-propulsion speed, number and fluid viscosity
        >>> b, vs, N, eta = 1.0, 1.0, 128, 0.1

        >>> #initialise
        >>> r = pystokes.utils.initialCondition(N)  # initial random distribution of positions
        >>> p = np.zeros(3*N); p[2*N:3*N] = -1    # initial orientation of the colloids
        >>> 
        >>> rbm   = pystokes.unbounded.Rbm(radius=b, particles=N, viscosity=eta)
        >>> force = pystokes.forceFields.Forces(particles=N)
        >>> 
        >>> def rhs(rp):
        >>>     # assign fresh values at each time step
        >>>     r = rp[0:3*N];   p = rp[3*N:6*N]
        >>>     F, v, o = np.zeros(3*N), np.zeros(3*N), np.zeros(3*N)
        >>> 
        >>>     force.lennardJonesWall(F, r, lje=0.01, ljr=5, wlje=1.2, wljr=3.4)
        >>>     rbm.mobilityTT(v, r, F)
        >>>     return np.concatenate( (v,o) )
        >>> 
        >>> # simulate the resulting system
        >>> Tf, Nts = 150, 200
        >>> pystokes.utils.simulate(np.concatenate((r,p)), 
        >>>      Tf,Nts,rhs,integrator='odeint', filename='crystallization')
        """


        cdef int N  = self.N, i, j, Z=2*N
        cdef double dx, dy, dz, idr, idr2, vx, vy, vz, vv1, vv2, aa = (2.0*self.b*self.b)/3.0 
        cdef double mu=self.mu, muv=self.muv        
        
        for i in prange(N, nogil=True):
            vx=0; vy=0;   vz=0;
            for j in range(N):
                if i != j:
                    dx = r[i]   - r[j]
                    dy = r[i+N] - r[j+N]
                    dz = r[i+Z] - r[j+Z] 
                    idr = 1.0/sqrt( dx*dx + dy*dy + dz*dz )
                    idr2 = idr*idr
                    
                    vv1 = (1+aa*idr2)*idr 
                    vv2 = (1-3*aa*idr2)*( F[j]*dx + F[j+N]*dy + F[j+Z]*dz )*idr2*idr
                    vx += vv1*F[j]    + vv2*dx 
                    vy += vv1*F[j+N] + vv2*dy 
                    vz += vv1*F[j+Z] + vv2*dz 

            v[i]   += mu*F[i]   + muv*vx
            v[i+N] += mu*F[i+N] + muv*vy
            v[i+Z] += mu*F[i+Z] + muv*vz
        return 
               

    cpdef mobilityTR(self, double [:] v, double [:] r, double [:] T):
        """
        Compute velocity due to body torque using :math:`v=\mu^{TR}\cdot T` 
        ...

        Parameters
        ----------
        v: np.array
            An array of velocities
            An array of size 3*N,
        r: np.array
            An array of positions
            An array of size 3*N,
        T: np.array
            An array of torques
            An array of size 3*N,
        """


        cdef int N = self.N, i, j, Z=2*N 
        cdef double dx, dy, dz, idr, idr3, vx, vy, vz
        cdef double muv=self.muv       
        
        for i in prange(N, nogil=True):
            vx=0; vy=0;   vz=0;
            for j in range(N):
                if i != j:
                    dx = r[i]    - r[j]
                    dy = r[i+N] - r[j+N]
                    dz = r[i+Z] - r[j+Z] 
                    idr = 1.0/sqrt( dx*dx + dy*dy + dz*dz )
                    idr3 = idr*idr*idr
                    vx += (T[j+N]*dz - dy*T[j+Z])*idr3
                    vy += (T[j+Z]*dx - dz*T[j]   )*idr3
                    vz += (T[j]  *dy - dx*T[j+N] )*idr3

            v[i]   += muv*vx
            v[i+N] += muv*vy
            v[i+Z] += muv*vz
        return 

    
    cpdef propulsionT2s(self, double [:] v, double [:] r, double [:] V2s):
        """
        Compute velocity due to 2s mode of the slip :math:`v=\pi^{T,2s}\cdot V^{(2s)}` 
        ...

        Parameters
        ----------
        v: np.array
            An array of velocities
            An array of size 3*N,
        r: np.array
            An array of positions
            An array of size 3*N,
        V2s: np.array
            An array of 2s mode of the slip
            An array of size 5*N,
        """

        cdef int N = self.N, i, j, Z=2*N, Z1=3*N, Z2=4*N
        cdef double dx, dy, dz, dr, idr,  idr3
        cdef double aa=(self.b*self.b*8.0)/3.0, vv1, vv2, aidr2
        cdef double vx, vy, vz, 
        cdef double sxx, sxy, sxz, syz, syy, srr, srx, sry, srz, mus = -(28.0*self.b*self.b)/24 
 
        for i in prange(N, nogil=True):
            vx=0; vy=0;   vz=0;
            for j in range(N):
                if i != j:
                    sxx = V2s[j]
                    syy = V2s[j+N]
                    sxy = V2s[j+Z]
                    sxz = V2s[j+Z1]
                    syz = V2s[j+Z2]
                    dx = r[i]   - r[j]
                    dy = r[i+N] - r[j+N]
                    dz = r[i+Z] - r[j+Z] 
                    idr = 1.0/sqrt( dx*dx + dy*dy + dz*dz )
                    idr3 = idr*idr*idr      
                    aidr2  = aa*idr*idr
                    
                    srr = (sxx*(dx*dx-dz*dz) + syy*(dy*dy-dz*dz) +  2*sxy*dx*dy + 2*sxz*dx*dz  +  2*syz*dy*dz)*idr*idr
                    srx = sxx*dx +  sxy*dy + sxz*dz  
                    sry = sxy*dx +  syy*dy + syz*dz  
                    srz = sxz*dx +  syz*dy - (sxx+syy)*dz 
                    
                    vv1 = 3*(1-aidr2)*srr*idr3
                    vv2 = 1.2*aidr2*idr3
                    vx +=  vv1*dx + vv2*srx
                    vy +=  vv1*dy + vv2*sry
                    vz +=  vv1*dz + vv2*srz
            
            v[i]   += vx*mus
            v[i+N] += vy*mus
            v[i+Z] += vz*mus

        return 

    
    cpdef propulsionT3t(self, double [:] v, double [:] r, double [:] V3t):
        """
        Compute velocity due to 3t mode of the slip :math:`v=\pi^{T,3t}\cdot V^{(3t)}` 
        ...

        Parameters
        ----------
        v: np.array
            An array of velocities
            An array of size 3*N,
        r: np.array
            An array of positions
            An array of size 3*N,
        V3t: np.array
            An array of 3t mode of the slip
            An array of size 3*N,
        """

        cdef int N = self.N, i, j, Z=2*N  
        cdef double dx, dy, dz, idr, idr3, V3tdotidr, vx, vy, vz, mud = 3.0*self.b*self.b*self.b/5, mud1 = -1.0*(self.b**3)/5
 
        for i in prange(N, nogil=True):
            vx=0; vy=0;   vz=0; 
            for j in range(N):
                if i != j: 
                    dx = r[ i]  - r[j]
                    dy = r[i+N] - r[j+N]
                    dz = r[i+Z] - r[j+Z] 
                    idr = 1.0/sqrt( dx*dx + dy*dy + dz*dz )
                    idr3 = idr*idr*idr 
                    V3tdotidr = (V3t[j]*dx + V3t[j+N]*dy + V3t[j+Z]*dz)*idr*idr

                    vx += (V3t[j]   - 3.0*V3tdotidr*dx )*idr3
                    vy += (V3t[j+N] - 3.0*V3tdotidr*dy )*idr3
                    vz += (V3t[j+Z] - 3.0*V3tdotidr*dz )*idr3
            
            v[i]   += mud1*vx
            v[i+N] += mud1*vy
            v[i+Z] += mud1*vz
        return 

    

    cpdef propulsionT3a(self, double [:] v, double [:] r, double [:]  V3a):
        """
        Compute velocity due to 3a mode of the slip :math:`v=\pi^{T,3a}\cdot  V3a` 
        ...

        Parameters
        ----------
        v: np.array
            An array of velocities
            An array of size 3*N,
        r: np.array
            An array of positions
            An array of size 3*N,
        V3a: np.array
            An array of 3a mode of the slip
            An array of size 5*N,
        """

        cdef int N = self.N, i, j, Z=2*N 
        cdef double dx, dy, dz, idr, idr5, vxx, vyy, vxy, vxz, vyz, vrx, vry, vrz
        cdef double mud = 13.0*self.b*self.b*self.b/12
 
        for i in prange(N, nogil=True):
            for j in range(N):
                if i != j:
                    vxx = V3a[j]
                    vyy = V3a[j+N]
                    vxy = V3a[j+2*N]
                    vxz = V3a[j+3*N]
                    vyz = V3a[j+4*N]
                    dx = r[i]   - r[j]
                    dy = r[i+N] - r[j+N]
                    dz = r[i+Z] - r[j+Z] 
                    idr = 1.0/sqrt( dx*dx + dy*dy + dz*dz)
                    idr5 = idr*idr*idr*idr*idr
                    vrx = vxx*dx +  vxy*dy + vxz*dz  
                    vry = vxy*dx +  vyy*dy + vyz*dz  
                    vrz = vxz*dx +  vyz*dy - (vxx+vyy)*dz 

                    v[i]     += mud * (vry * dz - vrz * dy) * idr5
                    v[i+N]   += mud * (vrz * dx - vrx * dz) * idr5
                    v[i+2*N] += mud * (vrx * dy - vry * dx) * idr5
        return


    cpdef propulsionT3s(self, double [:] v, double [:] r, double [:] V3s):
        """
        Compute velocity due to 3s mode of the slip :math:`v=\pi^{T,3s}\cdot V^{(3s)}` 
        ...

        Parameters
        ----------
        v: np.array
            An array of velocities
            An array of size 3*N,
        r: np.array
            An array of positions
            An array of size 3*N,
        V3s: np.array
            An array of 3s mode of the slip
            An array of size 7*N,
        """

        cdef int N = self.N, i, j 
        cdef double dx, dy, dz, idr, idr5, idr7, aidr2, grrr, grrx, grry, grrz, gxxx, gyyy, gxxy, gxxz, gxyy, gxyz, gyyz
 
        for i in prange(N, nogil=True):
             for j in range(N):
                if i != j:
                    gxxx  = V3s[j]
                    gyyy = V3s[j+N]
                    gxxy  = V3s[j+2*N]
                    gxxz  = V3s[j+3*N]
                    gxyy = V3s[j+4*N]
                    gxyz = V3s[j+5*N]
                    gyyz = V3s[j+6*N]
                    dx = r[i]   - r[j]
                    dy = r[i+N] - r[j+N]
                    dz = r[i+2*N] - r[j+2*N] 
                    idr = 1.0/sqrt( dx*dx + dy*dy + dz*dz )
                    idr5 = idr*idr*idr*idr*idr      
                    idr7 = idr5*idr*idr     
                    aidr2 = (10.0/3)*self.b*self.b*idr*idr
                    
                    grrr = gxxx*dx*(dx*dx-3*dz*dz) + 3*gxxy*dy*(dx*dx-dz*dz) + gxxz*dz*(3*dx*dx-dz*dz) +\
                       3*gxyy*dx*(dy*dy-dz*dz) + 6*gxyz*dx*dy*dz + gyyy*dy*(dy*dy-3*dz*dz) +  gyyz*dz*(3*dy*dy-dz*dz) 
                    grrx = gxxx*(dx*dx-dz*dz) + gxyy*(dy*dy-dz*dz) + 2*gxxy*dx*dy + 2*gxxz*dx*dz  +  2*gxyz*dy*dz
                    grry = gxxy*(dx*dx-dz*dz) + gyyy*(dy*dy-dz*dz) + 2*gxyy*dx*dy + 2*gxyz*dx*dz  +  2*gyyz*dy*dz
                    grrz = gxxz*(dx*dx-dz*dz) + gyyz*(dy*dy-dz*dz) + 2*gxyz*dx*dy - 2*(gxxx+gxyy)*dx*dz  - 2*(gxxy+gyyy)*dy*dz
                  
                    v[i]      += 3*(1-(15.0/7)*aidr2)*grrx*idr5 - 15*(1-aidr2)*grrr*dx*idr7
                    v[i+N]   += 3*(1-(15.0/7)*aidr2)*grry*idr5 - 15*(1-aidr2)*grrr*dy*idr7
                    v[i+2*N] += 3*(1-(15.0/7)*aidr2)*grrz*idr5 - 15*(1-aidr2)*grrr*dz*idr7
        return

    

    cpdef propulsionT4a(self, double [:] v, double [:] r, double [:] V4a):
        """
        Compute velocity due to 4a mode of the slip :math:`v=\pi^{T,4a}\cdot V^{(4a)}` 
        ...

        Parameters
        ----------
        v: np.array
            An array of velocities
            An array of size 3*N,
        r: np.array
            An array of positions
            An array of size 3*N,
        V4a: np.array
            An array of 4a mode of the slip
            An array of size 7*N,
        """

        cdef int N = self.N, i, j 
        cdef double dx, dy, dz, idr, idr7
        cdef double mrrx, mrry, mrrz, mxxx, myyy, mxxy, mxxz, mxyy, mxyz, myyz
        cdef double mud = -363/8 * self.b ** 4
 
        for i in prange(N, nogil=True):
            for j in range(N):
                if i != j:
                    mxxx = V4a[j]
                    myyy = V4a[j+N]
                    mxxy = V4a[j+2*N]
                    mxxz = V4a[j+3*N]
                    mxyy = V4a[j+4*N]
                    mxyz = V4a[j+5*N]
                    myyz = V4a[j+6*N]
                    dx = r[i]   - r[j]
                    dy = r[i+N] - r[j+N]
                    dz = r[i+2*N] - r[j+2*N] 
                    idr = 1.0/sqrt( dx*dx + dy*dy + dz*dz )
                    idr7 = idr*idr*idr*idr*idr*idr*idr
                    mrrx = mxxx*(dx*dx-dz*dz) + mxyy*(dy*dy-dz*dz) +  2*mxxy*dx*dy + 2*mxxz*dx*dz  +  2*mxyz*dy*dz
                    mrry = mxxy*(dx*dx-dz*dz) + myyy*(dy*dy-dz*dz) +  2*mxyy*dx*dy + 2*mxyz*dx*dz  +  2*myyz*dy*dz
                    mrrz = mxxz*(dx*dx-dz*dz) + myyz*(dy*dy-dz*dz) +  2*mxyz*dx*dy - 2*(mxxx+mxyy)*dx*dz  - 2*(mxxy+myyy)*dy*dz
                    
                    v[i]     -= mud*( dy*mrrz - dz*mrry )*idr7
                    v[i+N]   -= mud*( dz*mrrx - dx*mrrz )*idr7
                    v[i+2*N] -= mud*( dx*mrry - dy*mrrx )*idr7
        return


    
    ## Angular velocities
    cpdef mobilityRT(self, double [:] o, double [:] r, double [:] F):
        """
        Compute angular velocity due to body forces using :math:`o=\mu^{RT}\cdot F` 
        ...

        Parameters
        ----------
        o: np.array
            An array of angular velocities
            An array of size 3*N,
        r: np.array
            An array of positions
            An array of size 3*N,
        F: np.array
            An array of forces
            An array of size 3*N,
        """

        cdef int N = self.N, i, j, Z=2*N 
        cdef double dx, dy, dz, idr, idr3, ox, oy, oz, muv=self.muv
 
        for i in prange(N, nogil=True):
            ox=0;   oy=0;   oz=0;
            for j in range(N):
                if i != j:
                    dx = r[i]    - r[j]
                    dy = r[i+N] - r[j+N]
                    dz = r[i+Z] - r[j+Z] 
                    idr = 1.0/sqrt( dx*dx + dy*dy + dz*dz )
                    idr3 = idr*idr*idr

                    ox += (F[j+N]*dz - F[j+Z]*dy )*idr3
                    oy += (F[j+Z]*dx - F[j]  *dz )*idr3
                    oz += (F[j]  *dy - F[j+N]*dx )*idr3
            
            o[i]   += muv*ox
            o[i+N] += muv*oy
            o[i+Z] += muv*oz
        return  


        
    cpdef mobilityRR(   self, double [:] o, double [:] r, double [:] T):
        """
        Compute angular velocity due to body torques using :math:`o=\mu^{RR}\cdot T` 
        ...

        Parameters
        ----------
        o: np.array
            An array of angular velocities
            An array of size 3*N,
        r: np.array
            An array of positions
            An array of size 3*N,
        T: np.array
            An array of forces
            An array of size 3*N,
        """

        cdef int N = self.N, i, j, Z=2*N 
        cdef double dx, dy, dz, idr, idr3, Tdotidr, ox, oy, oz, mur=self.mur, muv=self.muv 
 
        for i in prange(N, nogil=True):
            ox=0;   oy=0;   oz=0;
            for j in range(N):
                if i != j:
                    dx = r[i]   - r[j]
                    dy = r[i+N] - r[j+N]
                    dz = r[i+Z] - r[j+Z] 
                    idr = 1.0/sqrt( dx*dx + dy*dy + dz*dz )
                    idr3 = idr*idr*idr
                    Tdotidr = ( T[j]*dx + T[j+N]*dy + T[j+Z]*dz )*idr*idr

                    ox += ( T[j]   - 3*Tdotidr*dx )*idr3
                    oy += ( T[j+N] - 3*Tdotidr*dy )*idr3
                    oz += ( T[j+Z] - 3*Tdotidr*dz )*idr3
            
            o[i]   += mur*T[i]   - 0.5*muv*ox ##changed factor 0.5 here
            o[i+N] += mur*T[i+N] - 0.5*muv*oy
            o[i+Z] += mur*T[i+Z] - 0.5*muv*oz
        return  


    
    cpdef propulsionR2s(self, double [:] o, double [:] r, double [:] V2s):
        """
        Compute angular velocity due to 2s mode of the slip :math:`v=\pi^{R,2s}\cdot V^{(2s)}` 
        ...

        Parameters
        ----------
        o: np.array
            An array of angular velocities
            An array of size 3*N,
        r: np.array
            An array of positions
            An array of size 3*N,
        V2s: np.array
            An array of 2s mode of the slip
            An array of size 5*N,
        """

        cdef int N = self.N, i, j, Z=2*N 
        cdef double dx, dy, dz, idr, idr5, ox, oy, oz
        cdef double sxx, sxy, sxz, syz, syy, srr, srx, sry, srz, mus = -(28.0*self.b*self.b)/24
 
        for i in prange(N, nogil=True):
            ox=0;   oy=0;   oz=0;
            for j in range(N):
                if i != j:
                    sxx = V2s[j]
                    syy = V2s[j+N]
                    sxy = V2s[j+2*N]
                    sxz = V2s[j+3*N]
                    syz = V2s[j+4*N]
                    dx = r[i]     - r[j]
                    dy = r[i+N]   - r[j+N]
                    dz = r[i+2*N] - r[j+2*N] 
                    idr = 1.0/sqrt( dx*dx + dy*dy + dz*dz )
                    idr5 = idr*idr*idr*idr*idr      
                    srx = sxx*dx +  sxy*dy + sxz*dz  
                    sry = sxy*dx +  syy*dy + syz*dz  
                    srz = sxz*dx +  syz*dy - (sxx+syy)*dz 

                    ox += 3*(sry*dz - srz*dy )*idr5
                    oy += 3*(srz*dx - srx*dz )*idr5
                    oz += 3*(srx*dy - sry*dx )*idr5
                    
            o[i]   += ox*mus
            o[i+N]  += oy*mus
            o[i+Z] += oz*mus
        return                 
    
    
    cpdef propulsionR3a(  self, double [:] o, double [:] r, double [:] V3a):
        """
        Compute angular velocity due to 3a mode of the slip :math:`v=\pi^{R,3a}\cdot V^{(3a)}` 
        ...

        Parameters
        ----------
        o: np.array
            An array of angular velocities
            An array of size 3*N,
        r: np.array
            An array of positions
            An array of size 3*N,
        V: np.array
            An array of 3a mode of the slip
            An array of size 5*N,
        """

        cdef int N = self.N, i, j, Z=2*N 
        cdef double dx, dy, dz, idr, idr2, idr5, vxx, vyy, vxy, vxz, vyz, vrr, vrx, vry, vrz
        cdef double mud = 13.0 / 24.0 * self.b * self.b * self.b
 
        for i in prange(N, nogil=True):
             for j in range(N):
                if i != j:
                    vxx  = V3a[j]
                    vyy = V3a[j+N]
                    vxy = V3a[j+2*N]
                    vxz = V3a[j+3*N]
                    vyz = V3a[j+4*N]
                    dx  = r[i]  - r[j]
                    dy = r[i+N] - r[j+N]
                    dz = r[i+Z] - r[j+Z] 
                    idr = 1.0/sqrt( dx*dx + dy*dy + dz*dz )
                    idr5 = idr*idr*idr*idr*idr      
                    vrr = (vxx*(dx*dx-dz*dz) + vyy*(dy*dy-dz*dz) +  2*vxy*dx*dy + 2*vxz*dx*dz  +  2*vyz*dy*dz)*idr*idr
                    vrx = vxx*dx +  vxy*dy + vxz*dz  
                    vry = vxy*dx +  vyy*dy + vyz*dz  
                    vrz = vxz*dx +  vyz*dy - (vxx+vyy)*dz 

                    o[i]     +=  mud * (-2 * vrx + 5 * vrr * dx)*idr5
                    o[i+N]   +=  mud * (-2 * vry + 5 * vrr * dy) * idr5
                    o[i+2*N] +=  mud * (-2 * vrz + 5 * vrr * dz) * idr5
        return


    
    cpdef propulsionR3s(  self, double [:] o, double [:] r, double [:] V3s):
        """
        Compute angular velocity due to 3s mode of the slip :math:`v=\pi^{R,3s}\cdot V^{(3s)}` 
        ...

        Parameters
        ----------
        o: np.array
            An array of angular velocities
            An array of size 3*N,
        r: np.array
            An array of positions
            An array of size 3*N,
        V3s: np.array
            An array of 3s mode of the slip
            An array of size 7*N,
        """


        cdef int N = self.N, i, j 
        cdef double dx, dy, dz, idr, idr7, grrx, grry, grrz, gxxx, gyyy, gxxy, gxxz, gxyy, gxyz, gyyz
 
        for i in prange(N, nogil=True):
            for j in range(N):
                if i != j:
                    gxxx = V3s[j]
                    gyyy = V3s[j+N]
                    gxxy = V3s[j+2*N]
                    gxxz = V3s[j+3*N]
                    gxyy = V3s[j+4*N]
                    gxyz = V3s[j+5*N]
                    gyyz = V3s[j+6*N]
                    dx = r[i]   - r[j]
                    dy = r[i+N] - r[j+N]
                    dz = r[i+2*N] - r[j+2*N] 
                    idr = 1.0/sqrt( dx*dx + dy*dy + dz*dz )
                    idr7 = idr*idr*idr*idr*idr*idr*idr     
                    
                    grrx = gxxx*(dx*dx-dz*dz) + gxyy*(dy*dy-dz*dz) +  2*gxxy*dx*dy + 2*gxxz*dx*dz  +  2*gxyz*dy*dz
                    grry = gxxy*(dx*dx-dz*dz) + gyyy*(dy*dy-dz*dz) +  2*gxyy*dx*dy + 2*gxyz*dx*dz  +  2*gyyz*dy*dz
                    grrz = gxxz*(dx*dx-dz*dz) + gyyz*(dy*dy-dz*dz) +  2*gxyz*dx*dy - 2*(gxxx+gxyy)*dx*dz  - 2*(gxxy+gyyy)*dy*dz

                    o[i]      += 15*( dy*grrz - dz*grry )*idr7
                    o[i+N]   += 15*( dz*grrx - dx*grrz )*idr7
                    o[i+2*N] += 15*( dx*grry - dy*grrx )*idr7
        return                 


    
    cpdef propulsionR4a(  self, double [:] o, double [:] r, double [:] V4a):
        """
        Compute angular velocity due to 4a mode of the slip :math:`v=\pi^{R,4a}\cdot V^{(4a)}` 
        ...

        Parameters
        ----------
        o: np.array
            An array of angular velocities
            An array of size 3*N,
        r: np.array
            An array of positions
            An array of size 3*N,
        V4a: np.array
            An array of 4a mode of the slip
            An array of size 7*N,
        """


        cdef int N = self.N, i, j 
        cdef double dx, dy, dz, idr, idr7, idr9, mrrr, mrrx, mrry, mrrz, mxxx, myyy, mxxy, mxxz, mxyy, mxyz, myyz
        cdef double mud = 363 / 80 * self.b ** 4
 
        for i in prange(N, nogil=True):
             for j in range(N):
                if i != j:
                    mxxx = V4a[j]
                    myyy = V4a[j+N]
                    mxxy = V4a[j+2*N]
                    mxxz = V4a[j+3*N]
                    mxyy = V4a[j+4*N]
                    mxyz = V4a[j+5*N]
                    myyz = V4a[j+6*N]
                    dx = r[i]   - r[j]
                    dy = r[i+N] - r[j+N]
                    dz = r[i+2*N] - r[j+2*N] 
                    idr = 1.0/sqrt( dx*dx + dy*dy + dz*dz )
                    idr7 = idr*idr*idr*idr*idr*idr*idr      
                    idr9 = idr7*idr*idr     
                    
                    mrrr = mxxx*dx*(dx*dx-3*dz*dz) + 3*mxxy*dy*(dx*dx-dz*dz) + mxxz*dz*(3*dx*dx-dz*dz) +\
                       3*mxyy*dx*(dy*dy-dz*dz) + 6*mxyz*dx*dy*dz + myyy*dy*(dy*dy-3*dz*dz) +  myyz*dz*(3*dy*dy-dz*dz) 
                    mrrx = mxxx*(dx*dx-dz*dz) + mxyy*(dy*dy-dz*dz) +  2*mxxy*dx*dy + 2*mxxz*dx*dz  +  2*mxyz*dy*dz
                    mrry = mxxy*(dx*dx-dz*dz) + myyy*(dy*dy-dz*dz) +  2*mxyy*dx*dy + 2*mxyz*dx*dz  +  2*myyz*dy*dz
                    mrrz = mxxz*(dx*dx-dz*dz) + myyz*(dy*dy-dz*dz) +  2*mxyz*dx*dy - 2*(mxxx+mxyy)*dx*dz  - 2*(mxxy+myyy)*dy*dz
                  
                    o[i]     += mud * (15 * mrrx * idr7 - 35 * mrrr * dx * idr9)
                    o[i+N]   += mud * (15 * mrry * idr7 - 35 * mrrr * dy * idr9)
                    o[i+2*N] += mud * (15 * mrrz * idr7 - 35 * mrrr * dz * idr9)
        return




## Flow at given points
@cython.wraparound(False)
@cython.boundscheck(False)
@cython.cdivision(True)
@cython.nonecheck(False)
cdef class Flow:
    """
    Flow at given points
    
    ...

    Parameters
    ----------
    radius: float
        Radius of the particles.    
    particles: int
        Number of particles 
    viscosity: viscosity of the fluid 
    gridpoints: int 
        Number of grid points
    Examples
    --------
    An example of the RBM

    """


    def __init__(self, radius=1, particles=1, viscosity=1, gridpoints=32):
        self.b  = radius
        self.N = particles
        self.Nt = gridpoints
        self.eta= viscosity

    cpdef flowField1s(self, double [:] vv, double [:] rt, double [:] r, double [:] F, double maskR=1.0):
        """
        Compute flow field at field points due body forces
        ...

        Parameters
        ----------
        vv: np.array
            An array of flow at field points
            An array of size 3*Nt,
        rt: np.array
            An array of field points
            An array of size 3*Nt,
        r: np.array
            An array of positions
            An array of size 3*N,
        F: np.array
            An array of body force
            An array of size 3*N,
    
        Examples
        --------
        An example of the Flow field due to $1s$ mode of force per unit area

        >>> import pystokes, numpy as np, matplotlib.pyplot as plt
        >>> 
        >>> # particle radius, self-propulsion speed, number and fluid viscosity
        >>> b, eta, N = 1.0, 1.0/6.0, 1
        >>> 
        >>> # initialize
        >>> r, p = np.array([0.0, 0.0, 3.4]), np.array([0.0, 1.0, 0])
        >>> F1s  = pystokes.utils.irreducibleTensors(1, p)
        >>> 
        >>> # space dimension , extent , discretization
        >>> dim, L, Ng = 3, 10, 64;
        >>> 
        >>> # instantiate the Flow class
        >>> flow = pystokes.unbounded.Flow(radius=b, particles=N, viscosity=eta, gridpoints=Ng*Ng)
        >>> 
        >>> # create grid, evaluate flow and plot
        >>> rr, vv = pystokes.utils.gridXY(dim, L, Ng)
        >>> flow.flowField1s(vv, rr, r, F1s)
        >>> pystokes.utils.plotStreamlinesXY(vv, rr, r, offset=6-1, density=1.4, title='1s')
        """

        cdef int N = self.N,  Nt = self.Nt
        cdef int i, ii, Z = 2*N
        cdef double dx, dy, dz, dr, idr, idr2, vv1, vv2, vx, vy, vz, radi=self.b
        cdef double muv = 1/(8*PI*self.eta), aa = self.b*self.b/3.0
        for i in prange(Nt, nogil=True):
            vx = 0.0; vy = 0.0; vz = 0.0;
            for ii in range(N):
                
                dx = rt[i]      - r[ii]
                dy = rt[i+Nt]   - r[ii+N]
                dz = rt[i+2*Nt] - r[ii+Z] 
                dr = sqrt( dx*dx + dy*dy + dz*dz )
                if dr>maskR:
                    idr = 1.0/dr
                    idr2= idr*idr
                    vv1 = (1+aa*idr2)*idr 
                    vv2 = (1-3*aa*idr2)*( F[ii]*dx + F[ii+N]*dy + F[ii+Z]*dz )*idr2*idr

                    vx += vv1*F[ii]     + vv2*dx
                    vy += vv1*F[ii+ N] + vv2*dy
                    vz += vv1*F[ii+ Z] + vv2*dz
            
            vv[i]         += vx*muv
            vv[i +   Nt]  += vy*muv
            vv[i +  2*Nt] += vz*muv
        return 


    cpdef flowField2a(self, double [:] vv, double [:] rt, double [:] r, double [:] T, double maskR=1.0):
        """
        Compute flow field at field points due body Torque 
        ...

        Parameters
        ----------
        vv: np.array
            An array of flow at field points
            An array of size 3*Nt,
        rt: np.array
            An array of field points
            An array of size 3*Nt,
        r: np.array
            An array of positions
            An array of size 3*N,
        T: np.array
            An array of body torque
            An array of size 3*N,
    
        Examples
        --------
        An example of the RBM 

        # Example 1: Flow field due to $2a$ mode of force per unit area
        >>> import pystokes, numpy as np, matplotlib.pyplot as plt
        >>> 
        >>> # particle radius, self-propulsion speed, number and fluid viscosity
        >>> b, eta, N = 1.0, 1.0/6.0, 1
        >>> 
        >>> # initialize
        >>> r, p = np.array([0.0, 0.0, 3.4]), np.array([0.0, 1.0, 0])
        >>> V2a  = pystokes.utils.irreducibleTensors(1, p)
        >>> 
        >>> # space dimension , extent , discretization
        >>> dim, L, Ng = 3, 10, 64;
        >>> 
        >>> # instantiate the Flow class
        >>> flow = pystokes.wallBounded.Flow(radius=b, particles=N, viscosity=eta, gridpoints=Ng*Ng)
        >>> 
        >>> # create grid, evaluate flow and plot
        >>> rr, vv = pystokes.utils.gridXY(dim, L, Ng)
        >>> flow.flowField2a(vv, rr, r, V2a)
        >>> pystokes.utils.plotStreamlinesXY(vv, rr, r, offset=6-1, density=1.4, title='2s')
        """

        cdef int N = self.N, Nt = self.Nt
        cdef int i, ii, Z = 2*N
        cdef double dx, dy, dz, dr, idr, idr3, vx, vy, vz, mur1 = 1.0/(8*PI*self.eta), radi=self.b
        for i in prange(Nt, nogil=True):
            vx = 0.0; vy = 0.0; vz = 0.0;
            for ii in range(N):
                dx = rt[i]      - r[ii]
                dy = rt[i+Nt]   - r[ii+N]
                dz = rt[i+2*Nt] - r[ii+Z] 
                dr = sqrt( dx*dx + dy*dy + dz*dz )
                if dr>maskR:
                    idr = 1.0/dr                
                    idr3 = idr*idr*idr
          
                    vx += ( dy*T[ii+Z] - dz*T[ii+N])*idr3
                    vy += ( dz*T[ii]    - dx*T[ii+Z])*idr3 
                    vz += ( dx*T[ii+N] - dy*T[ii]   )*idr3

            vv[i  ]      += vx*mur1
            vv[i + Nt]   += vy*mur1
            vv[i + 2*Nt] += vz*mur1
        return  


    cpdef flowField2s(self, double [:] vv, double [:] rt, double [:] r, double [:] V2s, double maskR=1.0):
        """
        Compute flow field at field points  due to 2s mode of the slip 
        ...

        Parameters
        ----------
        vv: np.array
            An array of flow at field points
            An array of size 3*Nt,
        rt: np.array
            An array of field points
            An array of size 3*Nt,
        r: np.array
            An array of positions
            An array of size 3*N,
        V2s: np.array
            An array of 2s mode of the slip
            An array of size 5*N,
        
        Examples
        --------
        An example of the Flow field due to $3t$ mode of active slip

        >>> import pystokes, numpy as np, matplotlib.pyplot as plt
        >>> 
        >>> # particle radius, self-propulsion speed, number and fluid viscosity
        >>> b, eta, N = 1.0, 1.0/6.0, 1
        >>> 
        >>> # initialize
        >>> r, p = np.array([0.0, 0.0, 3.4]), np.array([0.0, 1.0, 0])
        >>> V3t  = pystokes.utils.irreducibleTensors(1, p)
        >>> 
        >>> # space dimension , extent , discretization
        >>> dim, L, Ng = 3, 10, 64;
        >>> 
        >>> # instantiate the Flow class
        >>> flow = pystokes.wallBounded.Flow(radius=b, particles=N, viscosity=eta, gridpoints=Ng*Ng)
        >>> 
        >>> # create grid, evaluate flow and plot
        >>> rr, vv = pystokes.utils.gridXY(dim, L, Ng)
        >>> flow.flowField3t(vv, rr, r, V3t)
        >>> pystokes.utils.plotStreamlinesXY(vv, rr, r, offset=6-1, density=1.4, title='1s')
        """

        cdef int N = self.N, Nt = self.Nt
        cdef int i, ii, Z= 2*N, Z1= 3*N, Z2 = 4*N
        cdef double dx, dy, dz, dr, idr, idr3, aidr2, sxx, syy, sxy, sxz, syz, srr, srx, sry, srz
        cdef double aa = self.b**2, vv1, vv2, vx, vy, vz, mus = (28.0*self.b**3)/24, radi=self.b
        for i in prange(Nt, nogil=True):
            vx = 0.0;vy = 0.0; vz = 0.0;
            for ii in range(N):
                sxx = V2s[ii]
                syy = V2s[ii+N]
                sxy = V2s[ii+Z]
                sxz = V2s[ii+Z1]
                syz = V2s[ii+Z2]
                
                dx = rt[i]      - r[ii]
                dy = rt[i+Nt]   - r[ii+N]
                dz = rt[i+2*Nt] - r[ii+Z] 
                dr = sqrt( dx*dx + dy*dy + dz*dz )
                if dr>maskR:
                    idr = 1.0/dr
                    idr3 = idr*idr*idr      
                    aidr2  = aa*idr*idr

                    srr = (sxx*(dx*dx-dz*dz) + syy*(dy*dy-dz*dz) +  2*sxy*dx*dy + 2*sxz*dx*dz  +  2*syz*dy*dz)*idr*idr
                    srx = sxx*dx +  sxy*dy + sxz*dz  
                    sry = sxy*dx +  syy*dy + syz*dz  
                    srz = sxz*dx +  syz*dy - (sxx+syy)*dz 

                    vv1 = 3*(1-aidr2)*srr*idr3
                    vv2 = 1.2*aidr2*idr3

                    vx +=  vv1*dx + vv2*srx
                    vy +=  vv1*dy + vv2*sry
                    vz +=  vv1*dz + vv2*srz

            vv[i]      +=  vx*mus
            vv[i+Nt]   +=  vy*mus
            vv[i+2*Nt] +=  vz*mus
        return


    cpdef flowField3t(self, double [:] vv, double [:] rt, double [:] r, double [:] V3t, double maskR=1.0):
        """
        Compute flow field at field points due to 3t mode of the slip 
        ...

        Parameters
        ----------
        vv: np.array
            An array of flow at field points
            An array of size 3*Nt,
        rt: np.array
            An array of field points
            An array of size 3*Nt,
        r: np.array
            An array of positions
            An array of size 3*N,
        V3t: np.array
            An array of 3t mode of the slip
            An array of size 3*N,
 
        Examples
        --------
        An example of the Flow field due to $3t$ mode of active slip

        >>> import pystokes, numpy as np, matplotlib.pyplot as plt
        >>> 
        >>> # particle radius, self-propulsion speed, number and fluid viscosity
        >>> b, eta, N = 1.0, 1.0/6.0, 1
        >>> 
        >>> # initialize
        >>> r, p = np.array([0.0, 0.0, 3.4]), np.array([0.0, 1.0, 0])
        >>> V3t  = pystokes.utils.irreducibleTensors(1, p)
        >>> 
        >>> # space dimension , extent , discretization
        >>> dim, L, Ng = 3, 10, 64;
        >>> 
        >>> # instantiate the Flow class
        >>> flow = pystokes.wallBounded.Flow(radius=b, particles=N, viscosity=eta, gridpoints=Ng*Ng)
        >>> 
        >>> # create grid, evaluate flow and plot
        >>> rr, vv = pystokes.utils.gridXY(dim, L, Ng)
        >>> flow.flowField3t(vv, rr, r, V3t)
        >>> pystokes.utils.plotStreamlinesXY(vv, rr, r, offset=6-1, density=1.4, title='2s')
        """

        cdef int N = self.N, Nt = self.Nt
        cdef  int i, ii 
        cdef double dx, dy, dz, dr, idr, idr3, V3tdotidr, vx, vy, vz,mud1 = -1.0*(self.b**5)/10, radi=self.b
 
        for i in prange(Nt, nogil=True):
            vx =0.0; vy = 0.0; vz =0.0;
            for ii in range(N):
                dx = rt[i]      - r[ii]
                dy = rt[i+Nt]   - r[ii+N]
                dz = rt[i+2*Nt] - r[ii+2*N] 

                dr = sqrt( dx*dx + dy*dy + dz*dz )
                if dr>maskR:
                    idr = 1.0/dr
                    idr3 = idr*idr*idr 
                    V3tdotidr = (V3t[ii]*dx + V3t[ii+N]*dy + V3t[ii+2*N]*dz)*idr*idr

                    vx += (V3t[ii]     - 3.0*V3tdotidr*dx )*idr3
                    vy += (V3t[ii+N]   - 3.0*V3tdotidr*dy )*idr3
                    vz += (V3t[ii+2*N] - 3.0*V3tdotidr*dz )*idr3
        
            vv[i]      += vx*mud1
            vv[i+Nt]   += vy*mud1
            vv[i+2*Nt] += vz*mud1
        
        return 


    cpdef flowField3s(self, double [:] vv, double [:] rt, double [:] r, double [:] V3s, double maskR=1):
        """
        Compute flow field at field points  due to 3s mode of the slip 
        ...

        Parameters
        ----------
        vv: np.array
            An array of flow at field points
            An array of size 3*Nt,
        rt: np.array
            An array of field points
            An array of size 3*Nt,
        r: np.array
            An array of positions
            An array of size 3*N,
        V3s: np.array
            An array of 3s mode of the slip
            An array of size 7*N,
        """

        cdef int N = self.N, Nt = self.Nt
        cdef int i, ii, 
        cdef double dx, dy, dz, dr, idr, idr5, idr7, radi=self.b
        cdef double aidr2, grrr, grrx, grry, grrz, gxxx, gyyy, gxxy, gxxz, gxyy, gxyz, gyyz
        for i in prange(Nt, nogil=True):
            for ii in range(N):
                gxxx = V3s[ii]
                gyyy = V3s[ii+N]
                gxxy = V3s[ii+2*N]
                gxxz = V3s[ii+3*N]
                gxyy = V3s[ii+4*N]
                gxyz = V3s[ii+5*N]
                gyyz = V3s[ii+6*N]
                dx = rt[i]      - r[ii]
                dy = rt[i+Nt]   - r[ii+N]
                dz = rt[i+2*Nt] - r[ii+2*N] 
                dr = sqrt( dx*dx + dy*dy + dz*dz )
                if dr>maskR:
                    idr = 1.0/sqrt( dx*dx + dy*dy + dz*dz )
                    idr5 = idr*idr*idr*idr*idr      
                    idr7 = idr5*idr*idr     
                    aidr2 = self.b*self.b*idr*idr

                    grrr = gxxx*dx*(dx*dx-3*dz*dz) + 3*gxxy*dy*(dx*dx-dz*dz) + gxxz*dz*(3*dx*dx-dz*dz) +\
                           3*gxyy*dx*(dy*dy-dz*dz) + 6*gxyz*dx*dy*dz + gyyy*dy*(dy*dy-3*dz*dz) +  gyyz*dz*(3*dy*dy-dz*dz) 
                    grrx = gxxx*(dx*dx-dz*dz) + gxyy*(dy*dy-dz*dz) +  2*gxxy*dx*dy + 2*gxxz*dx*dz  +  2*gxyz*dy*dz
                    grry = gxxy*(dx*dx-dz*dz) + gyyy*(dy*dy-dz*dz) +  2*gxyy*dx*dy + 2*gxyz*dx*dz  +  2*gyyz*dy*dz
                    grrz = gxxz*(dx*dx-dz*dz) + gyyz*(dy*dy-dz*dz) +  2*gxyz*dx*dy - 2*(gxxx+gxyy)*dx*dz  - 2*(gxxy+gyyy)*dy*dz

                    vv[i]      += 3*(1-(15.0/7)*aidr2)*grrx*idr5 - 15*(1-aidr2)*grrr*dx*idr7
                    vv[i+Nt]   += 3*(1-(15.0/7)*aidr2)*grry*idr5 - 15*(1-aidr2)*grrr*dy*idr7
                    vv[i+2*Nt] += 3*(1-(15.0/7)*aidr2)*grrz*idr5 - 15*(1-aidr2)*grrr*dz*idr7
        return


    cpdef flowField3a(self, double [:] vv, double [:] rt, double [:] r, double [:] V3a, double maskR=1):
        """
        Compute flow field at field points  due to 3a mode of the slip 
        ...

        Parameters
        ----------
        vv: np.array
            An array of flow at field points
            An array of size 3*Nt,
        rt: np.array
            An array of field points
            An array of size 3*Nt,
        r: np.array
            An array of positions
            An array of size 3*N,
        V3a: np.array
            An array of 3a mode of the slip
            An array of size 5*N,
        """

        cdef int N = self.N, Nt = self.Nt
        cdef int i, ii
        cdef double dx, dy, dz, dr, idr, idr5, vxx, vyy, vxy, vxz, vyz, vrx, vry, vrz, radi=self.b
 
        for i in prange(Nt, nogil=True):
            for ii in range(N):
                vxx  = V3a[ii]
                vyy = V3a[ii+N]
                vxy = V3a[ii+2*N]
                vxz = V3a[ii+3*N]
                vyz = V3a[ii+4*N]
                dx = rt[i]      - r[ii]
                dy = rt[i+Nt]   - r[ii+N]
                dz = rt[i+2*Nt] - r[ii+2*N] 
                dr = sqrt( dx*dx + dy*dy + dz*dz )
                if dr>radi:
                    idr = 1.0/dr
                    idr5 = idr*idr*idr*idr*idr
                    vrx = vxx*dx +  vxy*dy + vxz*dz  
                    vry = vxy*dx +  vyy*dy + vyz*dz  
                    vrz = vxz*dx +  vyz*dy - (vxx+vyy)*dz 

                    vv[i]      += 8*( dy*vrz - dz*vry )*idr5
                    vv[i+Nt]   += 8*( dz*vrx - dx*vrz )*idr5
                    vv[i+2*Nt] += 8*( dx*vry - dy*vrx )*idr5 
        return 


    cpdef flowField4a(self, double [:] vv, double [:] rt, double [:] r, double [:] V4a, double maskR=1):
        """
        Compute flow field at field points  due to 4a mode of the slip 
        ...

        Parameters
        ----------
        vv: np.array
            An array of flow at field points
            An array of size 3*Nt,
        rt: np.array
            An array of field points
            An array of size 3*Nt,
        r: np.array
            An array of positions
            An array of size 3*N,
        V4a: np.array
            An array of 4a mode of the slip
            An array of size 7*N,
        """

        cdef int N = self.N, Nt = self.Nt
        cdef int i, ii
        cdef double dx, dy, dz, idr, idr7, dr, radi=self.b
        cdef double mrrx, mrry, mrrz, mxxx, myyy, mxxy, mxxz, mxyy, mxyz, myyz
 
        for i in prange(Nt, nogil=True):
            for ii in range(N):
                mxxx = V4a[ii]
                myyy = V4a[ii+N]
                mxxy = V4a[ii+2*N]
                mxxz = V4a[ii+3*N]
                mxyy = V4a[ii+4*N]
                mxyz = V4a[ii+5*N]
                myyz = V4a[ii+6*N]
                dx = rt[i]      - r[ii]
                dy = rt[i+Nt]   - r[ii+N]
                dz = rt[i+2*Nt] - r[ii+2*N] 
                dr = sqrt( dx*dx + dy*dy + dz*dz )
                if dr>radi:
                    idr = 1.0/dr                
                    idr7 = idr*idr*idr*idr*idr*idr*idr
                    mrrx = mxxx*(dx*dx-dz*dz) + mxyy*(dy*dy-dz*dz) +  2*mxxy*dx*dy + 2*mxxz*dx*dz  +  2*mxyz*dy*dz
                    mrry = mxxy*(dx*dx-dz*dz) + myyy*(dy*dy-dz*dz) +  2*mxyy*dx*dy + 2*mxyz*dx*dz  +  2*myyz*dy*dz
                    mrrz = mxxz*(dx*dx-dz*dz) + myyz*(dy*dy-dz*dz) +  2*mxyz*dx*dy - 2*(mxxx+mxyy)*dx*dz  - 2*(mxxy+myyy)*dy*dz

                    vv[i]      -= 6*( dy*mrrz - dz*mrry )*idr7
                    vv[i+Nt]   -= 6*( dz*mrrx - dx*mrrz )*idr7
                    vv[i+2*Nt] -= 6*( dx*mrry - dy*mrrx )*idr7
        return





@cython.wraparound(False)
@cython.boundscheck(False)
@cython.cdivision(True)
@cython.nonecheck(False)
cdef class PD:
    """
    Power Dissipation(PD)
    
    Methods in this class calculate the power dissipation
    using the inputs of -  power dissipation, arrays of positions, velocity or angular velocity, 
    along with an array of forces or torques or a slip mode

    The power dissipation is then updated by each method. 
    
    ...

    ----------
    radius: float
        Radius of the particles (a).    
    particles: int
        Number of particles (N)
    viscosity: float 
        Viscosity of the fluid (eta)

   """

    def __init__(self, radius=1, particles=1, viscosity=1.0):
        self.b      = radius
        self.N      = particles
        self.eta    = viscosity
        self.gammaT = 6*PI*self.eta*self.b
        self.gammaR = 8*PI*self.eta*self.b**3
        self.mu  = 1.0/self.gammaT
        self.muv = 1.0/(8*PI*self.eta)
        self.mur = 1.0/(self.gammaR)

        self.Mobility = np.zeros( (3*self.N, 3*self.N), dtype=np.float64)


    cpdef frictionTT(self, double depsilon, double [:] v, double [:] r):
        """
        Compute power dissipation due to translation using :math:`\dot{\epsilon}=V\cdot\gamma^{TT}\cdot V`
        ...
        Parameters
        ----------
        depsilon: energy dissipation,
        v: np.array
            An array of velocities
            An array of size 3*N,
        r: np.array
            An array of positions
            An array of size 3*N,
        ----------
        """


        cdef int N  = self.N, i, j, Z=2*N
        cdef double dx, dy, dz, idr, idr2, vx, vy, vz, vv1, vv2, aa = (2.0*self.b*self.b)/3.0 
        cdef double gT=self.gammaT, gg = -gT*gT, muv=self.muv        
        
        for i in prange(N, nogil=True):
            vx=0; vy=0;   vz=0;
            for j in range(N):
                if i != j:
                    dx = r[i]   - r[j]
                    dy = r[i+N] - r[j+N]
                    dz = r[i+Z] - r[j+Z] 
                    idr = 1.0/sqrt( dx*dx + dy*dy + dz*dz )
                    idr2 = idr*idr
                    
                    vv1 = (1+aa*idr2)*idr 
                    vv2 = (1-3*aa*idr2)*( v[j]*dx + v[j+N]*dy + v[j+Z]*dz )*idr2*idr
                    vx += vv1*v[j]    + vv2*dx 
                    vy += vv1*v[j+N] + vv2*dy 
                    vz += vv1*v[j+Z] + vv2*dz 

            depsilon += v[i] * (gT*v[i] + gg*muv*vx)
            depsilon += v[i + N] * (gT*v[i+N] + gg*muv*vy)
            depsilon += v[i+Z] * (gT*v[i+Z] + gg*muv*vz)
        return depsilon
               
   
    cpdef frictionTR(self, double depsilon, double [:] v, double [:] o, double [:] r):
        """
        Compute energy dissipation due to rotation using :math:`\dot{\epsilon}=V\cdot\gamma^{TR}\cdot \Omega`
        ...
        Parameters
        ----------
        depsilon: power dissipation,
        v: np.array
            An array of velocities
            An array of size 3*N,
        o: np.array
            An array of angular velocities
            An array of size 3*N,
        r: np.array
            An array of positions
            An array of size 3*N,
        ----------
        """


        cdef int N = self.N, i, j, Z=2*N 
        cdef double dx, dy, dz, idr, idr3, vx, vy, vz
        cdef double muv=self.muv, gg=-self.gammaT*self.gammaR       
        
        for i in prange(N, nogil=True):
            vx=0; vy=0;   vz=0;
            for j in range(N):
                if i != j:
                    dx = r[i]    - r[j]
                    dy = r[i+N] - r[j+N]
                    dz = r[i+Z] - r[j+Z] 
                    idr = 1.0/sqrt( dx*dx + dy*dy + dz*dz )
                    idr3 = idr*idr*idr
                    vx += -(dy*o[j+Z] -o[j+N]*dz )*idr3
                    vy += -(dz*o[j]    -o[j+Z]*dx )*idr3
                    vz += -(dx*o[j+N]  -o[j]   *dy )*idr3

            depsilon += v[i] * gg * muv*vx
            depsilon += v[i+N] * gg * muv*vy
            depsilon += v[i+Z] * gg *muv*vz
        return depsilon
    
    
    cpdef frictionT2s(self, double depsilon, double [:] V1s, double [:] V2s, double [:] r):
        """
        Compute energy dissipation due to 2s mode of the slip :math:`\dot{\epsilon}=V^{1s}\cdot\gamma^{T,2s}\cdot V^{2s}`
        ...
        Parameters
        ----------
        depsilon: power dissipation
        V1s: np.array
            An array of 1s mode of velocities
            An array of size 3*N,
        V2s: np.array
            An array of 2s mode of the slip
            An array of size 5*N,
        r: np.array
            An array of positions
            An array of size 3*N,
        ----------
        """

        cdef int N = self.N, i, j, Z=2*N, Z1=3*N, Z2=4*N
        cdef double dx, dy, dz, dr, idr,  idr3
        cdef double aa=(self.b*self.b*8.0)/3.0, vv1, vv2, aidr2
        cdef double vx, vy, vz, 
        cdef double sxx, sxy, sxz, syz, syy, srr, srx, sry, srz, mus = (28.0*self.b**3)/24 
        cdef double gT = self.gammaT
 
        for i in prange(N, nogil=True):
            vx=0; vy=0;   vz=0;
            for j in range(N):
                if i != j:
                    sxx = V2s[j]
                    syy = V2s[j+N]
                    sxy = V2s[j+Z]
                    sxz = V2s[j+Z1]
                    syz = V2s[j+Z2]
                    dx = r[i]    - r[j]
                    dy = r[i+N] - r[j+N]
                    dz = r[i+Z] - r[j+Z] 
                    idr = 1.0/sqrt( dx*dx + dy*dy + dz*dz )
                    idr3 = idr*idr*idr      
                    aidr2  = aa*idr*idr
                    
                    srr = (sxx*(dx*dx-dz*dz) + syy*(dy*dy-dz*dz) +  2*sxy*dx*dy + 2*sxz*dx*dz  +  2*syz*dy*dz)*idr*idr
                    srx = sxx*dx +  sxy*dy + sxz*dz  
                    sry = sxy*dx +  syy*dy + syz*dz  
                    srz = sxz*dx +  syz*dy - (sxx+syy)*dz 
                    
                    vv1 = 3*(1-aidr2)*srr*idr3
                    vv2 = 1.2*aidr2*idr3
                    vx +=  vv1*dx + vv2*srx
                    vy +=  vv1*dy + vv2*sry
                    vz +=  vv1*dz + vv2*srz
            
            depsilon += -V1s[i] * gT * vx*mus
            depsilon += -V1s[i+N] * gT * vy*mus
            depsilon += -V1s[i+Z] * gT * vz*mus

        return depsilon


    cpdef frictionT3t(self, double depsilon, double [:] V1s, double [:] V3t, double [:] r):
        """
        Compute energy dissipation due to 3t mode of the slip :math:`\dot{\epsilon}=V^{1s}\cdot\gamma^{T,3t}\cdot V^{3t}`
        ...
        Parameters
        ----------
        depsilon: power dissipation
        V1s: np.array
            An array of 1s mode of velocities
            An array of size 3*N,
        V3t: np.array
            An array of 3t mode of the slip
            An array of size 3*N,
        r: np.array
            An array of positions
            An array of size 3*N,
        ----------
        """

        cdef int N = self.N, i, j, Z=2*N  
        cdef double dx, dy, dz, idr, idr3, V3tdotidr, vx, vy, vz, mud = 3.0*self.b*self.b*self.b/5, mud1 = -1.0*(self.b**5)/10
        cdef double gammaT = self.gammaT
 
        for i in prange(N, nogil=True):
            vx=0; vy=0;   vz=0; 
            for j in range(N):
                if i != j: 
                    dx = r[ i]   - r[j]
                    dy = r[i+N] - r[j+N]
                    dz = r[i+Z] - r[j+Z] 
                    idr = 1.0/sqrt( dx*dx + dy*dy + dz*dz )
                    idr3 = idr*idr*idr 
                    V3tdotidr = (V3t[j]*dx + V3t[j+N]*dy + V3t[j+Z]*dz)*idr*idr

                    vx += (V3t[j]    - 3.0*V3tdotidr*dx )*idr3
                    vy += (V3t[j+N] - 3.0*V3tdotidr*dy )*idr3
                    vz += (V3t[j+Z] - 3.0*V3tdotidr*dz )*idr3
            
            depsilon += -V1s[i] * gammaT * mud1*vx
            depsilon += -V1s[i+N] * gammaT * mud1*vy
            depsilon += -V1s[i+Z] * gammaT * mud1*vz
        return depsilon


    ## Angular velocities
    cpdef frictionRT(self, double depsilon, double [:] v, double [:] o, double [:] r):
        """
        Compute energy dissipation due to rotation using :math:`\dot{\epsilon}=\Omega\cdot\gamma^{RT}\cdot V`
        ...
        Parameters
        ----------
        depsilon: np.array
                   An array of energy dissipation
                   An array of size 3*N,
        v: np.array
            An array of velocities
            An array of size 3*N,
        o: np.array
            An array of angular velocities
            An array of size 3*N,
        r: np.array
            An array of positions
            An array of size 3*N,
        ----------
        """

        cdef int N = self.N, i, j, Z=2*N 
        cdef double dx, dy, dz, idr, idr3, ox, oy, oz, muv=self.muv
        cdef double gg= -self.gammaT*self.gammaR
 
        for i in prange(N, nogil=True):
            ox=0;   oy=0;   oz=0;
            for j in range(N):
                if i != j:
                    dx = r[i]    - r[j]
                    dy = r[i+N] - r[j+N]
                    dz = r[i+Z] - r[j+Z] 
                    idr = 1.0/sqrt( dx*dx + dy*dy + dz*dz )
                    idr3 = idr*idr*idr

                    ox += (v[j+N]*dz - v[j+Z]*dy )*idr3
                    oy += (v[j+Z]*dx - v[j]   *dz )*idr3
                    oz += (v[j]   *dy - v[j+N]*dx )*idr3
            depsilon += o[i] * gg * muv*ox
            depsilon += o[i+N] * gg * muv*oy
            depsilon += o[i+Z] * gg * muv*oz
        return  depsilon

               
    cpdef frictionRR(self, double depsilon, double [:] o, double [:] r):
        """
        Compute energy dissipation due to translation using :math:`\dot{\epsilon}=\Omega\cdot\gamma^{RR}\cdot \Omega`
        ...
        Parameters
        ----------
        depsilon: power dissipation
        o: np.array
            An array of angular velocities
            An array of size 3*N,
        r: np.array
            An array of positions
            An array of size 3*N,
        ----------
        """

        cdef int N = self.N, i, j, Z=2*N 
        cdef double dx, dy, dz, idr, idr3, Odotidr, ox, oy, oz, gR=self.gammaR, gg=-gR*gR, muv=self.muv 
 
        for i in prange(N, nogil=True):
            ox=0;   oy=0;   oz=0;
            for j in range(N):
                if i != j:
                    dx = r[i]   - r[j]
                    dy = r[i+N] - r[j+N]
                    dz = r[i+Z] - r[j+Z] 
                    idr = 1.0/sqrt( dx*dx + dy*dy + dz*dz )
                    idr3 = idr*idr*idr
                    Odotidr = ( o[j]*dx + o[j+N]*dy + o[j+Z]*dz )*idr*idr

                    ox += ( o[j]    - 3*Odotidr*dx )*idr3
                    oy += ( o[j+N] - 3*Odotidr*dy )*idr3
                    oz += ( o[j+Z] - 3*Odotidr*dz )*idr3
            
            depsilon += o[i] * (gR*o[i]    - gg*0.5*muv*ox)
            depsilon += o[i+N] * (gR*o[i+N] - gg*0.5*muv*oy)
            depsilon += o[i+Z] * (gR*o[i+Z] - gg*0.5*muv*oz)
        return  depsilon
