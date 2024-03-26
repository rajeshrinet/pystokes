cimport cython
from libc.math cimport sqrt
from cython.parallel import prange
cdef double PI = 3.14159265359
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
        self.a   = radius
        self.N  = particles
        self.eta = viscosity
        self.mu  = 1.0/(6*PI*self.eta*self.a)
        self.muv = 1.0/(8*PI*self.eta)
        self.mur = 1.0/(8*PI*self.eta*self.a**3)

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


        cdef int N  = self.N, i, j, xx=2*N
        cdef double dx, dy, dz, idr, idr2, vx, vy, vz, vv1, vv2, aa = (2.0*self.a*self.a)/3.0 
        cdef double mu=self.mu, muv=self.muv        
        
        for i in prange(N, nogil=True):
            vx=0; vy=0;   vz=0;
            for j in range(N):
                if i != j:
                    dx = r[i]    - r[j]
                    dy = r[i+N] - r[j+N]
                    dz = r[i+xx] - r[j+xx] 
                    idr = 1.0/sqrt( dx*dx + dy*dy + dz*dz )
                    idr2 = idr*idr
                    
                    vv1 = (1+aa*idr2)*idr 
                    vv2 = (1-3*aa*idr2)*( F[j]*dx + F[j+N]*dy + F[j+xx]*dz )*idr2*idr
                    vx += vv1*F[j]    + vv2*dx 
                    vy += vv1*F[j+N] + vv2*dy 
                    vz += vv1*F[j+xx] + vv2*dz 

            v[i]    += mu*F[i]    + muv*vx
            v[i+N] += mu*F[i+N] + muv*vy
            v[i+xx] += mu*F[i+xx] + muv*vz
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


        cdef int N = self.N, i, j, xx=2*N 
        cdef double dx, dy, dz, idr, idr3, vx, vy, vz
        cdef double muv=self.muv       
        
        for i in prange(N, nogil=True):
            vx=0; vy=0;   vz=0;
            for j in range(N):
                if i != j:
                    dx = r[i]    - r[j]
                    dy = r[i+N] - r[j+N]
                    dz = r[i+xx] - r[j+xx] 
                    idr = 1.0/sqrt( dx*dx + dy*dy + dz*dz )
                    idr3 = idr*idr*idr
                    vx += -(dy*T[j+xx] -T[j+N]*dz )*idr3
                    vy += -(dz*T[j]    -T[j+xx]*dx )*idr3
                    vz += -(dx*T[j+N] -T[j]   *dy )*idr3

            v[i]    += muv*vx
            v[i+N] += muv*vy
            v[i+xx] += muv*vz
        return 
    
    
    cpdef propulsionT2s(self, double [:] v, double [:] r, double [:] S):
        """
        Compute velocity due to 2s mode of the slip :math:`v=\pi^{T,2s}\cdot S` 
        ...

        Parameters
        ----------
        v: np.array
            An array of velocities
            An array of size 3*N,
        r: np.array
            An array of positions
            An array of size 3*N,
        S: np.array
            An array of 2s mode of the slip
            An array of size 5*N,
        """

        cdef int N = self.N, i, j, xx=2*N, xx1=3*N, xx2=4*N
        cdef double dx, dy, dz, dr, idr,  idr3
        cdef double aa=(self.a*self.a*8.0)/3.0, vv1, vv2, aidr2
        cdef double vx, vy, vz, 
        cdef double sxx, sxy, sxz, syz, syy, srr, srx, sry, srz, mus = -(28.0*self.a*self.a)/24 
 
        for i in prange(N, nogil=True):
            vx=0; vy=0;   vz=0;
            for j in range(N):
                if i != j:
                    sxx = S[j]
                    syy = S[j+N]
                    sxy = S[j+xx]
                    sxz = S[j+xx1]
                    syz = S[j+xx2]
                    dx = r[i]    - r[j]
                    dy = r[i+N] - r[j+N]
                    dz = r[i+xx] - r[j+xx] 
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
            v[i+N]+= vy*mus
            v[i+xx]+= vz*mus

        return 


    cpdef propulsionT3t(self, double [:] v, double [:] r, double [:] D):
        """
        Compute velocity due to 3t mode of the slip :math:`v=\pi^{T,3t}\cdot D` 
        ...

        Parameters
        ----------
        v: np.array
            An array of velocities
            An array of size 3*N,
        r: np.array
            An array of positions
            An array of size 3*N,
        D: np.array
            An array of 3t mode of the slip
            An array of size 3*N,
        """

        cdef int N = self.N, i, j, xx=2*N  
        cdef double dx, dy, dz, idr, idr3, Ddotidr, vx, vy, vz, mud = 3.0*self.a*self.a*self.a/5, mud1 = -1.0*(self.a**3)/5
 
        for i in prange(N, nogil=True):
            vx=0; vy=0;   vz=0; 
            for j in range(N):
                if i != j: 
                    dx = r[ i]   - r[j]
                    dy = r[i+N] - r[j+N]
                    dz = r[i+xx] - r[j+xx] 
                    idr = 1.0/sqrt( dx*dx + dy*dy + dz*dz )
                    idr3 = idr*idr*idr 
                    Ddotidr = (D[j]*dx + D[j+N]*dy + D[j+xx]*dz)*idr*idr

                    vx += (D[j]    - 3.0*Ddotidr*dx )*idr3
                    vy += (D[j+N] - 3.0*Ddotidr*dy )*idr3
                    vz += (D[j+xx] - 3.0*Ddotidr*dz )*idr3
            
            v[i]   += mud1*vx
            v[i+N]+= mud1*vy
            v[i+xx]+= mud1*vz
        return 


    cpdef propulsionT3a(self, double [:] v, double [:] r, double [:] V):
        """
        Compute velocity due to 3a mode of the slip :math:`v=\pi^{T,3a}\cdot V` 
        ...

        Parameters
        ----------
        v: np.array
            An array of velocities
            An array of size 3*N,
        r: np.array
            An array of positions
            An array of size 3*N,
        V: np.array
            An array of 3a mode of the slip
            An array of size 5*N,
        """

        cdef int N = self.N, i, j 
        cdef double dx, dy, dz, idr, idr5, vxx, vyy, vxy, vxz, vyz, vrx, vry, vrz
 
        for i in prange(N, nogil=True):
            for j in range(N):
                if i != j:
                    vxx = V[j]
                    vyy = V[j+N]
                    vxy = V[j+2*N]
                    vxz = V[j+3*N]
                    vyz = V[j+4*N]
                    dx = r[i]      - r[j]
                    dy = r[i+N]   - r[j+N]
                    dz = r[i+2*N] - r[j+2*N] 
                    idr = 1.0/sqrt( dx*dx + dy*dy + dz*dz)
                    idr5 = idr*idr*idr*idr*idr
                    vrx = vxx*dx +  vxy*dy + vxz*dz  
                    vry = vxy*dx +  vyy*dy + vyz*dz  
                    vrz = vxz*dx +  vyz*dy - (vxx+vyy)*dz 

                    v[i]      -= 8*( dy*vrz - dz*vry )*idr5
                    v[i+N]   -= 8*( dz*vrx - dx*vrz )*idr5
                    v[i+2*N] -= 8*( dx*vry - dy*vrx )*idr5 
                else:
                    pass
        return


    cpdef propulsionT3s(self, double [:] v, double [:] r, double [:] G):
        """
        Compute velocity due to 3s mode of the slip :math:`v=\pi^{T,3s}\cdot G` 
        ...

        Parameters
        ----------
        v: np.array
            An array of velocities
            An array of size 3*N,
        r: np.array
            An array of positions
            An array of size 3*N,
        G: np.array
            An array of 3s mode of the slip
            An array of size 7*N,
        """

        cdef int N = self.N, i, j 
        cdef double dx, dy, dz, idr, idr5, idr7, aidr2, grrr, grrx, grry, grrz, gxxx, gyyy, gxxy, gxxz, gxyy, gxyz, gyyz
 
        for i in prange(N, nogil=True):
             for j in range(N):
                if i != j:
                    gxxx = G[j]
                    gyyy = G[j+N]
                    gxxy = G[j+2*N]
                    gxxz = G[j+3*N]
                    gxyy = G[j+4*N]
                    gxyz = G[j+5*N]
                    gyyz = G[j+6*N]
                    dx = r[i]      - r[j]
                    dy = r[i+N]   - r[j+N]
                    dz = r[i+2*N] - r[j+2*N] 
                    idr = 1.0/sqrt( dx*dx + dy*dy + dz*dz )
                    idr5 = idr*idr*idr*idr*idr      
                    idr7 = idr5*idr*idr     
                    aidr2 = (10.0/3)*self.a*self.a*idr*idr
                    
                    grrr = gxxx*dx*(dx*dx-3*dz*dz) + 3*gxxy*dy*(dx*dx-dz*dz) + gxxz*dz*(3*dx*dx-dz*dz) +\
                       3*gxyy*dx*(dy*dy-dz*dz) + 6*gxyz*dx*dy*dz + gyyy*dy*(dy*dy-3*dz*dz) +  gyyz*dz*(3*dy*dy-dz*dz) 
                    grrx = gxxx*(dx*dx-dz*dz) + gxyy*(dy*dy-dz*dz) +  2*gxxy*dx*dy + 2*gxxz*dx*dz  +  2*gxyz*dy*dz
                    grry = gxxy*(dx*dx-dz*dz) + gyyy*(dy*dy-dz*dz) +  2*gxyy*dx*dy + 2*gxyz*dx*dz  +  2*gyyz*dy*dz
                    grrz = gxxz*(dx*dx-dz*dz) + gyyz*(dy*dy-dz*dz) +  2*gxyz*dx*dy - 2*(gxxx+gxyy)*dx*dz  - 2*(gxxy+gyyy)*dy*dz
                  
                    v[i]      += 3*(1-(15.0/7)*aidr2)*grrx*idr5 - 15*(1-aidr2)*grrr*dx*idr7
                    v[i+N]   += 3*(1-(15.0/7)*aidr2)*grry*idr5 - 15*(1-aidr2)*grrr*dy*idr7
                    v[i+2*N] += 3*(1-(15.0/7)*aidr2)*grrz*idr5 - 15*(1-aidr2)*grrr*dz*idr7
                else:
                    pass 
        return


    cpdef propulsionT4a(self, double [:] v, double [:] r, double [:] M):
        """
        Compute velocity due to 4a mode of the slip :math:`v=\pi^{T,4a}\cdot M` 
        ...

        Parameters
        ----------
        v: np.array
            An array of velocities
            An array of size 3*N,
        r: np.array
            An array of positions
            An array of size 3*N,
        M: np.array
            An array of 4a mode of the slip
            An array of size 7*N,
        """

        cdef int N = self.N, i, j 
        cdef double dx, dy, dz, idr, idr7
        cdef double mrrx, mrry, mrrz, mxxx, myyy, mxxy, mxxz, mxyy, mxyz, myyz
 
        for i in prange(N, nogil=True):
            for j in range(N):
                if i != j:
                    mxxx = M[j]
                    myyy = M[j+N]
                    mxxy = M[j+2*N]
                    mxxz = M[j+3*N]
                    mxyy = M[j+4*N]
                    mxyz = M[j+5*N]
                    myyz = M[j+6*N]
                    dx = r[i]      - r[j]
                    dy = r[i+N]   - r[j+N]
                    dz = r[i+2*N] - r[j+2*N] 
                    idr = 1.0/sqrt( dx*dx + dy*dy + dz*dz )
                    idr7 = idr*idr*idr*idr*idr*idr*idr
                    mrrx = mxxx*(dx*dx-dz*dz) + mxyy*(dy*dy-dz*dz) +  2*mxxy*dx*dy + 2*mxxz*dx*dz  +  2*mxyz*dy*dz
                    mrry = mxxy*(dx*dx-dz*dz) + myyy*(dy*dy-dz*dz) +  2*mxyy*dx*dy + 2*mxyz*dx*dz  +  2*myyz*dy*dz
                    mrrz = mxxz*(dx*dx-dz*dz) + myyz*(dy*dy-dz*dz) +  2*mxyz*dx*dy - 2*(mxxx+mxyy)*dx*dz  - 2*(mxxy+myyy)*dy*dz
                    
                    v[i]      -= 6*( dy*mrrz - dz*mrry )*idr7
                    v[i+N]   -= 6*( dz*mrrx - dx*mrrz )*idr7
                    v[i+2*N] -= 6*( dx*mrry - dy*mrrx )*idr7
                else:
                    pass
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

        cdef int N = self.N, i, j, xx=2*N 
        cdef double dx, dy, dz, idr, idr3, ox, oy, oz, muv=self.muv
 
        for i in prange(N, nogil=True):
            ox=0;   oy=0;   oz=0;
            for j in range(N):
                if i != j:
                    dx = r[i]    - r[j]
                    dy = r[i+N] - r[j+N]
                    dz = r[i+xx] - r[j+xx] 
                    idr = 1.0/sqrt( dx*dx + dy*dy + dz*dz )
                    idr3 = idr*idr*idr

                    ox += (F[j+N]*dz - F[j+xx]*dy )*idr3
                    oy += (F[j+xx]*dx - F[j]   *dz )*idr3
                    oz += (F[j]   *dy - F[j+N]*dx )*idr3
            o[i]    += muv*ox
            o[i+N] += muv*oy
            o[i+xx] += muv*oz
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

        cdef int N = self.N, i, j, xx=2*N 
        cdef double dx, dy, dz, idr, idr3, Tdotidr, ox, oy, oz, mur=self.mur, muv=self.muv 
 
        for i in prange(N, nogil=True):
            ox=0;   oy=0;   oz=0;
            for j in range(N):
                if i != j:
                    dx = r[i]      - r[j]
                    dy = r[i+N]   - r[j+N]
                    dz = r[i+xx] - r[j+xx] 
                    idr = 1.0/sqrt( dx*dx + dy*dy + dz*dz )
                    idr3 = idr*idr*idr
                    Tdotidr = ( T[j]*dx + T[j+N]*dy + T[j+xx]*dz )*idr*idr

                    ox += ( T[j]    - 3*Tdotidr*dx )*idr3
                    oy += ( T[j+N] - 3*Tdotidr*dy )*idr3
                    oz += ( T[j+xx] - 3*Tdotidr*dz )*idr3
            
            o[i]    += mur*T[i]    - 0.5*muv*ox ##changed factor 0.5 here
            o[i+N] += mur*T[i+N] - 0.5*muv*oy
            o[i+xx] += mur*T[i+xx] - 0.5*muv*oz
        return  

    
    cpdef propulsionR2s(self, double [:] o, double [:] r, double [:] S):
        """
        Compute angular velocity due to 2s mode of the slip :math:`v=\pi^{R,2s}\cdot S` 
        ...

        Parameters
        ----------
        o: np.array
            An array of angular velocities
            An array of size 3*N,
        r: np.array
            An array of positions
            An array of size 3*N,
        S: np.array
            An array of 2s mode of the slip
            An array of size 5*N,
        """

        cdef int N = self.N, i, j, xx=2*N 
        cdef double dx, dy, dz, idr, idr5, ox, oy, oz
        cdef double sxx, sxy, sxz, syz, syy, srr, srx, sry, srz, mus = -(28.0*self.a*self.a)/24
 
        for i in prange(N, nogil=True):
            ox=0;   oy=0;   oz=0;
            for j in range(N):
                if i != j:
                    sxx = S[j]
                    syy = S[j+N]
                    sxy = S[j+2*N]
                    sxz = S[j+3*N]
                    syz = S[j+4*N]
                    dx = r[i]      - r[j]
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
                    
            o[i]    += ox*mus
            o[i+N] += oy*mus
            o[i+xx] += oz*mus
        return                 
    
    
    cpdef propulsionR3a(  self, double [:] o, double [:] r, double [:] V):
        """
        Compute angular velocity due to 3a mode of the slip :math:`v=\pi^{R,3a}\cdot V` 
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

        cdef int N = self.N, i, j 
        cdef double dx, dy, dz, idr, idr2, idr5, vxx, vyy, vxy, vxz, vyz, vrr, vrx, vry, vrz
 
        for i in prange(N, nogil=True):
             for j in range(N):
                if i != j:
                    vxx = V[j]
                    vyy = V[j+N]
                    vxy = V[j+2*N]
                    vxz = V[j+3*N]
                    vyz = V[j+4*N]
                    dx = r[i]      - r[j]
                    dy = r[i+N]   - r[j+N]
                    dz = r[i+2*N] - r[j+2*N] 
                    idr = 1.0/sqrt( dx*dx + dy*dy + dz*dz )
                    idr5 = idr*idr*idr*idr*idr      
                    vrr = (vxx*(dx*dx-dz*dz) + vyy*(dy*dy-dz*dz) +  2*vxy*dx*dy + 2*vxz*dx*dz  +  2*vyz*dy*dz)*idr*idr
                    vrx = vxx*dx +  vxy*dy + vxz*dz  
                    vry = vxy*dx +  vyy*dy + vyz*dz  
                    vrz = vxz*dx +  vyz*dy - (vxx+vyy)*dz 

                    o[i]      +=  ( 32*vrx- 20*vrr*dx )*idr5
                    o[i+N]   +=  ( 32*vry- 20*vrr*dy )*idr5
                    o[i+2*N] +=  ( 32*vrz- 20*vrr*dz )*idr5
                else :
                    pass 
        return


    cpdef propulsionR3s(  self, double [:] o, double [:] r, double [:] G):
        """
        Compute angular velocity due to 3s mode of the slip :math:`v=\pi^{R,3s}\cdot G` 
        ...

        Parameters
        ----------
        o: np.array
            An array of angular velocities
            An array of size 3*N,
        r: np.array
            An array of positions
            An array of size 3*N,
        G: np.array
            An array of 3s mode of the slip
            An array of size 7*N,
        """


        cdef int N = self.N, i, j 
        cdef double dx, dy, dz, idr, idr7, grrx, grry, grrz, gxxx, gyyy, gxxy, gxxz, gxyy, gxyz, gyyz
 
        for i in prange(N, nogil=True):
            for j in range(N):
                if i != j:
                    gxxx = G[j]
                    gyyy = G[j+N]
                    gxxy = G[j+2*N]
                    gxxz = G[j+3*N]
                    gxyy = G[j+4*N]
                    gxyz = G[j+5*N]
                    gyyz = G[j+6*N]
                    dx = r[i]      - r[j]
                    dy = r[i+N]   - r[j+N]
                    dz = r[i+2*N] - r[j+2*N] 
                    idr = 1.0/sqrt( dx*dx + dy*dy + dz*dz )
                    idr7 = idr*idr*idr*idr*idr*idr*idr     
                    
                    grrx = gxxx*(dx*dx-dz*dz) + gxyy*(dy*dy-dz*dz) +  2*gxxy*dx*dy + 2*gxxz*dx*dz  +  2*gxyz*dy*dz
                    grry = gxxy*(dx*dx-dz*dz) + gyyy*(dy*dy-dz*dz) +  2*gxyy*dx*dy + 2*gxyz*dx*dz  +  2*gyyz*dy*dz
                    grrz = gxxz*(dx*dx-dz*dz) + gyyz*(dy*dy-dz*dz) +  2*gxyz*dx*dy - 2*(gxxx+gxyy)*dx*dz  - 2*(gxxy+gyyy)*dy*dz

                    o[i]      += 15*( dy*grrz - dz*grry )*idr7
                    o[i+N]   += 15*( dz*grrx - dx*grrz )*idr7
                    o[i+2*N] += 15*( dx*grry - dy*grrx )*idr7
                else :
                    pass
        return                 


    cpdef propulsionR4a(  self, double [:] o, double [:] r, double [:] M):
        """
        Compute angular velocity due to 4a mode of the slip :math:`v=\pi^{R,4a}\cdot M` 
        ...

        Parameters
        ----------
        o: np.array
            An array of angular velocities
            An array of size 3*N,
        r: np.array
            An array of positions
            An array of size 3*N,
        M: np.array
            An array of 4a mode of the slip
            An array of size 7*N,
        """


        cdef int N = self.N, i, j 
        cdef double dx, dy, dz, idr, idr7, idr9, mrrr, mrrx, mrry, mrrz, mxxx, myyy, mxxy, mxxz, mxyy, mxyz, myyz
 
        for i in prange(N, nogil=True):
             for j in range(N):
                if i != j:
                    mxxx = M[j]
                    myyy = M[j+N]
                    mxxy = M[j+2*N]
                    mxxz = M[j+3*N]
                    mxyy = M[j+4*N]
                    mxyz = M[j+5*N]
                    myyz = M[j+6*N]
                    dx = r[i]      - r[j]
                    dy = r[i+N]   - r[j+N]
                    dz = r[i+2*N] - r[j+2*N] 
                    idr = 1.0/sqrt( dx*dx + dy*dy + dz*dz )
                    idr7 = idr*idr*idr*idr*idr*idr*idr      
                    idr9 = idr7*idr*idr     
                    
                    mrrr = mxxx*dx*(dx*dx-3*dz*dz) + 3*mxxy*dy*(dx*dx-dz*dz) + mxxz*dz*(3*dx*dx-dz*dz) +\
                       3*mxyy*dx*(dy*dy-dz*dz) + 6*mxyz*dx*dy*dz + myyy*dy*(dy*dy-3*dz*dz) +  myyz*dz*(3*dy*dy-dz*dz) 
                    mrrx = mxxx*(dx*dx-dz*dz) + mxyy*(dy*dy-dz*dz) +  2*mxxy*dx*dy + 2*mxxz*dx*dz  +  2*mxyz*dy*dz
                    mrry = mxxy*(dx*dx-dz*dz) + myyy*(dy*dy-dz*dz) +  2*mxyy*dx*dy + 2*mxyz*dx*dz  +  2*myyz*dy*dz
                    mrrz = mxxz*(dx*dx-dz*dz) + myyz*(dy*dy-dz*dz) +  2*mxyz*dx*dy - 2*(mxxx+mxyy)*dx*dz  - 2*(mxxy+myyy)*dy*dz
                  
                    o[i]      += 21*mrrr*dx*idr9 - 9*mrrx*idr7  
                    o[i+N]   += 21*mrrr*dy*idr9 - 9*mrry*idr7  
                    o[i+2*N] += 21*mrrr*dz*idr9 - 9*mrrz*idr7  
                else:
                    pass 
        return


    cpdef noiseTT(self, double [:] v, double [:] r):
        """
        Compute translation Brownian motion 
        ...

        Parameters
        ----------
        v: np.array
            An array of velocities
            An array of size 3*N,
        r: np.array
            An array of positions
            An array of size 3*N,
        """

        cdef int i, j, N=self.N, xx=2*N
        cdef double dx, dy, dz, idr, h2, hsq, idr2, idr3, idr4, idr5
        cdef double mu=self.mu, muv=2*mu*self.a*0.75, a2=self.a*self.a/3.0
        cdef double vx, vy, vz, mm=1/(.75*self.a)

        cdef double [:, :] M = self.Mobility
        cdef double [:]    Fr = np.random.normal(size=3*N)


        for i in prange(N, nogil=True):
            for j in range(N):
                dx = r[i]    - r[j]
                dy = r[i+N] - r[j+N]
                h2=2*r[j+xx]; hsq=r[j+xx]*r[j+xx]
                if i!=j:
                    dz = r[i+xx] - r[j+xx]
                    idr = 1.0/sqrt( dx*dx + dy*dy + dz*dz )
                    idr2=idr*idr;  idr3=idr*idr*idr
                    dx = dx*idr; dy=dy*idr; dz=dz*idr
                    #
                    M[i,    j   ] = (1 + dx*dx)*idr + a2*(2 - 6*dx*dx)*idr3
                    M[i+N, j+N] = (1 + dy*dy)*idr + a2*(2 - 6*dy*dy)*idr3
                    M[i+xx, j+xx] = (1 + dz*dz)*idr + a2*(2 - 6*dz*dz)*idr3
                    M[i,    j+N] = (    dx*dy)*idr + a2*(  - 6*dx*dy)*idr3
                    M[i,    j+xx] = (    dx*dz)*idr + a2*(  - 6*dx*dz)*idr3
                    M[i+N, j+xx] = (    dy*dz)*idr + a2*(  - 6*dy*dz)*idr3
                else:
                    # one-body mobility
                    M[i,    j   ] = mm
                    M[i+N, j+N] = mm
                    M[i+xx, j+xx] = mm
                    M[i,    j+N] = 0
                    M[i,    j+xx] = 0
                    M[i+N, j+xx] = 0


        for i in prange(N, nogil=True):
            for j in range(N):
                M[i,    j   ] = muv*M[i,    j   ]
                M[i+N, j+N] = muv*M[i+N, j+N]
                M[i+xx, j+xx] = muv*M[i+xx, j+xx]
                M[i,    j+N] = muv*M[i,    j+N]
                M[i,    j+xx] = muv*M[i,    j+xx]
                M[i+N, j+xx] = muv*M[i+N, j+xx]

                M[i+N, j   ] =     M[i,    j+N]
                M[i+xx, j   ] =     M[i,    j+xx]
                M[i+xx, j+N] =     M[i+N, j+xx]

        cdef double [:, :] L = np.linalg.cholesky(self.Mobility)

        for i in prange(N, nogil=True):
            vx=0; vy=0; vz=0;
            for j in range(N):
                vx += L[i   , j]*Fr[j] + L[i   , j+N]*Fr[j+N] + L[i   , j+xx]*Fr[j+xx]
                vy += L[i+N, j]*Fr[j] + L[i+N, j+N]*Fr[j+N] + L[i+N, j+xx]*Fr[j+xx]
                vz += L[i+xx, j]*Fr[j] + L[i+xx, j+N]*Fr[j+N] + L[i+xx, j+xx]*Fr[j+xx]
            v[i  ]  += vx
            v[i+N] += vy
            v[i+xx] += vz

        return


    cpdef noiseRR(self, double [:] o, double [:] r):
        """
        Compute rotational Brownian motion 
        ...

        Parameters
        ----------
        o: np.array
            An array of angular velocities
            An array of size 3*N,
        r: np.array
            An array of positions
            An array of size 3*N,
        """

        cdef int i, j, N=self.N, xx=2*N
        cdef double dx, dy, dz, idr, h2, hsq, idr2, idr3, idr4, idr5
        cdef double mur=1/(8*np.pi*self.eta), muv=0.25*sqrt(2.0)*mur, mm=4/(self.a**3)
        cdef double ox, oy, oz

        cdef double [:, :] M = self.Mobility
        cdef double [:]   Tr = np.random.normal(size=3*N)


        for i in prange(N, nogil=True):
            for j in range(N):
                dx = r[i]    - r[j]
                dy = r[i+N] - r[j+N]
                h2=2*r[j+xx]; hsq=r[j+xx]*r[j+xx]
                if i!=j:
                    dz = r[i+xx] - r[j+xx]
                    idr = 1.0/sqrt( dx*dx + dy*dy + dz*dz )
                    idr2=idr*idr;  idr3=idr*idr*idr
                    dx = dx*idr; dy=dy*idr; dz=dz*idr
                    #
                    M[i,    j   ] = (2 - 6*dx*dx)*idr3
                    M[i+N, j+N] = (2 - 6*dy*dy)*idr3
                    M[i+xx, j+xx] = (2 - 6*dz*dz)*idr3
                    M[i,    j+N] = (  - 6*dx*dy)*idr3
                    M[i,    j+xx] = (  - 6*dx*dz)*idr3
                    M[i+N, j+xx] = (  - 6*dy*dz)*idr3


        for i in prange(N, nogil=True):
            for j in range(N):
                M[i,    j   ] = muv*M[i,    j   ]
                M[i+N, j+N] = muv*M[i+N, j+N]
                M[i+xx, j+xx] = muv*M[i+xx, j+xx]
                M[i,    j+N] = muv*M[i,    j+N]
                M[i,    j+xx] = muv*M[i,    j+xx]
                M[i+N, j+xx] = muv*M[i+N, j+xx]

                M[i+N, j   ] =     M[i,    j+N]
                M[i+xx, j   ] =     M[i,    j+xx]
                M[i+xx, j+N] =     M[i+N, j+xx]

        cdef double [:, :] L = muv*np.linalg.cholesky(self.Mobility)
        for i in prange(N, nogil=True):
            ox=0; oy=0; oz=0;
            for j in range(N):
                ox += L[i   , j]*Tr[j] + L[i   , j+N]*Tr[j+N] + L[i   , j+xx]*Tr[j+xx]
                oy += L[i+N, j]*Tr[j] + L[i+N, j+N]*Tr[j+N] + L[i+N, j+xx]*Tr[j+xx]
                oz += L[i+xx, j]*Tr[j] + L[i+xx, j+N]*Tr[j+N] + L[i+xx, j+xx]*Tr[j+xx]
            o[i  ]  += ox
            o[i+N] += oy
            o[i+xx] += oz
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
        self.a  = radius
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
        cdef int i, ii, xx = 2*N
        cdef double dx, dy, dz, dr, idr, idr2, vv1, vv2, vx, vy, vz, radi=self.a
        cdef double muv = 1/(8*PI*self.eta), aa = self.a*self.a/3.0
        for i in prange(Nt, nogil=True):
            vx = 0.0; vy = 0.0; vz = 0.0;
            for ii in range(N):
                
                dx = rt[i]      - r[ii]
                dy = rt[i+Nt]   - r[ii+N]
                dz = rt[i+2*Nt] - r[ii+xx] 
                dr = sqrt( dx*dx + dy*dy + dz*dz )
                if dr>maskR:
                    idr = 1.0/dr
                    idr2= idr*idr
                    vv1 = (1+aa*idr2)*idr 
                    vv2 = (1-3*aa*idr2)*( F[ii]*dx + F[ii+N]*dy + F[ii+xx]*dz )*idr2*idr

                    vx += vv1*F[ii]     + vv2*dx
                    vy += vv1*F[ii+ N] + vv2*dy
                    vz += vv1*F[ii+ xx] + vv2*dz
            
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
        cdef int i, ii, xx = 2*N
        cdef double dx, dy, dz, dr, idr, idr3, vx, vy, vz, mur1 = 1.0/(8*PI*self.eta), radi=self.a
        for i in prange(Nt, nogil=True):
            vx = 0.0; vy = 0.0; vz = 0.0;
            for ii in range(N):
                dx = rt[i]      - r[ii]
                dy = rt[i+Nt]   - r[ii+N]
                dz = rt[i+2*Nt] - r[ii+xx] 
                dr = sqrt( dx*dx + dy*dy + dz*dz )
                if dr>maskR:
                    idr = 1.0/dr                
                    idr3 = idr*idr*idr
          
                    vx += ( dy*T[ii+xx] - dz*T[ii+N])*idr3
                    vy += ( dz*T[ii]    - dx*T[ii+xx])*idr3 
                    vz += ( dx*T[ii+N] - dy*T[ii]   )*idr3

            vv[i  ]      += vx*mur1
            vv[i + Nt]   += vy*mur1
            vv[i + 2*Nt] += vz*mur1
        return  


    cpdef flowField2s(self, double [:] vv, double [:] rt, double [:] r, double [:] S, double maskR=1.0):
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
        S: np.array
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
        cdef int i, ii, xx= 2*N, xx1= 3*N, xx2 = 4*N
        cdef double dx, dy, dz, dr, idr, idr3, aidr2, sxx, syy, sxy, sxz, syz, srr, srx, sry, srz
        cdef double aa = self.a**2, vv1, vv2, vx, vy, vz, mus = (28.0*self.a**3)/24, radi=self.a
        for i in prange(Nt, nogil=True):
            vx = 0.0;vy = 0.0; vz = 0.0;
            for ii in range(N):
                sxx = S[ii]
                syy = S[ii+N]
                sxy = S[ii+xx]
                sxz = S[ii+xx1]
                syz = S[ii+xx2]
                
                dx = rt[i]      - r[ii]
                dy = rt[i+Nt]   - r[ii+N]
                dz = rt[i+2*Nt] - r[ii+xx] 
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


    cpdef flowField3t(self, double [:] vv, double [:] rt, double [:] r, double [:] D, double maskR=1.0):
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
        D: np.array
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
        cdef double dx, dy, dz, dr, idr, idr3, Ddotidr, vx, vy, vz,mud1 = -1.0*(self.a**5)/10, radi=self.a
 
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
                    Ddotidr = (D[ii]*dx + D[ii+N]*dy + D[ii+2*N]*dz)*idr*idr

                    vx += (D[ii]      - 3.0*Ddotidr*dx )*idr3
                    vy += (D[ii+N]   - 3.0*Ddotidr*dy )*idr3
                    vz += (D[ii+2*N] - 3.0*Ddotidr*dz )*idr3
        
            vv[i]      += vx*mud1
            vv[i+Nt]   += vy*mud1
            vv[i+2*Nt] += vz*mud1
        
        return 


    cpdef flowField3s(self, double [:] vv, double [:] rt, double [:] r, double [:] G, double maskR=1):
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
        G: np.array
            An array of 3s mode of the slip
            An array of size 7*N,
        """

        cdef int N = self.N, Nt = self.Nt
        cdef int i, ii, 
        cdef double dx, dy, dz, dr, idr, idr5, idr7, radi=self.a
        cdef double aidr2, grrr, grrx, grry, grrz, gxxx, gyyy, gxxy, gxxz, gxyy, gxyz, gyyz
        for i in prange(Nt, nogil=True):
            for ii in range(N):
                gxxx = G[ii]
                gyyy = G[ii+N]
                gxxy = G[ii+2*N]
                gxxz = G[ii+3*N]
                gxyy = G[ii+4*N]
                gxyz = G[ii+5*N]
                gyyz = G[ii+6*N]
                dx = rt[i]      - r[ii]
                dy = rt[i+Nt]   - r[ii+N]
                dz = rt[i+2*Nt] - r[ii+2*N] 
                dr = sqrt( dx*dx + dy*dy + dz*dz )
                if dr>maskR:
                    idr = 1.0/sqrt( dx*dx + dy*dy + dz*dz )
                    idr5 = idr*idr*idr*idr*idr      
                    idr7 = idr5*idr*idr     
                    aidr2 = self.a*self.a*idr*idr

                    grrr = gxxx*dx*(dx*dx-3*dz*dz) + 3*gxxy*dy*(dx*dx-dz*dz) + gxxz*dz*(3*dx*dx-dz*dz) +\
                           3*gxyy*dx*(dy*dy-dz*dz) + 6*gxyz*dx*dy*dz + gyyy*dy*(dy*dy-3*dz*dz) +  gyyz*dz*(3*dy*dy-dz*dz) 
                    grrx = gxxx*(dx*dx-dz*dz) + gxyy*(dy*dy-dz*dz) +  2*gxxy*dx*dy + 2*gxxz*dx*dz  +  2*gxyz*dy*dz
                    grry = gxxy*(dx*dx-dz*dz) + gyyy*(dy*dy-dz*dz) +  2*gxyy*dx*dy + 2*gxyz*dx*dz  +  2*gyyz*dy*dz
                    grrz = gxxz*(dx*dx-dz*dz) + gyyz*(dy*dy-dz*dz) +  2*gxyz*dx*dy - 2*(gxxx+gxyy)*dx*dz  - 2*(gxxy+gyyy)*dy*dz

                    vv[i]      += 3*(1-(15.0/7)*aidr2)*grrx*idr5 - 15*(1-aidr2)*grrr*dx*idr7
                    vv[i+Nt]   += 3*(1-(15.0/7)*aidr2)*grry*idr5 - 15*(1-aidr2)*grrr*dy*idr7
                    vv[i+2*Nt] += 3*(1-(15.0/7)*aidr2)*grrz*idr5 - 15*(1-aidr2)*grrr*dz*idr7
        return


    cpdef flowField3a(self, double [:] vv, double [:] rt, double [:] r, double [:] V, double maskR=1):
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
        V: np.array
            An array of 3a mode of the slip
            An array of size 5*N,
        """

        cdef int N = self.N, Nt = self.Nt
        cdef int i, ii
        cdef double dx, dy, dz, dr, idr, idr5, vxx, vyy, vxy, vxz, vyz, vrx, vry, vrz, radi=self.a
 
        for i in prange(Nt, nogil=True):
            for ii in range(N):
                vxx = V[ii]
                vyy = V[ii+N]
                vxy = V[ii+2*N]
                vxz = V[ii+3*N]
                vyz = V[ii+4*N]
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


    cpdef flowField4a(self, double [:] vv, double [:] rt, double [:] r, double [:] M, double maskR=1):
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
        M: np.array
            An array of 4a mode of the slip
            An array of size 7*N,
        """

        cdef int N = self.N, Nt = self.Nt
        cdef int i, ii
        cdef double dx, dy, dz, idr, idr7, dr, radi=self.a
        cdef double mrrx, mrry, mrrz, mxxx, myyy, mxxy, mxxz, mxyy, mxyz, myyz
 
        for i in prange(Nt, nogil=True):
            for ii in range(N):
                mxxx = M[ii]
                myyy = M[ii+N]
                mxxy = M[ii+2*N]
                mxxz = M[ii+3*N]
                mxyy = M[ii+4*N]
                mxyz = M[ii+5*N]
                myyz = M[ii+6*N]
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

