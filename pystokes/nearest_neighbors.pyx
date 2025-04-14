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
    boxSize: float 
        Length of the box which is reperated periodicly in 3D
   """

    def __init__(self, radius=1, particles=1, viscosity=1.0, boxSize=10.0):
        self.a   = radius
        self.N  = particles
        self.eta = viscosity
        self.L   = boxSize
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
        """


        cdef int N  = self.N, i, j, xx=2*N
        cdef double dx, dy, dz, idr, idr2, vx, vy, vz, vv1, vv2, aa = (2.0*self.a*self.a)/3.0, L = self.L
        cdef double mu=self.mu, muv=self.muv        
        cdef int neighbors[2]
        
        for i in prange(N, nogil=True):
            vx=0; vy=0; vz=0;
            neighbors[0] = i - 1
            neighbors[1] = i + 1
            for j in neighbors:
                if i == 0 and j == i - 1:
                    j = N - 1
                    dx = r[i] - (r[j] - L)
                elif i == N - 1 and j == i + 1:
                    j = 0
                    dx = r[i] - (r[j] + L)
                else:
                    dx = r[i] - r[j]
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
            v[i+N]  += mu*F[i+N] + muv*vy
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
        cdef double dx, dy, dz, idr, idr3, vx, vy, vz, L = self.L
        cdef double muv=self.muv       
        cdef int neighbors[2]
        
        for i in prange(N, nogil=True):
            vx=0; vy=0; vz=0;
            neighbors[0] = i - 1
            neighbors[1] = i + 1
            for j in neighbors:
                if i == 0 and j == i - 1:
                    j = N - 1
                    dx = r[i] - (r[j] - L)
                elif i == N - 1 and j == i + 1:
                    j = 0
                    dx = r[i] - (r[j] + L)
                else:
                    dx = r[i] - r[j]
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
        cdef double dx, dy, dz, dr, idr,  idr3, L = self.L
        cdef double aa=(self.a*self.a*8.0)/3.0, vv1, vv2, aidr2
        cdef double vx, vy, vz, 
        cdef double sxx, sxy, sxz, syz, syy, srr, srx, sry, srz, mus = (28.0*self.a*self.a)/24 
        cdef int neighbors[2]
        
        for i in prange(N, nogil=True):
            vx=0; vy=0; vz=0;
            neighbors[0] = i - 1
            neighbors[1] = i + 1
            for j in neighbors:
                if i == 0 and j == i - 1:
                    j = N - 1
                    dx = r[i] - (r[j] - L)
                elif i == N - 1 and j == i + 1:
                    j = 0
                    dx = r[i] - (r[j] + L)
                else:
                    dx = r[i] - r[j]
                sxx = S[j]
                syy = S[j+N]
                sxy = S[j+xx]
                sxz = S[j+xx1]
                syz = S[j+xx2]
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
        cdef double L = self.L
        cdef int neighbors[2]
 
        for i in prange(N, nogil=True):
            vx=0; vy=0;   vz=0; 
            neighbors[0] = i - 1
            neighbors[1] = i + 1
            for j in neighbors:
                if i == 0 and j == i - 1:
                    j = N - 1
                    dx = r[i] - (r[j] - L)
                elif i == N - 1 and j == i + 1:
                    j = 0
                    dx = r[i] - (r[j] + L)
                else:
                    dx = r[i] - r[j]

                dy = r[i+N] - r[j+N]
                dz = r[i+xx] - r[j+xx] 
                idr = 1.0/sqrt( dx*dx + dy*dy + dz*dz )
                idr3 = idr*idr*idr 
                Ddotidr = (D[j]*dx + D[j+N]*dy + D[j+xx]*dz)*idr*idr

                vx += (D[j]    - 3.0*Ddotidr*dx )*idr3
                vy += (D[j+N] - 3.0*Ddotidr*dy )*idr3
                vz += (D[j+xx] - 3.0*Ddotidr*dz )*idr3
            
            v[i]   += mud1*vx
            v[i+N] += mud1*vy
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
        cdef double mud = 13.0*self.a*self.a*self.a/12, L = self.L
        cdef int neighbors[2]
 
        for i in prange(N, nogil=True):
            neighbors[0] = i - 1
            neighbors[1] = i + 1
            for j in neighbors:
                if i == 0 and j == i - 1:
                    j = N - 1
                    dx = r[i] - (r[j] - L)
                elif i == N - 1 and j == i + 1:
                    j = 0
                    dx = r[i] - (r[j] + L)
                else:
                    dx = r[i] - r[j]
                vxx = V[j]
                vyy = V[j+N]
                vxy = V[j+2*N]
                vxz = V[j+3*N]
                vyz = V[j+4*N]

                dy = r[i+N]   - r[j+N]
                dz = r[i+2*N] - r[j+2*N] 
                idr = 1.0/sqrt( dx*dx + dy*dy + dz*dz)
                idr5 = idr*idr*idr*idr*idr
                vrx = vxx*dx +  vxy*dy + vxz*dz  
                vry = vxy*dx +  vyy*dy + vyz*dz  
                vrz = vxz*dx +  vyz*dy - (vxx+vyy)*dz 

                v[i]     += mud * (vry * dz - vrz * dy) * idr5
                v[i+N]   += mud * (vrz * dx - vrx * dz) * idr5
                v[i+2*N] += mud * (vrx * dy - vry * dx) * idr5
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
        cdef double mud = -363/8 * self.a ** 4, L = self.L
        cdef int neighbors[2]
 
        for i in prange(N, nogil=True):
            neighbors[0] = i - 1
            neighbors[1] = i + 1
            for j in neighbors:
                if i == 0 and j == i - 1:
                    j = N - 1
                    dx = r[i] - (r[j] - L)
                elif i == N - 1 and j == i + 1:
                    j = 0
                    dx = r[i] - (r[j] + L)
                else:
                    dx = r[i] - r[j]

                mxxx = M[j]
                myyy = M[j+N]
                mxxy = M[j+2*N]
                mxxz = M[j+3*N]
                mxyy = M[j+4*N]
                mxyz = M[j+5*N]
                myyz = M[j+6*N]

                dy = r[i+N]   - r[j+N]
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

        cdef int N = self.N, i, j, xx=2*N 
        cdef double dx, dy, dz, idr, idr3, ox, oy, oz, muv=self.muv, L = self.L
        cdef int neighbors[2]
 
        for i in prange(N, nogil=True):
            ox=0;   oy=0;   oz=0;
            neighbors[0] = i - 1
            neighbors[1] = i + 1
            for j in neighbors:
                if i == 0 and j == i - 1:
                    j = N - 1
                    dx = r[i] - (r[j] - L)
                elif i == N - 1 and j == i + 1:
                    j = 0
                    dx = r[i] - (r[j] + L)
                else:
                    dx = r[i] - r[j]
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
        cdef double dx, dy, dz, idr, idr3, Tdotidr, ox, oy, oz, mur=self.mur, muv=self.muv, L = self.L
        cdef int neighbors[2]
 
        for i in prange(N, nogil=True):
            ox=0;   oy=0;   oz=0;
            neighbors[0] = i - 1
            neighbors[1] = i + 1
            for j in neighbors:
                if i == 0 and j == i - 1:
                    j = N - 1
                    dx = r[i] - (r[j] - L)
                elif i == N - 1 and j == i + 1:
                    j = 0
                    dx = r[i] - (r[j] + L)
                else:
                    dx = r[i] - r[j]
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
        cdef double dx, dy, dz, idr, idr5, ox, oy, oz, L = self.L
        cdef double sxx, sxy, sxz, syz, syy, srr, srx, sry, srz, mus = (28.0*self.a*self.a)/24
        cdef int neighbors[2]
 
        for i in prange(N, nogil=True):
            ox=0;   oy=0;   oz=0;
            neighbors[0] = i - 1
            neighbors[1] = i + 1
            for j in neighbors:
                if i == 0 and j == i - 1:
                    j = N - 1
                    dx = r[i] - (r[j] - L)
                elif i == N - 1 and j == i + 1:
                    j = 0
                    dx = r[i] - (r[j] + L)
                else:
                    dx = r[i] - r[j]
                sxx = S[j]
                syy = S[j+N]
                sxy = S[j+2*N]
                sxz = S[j+3*N]
                syz = S[j+4*N]
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
        cdef double mud = 13.0 / 24.0 * self.a * self.a * self.a, L = self.L
        cdef int neighbors[2]
 
        for i in prange(N, nogil=True):
            neighbors[0] = i - 1
            neighbors[1] = i + 1
            for j in neighbors:
                if i == 0 and j == i - 1:
                    j = N - 1
                    dx = r[i] - (r[j] - L)
                elif i == N - 1 and j == i + 1:
                    j = 0
                    dx = r[i] - (r[j] + L)
                else:
                    dx = r[i] - r[j]

                vxx = V[j]
                vyy = V[j+N]
                vxy = V[j+2*N]
                vxz = V[j+3*N]
                vyz = V[j+4*N]

                dy = r[i+N]   - r[j+N]
                dz = r[i+2*N] - r[j+2*N] 
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
        cdef double mud = 363 / 80 * self.a ** 4, L = self.L
        cdef int neighbors[2]
 
        for i in prange(N, nogil=True):
            neighbors[0] = i - 1
            neighbors[1] = i + 1
            for j in neighbors:
                if i == 0 and j == i - 1:
                    j = N - 1
                    dx = r[i] - (r[j] - L)
                elif i == N - 1 and j == i + 1:
                    j = 0
                    dx = r[i] - (r[j] + L)
                else:
                    dx = r[i] - r[j]

                mxxx = M[j]
                myyy = M[j+N]
                mxxy = M[j+2*N]
                mxxz = M[j+3*N]
                mxyy = M[j+4*N]
                mxyz = M[j+5*N]
                myyz = M[j+6*N]

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
                
                o[i]     += mud * (15 * mrrx * idr7 - 35 * mrrr * dx * idr9)
                o[i+N]   += mud * (15 * mrry * idr7 - 35 * mrrr * dy * idr9)
                o[i+2*N] += mud * (15 * mrrz * idr7 - 35 * mrrr * dz * idr9)
        return