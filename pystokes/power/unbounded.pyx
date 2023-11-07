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
cdef class PD:
    """
    Power Dissipation(PD)
    
    Methods in this class calculate the power dissipation
    using the inputs of -  power dissipation, arrays of positions, velocity or angular velocity, 
    along with an array of forces or torques or a slip mode

    The power dissipation is then update by each method. 
    
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
        self.gammaT = 6*PI*self.eta*self.a
        self.gammaR = 8*PI*self.eta*self.a**3
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


        cdef int N  = self.N, i, j, xx=2*N
        cdef double dx, dy, dz, idr, idr2, vx, vy, vz, vv1, vv2, aa = (2.0*self.a*self.a)/3.0 
        cdef double gT=self.gammaT, gg = -gT*gT, muv=self.muv        
        
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
                    vv2 = (1-3*aa*idr2)*( v[j]*dx + v[j+N]*dy + v[j+xx]*dz )*idr2*idr
                    vx += vv1*v[j]    + vv2*dx 
                    vy += vv1*v[j+N] + vv2*dy 
                    vz += vv1*v[j+xx] + vv2*dz 

            depsilon += v[i] * (gT*v[i] + gg*muv*vx)
            depsilon += v[i + N] * (gT*v[i+N] + gg*muv*vy)
            depsilon += v[i+xx] * (gT*v[i+xx] + gg*muv*vz)
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


        cdef int N = self.N, i, j, xx=2*N 
        cdef double dx, dy, dz, idr, idr3, vx, vy, vz
        cdef double muv=self.muv, gg=-self.gammaT*self.gammaR       
        
        for i in prange(N, nogil=True):
            vx=0; vy=0;   vz=0;
            for j in range(N):
                if i != j:
                    dx = r[i]    - r[j]
                    dy = r[i+N] - r[j+N]
                    dz = r[i+xx] - r[j+xx] 
                    idr = 1.0/sqrt( dx*dx + dy*dy + dz*dz )
                    idr3 = idr*idr*idr
                    vx += -(dy*o[j+xx] -o[j+N]*dz )*idr3
                    vy += -(dz*o[j]    -o[j+xx]*dx )*idr3
                    vz += -(dx*o[j+N]  -o[j]   *dy )*idr3

            depsilon += v[i] * gg * muv*vx
            depsilon += v[i+N] * gg * muv*vy
            depsilon += v[i+xx] * gg *muv*vz
        return depsilon
    
    
    cpdef frictionT2s(self, double depsilon, double [:] V1s, double [:] S, double [:] r):
        """
        Compute energy dissipation due to 2s mode of the slip :math:`\dot{\epsilon}=V^{1s}\cdot\gamma^{T,2s}\cdot V^{2s}`
        ...
        Parameters
        ----------
        depsilon: power dissipation
        V1s: np.array
            An array of 1s mode of velocities
            An array of size 3*N,
        S: np.array
            An array of 2s mode of the slip
            An array of size 5*N,
        r: np.array
            An array of positions
            An array of size 3*N,
        ----------
        """

        cdef int N = self.N, i, j, xx=2*N, xx1=3*N, xx2=4*N
        cdef double dx, dy, dz, dr, idr,  idr3
        cdef double aa=(self.a*self.a*8.0)/3.0, vv1, vv2, aidr2
        cdef double vx, vy, vz, 
        cdef double sxx, sxy, sxz, syz, syy, srr, srx, sry, srz, mus = (28.0*self.a**3)/24 
        cdef double gT = self.gammaT
 
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
            
            depsilon += -V1s[i] * gT * vx*mus
            depsilon += -V1s[i+N] * gT * vy*mus
            depsilon += -V1s[i+xx] * gT * vz*mus

        return depsilon


    cpdef frictionT3t(self, double depsilon, double [:] V1s, double [:] D, double [:] r):
        """
        Compute energy dissipation due to 3t mode of the slip :math:`\dot{\epsilon}=V^{1s}\cdot\gamma^{T,3t}\cdot V^{3t}`
        ...
        Parameters
        ----------
        depsilon: power dissipation
        V1s: np.array
            An array of 1s mode of velocities
            An array of size 3*N,
        D: np.array
            An array of 3t mode of the slip
            An array of size 3*N,
        r: np.array
            An array of positions
            An array of size 3*N,
        ----------
        """

        cdef int N = self.N, i, j, xx=2*N  
        cdef double dx, dy, dz, idr, idr3, Ddotidr, vx, vy, vz, mud = 3.0*self.a*self.a*self.a/5, mud1 = -1.0*(self.a**5)/10
        cdef double gammaT = self.gammaT
 
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
            
            depsilon += -V1s[i] * gammaT * mud1*vx
            depsilon += -V1s[i+N] * gammaT * mud1*vy
            depsilon += -V1s[i+xx] * gammaT * mud1*vz
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

        cdef int N = self.N, i, j, xx=2*N 
        cdef double dx, dy, dz, idr, idr3, ox, oy, oz, muv=self.muv
        cdef double gg= -self.gammaT*self.gammaR
 
        for i in prange(N, nogil=True):
            ox=0;   oy=0;   oz=0;
            for j in range(N):
                if i != j:
                    dx = r[i]    - r[j]
                    dy = r[i+N] - r[j+N]
                    dz = r[i+xx] - r[j+xx] 
                    idr = 1.0/sqrt( dx*dx + dy*dy + dz*dz )
                    idr3 = idr*idr*idr

                    ox += (v[j+N]*dz - v[j+xx]*dy )*idr3
                    oy += (v[j+xx]*dx - v[j]   *dz )*idr3
                    oz += (v[j]   *dy - v[j+N]*dx )*idr3
            depsilon += o[i] * gg * muv*ox
            depsilon += o[i+N] * gg * muv*oy
            depsilon += o[i+xx] * gg * muv*oz
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

        cdef int N = self.N, i, j, xx=2*N 
        cdef double dx, dy, dz, idr, idr3, Odotidr, ox, oy, oz, gR=self.gammaR, gg=-gR*gR, muv=self.muv 
 
        for i in prange(N, nogil=True):
            ox=0;   oy=0;   oz=0;
            for j in range(N):
                if i != j:
                    dx = r[i]      - r[j]
                    dy = r[i+N]   - r[j+N]
                    dz = r[i+xx] - r[j+xx] 
                    idr = 1.0/sqrt( dx*dx + dy*dy + dz*dz )
                    idr3 = idr*idr*idr
                    Odotidr = ( o[j]*dx + o[j+N]*dy + o[j+xx]*dz )*idr*idr

                    ox += ( o[j]    - 3*Odotidr*dx )*idr3
                    oy += ( o[j+N] - 3*Odotidr*dy )*idr3
                    oz += ( o[j+xx] - 3*Odotidr*dz )*idr3
            
            depsilon += o[i] * (gR*o[i]    - gg*0.5*muv*ox)
            depsilon += o[i+N] * (gR*o[i+N] - gg*0.5*muv*oy)
            depsilon += o[i+xx] * (gR*o[i+xx] - gg*0.5*muv*oz)
        return  depsilon
