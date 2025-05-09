cimport cython
from libc.math cimport sqrt
from cython.parallel import prange
import numpy as np
cimport numpy as np
cdef double PI = 3.14159265359



@cython.boundscheck(False)
@cython.cdivision(True)
@cython.nonecheck(False)
@cython.wraparound(False)
cdef class Rbm:                         
    """
    Rigid body motion (RBM)
    
    ...

    ----------
    radius: float
        Radius of the particles (a).    
    particles: int
        Number of particles (N)
    viscosity: float 
        Viscosity of the fluid (eta)
    """
 

    def __init__(self, b, N, eta):
        self.b   = b                 # radius of the particles
        self.N   = N                # number of particles
        self.eta = eta                # number of particles

    cpdef mobilityTT(self, double [:] v, double [:] r, double [:] F, double H):
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
        H: float 
            Height of the Hele-Shaw cell 
        """

        cdef int i, j, N=self.N, Z=2*N
        cdef double dx, dy, dz, idr, idr2, Fdotidr2, h2, hsq, tempF
        cdef double vx, vy, vz, tH = 2*H
        cdef double mu = 1.0/(6*PI*self.eta*self.b), mu1 = mu*self.b*0.75, a2=self.b*self.b/3.0
        cdef double fac0 = 3/(PI*self.eta*H*H*H), fac1, fac2
 
        for i in prange(N, nogil=True):
            vx=0; vy=0; vz=0;
            for j in range(N):
                dx = r[i]    - r[j]
                dy = r[i+N]  - r[j+N]
                h2  =  2*r[j+Z]; hsq=r[j+Z]*r[j+Z]
                if i!=j:
                    idr = 1.0/sqrt( dx*dx + dy*dy )
                    idr2=idr*idr
                    Fdotidr2 = (F[j] * dx + F[j+N] * dy )*idr2
                    #
                    fac1 = fac0*(H-r[i+Z])*(H-r[j+Z])*r[i+Z]
                    vx += fac1*(0.5*F[j]    + Fdotidr2*dx)*idr2 
                    vy += fac1*(0.5*F[j+N] + Fdotidr2*dy)*idr2 
                    
            v[i  ]  += mu*F[i]    + mu1*vx 
            v[i+N] += mu*F[i+N] + mu1*vy
        return 
    
    
    cpdef propulsionT2s(self, double [:] v, double [:] r, double [:] V2s, double H):
        """
        Compute velocity due to 2s mode of the slip 
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
            An array of forces
            An array of size 5*N,
        H: float 
            Height of the Hele-Shaw cell 
        """

        cdef int N=self.N, i, j, Z=2*N, xx1=3*N , xx2=4*N   
        cdef double dx, dy, dz, idr, idr2, idr4, idr6, idr7, aidr2, trS, h2, hsq
        cdef double sxx, syy, szz, sxy, syx, syz, szy, sxz, szx, srr, srx, sry, srz
        cdef double Sljrlx, Sljrly, Sljrlz, Sljrjx, Sljrjy, Sljrjz 
        cdef double vx, vy, vz, mus =-(28.0*self.b**3)/24, tH = 2*H
        cdef double fac0 = 3/(PI*self.eta*H*H*H), fac1, fac2

        for i in prange(N, nogil=True):
            vx=0; vy=0;   vz=0;
            for j in  range(N):
                h2 = 2*r[j+Z]; hsq = r[j+Z]*r[j+Z];
                sxx = V2s[j]  ; syy = V2s[j+N]; szz = -sxx-syy;
                sxy = V2s[j+Z]; syx = sxy;
                sxz = V2s[j+xx1]; szx = sxz;
                syz = V2s[j+xx2]; szy = syz;
                dx = r[i]   - r[j]
                dy = r[i+N] - r[j+N]
                if i!=j:
                    idr  = 1.0/sqrt( dx*dx + dy*dy);
                    idr4 = idr*idr*idr*idr; idr6 = idr4*idr*idr; 
                    fac1 = fac0*(H-r[i+Z])*(H-r[j+Z])*r[i+Z]

                    srx = fac1*(sxx*dx +  sxy*dy )*idr4; 
                    sry = fac1*(sxy*dx +  syy*dy )*idr4;
                    srz = fac1*(sxz*dx +  syz*dy )*idr4;
                    srr = fac1*(sxx*dx*dx + syy*dy*dy + 2*sxy*dx*dy )*idr6;
                    
                    vx += -2*srx + 4*srr*dx;
                    vy += -2*sry + 4*srr*dy;
                     
            v[i]    += vx*mus
            v[i+N] += vy*mus
            v[i+Z] += vz*mus
        return

   
    cpdef propulsionT3t(self, double [:] v, double [:] r, double [:] V3t, double H):
        """
        Compute velocity due to 3t mode of the slip 
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
            An array of forces
            An array of size 3*N,
        H: float 
            Height of the Hele-Shaw cell 
        """

        cdef int N=self.N, i, j, Z=2*N
        cdef double dx, dy, dz, idr, idr2, idr5, V3tdotidr, tempD, hsq, h2, tH=2*H
        cdef double vx, vy, vz, mud = 3.0*self.b*self.b*self.b/5
        cdef double fac0 = 6/(PI*self.eta*H*H*H), fac1, fac2

        for i in prange(N, nogil=True):
            vx=0; vy=0; vz=0;
            for j in range(N):
                dx = r[i]    - r[j]
                dy = r[i+N]  - r[j+N]
                h2  =  2*r[j+Z]
                if i!=j:
                    idr = 1.0/sqrt( dx*dx + dy*dy)
                    idr2=idr*idr
                    V3tdotidr = (V3t[j]*dx + V3t[j+N]*dy)*idr2
                    
                    fac1 = -fac0*r[j+Z]*(H-r[j+Z])
                    vx += fac1*(0.5*V3t[j]    - V3tdotidr*dx)*idr2
                    vy += fac1*(0.5*V3t[j+N] - V3tdotidr*dy)*idr2
            v[i]    += vx*mud
            v[i+N] += vy*mud
            v[i+Z] += vz*mud





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

    """

    def __init__(self, radius=1, particles=1, viscosity=1, gridpoints=32):
        self.b  = radius
        self.N = particles
        self.Nt = gridpoints
        self.eta= viscosity


    cpdef flowField1s(self, double [:] vv, double [:] rt, double [:] r, double [:] F, double H):
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
        H: float 
            Height of the Hele-Shaw cell 
        """

        cdef int i, j, N=self.N, Z=2*N, Nt=self.Nt
        cdef double dx, dy, dz, idr, idr3, idr5, Fdotidr, h2, hsq, tempF
        cdef double vx, vy, vz, tH = 2*H
        cdef double mu = 1.0/(6*PI*self.eta*self.b), mu1 = mu*self.b*0.75, a2=self.b*self.b/3.0
        cdef double fac0 = 3/(PI*self.eta*H*H*H), fac1, fac2
 
        for i in prange(Nt, nogil=True):
            vx=0; vy=0; vz=0;
            for j in range(N):
                dx = rt[i]    - r[j]
                dy = rt[i+Nt]  - r[j+N]
                idr = 1.0/sqrt( dx*dx + dy*dy )
                Fdotidr = (F[j] * dx + F[j+N] * dy )*idr*idr
                #
                fac1 = (H-rt[i+2*Nt])*(H-r[j+Z])*rt[i+2*Nt]*r[j+Z]
                vx += fac1*(0.5*F[j]    - Fdotidr*dx)*idr*idr 
                vy += fac1*(0.5*F[j+N] - Fdotidr*dy)*idr*idr
                    
            vv[i  ]  += mu1*vx 
            vv[i+Nt] += mu1*vy 



    cpdef flowField2s(self, double [:] vv, double [:] rt, double [:] r, double [:] V2s, double H):
        """
        Compute flow field at field points due to 2s mode of th slip 
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
        H: float 
            Height of the Hele-Shaw cell 
        """
        cdef int i, j, N=self.N, Z=2*N, Nt=self.Nt, xx1=3*N, xx2=4*N
        cdef double dx, dy, dz, idr, idr2, idr4, idr6, idr7, aidr2, trS, h2, hsq
        cdef double sxx, syy, szz, sxy, syx, syz, szy, sxz, szx, srr, srx, sry, srz
        cdef double Sljrlx, Sljrly, Sljrlz, Sljrjx, Sljrjy, Sljrjz 
        cdef double vx, vy, vz, mus =-(28.0*self.b**3)/24, tH = 2*H
        cdef double fac0 = 3/(PI*self.eta*H*H*H), fac1, fac2

 
        for i in prange(Nt, nogil=True):
            vx=0; vy=0; vz=0;
            for j in range(N):
                sxx = V2s[j]  ; syy = V2s[j+N]; szz = -sxx-syy;
                sxy = V2s[j+Z]; syx = sxy;
                sxz = V2s[j+xx1]; szx = sxz;
                syz = V2s[j+xx2]; szy = syz;
                dx = rt[i]    - r[j]
                dy = rt[i+Nt] - r[j+N]
                idr  = 1.0/sqrt( dx*dx + dy*dy);
                idr4 = idr*idr*idr*idr; idr6 = idr4*idr*idr; 
                fac1 = (H-rt[i+2*Nt])*(H-r[j+Z])*rt[i+2*Nt]*idr4

                srx = fac1*(sxx*dx    +  sxy*dy ); 
                sry = fac1*(sxy*dx    +  syy*dy );
                srr = fac1*(sxx*dx*dx + syy*dy*dy + 2*sxy*dx*dy )*idr*idr;
                
                vx += -2*srx + 4*srr*dx;
                vy += -2*sry + 4*srr*dy;
                    
            vv[i  ]  += mus*vx 
            vv[i+Nt] += mus*vy


    cpdef flowField3t(self, double [:] vv, double [:] rt, double [:] r, double [:] V3t, double H):
        """
        Compute flow field at field points due to 3t mode of th slip 
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
            An array of 2s mode of the slip 
            An array of size r*N,
        H: float 
            Height of the Hele-Shaw cell 
        """
        cdef int i, j, N=self.N, Z=2*N, Nt=self.Nt
        cdef double dx, dy, dz, idr, idr2, idr5, Fdotidr, h2, hsq, tempF
        cdef double vx, vy, vz, tH = 2*H, V3tdotidr
        cdef double mu = 1.0/(6*PI*self.eta*self.b), mu1 = mu*self.b*0.75, a2=self.b*self.b/3.0
        cdef double fac0 = 3/(PI*self.eta*H*H*H), fac1, fac2
 
        for i in prange(Nt, nogil=True):
            vx=0; vy=0; vz=0;
            for j in range(N):
                dx = rt[i]   - r[j]
                dy = rt[i+Nt] - r[j+N]
                idr = 1.0/sqrt( dx*dx + dy*dy)
                idr2=idr*idr
                V3tdotidr = (V3t[j]*dx + V3t[j+N]*dy)*idr2
                
                fac1 = fac0*(H-rt[i+2*Nt])*(H-r[j+Z])*rt[i+2*Nt]*r[j+Z]
                vx += fac1*(0.5*V3t[j]    - V3tdotidr*dx)*idr2
                vy += fac1*(0.5*V3t[j+N]  - V3tdotidr*dy)*idr2
                    
            vv[i  ]  += mu1*vx 
            vv[i+Nt] += mu1*vy
