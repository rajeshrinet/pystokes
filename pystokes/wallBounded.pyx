cimport cython
from libc.math cimport sqrt
from cython.parallel import prange
cdef double PI = 3.14159265359
import numpy as np
cimport numpy as np

@cython.boundscheck(False)
@cython.cdivision(True)
@cython.nonecheck(False)
@cython.wraparound(False)
cdef class Rbm:
    """
    Rigid body motion (RBM)
    
    ...

    Parameters
    ----------
    radius: float
        Radius of the particles (a)
    particles: int
        Number of particles (Np) 
    viscosity: float  
        Viscosity of the fluid (eta)
    
    """

    def __init__(self, radius=1, particles=1, viscosity=1.0):
        self.a   = radius
        self.Np  = particles
        self.eta = viscosity
        self.mu  = 1.0/(6*PI*self.eta*self.a)

        self.Mobility = np.zeros( (3*self.Np, 3*self.Np), dtype=np.float64)


    cpdef mobilityTT(self, double [:] v, double [:] r, double [:] F):
        """
        Compute velocity due to body forces using :math:`v=\mu^{TT}\cdot F` 
        ...

        Parameters
        ----------
        v: np.array
            An array of velocities
            An array of size 3*Np,
        r: np.array
            An array of positions
            An array of size 3*Np,
        F: np.array
            An array of forces
            An array of size 3*Np,
    
        Examples
        --------
        An example of the RBM 

        >>> import pystokes, numpy as np, matplotlib.pyplot as plt
        >>> # particle radius, self-propulsion speed, number and fluid viscosity
        >>> b, vs, Np, eta = 1.0, 1.0, 128, 0.1
        >>> #initialise
        >>> r = pystokes.utils.initialCondition(Np)  # initial random distribution of positions
        >>> p = np.zeros(3*Np); p[2*Np:3*Np] = -1    # initial orientation of the colloids
        >>> 
        >>> rbm = pystokes.wallBounded.Rbm(radius=b, particles=Np, viscosity=eta)
        >>> force = pystokes.forceFields.Forces(particles=Np)
        >>> 
        >>> def rhs(rp):
        >>>     # assign fresh values at each time step
        >>>     r = rp[0:3*Np];   p = rp[3*Np:6*Np]
        >>>     F, v, o = np.zeros(3*Np), np.zeros(3*Np), np.zeros(3*Np)
        >>> 
        >>>     force.lennardJonesWall(F, r, lje=0.01, ljr=5, wlje=1.2, wljr=3.4)
        >>>     rbm.mobilityTT(v, r, F)
        >>>     return np.concatenate( (v,o) )
        >>> 
        >>> # simulate the resulting system
        >>> Tf, Npts = 150, 200
        >>> pystokes.utils.simulate(np.concatenate((r,p)), 
        >>>    Tf,Npts,rhs,integrator='odeint', filename='crystallization')
        """

        cdef int i, j, Np=self.Np, xx=2*Np
        cdef double dx, dy, dz, idr, idr3, idr5, Fdotidr, h2, hsq, tempF
        cdef double vx, vy, vz
        cdef double mu=self.mu, mu1 = mu*self.a*0.75, a2=self.a*self.a/3.0

        for i in prange(Np, nogil=True):
            vx=0; vy=0; vz=0;
            for j in range(Np):
                dx = r[i]    - r[j]
                dy = r[i+Np]  - r[j+Np]
                h2  =  2*r[j+xx]; hsq=r[j+xx]*r[j+xx]
                if i!=j:
                    dz = r[i+xx]  - r[j+xx]
                    idr = 1.0/sqrt( dx*dx + dy*dy + dz*dz )
                    idr3=idr*idr*idr
                    Fdotidr = (F[j] * dx + F[j+Np] * dy + F[j+xx] * dz)*idr*idr
                    #
                    vx += (F[j]   +Fdotidr*dx)*idr + a2*(2*F[j]   -6*Fdotidr*dx)*idr3
                    vy += (F[j+Np]+Fdotidr*dy)*idr + a2*(2*F[j+Np]-6*Fdotidr*dy)*idr3
                    vz += (F[j+xx]+Fdotidr*dz)*idr + a2*(2*F[j+xx]-6*Fdotidr*dz)*idr3

                    ##contributions from the image
                    dz = r[i+xx] + r[j+xx]
                    idr = 1.0/sqrt( dx*dx + dy*dy + dz*dz )
                    idr3 = idr*idr*idr
                    idr5 = idr3*idr*idr
                    Fdotidr = ( F[j]*dx + F[j+Np]*dy + F[j+xx]*dz )*idr*idr

                    vx += -(F[j]   +Fdotidr*dx)*idr - a2*(2*F[j]   -6*Fdotidr*dx)*idr3
                    vy += -(F[j+Np]+Fdotidr*dy)*idr - a2*(2*F[j+Np]-6*Fdotidr*dy)*idr3
                    vz += -(F[j+xx]+Fdotidr*dz)*idr - a2*(2*F[j+xx]-6*Fdotidr*dz)*idr3

                    tempF  = -F[j+xx]     # F_i = M_ij F_j, reflection of the strength
                    Fdotidr = ( F[j]*dx + F[j+Np]*dy + tempF*dz )*idr*idr

                    vx += -h2*(dz*(F[j]   - 3*Fdotidr*dx) + tempF*dx)*idr3
                    vy += -h2*(dz*(F[j+Np]- 3*Fdotidr*dy) + tempF*dy)*idr3
                    vz += -h2*(dz*(tempF  - 3*Fdotidr*dz) + tempF*dz)*idr3 + h2*Fdotidr*idr

                    vx += hsq*( 2*F[j]   - 6*Fdotidr*dx )*idr3
                    vy += hsq*( 2*F[j+Np]- 6*Fdotidr*dy )*idr3
                    vz += hsq*( 2*tempF  - 6*Fdotidr*dz )*idr3

                    vx += 12*a2*dz*( dz*F[j]   - 5*dz*Fdotidr*dx + 2*tempF*dx )*idr5
                    vy += 12*a2*dz*( dz*F[j+Np]- 5*dz*Fdotidr*dy + 2*tempF*dy )*idr5
                    vz += 12*a2*dz*( dz*tempF  - 5*dz*Fdotidr*dz + 2*tempF*dz )*idr5

                    vx += -h2*6*a2*(dz*F[j]   -5*Fdotidr*dx*dz + tempF*dx)*idr5
                    vy += -h2*6*a2*(dz*F[j+Np]-5*Fdotidr*dy*dz+ tempF*dy)*idr5
                    vz += -h2*6*a2*(dz*tempF  -5*Fdotidr*dz*dz + tempF*dz)*idr5 -6*a2*h2*Fdotidr*idr3
                else:
                    ''' self contribution from the image point'''
                    dz = r[i+xx] + r[j+xx]
                    idr = 1.0/dz
                    idr3 = idr*idr*idr
                    idr5 = idr3*idr*idr
                    Fdotidr = F[j+xx]*dz*idr*idr

                    vx += -(F[j]               )*idr - a2*(2*F[j]    )*idr3
                    vy += -(F[j+Np]            )*idr - a2*(2*F[j+Np] )*idr3
                    vz += -(F[j+xx]+Fdotidr*dz )*idr - a2*(2*F[j+xx] - 6*Fdotidr*dz )*idr3

                    tempF  = -F[j+xx]     # F_i = M_ij F_j, reflection of the strength
                    Fdotidr = tempF*dz*idr*idr

                    vx += -h2*(dz*(F[j]  ) + tempF*dx)*idr3
                    vy += -h2*(dz*(F[j+Np]) + tempF*dy)*idr3
                    vz += -h2*(dz*(tempF -3*Fdotidr*dz) + tempF*dz)*idr3 + h2*Fdotidr*idr

                    vx += hsq*( 2*F[j]   )*idr3
                    vy += hsq*( 2*F[j+Np])*idr3
                    vz += hsq*( 2*tempF  - 6*Fdotidr*dz )*idr3

                    vx += 12*a2*dz*( dz*F[j]   )*idr5
                    vy += 12*a2*dz*( dz*F[j+Np])*idr5
                    vz += 12*a2*dz*( dz*tempF  - 5*dz*Fdotidr*dz + 2*tempF*dz )*idr5

                    vx += -h2*6*a2*(dz*F[j]   )*idr5
                    vy += -h2*6*a2*(dz*F[j+Np])*idr5
                    vz += -h2*6*a2*(dz*tempF  -5*Fdotidr*dz*dz+tempF*dz)*idr5 -6*a2*h2*Fdotidr*idr3

            v[i  ]  += mu*F[i]    + mu1*vx
            v[i+Np] += mu*F[i+Np] + mu1*vy
            v[i+xx] += mu*F[i+xx] + mu1*vz
        return


    cpdef mobilityTR(self, double [:] v, double [:] r, double [:] T):
        """
        Compute velocity due to body torque using :math:`v=\mu^{TR}\cdot T` 
        ...

        Parameters
        ----------
        v: np.array
            An array of velocities
            An array of size 3*Np,
        r: np.array
            An array of positions
            An array of size 3*Np,
        T: np.array
            An array of torques
            An array of size 3*Np,
        """

        cdef int Np = self.Np, i, j, xx=2*Np
        cdef double dx, dy, dz, idr, idr3, rlz, Tdotidr, h2,
        cdef double vx, vy, vz,
        cdef double mu1 = 1/(8*PI*self.eta)

        for i in prange(Np, nogil=True):
            vx=0; vy=0; vz=0;
            for j in range(Np):
                dx = r[i]   - r[j]
                dy = r[i+Np] - r[j+Np]
                h2 = 2*r[i+xx]
                if i != j:
                    #contributions from the source
                    dz = r[i+xx] - r[j+xx]
                    idr = 1.0/sqrt( dx*dx + dy*dy + dz*dz )
                    idr3 = idr*idr*idr

                    vx += (T[j+Np]*dz - T[j+xx]*dy )*idr3
                    vy += (T[j+xx]*dx - T[j]   *dz )*idr3
                    vz += (T[j]   *dy - T[j+Np]*dx )*idr3

                    #contributions from the image
                    dz = r[i+xx] + r[j+xx]
                    idr = 1.0/sqrt( dx*dx + dy*dy + dz*dz )
                    idr3 = idr*idr*idr

                    vx += -(T[j+Np]*dz - T[j+xx]*dy )*idr3
                    vy += -(T[j+xx]*dx - T[j]   *dz )*idr3
                    vz += -(T[j]   *dy - T[j+Np]*dx )*idr3

                    rlz = (dx*T[j+Np] - dy*T[j])*idr*idr
                    vx += (h2*(T[j+Np]-3*rlz*dx) + 6*dz*dx*rlz)*idr3
                    vy += (h2*(-T[j]  -3*rlz*dy) + 6*dz*dy*rlz)*idr3
                    vz += (h2*(       -3*rlz*dz) + 6*dz*dz*rlz)*idr3
                else:
                    ''' the self contribution from the image point'''
                    dz = r[i+xx] + r[j+xx]
                    idr = 1.0/dz
                    idr3 = idr*idr*idr

                    vx += -(T[j+Np]*dz )*idr3
                    vy += -(- T[j] *dz )*idr3

                    vx += h2*T[j+Np]*idr3
                    vy += -h2*T[j]*idr3

            v[i]    -= mu1*vx
            v[i+Np] -= mu1*vy
            v[i+xx] -= mu1*vz
        return


    cpdef propulsionT2s(self, double [:] v, double [:] r, double [:] S):
        """
        Compute velocity due to 2s mode of the slip :math:`v=\pi^{T,2s}\cdot S` 
        ...

        Parameters
        ----------
        v: np.array
            An array of velocities
            An array of size 3*Np,
        r: np.array
            An array of positions
            An array of size 3*Np,
        S: np.array
            An array of 2s mode of the slip
            An array of size 5*Np,
        """

        cdef int Np=self.Np, i, j, xx=2*Np, xx1=3*Np , xx2=4*Np
        cdef double dx, dy, dz, idr, idr2, idr3, idr5, idr4, aidr2, trS, h2, hsq
        cdef double sxx, syy, szz, sxy, syx, syz, szy, sxz, szx, srr, srx, sry, srz
        cdef double Sljrlx, Sljrly, Sljrlz, Sljrjx, Sljrjy, Sljrjz
        cdef double vx, vy, vz, mus=(28.0*self.a**3)/24

        for i in prange(Np, nogil=True):
            vx=0; vy=0;   vz=0;
            for j in  range(Np):
                h2 = 2*r[j+xx]; hsq = r[j+xx]*r[j+xx];
                sxx = S[j]  ; syy = S[j+Np]; szz = -sxx-syy;
                sxy = S[j+xx]; syx = sxy;
                sxz = S[j+xx1]; szx = sxz;
                syz = S[j+xx2]; szy = syz;
                dx = r[i]   - r[j]
                dy = r[i+Np] - r[j+Np]
                if i!=j:
                    dz = r[i+xx] - r[j+xx]
                    idr  = 1.0/sqrt( dx*dx + dy*dy + dz*dz );
                    idr2 = idr*idr; idr3 = idr2*idr; idr5 = idr3*idr2;
                    srx = (sxx*dx +  sxy*dy + sxz*dz );
                    sry = (sxy*dx +  syy*dy + syz*dz );
                    srz = (sxz*dx +  syz*dy + szz*dz );
                    srr = srx*dx + sry*dy + srz*dz

                    ## contributions from the source
                    vx += 3*srr*dx*idr5;
                    vy += 3*srr*dy*idr5;
                    vz += 3*srr*dz*idr5;

                    ## contributions from the image
                    dz = r[i+xx]+r[j+xx]
                    idr  = 1.0/sqrt( dx*dx + dy*dy + dz*dz )
                    idr2 = idr*idr; idr3 = idr2*idr; idr5 = idr3*idr2;

                    
                    #reflecting the first index of stresslet, S_jl M_lm
                    sxz=-sxz; syz=-syz; szz=-szz;     trS=sxx+syy+szz;
                    Sljrlx = sxx*dx +  sxy*dx + sxz*dx ;
                    Sljrly = syx*dy +  syy*dy + syz*dy ;
                    Sljrlz = szx*dz +  szy*dz + szz*dz ;
                    
                    Sljrjx = sxx*dx +  sxy*dy + sxz*dz ;
                    Sljrjy = syx*dx +  syy*dy + syz*dz ;
                    Sljrjz = szx*dx +  szy*dy + szz*dz ;
                    srx = sxx*dx + sxy*dy + sxz*dz
                    sry = syx*dx + syy*dy + syz*dz
                    srz = sxz*dx + syz*dy + szz*dz
                    srr = (srx*dx + sry*dy + srz*dz)*idr2

                    vx += (-Sljrlx + Sljrjx + trS*dx -3*srr*dx)*idr3
                    vy += (-Sljrly + Sljrjy + trS*dy -3*srr*dy)*idr3
                    vz += (-Sljrlz + Sljrjz + trS*dz -3*srr*dz)*idr3

                    vx += -2*(dz*(sxz-3*srz*dx*idr2)+ szz*dx)*idr3;
                    vy += -2*(dz*(syz-3*srz*dy*idr2)+ szz*dy)*idr3;
                    vz += -2*(dz*(szz-3*srz*dz*idr2)+ szz*dz - srz)*idr3;

                    vx += h2*( sxz-3*srz*dx*idr2)*idr3;
                    vy += h2*( syz-3*srz*dy*idr2)*idr3;
                    vz += h2*( szz-3*srz*dz*idr2)*idr3;

                    #reflecting both the indices of stresslet, S_jl M_lm M_jk
                    szx = -szx ; szy = -szy; szz = -szz;
                    srx = (sxx*dx +  sxy*dy + sxz*dz )
                    sry = (sxy*dx +  syy*dy + syz*dz )
                    srz = (sxz*dx +  syz*dy + szz*dz )
                    srr = (srx*dx + sry*dy + srz*dz)*idr2

                    vx += h2*( (dz*(-6*srx + 15*srr*dx)-3*srz*dx)*idr5 + (sxz)*idr3) ;
                    vy += h2*( (dz*(-6*sry + 15*srr*dy)-3*srz*dy)*idr5 + (syz)*idr3) ;
                    vz += h2*( (dz*(-6*srz + 15*srr*dz)-3*srz*dz)*idr5 + (szz + 3*srr)*idr3);

                    vx += hsq*(12*srx - 30*srr*dx)*idr5
                    vy += hsq*(12*sry - 30*srr*dy)*idr5
                    vz += hsq*(12*srz - 30*srr*dz)*idr5

                else:
                    ''' the self contribution from the image point'''
                    dz = r[i+xx]+r[j+xx]
                    idr  = 1.0/dz
                    idr2 = idr*idr; idr3 = idr2*idr; idr4 = idr3*idr

                    #reflecting the first index of stresslet, S_jl M_lm
                    sxz=-sxz; syz=-syz; szz=-szz;     trS=sxx+syy+szz;

                    Sljrlz = szx*dz +  szy*dz + szz*dz ;
                    Sljrjx = sxz*dz
                    Sljrjy = syz*dz
                    Sljrjz = szz*dz

                    vx += Sljrjx*idr3 ;
                    vy += Sljrjy*idr3 ;
                    vz += (-Sljrlz + Sljrjz + trS*dz -3*szz*dz)*idr3 ;

                    vx += -2*dz*sxz*idr3;
                    vy += -2*dz*syz*idr3;
                    vz += -2*dz*(-2*szz)*idr3;

                    vx +=      h2*sxz*idr3;
                    vy +=      h2*syz*idr3;
                    vz += h2*(-2*szz)*idr3;

                    #reflecting both the indices of stresslet, S_jl M_lm M_jk
                    szx = -szx ; syz = -syz; szz = -szz;

                    vx += h2*dz*(-5*sxz)*idr4
                    vy += h2*dz*(-5*syz)*idr4
                    vz += h2*dz*(10*szz)*idr4

                    vx += hsq*12*sxz*idr4
                    vy += hsq*12*syz*idr4
                    vz += hsq*(-18*szz)*idr4

            v[i]    += vx*mus
            v[i+Np] += vy*mus
            v[i+xx] += vz*mus
        return


    cpdef propulsionT3t(self, double [:] v, double [:] r, double [:] D):
        """
        Compute velocity due to 3t mode of the slip :math:`v=\pi^{T,3t}\cdot D` 
        ...

        Parameters
        ----------
        v: np.array
            An array of velocities
            An array of size 3*Np,
        r: np.array
            An array of positions
            An array of size 3*Np,
        D: np.array
            An array of 3t mode of the slip
            An array of size 3*Np,
        """

        cdef int Np=self.Np, i, j, xx=2*Np
        cdef double dx, dy, dz, idr, idr3, idr5, Ddotidr, tempD, hsq, h2
        cdef double vx, vy, vz, mud = 3.0*self.a*self.a*self.a/5, mu1 = -1.0*(self.a**5)/10

        for i in prange(Np, nogil=True):
            vx=0; vy=0; vz=0;
            for j in range(Np):
                dx = r[i]    - r[j]
                dy = r[i+Np]  - r[j+Np]
                h2  =  2*r[j+xx]
                if i!=j:
                    dz = r[i+xx] - r[j+xx]
                    idr = 1.0/sqrt( dx*dx + dy*dy + dz*dz )
                    idr3=idr*idr*idr
                    Ddotidr = (D[j]*dx + D[j+Np]*dy + D[j+xx]*dz)*idr*idr
                    #
                    vx += (2*D[j]    - 6*Ddotidr*dx)*idr3
                    vy += (2*D[j+Np] - 6*Ddotidr*dy)*idr3
                    vz += (2*D[j+xx] - 6*Ddotidr*dz)*idr3

                    ##contributions from the image
                    dz = r[i+xx] + r[j+xx]
                    idr = 1.0/sqrt( dx*dx + dy*dy + dz*dz )
                    idr3 = idr*idr*idr
                    idr5 = idr3*idr*idr
                    Ddotidr = (D[j]*dx + D[j+Np]*dy + D[j+xx]*dz)*idr*idr

                    vx += -(2*D[j]    - 6*Ddotidr*dx )*idr3
                    vy += -(2*D[j+Np] - 6*Ddotidr*dy )*idr3
                    vz += -(2*D[j+xx] - 6*Ddotidr*dz )*idr3

                    tempD = -D[j+xx]     # D_i = M_ij D_j, reflection of the strength
                    Ddotidr = ( D[j]*dx + D[j+Np]*dy + tempD*dz )*idr*idr

                    vx += 12*dz*( dz*D[j]   - 5*dz*Ddotidr*dx + 2*tempD*dx )*idr5
                    vy += 12*dz*( dz*D[j+Np]- 5*dz*Ddotidr*dy + 2*tempD*dy )*idr5
                    vz += 12*dz*( dz*tempD  - 5*dz*Ddotidr*dz + 2*tempD*dz )*idr5

                    vx += -6*h2*(dz*D[j]   -5*Ddotidr*dx*dz + tempD*dx)*idr5
                    vy += -6*h2*(dz*D[j+Np]-5*Ddotidr*dy*dz + tempD*dy)*idr5
                    vz += -6*h2*(dz*tempD  -5*Ddotidr*dz*dz + tempD*dz)*idr5 -6*h2*Ddotidr*idr3

                else:
                    ''' self contribution from the image point'''
                    dz = r[i+xx] + r[j+xx]
                    idr = 1.0/dz
                    idr3 = idr*idr*idr
                    idr5 = idr3*idr*idr
                    Ddotidr = D[j+xx]*dz*idr*idr

                    vx += -(2*D[j]    )*idr3
                    vy += -(2*D[j+Np] )*idr3
                    vz += -(2*D[j+xx] - 6*Ddotidr*dz )*idr3

                    tempD = -D[j+xx]     # D_i = M_ij D_j, reflection of the strength
                    Ddotidr = tempD*dz*idr*idr

                    vx += 12*dz*( dz*D[j]   )*idr5
                    vy += 12*dz*( dz*D[j+Np])*idr5
                    vz += 12*dz*( dz*tempD  - 5*dz*Ddotidr*dz + 2*tempD*dz )*idr5

                    vx += -6*h2*(dz*D[j]   )*idr5
                    vy += -6*h2*(dz*D[j+Np])*idr5
                    vz += -6*h2*(dz*tempD  -5*Ddotidr*dz*dz + tempD*dz)*idr5 -6*h2*Ddotidr*idr3

            v[i  ]  += mu1*vx
            v[i+Np] += mu1*vy
            v[i+xx] += mu1*vz
        return
   


    ## Angular Velocities
    cpdef mobilityRT(self, double [:] o, double [:] r, double [:] F):
        """
        Compute angular velocity due to body forces using :math:`o=\mu^{RT}\cdot F` 
        ...

        Parameters
        ----------
        o: np.array
            An array of angular velocities
            An array of size 3*Np,
        r: np.array
            An array of positions
            An array of size 3*Np,
        F: np.array
            An array of forces
            An array of size 3*Np,
        """


        cdef int Np = self.Np, i, j, xx=2*Np
        cdef double dx, dy, dz, idr, idr3, rlz, Fdotidr, h2
        cdef double ox, oy, oz, mu1=1.0/(8*PI*self.eta)

        for i in prange(Np, nogil=True):
            ox=0; oy=0; oz=0;
            for j in range(Np):
                dx = r[i]   - r[j]
                dy = r[i+Np] - r[j+Np]
                h2 = 2*r[i+xx]
                if i != j:
                    #contributions from the source
                    dz = r[i+xx] - r[j+xx]
                    idr = 1.0/sqrt( dx*dx + dy*dy + dz*dz )
                    idr3 = idr*idr*idr

                    ox += (F[j+Np]*dz - F[j+xx]*dy )*idr3
                    oy += (F[j+xx]*dx - F[j]   *dz )*idr3
                    oz += (F[j]   *dy - F[j+Np]*dx )*idr3

                    #contributions from the image
                    dz = r[i+xx] + r[j+xx]
                    idr = 1.0/sqrt( dx*dx + dy*dy + dz*dz )
                    idr3 = idr*idr*idr
                    rlz = (dx*F[j+Np] - dy*F[j])*idr*idr

                    ox += -(F[j+Np]*dz - F[j+xx]*dy )*idr3
                    oy += -(F[j+xx]*dx - F[j]   *dz )*idr3
                    oz += -(F[j]   *dy - F[j+Np]*dx )*idr3

                    ox += (h2*(F[j+Np]-3*rlz*dx) + 6*dz*dx*rlz)*idr3
                    oy += (h2*(-F[j]  -3*rlz*dy) + 6*dz*dy*rlz)*idr3
                    oz += (h2*(       -3*rlz*dz) + 6*dz*dz*rlz)*idr3

                else:
                    ''' the self contribution from the image point'''
                    dz = r[i+xx] + r[j+xx]
                    idr = 1.0/dz
                    idr3 = idr*idr*idr


                    ox += (h2*F[j+Np])*idr3
                    oy += (h2*-F[j]  )*idr3

            o[i  ]  += mu1*ox
            o[i+Np] += mu1*oy
            o[i+xx] += mu1*oz
        return


    cpdef mobilityRR(self, double [:] o, double [:] r, double [:] T):
        """
        Compute angular velocity due to body torques using :math:`o=\mu^{RR}\cdot T` 
        ...

        Parameters
        ----------
        o: np.array
            An array of angular velocities
            An array of size 3*Np,
        r: np.array
            An array of positions
            An array of size 3*Np,
        T: np.array
            An array of forces
            An array of size 3*Np,
        """

        cdef int Np=self.Np, i, j, xx=2*Np
        cdef double dx, dy, dz, idr, idr3, idr5, Tdotidr, tempT, hsq, h2
        cdef double ox, oy, oz, mut=1.0/(8*PI*self.eta*self.a**3), mu1=1.0/(8*4*PI*self.eta)

        for i in prange(Np, nogil=True):
            ox=0; oy=0; oz=0;
            for j in range(Np):
                dx = r[i]    - r[j]
                dy = r[i+Np]  - r[j+Np]
                h2  =  2*r[j+xx]
                if i!=j:
                    dz = r[i+xx] - r[j+xx]
                    idr = 1.0/sqrt( dx*dx + dy*dy + dz*dz )
                    idr3=idr*idr*idr
                    Tdotidr = (T[j]*dx + T[j+Np]*dy + T[j+xx]*dz)*idr*idr
                    #
                    ox += (2*T[j]    - 6*Tdotidr*dx)*idr3
                    oy += (2*T[j+Np] - 6*Tdotidr*dy)*idr3
                    oz += (2*T[j+xx] - 6*Tdotidr*dz)*idr3

                    ##contributions from the image
                    dz = r[i+xx] + r[j+xx]
                    idr = 1.0/sqrt( dx*dx + dy*dy + dz*dz )
                    idr3 = idr*idr*idr
                    idr5 = idr3*idr*idr
                    Tdotidr = (T[j]*dx + T[j+Np]*dy + T[j+xx]*dz)*idr*idr

                    ox += -(2*T[j]    - 6*Tdotidr*dx )*idr3
                    oy += -(2*T[j+Np] - 6*Tdotidr*dy )*idr3
                    oz += -(2*T[j+xx] - 6*Tdotidr*dz )*idr3

                    tempT = -T[j+xx]     # D_i = M_ij D_j, reflection of the strength
                    Tdotidr = ( T[j]*dx + T[j+Np]*dy + tempT*dz )*idr*idr

                    ox += 12*dz*( dz*T[j]   - 5*dz*Tdotidr*dx + 2*tempT*dx )*idr5
                    oy += 12*dz*( dz*T[j+Np]- 5*dz*Tdotidr*dy + 2*tempT*dy )*idr5
                    oz += 12*dz*( dz*tempT  - 5*dz*Tdotidr*dz + 2*tempT*dz )*idr5

                    ox += -6*h2*(dz*T[j]   -5*Tdotidr*dx*dz + tempT*dx)*idr5
                    oy += -6*h2*(dz*T[j+Np]-5*Tdotidr*dy*dz + tempT*dy)*idr5
                    oz += -6*h2*(dz*tempT  -5*Tdotidr*dz*dz + tempT*dz)*idr5 -6*h2*Tdotidr*idr3

                else:

                    ''' self contribution from the image point'''
                    dz = r[i+xx] + r[j+xx]
                    idr = 1.0/dz
                    idr3 = idr*idr*idr

                    ox += 4*T[j]*idr3
                    oy += 4*T[j+Np]*idr3
                    oz += 16*T[j+xx]*idr3

            o[i  ]  += mut*T[i  ]  - mu1*ox
            o[i+Np] += mut*T[i+Np] - mu1*oy
            o[i+xx] += mut*T[i+xx] - mu1*oz
        return
        
        
    cpdef propulsionR2s(self, double [:] o, double [:] r, double [:] S):
        cdef int Np=self.Np, i, j, xx=2*Np, xx1=3*Np , xx2=4*Np
        cdef double dx, dy, dz, idr, idr2, idr3, idr5, idr7
        cdef double sxx, syy, szz, sxy, syx, syz, szy, sxz, szx, srr, srx, sry, srz
        cdef double Sljrlx, Sljrly, Sljrlz, Sljrjx, Sljrjy, Sljrjz, rlz, smr3, smkrk3
        cdef double ox, oy, oz, mus = (28.0*self.a**3)/24, h

        for i in prange(Np, nogil=True):
            ox=0;   oy=0;   oz=0;
            sxz = S[i+xx1];
            syz = S[i+xx2];
            for j in range(Np):
                sxx = S[j    ]  ;
                syy = S[j+Np ];
                sxy = S[j+xx ];
                sxz = S[j+xx1];
                syz = S[j+xx2];
                if i != j:
                    #syx = sxy;
                    # szx = sxz;
                    # szy = syz;
                    #szz = -sxx-syy;
                    dx = r[i]   - r[j]
                    dy = r[i+Np] - r[j+Np]
                    h=r[j+xx]
                    dz = r[i+xx] - r[j+xx]
                    idr = 1.0/sqrt( dx*dx + dy*dy + dz*dz )
                    idr5 = idr*idr*idr*idr*idr
                    srx = sxx*dx +  sxy*dy + sxz*dz
                    sry = sxy*dx +  syy*dy + syz*dz
                    srz = sxz*dx +  syz*dy - (sxx+syy)*dz

                    ox += 3*(sry*dz - srz*dy )*idr5
                    oy += 3*(srz*dx - srx*dz )*idr5
                    oz += 3*(srx*dy - sry*dx )*idr5
                    #
                    ####contributions from the image
                    dz = r[i+xx] + r[j+xx]
                    idr = 1.0/sqrt( dx*dx + dy*dy + dz*dz )
                    idr5=idr*idr*idr*idr*idr
                    srx = sxx*dx +  sxy*dy + sxz*dz
                    sry = sxy*dx +  syy*dy + syz*dz
                    srz = sxz*dx +  syz*dy - (sxx+syy)*dz

                    ox += 3*(-sry*dz + srz*dy )*idr5
                    oy += 3*(-srz*dx + srx*dz )*idr5
                    oz += 3*(-srx*dy + sry*dx )*idr5

                    #rlz = (dx*syz - dy*sxz)*idr*idr
                    #ox += (2*syz  - 6*rlz*dx)*idr3
                    #oy += (-2*sxz - 6*rlz*dy)*idr3
                    #oz += (       - 6*rlz*dz)*idr3
                    #
                    ###reflecting the second index of stresslet, S_jl M_lm
                    #sxz=-sxz; syz=-syz; szz=-szz;
                    #
                    #smr3 = sxz*dy-syz*dx
                    #ox += 6*(dz*(sxx*dy-syx*dx) + smr3*dx)*idr5
                    #ox += 6*(dz*(sxy*dy-syy*dx) + smr3*dy)*idr5
                    #oz += 6*(dz*(sxz*dy-syz*dx) + smr3*dz)*idr5

                    #Sljrjx = sxx*dx +  sxy*dy + sxz*dz ;
                    #Sljrjy = syx*dx +  syy*dy + syz*dz ;
                    #Sljrjz = szx*dx +  szy*dy + szz*dz ;
                    #srr = (sxx*dx*dx + syy*dy*dy + szz*dz*dz +  2*sxy*dx*dy)*idr2 ;
                    #
                    #ox += 2*syz*idr3 - 3*(Sljrjy*dz - Sljrjz*dy)*idr5
                    #oy += 2*szx*idr3 - 3*(Sljrjz*dx - Sljrjx*dz)*idr5
                    #oz +=              3*(Sljrjx*dy - Sljrjy*dx)*idr5
                    #
                    #smkrk3 = 30*(dx*Sljrjy-dy*Sljrjx)*idr5*idr2
                    #ox += (h+dz)*smkrk3*dx
                    #ox += (h+dz)*smkrk3*dy
                    #oz += (h+dz)*smkrk3*dz

                    #ox += 2*syz*idr3  - 3*(Sljrjy*dz - Sljrjz*dy )*idr5
                    #oy += -2*szy*idr3 - 3*(Sljrjz*dx - Sljrjx*dz )*idr5
                    #oz +=             - 3*(Sljrjx*dy - Sljrjy*dx )*idr5
                    #
                    #ox += 6*h*(-Sljrjy + (sxx*dy-syx*dx))*idr5
                    #oy += 6*h*(Sljrjx  + (sxy*dy-syy*dx))*idr5
                    #ox +=                (sxz*dy-syz*dx)*idr5
                else:
                    ### self contributions from the image
                    dz = r[i+xx] + r[j+xx]
                    idr = 1.0/dz
                    idr3 = idr*idr*idr
                    # idr5 = idr3*idr*idr
                    ox += 3*(-syz)*idr3
                    oy += 3*(sxz )*idr3
                    ##reflecting the second index of stresslet, S_jl M_lm
                    #sxz=-sxz; syz=-syz; szz=-szz;
                    #
                    #Sljrjx = sxz*dz ;
                    #Sljrjy = syz*dz ;
                    #Sljrjz = szz*dz ;
                    #srr = szz*dz*dz*idr2 ;
                    #
                    #ox += 2*syz*idr3 - 3*(Sljrjy*dz)*idr5
                    #oy += 2*szx*idr3 - 3*(- Sljrjx*dz)*idr5
                    #
                    #smkrk3 = 30*(dx*Sljrjy-dy*Sljrjx)*idr5*idr2
                    #oz += (h+dz)*smkrk3*dz

                    #ox += 2*syz*idr3  - 3*(Sljrjy*dz )*idr5
                    #oy += -2*szy*idr3 - 3*(- Sljrjx*dz )*idr5
                    #
                    #ox += 6*h*(-Sljrjy )*idr5
                    #oy += 6*h*(Sljrjx  )*idr5

            o[i]    += ox
            o[i+Np] += oy
            o[i+xx] += oz
        return


    cpdef propulsionR3a(  self, double [:] o, double [:] r, double [:] V):
        '''
        approximation involved
        '''
        cdef int Np = self.Np, i, j, xx=2*Np
        cdef double dx, dy, dz, idr, idr2, idr4, idr5, vxx, vyy, vxy, vxz, vyz, vrr, vrx, vry, vrz
        cdef double ox, oy, oz

        for i in prange(Np, nogil=True):
            ox=0; oy=0; oz=0;
            for j in range(Np):
                vxx = V[j]
                vyy = V[j+Np]
                vxy = V[j+xx]
                vxz = V[j+3*Np]
                vyz = V[j+4*Np]
                if i != j:
                    pass
                    #dx = r[i]      - r[j]
                    #dy = r[i+Np]   - r[j+Np]
                    #dz = r[i+xx] - r[j+xx]
                    #idr = 1.0/sqrt( dx*dx + dy*dy + dz*dz )
                    #idr5 = idr*idr*idr*idr*idr
                    #vrr = (vxx*(dx*dx-dz*dz) + vyy*(dy*dy-dz*dz) +  2*vxy*dx*dy + 2*vxz*dx*dz  +  2*vyz*dy*dz)*idr*idr
                    #vrx = vxx*dx +  vxy*dy + vxz*dz
                    #vry = vxy*dx +  vyy*dy + vyz*dz
                    #vrz = vxz*dx +  vyz*dy - (vxx+vyy)*dz

                    #ox +=  5*( 6*vrx- 15*vrr*dx )*idr5
                    #oy +=  5*( 6*vry- 15*vrr*dy )*idr5
                    #oz +=  5*( 6*vrz- 15*vrr*dz )*idr5
                    #
                    ###contribution from the image point
                    #dz = r[i+xx] + r[j+xx]
                    #idr = 1.0/dz
                    #idr5 = idr*idr*idr*idr*idr
                    #vrr = (vxx*(dx*dx-dz*dz) + vyy*(dy*dy-dz*dz) +  2*vxy*dx*dy + 2*vxz*dx*dz  +  2*vyz*dy*dz)*idr*idr
                    #vrx = vxx*dx +  vxy*dy + vxz*dz
                    #vry = vxy*dx +  vyy*dy + vyz*dz
                    #vrz = vxz*dx +  vyz*dy - (vxx+vyy)*dz

                    #ox +=  -5*( 6*vrx- 15*vrr*dx )*idr5
                    #oy +=  -5*( 6*vry- 15*vrr*dy )*idr5
                    #oz +=  -5*( 6*vrz- 15*vrr*dz )*idr5
                else :
                    dz = r[i+xx] + r[j+xx]
                    idr = 1.0/dz
                    idr4 = idr*idr*idr*idr
                    #vrr = 45*(vxx+vyy)
                    #ox += -5*(6*vxz*dz)*idr4
                    #oy += -5*(6*vyz*dz)*idr4
                    oz += -45*(vxx+vyy)*idr4
            #o[i  ]  += ox
            #o[i+Np] += oy
            o[i+xx] += oz
        return


    cpdef propulsionR4a(self, double [:] o, double [:] r, double [:] M):
        '''
        approximation involved
        '''
        cdef int Np = self.Np, i, j, xx=2*Np
        cdef double ox, oy, oz, dx, dy, dz, idr, idr5, idr7, idr9, mu1 = 1
        cdef double mrrr, mrrx, mrry, mrrz, mxxx, myyy, mxxy, mxxz, mxyy, mxyz, myyz

        for i in prange(Np, nogil=True):
            ox=0; oy=0; oz=0;
            for j in range(Np):
                mxxx = M[j]
                myyy = M[j+Np]
                mxxy = M[j+2*Np]
                mxxz = M[j+3*Np]
                mxyy = M[j+4*Np]
                mxyz = M[j+5*Np]
                myyz = M[j+6*Np]
                if i != j:
                    pass
                    #dx = r[i]      - r[j]
                    #dy = r[i+Np]   - r[j+Np]
                    #dz = r[i+2*Np] - r[j+2*Np]
                    #idr = 1.0/sqrt( dx*dx + dy*dy + dz*dz )
                    #idr7 = idr*idr*idr*idr*idr*idr*idr
                    #idr9 = idr7*idr*idr
                    #
                    #mrrr = mxxx*dx*(dx*dx-3*dz*dz) + 3*mxxy*dy*(dx*dx-dz*dz) + mxxz*dz*(3*dx*dx-dz*dz) +\
                    #   3*mxyy*dx*(dy*dy-dz*dz) + 6*mxyz*dx*dy*dz + myyy*dy*(dy*dy-3*dz*dz) +  myyz*dz*(3*dy*dy-dz*dz)
                    #mrrx = mxxx*(dx*dx-dz*dz) + mxyy*(dy*dy-dz*dz) +  2*mxxy*dx*dy + 2*mxxz*dx*dz  +  2*mxyz*dy*dz
                    #mrry = mxxy*(dx*dx-dz*dz) + myyy*(dy*dy-dz*dz) +  2*mxyy*dx*dy + 2*mxyz*dx*dz  +  2*myyz*dy*dz
                    #mrrz = mxxz*(dx*dx-dz*dz) + myyz*(dy*dy-dz*dz) +  2*mxyz*dx*dy - 2*(mxxx+mxyy)*dx*dz  - 2*(mxxy+myyy)*dy*dz

                    #ox += -21*mrrr*dx*idr9 + 9*mrrx*idr7
                    #oy += -21*mrrr*dy*idr9 + 9*mrry*idr7
                    #oz += -21*mrrr*dz*idr9 + 9*mrrz*idr7
                    #''' self contribution from the image point'''
                    #dz = r[i+xx] + r[j+xx]
                    #idr = 1.0/dz
                    #idr7 = idr*idr*idr*idr*idr*idr*idr
                    #idr9 = idr7*idr*idr
                    #
                    #mrrr = mxxx*dx*(dx*dx-3*dz*dz) + 3*mxxy*dy*(dx*dx-dz*dz) + mxxz*dz*(3*dx*dx-dz*dz) +\
                    #   3*mxyy*dx*(dy*dy-dz*dz) + 6*mxyz*dx*dy*dz + myyy*dy*(dy*dy-3*dz*dz) +  myyz*dz*(3*dy*dy-dz*dz)
                    #mrrx = mxxx*(dx*dx-dz*dz) + mxyy*(dy*dy-dz*dz) +  2*mxxy*dx*dy + 2*mxxz*dx*dz  +  2*mxyz*dy*dz
                    #mrry = mxxy*(dx*dx-dz*dz) + myyy*(dy*dy-dz*dz) +  2*mxyy*dx*dy + 2*mxyz*dx*dz  +  2*myyz*dy*dz
                    #mrrz = mxxz*(dx*dx-dz*dz) + myyz*(dy*dy-dz*dz) +  2*mxyz*dx*dy - 2*(mxxx+mxyy)*dx*dz  - 2*(mxxy+myyy)*dy*dz

                    #ox += 21*mrrr*dx*idr9 - 9*mrrx*idr7
                    #oy += 21*mrrr*dy*idr9 - 9*mrry*idr7
                    #oz += 21*mrrr*dz*idr9 - 9*mrrz*idr7
                else:
                    ''' self contribution from the image point'''
                    dz = r[i+xx] + r[j+xx]
                    idr = 1.0/dz
                    idr5 = idr*idr*idr*idr*idr

                    #mrrr = mxxz +  myyz
                    #mrrx = mxxx*(-dz*dz) + mxyy*(-dz*dz)
                    #mrry = mxxy*(-dz*dz) + myyy*(-dz*dz)
                    #mrrz = mxxz*(-dz*dz) + myyz*(-dz*dz)

                    #ox #+= - 9*mrrx*idr7
                    #oy #+= - 9*mrry*idr7
                    oz +=  12*(mxxz+myyz)*idr5

            #o[i  ]  += ox
            #o[i+Np] += oy
            o[i+xx] += oz
        return

    ## Noise
    cpdef noiseTT(self, double [:] v, double [:] r):
        """
        Compute translation Brownian motion 
        ...

        Parameters
        ----------
        v: np.array
            An array of velocities
            An array of size 3*Np,
        r: np.array
            An array of positions
            An array of size 3*Np,
        """

        cdef int i, j, Np=self.Np, xx=2*Np
        cdef double dx, dy, dz, idr, h2, hsq, idr2, idr3, idr4, idr5
        cdef double mu=self.mu, mu1=2*mu*self.a*0.75, a2=self.a*self.a/3.0
        cdef double vx, vy, vz, mm=1/(.75*self.a)

        cdef double [:, :] M = self.Mobility
        cdef double [:]    Fr = np.random.normal(size=3*Np)


        for i in prange(Np, nogil=True):
            for j in range(Np):
                dx = r[i]    - r[j]
                dy = r[i+Np] - r[j+Np]
                h2=2*r[j+xx]; hsq=r[j+xx]*r[j+xx]
                if i!=j:
                    dz = r[i+xx] - r[j+xx]
                    idr = 1.0/sqrt( dx*dx + dy*dy + dz*dz )
                    idr2=idr*idr;  idr3=idr*idr*idr
                    dx = dx*idr; dy=dy*idr; dz=dz*idr
                    #
                    M[i,    j   ] = (1 + dx*dx)*idr + a2*(2 - 6*dx*dx)*idr3
                    M[i+Np, j+Np] = (1 + dy*dy)*idr + a2*(2 - 6*dy*dy)*idr3
                    M[i+xx, j+xx] = (1 + dz*dz)*idr + a2*(2 - 6*dz*dz)*idr3
                    M[i,    j+Np] = (    dx*dy)*idr + a2*(  - 6*dx*dy)*idr3
                    M[i,    j+xx] = (    dx*dz)*idr + a2*(  - 6*dx*dz)*idr3
                    M[i+Np, j+xx] = (    dy*dz)*idr + a2*(  - 6*dy*dz)*idr3

                    ###contributions from the image
                    dz = r[i+xx] + r[j+xx]
                    idr = 1.0/sqrt( dx*dx + dy*dy + dz*dz )
                    dx = dx*idr; dy=dy*idr; dz=dz*idr
                    idr2=idr*idr;  idr3=idr2*idr;  idr4=idr3*idr; idr5=idr4*idr

                    M[i,    j   ] += -(1 + dx*dx)*idr - a2*(2 - 6*dx*dx)*idr3
                    M[i+Np, j+Np] += -(1 + dy*dy)*idr - a2*(2 - 6*dy*dy)*idr3
                    M[i+xx, j+xx] += -(1 + dz*dz)*idr - a2*(2 - 6*dz*dz)*idr3
                    M[i,    j+Np] += -(    dx*dy)*idr - a2*(  - 6*dx*dy)*idr3
                    M[i,    j+xx] += -(    dx*dz)*idr - a2*(  - 6*dx*dz)*idr3
                    M[i+Np, j+xx] += -(    dy*dz)*idr - a2*(  - 6*dy*dz)*idr3

                    M[i,    j   ] += -h2*(dz*(1 - 3*dx*dx))*idr2
                    M[i+Np, j+Np] += -h2*(dz*(1 - 3*dy*dy))*idr2
                    M[i+xx, j+xx] += +h2*(dz*(1 - 3*dz*dz))*idr2
                    M[i,    j+Np] += -h2*(dz*(  - 3*dx*dy))*idr2
                    M[i,    j+xx] += +h2*(dz*(  - 3*dx*dz) + dx)*idr2
                    M[i+Np, j+xx] += +h2*(dz*(  - 3*dy*dz) + dy)*idr2

                    M[i,    j   ] += hsq*(2 - 6*dx*dx)*idr3
                    M[i+Np, j+Np] += hsq*(2 - 6*dy*dy)*idr3
                    M[i+xx, j+xx] +=-hsq*(2 - 6*dz*dz)*idr3
                    M[i,    j+Np] += hsq*(  - 6*dx*dy)*idr3
                    M[i,    j+xx] +=-hsq*(  - 6*dx*dz)*idr3
                    M[i+Np, j+xx] +=-hsq*(  - 6*dy*dz)*idr3

                    M[i,    j   ] += 12*a2*dz*(dz - 5*dx*dx)*idr3
                    M[i+Np, j+Np] += 12*a2*dz*(dz - 5*dy*dy)*idr3
                    M[i+xx, j+xx] +=-12*a2*dz*(dz - 5*dz*dz + 2*dz)*idr3
                    M[i,    j+Np] += 12*a2*dz*(   - 5*dx*dy)*idr3
                    M[i,    j+xx] +=-12*a2*dz*(   - 5*dx*dz + 2*dx)*idr3
                    M[i+Np, j+xx] +=-12*a2*dz*(   - 5*dy*dz + 2*dy)*idr3

                    M[i,    j   ] += h2*6*a2*(-dz   + 5*dx*dx*dz)*idr4
                    M[i+Np, j+Np] += h2*6*a2*(-dz   + 5*dy*dy*dz)*idr4
                    M[i+xx, j+xx] +=-h2*6*a2*(-3*dz + 5*dz*dz*dz)*idr4
                    M[i,    j+Np] += h2*6*a2*(        5*dx*dy*dz)*idr4
                    M[i,    j+xx] +=-h2*6*a2*(-dx   + 5*dx*dz*dz)*idr4
                    M[i+Np, j+xx] +=-h2*6*a2*(-dy   + 5*dy*dz*dz)*idr4
                else:
                    # one-body mobility
                    M[i,    j   ] = mm
                    M[i+Np, j+Np] = mm
                    M[i+xx, j+xx] = mm
                    M[i,    j+Np] = 0
                    M[i,    j+xx] = 0
                    M[i+Np, j+xx] = 0

                    ##self contribtion from the image point
                    dz = r[i+xx] + r[j+xx]
                    idr = 1.0/sqrt( dx*dx + dy*dy + dz*dz )
                    dx = dx*idr; dy=dy*idr; dz=dz*idr
                    idr2=idr*idr;  idr3=idr2*idr;  idr4=idr3*idr; idr5=idr4*idr

                    M[i,    j   ] += -(1 + dx*dx)*idr - a2*(2 - 6*dx*dx)*idr3
                    M[i+Np, j+Np] += -(1 + dy*dy)*idr - a2*(2 - 6*dy*dy)*idr3
                    M[i+xx, j+xx] += -(1 + dz*dz)*idr - a2*(2 - 6*dz*dz)*idr3
                    M[i,    j+Np] += -(    dx*dy)*idr - a2*(  - 6*dx*dy)*idr3
                    M[i,    j+xx] += -(    dx*dz)*idr - a2*(  - 6*dx*dz)*idr3
                    M[i+Np, j+xx] += -(    dy*dz)*idr - a2*(  - 6*dy*dz)*idr3

                    M[i,    j   ] += -h2*(dz*(1 - 3*dx*dx))*idr2
                    M[i+Np, j+Np] += -h2*(dz*(1 - 3*dy*dy))*idr2
                    M[i+xx, j+xx] += +h2*(dz*(1 - 3*dz*dz))*idr2
                    M[i,    j+Np] += -h2*(dz*(  - 3*dx*dy))*idr2
                    M[i,    j+xx] += +h2*(dz*(  - 3*dx*dz) + dx)*idr2
                    M[i+Np, j+xx] += +h2*(dz*(  - 3*dy*dz) + dy)*idr2

                    M[i,    j   ] += hsq*(2 - 6*dx*dx)*idr3
                    M[i+Np, j+Np] += hsq*(2 - 6*dy*dy)*idr3
                    M[i+xx, j+xx] +=-hsq*(2 - 6*dz*dz)*idr3
                    M[i,    j+Np] += hsq*(  - 6*dx*dy)*idr3
                    M[i,    j+xx] +=-hsq*(  - 6*dx*dz)*idr3
                    M[i+Np, j+xx] +=-hsq*(  - 6*dy*dz)*idr3

                    M[i,    j   ] += 12*a2*dz*(dz - 5*dx*dx)*idr3
                    M[i+Np, j+Np] += 12*a2*dz*(dz - 5*dy*dy)*idr3
                    M[i+xx, j+xx] +=-12*a2*dz*(dz - 5*dz*dz + 2*dz)*idr3
                    M[i,    j+Np] += 12*a2*dz*(   - 5*dx*dy)*idr3
                    M[i,    j+xx] +=-12*a2*dz*(   - 5*dx*dz + 2*dx)*idr3
                    M[i+Np, j+xx] +=-12*a2*dz*(   - 5*dy*dz + 2*dy)*idr3

                    M[i,    j   ] += h2*6*a2*(-dz   + 5*dx*dx*dz)*idr4
                    M[i+Np, j+Np] += h2*6*a2*(-dz   + 5*dy*dy*dz)*idr4
                    M[i+xx, j+xx] +=-h2*6*a2*(-3*dz + 5*dz*dz*dz)*idr4
                    M[i,    j+Np] += h2*6*a2*(        5*dx*dy*dz)*idr4
                    M[i,    j+xx] +=-h2*6*a2*(-dx   + 5*dx*dz*dz)*idr4
                    M[i+Np, j+xx] +=-h2*6*a2*(-dy   + 5*dy*dz*dz)*idr4

        for i in prange(Np, nogil=True):
            for j in range(Np):
                M[i,    j   ] = mu1*M[i,    j   ]
                M[i+Np, j+Np] = mu1*M[i+Np, j+Np]
                M[i+xx, j+xx] = mu1*M[i+xx, j+xx]
                M[i,    j+Np] = mu1*M[i,    j+Np]
                M[i,    j+xx] = mu1*M[i,    j+xx]
                M[i+Np, j+xx] = mu1*M[i+Np, j+xx]

                M[i+Np, j   ] =     M[i,    j+Np]
                M[i+xx, j   ] =     M[i,    j+xx]
                M[i+xx, j+Np] =     M[i+Np, j+xx]

        cdef double [:, :] L = np.linalg.cholesky(self.Mobility)

        for i in prange(Np, nogil=True):
            vx=0; vy=0; vz=0;
            for j in range(Np):
                vx += L[i   , j]*Fr[j] + L[i   , j+Np]*Fr[j+Np] + L[i   , j+xx]*Fr[j+xx]
                vy += L[i+Np, j]*Fr[j] + L[i+Np, j+Np]*Fr[j+Np] + L[i+Np, j+xx]*Fr[j+xx]
                vz += L[i+xx, j]*Fr[j] + L[i+xx, j+Np]*Fr[j+Np] + L[i+xx, j+xx]*Fr[j+xx]
            v[i  ]  += vx
            v[i+Np] += vy
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
            An array of size 3*Np,
        r: np.array
            An array of positions
            An array of size 3*Np,
        """

        cdef int i, j, Np=self.Np, xx=2*Np
        cdef double dx, dy, dz, idr, h2, hsq, idr2, idr3, idr4, idr5
        cdef double mur=1/(8*np.pi*self.eta), mu1=0.25*sqrt(2.0)*mur, mm=4/(self.a**3)
        cdef double ox, oy, oz

        cdef double [:, :] M = self.Mobility
        cdef double [:]   Tr = np.random.normal(size=3*Np)


        for i in prange(Np, nogil=True):
            for j in range(Np):
                dx = r[i]    - r[j]
                dy = r[i+Np] - r[j+Np]
                h2=2*r[j+xx]; hsq=r[j+xx]*r[j+xx]
                if i!=j:
                    dz = r[i+xx] - r[j+xx]
                    idr = 1.0/sqrt( dx*dx + dy*dy + dz*dz )
                    idr2=idr*idr;  idr3=idr*idr*idr
                    dx = dx*idr; dy=dy*idr; dz=dz*idr
                    #
                    M[i,    j   ] = (2 - 6*dx*dx)*idr3
                    M[i+Np, j+Np] = (2 - 6*dy*dy)*idr3
                    M[i+xx, j+xx] = (2 - 6*dz*dz)*idr3
                    M[i,    j+Np] = (  - 6*dx*dy)*idr3
                    M[i,    j+xx] = (  - 6*dx*dz)*idr3
                    M[i+Np, j+xx] = (  - 6*dy*dz)*idr3

                    ###contributions from the image
                    dz = r[i+xx] + r[j+xx]
                    idr = 1.0/sqrt( dx*dx + dy*dy + dz*dz )
                    dx = dx*idr; dy=dy*idr; dz=dz*idr
                    idr2=idr*idr;  idr3=idr2*idr;  idr4=idr3*idr

                    M[i,    j   ] += -(2 - 6*dx*dx)*idr3
                    M[i+Np, j+Np] += -(2 - 6*dy*dy)*idr3
                    M[i+xx, j+xx] += -(2 - 6*dz*dz)*idr3
                    M[i,    j+Np] += -(  - 6*dx*dy)*idr3
                    M[i,    j+xx] += -(  - 6*dx*dz)*idr3
                    M[i+Np, j+xx] += -(  - 6*dy*dz)*idr3

                    M[i,    j   ] += 12*dz*(dz - 5*dx*dx       )*idr3
                    M[i+Np, j+Np] += 12*dz*(dz - 5*dy*dy       )*idr3
                    M[i+xx, j+xx] +=-12*dz*(dz - 5*dz*dz + 2*dz)*idr3
                    M[i,    j+Np] += 12*dz*(   - 5*dx*dy       )*idr3
                    M[i,    j+xx] +=-12*dz*(   - 5*dx*dz + 2*dx)*idr3
                    M[i+Np, j+xx] +=-12*dz*(   - 5*dy*dz + 2*dy)*idr3

                    M[i,    j   ] += h2*6*(-dz   + 5*dx*dx*dz)*idr4
                    M[i+Np, j+Np] += h2*6*(-dz   + 5*dy*dy*dz)*idr4
                    M[i+xx, j+xx] +=-h2*6*(-3*dz + 5*dz*dz*dz)*idr4
                    M[i,    j+Np] += h2*6*(        5*dx*dy*dz)*idr4
                    M[i,    j+xx] +=-h2*6*(-dx   + 5*dx*dz*dz)*idr4
                    M[i+Np, j+xx] +=-h2*6*(-dy   + 5*dy*dz*dz)*idr4
                else:
                    # one-body mobility
                    M[i,    j   ] = mm
                    M[i+Np, j+Np] = mm
                    M[i+xx, j+xx] = mm
                    M[i,    j+Np] = 0
                    M[i,    j+xx] = 0
                    M[i+Np, j+xx] = 0

                    ##self contribtion from the image point
                    dz = r[i+xx] + r[j+xx]
                    idr = 1.0/sqrt( dx*dx + dy*dy + dz*dz )
                    dx = dx*idr; dy=dy*idr; dz=dz*idr
                    idr2=idr*idr;  idr3=idr2*idr;  idr4=idr3*idr

                    M[i,    j   ] += -(2 - 6*dx*dx)*idr3
                    M[i+Np, j+Np] += -(2 - 6*dy*dy)*idr3
                    M[i+xx, j+xx] += -(2 - 6*dz*dz)*idr3
                    M[i,    j+Np] += -(  - 6*dx*dy)*idr3
                    M[i,    j+xx] += -(  - 6*dx*dz)*idr3
                    M[i+Np, j+xx] += -(  - 6*dy*dz)*idr3

                    M[i,    j   ] += 12*dz*(dz - 5*dx*dx       )*idr3
                    M[i+Np, j+Np] += 12*dz*(dz - 5*dy*dy       )*idr3
                    M[i+xx, j+xx] +=-12*dz*(dz - 5*dz*dz + 2*dz)*idr3
                    M[i,    j+Np] += 12*dz*(   - 5*dx*dy       )*idr3
                    M[i,    j+xx] +=-12*dz*(   - 5*dx*dz + 2*dx)*idr3
                    M[i+Np, j+xx] +=-12*dz*(   - 5*dy*dz + 2*dy)*idr3

                    M[i,    j   ] += h2*6*(-dz   + 5*dx*dx*dz)*idr4
                    M[i+Np, j+Np] += h2*6*(-dz   + 5*dy*dy*dz)*idr4
                    M[i+xx, j+xx] +=-h2*6*(-3*dz + 5*dz*dz*dz)*idr4
                    M[i,    j+Np] += h2*6*(        5*dx*dy*dz)*idr4
                    M[i,    j+xx] +=-h2*6*(-dx   + 5*dx*dz*dz)*idr4
                    M[i+Np, j+xx] +=-h2*6*(-dy   + 5*dy*dz*dz)*idr4

        for i in prange(Np, nogil=True):
            for j in range(Np):
                M[i,    j   ] = mu1*M[i,    j   ]
                M[i+Np, j+Np] = mu1*M[i+Np, j+Np]
                M[i+xx, j+xx] = mu1*M[i+xx, j+xx]
                M[i,    j+Np] = mu1*M[i,    j+Np]
                M[i,    j+xx] = mu1*M[i,    j+xx]
                M[i+Np, j+xx] = mu1*M[i+Np, j+xx]

                M[i+Np, j   ] =     M[i,    j+Np]
                M[i+xx, j   ] =     M[i,    j+xx]
                M[i+xx, j+Np] =     M[i+Np, j+xx]

        cdef double [:, :] L = mu1*np.linalg.cholesky(self.Mobility)
        for i in prange(Np, nogil=True):
            ox=0; oy=0; oz=0;
            for j in range(Np):
                ox += L[i   , j]*Tr[j] + L[i   , j+Np]*Tr[j+Np] + L[i   , j+xx]*Tr[j+xx]
                oy += L[i+Np, j]*Tr[j] + L[i+Np, j+Np]*Tr[j+Np] + L[i+Np, j+xx]*Tr[j+xx]
                oz += L[i+xx, j]*Tr[j] + L[i+xx, j+Np]*Tr[j+Np] + L[i+xx, j+xx]*Tr[j+xx]
            o[i  ]  += ox
            o[i+Np] += oy
            o[i+xx] += oz
        return




@cython.boundscheck(False)
@cython.cdivision(True)
@cython.nonecheck(False)
@cython.wraparound(False)
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
        self.Np = particles
        self.Nt = gridpoints
        self.eta= viscosity



    cpdef flowField1s(self, double [:] vv, double [:] rt, double [:] r, double [:] F):
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
            An array of size 3*Np,
        F: np.array
            An array of body force
            An array of size 3*Np,
    
        Examples
        --------
        An example of the Flow field due to $1s$ mode of force per unit area

        >>> import pystokes, numpy as np, matplotlib.pyplot as plt
        >>> 
        >>> # particle radius, self-propulsion speed, number and fluid viscosity
        >>> b, eta, Np = 1.0, 1.0/6.0, 1
        >>> 
        >>> # initialize
        >>> r, p = np.array([0.0, 0.0, 3.4]), np.array([0.0, 1.0, 0])
        >>> F1s  = pystokes.utils.irreducibleTensors(1, p)
        >>> 
        >>> # space dimension , extent , discretization
        >>> dim, L, Ng = 3, 10, 64;
        >>> 
        >>> # instantiate the Flow class
        >>> flow = pystokes.wallBounded.Flow(radius=b, particles=Np, viscosity=eta, gridpoints=Ng*Ng)
        >>> 
        >>> # create grid, evaluate flow and plot
        >>> rr, vv = pystokes.utils.gridYZ(dim, L, Ng)
        >>> flow.flowField1s(vv, rr, r, F1s)
        >>> pystokes.utils.plotStreamlinesYZsurf(vv, rr, r, offset=6-1, density=1.4, title='1s')
        """

        cdef int i, j, Np=self.Np, Nt=self.Nt, xx=2*Np
        cdef double dx, dy, dz, idr, idr3, idr5, Fdotidr, tempF, hsq, h2
        cdef double vx, vy, vz, mu1=1.0/(8*PI*self.eta), a2=self.a*self.a/6.0

        for i in prange(Nt, nogil=True):
            vx=0; vy=0; vz=0;
            for j in range(Np):
                h2  =  2*r[j+xx]; hsq=r[j+xx]*r[j+xx]
                dx = rt[i]    - r[j]
                dy = rt[i+Nt] - r[j+Np]
                dz = rt[i+2*Nt]  - r[j+xx]
                idr = 1.0/sqrt( dx*dx + dy*dy + dz*dz )
                idr3=idr*idr*idr
                idr = 1.0/sqrt( dx*dx + dy*dy + dz*dz )
                idr3=idr*idr*idr
                Fdotidr = (F[j] * dx + F[j+Np] * dy + F[j+xx] * dz)*idr*idr
                #
                vx += (F[j]   +Fdotidr*dx)*idr + a2*(2*F[j]   -6*Fdotidr*dx)*idr3
                vy += (F[j+Np]+Fdotidr*dy)*idr + a2*(2*F[j+Np]-6*Fdotidr*dy)*idr3
                vz += (F[j+xx]+Fdotidr*dz)*idr + a2*(2*F[j+xx]-6*Fdotidr*dz)*idr3

                ##contributions from the image
                dz = rt[i+2*Nt] + r[j+xx]
                idr = 1.0/sqrt( dx*dx + dy*dy + dz*dz )
                idr3 = idr*idr*idr
                idr5 = idr3*idr*idr
                Fdotidr = ( F[j]*dx + F[j+Np]*dy + F[j+xx]*dz )*idr*idr

                vx += -(F[j]   +Fdotidr*dx)*idr - a2*(2*F[j]   -6*Fdotidr*dx)*idr3
                vy += -(F[j+Np]+Fdotidr*dy)*idr - a2*(2*F[j+Np]-6*Fdotidr*dy)*idr3
                vz += -(F[j+xx]+Fdotidr*dz)*idr - a2*(2*F[j+xx]-6*Fdotidr*dz)*idr3

                tempF  = -F[j+xx]     # F_i = M_ij F_j, reflection of the strength
                Fdotidr = ( F[j]*dx + F[j+Np]*dy + tempF*dz )*idr*idr

                vx += -h2*(dz*(F[j]   - 3*Fdotidr*dx) + tempF*dx)*idr3
                vy += -h2*(dz*(F[j+Np]- 3*Fdotidr*dy) + tempF*dy)*idr3
                vz += -h2*(dz*(tempF  - 3*Fdotidr*dz) + tempF*dz)*idr3 + h2*Fdotidr*idr

                vx += hsq*( 2*F[j]   - 6*Fdotidr*dx )*idr3
                vy += hsq*( 2*F[j+Np]- 6*Fdotidr*dy )*idr3
                vz += hsq*( 2*tempF  - 6*Fdotidr*dz )*idr3

                vx += 12*a2*dz*( dz*F[j]   - 5*dz*Fdotidr*dx + 2*tempF*dx )*idr5
                vy += 12*a2*dz*( dz*F[j+Np]- 5*dz*Fdotidr*dy + 2*tempF*dy )*idr5
                vz += 12*a2*dz*( dz*tempF  - 5*dz*Fdotidr*dz + 2*tempF*dz )*idr5

                vx += -h2*6*a2*(dz*F[j]   -5*Fdotidr*dx*dz + tempF*dx)*idr5
                vy += -h2*6*a2*(dz*F[j+Np]-5*Fdotidr*dy*dz+ tempF*dy)*idr5
                vz += -h2*6*a2*(dz*tempF  -5*Fdotidr*dz*dz + tempF*dz)*idr5 -6*a2*h2*Fdotidr*idr3

            vv[i  ]    += mu1*vx
            vv[i+Nt]   += mu1*vy
            vv[i+2*Nt] += mu1*vz
        return 
    
    
    cpdef flowField2a(  self, double [:] vv, double [:] rt, double [:] r, double [:] T):
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
            An array of size 3*Np,
        T: np.array
            An array of body torque
            An array of size 3*Np,
    
        Examples
        --------
        An example of the RBM 

        # Example 1: Flow field due to $2a$ mode of force per unit area
        >>> import pystokes, numpy as np, matplotlib.pyplot as plt
        >>> 
        >>> # particle radius, self-propulsion speed, number and fluid viscosity
        >>> b, eta, Np = 1.0, 1.0/6.0, 1
        >>> 
        >>> # initialize
        >>> r, p = np.array([0.0, 0.0, 3.4]), np.array([0.0, 1.0, 0])
        >>> V2a  = pystokes.utils.irreducibleTensors(1, p)
        >>> 
        >>> # space dimension , extent , discretization
        >>> dim, L, Ng = 3, 10, 64;
        >>> 
        >>> # instantiate the Flow class
        >>> flow = pystokes.wallBounded.Flow(radius=b, particles=Np, viscosity=eta, gridpoints=Ng*Ng)
        >>> 
        >>> # create grid, evaluate flow and plot
        >>> rr, vv = pystokes.utils.gridYZ(dim, L, Ng)
        >>> flow.flowField2a(vv, rr, r, V2a)
        >>> pystokes.utils.plotStreamlinesYZsurf(vv, rr, r, offset=6-1, density=1.4, title='2s')
        """

        cdef int Np = self.Np, i, j, xx=2*Np, Nt=self.Nt
        cdef double dx, dy, dz, idr, idr3, rlz, Tdotidr, h2, 
        cdef double vx, vy, vz, mu1 = 1.0/(8*PI*self.eta)
#        cdef int i, j, Np=self.Np, Nt=self.Nt, xx=2*Np
#        cdef double dx, dy, dz, idr, idr3, idr5, Fdotidr, tempF, hsq, h2
#        cdef double vx, vy, vz, mu1=1.0/(8*PI*self.eta), a2=self.a*self.a/6.0
 
        for i in prange(Nt, nogil=True):
            vx=0; vy=0; vz=0;
            for j in range(Np):
                dx = rt[i]   - r[j]   
                dy = rt[i+Nt] - r[j+Np]   
                h2 = 2*rt[i+Nt*2]
                #contributions from the source 
                dz = rt[i+2*Nt] - r[j+xx] 
                idr = 1.0/sqrt( dx*dx + dy*dy + dz*dz )
                idr3 = idr*idr*idr
                 
                vx += (T[j+Np]*dz - T[j+xx]*dy )*idr3
                vy += (T[j+xx]*dx - T[j]   *dz )*idr3
                vz += (T[j]   *dy - T[j+Np]*dx )*idr3
                    
                #contributions from the image 
                dz = rt[i+2*Nt] + r[j+xx]            
                idr = 1.0/sqrt( dx*dx + dy*dy + dz*dz )
                idr3 = idr*idr*idr
                
                vx += -(T[j+Np]*dz - T[j+xx]*dy )*idr3
                vy += -(T[j+xx]*dx - T[j]   *dz )*idr3
                vz += -(T[j]   *dy - T[j+Np]*dx )*idr3
                
                rlz = (dx*T[j+Np] - dy*T[j])*idr*idr
                vx += (h2*(T[j+Np]-3*rlz*dx) + 6*dz*dx*rlz)*idr3
                vy += (h2*(-T[j]  -3*rlz*dy) + 6*dz*dy*rlz)*idr3
                vz += (h2*(       -3*rlz*dz) + 6*dz*dz*rlz)*idr3

                dz = rt[i+2*Nt] + r[j+xx]            
                idr = 1.0/sqrt( dx*dx + dy*dy + dz*dz )
                idr3 = idr*idr*idr
                
                vx += -(T[j+Np]*dz )*idr3
                vy += -(- T[j] *dz )*idr3
                
                vx += h2*T[j+Np]*idr3
                vy += -h2*T[j]*idr3

            vv[i  ]  += mu1*vx 
            vv[i+Nt] += mu1*vy
            vv[i+2*Nt] += mu1*vz
        return 
    
   
    cpdef flowField2s(self, double [:] vv, double [:] rt, double [:] r, double [:] S):
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
            An array of size 3*Np,
        S: np.array
            An array of 2s mode of the slip
            An array of size 5*Np,
        
        Examples
        --------
        An example of the Flow field due to $3t$ mode of active slip

        >>> import pystokes, numpy as np, matplotlib.pyplot as plt
        >>> 
        >>> # particle radius, self-propulsion speed, number and fluid viscosity
        >>> b, eta, Np = 1.0, 1.0/6.0, 1
        >>> 
        >>> # initialize
        >>> r, p = np.array([0.0, 0.0, 3.4]), np.array([0.0, 1.0, 0])
        >>> V3t  = pystokes.utils.irreducibleTensors(1, p)
        >>> 
        >>> # space dimension , extent , discretization
        >>> dim, L, Ng = 3, 10, 64;
        >>> 
        >>> # instantiate the Flow class
        >>> flow = pystokes.wallBounded.Flow(radius=b, particles=Np, viscosity=eta, gridpoints=Ng*Ng)
        >>> 
        >>> # create grid, evaluate flow and plot
        >>> rr, vv = pystokes.utils.gridYZ(dim, L, Ng)
        >>> flow.flowField3t(vv, rr, r, V3t)
        >>> pystokes.utils.plotStreamlinesYZsurf(vv, rr, r, offset=6-1, density=1.4, title='1s')
        """
        cdef int Np=self.Np,  Nt=self.Nt, xx=2*Np, xx1=3*Np, xx2=4*Np
        cdef int i, j  
        cdef double dx, dy, dz, idr, idr2, idr3, idr5, idr7, aidr2, trS, h2, hsq
        cdef double sxx, syy, szz, sxy, syx, syz, szy, sxz, szx, srr, srx, sry, srz
        cdef double Sljrlx, Sljrly, Sljrlz, Sljrjx, Sljrjy, Sljrjz 
        cdef double vx, vy, vz, mus = (28.0*self.a**3)/24 

        for i in prange(Nt, nogil=True):
            vx=0; vy=0; vz=0;
            for j in  range(Np):
                sxx = S[j]  ; syy = S[j+Np]; szz = -sxx-syy;
                sxy = S[j+xx]; syx = sxy;
                sxz = S[j+xx1]; szx = sxz;
                syz = S[j+xx2]; szy = syz;
                
                dx = rt[i]   - r[j]
                dy = rt[i+Nt] - r[j+Np]
                dz = rt[i+2*Nt] - r[j+xx] 
                h2 = 2*r[j+xx]; hsq = r[j+xx]*r[j+xx];
                idr  = 1.0/sqrt( dx*dx + dy*dy + dz*dz );
                idr2 = idr*idr; idr3 = idr2*idr; idr5 = idr3*idr2; idr7 = idr5*idr2;
                srx = (sxx*dx +  sxy*dy + sxz*dz ); 
                sry = (sxy*dx +  syy*dy + syz*dz );
                srz = (sxz*dx +  syz*dy + szz*dz );
                srr = sxx*dx*dx + syy*dy*dy + szz*dz*dz + 2*sxy*dx*dy + 2*sxz*dx*dz + 2*syz*dy*dz;
                
                ## contributions from the source 
                vx += 3*srr*dx*idr5;
                vy += 3*srr*dy*idr5;
                vz += 3*srr*dz*idr5;
                 
                ## contributions from the image 
                dz = rt[i+2*Nt]+r[j+xx]
                idr  = 1.0/sqrt( dx*dx + dy*dy + dz*dz )
                idr2 = idr*idr; idr3 = idr2*idr; idr5 = idr3*idr2; idr7 = idr5*idr2;
                
                #reflecting the first index of stresslet, S_jl M_lm
                sxz=-sxz; syz=-syz; szz=-szz;     trS=sxx+syy+szz; 
                Sljrlx = sxx*dx +  sxy*dx + sxz*dx ; 
                Sljrly = syx*dy +  syy*dy + syz*dy ;
                Sljrlz = szx*dz +  szy*dz + szz*dz ;
                Sljrjx = sxx*dx +  sxy*dy + sxz*dz ; 
                Sljrjy = syx*dx +  syy*dy + syz*dz ;
                Sljrjz = szx*dx +  szy*dy + szz*dz ;
                srr = (sxx*dx*dx + syy*dy*dy + szz*dz*dz +  2*sxy*dx*dy)*idr2 ;
                srx = sxx*dx + syx*dy+szx*dz;
                sry = sxy*dx + syy*dy+szy*dz;
                srz = sxz*dx + syz*dy+szz*dz;
                
                vx += (-Sljrlx + Sljrjx + trS*dx -3*srr*dx)*idr3 ;
                vy += (-Sljrly + Sljrjy + trS*dy -3*srr*dy)*idr3 ;
                vz += (-Sljrlz + Sljrjz + trS*dz -3*srr*dz)*idr3 ;
                
                vx += -2*(dz*(sxz-3*srz*dx*idr2)+ szz*dx)*idr3;
                vy += -2*(dz*(syz-3*srz*dy*idr2)+ szz*dy)*idr3;
                vz += -2*(dz*(szz-3*srz*dz*idr2)+ szz*dz - srz)*idr3;
                
                vx += h2*( sxz-3*srz*dx*idr2)*idr3;
                vy += h2*( syz-3*srz*dy*idr2)*idr3;
                vz += h2*( szz-3*srz*dz*idr2)*idr3;
                
                #reflecting both the indices of stresslet, S_jl M_lm M_jk
                szx = -szx ; syz = -syz; szz = -szz;
                srx = sxx*dx + syx*dy+szx*dz;
                sry = sxy*dx + syy*dy+szy*dz;
                srz = sxz*dx + syz*dy+szz*dz;
                srr = (sxx*dx*dx + syy*dy*dy + szz*dz*dz + 2*sxy*dx*dy + 2*sxz*dx*dz + 2*syz*dy*dz)*idr2;
                
                vx += h2*(dz*(-6*srx + 15*srr*dx)-3*srz*dx)*idr5 + h2*(sxz)*idr3 ;
                vy += h2*(dz*(-6*sry + 15*srr*dy)-3*srz*dy)*idr5 + h2*(syz)*idr3 ;
                vz += h2*(dz*(-6*srz + 15*srr*dz)-3*srz*dz)*idr5 + h2*(szz + 3*srr)*idr3;

                vx += hsq*(12*srx - 30*srr*dx)*idr5 ;
                vy += hsq*(12*sry - 30*srr*dy)*idr5 ;
                vz += hsq*(12*srz - 30*srr*dz)*idr5 ;
     
            vv[i  ]    += mus*vx
            vv[i+Nt]   += mus*vy
            vv[i+2*Nt] += mus*vz
        return

   
    cpdef flowField3t(self, double [:] vv, double [:] rt, double [:] r, double [:] D):
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
            An array of size 3*Np,
        D: np.array
            An array of 3t mode of the slip
            An array of size 3*Np,
 
        Examples
        --------
        An example of the Flow field due to $3t$ mode of active slip

        >>> import pystokes, numpy as np, matplotlib.pyplot as plt
        >>> 
        >>> # particle radius, self-propulsion speed, number and fluid viscosity
        >>> b, eta, Np = 1.0, 1.0/6.0, 1
        >>> 
        >>> # initialize
        >>> r, p = np.array([0.0, 0.0, 3.4]), np.array([0.0, 1.0, 0])
        >>> V3t  = pystokes.utils.irreducibleTensors(1, p)
        >>> 
        >>> # space dimension , extent , discretization
        >>> dim, L, Ng = 3, 10, 64;
        >>> 
        >>> # instantiate the Flow class
        >>> flow = pystokes.wallBounded.Flow(radius=b, particles=Np, viscosity=eta, gridpoints=Ng*Ng)
        >>> 
        >>> # create grid, evaluate flow and plot
        >>> rr, vv = pystokes.utils.gridYZ(dim, L, Ng)
        >>> flow.flowField3t(vv, rr, r, V3t)
        >>> pystokes.utils.plotStreamlinesYZsurf(vv, rr, r, offset=6-1, density=1.4, title='1s')
        """

        cdef int i, j, Np=self.Np, Nt=self.Nt, xx=2*Np
        cdef double dx, dy, dz, idr, idr3, idr5, Ddotidr, tempD, hsq, h2
        cdef double vx, vy, vz, mud = 3.0*self.a*self.a*self.a/5, mu1 = -1.0*(self.a**5)/10

        for i in prange(Nt, nogil=True):
            vx=0; vy=0; vz=0;
            for j in range(Np):
                h2 = 2*r[j+xx]; hsq = r[j+xx]*r[j+xx];
                dx = rt[i]      - r[j]
                dy = rt[i+Nt]   - r[j+Np]
                dz = rt[i+2*Nt] - r[j+xx] 
                idr = 1.0/sqrt( dx*dx + dy*dy + dz*dz )
                idr3=idr*idr*idr
                Ddotidr = (D[j]*dx + D[j+Np]*dy + D[j+xx]*dz)*idr*idr
                #
                vx += (2*D[j]    - 6*Ddotidr*dx)*idr3
                vy += (2*D[j+Np] - 6*Ddotidr*dy)*idr3
                vz += (2*D[j+xx] - 6*Ddotidr*dz)*idr3
                
                ##contributions from the image 
                dz = rt[i+2*Nt] + r[j+xx]        
                idr = 1.0/sqrt( dx*dx + dy*dy + dz*dz )
                idr3 = idr*idr*idr
                idr5 = idr3*idr*idr
                Ddotidr = (D[j]*dx + D[j+Np]*dy + D[j+xx]*dz)*idr*idr
                
                vx += -(2*D[j]    - 6*Ddotidr*dx )*idr3
                vy += -(2*D[j+Np] - 6*Ddotidr*dy )*idr3
                vz += -(2*D[j+xx] - 6*Ddotidr*dz )*idr3
                
                tempD = -D[j+xx]     # D_i = M_ij D_j, reflection of the strength
                Ddotidr = ( D[j]*dx + D[j+Np]*dy + tempD*dz )*idr*idr
                
                vx += 12*dz*( dz*D[j]   - 5*dz*Ddotidr*dx + 2*tempD*dx )*idr5
                vy += 12*dz*( dz*D[j+Np]- 5*dz*Ddotidr*dy + 2*tempD*dy )*idr5
                vz += 12*dz*( dz*tempD  - 5*dz*Ddotidr*dz + 2*tempD*dz )*idr5

                vx += -6*h2*(dz*D[j]   -5*Ddotidr*dx*dz + tempD*dx)*idr5
                vy += -6*h2*(dz*D[j+Np]-5*Ddotidr*dy*dz + tempD*dy)*idr5
                vz += -6*h2*(dz*tempD  -5*Ddotidr*dz*dz + tempD*dz)*idr5 -6*h2*Ddotidr*idr3
                
            vv[i  ]    += mu1*vx
            vv[i+Nt]   += mu1*vy
            vv[i+2*Nt] += mu1*vz
        return 
    
