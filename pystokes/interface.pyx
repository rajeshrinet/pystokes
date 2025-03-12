cimport cython
from libc.math cimport sqrt
from cython.parallel import prange
cdef double PI = 3.14159265359
cdef double sqrt8 = 2.82842712475
cdef double sqrt2 = 1.41421356237
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

        
    cpdef mobilityTT(self, double [:] v, double [:] r, double [:] F, double ll=0):
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
        ll: float 
            viscosity ratio of the two fluids 
            Default is zero 
        
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
        >>> rbm = pystokes.interface.Rbm(radius=b, particles=N, viscosity=eta)
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
        >>>    Tf,Nts,rhs,integrator='odeint', filename='crystallization')
        """

        cdef int i, j, N=self.N, xx=2*N
        cdef double dx, dy, dz, idr, idr3, idr5, Fdotidr, h2, hsq, tempF
        cdef double vx, vy, vz, F1, F2, F3
        cdef double mu=self.mu, muv=self.muv, a2=self.b*self.b/3.0
        cdef double ll1 = (1-ll)/(1+ll), ll2 = ll/(1+ll);
        
        cdef double llp = 1.0/(1+ll);
        cdef double a = self.b
        cdef double h, hbar_inv, hbar_inv3, hbar_inv5
        cdef double muTTpara1 = 3*(2-3*ll)*llp/16.0, muTTpara2 = (1+2*ll)*llp/16.0
        cdef double muTTpara3 = - ll*llp/16.0
        cdef double muTTperp1 = - 3*(2+3*ll)*llp/8.0, muTTperp2 = (1+4*ll)*llp/8.0
        cdef double muTTperp3 = - ll*llp/8.0
        cdef double mux, muy, muz

        for i in prange(N, nogil=True):
            vx=0; vy=0; vz=0;
            for j in range(N):
                dx = r[i]    - r[j]
                dy = r[i+N]  - r[j+N]
                h2  =  2*r[j+xx]; hsq=r[j+xx]*r[j+xx]
                if i!=j:
                    dz = r[i+xx]  - r[j+xx]
                    idr = 1.0/sqrt( dx*dx + dy*dy + dz*dz )
                    idr3=idr*idr*idr
                    Fdotidr = (F[j] * dx + F[j+N] * dy + F[j+xx] * dz)*idr*idr
                    #
                    vx += (F[j]   +Fdotidr*dx)*idr + a2*(2*F[j]   -6*Fdotidr*dx)*idr3
                    vy += (F[j+N]+Fdotidr*dy)*idr  + a2*(2*F[j+N]-6*Fdotidr*dy)*idr3
                    vz += (F[j+xx]+Fdotidr*dz)*idr + a2*(2*F[j+xx]-6*Fdotidr*dz)*idr3

                    ##contributions from the image
                    F1 = ll1*F[j];   F2 = ll1*F[j+N];   F3 = -F[j+xx];
                    dz = r[i+xx] + r[j+xx]
                    idr = 1.0/sqrt( dx*dx + dy*dy + dz*dz )
                    idr3 = idr*idr*idr
                    idr5 = idr3*idr*idr
                    Fdotidr = ( F1*dx + F2*dy + F3*dz )*idr*idr
                    vx += (F1+Fdotidr*dx)*idr + a2*(2*F1-6*Fdotidr*dx)*idr3
                    vy += (F2+Fdotidr*dy)*idr + a2*(2*F2-6*Fdotidr*dy)*idr3
                    vz += (F3+Fdotidr*dz)*idr + a2*(2*F3-6*Fdotidr*dz)*idr3

                    tempF  = -F[j+xx]     # F_i = M_ij F_j, reflection of the strength
                    Fdotidr = ( F[j]*dx + F[j+N]*dy + tempF*dz )*idr*idr

                    vx += ll2*(-h2*(dz*(F[j]   - 3*Fdotidr*dx) + tempF*dx)*idr3)
                    vy += ll2*(-h2*(dz*(F[j+N]- 3*Fdotidr*dy) + tempF*dy)*idr3)
                    vz += ll2*(-h2*(dz*(tempF  - 3*Fdotidr*dz) + tempF*dz)*idr3 + h2*Fdotidr*idr)

                    vx += ll2*(hsq*( 2*F[j]   - 6*Fdotidr*dx )*idr3)
                    vy += ll2*(hsq*( 2*F[j+N]- 6*Fdotidr*dy )*idr3)
                    vz += ll2*(hsq*( 2*tempF  - 6*Fdotidr*dz )*idr3)

                    vx += ll2*(12*a2*dz*( dz*F[j]   - 5*dz*Fdotidr*dx + 2*tempF*dx )*idr5)
                    vy += ll2*(12*a2*dz*( dz*F[j+N]- 5*dz*Fdotidr*dy + 2*tempF*dy )*idr5)
                    vz += ll2*(12*a2*dz*( dz*tempF  - 5*dz*Fdotidr*dz + 2*tempF*dz )*idr5)

                    vx += ll2*(-h2*6*a2*(dz*F[j]   -5*Fdotidr*dx*dz + tempF*dx)*idr5)
                    vy += ll2*(-h2*6*a2*(dz*F[j+N]-5*Fdotidr*dy*dz+ tempF*dy)*idr5)
                    vz += ll2*(-h2*6*a2*(dz*tempF  -5*Fdotidr*dz*dz + tempF*dz)*idr5 -6*a2*h2*Fdotidr*idr3)
                else:
                    ''' self contribution from the image point'''
                    h = r[j+xx]
                    hbar_inv = a/h; hbar_inv3 = hbar_inv*hbar_inv*hbar_inv
                    hbar_inv5 = hbar_inv3*hbar_inv*hbar_inv
                    
                    mux = mu*(1 + muTTpara1*hbar_inv + muTTpara2*hbar_inv3 
                              + muTTpara3*hbar_inv5)
                    muy = mux
                    muz = mu*(1 + muTTperp1*hbar_inv + muTTperp2*hbar_inv3 
                              + muTTperp3*hbar_inv5)

            v[i  ]  += mux*F[i]    + muv*vx
            v[i+N] += muy*F[i+N] + muv*vy
            v[i+xx] += muz*F[i+xx] + muv*vz
        return


    cpdef mobilityTR(self, double [:] v, double [:] r, double [:] T, double ll=0):
        """
        Compute velocity due to body forces using :math:`v=\mu^{TR}\cdot T` 
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
            An array of forces
            An array of size 3*N,
        ll: float 
            viscosity ratio of the two fluids 
            Default is zero 
        """

        cdef int N = self.N, i, j, xx=2*N
        cdef double dx, dy, dz, idr, idr3, rlz, Tdotidr, h2,
        cdef double vx, vy, vz, T1, T2, T3
        cdef double muv=self.muv 
        cdef double ll1 = (1-ll)/(1+ll), ll2 = ll/(1+ll);
        
        cdef double llp = 1.0/(1+ll);
        cdef double a = self.b
        cdef double h, hbar_inv, hbar_inv2, hbar_inv4
        cdef double muTR0 = 4.0/(3*self.b*self.b)
        cdef double muTR1 = -3*llp/16.0, muTR2 = 3*ll2/32.0
        cdef double muTR

        for i in prange(N, nogil=True):
            vx=0; vy=0; vz=0;
            for j in range(N):
                dx = r[i]   - r[j]
                dy = r[i+N] - r[j+N]
                h2 = 2*r[i+xx]
                if i != j:
                    #contributions from the source
                    dz = r[i+xx] - r[j+xx]
                    idr = 1.0/sqrt( dx*dx + dy*dy + dz*dz )
                    idr3 = idr*idr*idr

                    vx += (T[j+N]*dz - T[j+xx]*dy )*idr3
                    vy += (T[j+xx]*dx - T[j]   *dz )*idr3
                    vz += (T[j]   *dy - T[j+N]*dx )*idr3

                    #contributions from the image
                    T1 = T[j];
                    T2 = T[j+N]
                    T3 = -T[j+2*N];
                    dz = r[i+xx] + r[j+xx]
                    idr = 1.0/sqrt( dx*dx + dy*dy + dz*dz )
                    idr3 = idr*idr*idr

                    vx += (T2*dz - T3*dy )*idr3
                    vy += (T3*dx - T1*dz )*idr3
                    vz += (T1*dy - T2*dx )*idr3

                    rlz = (dx*T[j+N] - dy*T[j])*idr*idr
                    vx += ll2*(h2*(T[j+N]-3*rlz*dx) + 6*dz*dx*rlz)*idr3
                    vy += ll2*(h2*(-T[j]  -3*rlz*dy) + 6*dz*dy*rlz)*idr3
                    vz += ll2*(h2*(       -3*rlz*dz) + 6*dz*dz*rlz)*idr3
                else:
                    ''' the self contribution from the image point'''
                    h = r[j+xx]
                    hbar_inv = a/h; hbar_inv2 = hbar_inv*hbar_inv
                    hbar_inv4 = hbar_inv2*hbar_inv*hbar_inv
                    
                    muTR = muTR0*(muTR1*hbar_inv2 + muTR2*hbar_inv4)
                    
                    T1 = T[j];
                    T2 = T[j+N]
                    
                    vx += -muTR*T2   #change sign here to make up for '-=' below...
                    vy += muTR*T1  #same here

            v[i]    += muv*vx   
            v[i+N]  += muv*vy
            v[i+xx] += muv*vz
        return


    cpdef propulsionT2s(self, double [:] v, double [:] r, double [:] V2s, double ll=0):
        """
        Compute velocity due to 2s mode of the slip :math:`v=\pi^{T,2s}\cdot V^{2s}` 
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

        cdef int N=self.N, i, j, xx=2*N, xx1=3*N , xx2=4*N
        cdef double dx, dy, dz, idr, idr2, idr3, idr5, idr7, aidr2, trS, h2, hsq
        cdef double sxx, syy, szz, sxy, syx, syz, szy, sxz, szx, srr, srx, sry, srz
        cdef double Sljrlx, Sljrly, Sljrlz, Sljrjx, Sljrjy, Sljrjz
        cdef double vx, vy, vz, mus =(28.0*self.b**3)/24
        cdef double ll2 = ll/(1+ll);
        
        cdef double llp = 1.0/(1+ll);
        cdef double a = self.b
        cdef double h, hbar_inv, hbar_inv2, hbar_inv4, hbar_inv6
        cdef double piT2s11 = 5*ll2/16.0, piT2s12 = -(1+3*ll)*llp/12.0
        cdef double piT2s13 = 5*ll2/48.0
        cdef double piT2s21 = -5*(2+3*ll)*llp/48.0, piT2s22 = (4+15*ll)*llp/48.0
        cdef double piT2s23 = -5*ll2/48.0
        cdef double piT2s1, piT2s2
        cdef double mus_inv = 1.0/mus

        for i in prange(N, nogil=True):
            vx=0; vy=0;   vz=0;
            for j in  range(N):
                h2 = 2*r[j+xx]; hsq = r[j+xx]*r[j+xx];
                sxx = V2s[j]  ; syy = V2s[j+N]; szz = -sxx-syy;
                sxy = V2s[j+xx]; syx = sxy;
                sxz = V2s[j+xx1]; szx = sxz;
                syz = V2s[j+xx2]; szy = syz;
                dx = r[i]   - r[j]
                dy = r[i+N] - r[j+N]
                if i!=j:
                    dz = r[i+xx] - r[j+xx]
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
                    dz = r[i+xx]+r[j+xx]
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
                    srx = sxx*dx + sxy*dy + sxz*dz
                    sry = syx*dx + syy*dy + syz*dz
                    srz = sxz*dx + syz*dy + szz*dz

                    vx += (Sljrlx - Sljrjx - trS*dx + 3*srr*dx)*idr3
                    vy += (Sljrly - Sljrjy - trS*dy + 3*srr*dy)*idr3
                    vz += (Sljrlz - Sljrjz - trS*dz + 3*srr*dz)*idr3

                    vx += -ll*2*(dz*(sxz-3*srz*dx*idr2)+ szz*dx)*idr3;
                    vy += -ll*2*(dz*(syz-3*srz*dy*idr2)+ szz*dy)*idr3;
                    vz += -ll*2*(dz*(szz-3*srz*dz*idr2)+ szz*dz - srz)*idr3;
                    
                    vx += ll*h2*( sxz-3*srz*dx*idr2)*idr3;
                    vy += ll*h2*( syz-3*srz*dy*idr2)*idr3;
                    vz += ll*h2*( szz-3*srz*dz*idr2)*idr3;
                    
                    #reflecting both the indices of stresslet, S_jl M_lm M_jk
                    szx = -szx ; syz = -syz; szz = -szz;
                    srx = ll*(sxx*dx +  sxy*dy + sxz*dz )
                    sry = ll*(sxy*dx +  syy*dy + syz*dz )
                    srz = ll*(sxz*dx +  syz*dy + szz*dz )
                    srr = ll*(sxx*dx*dx + syy*dy*dy + szz*dz*dz 
                            + 2*sxy*dx*dy + 2*sxz*dx*dz + 2*syz*dy*dz)*idr2;
                    
                    vx += ll*h2*( (dz*(-6*srx + 15*srr*dx)-3*srz*dx)*idr5 + (sxz)*idr3) ;
                    vy += ll*h2*( (dz*(-6*sry + 15*srr*dy)-3*srz*dy)*idr5 + (syz)*idr3) ;
                    vz += ll*h2*( (dz*(-6*srz + 15*srr*dz)-3*srz*dz)*idr5 + (szz + 3*srr)*idr3);

                    vx += ll*hsq*(12*srx - 30*srr*dx)*idr5
                    vy += ll*hsq*(12*sry - 30*srr*dy)*idr5
                    vz += ll*hsq*(12*srz - 30*srr*dz)*idr5

                else:
                    ''' the self contribution from the image point'''
                    h = r[j+xx]
                    hbar_inv = a/h; hbar_inv2 = hbar_inv*hbar_inv
                    hbar_inv4 = hbar_inv2*hbar_inv*hbar_inv
                    hbar_inv6 = hbar_inv4*hbar_inv2
                    
                    #implement superposition approximation only
                    piT2s1 = piT2s11*hbar_inv2 + piT2s12*hbar_inv4 + piT2s13*hbar_inv6
                    piT2s2 = piT2s21*hbar_inv2 + piT2s22*hbar_inv4 + piT2s23*hbar_inv6
                    
                    vx += mus_inv * 2*piT2s1*sxz
                    vy += mus_inv * 2*piT2s1*syz
                    vz += mus_inv * 3*(piT2s2*sxx + piT2s2*syy)

            v[i]    += vx*mus
            v[i+N] += vy*mus
            v[i+xx] += vz*mus
        return


    cpdef propulsionT3t(self, double [:] v, double [:] r, double [:] V3t, double ll=0):
        """
        Compute velocity due to 3t mode of the slip :math:`v=\pi^{T,3t}\cdot V^{3t}` 
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

        cdef int N=self.N, i, j, xx=2*N
        cdef double dx, dy, dz, idr, idr3, idr5, Ddotidr, tempD, hsq, h2, D1, D2, D3
        cdef double vx, vy, vz, mud = 3.0*self.b*self.b*self.b/5, muv = -1.0*(self.b**5)/10
        cdef double ll1 = (1-ll)/(1+ll), ll2 = ll/(1+ll);
        
        cdef double llp = 1.0/(1+ll);
        cdef double a = self.b
        cdef double h, hbar_inv, hbar_inv3, hbar_inv5
        cdef double piT3tpara1 = -(1+2*ll)*llp/80.0
        cdef double piT3tpara2 = ll2/40.0
        cdef double piT3tperp1 = -(1+4*ll)*llp/40.0
        cdef double piT3tperp2 = ll2/20.0
        cdef double pix, piy, piz

        for i in prange(N, nogil=True):
            vx=0; vy=0; vz=0;
            for j in range(N):
                dx = r[i]    - r[j]
                dy = r[i+N]  - r[j+N]
                h2  =  2*r[j+xx]
                if i!=j:
                    dz = r[i+xx] - r[j+xx]
                    idr = 1.0/sqrt( dx*dx + dy*dy + dz*dz )
                    idr3=idr*idr*idr
                    Ddotidr = (V3t[j]*dx + V3t[j+N]*dy + V3t[j+xx]*dz)*idr*idr
                    #
                    vx += (2*V3t[j]    - 6*Ddotidr*dx)*idr3
                    vy += (2*V3t[j+N] - 6*Ddotidr*dy)*idr3
                    vz += (2*V3t[j+xx] - 6*Ddotidr*dz)*idr3

                    ##contributions from the image
                    dz = r[i+xx] + r[j+xx]
                    idr = 1.0/sqrt( dx*dx + dy*dy + dz*dz )
                    idr3 = idr*idr*idr
                    idr5 = idr3*idr*idr
                    D1 = V3t[j];
                    D2 = V3t[j+N]
                    D3 = -V3t[j+2*N];
                    Ddotidr = (D1*dx + D2*dy + D3*dz)*idr*idr

                    vx += (2*D1 - 6*Ddotidr*dx )*idr3
                    vy += (2*D2 - 6*Ddotidr*dy )*idr3
                    vz += (2*D3 - 6*Ddotidr*dz )*idr3

                    tempD = -V3t[j+xx]     # D_i = M_ij D_j, reflection of the strength
                    Ddotidr = ( V3t[j]*dx + V3t[j+N]*dy + tempD*dz )*idr*idr
                    
                    vx += ll*12*dz*( dz*V3t[j]   - 5*dz*Ddotidr*dx + 2*tempD*dx )*idr5
                    vy += ll*12*dz*( dz*V3t[j+N]- 5*dz*Ddotidr*dy + 2*tempD*dy )*idr5
                    vz += ll*12*dz*( dz*tempD  - 5*dz*Ddotidr*dz + 2*tempD*dz )*idr5

                    vx += -ll*6*h2*(dz*V3t[j]   -5*Ddotidr*dx*dz + tempD*dx)*idr5
                    vy += -ll*6*h2*(dz*V3t[j+N]-5*Ddotidr*dy*dz + tempD*dy)*idr5
                    vz += -ll*6*h2*(dz*tempD  -5*Ddotidr*dz*dz + tempD*dz)*idr5 -6*h2*Ddotidr*idr3

                else:
                    ''' self contribution from the image point'''
                    h = r[j+xx]
                    hbar_inv = a/h; hbar_inv3 = hbar_inv*hbar_inv*hbar_inv
                    hbar_inv5 = hbar_inv3*hbar_inv*hbar_inv
                    
                    pix = piT3tpara1*hbar_inv3 + piT3tpara2*hbar_inv5
                    piy = pix
                    piz = piT3tperp1*hbar_inv3 + piT3tperp2*hbar_inv5

            v[i  ]  += pix*V3t[j]    + muv*vx
            v[i+N]  += piy*V3t[j+N] + muv*vy
            v[i+xx] += piz*V3t[j+xx] + muv*vz
        return
    
    

    
    ## Angular Velocities
    cpdef mobilityRT(self, double [:] o, double [:] r, double [:] F, double ll=0):
        cdef int N = self.N, i, j, xx=2*N
        cdef double dx, dy, dz, idr, idr3, rlz, Fdotidr, h2
        cdef double ox, oy, oz, muv=1.0/(8*PI*self.eta)
        cdef double ll1 = (1-ll)/(1+ll), ll2 = ll/(1+ll)
        
        cdef double llp = 1.0/(1+ll);
        cdef double a = self.b
        cdef double h, hbar_inv, hbar_inv2, hbar_inv4
        cdef double muRT0 = 4.0/(3*self.b*self.b)
        cdef double muRT1 = 3*llp/16.0, muRT2 = -3*ll2/32.0
        cdef double muRT, F1, F2

        for i in prange(N, nogil=True):
            ox=0; oy=0; oz=0;
            for j in range(N):
                dx = r[i]   - r[j]
                dy = r[i+N] - r[j+N]
                h2 = 2*r[i+xx]
                if i != j:
                    #contributions from the source
                    dz = r[i+xx] - r[j+xx]
                    idr = 1.0/sqrt( dx*dx + dy*dy + dz*dz )
                    idr3 = idr*idr*idr

                    ox += (F[j+N]*dz - F[j+xx]*dy )*idr3
                    oy += (F[j+xx]*dx - F[j]   *dz )*idr3
                    oz += (F[j]   *dy - F[j+N]*dx )*idr3

                    #contributions from the image
                    dz = r[i+xx] + r[j+xx]
                    idr = 1.0/sqrt( dx*dx + dy*dy + dz*dz )
                    idr3 = idr*idr*idr
                    rlz = (dx*F[j+N] - dy*F[j])*idr*idr

                    ox += (F[j+N]*dz + F[j+xx]*dy )*idr3
                    oy += (-F[j+xx]*dx - F[j]   *dz )*idr3
                    oz += (F[j]   *dy - F[j+N]*dx )*idr3

                    ox += (ll*h2*(F[j+N]-3*rlz*dx) + 6*dz*dx*rlz)*idr3
                    oy += (ll*h2*(-F[j]  -3*rlz*dy) + 6*dz*dy*rlz)*idr3
                    oz += (ll*h2*(       -3*rlz*dz) + 6*dz*dz*rlz)*idr3

                else:
                    ''' the self contribution from the image point'''
                    h = r[j+xx]
                    hbar_inv = a/h; hbar_inv2 = hbar_inv*hbar_inv
                    hbar_inv4 = hbar_inv2*hbar_inv*hbar_inv
                    
                    muRT = muRT0*(muRT1*hbar_inv2 + muRT2*hbar_inv4)
                    
                    F1 = F[j];
                    F2 = F[j+N]
                    
                    ox += muRT*F2
                    oy += -muRT*F1 
                   
            o[i  ]  += muv*ox
            o[i+N] += muv*oy
            o[i+xx] += muv*oz
        return


    cpdef mobilityRR(self, double [:] o, double [:] r, double [:] T, double ll=0):
        cdef int N=self.N, i, j, xx=2*N
        cdef double dx, dy, dz, idr, idr3, idr5, Tdotidr, tempT, hsq, h2
        cdef double ox, oy, oz, mur=self.mur, muv=self.muv
        
        cdef double llp = 1.0/(1+ll);
        cdef double a = self.b
        cdef double h, hbar_inv, hbar_inv3
        cdef double muRRpara = (1-5*ll)*llp/16.0
        cdef double muRRperp = (1-ll)*llp/8.0
        cdef double mux, muy, muz

        for i in prange(N, nogil=True):
            ox=0; oy=0; oz=0;
            for j in range(N):
                dx = r[i]    - r[j]
                dy = r[i+N]  - r[j+N]
                h2  =  2*r[j+xx]
                if i!=j:
                    dz = r[i+xx] - r[j+xx]
                    idr = 1.0/sqrt( dx*dx + dy*dy + dz*dz )
                    idr3=idr*idr*idr
                    Tdotidr = (T[j]*dx + T[j+N]*dy + T[j+xx]*dz)*idr*idr
                    #
                    ox += (2*T[j]    - 6*Tdotidr*dx)*idr3
                    oy += (2*T[j+N] - 6*Tdotidr*dy)*idr3
                    oz += (2*T[j+xx] - 6*Tdotidr*dz)*idr3

                    ##contributions from the image
                    dz = r[i+xx] + r[j+xx]
                    idr = 1.0/sqrt( dx*dx + dy*dy + dz*dz )
                    idr3 = idr*idr*idr
                    idr5 = idr3*idr*idr
                    Tdotidr = (T[j]*dx + T[j+N]*dy - T[j+xx]*dz)*idr*idr

                    ox += (2*T[j]    - 6*Tdotidr*dx )*idr3
                    oy += (2*T[j+N] - 6*Tdotidr*dy )*idr3
                    oz += (-2*T[j+xx] - 6*Tdotidr*dz )*idr3

                    tempT = -T[j+xx]     # D_i = M_ij D_j, reflection of the strength
                    Tdotidr = ( T[j]*dx + T[j+N]*dy + tempT*dz )*idr*idr
                    
                    ox += ll*12*dz*( dz*T[j]   - 5*dz*Tdotidr*dx + 2*tempT*dx )*idr5
                    oy += ll*12*dz*( dz*T[j+N]- 5*dz*Tdotidr*dy + 2*tempT*dy )*idr5
                    oz += ll*12*dz*( dz*tempT  - 5*dz*Tdotidr*dz + 2*tempT*dz )*idr5

                    ox += -ll*6*h2*(dz*T[j]   -5*Tdotidr*dx*dz + tempT*dx)*idr5
                    oy += -ll*6*h2*(dz*T[j+N]-5*Tdotidr*dy*dz + tempT*dy)*idr5
                    oz += -ll*6*h2*(dz*tempT  -5*Tdotidr*dz*dz + tempT*dz)*idr5 -6*h2*Tdotidr*idr3

                else:
                    ''' self contribution from the image point'''
                    h = r[j+xx]
                    hbar_inv = a/h; hbar_inv3 = hbar_inv*hbar_inv*hbar_inv
                    
                    mux = mur*(1 + muRRpara*hbar_inv3)
                    muy = mux
                    muz = mur*(1 + muRRperp*hbar_inv3)

            o[i  ]  += mux*T[i  ]  - muv*ox
            o[i+N] += muy*T[i+N] - muv*oy
            o[i+xx] += muz*T[i+xx] - muv*oz
        return
    
    
    cpdef propulsionR3t(self, double [:] o, double [:] r, double [:] V3t, double ll=0):
        """
        Compute angular velocity due to 3t mode of the slip :math:`o=\pi^{R,3t}\cdot V^{3t}` 
        ...

        Parameters
        ----------
        o:   np.array
             An array of angular velocities
             An array of size 3*N,
        r:   np.array
             An array of positions
             An array of size 3*N,
        V3t: np.array
             An array of 3t mode of the slip
             An array of size 3*N,
        """

        cdef int N=self.N, i, j, xx=2*N
        cdef double dx, dy, dz, idr, idr3, idr5, Tdotidr, tempT, hsq, h2
        cdef double ox, oy, oz, mur=self.mur, muv=self.muv
        cdef double ll1 = (1-ll)/(1+ll), ll2 = ll/(1+ll);
        
        cdef double llp = 1.0/(1+ll);
        cdef double a = self.b
        cdef double h, hbar_inv, hbar_inv4
        cdef double piR3t0 = 3*ll2/(80.0*a)
        cdef double piR3t, V1, V2

        for i in prange(N, nogil=True):
            ox=0; oy=0; oz=0;
            for j in range(N):
                dx = r[i]    - r[j]
                dy = r[i+N]  - r[j+N]
                h2  =  2*r[j+xx]
                
                if i!=j:
                    pass
                    ##
                    ##
                    ## to be determined 
                    ##
                    ##
                    
                else:
                    ''' self contribution from the image point'''
                    h = r[j+xx]
                    hbar_inv = a/h; hbar_inv4 = hbar_inv*hbar_inv*hbar_inv*hbar_inv
                    
                    piR3t = piR3t0*hbar_inv4
                    
                    V1 = V3t[j];
                    V2 = V3t[j+N]
                    
                    ox += piR3t*V2
                    oy += -piR3t*V1 
                   
            o[i  ]  += ox
            o[i+N] += oy
            o[i+xx] += oz
        return
    
    
    cpdef propulsionR2s(self, double [:] o, double [:] r, double [:] V2s, double ll=0):
        """
        Compute angular velocity due to 2s mode of the slip :math:`o=\pi^{R,2s}\cdot S` 
        ...

        Parameters
        ----------
        o:   np.array
             An array of angular velocities
             An array of size 3*N,
        r:   np.array
             An array of positions
             An array of size 3*N,
        V2s: np.array
             An array of 2s mode of the slip
             An array of size 5*N,
        """

        cdef int N=self.N, i, j, xx=2*N, xx1=3*N , xx2=4*N
        cdef double dx, dy, dz, idr, idr2, idr3, idr5, idr7, aidr2, trS, h2, hsq
        cdef double sxx, syy, szz, sxy, syx, syz, szy, sxz, szx, srr, srx, sry, srz
        cdef double Sljrlx, Sljrly, Sljrlz, Sljrjx, Sljrjy, Sljrjz
        cdef double ox, oy, oz, mur=self.mur, muv=self.muv
        cdef double ll2 = ll/(1+ll);
        
        cdef double llp = 1.0/(1+ll);
        cdef double a = self.b, a_inv = 1.0/a
        cdef double h, hbar_inv, hbar_inv3, hbar_inv5
        cdef double piR2s1 = 5.0/32.0, piR2s2 = -ll2/8.0
        cdef double piR2s

        for i in prange(N, nogil=True):
            ox=0; oy=0;  oz=0;
            for j in  range(N):
                h2 = 2*r[j+xx]; hsq = r[j+xx]*r[j+xx];
                sxx = V2s[j]  ; syy = V2s[j+N]; szz = -sxx-syy;
                sxy = V2s[j+xx]; syx = sxy;
                sxz = V2s[j+xx1]; szx = sxz;
                syz = V2s[j+xx2]; szy = syz;
                dx = r[i]   - r[j]
                dy = r[i+N] - r[j+N]
                if i!=j:
                     pass
                    ##
                    ##
                    ## to be determined 
                    ##
                    ##

                else:
                    ''' the self contribution from the image point'''
                    h = r[j+xx]
                    hbar_inv = a/h; hbar_inv3 = hbar_inv*hbar_inv*hbar_inv
                    hbar_inv5 = hbar_inv3*hbar_inv*hbar_inv
                    
                    piR2s = a_inv*(piR2s1*hbar_inv3 + piR2s2*hbar_inv5)
                    
                    ox += -2*piR2s*syz 
                    oy +=  2*piR2s*sxz

            o[i  ]  += ox
            o[i+N] += oy
            o[i+xx] += oz
        return



   

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
        >>> flow = pystokes.interface.Flow(radius=b, particles=N, viscosity=eta, gridpoints=Ng*Ng)
        >>> 
        >>> # create grid, evaluate flow and plot
        >>> rr, vv = pystokes.utils.gridYZ(dim, L, Ng)
        >>> flow.flowField1s(vv, rr, r, F1s)
        >>> pystokes.utils.plotStreamlinesYZsurf(vv, rr, r, offset=6-1, density=1.4, title='1s')
        """

        cdef int i, j, N=self.N, Nt=self.Nt, xx=2*N
        cdef double dx, dy, dz, idr, idr3, idr5, Fdotidr, tempF, hsq, h2, F3
        cdef double vx, vy, vz, muv=1.0/(8*PI*self.eta), a2=self.b*self.b/6.0

        for i in prange(Nt, nogil=True):
            vx=0; vy=0; vz=0;
            for j in range(N):
                h2  =  2*r[j+xx]; hsq=r[j+xx]*r[j+xx]
                dx = rt[i]    - r[j]
                dy = rt[i+Nt] - r[j+N]
                dz = rt[i+2*Nt]  - r[j+xx]
                idr = 1.0/sqrt( dx*dx + dy*dy + dz*dz )
                idr3=idr*idr*idr
                idr = 1.0/sqrt( dx*dx + dy*dy + dz*dz )
                idr3=idr*idr*idr
                Fdotidr = (F[j] * dx + F[j+N] * dy + F[j+xx] * dz)*idr*idr
                #
                vx += (F[j]   +Fdotidr*dx)*idr + a2*(2*F[j]   -6*Fdotidr*dx)*idr3
                vy += (F[j+N]+Fdotidr*dy)*idr + a2*(2*F[j+N]-6*Fdotidr*dy)*idr3
                vz += (F[j+xx]+Fdotidr*dz)*idr + a2*(2*F[j+xx]-6*Fdotidr*dz)*idr3

                ##contributions from the image
                dz = rt[i+2*Nt] + r[j+xx]
                idr = 1.0/sqrt( dx*dx + dy*dy + dz*dz )
                idr3 = idr*idr*idr
                idr5 = idr3*idr*idr 

                F3 = -F[j+xx]
                Fdotidr = ( F[j]*dx + F[j+N]*dy + F3*dz )*idr*idr

                vx += (F[j]   +Fdotidr*dx)*idr - a2*(2*F[j]   -6*Fdotidr*dx)*idr3
                vy += (F[j+N]+Fdotidr*dy)*idr - a2*(2*F[j+N]-6*Fdotidr*dy)*idr3
                vz += (F3     +Fdotidr*dz)*idr - a2*(2*F3     -6*Fdotidr*dz)*idr3

            vv[i  ]    += muv*vx
            vv[i+Nt]   += muv*vy
            vv[i+2*Nt] += muv*vz
        return 

    
    cpdef flowField2a(  self, double [:] vv, double [:] rt, double [:] r, double [:] T):
        """
        Compute flow field at field points due body torques
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
            An array of body torques
            An array of size 3*N,
    
        Examples
        --------
        An example of the Flow field due to $2a$ mode of force per unit area

        >>> import pystokes, numpy as np, matplotlib.pyplot as plt
        >>> 
        >>> # particle radius, self-propulsion speed, number and fluid viscosity
        >>> b, eta, N = 1.0, 1.0/6.0, 1
        >>> 
        >>> # initialize
        >>> r, T = np.array([0.0, 0.0, 3.4]), np.array([0.0, 1.0, 0])
        >>> 
        >>> # space dimension , extent , discretization
        >>> dim, L, Ng = 3, 10, 64;
        >>> 
        >>> # instantiate the Flow class
        >>> flow = pystokes.interface.Flow(radius=b, particles=N, viscosity=eta, gridpoints=Ng*Ng)
        >>> 
        >>> # create grid, evaluate flow and plot
        >>> rr, vv = pystokes.utils.gridYZ(dim, L, Ng)
        >>> flow.flowField2a(vv, rr, r, T)
        >>> pystokes.utils.plotStreamlinesYZsurf(vv, rr, r, offset=6-1, density=1.4, title='1s')
        """ 

        cdef int N = self.N, i, j, xx=2*N, Nt=self.Nt
        cdef double dx, dy, dz, idr, idr3, rlz, Tdotidr, h2, 
        cdef double vx, vy, vz, muv = 1.0/(8*PI*self.eta)
 
        for i in prange(N, nogil=True):
            vx=0; vy=0; vz=0;
            for j in range(Nt):
                dx = rt[i]   - r[j]   
                dy = rt[i+Nt] - r[j+N]   
                h2 = 2*rt[i+Nt*2]
                    #contributions from the source 
                dz = rt[i+2*Nt] - r[j+xx] 
                idr = 1.0/sqrt( dx*dx + dy*dy + dz*dz )
                idr3 = idr*idr*idr
                 
                vx += (T[j+N]*dz - T[j+xx]*dy )*idr3
                vy += (T[j+xx]*dx - T[j]   *dz )*idr3
                vz += (T[j]   *dy - T[j+N]*dx )*idr3
                    
                #contributions from the image 
                dz = rt[i+2*Nt] + r[j+xx]            
                idr = 1.0/sqrt( dx*dx + dy*dy + dz*dz )
                idr3 = idr*idr*idr
                
                vx += -(T[j+N]*dz - T[j+xx]*dy )*idr3
                vy += -(T[j+xx]*dx - T[j]   *dz )*idr3
                vz += -(T[j]   *dy - T[j+N]*dx )*idr3
                
                rlz = (dx*T[j+N] - dy*T[j])*idr*idr
                vx += (h2*(T[j+N]-3*rlz*dx) + 6*dz*dx*rlz)*idr3
                vy += (h2*(-T[j]  -3*rlz*dy) + 6*dz*dy*rlz)*idr3
                vz += (h2*(       -3*rlz*dz) + 6*dz*dz*rlz)*idr3

                ''' the self contribution from the image point''' 
                dz = rt[i+2*Nt] + r[j+xx]            
                idr = 1.0/dz
                idr3 = idr*idr*idr
                
                vx += -(T[j+N]*dz )*idr3
                vy += -(- T[j] *dz )*idr3
                
                vx += h2*T[j+N]*idr3
                vy += -h2*T[j]*idr3

            vv[i  ]  += muv*vx 
            vv[i+Nt] += muv*vy
            vv[i+2*Nt] += muv*vz
        return 
    
   
    cpdef flowField2s(self, double [:] vv, double [:] rt, double [:] r, double [:] V2s):
        """
        Compute flow field at field points  due to 2s mode of the slip 
        ...

        Parameters
        ----------
        vv:  np.array
             An array of flow at field points
             An array of size 3*Nt,
        rt:  np.array
             An array of field points
             An array of size 3*Nt,
        r:   np.array
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
        >>> flow = pystokes.interface.Flow(radius=b, particles=N, viscosity=eta, gridpoints=Ng*Ng)
        >>> 
        >>> # create grid, evaluate flow and plot
        >>> rr, vv = pystokes.utils.gridXY(dim, L, Ng)
        >>> flow.flowField3t(vv, rr, r, V3t)
        >>> pystokes.utils.plotStreamlinesXY(vv, rr, r, offset=6-1, density=1.4, title='1s')
        """

        cdef int N=self.N,  Nt=self.Nt, xx=2*N, xx1=3*N, xx2=4*N
        cdef int i, j  
        cdef double dx, dy, dz, idr, idr2, idr3, idr5, idr7, aidr2, trS, h2, hsq
        cdef double sxx, syy, szz, sxy, syx, syz, szy, sxz, szx, srr, srx, sry, srz
        cdef double Sljrlx, Sljrly, Sljrlz, Sljrjx, Sljrjy, Sljrjz 
        cdef double vx, vy, vz, mus = (28.0*self.b**3)/24 

        for i in prange(Nt, nogil=True):
            vx=0; vy=0; vz=0;
            for j in  range(N):
                sxx = V2s[j]  ; syy = V2s[j+N]; szz = -sxx-syy;
                sxy = V2s[j+xx]; syx = sxy;
                sxz = V2s[j+xx1]; szx = sxz;
                syz = V2s[j+xx2]; szy = syz;
                
                dx = rt[i]   - r[j]
                dy = rt[i+Nt] - r[j+N]
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
                sxz=-sxz; syz=-syz; szz=szz;     trS=sxx+syy+szz; 
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
                
                vx += -(-Sljrlx + Sljrjx + trS*dx -3*srr*dx)*idr3 ;
                vy += -(-Sljrly + Sljrjy + trS*dy -3*srr*dy)*idr3 ;
                vz += -(-Sljrlz + Sljrjz + trS*dz -3*srr*dz)*idr3 ;
                
            vv[i  ]    += mus*vx
            vv[i+Nt]   += mus*vy
            vv[i+2*Nt] += mus*vz

   
    cpdef flowField3t(self, double [:] vv, double [:] rt, double [:] r, double [:] V3t):
        """
        Compute flow field at field points due to 3t mode of the slip 
        ...

        Parameters
        ----------
        vv:  np.array
             An array of flow at field points
             An array of size 3*Nt,
        rt:  np.array
             An array of field points
             An array of size 3*Nt,
        r:   np.array
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
        >>> flow = pystokes.interface.Flow(radius=b, particles=N, viscosity=eta, gridpoints=Ng*Ng)
        >>> 
        >>> # create grid, evaluate flow and plot
        >>> rr, vv = pystokes.utils.gridXY(dim, L, Ng)
        >>> flow.flowField3t(vv, rr, r, V3t)
        >>> pystokes.utils.plotStreamlinesXY(vv, rr, r, offset=6-1, density=1.4, title='2s')
        """
        cdef int i, j, N=self.N, Nt=self.Nt, xx=2*N
        cdef double dx, dy, dz, idr, idr3, idr5, Ddotidr, tempD, hsq, h2, D3
        cdef double vx, vy, vz, mud = 3.0*self.b*self.b*self.b/5, muv = -1.0*(self.b**5)/10

        for i in prange(Nt, nogil=True):
            vx=0; vy=0; vz=0;
            for j in range(N):
                h2 = 2*r[j+xx]; hsq = r[j+xx]*r[j+xx];
                dx = rt[i]      - r[j]
                dy = rt[i+Nt]   - r[j+N]
                dz = rt[i+2*Nt] - r[j+xx] 
                idr = 1.0/sqrt( dx*dx + dy*dy + dz*dz )
                idr3=idr*idr*idr
                Ddotidr = (V3t[j]*dx + V3t[j+N]*dy + V3t[j+xx]*dz)*idr*idr
                #
                vx += (2*V3t[j]    - 6*Ddotidr*dx)*idr3
                vy += (2*V3t[j+N] - 6*Ddotidr*dy)*idr3
                vz += (2*V3t[j+xx] - 6*Ddotidr*dz)*idr3
                
                ##contributions from the image 
                dz = rt[i+2*Nt] + r[j+xx]        
                idr = 1.0/sqrt( dx*dx + dy*dy + dz*dz )
                idr3 = idr*idr*idr
                idr5 = idr3*idr*idr 
                D3 = -V3t[j+xx]
                Ddotidr = (V3t[j]*dx + V3t[j+N]*dy + D3*dz)*idr*idr
                
                vx += (2*V3t[j]    - 6*Ddotidr*dx )*idr3
                vy += (2*V3t[j+N] - 6*Ddotidr*dy )*idr3
                vz += (2*D3      - 6*Ddotidr*dz )*idr3

            vv[i  ]    += muv*vx
            vv[i+Nt]   += muv*vy
            vv[i+2*Nt] += muv*vz
        return 
    
    
