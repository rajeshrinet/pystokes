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
cdef class PD:
    """
    Power Dissipation (PD)
    
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
        self.a   = radius
        self.N  = particles
        self.eta = viscosity
        self.gammaT = 6*PI*self.eta*self.a
        self.gammaR = 8*PI*self.eta*self.a**3
        self.mu  = 1.0/self.gammaT
        self.muv = 1.0/(8*PI*self.eta)
        self.mur = 1.0/self.gammaR

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

        cdef int i, j, N=self.N, xx=2*N
        cdef double dx, dy, dz, idr, idr3, idr5, Vdotidr, h2, hsq, tempV
        cdef double vx, vy, vz
        cdef double mu=self.mu, muv=self.muv, a2=self.a*self.a/3.0
        cdef double gT=self.gammaT, gg = -gT*gT
        
        cdef double a = self.a
        cdef double h, hbar_inv, hbar_inv3, hbar_inv5
        cdef double muTTpara1 = -9./16., muTTpara2 = 1./8.
        cdef double muTTpara3 = -1./16.
        cdef double muTTperp1 = -9./8., muTTperp2 = 1./2.
        cdef double muTTperp3 = -1./8.
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
                    Vdotidr = (v[j] * dx + v[j+N] * dy + v[j+xx] * dz)*idr*idr
                    #
                    vx += (v[j]   +Vdotidr*dx)*idr + a2*(2*v[j]   -6*Vdotidr*dx)*idr3
                    vy += (v[j+N]+Vdotidr*dy)*idr + a2*(2*v[j+N]-6*Vdotidr*dy)*idr3
                    vz += (v[j+xx]+Vdotidr*dz)*idr + a2*(2*v[j+xx]-6*Vdotidr*dz)*idr3

                    ##contributions from the image
                    dz = r[i+xx] + r[j+xx]
                    idr = 1.0/sqrt( dx*dx + dy*dy + dz*dz )
                    idr3 = idr*idr*idr
                    idr5 = idr3*idr*idr
                    Vdotidr = ( v[j]*dx + v[j+N]*dy + v[j+xx]*dz )*idr*idr

                    vx += -(v[j]   +Vdotidr*dx)*idr - a2*(2*v[j]   -6*Vdotidr*dx)*idr3
                    vy += -(v[j+N]+Vdotidr*dy)*idr - a2*(2*v[j+N]-6*Vdotidr*dy)*idr3
                    vz += -(v[j+xx]+Vdotidr*dz)*idr - a2*(2*v[j+xx]-6*Vdotidr*dz)*idr3

                    tempV  = -v[j+xx]     # F_i = M_ij F_j, reflection of the strength
                    Vdotidr = ( v[j]*dx + v[j+N]*dy + tempV*dz )*idr*idr

                    vx += -h2*(dz*(v[j]   - 3*Vdotidr*dx) + tempV*dx)*idr3
                    vy += -h2*(dz*(v[j+N]- 3*Vdotidr*dy) + tempV*dy)*idr3
                    vz += -h2*(dz*(tempV  - 3*Vdotidr*dz) + tempV*dz)*idr3 + h2*Vdotidr*idr

                    vx += hsq*( 2*v[j]   - 6*Vdotidr*dx )*idr3
                    vy += hsq*( 2*v[j+N]- 6*Vdotidr*dy )*idr3
                    vz += hsq*( 2*tempV  - 6*Vdotidr*dz )*idr3

                    vx += 12*a2*dz*( dz*v[j]   - 5*dz*Vdotidr*dx + 2*tempV*dx )*idr5
                    vy += 12*a2*dz*( dz*v[j+N]- 5*dz*Vdotidr*dy + 2*tempV*dy )*idr5
                    vz += 12*a2*dz*( dz*tempV  - 5*dz*Vdotidr*dz + 2*tempV*dz )*idr5

                    vx += -h2*6*a2*(dz*v[j]   -5*Vdotidr*dx*dz + tempV*dx)*idr5
                    vy += -h2*6*a2*(dz*v[j+N]-5*Vdotidr*dy*dz+ tempV*dy)*idr5
                    vz += -h2*6*a2*(dz*tempV  -5*Vdotidr*dz*dz + tempV*dz)*idr5 -6*a2*h2*Vdotidr*idr3
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

            depsilon += v[i] * gg * (-mux*v[i]    + muv*vx)
            depsilon += v[i+N] * gg * (-muy*v[i+N] + muv*vy)
            depsilon += v[i+xx] * gg * (-muz*v[i+xx] + muv*vz)
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
        cdef double dx, dy, dz, idr, idr3, rlz, Tdotidr, h2,
        cdef double vx, vy, vz, T1, T2, T3
        cdef double muv=self.muv, gg=-self.gammaT*self.gammaR
        
        cdef double a = self.a
        cdef double h, hbar_inv, hbar_inv2, hbar_inv4
        cdef double muTR0 = 4.0/(3*self.a*self.a)
        cdef double muTR2 = 3./32.0
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

                    vx += (o[j+N]*dz - o[j+xx]*dy )*idr3
                    vy += (o[j+xx]*dx - o[j]   *dz )*idr3
                    vz += (o[j]   *dy - o[j+N]*dx )*idr3

                    #contributions from the image
                    dz = r[i+xx] + r[j+xx]
                    idr = 1.0/sqrt( dx*dx + dy*dy + dz*dz )
                    idr3 = idr*idr*idr

                    vx += -(o[j+N]*dz - o[j+xx]*dy )*idr3
                    vy += -(o[j+xx]*dx - o[j]   *dz )*idr3
                    vz += -(o[j]   *dy - o[j+N]*dx )*idr3

                    rlz = (dx*o[j+N] - dy*o[j])*idr*idr
                    vx += (h2*(o[j+N]-3*rlz*dx) + 6*dz*dx*rlz)*idr3
                    vy += (h2*(-o[j]  -3*rlz*dy) + 6*dz*dy*rlz)*idr3
                    vz += (h2*(       -3*rlz*dz) + 6*dz*dz*rlz)*idr3
                else:
                    ''' the self contribution from the image point'''
                    h = r[j+xx]
                    hbar_inv = a/h; hbar_inv2 = hbar_inv*hbar_inv
                    hbar_inv4 = hbar_inv2*hbar_inv*hbar_inv
                    
                    muTR = muTR0*muTR2*hbar_inv4
                    
                    T1 = o[j];
                    T2 = o[j+N]
                    
                    vx += muTR*T2   #change sign here to make up for '-=' below...
                    vy += -muTR*T1  #same here

            depsilon -= v[i] * gg * muv*vx
            depsilon -= v[i+N] * gg * muv*vy
            depsilon -= v[i+xx] * gg * muv*vz
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

        cdef int N=self.N, i, j, xx=2*N, xx1=3*N , xx2=4*N
        cdef double dx, dy, dz, idr, idr2, idr3, idr5, idr4, aidr2, trS, h2, hsq
        cdef double sxx, syy, szz, sxy, syx, syz, szy, sxz, szx, srr, srx, sry, srz
        cdef double Sljrlx, Sljrly, Sljrlz, Sljrjx, Sljrjy, Sljrjz
        cdef double vx, vy, vz, mus=(28.0*self.a**3)/24
        
        cdef double a = self.a
        cdef double h, hbar_inv, hbar_inv2, hbar_inv4, hbar_inv6
        cdef double piT2s11 = 5./16.0, piT2s12 = -1./4.
        cdef double piT2s13 = 5./48.0
        cdef double piT2s21 = -15./48.0, piT2s22 = 15./48.0
        cdef double piT2s23 = -5./48.0
        cdef double piT2s1, piT2s2
        cdef double mus_inv = 1.0/mus, gT = self.gammaT

        for i in prange(N, nogil=True):
            vx=0; vy=0;   vz=0;
            for j in  range(N):
                h2 = 2*r[j+xx]; hsq = r[j+xx]*r[j+xx];
                sxx = S[j]  ; syy = S[j+N]; szz = -sxx-syy;
                sxy = S[j+xx]; syx = sxy;
                sxz = S[j+xx1]; szx = sxz;
                syz = S[j+xx2]; szy = syz;
                dx = r[i]   - r[j]
                dy = r[i+N] - r[j+N]
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
                    h = r[j+xx]
                    hbar_inv = a/h; hbar_inv2 = hbar_inv*hbar_inv
                    hbar_inv4 = hbar_inv2*hbar_inv*hbar_inv
                    hbar_inv6 = hbar_inv4*hbar_inv2
                    
                    #implement superposition approximation only
                    piT2s1 = piT2s11*hbar_inv2 + piT2s12*hbar_inv4 + piT2s13*hbar_inv6
                    piT2s2 = piT2s21*hbar_inv2 + piT2s22*hbar_inv4 + piT2s23*hbar_inv6
                    
                    vx += -mus_inv * 2*piT2s1*sxz
                    vy += -mus_inv * 2*piT2s1*syz
                    vz += -mus_inv * 3*(piT2s2*sxx + piT2s2*syy)

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

        cdef int N=self.N, i, j, xx=2*N
        cdef double dx, dy, dz, idr, idr3, idr5, Ddotidr, tempD, hsq, h2
        cdef double vx, vy, vz, mud = 3.0*self.a*self.a*self.a/5, muv = -1.0*(self.a**5)/10
        
        cdef double a = self.a
        cdef double h, hbar_inv, hbar_inv3, hbar_inv5
        cdef double piT3tpara1 = -1./40.
        cdef double piT3tpara2 = 1./40.
        cdef double piT3tperp1 = -1./10.
        cdef double piT3tperp2 = 1./20.
        cdef double pix, piy, piz
        cdef double gT = self.gammaT

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
                    Ddotidr = (D[j]*dx + D[j+N]*dy + D[j+xx]*dz)*idr*idr
                    #
                    vx += (2*D[j]    - 6*Ddotidr*dx)*idr3
                    vy += (2*D[j+N] - 6*Ddotidr*dy)*idr3
                    vz += (2*D[j+xx] - 6*Ddotidr*dz)*idr3

                    ##contributions from the image
                    dz = r[i+xx] + r[j+xx]
                    idr = 1.0/sqrt( dx*dx + dy*dy + dz*dz )
                    idr3 = idr*idr*idr
                    idr5 = idr3*idr*idr
                    Ddotidr = (D[j]*dx + D[j+N]*dy + D[j+xx]*dz)*idr*idr

                    vx += -(2*D[j]    - 6*Ddotidr*dx )*idr3
                    vy += -(2*D[j+N] - 6*Ddotidr*dy )*idr3
                    vz += -(2*D[j+xx] - 6*Ddotidr*dz )*idr3

                    tempD = -D[j+xx]     # D_i = M_ij D_j, reflection of the strength
                    Ddotidr = ( D[j]*dx + D[j+N]*dy + tempD*dz )*idr*idr

                    vx += 12*dz*( dz*D[j]   - 5*dz*Ddotidr*dx + 2*tempD*dx )*idr5
                    vy += 12*dz*( dz*D[j+N]- 5*dz*Ddotidr*dy + 2*tempD*dy )*idr5
                    vz += 12*dz*( dz*tempD  - 5*dz*Ddotidr*dz + 2*tempD*dz )*idr5

                    vx += -6*h2*(dz*D[j]   -5*Ddotidr*dx*dz + tempD*dx)*idr5
                    vy += -6*h2*(dz*D[j+N]-5*Ddotidr*dy*dz + tempD*dy)*idr5
                    vz += -6*h2*(dz*tempD  -5*Ddotidr*dz*dz + tempD*dz)*idr5 -6*h2*Ddotidr*idr3

                else:
                    ''' self contribution from the image point'''
                    h = r[j+xx]
                    hbar_inv = a/h; hbar_inv3 = hbar_inv*hbar_inv*hbar_inv
                    hbar_inv5 = hbar_inv3*hbar_inv*hbar_inv
                    
                    pix = piT3tpara1*hbar_inv3 + piT3tpara2*hbar_inv5
                    piy = pix
                    piz = piT3tperp1*hbar_inv3 + piT3tperp2*hbar_inv5

            depsilon -= V1s[i] * gT * (-pix*D[j]    + muv*vx)
            depsilon -= V1s[i+N] * gT * (-piy*D[j+N] + muv*vy)
            depsilon -= V1s[i+xx] * gT * (-piz*D[j+xx] + muv*vz)
        return depsilon
   


    ## Angular Velocities
    cpdef frictionRT(self, double depsilon, double [:]v, double [:] o, double [:] r):
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
        cdef double dx, dy, dz, idr, idr3, rlz, Fdotidr, h2
        cdef double ox, oy, oz, muv=self.muv
        
        cdef double a = self.a
        cdef double h, hbar_inv, hbar_inv2, hbar_inv4
        cdef double muRT0 = 4.0/(3*self.a*self.a)
        cdef double muRT2 = -3./32.0
        cdef double muRT, F1, F2
        cdef double gg = -self.gammaT * self.gammaR

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

                    ox += (v[j+N]*dz - v[j+xx]*dy )*idr3
                    oy += (v[j+xx]*dx - v[j]   *dz )*idr3
                    oz += (v[j]   *dy - v[j+N]*dx )*idr3

                    #contributions from the image
                    dz = r[i+xx] + r[j+xx]
                    idr = 1.0/sqrt( dx*dx + dy*dy + dz*dz )
                    idr3 = idr*idr*idr
                    rlz = (dx*v[j+N] - dy*v[j])*idr*idr

                    ox += -(v[j+N]*dz - v[j+xx]*dy )*idr3
                    oy += -(v[j+xx]*dx - v[j]   *dz )*idr3
                    oz += -(v[j]   *dy - v[j+N]*dx )*idr3

                    ox += (h2*(v[j+N]-3*rlz*dx) + 6*dz*dx*rlz)*idr3
                    oy += (h2*(-v[j]  -3*rlz*dy) + 6*dz*dy*rlz)*idr3
                    oz += (h2*(       -3*rlz*dz) + 6*dz*dz*rlz)*idr3

                else:
                    ''' the self contribution from the image point'''
                    h = r[j+xx]
                    hbar_inv = a/h; hbar_inv2 = hbar_inv*hbar_inv
                    hbar_inv4 = hbar_inv2*hbar_inv*hbar_inv
                    
                    muRT = muRT0*muRT2*hbar_inv4
                    
                    F1 = v[j];
                    F2 = v[j+N]
                    
                    ox += -muRT*F2
                    oy += muRT*F1 

            depsilon += o[i] * gg * muv*ox
            depsilon += o[i+N] * gg * muv*oy
            depsilon += o[i+xx] * gg * muv*oz
        return depsilon


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

        cdef int N=self.N, i, j, xx=2*N
        cdef double dx, dy, dz, idr, idr3, idr5, Odotidr, tempO, hsq, h2
        cdef double ox, oy, oz, mur=self.mur, muv=self.muv
        
        cdef double a = self.a
        cdef double h, hbar_inv, hbar_inv3
        cdef double muRRpara = -5./16.0
        cdef double muRRperp = -1./8.0
        cdef double mux, muy, muz
        cdef double gg = -self.gammaR * self.gammaR

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
                    Odotidr = (o[j]*dx + o[j+N]*dy + o[j+xx]*dz)*idr*idr
                    #
                    ox += (2*o[j]    - 6*Odotidr*dx)*idr3
                    oy += (2*o[j+N] - 6*Odotidr*dy)*idr3
                    oz += (2*o[j+xx] - 6*Odotidr*dz)*idr3

                    ##contributions from the image
                    dz = r[i+xx] + r[j+xx]
                    idr = 1.0/sqrt( dx*dx + dy*dy + dz*dz )
                    idr3 = idr*idr*idr
                    idr5 = idr3*idr*idr
                    Odotidr = (o[j]*dx + o[j+N]*dy + o[j+xx]*dz)*idr*idr

                    ox += -(2*o[j]    - 6*Odotidr*dx )*idr3
                    oy += -(2*o[j+N] - 6*Odotidr*dy )*idr3
                    oz += -(2*o[j+xx] - 6*Odotidr*dz )*idr3

                    tempO = -o[j+xx]     # D_i = M_ij D_j, reflection of the strength
                    Odotidr = ( o[j]*dx + o[j+N]*dy + tempO*dz )*idr*idr

                    ox += 12*dz*( dz*o[j]   - 5*dz*Odotidr*dx + 2*tempO*dx )*idr5
                    oy += 12*dz*( dz*o[j+N]- 5*dz*Odotidr*dy + 2*tempO*dy )*idr5
                    oz += 12*dz*( dz*tempO  - 5*dz*Odotidr*dz + 2*tempO*dz )*idr5

                    ox += -6*h2*(dz*o[j]   -5*Odotidr*dx*dz + tempO*dx)*idr5
                    oy += -6*h2*(dz*o[j+N]-5*Odotidr*dy*dz + tempO*dy)*idr5
                    oz += -6*h2*(dz*tempO  -5*Odotidr*dz*dz + tempO*dz)*idr5 -6*h2*Odotidr*idr3

                else:

                    ''' self contribution from the image point'''
                    h = r[j+xx]
                    hbar_inv = a/h; hbar_inv3 = hbar_inv*hbar_inv*hbar_inv
                    
                    mux = mur*(1 + muRRpara*hbar_inv3)
                    muy = mux
                    muz = mur*(1 + muRRperp*hbar_inv3)

            depsilon += o[i] * gg * (-mux*o[i]  - muv*ox)
            depsilon += o[i+N] * gg * (-muy*o[i+N] - muv*oy)
            depsilon += o[i+xx] * gg * (-muz*o[i+xx] - muv*oz)
        return depsilon