cimport cython
from libc.math cimport sqrt, exp, pow, erfc, sin, cos
from cython.parallel import prange
cdef double PI = 3.14159265359
cdef double IPI = (2/sqrt(PI))

@cython.boundscheck(False)
@cython.cdivision(True)
@cython.nonecheck(False)
@cython.wraparound(False)
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
    def __init__(self, radius=1, particles=1, viscosity=1.0, boxSize=10, xi=123456789):
        self.b   = radius
        self.N  = particles
        self.eta = viscosity
        self.L   = boxSize 
        if xi==123456789:
            self.xi = sqrt(PI)/boxSize 
            #Nijboer and De Wette have shown that \pi^{1/2}/V^{1/3} is a good choice for cubic lattices 
        else:
            self.xi = xi 

        self.mu  = 1.0/(6*PI*self.eta*self.b)
        self.muv = 1.0/(8*PI*self.eta)


    cpdef mobilityTT(self, double [:] v, double [:] r, double [:] F, int Nb=6, int Nm=6, double xi0=123456789):
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
        Nb: int 
            Number of periodic boxed summed 
            Default is 10
        Nm: int 
            Number of Fourier modes summed
            Default is 1
        """

        cdef int N=self.N, N1=-(Nm/2)+1, N2=(Nm/2)+1, i, j, ii, jj, kk, Z=2*N, Nbb=2*Nb+1
        cdef double L=self.L,  xi=self.xi, ixi2, siz=Nb*L, mu=self.mu, muv=self.muv
        cdef double a2=self.b*self.b/3, aidr2, k0=2*PI/L, fac=8*PI/(L*L*L), 
        cdef double xdr, xdr2, xdr3, A, B, A1, B1, fdotir, e1, erxdr, m20, xd1, yd1, zd1, mt, mpp
        cdef double xd, yd, zd, dx, dy, dz, idr, kx, ky, kz, k2, ik2, cc, fdotik, vx, vy, vz, fx, fy, fz
        
        if xi0 != 123456789:
            xi = xi0 
        ixi2=1/(xi*xi)
        mt=IPI*xi*self.b*(-3+20*xi*xi*self.b*self.b/3.0); mpp=mu*(1+mt)    
        # include M^2(r=0); see Eq.15, Beenakker JCP 85(3) 1986 

        for i in prange(N, nogil=True):
            vx=0;  vy=0;  vz=0;
            for j in range(N):
                xd=r[i]  -r[j];       xd1=xd-siz; 
                yd=r[i+N]-r[j+N];     yd1=yd-siz;  
                zd=r[i+Z]-r[j+Z];     zd1=zd-siz;
                fx=F[j];  fy=F[j+N];  fz=F[j+Z];

                for ii in range(Nbb):
                    dx = xd1 + ii*L 
                    for jj in range(Nbb):               
                        dy = yd1 + jj*L 
                        for kk in range(Nbb):                 
                            dz = zd1 + kk*L
                            if ii==jj==kk==Nb and i==j:
                                pass
                            else:    
                                idr = 1.0/sqrt( dx*dx + dy*dy + dz*dz)
                                xdr=xi/idr;  xdr2=xdr*xdr;  xdr3=xdr2*xdr;  
                                erxdr = erfc(xdr);  e1=IPI*exp(-xdr2);
                                fdotir = (fx*dx + fy*dy + fz*dz)*idr*idr
                                aidr2 = a2*idr*idr
                                
                                A = erxdr + e1*(2*xdr3-3*xdr)
                                B = erxdr + e1*(xdr - 2*xdr3)
                                A += (2*erxdr  + e1*( 2*xdr + 28*xdr3 - 40*xdr3*xdr2 + 8*xdr3*xdr3*xdr ))*aidr2 # finite size correction
                                B += (-6*erxdr + e1*(-6*xdr - 4*xdr3  + 32*xdr3*xdr2 - 8*xdr3*xdr3*xdr ))*aidr2  #finite size 
                                A = A*idr
                                B = B*fdotir*idr

                                vx += A*fx + B*dx
                                vy += A*fy + B*dy
                                vz += A*fz + B*dz

                # Fourier space sum
                for ii in range(N1, N2):
                    kx = k0*ii;
                    for jj in range(N1, N2):               
                        ky = k0*jj;
                        for kk in range(N1, N2):                 
                            kz = k0*kk;
                            if kx != 0 or ky != 0 or kz != 0:
                                k2 = (kx*kx + ky*ky + kz*kz); ik2=1/k2
                                fdotik = (fx*kx + fy*ky + fz*kz )*ik2
                                cc = fac*(1-a2*k2)*cos( kx*xd+ky*yd+kz*zd )*(1 + 0.25*k2*ixi2 + 0.125*ixi2*ixi2*k2*k2)*exp(-0.25*ixi2*k2)*ik2

                                vx += cc*(fx - fdotik*kx) 
                                vy += cc*(fy - fdotik*ky) 
                                vz += cc*(fz - fdotik*kz) 
        
            v[i]   += mpp*F[i]   + muv*vx 
            v[i+N] += mpp*F[i+N] + muv*vy 
            v[i+Z] += mpp*F[i+Z] + muv*vz
        return 
    
    
    cpdef mobilityTR(   self, double [:] v, double [:] r, double [:] T, int Nb=6, int Nm=6, double xi0=123456789):
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
        Nb: int 
            Number of periodic boxed summed 
            Default is 6
        Nm: int 
            Number of Fourier modes summed
            Default is 6
        """

        cdef: 
            double L = self.L,  xi=self.xi, ixi2, vx, vy, vz, muv=0.5*self.muv
            int N = self.N, N1 = -(Nm/2)+1, N2 =  (Nm/2)+1, i, i1, j, j1, ii, jj, kk, Z=2*N
            double xdr, xdr2, xdr3, e1, erxdr, fac=8*PI/(L*L*L),
            double dx, dy, dz, idr, idr3, kx, ky, kz, k2, cc, D 
        if xi0 != 123456789:
            xi = xi0 
        ixi2=1/(xi*xi)


        for i in prange(N, nogil=True):
            vx=0; vy=0; vz=0
            for j in range(N):
                for ii in range(2*Nb+1):
                    for jj in range(2*Nb+1):               
                        for kk in range(2*Nb+1):                 
                            if ii==jj==kk==Nb and i==j:
                                pass
                            else:    
                                dx = r[i]   - r[j]-Nb*L + ii*L 
                                dy = r[i+N] - r[j+N]-Nb*L + jj*L 
                                dz = r[i+Z] - r[j+Z]-Nb*L + kk*L
                                idr = 1.0/sqrt( dx*dx + dy*dy + dz*dz)
                                idr3 = idr*idr*idr
                                xdr    = xi/idr;    erxdr = erfc(xdr) 
                                xdr2   = xdr*xdr ; e1 = IPI*exp(-xdr2);
                                D      = (2*erfc(xdr) + e1*(2*xdr - 12*xdr2*xdr + 4*xdr2*xdr2*xdr))*idr3

                                vx -= D*(dy*T[j+Z] - dz*T[j+N])
                                vy -= D*(dz*T[j]   - dx*T[j+Z])
                                vz -= D*(dx*T[j+N] - dy*T[j  ])
        # Fourier space sum
        for i in prange(N, nogil=True):
            for j  in range(N):
                dx = r[i]  -r[j]
                dy = r[i+N]-r[j+N]
                dz = r[i+Z]-r[j+Z]
                for ii in range(N1, N2):
                    kx = (2*PI/L)*ii;
                    for jj in range(N1, N2):               
                        ky = (2*PI/L)*jj;
                        for kk in range(N1, N2):                 
                            kz = (2*PI/L)*kk;
                            if kx != 0 or ky != 0 or kz != 0:
                                k2 = kx*kx + ky*ky + kz*kz    
                                cc = fac*sin( kx*dx+ky*dy+kz*dz )* (1 + 0.25*k2*ixi2 + 0.125*ixi2*ixi2*k2*k2)*exp(-0.25*ixi2*k2)/(k2)

                                vx += cc*( T[j+N]*kz - T[j+Z]*ky  ) 
                                vy += cc*( T[j+Z]*kx - T[j]  *kz  ) 
                                vz += cc*( T[j]  *ky - T[j+N]*kx  ) 
            v[i]   += muv*vx 
            v[i+N] += muv*vy 
            v[i+Z] += muv*vz 
        return 
    
    
    cpdef propulsionT2s(self, double [:] v, double [:] r, double [:] V2s, int Nb=6, int Nm=6, double xi0=123456789):
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
        Nb: int 
            Number of periodic boxed summed 
            Default is 6
        Nm: int 
            Number of Fourier modes summed
            Default is 6
        """

        cdef: 
            int N=self.N, N1=-(Nm/2)+1, N2=(Nm/2)+1, i, j, ii, jj, kk, Z=2*N, Nbb=2*Nb+1, xx1=3*N, xx2=4*N
            double L = self.L,  xi=self.xi, siz=Nb*L, ixi2
            double xdr, xdr2, xdr3, xdr5,  E1, E2, erxdr, e1, sxx, syy, sxy, sxz, syz, srr, srx, sry, srz
            double dx, dy, dz, idr, idr3, kx, ky, kz, k2, ik2, cc, kdotr, vx, vy, vz, k0=2*PI/L, ixk2, fac=8*PI/(L*L*L)
            double a2 = self.b*self.b*4.0/15, aidr2, xd1, yd1, zd1, xd, yd, zd, mus = (28.0*self.b**3)/24 
        if xi0 != 123456789:
            xi = xi0 
        ixi2 = 1/(xi*xi)
        
        for i in prange(N, nogil=True):
            vx=0; vy=0; vz=0;
            for j in range(N):
                sxx = V2s[j]
                syy = V2s[j+N]
                sxy = V2s[j+Z]
                sxz = V2s[j+xx1]
                syz = V2s[j+xx2]
                xd=r[i]-r[j];          xd1=xd-siz; 
                yd=r[i+N]-r[j+N];    yd1=yd-siz;  
                zd=r[i+Z]-r[j+Z];    zd1=zd-siz;
                
                for ii in range(Nbb):
                    dx = xd1 + ii*L 
                    for jj in range(Nbb):               
                        dy = yd1 + jj*L 
                        for kk in range(Nbb):                 
                            dz = zd1 + kk*L
                            if ii==jj==kk==Nb and i==j:
                                pass
                            else:    
                                idr = 1.0/sqrt( dx*dx + dy*dy + dz*dz)
                                idr3 = idr*idr*idr; aidr2=a2*idr*idr
                                xdr = xi/idr; xdr2=xdr*xdr; xdr3 = xdr2*xdr; xdr5 = xdr3*xdr2;
                                erxdr   = erfc(xdr);   e1  = IPI*exp(-xdr2);
                                E1 =  e1*(8*xdr3 - 4*xdr5 ) 
                                E2 = -3*erxdr + e1*(-3*xdr - 2*xdr3  + 4*xdr5 ) 
                                
                                srx = sxx*dx + sxy*dy + sxz*dz  
                                sry = sxy*dx + syy*dy + syz*dz  
                                srz = sxz*dx + syz*dy - (sxx+syy)*dz 
                                srr = srx*dx + sry*dy + srz*dz

                                # E1 += -12*erxdr+ e1*(-12*xdr - 8*xdr3  - 104*xdr5 + 104*xdr5*xdr2 - 16*xdr3*xdr3*xdr3)*aidr2
                                # E2 += 30*erxdr + e1*(30*xdr  + 20*xdr3 + 8*xdr5   - 80*xdr5*xdr2  + 16*xdr3*xdr3*xdr3)*aidr2
                                E1  = E1*idr3
                                E2  = E2*srr*idr*idr*idr3

                                vx += E1*srx + E2*dx
                                vy += E1*sry + E2*dy
                                vz += E1*srz + E2*dz
                #Fourier part
                for ii in range(N1, N2):
                    kx = k0*ii;
                    for jj in range(N1, N2):               
                        ky = k0*jj;
                        for kk in range(N1, N2):                 
                            kz = k0*kk;
                            if kx != 0 or ky != 0 or kz != 0:  
                                k2 = (kx*kx + ky*ky + kz*kz); ik2=1.0/k2    
                                ixk2 = 0.25*k2*ixi2
                                #cc = -fac*(1-a2*k2)*sin(kx*xd+ky*yd+kz*zd)*(1+ixk2+2*ixk2*ixk2)*exp(-ixk2)*ik2
                                cc = -fac*sin(kx*xd+ky*yd+kz*zd)*(1+ixk2+2*ixk2*ixk2)*exp(-ixk2)*ik2

                                srx = sxx*kx + sxy*ky + sxz*kz  
                                sry = sxy*kx + syy*ky + syz*kz  
                                srz = sxz*kx + syz*ky - (sxx+syy)*kz 
                                srr = (srx*kx + sry*ky + srz*kz)*ik2
                                
                                vx += cc* (srx - srr*kx) 
                                vy += cc* (sry - srr*ky)
                                vz += cc* (srz - srr*kz)

            v[i]   += mus*vx
            v[i+N] += mus*vy
            v[i+Z] += mus*vz
        return 
      

    cpdef propulsionT3t(self, double [:] v, double [:] r, double [:] V3t, int Nb=6, int Nm=6, double xi0=123456789):
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
        V3t: np.array
            An array of 3t mode of the slip
            An array of size 3*N,
        Nb: int 
            Number of periodic boxed summed 
            Default is 6
        Nm: int 
            Number of Fourier modes summed
            Default is 6
        """

        cdef double L = self.L,  xi=self.xi, siz=Nb*L, k0=(2*PI/L), fac=8.0*PI/(L*L*L)
        cdef double ixi2, vx, vy, vz
        cdef int N = self.N, N1=-(Nm/2)+1, N2=(Nm/2)+1, i, i1, j, j1, ii, jj, kk, Z=2*N, Nbb=2*Nb+1
        cdef double xdr, xdr2, xdr3, A1, B1, V3tdotik2, Ddotidr2, e1, erxdr, dx, dy, dz, idr, idr3,  kx, ky, kz, k2, cc
        cdef double b3=self.b*self.b*self.b, muv=-(b3)/20, xi2
        cdef double xd, yd, zd, xd1, yd1, zd1
        if xi0 != 123456789:
            xi = xi0 
        xi2 = xi*xi 
        ixi2 = 1/(xi2)
        mud = 0.2# - IPI*(4*xi2*xi*b3/3.0) ## adding the M^2(r=0) contribution

        
        for i in prange(N, nogil=True):
            vx=0;  vy=0; vz=0;
            for j in range(N):
                xd=r[i]-r[j];          xd1=xd-siz; 
                yd=r[i+N]-r[j+N];    yd1=yd-siz;  
                zd=r[i+Z]-r[j+Z];    zd1=zd-siz;
                
                for ii in range(Nbb):
                    dx = xd1 + ii*L 
                    for jj in range(Nbb):               
                        dy = yd1 + jj*L 
                        for kk in range(Nbb):                 
                            dz = zd1 + kk*L
                            if ii==jj==kk==Nb and i==j:
                                pass
                            else:    
                                idr = 1.0/sqrt( dx*dx + dy*dy + dz*dz)
                                idr3=idr*idr*idr
                                xdr = xi/idr; xdr2=xdr*xdr; xdr3 = xdr2*xdr;
                                erxdr   = erfc(xdr); e1=IPI*exp(-xdr2);  
                                A1 = (2*erxdr  + e1*( 2*xdr+28*xdr3-40*xdr3*xdr2+8*xdr3*xdr3*xdr ))*idr3
                                B1 = (-6*erxdr + e1*(-6*xdr-4*xdr3 +32*xdr3*xdr2-8*xdr3*xdr3*xdr ))*idr3
                                B1 = B1*(V3t[j]*dx + V3t[j+N]*dy + V3t[j+Z]*dz )*idr*idr

                                vx += A1*V3t[j]   + B1*dx
                                vy += A1*V3t[j+N] + B1*dy
                                vz += A1*V3t[j+Z] + B1*dz
                #Fourier part
                for ii in range(N1, N2):
                    kx = k0*ii;
                    for jj in range(N1, N2):               
                        ky = k0*jj;
                        for kk in range(N1, N2):                 
                            kz = k0*kk;
                            if kx != 0 or ky != 0 or kz != 0:  
                                k2 = (kx*kx + ky*ky + kz*kz)    
                                cc = -fac*cos(kx*xd+ky*yd+kz*zd)*(1 + 0.25*k2*ixi2 + 0.125*ixi2*ixi2*k2*k2)*exp(-0.25*ixi2*k2)
                                V3tdotik2  = (V3t[j]*kx + V3t[j+N]*ky + V3t[j+Z]*kz)/k2
                                
                                vx += cc*( V3t[j]   - V3tdotik2*kx ) 
                                vy += cc*( V3t[j+N] - V3tdotik2*ky ) 
                                vz += cc*( V3t[j+Z] - V3tdotik2*kz ) 

            v[i]   += mud*V3t[i]    + muv*vx
            v[i+N] += mud*V3t[i+N]  + muv*vy
            v[i+Z] += mud*V3t[i+Z] + muv*vz
        return


    cpdef propulsionT3s(  self, double [:] v, double [:] r, double [:] V3s, int Nb=6, int Nm=6, double xi0=123456789):
        """
        Compute velocity due to 3s mode of the slip :math:`v=\pi^{T,3s}\cdot V^{3s}` 
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
        Nb: int 
            Number of periodic boxed summed 
            Default is 6
        Nm: int 
            Number of Fourier modes summed
            Default is 6
        """
        cdef: 
            double L = self.L,  xi=self.xi  
            double ixi2, vx, vy, vz
            int N = self.N, N1 = -(Nm/2)+1, N2 =  (Nm/2)+1, i, j,  ii, jj, kk, Z=2*N
            double xdr, xdr2, xdr3, xdr5, xdr7, e1, erxdr, D1, D2, D11, D22
            double dx, dy, dz, idr, idr5, idr7, kx, ky, kz, k2, cc, fac=8*PI/(L*L*L)
            double aidr2, grrr, grrx, grry, grrz, gxxx, gyyy, gxxy, gxxz, gxyy, gxyz, gyyz
            double a2 = self.b*self.b*5/21
        if xi0 != 123456789:
            xi = xi0 
        ixi2 = 1/(xi*xi)
        
        for i in prange(N, nogil=True):
            vx=0; vy=0; vz=0
            for j in range(N):
                gxxx = V3s[j]
                gyyy = V3s[j+N]
                gxxy = V3s[j+2*N]
                gxxz = V3s[j+3*N]
                gxyy = V3s[j+4*N]
                gxyz = V3s[j+5*N]
                gyyz = V3s[j+6*N]
                dx = r[i]  -r[j]
                dy = r[i+N]-r[j+N]
                dz = r[i+Z]-r[j+Z]
                
                for ii in range(2*Nb+1):
                    for jj in range(2*Nb+1):               
                        for kk in range(2*Nb+1):                 
                            if ii==Nb and jj==Nb and kk==Nb and i==j:
                                pass
                            else:    
                                dx = dx - Nb*L + ii*L 
                                dy = dy - Nb*L + jj*L 
                                dz = dz - Nb*L + kk*L
                                idr = 1.0/sqrt( dx*dx + dy*dy + dz*dz)
                                idr5   = idr*idr*idr*idr*idr 
                                idr7   = idr5*idr*idr
                                xdr    = xi/idr 
                                xdr = xi/idr; xdr2=xdr*xdr; xdr3=xdr2*xdr; xdr5 = xdr3*xdr2
                                erxdr   = erfc(xdr); e1=IPI*exp(-xdr2); xdr7= xdr5*xdr2 
                                D1 = ( -3*erxdr + e1*(-3*xdr  -2*xdr3 - 20*xdr5 + 8*xdr7)+ 4*a2*idr*idr )*idr5    ;
                                D2 = ( 15*erxdr + e1*(15*xdr +10*xdr3 +4*xdr5 - 8*xdr7)  - 28*a2*idr*idr )*idr7;
                                D11 = a2*(-9*xdr+9*xdr3-267*xdr5+632*xdr7-296*xdr7*xdr2+32*xdr7*xdr2*xdr2)*idr7
                                D22 = a2*(45*xdr-45*xdr3-60*xdr5+320*xdr7-272*xdr7*xdr2-32*xdr7*xdr2*xdr2)*idr7*idr*idr


                                grrr = gxxx*dx*(dx*dx-3*dz*dz) + 3*gxxy*dy*(dx*dx-dz*dz) + gxxz*dz*(3*dx*dx-dz*dz) +\
                                   3*gxyy*dx*(dy*dy-dz*dz) + 6*gxyz*dx*dy*dz + gyyy*dy*(dy*dy-3*dz*dz) +  gyyz*dz*(3*dy*dy-dz*dz) 
                                grrx = gxxx*(dx*dx-dz*dz) + gxyy*(dy*dy-dz*dz) +  2*gxxy*dx*dy + 2*gxxz*dx*dz  +  2*gxyz*dy*dz
                                grry = gxxy*(dx*dx-dz*dz) + gyyy*(dy*dy-dz*dz) +  2*gxyy*dx*dy + 2*gxyz*dx*dz  +  2*gyyz*dy*dz
                                grrz = gxxz*(dx*dx-dz*dz) + gyyz*(dy*dy-dz*dz) +  2*gxyz*dx*dy - 2*(gxxx+gxyy)*dx*dz  - 2*(gxxy+gyyy)*dy*dz
                                D1 = D1 + D11;  D2 = D2 + D22; 
                                vx += D1*grrx - D2*grrr*dx
                                vy += D1*grry - D2*grrr*dy
                                vz += D1*grrz - D2*grrr*dz
                #Fourier part
                for ii in range(N1, N2):
                    kx = (2*PI/L)*ii;
                    for jj in range(N1, N2):               
                        ky = (2*PI/L)*jj;
                        for kk in range(N1, N2):                 
                            kz = (2*PI/L)*kk;
                            if kx == 0 and ky == 0 and kz == 0:  
                                pass
                            else:    
                                k2 = (kx*kx + ky*ky + kz*kz)    
                                cc = -fac*(1-a2*k2)*cos( kx*dx+ky*dy+kz*dz )*(1 + 0.25*k2*ixi2 + 0.125*ixi2*ixi2*k2*k2)*exp(-0.25*ixi2*k2)
                
                                grrr = (gxxx*kx*(kx*kx-3*kz*kz) + 3*gxxy*ky*(kx*kx-kz*kz) + gxxz*kz*(3*kx*kx-kz*kz) +\
                                   3*gxyy*kx*(ky*ky-kz*kz) + 6*gxyz*kx*ky*kz + gyyy*ky*(ky*ky-3*kz*kz) +  gyyz*kz*(3*ky*ky-kz*kz) )/k2
                                grrx = gxxx*(kx*kx-kz*kz) + gxyy*(ky*ky-kz*kz) +  2*gxxy*kx*ky + 2*gxxz*kx*kz  +  2*gxyz*ky*kz
                                grry = gxxy*(kx*kx-kz*kz) + gyyy*(ky*ky-kz*kz) +  2*gxyy*kx*ky + 2*gxyz*kx*kz  +  2*gyyz*ky*kz
                                grrz = gxxz*(kx*kx-kz*kz) + gyyz*(ky*ky-kz*kz) +  2*gxyz*kx*ky - 2*(gxxx+gxyy)*kx*kz  - 2*(gxxy+gyyy)*ky*kz
                                
                                vx += cc*(grrx - grrr*kx) 
                                vy += cc*(grry - grrr*ky) 
                                vz += cc*(grrz - grrr*kz) 
            v[i]   += vx
            v[i+N]+= vy
            v[i+Z]+= vz

        return

    
    cpdef propulsionT3a(  self, double [:] v, double [:] r, double [:] V3a, int Nb=6, int Nm=6, double xi0=123456789):
        """
        Compute velocity due to 3a mode of the slip :math:`v=\pi^{T,3a}\cdot V^{3a}` 
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
        Nb: int 
            Number of periodic boxed summed 
            Default is 6
        Nm: int 
            Number of Fourier modes summed
            Default is 6
        """

        cdef: 
            double L = self.L,  xi=self.xi, ixi2
            int N = self.N, N1 = -(Nm/2)+1, N2 =  (Nm/2)+1, i, i1, j, j1, j2, ii, jj, kk, Z=2*N
            double dx, dy, dz, idr, idr5, vxx, vyy, vxy, vxz, vyz, vrx, vry, vrz, vkx, vky, vkz, fac=8*PI/(L*L*L)
            double s1, kx, ky, kz, k2, xdr, xdr2, cc 
        if xi0 != 123456789:
            xi = xi0 
        ixi2 = 1/(xi*xi)

 
        for i in prange(N, nogil=True):
            for j in range(N):
                vxx = V3a[j]
                vyy = V3a[j+N]
                vxy = V3a[j+2*N]
                vxz = V3a[j+3*N]
                vyz = V3a[j+4*N]
                for ii in range(2*Nb+1):
                    for jj in range(2*Nb+1):               
                        for kk in range(2*Nb+1):                 
                            if ii==Nb and jj==Nb and kk==Nb and i==j:
                                pass
                            else:    
                                dx = r[i]   - r[j]     -Nb*L + ii*L 
                                dy = r[i+N] - r[j+N]  -Nb*L + jj*L 
                                dz = r[i+Z] - r[j+Z]-Nb*L + kk*L
                                idr = 1.0/sqrt( dx*dx + dy*dy + dz*dz )
                                idr5 = pow(idr, 5)
                                xdr = xi/idr; xdr2 = xdr*xdr;  
                                s1 = -6*erfc(xdr) + IPI*(-16*xdr2*xdr + 32*xdr2*xdr2*xdr - 8*xdr2*xdr2*xdr2*xdr)*exp(-xdr2)

                                vrx = vxx*dx +  vxy*dy + vxz*dz  
                                vry = vxy*dx +  vyy*dy + vyz*dz  
                                vrz = vxz*dx +  vyz*dy - (vxx+vyy)*dz 
                                
                                v[i]    -= s1*( dy*vrz - dz*vry )*idr5
                                v[i+N] -= s1*( dz*vrx - dx*vrz )*idr5
                                v[i+Z] -= s1*( dx*vry - dy*vrx )*idr5
                #Fourier part
                N1 = -(Nm/2)+1
                N2 =  (Nm/2)+1
                dx = r[i]      - r[j]
                dy = r[i+N]   - r[j+N]
                dz = r[i+Z] - r[j+Z] 
                for ii in range(N1, N2):
                    kx = (2*PI/L)*ii;
                    for jj in range(N1, N2):               
                        ky = (2*PI/L)*jj;
                        for kk in range(N1, N2):                 
                            kz = (2*PI/L)*kk;
                            if kx != 0 or ky != 0 or kz != 0:  
                                k2 = (kx*kx + ky*ky + kz*kz)    
                                cc = 8*PI*(1 + 0.25*k2*ixi2 + 0.125*ixi2*ixi2*k2*k2)*exp(-0.25*ixi2*k2)/(k2*L*L*L)
                                vkx = vxx*kx +  vxy*ky + vxz*kz  
                                vky = vxy*kx +  vyy*ky + vyz*kz  
                                vkz = vxz*kx +  vyz*ky - (vxx+vyy)*kz 
                                
                                v[i]   += cc*( ky*vkz - kz*vky) 
                                v[i+N] += cc*( kz*vkx - kx*vkz) 
                                v[i+Z] += cc*( kx*vky - ky*vkx) 
        return


    cpdef propulsionT4a(  self, double [:] v, double [:] r, double [:] V4a, int Nb=6, int Nm=6, double xi0=123456789):
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
        V4a: np.array
            An array of 4a mode of the slip
            An array of size 7*N,
        Nb: int 
            Number of periodic boxed summed 
            Default is 6
        Nm: int 
            Number of Fourier modes summed
            Default is 6
        """

        cdef: 
            double L = self.L,  xi=self.xi, fac=8*PI/(L*L*L)
            double ixi2, vx, vy, vz
            int N = self.N, N1 = -(Nm/2)+1, N2 =  (Nm/2)+1, i, i1, j, j1, j2, ii, jj, kk, Z=2*N
            double dx, dy, dz, idr, idr7, mrrx, mrry, mrrz, mkkx, mkky, mkkz, mxxx, myyy, mxxy, mxxz, mxyy, mxyz, myyz, 
            double s2, kx, ky, kz, k2, xdr, e1, xdr2, cc, 
        if xi0 != 123456789:
            xi = xi0 
        ixi2 = 1/(xi*xi)


        for i in prange(N, nogil=True):
            vx=0; vy=0; vz=0;
            for j in range(N):
                mxxx = V4a[j]
                myyy = V4a[j+N]
                mxxy = V4a[j+2*N]
                mxxz = V4a[j+3*N]
                mxyy = V4a[j+4*N]
                mxyz = V4a[j+5*N]
                myyz = V4a[j+6*N]

                for ii in range(2*Nb+1):
                    for jj in range(2*Nb+1):               
                        for kk in range(2*Nb+1):                 
                            if ii==Nb and jj==Nb and kk==Nb and i==j:
                                pass
                            else:    
                                dx = r[i]   - r[j]     -Nb*L + ii*L 
                                dy = r[i+N] - r[j+N]  -Nb*L + jj*L 
                                dz = r[i+Z] - r[j+Z]-Nb*L + kk*L
                                idr = 1.0/sqrt( dx*dx + dy*dy + dz*dz )
                                idr7 = pow(idr, 7)
                                xdr = xi/idr; xdr2 = xdr*xdr; 
                                s2 = 30*erfc(xdr) + IPI*xdr*(6 + 32*xdr2 -32*xdr2*xdr2 - 80*xdr2*xdr2*xdr2 + 16*xdr2*xdr2*xdr2*xdr2)*exp(-xdr2)
                                mrrx = mxxx*(dx*dx-dz*dz) + mxyy*(dy*dy-dz*dz) +  2*mxxy*dx*dy + 2*mxxz*dx*dz  +  2*mxyz*dy*dz
                                mrry = mxxy*(dx*dx-dz*dz) + myyy*(dy*dy-dz*dz) +  2*mxyy*dx*dy + 2*mxyz*dx*dz  +  2*myyz*dy*dz
                                mrrz = mxxz*(dx*dx-dz*dz) + myyz*(dy*dy-dz*dz) +  2*mxyz*dx*dy - 2*(mxxx+mxyy)*dx*dz - 2*(mxxy+myyy)*dy*dz
                                
                                vx  += -s2*( dy*mrrz - dz*mrry )*idr7
                                vy  += -s2*( dz*mrrx - dx*mrrz )*idr7
                                vz  += -s2*( dx*mrry - dy*mrrx )*idr7
                #Fourier part
                N1 = -(Nm/2)+1
                N2 =  (Nm/2)+1
                dx = r[i]  - r[j]     
                dy = r[i+N]- r[j+N]  
                dz = r[i+Z]- r[j+Z]
                for ii in range(N1, N2):
                    kx = (2*PI/L)*ii;
                    for jj in range(N1, N2):               
                        ky = (2*PI/L)*jj;
                        for kk in range(N1, N2):                 
                            kz = (2*PI/L)*kk;
                            if kx != 0 or ky != 0 or kz != 0:  
                                k2 = (kx*kx + ky*ky + kz*kz)    
                                cc = 8*PI*sin( kx*dx+ky*dy+kz*dz )*(1 + 0.25*k2*ixi2 + 0.125*ixi2*ixi2*k2*k2)*exp(-0.25*ixi2*k2)/(k2*L*L*L)
                                mkkx = mxxx*(kx*kx-kz*kz) + mxyy*(ky*ky-kz*kz) +  2*mxxy*kx*ky + 2*mxxz*kx*kz  +  2*mxyz*ky*kz
                                mkky = mxxy*(kx*kx-kz*kz) + myyy*(ky*ky-kz*kz) +  2*mxyy*kx*ky + 2*mxyz*kx*kz  +  2*myyz*ky*kz
                                mkkz = mxxz*(kx*kx-kz*kz) + myyz*(ky*ky-kz*kz) +  2*mxyz*kx*ky - 2*(mxxx+mxyy)*kx*kz-2*(mxxy+myyy)*ky*kz
                                
                                vx += cc*( ky*mkkz - kz*mkky) 
                                vy += cc*( kz*mkkx - kx*mkkz) 
                                vz += cc*( kx*mkky - ky*mkkx) 
            v[i]    += vx
            v[i+N] += vy
            v[i+Z]+= vz
        return


    ## Angular velocities

    cpdef mobilityRT(self, double [:] o, double [:] r, double [:] F, int Nb=6, int Nm=6, double xi0=123456789):
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
        Nb: int 
            Number of periodic boxed summed 
            Default is 6
        Nm: int 
            Number of Fourier modes summed
            Default is 6
        """

        cdef: 
            int N = self.N, N1 = -(Nm/2)+1, N2 =  (Nm/2)+1, i, i1, j, j1, ii, jj, kk, Z=2*N
            double L = self.L,  xi=self.xi, fac=8*PI/(L*L*L), muv=0.5*self.muv
            double  ixi2, ox, oy, oz
            double xdr, xdr2, xdr3, e1, erxdr 
            double dx, dy, dz, idr, idr3, kx, ky, kz, k2, cc, D 
        if xi0 != 123456789:
            xi = xi0 
        ixi2 = 1/(xi*xi)


        for i in prange(N, nogil=True):
            ox=0; oy=0; oz=0
            for j in range(N):
                for ii in range(2*Nb+1):
                    for jj in range(2*Nb+1):               
                        for kk in range(2*Nb+1):                 
                            if ii==jj==kk==Nb and i==j:
                                pass
                            else:    
                                dx = r[i]   - r[j]  -Nb*L + ii*L 
                                dy = r[i+N] - r[j+N]-Nb*L + jj*L 
                                dz = r[i+Z] - r[j+Z]-Nb*L + kk*L
                                idr = 1.0/sqrt( dx*dx + dy*dy + dz*dz)
                                idr3 = idr*idr*idr
                                xdr    = xi/idr;    erxdr = erfc(xdr) 
                                xdr2   = xdr*xdr ; e1 = IPI*exp(-xdr2);
                                D      = (2*erfc(xdr) + e1*(2*xdr - 12*xdr2*xdr + 4*xdr2*xdr2*xdr))*idr3
                                
                                ox += D*(F[j+N]*dz - F[j+Z]*dy )
                                oy += D*(F[j+Z]*dx - F[j]*dz   )
                                oz += D*(F[j]*dy    - F[j+N]*dx)
        # Fourier space sum
        for i in prange(N, nogil=True):
            i1 = i*3
            for j  in range(N):
                j1 = j*3
                dx = r[i]  -r[j]
                dy = r[i+N]-r[j+N]
                dz = r[i+Z]-r[j+Z]
                for ii in range(N1, N2):
                    kx = (2*PI/L)*ii;
                    for jj in range(N1, N2):               
                        ky = (2*PI/L)*jj;
                        for kk in range(N1, N2):                 
                            kz = (2*PI/L)*kk;
                            if kx != 0 or ky != 0 or kz != 0:
                                k2 = kx*kx + ky*ky + kz*kz    
                                cc = fac*sin( kx*dx+ky*dy+kz*dz )* (1 + 0.25*k2*ixi2 + 0.125*ixi2*ixi2*k2*k2)*exp(-0.25*ixi2*k2)/(k2)

                                ox += cc*( F[j+Z]*ky - F[j+N]*kz) 
                                oy += cc*( F[j]*kz   - F[j+Z]*kx) 
                                oz += cc*( F[j+N]*kx - F[j]*ky  ) 
            o[i]   += muv*ox
            o[i+N] += muv*oy
            o[i+Z] += muv*oz
        return 


    cpdef mobilityRR(   self, double [:] o, double [:] r, double [:] T, int Nb=6, int Nm=6, double xi0=123456789):
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
        Nb: int 
            Number of periodic boxed summed 
            Default is 6
        Nm: int 
            Number of Fourier modes summed
            Default is 6
        """
        cdef: 
            double L = self.L,  xi=self.xi 
            double ixi2, ox, oy, oz, fac=8*PI/(L*L*L), muv=self.muv
            int N = self.N, N1 = -(Nm/2)+1, N2 =  (Nm/2)+1, i, i1, j, j1, ii, jj, kk,  Z=2*N
            double xdr, xdr2, A1, B1, Tdotik2, Tdotidr2, e1, erxdr, dx, dy, dz, idr,  kx, ky, kz, k2, cc
        if xi0 != 123456789:
            xi = xi0 
        ixi2 = 1/(xi*xi)

        
        for i in prange(N, nogil=True):
            ox=0; oy=0; oz=0
            for j in range(N):
                dx = r[i]  -r[j]
                dy = r[i+N]-r[j+N]
                dz = r[i+Z]-r[j+Z]
              
                for ii in range(2*Nb+1):
                    for jj in range(2*Nb+1):               
                        for kk in range(2*Nb+1):                 
                            if ii==jj==kk==Nb and i==j:
                                pass
                            else:    
                                dx = dx - Nb*L + ii*L 
                                dy = dy - Nb*L + jj*L 
                                dz = dz - Nb*L + kk*L
                                idr = 1.0/sqrt( dx*dx + dy*dy + dz*dz)
                                xdr = xi/idr; xdr2=xdr*xdr; 
                                erxdr   = erfc(xdr); e1=IPI*exp(-xdr2);  
                                Tdotidr2 = (T[j]*dx + T[j+N]*dy + T[j+Z]*dz )*idr*idr
                                A1    =  2*erxdr*idr*idr + e1*( 8*xdr2*xdr2*xdr*xi*xi - 40*xdr2*xdr*xi*xi +28*xdr*xi*xi + 2*xi*idr)
                                B1    = -6*erxdr*idr*idr + e1*(-8*xdr2*xdr2*xdr*xi*xi + 32*xdr2*xdr*xi*xi - 4*xdr*xi*xi - 6*xi*idr )
                              
                                ox += -(A1*T[j]  *idr + B1*Tdotidr2*dx)*idr;
                                oy += -(A1*T[j+N]*idr + B1*Tdotidr2*dy)*idr;
                                oz += -(A1*T[j+Z]*idr + B1*Tdotidr2*dz)*idr;
                #Fourier part
                for ii in range(N1, N2):
                    kx = (2*PI/L)*ii;
                    for jj in range(N1, N2):               
                        ky = (2*PI/L)*jj;
                        for kk in range(N1, N2):                 
                            kz = (2*PI/L)*kk;
                            if kx != 0 or ky != 0 or kz != 0:  
                                k2 = (kx*kx + ky*ky + kz*kz)    
                                cc = -fac*cos(kx*dx+ky*dy+kz*dz)*(1 + 0.25*k2*ixi2 + 0.125*ixi2*ixi2*k2*k2)*exp(-0.25*ixi2*k2)
                                Tdotik2  = (T[j]*kx + T[j+N]*ky + T[j+Z]*kz)/k2
                                
                                ox += cc*( T[j]   - Tdotik2*kx ) 
                                oy += cc*( T[j+N] - Tdotik2*ky ) 
                                oz += cc*( T[j+Z] - Tdotik2*kz ) 
            o[i]   += muv*ox
            o[i+N] += muv*oy
            o[i+Z] += muv*oz
        return

    
    cpdef propulsionR2s(self, double [:] o, double [:] r, double [:] V2s, int Nb=6, int Nm=6, double xi0=123456789):
        """
        Compute angular velocity due to 2s mode of the slip :math:`v=\pi^{R,2s}\cdot V^{2s}` 
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
        Nb: int 
            Number of periodic boxed summed 
            Default is 6
        Nm: int 
            Number of Fourier modes summed
            Default is 6
        """

        cdef: 
            double L = self.L,  xi=self.xi 
            double ixi2, ox, oy, oz, fac=8*PI/(L*L*L)
            int N = self.N, N1 = -(Nm/2)+1, N2 =  (Nm/2)+1, i, i1, j, j1, j2, ii, jj, kk, Z=2*N
            double dx, dy, dz, idr, idr5, sxx, syy, sxy, sxz, syz, srx, sry, srz, skx, sky, skz, s1
            double kx, ky, kz, k2, xdr, xdr2, cc, mus = (28.0*self.b**3)/24
        if xi0 != 123456789:
            xi = xi0 
        ixi2 = 1/(xi*xi)

 
        for i in prange(N, nogil=True):
            ox=0; oy=0; oz=0
            for j in range(N):
                sxx = V2s[j]
                syy = V2s[j+N]
                sxy = V2s[j+2*N]
                sxz = V2s[j+3*N]
                syz = V2s[j+4*N]

                for ii in range(2*Nb+1):
                    for jj in range(2*Nb+1):               
                        for kk in range(2*Nb+1):                 
                            if ii==Nb and jj==Nb and kk==Nb and i==j:
                                pass
                            else:    
                                dx = r[i]   - r[j]  -Nb*L + ii*L 
                                dy = r[i+N] - r[j+N]-Nb*L + jj*L 
                                dz = r[i+Z] - r[j+Z]-Nb*L + kk*L
                                idr = 1.0/sqrt( dx*dx + dy*dy + dz*dz )
                                idr5 = pow(idr, 5)
                                xdr = xi/idr; xdr2 = xdr*xdr;
                                s1 = -6*erfc(xdr) + IPI*(-16*xdr2*xdr + 32*xdr2*xdr2*xdr - 8*xdr2*xdr2*xdr2*xdr)*exp(-xdr2)
                                xdr = xi/idr; xdr2 = xdr*xdr; 
                                srx = sxx*dx +  sxy*dy + sxz*dz  
                                sry = sxy*dx +  syy*dy + syz*dz  
                                srz = sxz*dx +  syz*dy - (sxx+syy)*dz 
                                
                                ox += -s1*( dy*srz - dz*sry )*idr5
                                oy += -s1*( dz*srx - dx*srz )*idr5
                                oz += -s1*( dx*sry - dy*srx )*idr5
                #Fourier part
                dx = r[i]   - r[j]
                dy = r[i+N] - r[j+N]
                dz = r[i+Z] - r[j+Z] 
                for ii in range(N1, N2):
                    kx = (2*PI/L)*ii;
                    for jj in range(N1, N2):               
                        ky = (2*PI/L)*jj;
                        for kk in range(N1, N2):                 
                            kz = (2*PI/L)*kk;
                            if kx != 0 or ky != 0 or kz != 0:  
                                k2 = (kx*kx + ky*ky + kz*kz)    
                                cc = fac*(1 + 0.25*k2*ixi2 + 0.125*ixi2*ixi2*k2*k2)*exp(-0.25*ixi2*k2)/(k2)
                                skx = sxx*kx +  sxy*ky + sxz*kz  
                                sky = sxy*kx +  syy*ky + syz*kz  
                                skz = sxz*kx +  syz*ky - (sxx+syy)*kz 
                                
                                ox += cc*( ky*skz - kz*sky) 
                                oy += cc*( kz*skx - kx*skz) 
                                oz += cc*( kx*sky - ky*skx) 
            o[i]    += mus*ox
            o[i+N] += mus*oy
            o[i+Z]+= mus*oz
        pass


    cpdef propulsionR3s(  self, double [:] o, double [:] r, double [:] V3s, int Nb=6, int Nm=6, double xi0=123456789):
        """
        Compute angular velocity due to 3s mode of the slip :math:`v=\pi^{R,3s}\cdot V^{3s}` 
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
        Nb: int 
            Number of periodic boxed summed 
            Default is 6
        Nm: int 
            Number of Fourier modes summed
            Default is 6
        """

        cdef: 
            double L = self.L,  xi=self.xi, 
            double ixi2, fac=8*PI/(L*L*L)
            int N = self.N, N1 = -(Nm/2)+1, N2 =  (Nm/2)+1, i, i1, j, j1, j2, ii, jj, kk, Z=2*N
            double dx, dy, dz, idr, idr7, grrx, grry, grrz, gkkx, gkky, gkkz, gxxx, gyyy, gxxy, gxxz, gxyy, gxyz, gyyz, 
            double s2, kx, ky, kz, k2, xdr, e1, xdr2, xdr3, xdr5, cc,
        if xi0 != 123456789:
            xi = xi0 
        ixi2 = 1/(xi*xi)


        for i in prange(N, nogil=True):
            for j in range(N):
                gxxx = V3s[j]
                gyyy = V3s[j+N]
                gxxy = V3s[j+2*N]
                gxxz = V3s[j+3*N]
                gxyy = V3s[j+4*N]
                gxyz = V3s[j+5*N]
                gyyz = V3s[j+6*N]
                for ii in range(2*Nb+1):
                    for jj in range(2*Nb+1):               
                        for kk in range(2*Nb+1):                 
                            if ii==Nb and jj==Nb and kk==Nb and i==j:
                                pass
                            else:    
                                dx = r[i]   - r[j]  -Nb*L + ii*L 
                                dy = r[i+N] - r[j+N]-Nb*L + jj*L 
                                dz = r[i+Z] - r[j+Z]-Nb*L + kk*L
                                idr = 1.0/sqrt( dx*dx + dy*dy + dz*dz )
                                idr7 = pow(idr, 7)
                                xdr = xi/idr; xdr2=xdr*xdr; xdr3=xdr2*xdr;   xdr5=xdr3*xdr3;
                                s2 = 30*erfc(xdr) + IPI*(6*xdr + 32*xdr3 - 32*xdr5 - 80*xdr5*xdr2 + 16*xdr5*xdr3*xdr)*exp(-xdr2)

                                grrx = gxxx*(dx*dx-dz*dz) + gxyy*(dy*dy-dz*dz) +  2*gxxy*dx*dy + 2*gxxz*dx*dz  +  2*gxyz*dy*dz
                                grry = gxxy*(dx*dx-dz*dz) + gyyy*(dy*dy-dz*dz) +  2*gxyy*dx*dy + 2*gxyz*dx*dz  +  2*gyyz*dy*dz
                                grrz = gxxz*(dx*dx-dz*dz) + gyyz*(dy*dy-dz*dz) +  2*gxyz*dx*dy - 2*(gxxx+gxyy)*dx*dz - 2*(gxxy+gyyy)*dy*dz
                                
                                o[i]   -= s2*( dy*grrz - dz*grry )*idr7
                                o[i+N] -= s2*( dz*grrx - dx*grrz )*idr7
                                o[i+Z] -= s2*( dx*grry - dy*grrx )*idr7
                #Fourier part
                dx = r[i]   - r[j]     
                dy = r[i+N] - r[j+N]  
                dz = r[i+Z] - r[j+Z]
                for ii in range(N1, N2):
                    kx = (2*PI/L)*ii;
                    for jj in range(N1, N2):               
                        ky = (2*PI/L)*jj;
                        for kk in range(N1, N2):                 
                            kz = (2*PI/L)*kk;
                            if kx != 0 or ky != 0 or kz != 0:  
                                k2 = (kx*kx + ky*ky + kz*kz)     
                                cc = 8*PI*sin( kx*dx+ky*dy+kz*dz )*(1 + 0.25*k2*ixi2 + 0.125*ixi2*ixi2*k2*k2)*exp(-0.25*ixi2*k2)/(k2*L*L*L)
                                gkkx = gxxx*(kx*kx-kz*kz) + gxyy*(ky*ky-kz*kz) +  2*gxxy*kx*ky + 2*gxxz*kx*kz  +  2*gxyz*ky*kz
                                gkky = gxxy*(kx*kx-kz*kz) + gyyy*(ky*ky-kz*kz) +  2*gxyy*kx*ky + 2*gxyz*kx*kz  +  2*gyyz*ky*kz
                                gkkz = gxxz*(kx*kx-kz*kz) + gyyz*(ky*ky-kz*kz) +  2*gxyz*kx*ky - 2*(gxxx+gxyy)*kx*kz-2*(gxxy+gyyy)*ky*kz
                                
                                o[i]   += cc*( ky*gkkz - kz*gkky) 
                                o[i+N] += cc*( kz*gkkx - kx*gkkz) 
                                o[i+Z] += cc*( kx*gkky - ky*gkkx) 
                            else:    
                                pass
        return


    cpdef propulsionR3a(  self, double [:] o, double [:] r, double [:] V3a, int Nb=6, int Nm=6, double xi0=123456789):
        """
        Compute angular velocity due to 3a mode of the slip :math:`v=\pi^{R,3a}\cdot V^{3a}` 
        ...

        Parameters
        ----------
        o: np.array
            An array of angular velocities
            An array of size 3*N,
        r: np.array
            An array of positions
            An array of size 3*N,
        V3a: np.array
            An array of 3a mode of the slip
            An array of size 5*N,
        Nb: int 
            Number of periodic boxed summed 
            Default is 6
        Nm: int 
            Number of Fourier modes summed
            Default is 6
        """

        cdef: 
            double L = self.L,  xi=self.xi   
            double ixi2, fac=8*PI/(L*L*L)
            int N = self.N, N1 = -(Nm/2)+1, N2 =  (Nm/2)+1, i, i1, j, j1, j2, ii, jj, kk, Z=2*N
            double dx, dy, dz, idr, idr2, idr5, vxx, vyy, vxy, vxz, vyz, vrr, vrx, vry, vrz, vkx, vky, vkz, vkk
            double kx, ky, kz, k2,  xdr, e1, xdr2, cc,  s1, s3
        if xi0 != 123456789:
            xi = xi0 
        ixi2 = 1/(xi*xi)

 
        for i in prange(N, nogil=True):
            for j in range(N):
                vxx = V3a[j]
                vyy = V3a[j+N]
                vxy = V3a[j+2*N]
                vxz = V3a[j+3*N]
                vyz = V3a[j+4*N]
                for ii in range(2*Nb+1):
                    for jj in range(2*Nb+1):               
                        for kk in range(2*Nb+1):                 
                            if ii==jj==kk==Nb and i==j:
                                pass
                            else:    
                                dx = r[i]   - r[j]  -Nb*L + ii*L 
                                dy = r[i+N] - r[j+N]-Nb*L + jj*L 
                                dz = r[i+Z] - r[j+Z]-Nb*L + kk*L
                                idr = 1.0/sqrt( dx*dx + dy*dy + dz*dz )
                                
                                xdr = xi/idr; xdr2 = xdr*xdr; e1 = IPI*exp(-xdr2);
                                s1 = -6*erfc(xdr) + e1*(-16*xdr2*xdr + 32*xdr2*xdr2*xdr - 8*xdr2*xdr2*xdr2*xdr)
                                s3 = e1*xdr*(6-48*xdr2+192*xdr2*xdr2-120*xdr2*xdr2*xdr2+16*xdr2*xdr2*xdr2*xdr2)
                                idr5 = idr*idr*idr*idr*idr      
                                vrr = (vxx*(dx*dx-dz*dz) + vyy*(dy*dy-dz*dz) +  2*vxy*dx*dy + 2*vxz*dx*dz  +  2*vyz*dy*dz)*idr*idr
                                vrx = vxx*dx +  vxy*dy + vxz*dz  
                                vry = vxy*dx +  vyy*dy + vyz*dz  
                                vrz = vxz*dx +  vyz*dy - (vxx+vyy)*dz 

                                o[i]   +=  ( (6*s1+s3)*vrx- (5*s1-s3)*vrr*dx )*idr5
                                o[i+N]+=  ( (6*s1+s3)*vry- (5*s1-s3)*vrr*dy )*idr5
                                o[i+Z] +=  ( (6*s1+s3)*vrz- (5*s1-s3)*vrr*dz )*idr5
                #Fourier part
                dx = r[i]   - r[j]
                dy = r[i+N] - r[j+N]
                dz = r[i+Z] - r[j+Z] 
                for ii in range(N1, N2):
                    kx = (2*PI/L)*ii;
                    for jj in range(N1, N2):               
                        ky = (2*PI/L)*jj;
                        for kk in range(N1, N2):                 
                            kz = (2*PI/L)*kk;
                            if kx != 0 or ky != 0 or kz != 0:  
                                k2 = (kx*kx + ky*ky + kz*kz)    
                                cc = 8*PI*(1 + 0.25*k2*ixi2 + 0.125*ixi2*ixi2*k2*k2)*exp(-0.25*ixi2*k2)/(k2*L*L*L)
                                vkk = (vxx*(kx*kx-kz*kz) + vyy*(ky*ky-kz*kz) +  2*vxy*kx*ky + 2*vxz*kx*kz  +  2*vyz*ky*kz)
                                vkx = vxx*kx +  vxy*ky + vxz*kz  
                                vky = vxy*kx +  vyy*ky + vyz*kz  
                                vkz = vxz*kx +  vyz*ky - (vxx+vyy)*kz 
                                
                                o[i]   += cc*(vkx*k2 - vkk*kx) 
                                o[i+N] += cc*(vkx*k2 - vkk*kx) 
                                o[i+Z] += cc*(vkx*k2 - vkk*kx) 
        return


    cpdef propulsionR4a(  self, double [:] o, double [:] r, double [:] V4a, int Nb=6, int Nm=6, double xi0=123456789):
        """
        Compute angular velocity due to 4a mode of the slip :math:`v=\pi^{R,4a}\cdot V^{4a}` 
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
        Nb: int 
            Number of periodic boxed summed 
            Default is 6
        Nm: int 
            Number of Fourier modes summed
            Default is 6
        """

        cdef: 
            double L = self.L,  xi=self.xi
            double ixi2, ox, oy, oz, fac=8*PI/(L*L*L)
            int N = self.N, N1 = -(Nm/2)+1, N2 =  (Nm/2)+1, i, i1, j, j1, j2, ii, jj, kk, Z=2*N
            double dx, dy, dz, idr, idr2, idr7
            double mrrr, mkkk, mrrx, mrry, mrrz, mkkx, mkky, mkkz, mxxx, myyy, mxxy, mxxz, mxyy, mxyz, myyz, 
            double kx, ky, kz, k2, xdr, e1, xdr2, xdr4, cc, s2, s4
        if xi0 != 123456789:
            xi = xi0 
        ixi2 = 1/(xi*xi)

 
        for i in prange(N, nogil=True):
            ox=0; oy=0; oz=0;
            for j in range(N):
                mxxx = V4a[j]
                myyy = V4a[j+N  ]
                mxxy = V4a[j+2*N]
                mxxz = V4a[j+3*N]
                mxyy = V4a[j+4*N]
                mxyz = V4a[j+5*N]
                myyz = V4a[j+6*N]
                for ii in range(2*Nb+1):
                    for jj in range(2*Nb+1):               
                        for kk in range(2*Nb+1):                 
                            if ii==Nb and jj==Nb and kk==Nb and i==j:
                                pass
                            else:    
                                dx = r[i]   - r[j]     -Nb*L + ii*L 
                                dy = r[i+N] - r[j+N]  -Nb*L + jj*L 
                                dz = r[i+Z] - r[j+Z]-Nb*L + kk*L
                                idr = 1.0/sqrt( dx*dx + dy*dy + dz*dz )
                                idr2 = idr*idr
                                idr7 = idr2*idr2*idr2*idr
                                xdr = xi/idr; xdr2 = xdr*xdr; xdr4 = xdr2*xdr2; e1 = IPI*exp(-xdr2);
                                s2 = 30*erfc(xdr) + e1*xdr*(6 + 32*xdr2 - 32*xdr4 - 80*xdr4*xdr2 + 16*xdr4*xdr4)

                                s4  =e1*xdr*(-24 + 84*xdr2 + 160*xdr4 - 336*xdr4*xdr2 + 304*xdr4*xdr4 -32*xdr4*xdr4*xdr2)
                                mrrr = mxxx*dx*(dx*dx-3*dz*dz) + 3*mxxy*dy*(dx*dx-dz*dz) + mxxz*dz*(3*dx*dx-dz*dz) +\
                                   3*mxyy*dx*(dy*dy-dz*dz) + 6*mxyz*dx*dy*dz + myyy*dy*(dy*dy-3*dz*dz) +  myyz*dz*(3*dy*dy-dz*dz) 
                                mrrx = mxxx*(dx*dx-dz*dz) + mxyy*(dy*dy-dz*dz) +  2*mxxy*dx*dy + 2*mxxz*dx*dz  +  2*mxyz*dy*dz
                                mrry = mxxy*(dx*dx-dz*dz) + myyy*(dy*dy-dz*dz) +  2*mxyy*dx*dy + 2*mxyz*dx*dz  +  2*myyz*dy*dz
                                mrrz = mxxz*(dx*dx-dz*dz) + myyz*(dy*dy-dz*dz) +  2*mxyz*dx*dy - 2*(mxxx+mxyy)*dx*dz - 2*(mxxy+myyy)*dy*dz
                                
                                ox += (3*s2-s4)*mrrx*idr7 - (7*s2+s4)*mrrr*dx*idr7*idr2
                                oy += (3*s2-s4)*mrry*idr7 - (7*s2+s4)*mrrr*dy*idr7*idr2
                                oz += (3*s2-s4)*mrrz*idr7 - (7*s2+s4)*mrrr*dz*idr7*idr2
                #Fourier part
                dx = r[i]  - r[j]     
                dy = r[i+N]- r[j+N]  
                dz = r[i+Z]- r[j+Z]
                for ii in range(N1, N2):
                    kx = (2*PI/L)*ii;
                    for jj in range(N1, N2):               
                        ky = (2*PI/L)*jj;
                        for kk in range(N1, N2):                 
                            kz = (2*PI/L)*kk;
                            if kx != 0 or ky != 0 or kz != 0:  
                                k2 = (kx*kx + ky*ky + kz*kz)     
                                cc = 8*PI*cos( kx*dx+ky*dy+kz*dz )*(1 + 0.25*k2*ixi2 + 0.125*ixi2*ixi2*k2*k2)*exp(-0.25*ixi2*k2)/(k2*L*L*L)
                                mkkk = mxxx*kx*(kx*kx-3*kz*kz) + 3*mxxy*ky*(kx*kx-kz*kz) + mxxz*kz*(3*kx*kx-kz*kz) +\
                                   3*mxyy*kx*(ky*ky-kz*kz) + 6*mxyz*kx*ky*kz + myyy*ky*(ky*ky-3*kz*kz) +  myyz*kz*(3*ky*ky-kz*kz) 
                                mkkx = mxxx*(kx*kx-kz*kz) + mxyy*(ky*ky-kz*kz) +  2*mxxy*kx*ky + 2*mxxz*kx*kz  +  2*mxyz*ky*kz
                                mkky = mxxy*(kx*kx-kz*kz) + myyy*(ky*ky-kz*kz) +  2*mxyy*kx*ky + 2*mxyz*kx*kz  +  2*myyz*ky*kz
                                mkkz = mxxz*(kx*kx-kz*kz) + myyz*(ky*ky-kz*kz) +  2*mxyz*kx*ky - 2*(mxxx+mxyy)*kx*kz-2*(mxxy+myyy)*ky*kz
                               
                                ox += cc*( mkkx*k2 - mkkk*kx) 
                                oy += cc*( mkky*k2 - mkkk*ky) 
                                oz += cc*( mkkz*k2 - mkkk*kz) 
            o[i]   += ox
            o[i+N] += oy
            o[i+Z] += oz
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
    boxSize: int 
        Box size

    """
    def __init__(self, radius=1, particles=1, viscosity=1, gridpoints=32, boxSize=10, xi=123456789):
        self.b  = radius
        self.N = particles
        self.Nt = gridpoints
        self.eta= viscosity
        self.L  = boxSize

        if xi==123456789:
            self.xi = sqrt(PI)/boxSize 
            #Nijboer and De Wette have shown that \pi^{1/2}/V^{1/3} is a good choice for cubic lattices 
        else:
            self.xi = xi 

    cpdef flowField1s(self, double [:] vv, double [:] rt, double [:] r, double [:] F, int Nb=6, int Nm=6, double xi0=123456789):
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
        Nb: int 
            Number of periodic boxed summed 
            Default is 6
        Nm: int 
            Number of Fourier modes summed
            Default is 6
    
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
        >>> flow = pystokes.periodic.Flow(radius=b, particles=N, viscosity=eta, gridpoints=Ng*Ng,
        >>>         boxSize=L)
        >>> 
        >>> # create grid, evaluate flow and plot
        >>> rr, vv = pystokes.utils.gridXY(dim, L, Ng)
        >>> flow.flowField1s(vv, rr, r, F1s)
        >>> pystokes.utils.plotStreamlinesXY(vv, rr, r, offset=6-1, density=1.4, title='1s')
        """
        cdef int N=self.N, Nt=self.Nt, N1=-(Nm/2)+1, N2=(Nm/2)+1, i, j, ii, jj, kk, Z=2*N, Nbb=2*Nb+1
        cdef double L=self.L,  xi=self.xi, ixi2, mu=1.0/(6*PI*self.eta*self.b), muv=mu*self.b*0.75, siz=Nb*L
        cdef double a2=0*self.b*self.b/6, k0=2*PI/L, fac=8*PI/(L*L*L), mt= IPI*xi*self.b*(-3+20*xi*xi*self.b*self.b/3.0), mpp=mu*(1+mt)   # include M^2(r=0)
        cdef double xdr, xdr2, xdr3, A, B, A1, B1, fdotir, e1, erxdr, m20, xd1, yd1, zd1
        cdef double xd, yd, zd, dx, dy, dz, idr, kx, ky, kz, k2, ik2, cc, fdotik, vx, vy, vz, fx, fy, fz
        if xi0 != 123456789:
            xi = xi0 
        ixi2 = 1/(xi*xi)

        
        for i in prange(Nt, nogil=True):
            vx=0;  vy=0;  vz=0;
            for j in range(N):
                xd=rt[i]    -r[j];          xd1=xd-siz; 
                yd=rt[i+Nt]  -r[j+N];    yd1=yd-siz;  
                zd=rt[i+2*Nt]-r[j+Z];    zd1=zd-siz;
                fx=F[j];  fy=F[j+N];  fz=F[j+Z];

                for ii in range(Nbb):
                    dx = xd1 + ii*L 
                    for jj in range(Nbb):               
                        dy = yd1 + jj*L 
                        for kk in range(Nbb):                 
                            dz = zd1 + kk*L
                            idr = 1.0/sqrt( dx*dx + dy*dy + dz*dz)
                            xdr=xi/idr;  xdr2=xdr*xdr;  xdr3=xdr2*xdr;  
                            erxdr = erfc(xdr);  e1=IPI*exp(-xdr2);
                            fdotir = (fx*dx + fy*dy + fz*dz)*idr*idr
                            A = erxdr + e1*(2*xdr3-3*xdr)
                            B = erxdr + e1*(xdr - 2*xdr3)
                            A += (2*erxdr  + e1*( 2*xdr + 28*xdr3 - 40*xdr3*xdr2 + 8*xdr3*xdr3*xdr ))*idr*idr*a2 # finite size correction
                            B += (-6*erxdr + e1*(-6*xdr - 4*xdr3  + 32*xdr3*xdr2 - 8*xdr3*xdr3*xdr ))*idr*idr*a2  #finite size 
                            vx += ( A*fx + B*fdotir*dx)*idr
                            vy += ( A*fy + B*fdotir*dy)*idr
                            vz += ( A*fz + B*fdotir*dz)*idr
                # Fourier space sum
                for ii in range(N1, N2):
                    kx = k0*ii;
                    for jj in range(N1, N2):               
                        ky = k0*jj;
                        for kk in range(N1, N2):                 
                            kz = k0*kk;
                            if kx != 0 or ky != 0 or kz != 0:
                                k2 = (kx*kx + ky*ky + kz*kz); ik2=1/k2
                                fdotik = (fx*kx + fy*ky + fz*kz )*ik2
                                cc = fac*(1-a2*k2)*cos( kx*xd+ky*yd+kz*zd )*(1 + 0.25*k2*ixi2 + 0.125*ixi2*ixi2*k2*k2)*exp(-0.25*ixi2*k2)*ik2

                                vx += cc*(fx - fdotik*kx) 
                                vy += cc*(fy - fdotik*ky) 
                                vz += cc*(fz - fdotik*kz) 
        
            vv[i]      += muv*vx 
            vv[i+Nt]   += muv*vy 
            vv[i+2*Nt] += muv*vz 
        return 
   

    cpdef flowField2s(self, double [:] vv, double [:] rt, double [:] r, double [:] V2s, int Nb=6, int Nm=6, double xi0=123456789):
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
        Nb: int 
            Number of periodic boxed summed 
            Default is 6
        Nm: int 
            Number of Fourier modes summed
            Default is 6
        """

        cdef: 
            int N=self.N,Nt=self.Nt, N1=-(Nm/2)+1, N2=(Nm/2)+1, i, j, ii, jj, kk, Z=2*N, Nbb=2*Nb+1
            double L = self.L,  xi=self.xi, siz=Nb*L, ixi2
            double xdr, xdr2, xdr3, xdr5,  D, E, erxdr, e1, sxx, syy, sxy, sxz, syz, srr, srx, sry, srz
            double dx, dy, dz, idr, idr3, kx, ky, kz, k2, cc, kdotr, vx, vy, vz, k0=2*PI/L, ixk2, fac=8*PI/(L*L*L)
            double a2 = self.b*self.b*4.0/15,aidr2, xd1, yd1, zd1, xd, yd, zd, mus = (28.0*self.b**3)/24, ik2
        if xi0 != 123456789:
            xi = xi0 
        ixi2 = 1/(xi*xi)

        
        for i in prange(Nt, nogil=True):
            vx=0; vy=0; vz=0;
            for j in range(N):
                sxx = V2s[j]
                syy = V2s[j+N]
                sxy = V2s[j+2*N]
                sxz = V2s[j+3*N]
                syz = V2s[j+4*N]
                xd=rt[i]-r[j];          xd1=xd-siz; 
                yd=rt[i+Nt]-r[j+N];    yd1=yd-siz;  
                zd=rt[i+2*Nt]-r[j+Z];    zd1=zd-siz;
                
                for ii in range(Nbb):
                    dx = xd1 + ii*L 
                    for jj in range(Nbb):               
                        dy = yd1 + jj*L 
                        for kk in range(Nbb):                 
                            dz = zd1 + kk*L
                            idr = 1.0/sqrt( dx*dx + dy*dy + dz*dz)
                            idr3 = idr*idr*idr; aidr2=a2*idr*idr
                            xdr = xi/idr; xdr2=xdr*xdr; xdr3 = xdr2*xdr; xdr5 = xdr3*xdr2;
                            erxdr   = erfc(xdr);   e1  = IPI*exp(-xdr2);
                            D =  e1*(8*xdr3 - 4*xdr5 ) 
                            E = -3*erxdr + e1*(-3*xdr - 2*xdr3  + 4*xdr5 ) 
                            
                            srx = sxx*dx + sxy*dy + sxz*dz  
                            sry = sxy*dx + syy*dy + syz*dz  
                            srz = sxz*dx + syz*dy - (sxx+syy)*dz 
                            srr = srx*dx + sry*dy + srz*dz

                            # D += -12*erxdr+ e1*(-12*xdr - 8*xdr3  - 104*xdr5 + 104*xdr5*xdr2 - 16*xdr3*xdr3*xdr3)*aidr2
                            # E += 30*erxdr + e1*(30*xdr  + 20*xdr3 + 8*xdr5   - 80*xdr5*xdr2  + 16*xdr3*xdr3*xdr3)*aidr2
                            D  = D*idr3
                            E  = E*srr*idr*idr*idr3

                            vx += D*srx + E*dx
                            vy += D*sry + E*dy
                            vz += D*srz + E*dz
                #Fourier part
                for ii in range(N1, N2):
                    kx = k0*ii;
                    for jj in range(N1, N2):               
                        ky = k0*jj;
                        for kk in range(N1, N2):                 
                            kz = k0*kk;
                            if kx != 0 or ky != 0 or kz != 0:  
                                k2 = (kx*kx + ky*ky + kz*kz); ik2=1.0/k2    
                                ixk2 = 0.25*k2*ixi2
                                cc = -fac*(1-a2*k2)*sin(kx*xd+ky*yd+kz*zd)*(1+ixk2+2*ixk2*ixk2)*exp(-ixk2)*k2

                                srx = sxx*kx + sxy*ky + sxz*kz  
                                sry = sxy*kx + syy*ky + syz*kz  
                                srz = sxz*kx + syz*ky - (sxx+syy)*kz 
                                srr = (srx*kx + sry*ky + srz*kz)*ik2
                                
                                vx += cc* (srx - srr*kx) 
                                vy += cc* (sry - srr*ky)
                                vz += cc* (srz - srr*kz)
            vv[i]      += mus*vx
            vv[i+Nt]   += mus*vy
            vv[i+2*Nt] += mus*vz
        return 
    
    
    cpdef flowField3t(self, double [:] vv, double [:] rt, double [:] r, double [:] V3t, int Nb=16, int Nm=16, double xi0=123456789):
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
        Nb: int 
            Number of periodic boxed summed 
            Default is 6
        Nm: int 
            Number of Fourier modes summed
            Default is 6
        """
 
        cdef: 
            double L = self.L,  xi=self.xi, siz=Nb*L, k0=(2*PI/L), fac=8*PI/(L*L*L)
            double ixi2, vx, vy, vz
            int Nt=self.Nt,N = self.N, N1 = -(Nm/2)+1, N2 =  (Nm/2)+1, i, i1, j, j1, ii, jj, kk, Z=2*N, Nbb=2*Nb+1
            double xdr, xdr2, xdr3, A1, B1, V3tdotik2, Ddotidr2, e1, erxdr, dx, dy, dz, idr, idr5,  kx, ky, kz, k2, cc
            double mud1 = -1.0*(self.b**3)/5
            double xd, yd, zd, xd1, yd1, zd1
        if xi0 != 123456789:
            xi = xi0 
        ixi2 = 1/(xi*xi)

        
        for i in prange(Nt, nogil=True):
            vx=0;  vy=0; vz=0;
            for j in range(N):
                xd=rt[i]-r[j];          xd1=xd-siz; 
                yd=rt[i+Nt]-r[j+N];    yd1=yd-siz;  
                zd=rt[i+2*Nt]-r[j+Z];    zd1=zd-siz;
                
                for ii in range(Nbb):
                    dx = xd1 + ii*L 
                    for jj in range(Nbb):               
                        dy = yd1 + jj*L 
                        for kk in range(Nbb):                 
                            dz = zd1 + kk*L
                            idr = 1.0/sqrt( dx*dx + dy*dy + dz*dz)
                            idr5=idr*idr*idr*idr*idr
                            xdr = xi/idr; xdr2=xdr*xdr; xdr3 = xdr2*xdr;
                            erxdr   = erfc(xdr); e1=IPI*exp(-xdr2);  
                            A1 = (2*erxdr  + e1*( 2*xdr+28*xdr3-40*xdr3*xdr2+8*xdr3*xdr3*xdr ))*idr5 
                            B1 = (-6*erxdr + e1*(-6*xdr-4*xdr3 +32*xdr3*xdr2-8*xdr3*xdr3*xdr ))*idr5 
                            Ddotidr2 = (V3t[j]*dx + V3t[j+N]*dy + V3t[j+Z]*dz )*idr*idr
                          
                            vx += A1*V3t[j]    + B1*Ddotidr2*dx
                            vy += A1*V3t[j+N] + B1*Ddotidr2*dy
                            vz += A1*V3t[j+Z] + B1*Ddotidr2*dz
                #Fourier part
                for ii in range(N1, N2):
                    kx = k0*ii;
                    for jj in range(N1, N2):               
                        ky = k0*jj;
                        for kk in range(N1, N2):                 
                            kz = k0*kk;
                            if kx != 0 or ky != 0 or kz != 0:  
                                k2 = (kx*kx + ky*ky + kz*kz)    
                                cc = -fac*cos(kx*xd+ky*yd+kz*zd)*(1 + 0.25*k2*ixi2 + 0.125*ixi2*ixi2*k2*k2)*exp(-0.25*ixi2*k2)
                                V3tdotik2  = (V3t[j]*kx + V3t[j+N]*ky + V3t[j+Z]*kz)/k2
                                
                                vx += cc*( V3t[j]   - V3tdotik2*kx ) 
                                vy += cc*( V3t[j+N] - V3tdotik2*ky ) 
                                vz += cc*( V3t[j+Z] - V3tdotik2*kz ) 
            vv[i]      += mud1*vx
            vv[i+Nt]   += mud1*vy
            vv[i+2*Nt] += mud1*vz
        return
