cimport cython
from libc.math cimport sqrt, pow, exp
from cython.parallel import prange


@cython.boundscheck(False)
@cython.cdivision(True)
@cython.nonecheck(False)
@cython.wraparound(False)
cdef class Forces:
    """
    Computes forces in a system of colloidal particles

    Methods in the Forces class take input,
        * arrays of positions, forces 
        * parameters for a given potential
    
    The array of forces is then update by each method. 

    ...

    Parameters
    ----------
    particles: int
        Number of particles (N)


    """
    def __init__(self, particles=1):
        self.N = particles

        
    cpdef VdW(self, double [:] F, double [:] r, double A=0, double a0=0):
        """
        generic van der Waals attraction to a wall at z=0 with Hamaker constant a
        """
        cdef int N = self.N, i, j, Z= 2*N
        cdef double fz, iz, iz2
        
        for i in prange(N, nogil=True):
            iz  = 1./r[i+Z]
            iz2 = iz*iz
            fz = -1./6*a0*A*iz2
            F[i+Z] += fz
        return

    
    cpdef dlvo(self, double [:] F, double [:] r, double B=1., double kap=0.1, double A=1.):
        """
        generic DLVO interaction used for example in thesis
        """
        cdef int N = self.N, i, j, Z= 2*N
        cdef double dx, dy, dz, dr, idr, idr3, fx, fy, fz, fac, facinv, dlvo_fac

        for i in prange(N,nogil=True):
            fx = 0.0; fy = 0.0; fz = 0.0;
            for j in range(N):
                dx = r[i   ] - r[j   ]
                dy = r[i+N] - r[j+N]
                dz = r[i+Z] - r[j+Z]
                dr = sqrt(dx*dx + dy*dy + dz*dz)
                if i != j:
                    idr  = 1.0/dr
                    idr3 = idr*idr*idr
                    fac = 1.0 + exp(kap*dr)
                    facinv = 1.0/fac
                    
                    dlvo_fac = kap*B*idr*facinv - A*idr3

                    fx += dlvo_fac*dx
                    fy += dlvo_fac*dy
                    fz += dlvo_fac*dz
                    
            F[i]   += fx
            F[i+N] += fy
            F[i+Z] += fz
        return
        
        

    cpdef lennardJones(self, double [:] F, double [:] r, double lje=0.01, double ljr=3):
        """
        The standard Lennard-Jones potential truncated at the minimum (aslo called WCA potential)
        
            We choose \phi(r) = lje/12 (rr^12 - 2*rr^6 ) + lje/12,  as the standard WCA potential.
            ljr: minimum of the LJ potential and rr=ljr/r.

        ...

        Parameters
        ----------
        r: np.array
            An array of positions
            An array of size 3*N,
        F: np.array
            An array of forces
            An array of size 3*N,
        lje: float
            Strength of the LJ 
        ljr: float
            Range of the LJ 


        """

        cdef int N = self.N, i, j, Z=2*N
        cdef double dx, dy, dz, dr2, idr, rminbyr, fac, fx, fy, fz

        for i in prange(N,nogil=True):
            fx = 0.0; fy = 0.0; fz = 0.0;
            for j in range(N):
                dx = r[i   ] - r[j   ]
                dy = r[i+N] - r[j+N]
                dz = r[i+Z] - r[j+Z]
                dr2 = dx*dx + dy*dy + dz*dz
                if i != j and dr2 < (ljr*ljr):
                    idr     = 1.0/sqrt(dr2)
                    rminbyr = ljr*idr
                    fac   = lje*(pow(rminbyr, 12) - pow(rminbyr, 6))*idr*idr

                    fx += fac*dx
                    fy += fac*dy
                    fz += fac*dz
                    
            F[i]   += fx
            F[i+N] += fy
            F[i+Z] += fz
        return


    cpdef lennardJonesWall(self, double [:] F, double [:] r, double lje=0.0100, double ljr=3, double wlje=0.01, double wljr=3.0):
        """
        The standard Lennard-Jones potential truncated at the minimum (aslo called WCA potential)
        
            We choose \phi(r) = lje/12 (rr^12 - 2*rr^6 ) + lje/12,  as the standard WCA potential.
            ljr: minimum of the LJ potential and rr=ljr/r.

        ...

        Parameters
        ----------
        r: np.array
            An array of positions
            An array of size 3*N,
        F: np.array
            An array of forces
            An array of size 3*N,
        lje: float
            Strength of the LJ 
        ljr: float
            Range of the LJ 


        """

        cdef int N=self.N, i, j, Z=2*N
        cdef double dx, dy, dz, dr, idr, rminbyr, fac, fx, fy, fz, hh

        for i in prange(N, nogil=True):
            fx = 0.0; fy = 0.0; fz = 0.0;
            hh = r[i+Z]
            if hh<wljr:
                idr = 1/hh
                rminbyr = wljr*idr
                fac   = wlje*(pow(rminbyr, 12) - pow(rminbyr, 6))*idr
                fz += fac       ##LJ from the wall

            for j in range(N):
                dx = r[i   ] - r[j   ]
                dy = r[i+N] - r[j+N]
                dz = r[i+Z] - r[j+Z]
                dr = sqrt(dx*dx + dy*dy + dz*dz)
                if i != j and dr < ljr:
                    idr     = 1.0/dr
                    rminbyr = ljr*idr
                    fac   = lje*(pow(rminbyr, 12) - pow(rminbyr, 6))*idr*idr

                    fx += fac*dx
                    fy += fac*dy
                    fz += fac*dz

            F[i]   += fx
            F[i+N] += fy
            F[i+Z] += fz
        return

    
    cpdef softSpringWall(self, double [:] F, double [:] r, double pk=0.0100, double prmin=3, double prmax=4,
                         double wlje= 0.001, double wljr = 1.5):
        '''
        lj potential fron wall to particles and spring between particles
        F = -k(r-rmin)
        
        ...
        
        Parameters
        ----------
        r: np.array
            An array of positions
            An array of size 3*N,
        F: np.array
            An array of forces
            An array of size 3*N,
        pk: float
            Strength of harmonic potential between particles
        prmin: float
            Minimum of harmonic potential
        prmax: float
            Cutoff distance of harmonic potential
        
        wlje: float
            Strength of the LJ from wall
        wljr: float
            Range of the LJ from wall

        '''
        cdef int N=self.N, i, j, Z=2*N
        cdef double dx, dy, dz, dr, idr, rminbyr, fac, fx, fy, fz, hh, facss,abshh,sgnhh

        for i in prange(N, nogil=True):
            fx = 0.0; fy = 0.0; fz = 0.0;
            hh = r[i+Z]
            abshh=sqrt(hh*hh)
            sgnhh=hh/abshh
            idr = 1/hh
            if abshh<wljr:
                  rminbyr = 1.5*idr
                  fac   = 0.1*(pow(rminbyr, 12) - pow(rminbyr, 6))*idr
                  fz+=fac
                # rminbyr = wljr*idr
                # fac   = wlje*(pow(rminbyr, 12) - pow(rminbyr, 6))*idr
                # fz += fac      ##LJ from the wall

            for j in range(N):
                dx = r[i  ] - r[j   ]
                dy = r[i+N] - r[j+N]
                dz = r[i+Z] - r[j+Z]
                dr = sqrt(dx*dx + dy*dy + dz*dz)
                if i != j and dr < prmax:
                    idr     = 1.0/dr
                    fac   = pk*(prmin-dr)

                    fx += fac*dx*idr
                    fy += fac*dy*idr
                    fz += fac*dz*idr

            F[i]   += fx
            F[i+N] += fy
            F[i+Z] += fz
        return

        
    cpdef softSpringLJWall(self, double [:] F, double [:] r, double pk=0.0100, double prmin=3, double prmax=4,
                         double lje = 0.001, double ljr = 3, double wlje= 0.001, double wljr = 1.5):
        '''
        lj potential fron wall to particles and spring and lj between particles
        F = -k(r-rmin)
        lj stabilises numerical solver when particles get close
        
        ...
        
        Parameters
        ----------
        r: np.array
            An array of positions
            An array of size 3*N,
        F: np.array
            An array of forces
            An array of size 3*N,
        pk: float
            Strength of harmonic potential between particles
        prmin: float
            Minimum of harmonic potential
        prmax: float
            Cutoff distance of harmonic potential
        lje: LJ strength between particles
        ljr: LJ min between particles
        
        wlje: float
            Strength of the LJ from wall
        wljr: float
            Range of the LJ from wall
        '''
        cdef int N=self.N, i, j, Z=2*N
        cdef double dx, dy, dz, dr, idr, rminbyr, fac, fx, fy, fz, hh, facss,abshh,sgnhh

        for i in prange(N, nogil=True):
            fx = 0.0; fy = 0.0; fz = 0.0;
            hh = r[i+Z]
            abshh=sqrt(hh*hh)
            sgnhh=hh/abshh
            idr = 1/hh
            if abshh<wljr:
                  rminbyr = 1.5*idr
                  fac   = 0.1*(pow(rminbyr, 12) - pow(rminbyr, 6))*idr
                  fz+=fac
                # rminbyr = wljr*idr
                # fac   = wlje*(pow(rminbyr, 12) - pow(rminbyr, 6))*idr
                # fz += fac      ##LJ from the wall

            for j in range(N):
                dx = r[i  ] - r[j   ]
                dy = r[i+N] - r[j+N]
                dz = r[i+Z] - r[j+Z]
                dr = sqrt(dx*dx + dy*dy + dz*dz)
                if i != j and dr < prmax:
                    idr     = 1.0/dr
                    fac   = pk*(prmin-dr)
                    fx += fac*dx*idr
                    fy += fac*dy*idr
                    fz += fac*dz*idr
                    if dr<ljr:
                        idr     = 1.0/dr
                        rminbyr = ljr*idr
                        fac   = lje*(pow(rminbyr, 12) - pow(rminbyr, 6))*idr*idr
                        fx += fac*dx
                        fy += fac*dy
                        fz += fac*dz

                    

            F[i]   += fx
            F[i+N] += fy
            F[i+Z] += fz
        return
        

    cpdef harmonicRepulsionPPPW(self, double [:] F, double [:] r, double partE=10, double partR=5, double wallE=2, double wallR=5.0):
        cdef int N = self.N, i, j, Z= 2*N
        cdef double dx, dy, dz, dr, idr, rminbyr, fac, fx, fy, fz, hh

        for i in prange(N, nogil=True):
            fx = 0.0; fy = 0.0; fz = 0.0;
            hh = r[i+Z]
            if hh<wallR:
                fz += wallE*(wallR-hh)

            for j in range(N):
                dx = r[i  ] - r[j   ]
                dy = r[i+N] - r[j+N]
                dz = r[i+Z] - r[j+Z]
                dr = sqrt(dx*dx + dy*dy + dz*dz)
                if i != j and dr < partR:
                    idr     = 1.0/dr
                    fac = partE*(partR - dr)*idr
                    fx += fac*dx
                    fy += fac*dy
                    fz += fac*dz

            F[i]   += fx
            F[i+N] += fy
            F[i+Z] += fz
        return


    cpdef lennardJonesXWall(self, double [:] F, double [:] r, double wlje=0.12, double wljr=3.0):
        """
        The standard Lennard-Jones potential truncated at the minimum (aslo called WCA potential)
        
            We choose \phi(r) = lje/12 (rr^12 - 2*rr^6 ) + lje/12,  as the standard WCA potential.
            ljr: minimum of the LJ potential and rr=ljr/r.
            This force is only in z-direction due to wall at x=0

        ...

        Parameters
        ----------
        r: np.array
            An array of positions
            An array of size 3*N,
        F: np.array
            An array of forces
            An array of size 3*N,
        lje: float
            Strength of the LJ 
        ljr: float
            Range of the LJ 


        """

        cdef int N=self.N, i, j, Z=2*N
        cdef double dx, dy, dz, dr, idr, rminbyr, fac, fx, fy, fz, hh

        for i in prange(N, nogil=True):
            fx = 0.0; fy = 0.0; fz = 0.0;
            hh = r[i]+11
            if hh<wljr:
                idr = 1/hh
                rminbyr = wljr*idr
                fac   = wlje*(pow(rminbyr, 12) - pow(rminbyr, 6))*idr
                fx += fac       ##LJ from the wall

            F[i]    += fx
        return

        
    cpdef staticlennardJones(self, double [:] F, double [:] r, double [:] rS, 
                           double lje=0.0100, double ljr=3, double a=1):
        '''
        non-dynamical static particles useful for simulating infinite crystal
        
           ...

        Parameters
        ----------
        r: np.array
            An array of positions
            An array of size 3*N,
        F: np.array
            An array of forces
            An array of size 3*N,
        rS: float
            positions of non-dynamic particles
    
        '''
        cdef int N=self.N, i, j, Z=2*N, Ns = len(rS)/3
        cdef double dx, dy, dz, dr, idr, fac=3*a/4, spring, fx, fy, fz,rminbyr
        for i in prange(N, nogil=True):
            fx=0; fy=0; fz=0;
            for j in range(Ns):
                dx = r[i  ] - rS[j   ]
                dy = r[i+N] - rS[j+Ns]
                dz = r[i+Z] - rS[j+Ns*2]
                dr = sqrt(dx*dx + dy*dy + dz*dz)
                if dr < ljr:
                    idr     = 1.0/dr
                    rminbyr = ljr*idr
                    fac   = lje*(pow(rminbyr, 12) - pow(rminbyr, 6))*idr*idr

                    fx += fac*dx
                    fy += fac*dy
                    fz += fac*dz
                    
                
            F[i]    += fx
            F[i+N] += fy
            F[i+Z] += fz
        return
        
    
    cpdef staticHarmonic(self, double [:] F, double [:] r, double [:] rS, 
                           double pk=0.0100, double prmin=3, double prmax=4,double a=1):
        '''
        non-dynamical static particles useful for simulating infinite crystal
        
           ...

        Parameters
        ----------
        r: np.array
            An array of positions
            An array of size 3*N,
        F: np.array
            An array of forces
            An array of size 3*N,
        rS: float
            positions of non-dynamic particles
    
        '''
        cdef int N=self.N, i, j, Z=2*N, Ns = len(rS)/3
        cdef double dx, dy, dz, dr, idr, fac=3*a/4, spring, fx, fy, fz
        for i in prange(N, nogil=True):
            fx=0; fy=0; fz=0;
            for j in range(Ns):
                dx = r[i  ] - rS[j   ]
                dy = r[i+N] - rS[j+Ns]
                dz = r[i+Z] - rS[j+Ns*2]
                dr = sqrt(dx*dx + dy*dy + dz*dz)
                if dr < prmax:
                    '''soft spring repulsion to keep particles away'''
                    idr=1/dr
                    spring= pk*(prmin-dr)
                    fx += spring*dx*idr
                    fy += spring*dy*idr
                    fz += spring*dz*idr
                    
                
            F[i]   += fx
            F[i+N] += fy
            F[i+Z] += fz
        return


    cpdef harmonicConfinement(self, double [:] F, double [:] r, double cn):
        '''
        Forces on colloids in a harmonic trap
        '''
        cdef int N = self.N, i
        for i in prange(3*N, nogil=True):
            F[i] -= cn*r[i]
        return


    cpdef opticalConfinement(self, double [:] F, double [:] r, double [:] r0, double [:] k):
        """
        Force on colloids in optical traps of varying stiffnesses
        
        ...

        Parameters
        ----------
        r: np.array
            An array of positions
            An array of size 3*N,
        r0: np.array
            An array of trap centers
            An array of size 3*N,
        F: np.array
            An array of forces
            An array of size 3*N,
        k: float 
            Stiffness of the trap
        """

        cdef int N = self.N, i, i1, Z= 2*N
        for i in prange(N, nogil=True):
            F[i  ] -= k[i]*(r[i]   - r0[i]  )
            F[i+N] -= k[i]*(r[i+N] - r0[i+N])
            F[i+Z] -= k[i]*(r[i+Z] - r0[i+Z])
        return


    cpdef sedimentation(self, double [:] F, double g):
        """
        Force on colloids in sedimentation 
        
        ...

        Parameters
        ----------
        r: np.array
            An array of positions
            An array of size 3*N,
        F: np.array
            An array of forces
            An array of size 3*N,
        g: float 
            Gravity 
        """
        cdef int N = self.N, i, Z= 2*N
        for i in prange(N, nogil=True):
#            F[i   ] +=  0
#            F[i+N] +=  0
            F[i+Z] +=  g
        return


    cpdef membraneConfinement(self, double [:] F, double [:] r, double cn, double r0):
        """
        Force on colloids in membraneSurface 
        
        ...

        Parameters
        ----------
        r: np.array
            An array of positions
            An array of size 3*N,
        r0: np.array
            An array of trap centers
            An array of size 3*N,
        F: np.array
            An array of forces
            An array of size 3*N,
        cn: float 
            Stiffness of the trap
        """

        cdef int N = self.N, Z= 2*N
        cdef int i
        cdef double r0byr
        for i in prange(N, nogil=True):
            r0byr = (1.0*r0)/sqrt( r[i]*r[i] + r[i+N]*r[i+N] + r[i+Z]*r[i+Z] )
            F[i]   -= cn*(1 - r0byr)*r[i]
            F[i+N] -= cn*(1 - r0byr)*r[i+N]
            F[i+Z] -= cn*(1 - r0byr)*r[i+Z]
        return


    cpdef membraneBound(self, double [:] F, double [:] r, double cn, double r0):
        """
        Force on colloids in membraneSurface 
        
        ...

        Parameters
        ----------
        r: np.array
            An array of positions
            An array of size 3*N,
        r0: np.array
            An array of trap centers
            An array of size 3*N,
        F: np.array
            An array of forces
            An array of size 3*N,
        cn: float 
            Stiffness of the trap
        """
        cdef int N = self.N, Z= 2*N
        cdef int i
        cdef double r0byr, dist
        for i in prange(N, nogil=True):
            dist = sqrt( r[i]*r[i] + r[i+N]*r[i+N] + r[i+Z]*r[i+Z] )
            if  (dist - r0) > 0.0:
                r0byr = (1.0*r0)/dist
                F[i]   -= cn*(1 - r0byr)*r[i]
                F[i+N] -= cn*(1 - r0byr)*r[i+N]
                F[i+Z] -= cn*(1 - r0byr)*r[i+Z]
            else:
                pass
        return


    cpdef spring(self, double [:] F, double [:] r, double bondLength, double springModulus):
        """
        Force on colloids connected by a spring in a single polymer
        
        ...

        Parameters
        ----------
        r: np.array
            An array of positions
        F: np.array
            An array of forces
            An array of size 3*N,
            An array of size 3*N,
        bondLength: float
            The size of natural spring 
        springModulus: float 
            Stiffness of the trap
        """

        cdef int N = self.N, Z= 2*N
        cdef int Nf=1, Nm=N
        cdef:
            int i, ii, ip, jp, kp
            double dx12, dy12, dz12, idr12, dx23, dy23, dz23, idr23,
            double F_bend, F_spring, f1, f3, cos1

        # Spring Force 
        for ii in prange(Nf,nogil=True):
            for i in range(Nm-1):
                ip = Nm*ii + i
                jp = Nm*ii + i + 1
                dx12 = r[ip]    - r[jp]
                dy12 = r[ip+N] - r[jp+N]
                dz12 = r[ip+Z] - r[jp+Z] #
                F_spring = springModulus*(bondLength/sqrt(dx12*dx12+dy12*dy12+dz12*dz12) - 1.0) # Scalar part of spring force

                F[ip]      += F_spring*dx12
                F[ip + N] += F_spring*dy12
                F[ip + Z] += F_spring*dz12
                F[jp]      -= F_spring*dx12
                F[jp + N] -= F_spring*dy12
                F[jp + Z] -= F_spring*dz12 
        return


    cpdef multipolymers(self, int Nf, double [:] F, double [:] r, double bondLength, double springModulus, double bendModulus, double twistModulus):
        """
        Force on colloids in many polymers connected by a spring 
        
        ...

        Parameters
        ----------
        r: np.array
            An array of positions
        F: np.array
            An array of forces
            An array of size 3*N,
            An array of size 3*N,
        bondLength: float
            The size of natural spring 
        springModulus: float 
            Stiffness of the trap
        """
        cdef int N = self.N, Z= 2*N
        cdef:
            int i, ii, ip, jp, kp
            int Nm = N/Nf # int/int problem might happen
            double dx12, dy12, dz12, idr12, dx23, dy23, dz23, idr23,
            double F_bend, F_spring, f1, f3, cos1

        # Spring Force
        for ii in prange(Nf,nogil=True):
            for i in range(Nm-1):
                ip = Nm*ii + i
                jp = Nm*ii + i + 1
                dx12 = r[ip]    - r[jp]
                dy12 = r[ip+N] - r[jp+N]
                dz12 = r[ip+Z] - r[jp+Z] #
                F_spring = springModulus*(bondLength/sqrt(dx12*dx12+dy12*dy12+dz12*dz12) - 1.0) # Scalar part of spring force

                F[ip]     += F_spring*dx12
                F[ip + N] += F_spring*dy12
                F[ip + Z] += F_spring*dz12
                F[jp]     -= F_spring*dx12
                F[jp + N] -= F_spring*dy12
                F[jp + Z] -= F_spring*dz12

        # Bending Force
        for ii in prange(Nf,nogil=True):
            for i in range(Nm-2):
                ip = (Nm*ii+i)
                jp = (Nm*ii+i+1)
                kp = (Nm*ii+i+2)
                dx12 = r[ip]   - r[jp]
                dy12 = r[ip+N] - r[jp+N]
                dz12 = r[ip+Z] - r[jp+Z] #
                idr12 = 1.0/sqrt( dx12*dx12 + dy12*dy12 + dz12*dz12 )

                dx23 = r[jp]   - r[kp]
                dy23 = r[jp+N] - r[kp+N]
                dz23 = r[jp+Z] - r[kp+Z] #
                idr23 = 1.0/sqrt( dx23*dx23 + dy23*dy23 + dz23*dz23 )

                cos1 = (dx12*dx23 + dy12*dy23 + dz12*dz23)
                F_bend = bendModulus*idr12*idr23/bondLength

                f1 = F_bend*( dx23 - dx12*cos1*idr12*idr12)
                f3 = F_bend*(-dx12 + dx23*cos1*idr23*idr23)

                F[ip] += f1
                F[jp] -= f1+f3
                F[kp] += f3

                f1 = F_bend*( dy23 - dy12*cos1*idr12*idr12)
                f3 = F_bend*(-dy12 + dy23*cos1*idr23*idr23)

                F[ip+N] += f1
                F[jp+N] -= f1+f3
                F[kp+N] += f3

                f1 = F_bend*( dz23 - dz12*cos1*idr12*idr12)
                f3 = F_bend*(-dz12 + dz23*cos1*idr23*idr23)

                F[ip+Z] += f1
                F[jp+Z] -= f1+f3
                F[kp+Z] += f3
        return

    
    cpdef multiRingpolymers(self, int Nf, double [:] F, double [:] r, double bondLength, double springModulus, double bendModulus, double twistModulus):
        """
        Force on colloids connected by a spring in a ring polymer
        
        ...

        Parameters
        ----------
        r: np.array
            An array of positions
        F: np.array
            An array of forces
            An array of size 3*N,
            An array of size 3*N,
        bondLength: float
            The size of natural spring 
        springModulus: float 
            Stiffness of the trap
        """
        cdef int N = self.N, Z= 2*N
        cdef:
             int i, ii, ip, jp, kp
             int Nm = N/Nf # int/int problem might happen
             double dx12, dy12, dz12, idr12, dx23, dy23, dz23, idr23,
             double F_bend, F_spring, f1, f3, cos1

        # Spring Force
        for ii in prange(Nf, nogil=True):
            for i in range(Nm):
                ip = (ii*Nm + i)
                jp = (ii*Nm +((i+1) % Nm)) #NOTE : Nm or Nm-1
                dx12 = r[ip]   - r[jp]
                dy12 = r[ip+N] - r[jp+N]
                dz12 = r[ip+Z] - r[jp+Z]
                F_spring = springModulus*(bondLength/sqrt(dx12*dx12+dy12*dy12+dz12*dz12) - 1.0) # Scalar part of spring force

                F[ip]     += F_spring*dx12
                F[ip + N] += F_spring*dy12
                F[ip + Z] += F_spring*dz12
                F[jp]     -= F_spring*dx12
                F[jp + N] -= F_spring*dy12
                F[jp + Z] -= F_spring*dz12

        # Bending Force
        for ii in prange(Nf,nogil=True):
            for i in range(Nm):
                ip = ( ii*Nm + i  )
                jp = ( ii*Nm +((i+1) % Nm))
                kp = ( ii*Nm +((i+2) % Nm))
                dx12 = r[ip]      - r[jp]
                dy12 = r[ip+N]   - r[jp+N]
                dz12 = r[ip+Z]   - r[jp+Z] #
                idr12 = 1.0/sqrt( dx12*dx12 + dy12*dy12 + dz12*dz12 )

                dx23 = r[jp]   - r[kp]
                dy23 = r[jp+N] - r[kp+N]
                dz23 = r[jp+Z] - r[kp+Z] #
                idr23 = 1.0/sqrt( dx23*dx23 + dy23*dy23 + dz23*dz23 )

                cos1 = (dx12*dx23 + dy12*dy23 + dz12*dz23)
                F_bend = bendModulus*idr12*idr23/bondLength

                f1 = F_bend*( dx23 - dx12*cos1*idr12*idr12)
                f3 = F_bend*(-dx12 + dx23*cos1*idr23*idr23)

                F[ip] += f1
                F[jp] -= f1+f3
                F[kp] += f3

                f1 = F_bend*( dy23 - dy12*cos1*idr12*idr12)
                f3 = F_bend*(-dy12 + dy23*cos1*idr23*idr23)

                F[ip+N] += f1
                F[jp+N] -= f1+f3
                F[kp+N] += f3

                f1 = F_bend*( dz23 - dz12*cos1*idr12*idr12)
                f3 = F_bend*(-dz12 + dz23*cos1*idr23*idr23)

                F[ip+Z] += f1
                F[jp+Z] -= f1+f3
                F[kp+Z] += f3
        return


    cpdef membraneSurface(self, int Nmx, int Nmy, double [:] F, double [:] r, double bondLength, double springModulus, double bendModulus):
        """
        Force on colloids connected as a membrane 
        
        ...

        Parameters
        ----------
        r: np.array
            An array of positions
            An array of size 3*N,
        F: np.array
            An array of forces
            An array of size 3*N,
        bondLength: float
            The size of natural spring 
        springModulus: float 
            Stiffness of the trap
        bendModulus: float 
            Bending cost
        """


        cdef int N = self.N, Z= 2*N
        cdef:
            int i1, i2, ip, jp, kp
            double dx12, dy12, dz12, idr12, dx23, dy23, dz23, idr23, cos1, f1, f3
            double F_spring, F_bend
            # pointers in the Position
            double * ri = & r[0]
            double * rj = & r[0]
            double * rk = & r[0]
            # pointers in the Force
            double * Fi = & F[0]
            double * Fj = & F[0]
            double * Fk = & F[0]

        # Spring Force in x direction : first loop in y
        # This loop should be faster compare to the  next for
        for i2 in prange(Nmy,nogil=True):
            for i1 in range(Nmx-1):
                ip = Nmx*i2 + i1
                jp = Nmx*i2 + (i1 + 1)
                dx12 = ri[ip]   - rj[jp]
                dy12 = ri[ip+N] - rj[jp+N]
                dz12 = ri[ip+Z] - rj[jp+Z]
                F_spring = springModulus*(bondLength/sqrt(dx12*dx12+dy12*dy12+dz12*dz12) - 1.0) # Scalar part of spring force

                Fi[ip]     += F_spring*dx12
                Fi[ip + N] += F_spring*dy12
                Fi[ip + Z] += F_spring*dz12
                Fj[jp]     -= F_spring*dx12
                Fj[jp + N] -= F_spring*dy12
                Fj[jp + Z] -= F_spring*dz12

        # Spring Force in y direction : first loop in x
        # this should be a bit slow as pointer jumps 3*Nmx in every inner loop step
        for i1 in prange(Nmx,nogil=True):
            for i2 in range(Nmy-1):
                ip = Nmx*i2 + i1
                jp = Nmx*(i2+1) + i1
                dx12 = ri[ip]   - rj[jp]
                dy12 = ri[ip+N] - rj[jp+N]
                dz12 = ri[ip+Z] - rj[jp+Z]
                F_spring = springModulus*(bondLength/sqrt(dx12*dx12+dy12*dy12+dz12*dz12) - 1.0)

                Fi[ip]     += F_spring*dx12
                Fi[ip + N] += F_spring*dy12
                Fi[ip + Z] += F_spring*dz12
                Fj[jp]     -= F_spring*dx12
                Fj[jp + N] -= F_spring*dy12
                Fj[jp + Z] -= F_spring*dz12

        # Bending Force
        for i2 in prange(Nmy,nogil=True):
            for i1 in range(Nmx-2):
                ip = Nmx*i2 + i1
                jp = Nmx*i2 + (i1+1)
                kp = Nmx*i2 + (i1+2)

                dx12 = ri[ip]   - rj[jp]
                dy12 = ri[ip+N] - rj[jp+N]
                dz12 = ri[ip+Z] - rj[jp+Z] #
                idr12 = 1.0/sqrt( dx12*dx12 + dy12*dy12 + dz12*dz12 )

                dx23 = rj[jp]   - rk[kp]
                dy23 = rj[jp+N] - rk[kp+N]
                dz23 = rj[jp+Z] - rk[kp+Z] #
                idr23 = 1.0/sqrt( dx23*dx23 + dy23*dy23 + dz23*dz23 )

                cos1 = (dx12*dx23 + dy12*dy23 + dz12*dz23)
                F_bend = bendModulus*idr12*idr23/bondLength

                f1 = F_bend*( dx23 - dx12*cos1*idr12*idr12)
                f3 = F_bend*(-dx12 + dx23*cos1*idr23*idr23)

                Fi[ip] += f1
                Fj[jp] -= f1+f3
                Fk[kp] += f3

                f1 = F_bend*( dy23 - dy12*cos1*idr12*idr12)
                f3 = F_bend*(-dy12 + dy23*cos1*idr23*idr23)

                Fi[ip+N] += f1
                Fj[jp+N] -= f1+f3
                Fk[kp+N] += f3

                f1 = F_bend*( dz23 - dz12*cos1*idr12*idr12)
                f3 = F_bend*(-dz12 + dz23*cos1*idr23*idr23)

                Fi[ip+Z] += f1
                Fj[jp+Z] -= f1+f3
                Fk[kp+Z] += f3
        for i1 in prange(Nmx,nogil=True):
            for i2 in range(Nmy-2):
                ip = Nmx*i2     + i1
                jp = Nmx*(i2+1) + i1
                kp = Nmx*(i2+2) + i1

                dx12 = ri[ip]   - rj[jp]
                dy12 = ri[ip+N] - rj[jp+N]
                dz12 = ri[ip+Z] - rj[jp+Z] #
                idr12 = 1.0/sqrt( dx12*dx12 + dy12*dy12 + dz12*dz12 )

                dx23 = rj[jp]   - rk[kp]
                dy23 = rj[jp+N] - rk[kp+N]
                dz23 = rj[jp+Z] - rk[kp+Z] #
                idr23 = 1.0/sqrt( dx23*dx23 + dy23*dy23 + dz23*dz23 )

                cos1 = (dx12*dx23 + dy12*dy23 + dz12*dz23)
                F_bend = bendModulus*idr12*idr23/bondLength

                f1 = F_bend*( dx23 - dx12*cos1*idr12*idr12)
                f3 = F_bend*(-dx12 + dx23*cos1*idr23*idr23)

                Fi[ip] += f1
                Fj[jp] -= f1+f3
                Fk[kp] += f3

                f1 = F_bend*( dy23 - dy12*cos1*idr12*idr12)
                f3 = F_bend*(-dy12 + dy23*cos1*idr23*idr23)

                Fi[ip+N] += f1
                Fj[jp+N] -= f1+f3
                Fk[kp+N] += f3

                f1 = F_bend*( dz23 - dz12*cos1*idr12*idr12)
                f3 = F_bend*(-dz12 + dz23*cos1*idr23*idr23)

                Fi[ip+Z] += f1
                Fj[jp+Z] -= f1+f3
                Fk[kp+Z] += f3
        return


    cpdef Cosserat(self, double [:] F, double [:] r, double [:] e1, double [:] e2, double [:] e3, double Lambda, double d):
        """
        The force for the Cosserat solids

        ...

        Parameter
        ----------
        r: np.array
            An array of positions
            An array of size 3*N,
        F: np.array
            An array of forces
            An array of size 3*N,
        e1, e2, e3: np.array
                    Arrays of Orientations
                    Arrays of size 3*N
        Lambda: float
            Strength of the force 
        d: float 
            particle distance
        """
        cdef int N = self.N, i, k, xx = 2*N, ip, im
        cdef double L = N*d
        cdef double dxp, dxm, dyp, dym, dzp, dzm

        for i in prange(N,nogil=True):
            ip = (i + 1) % N
            im = (i - 1 + N) % N

            dxp = r[ip] - r[i]
            dxm = r[i] - r[im]
            dyp = r[N + ip] - r[N + i]
            dym = r[N + i] - r[N + im]
            dzp = r[xx + ip] - r[xx + i]
            dzm = r[xx + i] - r[xx + im]

            if i == 0:
                dxm = r[i] - r[im] + L
            elif i == N - 1:
                dxp = r[ip] - r[i] + L

            for k in range(3):
                F[i+k*N] += 0.5 * Lambda * ((e1[i] * dxp + e1[N+i] * dyp + e1[xx+i] * dzp - d) * e1[i+k*N] \
                                          + (e2[i] * dxp + e2[N+i] * dyp + e2[xx+i] * dzp) * e2[i+k*N] \
                                          + (e3[i] * dxp + e3[N+i] * dyp + e3[xx+i] * dzp) * e3[i+k*N])
                F[i+k*N] -= 0.5 * Lambda * ((e1[i] * dxm + e1[N+i] * dym + e1[xx+i] * dzm - d) * e1[i+k*N] \
                                          + (e2[i] * dxm + e2[N+i] * dym + e2[xx+i] * dzm) * e2[i+k*N] \
                                          + (e3[i] * dxm + e3[N+i] * dym + e3[xx+i] * dzm) * e3[i+k*N])
                F[i+k*N] += 0.5 * Lambda * ((e1[ip] * dxp + e1[N+ip] * dyp + e1[xx+ip] * dzp - d) * e1[ip+k*N] \
                                          + (e2[ip] * dxp + e2[N+ip] * dyp + e2[xx+ip] * dzp) * e2[ip+k*N] \
                                          + (e3[ip] * dxp + e3[N+ip] * dyp + e3[xx+ip] * dzp) * e3[ip+k*N])
                F[i+k*N] -= 0.5 * Lambda * ((e1[im] * dxm + e1[N+im] * dym + e1[xx+im] * dzm - d) * e1[im+k*N] \
                                          + (e2[im] * dxm + e2[N+im] * dym + e2[xx+im] * dzm) * e2[im+k*N] \
                                          + (e3[im] * dxm + e3[N+im] * dym + e3[xx+im] * dzm) * e3[im+k*N])
        return 

@cython.boundscheck(False)
@cython.cdivision(True)
@cython.nonecheck(False)
@cython.wraparound(False)
cdef class Torques:
    """
    Computes torques in a system of colloidal particles

    Methods in the Torques class take input,
        * arrays of positions, Torques 
        * parameters for a given potential
    
    The array of torques is then update by each method. 

    ...

    Parameters
    ----------
    particles: int
        Number of particles (N)


    """
    def __init__(self, particles=1):
        self.N = particles 


    cpdef bottomHeaviness(self, double [:] T, double [:] p, double bh=1.0):
        """
        Torque due to bottom-heaviness
        
        ...

        Parameters
        ----------
        p: np.array
            An array of Orientations
            An array of size 3*N,
        F: np.array
            An array of Torques
            An array of size 3*N,
        bh: float 
            bottomHeaviness 
        """
        cdef int N = self.N, i, Z= 2*N
        for i in prange(N, nogil=True):
            T[i   ] +=  -bh*p[i+N]  # torque is bh\times p
            T[i+N] +=  bh*p[i]
            T[i+Z] +=  0
        return

    
    cpdef magnetic(self, double[:] T, double [:] p, double m0, double Bx, double By, double Bz):
        """
        
        Torque due to magnetotaxis. 
        The torque on microparticle in an external magnetic field ${\bf B}$  is ${\bf T} = {\bf m\times B}$
        We assume $m = m_0 \bf p$
        
        ...
        Parameters
        ----------
        p: np.array
            An array of Orientations
            An array of size 3*N,
        T: np.array
            An array of Torques
            An array of size 3*N,
        m0: float 
            magnetic moment
        Bx,By,Bz : float
            magnetic field components
        """
  
        cdef int N = self.N, i, Z= 2*N
        for i in prange(N, nogil=True):
            T[i  ]  += m0*(p[i+N]*Bz  -p[i+Z]*By)
            T[i+N]  += m0*(p[i+Z]*Bx -p[i]*Bz)
            T[i+Z] += m0*(p[i]*By    -p[i+N]*Bx)
        return 


    cpdef Cosserat(self, double [:] T, double [:] r, double [:] e1, double [:] e2, double [:] e3, double Lambda, double mu, double d):
        """
        The torque for the Cosserat solids

        ...

        Parameter
        ----------
        r: np.array
            An array of positions
            An array of size 3*N,
        T: np.array
            An array of torques
            An array of size 3*N,
        e1, e2, e3: np.array
                    Arrays of Orientations
                    Arrays of size 3*N
        Lambda: float
            Strength of the force 
        mu: float 
            Strength of the torque
        d: float 
            particle distance
        """
        cdef int N = self.N, i, j, k, xx = 2*N, ip, im
        cdef double L = N*d
        cdef double drp[3], drm[3], e[3][3], ep[3][3], em[3][3]
        cdef double e_dot_r, e_dot_e
        cdef double cross[3]

        for i in prange(N,nogil=True):
            ip = (i + 1) % N
            im = (i - 1 + N) % N

            drp[0] = r[ip] - r[i]
            drp[1] = r[N+ip] - r[N+i]
            drp[2] = r[xx + ip] - r[xx + i]

            drm[0] = r[i] - r[im]
            drm[1] = r[N + i] - r[N + im]
            drm[2] = r[xx + i] - r[xx + im]

            if i == 0:
                drm[0] += L
            elif i == N - 1:
                drp[0] += L
            
            for k in range(3):
                e[0][k] = e1[k*N+i]
                e[1][k] = e2[k*N+i]
                e[2][k] = e3[k*N+i]

                ep[0][k] = e1[k*N+ip]
                ep[1][k] = e2[k*N+ip]
                ep[2][k] = e3[k*N+ip]

                em[0][k] = e1[k*N+im]
                em[1][k] = e2[k*N+im]
                em[2][k] = e3[k*N+im]

            for j in range(3):
                e_dot_r  = e[j][0] * drp[0] + e[j][1] * drp[1] + e[j][2] * drp[2]
                cross[0] = e[j][1] * drp[2] - e[j][2] * drp[1]
                cross[1] = e[j][2] * drp[0] - e[j][0] * drp[2]
                cross[2] = e[j][0] * drp[1] - e[j][1] * drp[0]
                for k in range(3):
                    T[i+k*N] -= 0.5 * Lambda * (e_dot_r - (d if j == 0 else 0)) * cross[k]

                e_dot_r  = e[j][0] * drm[0] + e[j][1] * drm[1] + e[j][2] * drm[2]
                cross[0] = e[j][1] * drm[2] - e[j][2] * drm[1]
                cross[1] = e[j][2] * drm[0] - e[j][0] * drm[2]
                cross[2] = e[j][0] * drm[1] - e[j][1] * drm[0]
                for k in range(3):
                    T[i+k*N] -= 0.5 * Lambda * (e_dot_r - (d if j == 0 else 0)) * cross[k]

            for j in range(3):
                e_dot_e = e[j][0] * ep[(j+1)%3][0] + e[j][1] * ep[(j+1)%3][1] + e[j][2] * ep[(j+1)%3][2] 
                cross[0] = e[j][1] * ep[(j+1)%3][2] - e[j][2] * ep[(j+1)%3][1]
                cross[1] = e[j][2] * ep[(j+1)%3][0] - e[j][0] * ep[(j+1)%3][2]
                cross[2] = e[j][0] * ep[(j+1)%3][1] - e[j][1] * ep[(j+1)%3][0]
                for k in range(3):
                    T[i+k*N] -= 0.5 * mu * (e_dot_e) * cross[k]
                
                e_dot_e  = e[j][0] * em[(j+2)%3][0] + e[j][1] * em[(j+2)%3][1] + e[j][2] * em[(j+2)%3][2]
                cross[0] = e[j][1] * em[(j+2)%3][2] - e[j][2] * em[(j+2)%3][1]
                cross[1] = e[j][2] * em[(j+2)%3][0] - e[j][0] * em[(j+2)%3][2]
                cross[2] = e[j][0] * em[(j+2)%3][1] - e[j][1] * em[(j+2)%3][0]
                for k in range(3):
                    T[i+k*N] -= 0.5 * mu * (e_dot_e) * cross[k]
        return