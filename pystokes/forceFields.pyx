cimport cython
from libc.math cimport sqrt, pow
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
        Number of particles (Np)


    """
    def __init__(self, particles=1):
        self.Np = particles


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
            An array of size 3*Np,
        F: np.array
            An array of forces
            An array of size 3*Np,
        lje: float
            Strength of the LJ 
        ljr: float
            Range of the LJ 


        """

        cdef int Np = self.Np, i, j, xx = 2*Np
        cdef double dx, dy, dz, dr2, idr, rminbyr, fac, fx, fy, fz

        for i in prange(Np,nogil=True):
            fx = 0.0; fy = 0.0; fz = 0.0;
            for j in range(Np):
                dx = r[i   ] - r[j   ]
                dy = r[i+Np] - r[j+Np]
                dz = r[i+xx] - r[j+xx]
                dr2 = dx*dx + dy*dy + dz*dz
                if i != j and dr2 < (ljr*ljr):
                    idr     = 1.0/sqrt(dr2)
                    rminbyr = ljr*idr
                    fac   = lje*(pow(rminbyr, 12) - pow(rminbyr, 6))*idr*idr

                    fx += fac*dx
                    fy += fac*dy
                    fz += fac*dz
            F[i]    += fx
            F[i+Np] += fy
            F[i+xx] += fz
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
            An array of size 3*Np,
        F: np.array
            An array of forces
            An array of size 3*Np,
        lje: float
            Strength of the LJ 
        ljr: float
            Range of the LJ 


        """

        cdef int Np=self.Np, i, j, xx=2*Np
        cdef double dx, dy, dz, dr, idr, rminbyr, fac, fx, fy, fz, hh

        for i in prange(Np, nogil=True):
            fx = 0.0; fy = 0.0; fz = 0.0;
            hh = r[i+xx]
            if hh<wljr:
                idr = 1/hh
                rminbyr = wljr*idr
                fac   = wlje*(pow(rminbyr, 12) - pow(rminbyr, 6))*idr
                fz += fac       ##LJ from the wall

            for j in range(Np):
                dx = r[i   ] - r[j   ]
                dy = r[i+Np] - r[j+Np]
                dz = r[i+xx] - r[j+xx]
                dr = sqrt(dx*dx + dy*dy + dz*dz)
                if i != j and dr < ljr:
                    idr     = 1.0/dr
                    rminbyr = ljr*idr
                    fac   = lje*(pow(rminbyr, 12) - pow(rminbyr, 6))*idr*idr

                    fx += fac*dx
                    fy += fac*dy
                    fz += fac*dz

            F[i]    += fx
            F[i+Np] += fy
            F[i+xx] += fz
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
            An array of size 3*Np,
        F: np.array
            An array of forces
            An array of size 3*Np,
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
        cdef int Np=self.Np, i, j, xx=2*Np
        cdef double dx, dy, dz, dr, idr, rminbyr, fac, fx, fy, fz, hh, facss,abshh,sgnhh

        for i in prange(Np, nogil=True):
            fx = 0.0; fy = 0.0; fz = 0.0;
            hh = r[i+xx]
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

            for j in range(Np):
                dx = r[i   ] - r[j   ]
                dy = r[i+Np] - r[j+Np]
                dz = r[i+xx] - r[j+xx]
                dr = sqrt(dx*dx + dy*dy + dz*dz)
                if i != j and dr < prmax:
                    idr     = 1.0/dr
                    fac   = pk*(prmin-dr)

                    fx += fac*dx*idr
                    fy += fac*dy*idr
                    fz += fac*dz*idr

            F[i]    += fx
            F[i+Np] += fy
            F[i+xx] += fz
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
            An array of size 3*Np,
        F: np.array
            An array of forces
            An array of size 3*Np,
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
        cdef int Np=self.Np, i, j, xx=2*Np
        cdef double dx, dy, dz, dr, idr, rminbyr, fac, fx, fy, fz, hh, facss,abshh,sgnhh

        for i in prange(Np, nogil=True):
            fx = 0.0; fy = 0.0; fz = 0.0;
            hh = r[i+xx]
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

            for j in range(Np):
                dx = r[i   ] - r[j   ]
                dy = r[i+Np] - r[j+Np]
                dz = r[i+xx] - r[j+xx]
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

                    

            F[i]    += fx
            F[i+Np] += fy
            F[i+xx] += fz
        return
        

    cpdef harmonicRepulsionPPPW(self, double [:] F, double [:] r, double partE=10, double partR=5, double wallE=2, double wallR=5.0):
        cdef int Np = self.Np, i, j, xx = 2*Np
        cdef double dx, dy, dz, dr, idr, rminbyr, fac, fx, fy, fz, hh

        for i in prange(Np, nogil=True):
            fx = 0.0; fy = 0.0; fz = 0.0;
            hh = r[i+xx]
            if hh<wallR:
                fz += wallE*(wallR-hh)

            for j in range(Np):
                dx = r[i   ] - r[j   ]
                dy = r[i+Np] - r[j+Np]
                dz = r[i+xx] - r[j+xx]
                dr = sqrt(dx*dx + dy*dy + dz*dz)
                if i != j and dr < partR:
                    idr     = 1.0/dr
                    fac = partE*(partR - dr)*idr
                    fx += fac*dx
                    fy += fac*dy
                    fz += fac*dz

            F[i]    += fx
            F[i+Np] += fy
            F[i+xx] += fz
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
            An array of size 3*Np,
        F: np.array
            An array of forces
            An array of size 3*Np,
        lje: float
            Strength of the LJ 
        ljr: float
            Range of the LJ 


        """

        cdef int Np=self.Np, i, j, xx=2*Np
        cdef double dx, dy, dz, dr, idr, rminbyr, fac, fx, fy, fz, hh

        for i in prange(Np, nogil=True):
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
            An array of size 3*Np,
        F: np.array
            An array of forces
            An array of size 3*Np,
        rS: float
            positions of non-dynamic particles
    
        '''
        cdef int Np=self.Np, i, j, xx=2*Np, Ns = len(rS)/3
        cdef double dx, dy, dz, dr, idr, fac=3*a/4, spring, fx, fy, fz,rminbyr
        for i in prange(Np, nogil=True):
            fx=0; fy=0; fz=0;
            for j in range(Ns):
                dx = r[i   ] - rS[j   ]
                dy = r[i+Np] - rS[j+Ns]
                dz = r[i+xx] - rS[j+Ns*2]
                dr = sqrt(dx*dx + dy*dy + dz*dz)
                if dr < ljr:
                    idr     = 1.0/dr
                    rminbyr = ljr*idr
                    fac   = lje*(pow(rminbyr, 12) - pow(rminbyr, 6))*idr*idr

                    fx += fac*dx
                    fy += fac*dy
                    fz += fac*dz
                    
                
            F[i]    += fx
            F[i+Np] += fy
            F[i+xx] += fz
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
            An array of size 3*Np,
        F: np.array
            An array of forces
            An array of size 3*Np,
        rS: float
            positions of non-dynamic particles
    
        '''
        cdef int Np=self.Np, i, j, xx=2*Np, Ns = len(rS)/3
        cdef double dx, dy, dz, dr, idr, fac=3*a/4, spring, fx, fy, fz
        for i in prange(Np, nogil=True):
            fx=0; fy=0; fz=0;
            for j in range(Ns):
                dx = r[i   ] - rS[j   ]
                dy = r[i+Np] - rS[j+Ns]
                dz = r[i+xx] - rS[j+Ns*2]
                dr = sqrt(dx*dx + dy*dy + dz*dz)
                if dr < prmax:
                    '''soft spring repulsion to keep particles away'''
                    idr=1/dr
                    spring= pk*(prmin-dr)
                    fx += spring*dx*idr
                    fy += spring*dy*idr
                    fz += spring*dz*idr
                    
                
            F[i]    += fx
            F[i+Np] += fy
            F[i+xx] += fz
        return


    cpdef harmonicConfinement(self, double [:] F, double [:] r, double cn):
        '''
        Forces on colloids in a harmonic trap
        '''
        cdef int Np = self.Np, i
        for i in prange(3*Np, nogil=True):
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
            An array of size 3*Np,
        r0: np.array
            An array of trap centers
            An array of size 3*Np,
        F: np.array
            An array of forces
            An array of size 3*Np,
        k: float 
            Stiffness of the trap
        """

        cdef int Np = self.Np, i, i1, xx = 2*Np
        for i in prange(Np, nogil=True):
            F[i  ]  -= k[i]*(r[i]    - r0[i]  )
            F[i+Np] -= k[i]*(r[i+Np] - r0[i+Np])
            F[i+xx] -= k[i]*(r[i+xx] - r0[i+xx])
        return


    cpdef sedimentation(self, double [:] F, double g):
        """
        Force on colloids in sedimentation 
        
        ...

        Parameters
        ----------
        r: np.array
            An array of positions
            An array of size 3*Np,
        F: np.array
            An array of forces
            An array of size 3*Np,
        g: float 
            Gravity 
        """
        cdef int Np = self.Np, i, xx = 2*Np
        for i in prange(Np, nogil=True):
#            F[i   ] +=  0
#            F[i+Np] +=  0
            F[i+xx] +=  g
        return


    cpdef membraneConfinement(self, double [:] F, double [:] r, double cn, double r0):
        """
        Force on colloids in membraneSurface 
        
        ...

        Parameters
        ----------
        r: np.array
            An array of positions
            An array of size 3*Np,
        r0: np.array
            An array of trap centers
            An array of size 3*Np,
        F: np.array
            An array of forces
            An array of size 3*Np,
        cn: float 
            Stiffness of the trap
        """

        cdef int Np = self.Np, xx = 2*Np
        cdef int i
        cdef double r0byr
        for i in prange(Np, nogil=True):
            r0byr = (1.0*r0)/sqrt( r[i]*r[i] + r[i+Np]*r[i+Np] + r[i+xx]*r[i+xx] )
            F[i]   -= cn*(1 - r0byr)*r[i]
            F[i+Np] -= cn*(1 - r0byr)*r[i+Np]
            F[i+xx] -= cn*(1 - r0byr)*r[i+xx]
        return


    cpdef membraneBound(self, double [:] F, double [:] r, double cn, double r0):
        """
        Force on colloids in membraneSurface 
        
        ...

        Parameters
        ----------
        r: np.array
            An array of positions
            An array of size 3*Np,
        r0: np.array
            An array of trap centers
            An array of size 3*Np,
        F: np.array
            An array of forces
            An array of size 3*Np,
        cn: float 
            Stiffness of the trap
        """
        cdef int Np = self.Np, xx = 2*Np
        cdef int i
        cdef double r0byr, dist
        for i in prange(Np, nogil=True):
            dist = sqrt( r[i]*r[i] + r[i+Np]*r[i+Np] + r[i+xx]*r[i+xx] )
            if  (dist - r0) > 0.0:
                r0byr = (1.0*r0)/dist
                F[i]   -= cn*(1 - r0byr)*r[i]
                F[i+Np] -= cn*(1 - r0byr)*r[i+Np]
                F[i+xx] -= cn*(1 - r0byr)*r[i+xx]
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
            An array of size 3*Np,
            An array of size 3*Np,
        bondLength: float
            The size of natural spring 
        springModulus: float 
            Stiffness of the trap
        """

        cdef int Np = self.Np, xx = 2*Np
        cdef int Nf=1, Nm=Np
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
                dy12 = r[ip+Np] - r[jp+Np]
                dz12 = r[ip+xx] - r[jp+xx] #
                F_spring = springModulus*(bondLength/sqrt(dx12*dx12+dy12*dy12+dz12*dz12) - 1.0) # Scalar part of spring force

                F[ip]      += F_spring*dx12
                F[ip + Np] += F_spring*dy12
                F[ip + xx] += F_spring*dz12
                F[jp]      -= F_spring*dx12
                F[jp + Np] -= F_spring*dy12
                F[jp + xx] -= F_spring*dz12 
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
            An array of size 3*Np,
            An array of size 3*Np,
        bondLength: float
            The size of natural spring 
        springModulus: float 
            Stiffness of the trap
        """
        cdef int Np = self.Np, xx = 2*Np
        cdef:
            int i, ii, ip, jp, kp
            int Nm = Np/Nf # int/int problem might happen
            double dx12, dy12, dz12, idr12, dx23, dy23, dz23, idr23,
            double F_bend, F_spring, f1, f3, cos1

        # Spring Force
        for ii in prange(Nf,nogil=True):
            for i in range(Nm-1):
                ip = Nm*ii + i
                jp = Nm*ii + i + 1
                dx12 = r[ip]    - r[jp]
                dy12 = r[ip+Np] - r[jp+Np]
                dz12 = r[ip+xx] - r[jp+xx] #
                F_spring = springModulus*(bondLength/sqrt(dx12*dx12+dy12*dy12+dz12*dz12) - 1.0) # Scalar part of spring force

                F[ip]      += F_spring*dx12
                F[ip + Np] += F_spring*dy12
                F[ip + xx] += F_spring*dz12
                F[jp]      -= F_spring*dx12
                F[jp + Np] -= F_spring*dy12
                F[jp + xx] -= F_spring*dz12

        # Bending Force
        for ii in prange(Nf,nogil=True):
            for i in range(Nm-2):
                ip = (Nm*ii+i)
                jp = (Nm*ii+i+1)
                kp = (Nm*ii+i+2)
                dx12 = r[ip]   - r[jp]
                dy12 = r[ip+Np] - r[jp+Np]
                dz12 = r[ip+xx] - r[jp+xx] #
                idr12 = 1.0/sqrt( dx12*dx12 + dy12*dy12 + dz12*dz12 )

                dx23 = r[jp]   - r[kp]
                dy23 = r[jp+Np] - r[kp+Np]
                dz23 = r[jp+xx] - r[kp+xx] #
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

                F[ip+Np] += f1
                F[jp+Np] -= f1+f3
                F[kp+Np] += f3

                f1 = F_bend*( dz23 - dz12*cos1*idr12*idr12)
                f3 = F_bend*(-dz12 + dz23*cos1*idr23*idr23)

                F[ip+xx] += f1
                F[jp+xx] -= f1+f3
                F[kp+xx] += f3
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
            An array of size 3*Np,
            An array of size 3*Np,
        bondLength: float
            The size of natural spring 
        springModulus: float 
            Stiffness of the trap
        """
        cdef int Np = self.Np, xx = 2*Np
        cdef:
             int i, ii, ip, jp, kp
             int Nm = Np/Nf # int/int problem might happen
             double dx12, dy12, dz12, idr12, dx23, dy23, dz23, idr23,
             double F_bend, F_spring, f1, f3, cos1

        # Spring Force
        for ii in prange(Nf, nogil=True):
            for i in range(Nm):
                ip = (ii*Nm + i)
                jp = (ii*Nm +((i+1) % Nm)) #NOTE : Nm or Nm-1
                dx12 = r[ip]   - r[jp]
                dy12 = r[ip+Np] - r[jp+Np]
                dz12 = r[ip+xx] - r[jp+xx]
                F_spring = springModulus*(bondLength/sqrt(dx12*dx12+dy12*dy12+dz12*dz12) - 1.0) # Scalar part of spring force

                F[ip]     += F_spring*dx12
                F[ip + Np] += F_spring*dy12
                F[ip + xx] += F_spring*dz12
                F[jp]     -= F_spring*dx12
                F[jp + Np] -= F_spring*dy12
                F[jp + xx] -= F_spring*dz12

        # Bending Force
        for ii in prange(Nf,nogil=True):
            for i in range(Nm):
                ip = ( ii*Nm + i  )
                jp = ( ii*Nm +((i+1) % Nm))
                kp = ( ii*Nm +((i+2) % Nm))
                dx12 = r[ip]      - r[jp]
                dy12 = r[ip+Np]   - r[jp+Np]
                dz12 = r[ip+xx]   - r[jp+xx] #
                idr12 = 1.0/sqrt( dx12*dx12 + dy12*dy12 + dz12*dz12 )

                dx23 = r[jp]   - r[kp]
                dy23 = r[jp+Np] - r[kp+Np]
                dz23 = r[jp+xx] - r[kp+xx] #
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

                F[ip+Np] += f1
                F[jp+Np] -= f1+f3
                F[kp+Np] += f3

                f1 = F_bend*( dz23 - dz12*cos1*idr12*idr12)
                f3 = F_bend*(-dz12 + dz23*cos1*idr23*idr23)

                F[ip+xx] += f1
                F[jp+xx] -= f1+f3
                F[kp+xx] += f3
        return


    cpdef membraneSurface(self, int Nmx, int Nmy, double [:] F, double [:] r, double bondLength, double springModulus, double bendModulus):
        """
        Force on colloids connected as a membrane 
        
        ...

        Parameters
        ----------
        r: np.array
            An array of positions
            An array of size 3*Np,
        F: np.array
            An array of forces
            An array of size 3*Np,
        bondLength: float
            The size of natural spring 
        springModulus: float 
            Stiffness of the trap
        bendModulus: float 
            Bending cost
        """


        cdef int Np = self.Np, xx = 2*Np
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
                dy12 = ri[ip+Np] - rj[jp+Np]
                dz12 = ri[ip+xx] - rj[jp+xx]
                F_spring = springModulus*(bondLength/sqrt(dx12*dx12+dy12*dy12+dz12*dz12) - 1.0) # Scalar part of spring force

                Fi[ip]     += F_spring*dx12
                Fi[ip + Np] += F_spring*dy12
                Fi[ip + xx] += F_spring*dz12
                Fj[jp]     -= F_spring*dx12
                Fj[jp + Np] -= F_spring*dy12
                Fj[jp + xx] -= F_spring*dz12

        # Spring Force in y direction : first loop in x
        # this should be a bit slow as pointer jumps 3*Nmx in every inner loop step
        for i1 in prange(Nmx,nogil=True):
            for i2 in range(Nmy-1):
                ip = Nmx*i2 + i1
                jp = Nmx*(i2+1) + i1
                dx12 = ri[ip]   - rj[jp]
                dy12 = ri[ip+Np] - rj[jp+Np]
                dz12 = ri[ip+xx] - rj[jp+xx]
                F_spring = springModulus*(bondLength/sqrt(dx12*dx12+dy12*dy12+dz12*dz12) - 1.0)

                Fi[ip]     += F_spring*dx12
                Fi[ip + Np] += F_spring*dy12
                Fi[ip + xx] += F_spring*dz12
                Fj[jp]     -= F_spring*dx12
                Fj[jp + Np] -= F_spring*dy12
                Fj[jp + xx] -= F_spring*dz12

        # Bending Force
        for i2 in prange(Nmy,nogil=True):
            for i1 in range(Nmx-2):
                ip = Nmx*i2 + i1
                jp = Nmx*i2 + (i1+1)
                kp = Nmx*i2 + (i1+2)

                dx12 = ri[ip]   - rj[jp]
                dy12 = ri[ip+Np] - rj[jp+Np]
                dz12 = ri[ip+xx] - rj[jp+xx] #
                idr12 = 1.0/sqrt( dx12*dx12 + dy12*dy12 + dz12*dz12 )

                dx23 = rj[jp]   - rk[kp]
                dy23 = rj[jp+Np] - rk[kp+Np]
                dz23 = rj[jp+xx] - rk[kp+xx] #
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

                Fi[ip+Np] += f1
                Fj[jp+Np] -= f1+f3
                Fk[kp+Np] += f3

                f1 = F_bend*( dz23 - dz12*cos1*idr12*idr12)
                f3 = F_bend*(-dz12 + dz23*cos1*idr23*idr23)

                Fi[ip+xx] += f1
                Fj[jp+xx] -= f1+f3
                Fk[kp+xx] += f3
        for i1 in prange(Nmx,nogil=True):
            for i2 in range(Nmy-2):
                ip = Nmx*i2     + i1
                jp = Nmx*(i2+1) + i1
                kp = Nmx*(i2+2) + i1

                dx12 = ri[ip]   - rj[jp]
                dy12 = ri[ip+Np] - rj[jp+Np]
                dz12 = ri[ip+xx] - rj[jp+xx] #
                idr12 = 1.0/sqrt( dx12*dx12 + dy12*dy12 + dz12*dz12 )

                dx23 = rj[jp]   - rk[kp]
                dy23 = rj[jp+Np] - rk[kp+Np]
                dz23 = rj[jp+xx] - rk[kp+xx] #
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

                Fi[ip+Np] += f1
                Fj[jp+Np] -= f1+f3
                Fk[kp+Np] += f3

                f1 = F_bend*( dz23 - dz12*cos1*idr12*idr12)
                f3 = F_bend*(-dz12 + dz23*cos1*idr23*idr23)

                Fi[ip+xx] += f1
                Fj[jp+xx] -= f1+f3
                Fk[kp+xx] += f3
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
        Number of particles (Np)


    """
    def __init__(self, particles=1):
        self.Np = particles 


    cpdef bottomHeaviness(self, double [:] T, double [:] p, double bh=1.0):
        """
        Torque due to bottom-heaviness
        
        ...

        Parameters
        ----------
        p: np.array
            An array of Orientations
            An array of size 3*Np,
        F: np.array
            An array of Torques
            An array of size 3*Np,
        bh: float 
            bottomHeaviness 
        """
        cdef int Np = self.Np, i, xx = 2*Np
        for i in prange(Np, nogil=True):
            T[i   ] +=  -bh*p[i+Np]  # torque is bh\times p
            T[i+Np] +=  bh*p[i]
            T[i+xx] +=  0
        return

