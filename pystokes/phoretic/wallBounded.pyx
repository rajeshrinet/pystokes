
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
cdef class Phoresis:
    """
    Phoresis 
    
    ...

    Parameters
    ----------
    radius: float
        Radius of the particles.    
    particles: int
        Number of particles 
    phoreticConstant: float 
        PhoreticConstant 
    Examples
    --------
    An example of Phoresis

    """
    def __init__(self, radius=1, particles=1, phoreticConstant=1.0):
        self.a  = radius
        self.Np = particles
        self.D  = phoreticConstant

    cpdef elastance00(self, double [:] C0, double [:] r, double [:] J0):
        cdef int i, j, Np=self.Np, xx=2*Np
        cdef double dx, dy, dz, idr, h2, hsq, idr2, A1
        cdef double vx, vy, vz, ii , cc=0 ## cc is the concentration constant
        cdef double mud = J0[0]/(4*PI*self.D)

        for i in prange(Np, nogil=True):
            cc=0;
            for j in range(Np):
                if i!=j:
                    dx = r[i]    - r[j]
                    dy = r[i+Np]  - r[j+Np]
                    dz = r[i+xx]  - r[j+xx]
                    idr = 1.0/sqrt(dx*dx + dy*dy + dz*dz)
                    cc += idr

                    ###contributions from the image 
                    dz = r[i+xx] + r[j+xx]
                    idr = 1.0/sqrt( dx*dx + dy*dy + dz*dz )
                    cc += idr
                else:
                    ''' self contribution from the image point''' 
                    dz = r[i+xx] + r[j+xx]    
                    cc += 1/(dz)
    
            C0[i  ]  += mud*cc 
        return 
    

    cpdef elastance10(self, double [:] C1, double [:] r, double [:] J0):
        cdef int i, j, Np=self.Np, xx=2*Np
        cdef double dx, dy, dz, idr, h2, hsq, idr2, A1
        cdef double vx, vy, vz, ii , cc=1 ## cc is the concentration constant
        cdef double mud = J0[0]/(4*PI*self.D)

        for i in prange(Np, nogil=True):
            vx=0; vy=0; vz=0;
            for j in range(Np):
                if i!=j:
                    dx = r[i]    - r[j]
                    dy = r[i+Np]  - r[j+Np]
                    dz = r[i+xx]  - r[j+xx]
                    idr = 1.0/sqrt(dx*dx + dy*dy + dz*dz)
                    A1 = idr*idr*idr
                    vx += A1*dx
                    vy += A1*dy
                    vz += A1*dz

                    ###contributions from the image 
                    dz = r[i+xx] + r[j+xx]
                    idr = 1.0/sqrt( dx*dx + dy*dy + dz*dz )
                    A1  = idr*idr*idr
                    vx += A1*dx
                    vy += A1*dy
                    vz += A1*dz
                else:
                    ''' self contribution from the image point''' 
                    dz = r[i+xx] + r[j+xx]    
                    vz += 1/(dz*dz)
    
            C1[i  ]  += mud*vx 
            C1[i+Np] += mud*vy
            C1[i+xx] += mud*vz
        return 


    cpdef elastance11(self, double [:] C1, double [:] r, double [:] J1):
        cdef int i, j, Np=self.Np, xx=2*Np
        cdef double dx, dy, dz, idr, h2, hsq, idr2, A1, B1
        cdef double vx, vy, vz, ii , cc=1 ## cc is the concentration constant
        cdef double mud = 5/(PI*PI**self.a)
 
        for i in prange(Np, nogil=True):
            vx=0; vy=0; vz=0;
            for j in range(Np):
                if i!=j:
                    dx = r[i]    - r[j]
                    dy = r[i+Np] - r[j+Np]
                    dz = r[i+xx] - r[j+xx] 
                    idr = 1.0/sqrt( dx*dx + dy*dy + dz*dz)
                    A1 = idr*idr*idr
                    B1  = 3*(J1[j]*dx + J1[j+Np]*dy + J1[j+xx]*dz)*idr*idr
                    vx += A1*(J1[j]    - B1*dx)
                    vy += A1*(J1[j+Np] - B1*dy)
                    vz += A1*(J1[j+xx] - B1*dz)

                    ###contributions from the image 
                    dz = r[i+xx]  + r[j+xx]
                    idr = 1.0/sqrt( dx*dx + dy*dy + dz*dz)
                    A1 = idr*idr*idr
                    B1  = 3*(J1[j]*dx + J1[j+Np]*dy + J1[j+xx]*dz)*idr*idr
                    vx += A1*(J1[j]    - B1*dx)
                    vy += A1*(J1[j+Np] - B1*dy)
                    vz += A1*(J1[j+xx] - B1*dz)

                else:
                   ''' self contribution from the image point'''
                   dz = 2*r[i+xx] ; A1=1/(dz*dz*dz)
                   vx += A1*(   J1[j]    )
                   vy += A1*(   J1[j+Np] )
                   vy += A1*(-2*J1[j+xx] )

            C1[i  ]  += mud*J1[i  ]  + mud*vx
            C1[i+Np] += mud*J1[i+Np] + mud*vy
            C1[i+xx] += mud*J1[i+xx] + mud*vz
        return




@cython.boundscheck(False)
@cython.cdivision(True)
@cython.nonecheck(False)
@cython.wraparound(False)
cdef class Field:
    """
    Phoretic field at given points
    
    ...

    Parameters
    ----------
    radius: float
        Radius of the particles.    
    particles: int
        Number of particles 
    phoreticConstant: float
        PhoreticConstant of the fluid 
    gridpoints: int 
        Number of grid points
    """

    def __init__(self, radius=1, particles=1, phoreticConstant=1, gridpoints=32):
        self.a  = radius
        self.Np = particles
        self.Nt = gridpoints
        self.D  = phoreticConstant


    cpdef phoreticField0(self, double [:] c, double [:] rt, double [:] r, double [:] J0):
        cdef int Np = self.Np, Nt = self.Nt, xx = 2*Np
        cdef int i, j
        cdef double dx, dy, dz, idr, idr3, cc, mu1=J0[0]/(4*PI*self.D)
 
        for i in prange(Nt, nogil=True):
            cc=0
            for j in range(Np):
                dx = rt[i]      - r[j]
                dy = rt[i+Nt]   - r[j+Np]
                dz = rt[i+2*Nt] - r[j+xx] 
                idr = 1.0/sqrt( dx*dx + dy*dy + dz*dz )
                cc +=idr
                
                ##contributions from the image 
                dz  = rt[i+2*Nt] + r[j+xx]        
                idr = 1.0/sqrt( dx*dx + dy*dy + dz*dz )
                cc  += idr

            c[i] += mu1*cc
        return 
    
    
    cpdef phoreticField1(self, double [:] c, double [:] rt, double [:] r, double [:] J1):
        cdef int Np = self.Np, Nt = self.Nt, xx = 2*Np
        cdef int i, j
        cdef double dx, dy, dz, idr, idr3, cc, mu1=1.0/(4*PI*self.D)
 
        for i in prange(Nt, nogil=True):
            cc=0
            for j in range(Np):
                dx = rt[i]      - r[j]
                dy = rt[i+Nt]   - r[j+Np]
                dz = rt[i+2*Nt] - r[j+2*Np] 

                idr  = 1.0/sqrt( dx*dx + dy*dy + dz*dz )
                idr3 = idr*idr*idr 

                cc  += idr3*(dx*J1[j] + dy*J1[j+Np] + dz*J1[j+2*Np])
                
                ##contributions from the image 
                dz   = rt[i+2*Nt] + r[j+xx]        
                idr  = 1.0/sqrt( dx*dx + dy*dy + dz*dz )
                idr3 = idr*idr*idr 

                cc  += idr3*(dx*J1[j] + dy*J1[j+Np] + dz*J1[j+2*Np])

            c[i] += mu1*cc
        return 
    
    
    cpdef phoreticField2(self, double [:] c, double [:] rt, double [:] r, double [:] J2):
        cdef int Np = self.Np, Nt = self.Nt, xx = 2*Np
        cdef int i, j
        cdef double dx, dy, dz, idr, idr5, cc, mu1=1.0/(4*PI*self.D), jxx, jyy, jxy, jyz, jxz
 
        for i in prange(Nt, nogil=True):
            cc=0
            for j in range(Np):
                jxx = J2[j]
                jyy = J2[j+Np]
                jxy = J2[j+2*Np]
                jxz = J2[j+3*Np]
                jyz = J2[j+4*Np]

                dx = rt[i]      - r[j]
                dy = rt[i+Nt]   - r[j+Np]
                dz = rt[i+2*Nt] - r[j+2*Np] 

                idr  = 1.0/sqrt( dx*dx + dy*dy + dz*dz )
                idr5 = idr*idr*idr*idr*idr 

                cc  += jxx*dx*dx + jyy*dy*dy + (-jxx-jyy)*dz*dz
                cc  +=(jxy*dx*dy + jxz*dx*dz+ jyz*dy*dz)*2
                
                ##contributions from the image 
                dz   = rt[i+2*Nt] + r[j+xx]        
                idr  = 1.0/sqrt( dx*dx + dy*dy + dz*dz )
                idr  = 1.0/sqrt( dx*dx + dy*dy + dz*dz )
                idr5 = idr*idr*idr*idr*idr 

                cc  += jxx*dx*dx + jyy*dy*dy + (-jxx-jyy)*dz*dz
                cc  +=(jxy*dx*dy + jxz*dx*dz+ jyz*dy*dz)*2

            c[i] += mu1*cc
        return 
