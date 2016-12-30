cimport cython
from libc.math cimport sqrt, pow
from cython.parallel import prange

@cython.boundscheck(False)
@cython.cdivision(True)
@cython.nonecheck(False)
@cython.wraparound(False)
cdef class Forces:
    def __init__(self, Np_):
        self.Np = Np_
        pass

    cpdef lennardJones(self, double [:] F, double [:] r, double ljeps=0.01, double ljrmin=3):
        cdef int Np = self.Np, i, j, xx = 2*Np
        cdef double dx, dy, dz, dr2, idr, rminbyr, fac, fx, fy, fz

        for i in prange(Np,nogil=True):
            fx = 0.0; fy = 0.0; fz = 0.0;
            for j in range(Np):
                dx = r[i   ] - r[j   ]
                dy = r[i+Np] - r[j+Np]
                dz = r[i+xx] - r[j+xx] 
                dr2 = dx*dx + dy*dy + dz*dz
                if i != j and dr2 < (ljrmin*ljrmin):
                    idr     = 1.0/sqrt(dr2)
                    rminbyr = ljrmin*idr 
                    fac   = ljeps*(pow(rminbyr, 12) - pow(rminbyr, 6))*idr*idr
                
                    fx += fac*dx 
                    fy += fac*dy
                    fz += fac*dz
            
            F[i]    += fx 
            F[i+Np] += fy 
            F[i+xx] += fz 
        return  

    


    cpdef harmonicConfinement(self, double [:] F, double [:] r, double cn):
        cdef int Np = self.Np, i
        for i in prange(3*Np, nogil=True):
            F[i] -= cn*r[i]
        return 
    

    cpdef opticalConfinement(self, double [:] F, double [:] r, double [:] r0, double [:] k):
        cdef int Np = self.Np, i, i1, xx = 2*Np
        for i in prange(Np, nogil=True):
            F[i  ]  -= k[i]*(r[i]    - r0[i]  ) 
            F[i+Np] -= k[i]*(r[i+Np] - r0[i+Np]) 
            F[i+xx] -= k[i]*(r[i+xx] - r0[i+xx]) 
        return 
    
    
    cpdef sedimentation(self, double [:] F, double g):
        cdef int Np = self.Np, i, xx = 2*Np 
        for i in prange(Np, nogil=True):
#            F[i   ] +=  0
#            F[i+Np] +=  0
            F[i+xx] +=  g
        return 

   
