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
cdef class Rbm:

    def __init__(self, radius=1, particles=1, viscosity=1.0):
        self.a   = radius
        self.Np  = particles
        self.eta = viscosity
        self.mu  = 1.0/(6*PI*self.eta*self.a)

        self.Mobility = np.zeros( (3*self.Np, 3*self.Np), dtype=np.float64)

    cpdef mobilityTT(self, double [:] v, double [:] r, double [:] F):
        cdef int Np  = self.Np, i, j, xx=2*Np
        cdef double dx, dy, dz, idr, idr2, vx, vy, vz, vv1, vv2, aa = (2.0*self.a*self.a)/3.0 
        cdef double mu = 1.0/(6*PI*self.eta*self.a), mu1 = mu*self.a*0.75       
        
        for i in prange(Np, nogil=True):
            vx=0; vy=0;   vz=0;
            for j in range(Np):
                if i != j:
                    dx = r[i]    - r[j]
                    dy = r[i+Np] - r[j+Np]
                    dz = r[i+xx] - r[j+xx] 
                    idr = 1.0/sqrt( dx*dx + dy*dy + dz*dz )
                    idr2 = idr*idr
                    
                    vv1 = (1+aa*idr2)*idr 
                    vv2 = (1-3*aa*idr2)*( F[j]*dx + F[j+Np]*dy + F[j+xx]*dz )*idr2*idr
                    vx += vv1*F[j]    + vv2*dx 
                    vy += vv1*F[j+Np] + vv2*dy 
                    vz += vv1*F[j+xx] + vv2*dz 

            v[i]    += mu*F[i]    + mu1*vx
            v[i+Np] += mu*F[i+Np] + mu1*vy
            v[i+xx] += mu*F[i+xx] + mu1*vz
        return 
               
   
    cpdef mobilityTR(self, double [:] v, double [:] r, double [:] T):
        cdef int Np = self.Np, i, j, xx=2*Np 
        cdef double dx, dy, dz, idr, idr3, vx, vy, vz
        cdef double mu1 = 1.0/(8*PI*self.eta)       
        
        for i in prange(Np, nogil=True):
            vx=0; vy=0;   vz=0;
            for j in range(Np):
                if i != j:
                    dx = r[i]    - r[j]
                    dy = r[i+Np] - r[j+Np]
                    dz = r[i+xx] - r[j+xx] 
                    idr = 1.0/sqrt( dx*dx + dy*dy + dz*dz )
                    idr3 = idr*idr*idr
                    vx += (dy*T[j+xx] -T[j+Np]*dz )*idr3
                    vy += (dz*T[j]    -T[j+xx]*dx )*idr3
                    vz += (dx*T[j+Np] -T[j]   *dy )*idr3

            v[i]    += mu1*vx
            v[i+Np] += mu1*vy
            v[i+xx] += mu1*vz
        return 
    
    
    cpdef propulsionT2s(self, double [:] v, double [:] r, double [:] S):
        cdef int Np = self.Np, i, j, xx=2*Np, xx1=3*Np, xx2=4*Np
        cdef double dx, dy, dz, dr, idr,  idr3
        cdef double aa=(self.a*self.a*8.0)/3.0, vv1, vv2, aidr2
        cdef double vx, vy, vz, 
        cdef double sxx, sxy, sxz, syz, syy, srr, srx, sry, srz, mus = (28.0*self.a**3)/24 
 
        for i in prange(Np, nogil=True):
            vx=0; vy=0;   vz=0;
            for j in range(Np):
                if i != j:
                    sxx = S[j]
                    syy = S[j+Np]
                    sxy = S[j+xx]
                    sxz = S[j+xx1]
                    syz = S[j+xx2]
                    dx = r[i]    - r[j]
                    dy = r[i+Np] - r[j+Np]
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
            
            v[i]   += vx*mus
            v[i+Np]+= vy*mus
            v[i+xx]+= vz*mus

        return 


    cpdef propulsionT3t(self, double [:] v, double [:] r, double [:] D):
        cdef int Np = self.Np, i, j, xx=2*Np  
        cdef double dx, dy, dz, idr, idr3, Ddotidr, vx, vy, vz, mud = 3.0*self.a*self.a*self.a/5, mud1 = -1.0*(self.a**5)/10
 
        for i in prange(Np, nogil=True):
            vx=0; vy=0;   vz=0; 
            for j in range(Np):
                if i != j: 
                    dx = r[ i]   - r[j]
                    dy = r[i+Np] - r[j+Np]
                    dz = r[i+xx] - r[j+xx] 
                    idr = 1.0/sqrt( dx*dx + dy*dy + dz*dz )
                    idr3 = idr*idr*idr 
                    Ddotidr = (D[j]*dx + D[j+Np]*dy + D[j+xx]*dz)*idr*idr

                    vx += (D[j]    - 3.0*Ddotidr*dx )*idr3
                    vy += (D[j+Np] - 3.0*Ddotidr*dy )*idr3
                    vz += (D[j+xx] - 3.0*Ddotidr*dz )*idr3
            
            v[i]   += mud1*vx
            v[i+Np]+= mud1*vy
            v[i+xx]+= mud1*vz
        return 


    cpdef propulsionT3a(self, double [:] v, double [:] r, double [:] V):
        cdef int Np = self.Np, i, j 
        cdef double dx, dy, dz, idr, idr5, vxx, vyy, vxy, vxz, vyz, vrx, vry, vrz
 
        for i in prange(Np, nogil=True):
            for j in range(Np):
                if i != j:
                    vxx = V[j]
                    vyy = V[j+Np]
                    vxy = V[j+2*Np]
                    vxz = V[j+3*Np]
                    vyz = V[j+4*Np]
                    dx = r[i]      - r[j]
                    dy = r[i+Np]   - r[j+Np]
                    dz = r[i+2*Np] - r[j+2*Np] 
                    idr = 1.0/sqrt( dx*dx + dy*dy + dz*dz)
                    idr5 = idr*idr*idr*idr*idr
                    vrx = vxx*dx +  vxy*dy + vxz*dz  
                    vry = vxy*dx +  vyy*dy + vyz*dz  
                    vrz = vxz*dx +  vyz*dy - (vxx+vyy)*dz 

                    v[i]      -= 8*( dy*vrz - dz*vry )*idr5
                    v[i+Np]   -= 8*( dz*vrx - dx*vrz )*idr5
                    v[i+2*Np] -= 8*( dx*vry - dy*vrx )*idr5 
                else:
                    pass
        return


    cpdef propulsionT3s(self, double [:] v, double [:] r, double [:] G):
        cdef int Np = self.Np, i, j 
        cdef double dx, dy, dz, idr, idr5, idr7, aidr2, grrr, grrx, grry, grrz, gxxx, gyyy, gxxy, gxxz, gxyy, gxyz, gyyz
 
        for i in prange(Np, nogil=True):
             for j in range(Np):
                if i != j:
                    gxxx = G[j]
                    gyyy = G[j+Np]
                    gxxy = G[j+2*Np]
                    gxxz = G[j+3*Np]
                    gxyy = G[j+4*Np]
                    gxyz = G[j+5*Np]
                    gyyz = G[j+6*Np]
                    dx = r[i]      - r[j]
                    dy = r[i+Np]   - r[j+Np]
                    dz = r[i+2*Np] - r[j+2*Np] 
                    idr = 1.0/sqrt( dx*dx + dy*dy + dz*dz )
                    idr5 = idr*idr*idr*idr*idr      
                    idr7 = idr5*idr*idr     
                    aidr2 = (10.0/3)*self.a*self.a*idr*idr
                    
                    grrr = gxxx*dx*(dx*dx-3*dz*dz) + 3*gxxy*dy*(dx*dx-dz*dz) + gxxz*dz*(3*dx*dx-dz*dz) +\
                       3*gxyy*dx*(dy*dy-dz*dz) + 6*gxyz*dx*dy*dz + gyyy*dy*(dy*dy-3*dz*dz) +  gyyz*dz*(3*dy*dy-dz*dz) 
                    grrx = gxxx*(dx*dx-dz*dz) + gxyy*(dy*dy-dz*dz) +  2*gxxy*dx*dy + 2*gxxz*dx*dz  +  2*gxyz*dy*dz
                    grry = gxxy*(dx*dx-dz*dz) + gyyy*(dy*dy-dz*dz) +  2*gxyy*dx*dy + 2*gxyz*dx*dz  +  2*gyyz*dy*dz
                    grrz = gxxz*(dx*dx-dz*dz) + gyyz*(dy*dy-dz*dz) +  2*gxyz*dx*dy - 2*(gxxx+gxyy)*dx*dz  - 2*(gxxy+gyyy)*dy*dz
                  
                    v[i]      += 3*(1-(15.0/7)*aidr2)*grrx*idr5 - 15*(1-aidr2)*grrr*dx*idr7
                    v[i+Np]   += 3*(1-(15.0/7)*aidr2)*grry*idr5 - 15*(1-aidr2)*grrr*dy*idr7
                    v[i+2*Np] += 3*(1-(15.0/7)*aidr2)*grrz*idr5 - 15*(1-aidr2)*grrr*dz*idr7
                else:
                    pass 
        return


    cpdef propulsionT4a(self, double [:] v, double [:] r, double [:] M):
        cdef int Np = self.Np, i, j 
        cdef double dx, dy, dz, idr, idr7
        cdef double mrrx, mrry, mrrz, mxxx, myyy, mxxy, mxxz, mxyy, mxyz, myyz
 
        for i in prange(Np, nogil=True):
            for j in range(Np):
                if i != j:
                    mxxx = M[j]
                    myyy = M[j+Np]
                    mxxy = M[j+2*Np]
                    mxxz = M[j+3*Np]
                    mxyy = M[j+4*Np]
                    mxyz = M[j+5*Np]
                    myyz = M[j+6*Np]
                    dx = r[i]      - r[j]
                    dy = r[i+Np]   - r[j+Np]
                    dz = r[i+2*Np] - r[j+2*Np] 
                    idr = 1.0/sqrt( dx*dx + dy*dy + dz*dz )
                    idr7 = idr*idr*idr*idr*idr*idr*idr
                    mrrx = mxxx*(dx*dx-dz*dz) + mxyy*(dy*dy-dz*dz) +  2*mxxy*dx*dy + 2*mxxz*dx*dz  +  2*mxyz*dy*dz
                    mrry = mxxy*(dx*dx-dz*dz) + myyy*(dy*dy-dz*dz) +  2*mxyy*dx*dy + 2*mxyz*dx*dz  +  2*myyz*dy*dz
                    mrrz = mxxz*(dx*dx-dz*dz) + myyz*(dy*dy-dz*dz) +  2*mxyz*dx*dy - 2*(mxxx+mxyy)*dx*dz  - 2*(mxxy+myyy)*dy*dz
                    
                    v[i]      -= 6*( dy*mrrz - dz*mrry )*idr7
                    v[i+Np]   -= 6*( dz*mrrx - dx*mrrz )*idr7
                    v[i+2*Np] -= 6*( dx*mrry - dy*mrrx )*idr7
                else:
                    pass
        return


    ## Angular velocities
    cpdef mobilityRT(self, double [:] o, double [:] r, double [:] F):
        cdef int Np = self.Np, i, j, xx=2*Np 
        cdef double dx, dy, dz, idr, idr3, ox, oy, oz, mu1 = 1.0/(8*PI*self.eta)
 
        for i in prange(Np, nogil=True):
            ox=0;   oy=0;   oz=0;
            for j in range(Np):
                if i != j:
                    dx = r[i]    - r[j]
                    dy = r[i+Np] - r[j+Np]
                    dz = r[i+xx] - r[j+xx] 
                    idr = 1.0/sqrt( dx*dx + dy*dy + dz*dz )
                    idr3 = idr*idr*idr

                    ox += (F[j+Np]*dz - F[j+xx]*dy )*idr3
                    oy += (F[j+xx]*dx - F[j]   *dz )*idr3
                    oz += (F[j]   *dy - F[j+Np]*dx )*idr3
            o[i]    += mu1*ox
            o[i+Np] += mu1*oy
            o[i+xx] += mu1*oz
        return  

               
    cpdef mobilityRR(   self, double [:] o, double [:] r, double [:] T):
        cdef int Np = self.Np, i, j, xx=2*Np 
        cdef double dx, dy, dz, idr, idr3, Tdotidr, ox, oy, oz, mur = 1.0/(8*PI*self.eta*self.a**3),  mu1 = 1.0/(8*PI*self.eta) 
 
        for i in prange(Np, nogil=True):
            ox=0;   oy=0;   oz=0;
            for j in range(Np):
                if i != j:
                    dx = r[i]      - r[j]
                    dy = r[i+Np]   - r[j+Np]
                    dz = r[i+xx] - r[j+xx] 
                    idr = 1.0/sqrt( dx*dx + dy*dy + dz*dz )
                    idr3 = idr*idr*idr
                    Tdotidr = ( T[j]*dx + T[j+Np]*dy + T[j+xx]*dz )*idr*idr

                    ox += ( T[j]    - 3*Tdotidr*dx )*idr3
                    oy += ( T[j+Np] - 3*Tdotidr*dy )*idr3
                    oz += ( T[j+xx] - 3*Tdotidr*dz )*idr3
            
            o[i]    += mur*T[i]    - mu1*ox
            o[i+Np] += mur*T[i+Np] - mu1*oy
            o[i+xx] += mur*T[i+xx] - mu1*oz
        return  

    
    cpdef propulsionR2s(self, double [:] o, double [:] r, double [:] S):
        cdef int Np = self.Np, i, j, xx=2*Np 
        cdef double dx, dy, dz, idr, idr5, ox, oy, oz
        cdef double sxx, sxy, sxz, syz, syy, srr, srx, sry, srz, mus = (28.0*self.a*self.a*self.a)/24
 
        for i in prange(Np, nogil=True):
            ox=0;   oy=0;   oz=0;
            for j in range(Np):
                if i != j:
                    sxx = S[j]
                    syy = S[j+Np]
                    sxy = S[j+2*Np]
                    sxz = S[j+3*Np]
                    syz = S[j+4*Np]
                    dx = r[i]      - r[j]
                    dy = r[i+Np]   - r[j+Np]
                    dz = r[i+2*Np] - r[j+2*Np] 
                    idr = 1.0/sqrt( dx*dx + dy*dy + dz*dz )
                    idr5 = idr*idr*idr*idr*idr      
                    srx = sxx*dx +  sxy*dy + sxz*dz  
                    sry = sxy*dx +  syy*dy + syz*dz  
                    srz = sxz*dx +  syz*dy - (sxx+syy)*dz 

                    ox += 3*(sry*dz - srz*dy )*idr5
                    oy += 3*(srz*dx - srx*dz )*idr5
                    oz += 3*(srx*dy - sry*dx )*idr5
                    
            o[i]    += ox*mus
            o[i+Np] += oy*mus
            o[i+xx] += oz*mus
        return                 
    
    
    cpdef propulsionR3a(  self, double [:] o, double [:] r, double [:] V):
        cdef int Np = self.Np, i, j 
        cdef double dx, dy, dz, idr, idr2, idr5, vxx, vyy, vxy, vxz, vyz, vrr, vrx, vry, vrz
 
        for i in prange(Np, nogil=True):
             for j in range(Np):
                if i != j:
                    vxx = V[j]
                    vyy = V[j+Np]
                    vxy = V[j+2*Np]
                    vxz = V[j+3*Np]
                    vyz = V[j+4*Np]
                    dx = r[i]      - r[j]
                    dy = r[i+Np]   - r[j+Np]
                    dz = r[i+2*Np] - r[j+2*Np] 
                    idr = 1.0/sqrt( dx*dx + dy*dy + dz*dz )
                    idr5 = idr*idr*idr*idr*idr      
                    vrr = (vxx*(dx*dx-dz*dz) + vyy*(dy*dy-dz*dz) +  2*vxy*dx*dy + 2*vxz*dx*dz  +  2*vyz*dy*dz)*idr*idr
                    vrx = vxx*dx +  vxy*dy + vxz*dz  
                    vry = vxy*dx +  vyy*dy + vyz*dz  
                    vrz = vxz*dx +  vyz*dy - (vxx+vyy)*dz 

                    o[i]      +=  ( 32*vrx- 20*vrr*dx )*idr5
                    o[i+Np]   +=  ( 32*vry- 20*vrr*dy )*idr5
                    o[i+2*Np] +=  ( 32*vrz- 20*vrr*dz )*idr5
                else :
                    pass 
        return


    cpdef propulsionR3s(  self, double [:] o, double [:] r, double [:] G):
        cdef int Np = self.Np, i, j 
        cdef double dx, dy, dz, idr, idr7, grrx, grry, grrz, gxxx, gyyy, gxxy, gxxz, gxyy, gxyz, gyyz
 
        for i in prange(Np, nogil=True):
            for j in range(Np):
                if i != j:
                    gxxx = G[j]
                    gyyy = G[j+Np]
                    gxxy = G[j+2*Np]
                    gxxz = G[j+3*Np]
                    gxyy = G[j+4*Np]
                    gxyz = G[j+5*Np]
                    gyyz = G[j+6*Np]
                    dx = r[i]      - r[j]
                    dy = r[i+Np]   - r[j+Np]
                    dz = r[i+2*Np] - r[j+2*Np] 
                    idr = 1.0/sqrt( dx*dx + dy*dy + dz*dz )
                    idr7 = idr*idr*idr*idr*idr*idr*idr     
                    
                    grrx = gxxx*(dx*dx-dz*dz) + gxyy*(dy*dy-dz*dz) +  2*gxxy*dx*dy + 2*gxxz*dx*dz  +  2*gxyz*dy*dz
                    grry = gxxy*(dx*dx-dz*dz) + gyyy*(dy*dy-dz*dz) +  2*gxyy*dx*dy + 2*gxyz*dx*dz  +  2*gyyz*dy*dz
                    grrz = gxxz*(dx*dx-dz*dz) + gyyz*(dy*dy-dz*dz) +  2*gxyz*dx*dy - 2*(gxxx+gxyy)*dx*dz  - 2*(gxxy+gyyy)*dy*dz

                    o[i]      += 15*( dy*grrz - dz*grry )*idr7
                    o[i+Np]   += 15*( dz*grrx - dx*grrz )*idr7
                    o[i+2*Np] += 15*( dx*grry - dy*grrx )*idr7
                else :
                    pass
        return                 


    cpdef propulsionR4a(  self, double [:] o, double [:] r, double [:] M):
        cdef int Np = self.Np, i, j 
        cdef double dx, dy, dz, idr, idr7, idr9, mrrr, mrrx, mrry, mrrz, mxxx, myyy, mxxy, mxxz, mxyy, mxyz, myyz
 
        for i in prange(Np, nogil=True):
             for j in range(Np):
                if i != j:
                    mxxx = M[j]
                    myyy = M[j+Np]
                    mxxy = M[j+2*Np]
                    mxxz = M[j+3*Np]
                    mxyy = M[j+4*Np]
                    mxyz = M[j+5*Np]
                    myyz = M[j+6*Np]
                    dx = r[i]      - r[j]
                    dy = r[i+Np]   - r[j+Np]
                    dz = r[i+2*Np] - r[j+2*Np] 
                    idr = 1.0/sqrt( dx*dx + dy*dy + dz*dz )
                    idr7 = idr*idr*idr*idr*idr*idr*idr      
                    idr9 = idr7*idr*idr     
                    
                    mrrr = mxxx*dx*(dx*dx-3*dz*dz) + 3*mxxy*dy*(dx*dx-dz*dz) + mxxz*dz*(3*dx*dx-dz*dz) +\
                       3*mxyy*dx*(dy*dy-dz*dz) + 6*mxyz*dx*dy*dz + myyy*dy*(dy*dy-3*dz*dz) +  myyz*dz*(3*dy*dy-dz*dz) 
                    mrrx = mxxx*(dx*dx-dz*dz) + mxyy*(dy*dy-dz*dz) +  2*mxxy*dx*dy + 2*mxxz*dx*dz  +  2*mxyz*dy*dz
                    mrry = mxxy*(dx*dx-dz*dz) + myyy*(dy*dy-dz*dz) +  2*mxyy*dx*dy + 2*mxyz*dx*dz  +  2*myyz*dy*dz
                    mrrz = mxxz*(dx*dx-dz*dz) + myyz*(dy*dy-dz*dz) +  2*mxyz*dx*dy - 2*(mxxx+mxyy)*dx*dz  - 2*(mxxy+myyy)*dy*dz
                  
                    o[i]      += 21*mrrr*dx*idr9 - 9*mrrx*idr7  
                    o[i+Np]   += 21*mrrr*dy*idr9 - 9*mrry*idr7  
                    o[i+2*Np] += 21*mrrr*dz*idr9 - 9*mrrz*idr7  
                else:
                    pass 
        return


    cpdef calcNoiseMuTT(self, double [:] v, double [:] r):
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
                else:
                    # one-body mobility
                    M[i,    j   ] = mm
                    M[i+Np, j+Np] = mm
                    M[i+xx, j+xx] = mm
                    M[i,    j+Np] = 0
                    M[i,    j+xx] = 0
                    M[i+Np, j+xx] = 0


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

        #'''to check the one-body solution near a plane wall'''
        #muPerp = mu*(1 - 9*self.a/(8*r[2]) + 0.5*(self.a/r[2])**3 ),
        #muParl = mu*(1 - 9*self.a/(16*r[2]) + 0.125*(self.a/r[2])**3 )
        #print self.Mobility/sqrt(2), muParl, muPerp  # note that there is a factor of sqrt(2)
        return


## Flow at given points
@cython.wraparound(False)
@cython.boundscheck(False)
@cython.cdivision(True)
@cython.nonecheck(False)
cdef class Flow:
    def __init__(self, radius=1, particles=1, viscosity=1, gridpoints=32):
        self.a  = radius
        self.Np = particles
        self.Nt = gridpoints
        self.eta= viscosity

    cpdef flowField1s(self, double [:] vv, double [:] rt, double [:] r, double [:] F):
        cdef int Np = self.Np,  Nt = self.Nt
        cdef int i, ii, xx = 2*Np
        cdef double dx, dy, dz, idr, idr2, vv1, vv2, vx, vy, vz, mu1 = 1/(8*PI*self.eta), aa = self.a*self.a/3.0
        for i in prange(Nt, nogil=True):
            vx = 0.0; vy = 0.0; vz = 0.0;
            for ii in range(Np):
                
                dx = rt[i]      - r[ii]
                dy = rt[i+Nt]   - r[ii+Np]
                dz = rt[i+2*Nt] - r[ii+xx] 
                idr = 1.0/sqrt( dx*dx + dy*dy + dz*dz )
                idr2= idr*idr
                vv1 = (1+aa*idr2)*idr 
                vv2 = (1-3*aa*idr2)*( F[ii]*dx + F[ii+Np]*dy + F[ii+xx]*dz )*idr2*idr
                
                vx += vv1*F[ii]     + vv2*dx
                vy += vv1*F[ii+ Np] + vv2*dy
                vz += vv1*F[ii+ xx] + vv2*dz
            
            vv[i]         += vx*mu1
            vv[i +   Nt]  += vy*mu1
            vv[i +  2*Nt] += vz*mu1
        return 
               
    cpdef flowField2a(self, double [:] vv, double [:] rt, double [:] r, double [:] T):
        cdef int Np = self.Np, Nt = self.Nt
        cdef int i, ii, xx = 2*Np
        cdef double dx, dy, dz, idr, idr3, vx, vy, vz, mur1 = 1.0/(8*PI*self.eta)
        for i in prange(Nt, nogil=True):
            vx = 0.0; vy = 0.0; vz = 0.0;
            for ii in range(Np):
                dx = rt[i]      - r[ii]
                dy = rt[i+Nt]   - r[ii+Np]
                dz = rt[i+2*Nt] - r[ii+xx] 
                idr = 1.0/sqrt( dx*dx + dy*dy + dz*dz )
                idr3 = idr*idr*idr
          
                vx += ( dy*T[ii+xx] - dz*T[ii+Np])*idr3
                vy += ( dz*T[ii]    - dx*T[ii+xx])*idr3 
                vz += ( dx*T[ii+Np] - dy*T[ii]   )*idr3

            vv[i  ]      += vx*mur1
            vv[i + Nt]   += vy*mur1
            vv[i + 2*Nt] += vz*mur1
        return  

    cpdef flowField2s(self, double [:] vv, double [:] rt, double [:] r, double [:] S):
        cdef int Np = self.Np, Nt = self.Nt
        cdef int i, ii, xx= 2*Np, xx1= 3*Np, xx2 = 4*Np
        cdef double dx, dy, dz, idr, idr3, aidr2, sxx, syy, sxy, sxz, syz, srr, srx, sry, srz
        cdef double aa = self.a**2, vv1, vv2, vx, vy, vz, mus = (28.0*self.a**3)/24 
        for i in prange(Nt, nogil=True):
            vx = 0.0;vy = 0.0; vz = 0.0;
            for ii in range(Np):
                sxx = S[ii]
                syy = S[ii+Np]
                sxy = S[ii+xx]
                sxz = S[ii+xx1]
                syz = S[ii+xx2]
                
                dx = rt[i]      - r[ii]
                dy = rt[i+Nt]   - r[ii+Np]
                dz = rt[i+2*Nt] - r[ii+xx] 
                
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
                 
            vv[i]      +=  vx*mus
            vv[i+Nt]   +=  vy*mus
            vv[i+2*Nt] +=  vz*mus
        return


    cpdef flowField3t(self, double [:] vv, double [:] rt, double [:] r, double [:] D):
        cdef int Np = self.Np, Nt = self.Nt
        cdef  int i, ii 
        cdef double dx, dy, dz, idr, idr3, Ddotidr, vx, vy, vz,mud1 = -1.0*(self.a**5)/10
 
        for i in prange(Nt, nogil=True):
            vx =0.0; vy = 0.0; vz =0.0;
            for ii in range(Np):
                dx = rt[i]      - r[ii]
                dy = rt[i+Nt]   - r[ii+Np]
                dz = rt[i+2*Nt] - r[ii+2*Np] 
                idr = 1.0/sqrt( dx*dx + dy*dy + dz*dz )
                idr3 = idr*idr*idr 
                Ddotidr = (D[ii]*dx + D[ii+Np]*dy + D[ii+2*Np]*dz)*idr*idr

                vx += (D[ii]      - 3.0*Ddotidr*dx )*idr3
                vy += (D[ii+Np]   - 3.0*Ddotidr*dy )*idr3
                vz += (D[ii+2*Np] - 3.0*Ddotidr*dz )*idr3
        
            vv[i]      += vx*mud1
            vv[i+Nt]   += vy*mud1
            vv[i+2*Nt] += vz*mud1
        
        return 


    cpdef flowField3s(self, double [:] vv, double [:] rt, double [:] r, double [:] G):
        cdef int Np = self.Np, Nt = self.Nt
        cdef int i, ii, 
        cdef double dx, dy, dz, idr, idr5, idr7
        cdef double aidr2, grrr, grrx, grry, grrz, gxxx, gyyy, gxxy, gxxz, gxyy, gxyz, gyyz
        for i in prange(Nt, nogil=True):
            for ii in range(Np):
                gxxx = G[ii]
                gyyy = G[ii+Np]
                gxxy = G[ii+2*Np]
                gxxz = G[ii+3*Np]
                gxyy = G[ii+4*Np]
                gxyz = G[ii+5*Np]
                gyyz = G[ii+6*Np]
                dx = rt[i]      - r[ii]
                dy = rt[i+Nt]   - r[ii+Np]
                dz = rt[i+2*Nt] - r[ii+2*Np] 
                idr = 1.0/sqrt( dx*dx + dy*dy + dz*dz )
                idr5 = idr*idr*idr*idr*idr      
                idr7 = idr5*idr*idr     
                aidr2 = self.a*self.a*idr*idr
                
                grrr = gxxx*dx*(dx*dx-3*dz*dz) + 3*gxxy*dy*(dx*dx-dz*dz) + gxxz*dz*(3*dx*dx-dz*dz) +\
                       3*gxyy*dx*(dy*dy-dz*dz) + 6*gxyz*dx*dy*dz + gyyy*dy*(dy*dy-3*dz*dz) +  gyyz*dz*(3*dy*dy-dz*dz) 
                grrx = gxxx*(dx*dx-dz*dz) + gxyy*(dy*dy-dz*dz) +  2*gxxy*dx*dy + 2*gxxz*dx*dz  +  2*gxyz*dy*dz
                grry = gxxy*(dx*dx-dz*dz) + gyyy*(dy*dy-dz*dz) +  2*gxyy*dx*dy + 2*gxyz*dx*dz  +  2*gyyz*dy*dz
                grrz = gxxz*(dx*dx-dz*dz) + gyyz*(dy*dy-dz*dz) +  2*gxyz*dx*dy - 2*(gxxx+gxyy)*dx*dz  - 2*(gxxy+gyyy)*dy*dz
             
                vv[i]      += 3*(1-(15.0/7)*aidr2)*grrx*idr5 - 15*(1-aidr2)*grrr*dx*idr7
                vv[i+Nt]   += 3*(1-(15.0/7)*aidr2)*grry*idr5 - 15*(1-aidr2)*grrr*dy*idr7
                vv[i+2*Nt] += 3*(1-(15.0/7)*aidr2)*grrz*idr5 - 15*(1-aidr2)*grrr*dz*idr7
        return


    cpdef flowField3a(self, double [:] vv, double [:] rt, double [:] r, double [:] V):
        cdef int Np = self.Np, Nt = self.Nt
        cdef int i, ii
        cdef double dx, dy, dz, idr, idr5, vxx, vyy, vxy, vxz, vyz, vrx, vry, vrz
 
        for i in prange(Nt, nogil=True):
            for ii in range(Np):
                vxx = V[ii]
                vyy = V[ii+Np]
                vxy = V[ii+2*Np]
                vxz = V[ii+3*Np]
                vyz = V[ii+4*Np]
                dx = rt[i]      - r[ii]
                dy = rt[i+Nt]   - r[ii+Np]
                dz = rt[i+2*Nt] - r[ii+2*Np] 
                idr = 1.0/sqrt( dx*dx + dy*dy + dz*dz)
                idr5 = idr*idr*idr*idr*idr
                vrx = vxx*dx +  vxy*dy + vxz*dz  
                vry = vxy*dx +  vyy*dy + vyz*dz  
                vrz = vxz*dx +  vyz*dy - (vxx+vyy)*dz 

                vv[i]      += 8*( dy*vrz - dz*vry )*idr5
                vv[i+Nt]   += 8*( dz*vrx - dx*vrz )*idr5
                vv[i+2*Nt] += 8*( dx*vry - dy*vrx )*idr5 
        return 


    cpdef flowField4a(self, double [:] vv, double [:] rt, double [:] r, double [:] M):
        cdef int Np = self.Np, Nt = self.Nt
        cdef int i, ii
        cdef double dx, dy, dz, idr, idr7
        cdef double mrrx, mrry, mrrz, mxxx, myyy, mxxy, mxxz, mxyy, mxyz, myyz
 
        for i in prange(Nt, nogil=True):
            for ii in range(Np):
                mxxx = M[ii]
                myyy = M[ii+Np]
                mxxy = M[ii+2*Np]
                mxxz = M[ii+3*Np]
                mxyy = M[ii+4*Np]
                mxyz = M[ii+5*Np]
                myyz = M[ii+6*Np]
                dx = rt[i]      - r[ii]
                dy = rt[i+Nt]   - r[ii+Np]
                dz = rt[i+2*Nt] - r[ii+2*Np] 
                idr = 1.0/sqrt( dx*dx + dy*dy + dz*dz )
                idr7 = idr*idr*idr*idr*idr*idr*idr
                mrrx = mxxx*(dx*dx-dz*dz) + mxyy*(dy*dy-dz*dz) +  2*mxxy*dx*dy + 2*mxxz*dx*dz  +  2*mxyz*dy*dz
                mrry = mxxy*(dx*dx-dz*dz) + myyy*(dy*dy-dz*dz) +  2*mxyy*dx*dy + 2*mxyz*dx*dz  +  2*myyz*dy*dz
                mrrz = mxxz*(dx*dx-dz*dz) + myyz*(dy*dy-dz*dz) +  2*mxyz*dx*dy - 2*(mxxx+mxyy)*dx*dz  - 2*(mxxy+myyy)*dy*dz
                
                vv[i]      -= 6*( dy*mrrz - dz*mrry )*idr7
                vv[i+Nt]   -= 6*( dz*mrrx - dx*mrrz )*idr7
                vv[i+2*Nt] -= 6*( dx*mrry - dy*mrrx )*idr7
        return

