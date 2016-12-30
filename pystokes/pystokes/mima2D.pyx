cimport cython
from libc.math cimport sqrt, exp, pow, erfc, sin, cos, int
from cython.parallel import prange
import numpy as np
cimport numpy as np
cdef double PI = 3.14159265359
DTYPE   = np.float

cdef extern from "complex.h" nogil:
    double complex cexp(double complex x)


@cython.wraparound(False)
@cython.boundscheck(False)
@cython.nonecheck(False)
cdef class Flow:
    def __init__(self, a_, Np_, Lx_, Ly_, Nx_, Ny_):
        self.a = a_
        self.Np = Np_
        self.Lx = Lx_
        self.Ly = Ly_
        self.Nx = Nx_
        self.Ny = Ny_

        self.facx = 2*PI/Nx_
        self.facy = 2*PI/Ny_
        
        self.fx0 = np.zeros((Nx_, Ny_), dtype=DTYPE) 
        self.fk0 = np.zeros((Nx_, Ny_), dtype=np.complex128)
        self.fkx = np.zeros((Nx_, Ny_), dtype=np.complex128) 
        self.fky = np.zeros((Nx_, Ny_), dtype=np.complex128)
        self.vkx = np.zeros((Nx_, Ny_), dtype=np.complex128) 
        self.vky = np.zeros((Nx_, Ny_), dtype=np.complex128)
       

    
    cpdef stokesletV(self, np.ndarray v, double [:] r, double [:] F, double sigma=1, int NN=20):
        cdef:
            int i, nx, ny , x, y, Nx, Ny, Np, jx, jy
            double arg, facx, facy, kx, ky,
            double scale = pow(2*PI*sigma*sigma, 2/2) 
            double kdotdr, a2, k2
        Nx, Ny = self.Nx, self.Ny 
        facx, facy = self.facx, self.facy
        Np, a2 = self.Np, self.a*self.a/6.0
        
        self.fourierFk(sigma, scale)  # construct the fk0 
       
        cdef complex [:, :] fk0 = self.fk0
        cdef complex [:, :] fkx = self.fkx
        cdef complex [:, :] fky = self.fky

        for jy in prange(Ny, nogil=True): 
            for jx in range(Nx): 
                for i in range(0, 3*Np, 3):
                    kx = jx*facx if jx <= Nx / 2 else (-Nx+jx)*facx
                    ky = jy*facy if jy <= Ny / 2 else (-Ny+jy)*facy
                    kdotdr = kx*(r[i]-Nx/2) + ky*(r[i+1]-Ny/2)
                    k2 = kx*kx + ky*ky
                    
                    fkx[jy, jx] += fk0[jy, jx]*F[i]*cexp(1j*kdotdr)*(1-a2*k2)
                    fky[jy, jx] += fk0[jy, jx]*F[i+1]*cexp(1j*kdotdr)*(1-a2*k2)
        
        self.solve( v, np.concatenate(( self.fkx.ravel(), self.fky.ravel() )) )
        return

    cpdef stressletV(self, np.ndarray v, double [:] r, double [:] S, double sigma=1, int NN=20):
        cdef:
            int i, nx, ny , x, y, Nx, Ny, Np, jx, jy
            double arg, facx, facy, kx, ky, skk, skx, sky, 
            double scale = pow(2*PI*sigma*sigma, 2/2) 
            double kdotdr, a2, k2
        Nx, Ny = self.Nx, self.Ny 
        facx, facy = self.facx, self.facy
        Np, a2 = self.Np, self.a*self.a*4.0/15
        
        self.fourierFk(sigma, scale)  # construct the fk0 
       
        cdef complex [:, :] fk0 = self.fk0
        cdef complex [:, :] fkx = self.fkx
        cdef complex [:, :] fky = self.fky

        for jy in prange(Ny, nogil=True): 
            for jx in range(Nx): 
                for i in range(0, 3*Np, 3):
                    kx = jx*facx if jx <= Nx / 2 else (-Nx+jx)*facx
                    ky = jy*facy if jy <= Ny / 2 else (-Ny+jy)*facy
                    kdotdr = kx*(r[i]-Nx/2) + ky*(r[i+1]-Ny/2)
                    k2 = kx*kx + ky*ky
                    skx = S[i]*kx    + S[i+1]*ky
                    sky = S[i+1]*kx - S[i]*ky
                    
                    fkx[jy, jx] += -1j*fk0[jy, jx]*skx*cexp(1j*kdotdr)*(1-a2*k2)
                    fky[jy, jx] += -1j*fk0[jy, jx]*sky*cexp(1j*kdotdr)*(1-a2*k2)
        
        self.solve( v, np.concatenate(( self.fkx.ravel(), self.fky.ravel() )) )
        return


    cpdef potDipoleV(self, np.ndarray v, double [:] r, double [:] D, double sigma=3, int NN=20):
        cdef:
            int i, nx, ny , x, y, Nx, Ny, Np, jx, jy
            double arg, facx, facy, kx, ky, skk, skx, sky, 
            double scale = pow(2*PI*sigma*sigma, 2/2) 
            double kdotdr, a2, k2
        Nx, Ny = self.Nx, self.Ny 
        facx, facy = self.facx, self.facy
        Np, a2 = self.Np, self.a*self.a*4.0/15
        
        self.fourierFk(sigma, scale)  # construct the fk0 
       
        cdef complex [:, :] fk0 = self.fk0
        cdef complex [:, :] fkx = self.fkx
        cdef complex [:, :] fky = self.fky

        for jy in prange(Ny, nogil=True): 
            for jx in range(Nx): 
                for i in range(0, 3*Np, 3):
                    kx = jx*facx if jx <= Nx / 2 else (-Nx+jx)*facx
                    ky = jy*facy if jy <= Ny / 2 else (-Ny+jy)*facy
                    kdotdr = kx*(r[i]-Nx/2) + ky*(r[i+1]-Ny/2)
                    k2 = kx*kx + ky*ky
                    
                    fkx[jy, jx] += fk0[jy, jx]*D[i]   *cexp(1j*kdotdr)
                    fky[jy, jx] += fk0[jy, jx]*D[i+1]*cexp(1j*kdotdr)
        
        self.solve( v, np.concatenate(( self.fkx.ravel(), self.fky.ravel() )) )
        return

    
    cpdef septletV(self, np.ndarray v, double [:] r, double [:] p, double sigma=3, int NN=20):
        cdef:
            int i, nx, ny , x, y, Nx, Ny, Np, jx, jy
            double arg, facx, facy, kx, ky, skk, skx, sky, 
            double scale = pow(2*PI*sigma*sigma, 2/2) 
            double kdotdr, pdotk, a2, k2, sepx, sepy, pdotk2
        Nx, Ny = self.Nx, self.Ny 
        facx, facy = self.facx, self.facy
        Np, a2 = self.Np, self.a*self.a*5.0/17
        
        self.fourierFk(sigma, scale)  # construct the fk0 
       
        cdef complex [:, :] fk0 = self.fk0
        cdef complex [:, :] fkx = self.fkx
        cdef complex [:, :] fky = self.fky

        for jy in prange(Ny, nogil=True): 
            for jx in range(Nx): 
                for i in range(0, 3*Np, 3):
                    kx = jx*facx if jx <= Nx / 2 else (-Nx+jx)*facx
                    ky = jy*facy if jy <= Ny / 2 else (-Ny+jy)*facy
                    kdotdr = kx*(r[i]-Nx/2) + ky*(r[i+1]-Ny/2)
                    k2 = kx*kx + ky*ky
                    pdotk  = p[i]*kx + p[i+1]*ky
                    pdotk2 = pdotk*pdotk
                    sepx = p[i]*pdotk2 - (1/5)*(  k2*p[i] + 2*pdotk*kx   )
                    sepy = p[i + Np]*pdotk2 - (1/5)*(  k2*p[i +Np] + 2*pdotk*ky   )
                    
                    fkx[jy, jx] += fk0[jy, jx]*cexp(-1j * kdotdr)*sepx*(1-a2*k2) 
                    fky[jy, jx] += fk0[jy, jx]*cexp(-1j * kdotdr)*sepy*(1-a2*k2)
        
        self.solve( v, np.concatenate(( self.fkx.ravel(), self.fky.ravel() )) )
        return
   

    
    #####
    cpdef solve(self, np.ndarray v, np.ndarray fk, ):
        ''' solve takes two numpy arrays v and fk. Here, fk is 
        the Fourier transform of the force field on the grid and v is an empty 
        arrray of same size. It return  the updated value of v = f(1-k k/k^2)/k^2''' 
        cdef int jx, jy, Nx, Ny
        cdef double facx, facy,  a2, k2, kx, ky, kz
        cdef double complex fdotk
        Nx, Ny = self.Nx, self.Ny 
        facx, facy = self.facx, self.facy
        cdef:
            complex [:,:] fkx = fk[0:Nx*Ny       ].reshape(Nx, Ny) 
            complex [:,:] fky = fk[Nx*Ny:2*Nx*Ny ].reshape(Nx, Ny) 
            complex [:,:] vkx = self.vkx
            complex [:,:] vky = self.vky
        
        for jy in prange(Ny, nogil=True): 
            for jx in range(0, Nx): 
                kx = jx*facx if jx <= Nx / 2 else (-Nx+jx)*facx
                ky = jy*facy if jy <= Ny / 2 else (-Ny+jy)*facy
                if kx !=0  or ky != 0.0:
                    k2 = kx*kx + ky*ky 
                    fdotk = kx*fkx[jy, jx] + ky*fky[jy, jx]
                    vkx[jy, jx] = ( fkx[jy, jx] - fdotk*((kx / k2)) ) / k2
                    vky[jy, jx] = ( fky[jy, jx] - fdotk*((ky / k2)) ) / k2 
                else:
                    pass
        vkx[0, 0] = 0.0;        vky[0, 0] = 0.0;
        
        v[0:Nx*Ny]       = np.real(np.fft.ifftn(self.vkx)).ravel()
        v[Nx*Ny:2*Ny*Ny] = np.real(np.fft.ifftn(self.vky)).ravel()
        return


    cdef fourierFk(self, double sigma, double scale):
        ''' this module construct an initial Gaussian mollified force force on a grid.
        And returns the Fourier transform of this which can then be used by the method solve()'''
        cdef:
            int Nx, Ny, nx, ny
            double [:, :] fx0 = self.fx0
            double arg
        Nx, Ny = self.Nx, self.Ny 
       
        for ny in prange(Ny, nogil=True):
            for nx in range(Nx):
                arg = ( (nx-Nx/2 )*(nx-Nx/2) + (ny -Ny/2)*(ny-Ny/2) )/ (2*sigma*sigma)
                fx0[ny, nx] += exp(-arg)*scale
 
        self.fk0 = np.fft.fft2(self.fx0)
        return
    

    cpdef interpolate(self, double [:] V, double [:] r, np.ndarray vv, double sigma=3, int NN=20):
        cdef int i, nx, ny, nz, x, y, Nx, Ny, Np
        cdef double arg, pdotdr 
        cdef double scale = pow(2*PI*sigma*sigma, 2/2) 
        
            
        Nx, Ny = self.Nx, self.Ny
        Np = self.Np    
        cdef double [:,:] vx = vv[       :Nx*Ny  ] 
        cdef double [:,:] vy = vv[Nx*Ny  :2*Nx*Ny]
            
        
        for ny in range(2*NN+2):
            for nx in range(2*NN+2):
                for i in range(0, 3*Np, 3):

                    x = int(r[i])        #NOTE works for positive coordinates!
                    y = int(r[i+1]) 
                    x = x - NN + nx
                    y = y - NN + ny
                    
                    arg = ( (x - r[i])*(x-r[i]) + (y - r[i+1])*(y-r[i+1]) )/ (2 * sigma*sigma)
                    x = x % Nx
                    y = y % Ny
                
                V[i]     += exp(-arg)*scale*vx[y, x]
                V[i+1]   += exp(-arg)*scale*vy[y, x]
