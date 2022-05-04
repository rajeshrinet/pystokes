import numpy as np
from scipy.sparse.linalg import bicgstab, LinearOperator
import wallME as me

PI = 3.14159265359

class Rbm: ##in pystokes style
    
    def __init__(self, radius=1., viscosity=1.0, tolerance=1e-05):
        self.b  = radius
        self.eta = viscosity
        
        ## tolerance for Krylov iterative solver
        self.tol = tolerance
        
        ## single-layer matrix elements for i=j
        self.g1s = 1./(6*PI*self.eta*self.b)
        self.g2a = 1./(4*PI*self.eta*self.b)
        
        ## total dim of higher modes 
        self.dimH = 48
        
        ## used dimensions of 2s
        self.dim2s = 9
        
        ## for starting point of Krylov solver: free space solution
        self.g2s = 3./(20*PI*self.eta*self.b)
        self.g3t = 1./(2*PI*self.eta*self.b)
        self.g3a = 3./(2*PI*self.eta*self.b)
        self.g3s = 6./(7*PI*self.eta*self.b)

        self.GoHH_diag = np.concatenate([np.full(9, self.g2s), np.full(3, self.g3t), np.full(9, self.g3a), np.full(27, self.g3s)]) ## define in me file?
        
        
    def krylovSolve(self, v, o, r, F, T, S, D):
        b = self.b
        eta = self.eta
        
        VH = np.zeros(self.dimH)
        FH, exitCode = self.get_FH(r, F, T, S, D)
        
        h = r[2]
                    
        VH[0:self.dim2s]  = S
        VH[self.dim2s:self.dim2s+3]  = D 
        
        ## interactions with image
        me.G1s1sF(v, h, b,eta, F) ##update v and o directly? 
        me.G1s2aT(v, h, b,eta, T)
        me.G1sHFH(v, h, b,eta, FH)
        me.K1sHVH(v, h, b,eta, VH)
        
        me.G2a1sF(o, h, b,eta, F)
        me.G2a2aT(o, h, b,eta, T)
        me.G2aHFH(o, h, b,eta, FH)
        me.K2aHVH(o, h, b,eta, VH)
        
        ## self-interaction
        v += self.g1s*F# + 0.2*D
        o += 0.5/(b*b) * self.g2a*T
        
        return    
    
        
        
    def get_FH(self, r, F, T, S, D):
        b = self.b
        eta = self.eta
        
        self.r = r ##for construction of GHHFH
        
        
        ## dimH is dimension of 2s + 3t + 3a + 3s
        VH = np.zeros(self.dimH)
        
        KHHVH = np.zeros([self.dimH])
        GH1sF = np.zeros([self.dimH])
        GH2aT = np.zeros([self.dimH])
        
        h = r[2]
                    
        VH[0:self.dim2s]  = S
        VH[self.dim2s:self.dim2s+3]  = D
        
        ## interactions with image
        me.KHHVH(KHHVH, h, b,eta, VH)
        me.GH1sF(GH1sF, h, b,eta, F)
        me.GH2aT(GH2aT, h, b,eta, T)

        ## self-interaction
        me.KoHHVH(KHHVH, b,eta, VH)
            
        rhs = KHHVH + GH1sF + 1./b * GH2aT 
        x0 = rhs/self.GoHH_diag  #start at the free-space solution
        
        GHHFH = LinearOperator((self.dimH, self.dimH), matvec = self.GHHFH)
            
        return bicgstab(GHHFH, rhs, x0, self.tol)
    
    
    def GHHFH(self, FH):
        b = self.b
        eta = self.eta
        
        r = self.r
        
        GHHFH = np.zeros(self.dimH)
        
        h = r[2]
        
        ## interactions with image
        me.GHHFH(GHHFH, h, b,eta, FH)
        
        ## self-interaction
        me.GoHHFH(GHHFH, b,eta, FH)
                    
        return GHHFH