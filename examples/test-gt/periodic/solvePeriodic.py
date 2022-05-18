import numpy as np
from math import *
from scipy.sparse.linalg import bicgstab, LinearOperator
import periodic_1_4 as me

PI = 3.14159265359

class Rbm:
    
    def __init__(self, radius=1., viscosity=1.0, boxSize=10, xi = 123456789, tolerance=1e-05):
        self.b  = radius
        self.eta = viscosity
        self.L = boxSize
        
        if xi==123456789:
            self.xi = sqrt(PI)/boxSize 
            #Nijboer and De Wette have shown that \pi^{1/2}/V^{1/3} is a good choice for cubic lattices 
        else:
            self.xi = xi 
        
        ## tolerance for Krylov iterative solver
        self.tol = tolerance
        
        ## single-layer matrix elements for i=j (not gamma)
        self.g1s = 1./(6*PI*self.eta*self.b)
        self.g2a = 1./(4*PI*self.eta*self.b)
        
        ## total dim of higher modes 
        self.dimH = 20
        
        ## used dimensions of 2s
        self.dim2s = 5
        
        ## for starting point of Krylov solver: free space solution
        self.gamma2s = 4*PI*self.eta*self.b
        self.gamma3t = 4*PI*self.eta*self.b/5.
        self.gamma3a = 8*PI*self.eta*self.b/15.
        self.gamma3s = 19*PI*self.eta*self.b/30.
        
        self.gammaH = np.concatenate([np.full(self.dim2s, self.gamma2s),
                                      np.full(3, self.gamma3t),
                                      np.full(self.dim2s, self.gamma3a),
                                      np.full(self.dimH-2*self.dim2s-3, self.gamma3s)])
        
        ## subtract M2(r=0), see Beenakker
        self.M20 = self.g1s*(1-6*1/sqrt(PI)*self.xi*self.b + 40/3*1/sqrt(PI)*self.xi**3*self.b**3)
        
        
        
    def krylovSolve(self, v, o, F, T, S, D):
        b = self.b
        eta = self.eta
        xi = self.xi
        L = self.L
        M20 = self.M20
        
        VH = np.zeros(self.dimH)
        FH, exitCode = self.get_FH(F, T, S, D)
                    
        VH[0:self.dim2s]  = S
        VH[self.dim2s:self.dim2s+3]  = D 
        
        ## interactions with periodic lattice
        me.G1s1sF(v, L,xi, b,eta, F)
        me.G1s2aT(v, L,xi, b,eta, T)
        me.G1sHFH(v, L,xi, b,eta, FH)
        me.K1sHVH(v, L,xi, b,eta, VH) 
        
        me.G2a1sF(o, L,xi, b,eta, F)
        me.G2a2aT(o, L,xi, b,eta, T)
        me.G2aHFH(o, L,xi, b,eta, FH)
        me.K2aHVH(o, L,xi, b,eta, VH)
        
        ## self-interaction, subtract M2(r=0)
        v += self.g1s*F + 0.2*D + M20*F
        o += 0.5/(b*b) * self.g2a*T
        print(M20)
        
        return
    
    
    def get_FH(self, F, T, S, D):
        b = self.b
        eta = self.eta
        xi = self.xi
        L = self.L

        ## dimH is dimension of 2s + 3t + 3a + 3s
        VH = np.zeros(self.dimH)
        
        KHHVH = np.zeros([self.dimH])
        GH1sF = np.zeros([self.dimH])
        GH2aT = np.zeros([self.dimH])
                    
        VH[0:self.dim2s]  = S
        VH[self.dim2s:self.dim2s+3]  = D
        
        ## interactions with periodic lattice
        me.KHHVH(KHHVH, L,xi, b,eta, VH)
        me.GH1sF(GH1sF, L,xi, b,eta, F)
        me.GH2aT(GH2aT, L,xi, b,eta, T)
        

        ## self-interaction
        me.KoHHVH(KHHVH, b,eta, VH)
            
        rhs = KHHVH + GH1sF + 1./b * GH2aT 
        FH0 = -self.gammaH*VH  #start at the one-body solution
        
        GHHFH = LinearOperator((self.dimH, self.dimH), matvec = self.GHHFH)
            
        return bicgstab(GHHFH, rhs, x0=FH0, tol=self.tol)
    
    
    def GHHFH(self, FH):
        b = self.b
        eta = self.eta
        xi = self.xi
        L = self.L
        
        GHHFH = np.zeros(self.dimH)
        
        ## interactions with periodic lattice
        me.GHHFH(GHHFH, L,xi, b,eta, FH)
        
        ## self-interaction
        me.GoHHFH(GHHFH, b,eta, FH)
                    
        return GHHFH