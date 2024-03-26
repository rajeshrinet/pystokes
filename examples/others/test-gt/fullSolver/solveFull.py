import numpy as np
from scipy.sparse.linalg import bicgstab, LinearOperator
#import freeSpaceME as me  ##changed dimensions below to only use symmetry (not tracelessness) of irreducible tensors
import fullME as me   ##including symmetry factors  

PI = 3.14159265359

class linearSolve_krylov:
    
    def __init__(self, radius=1., particles=1, viscosity=1.0, tolerance=1e-05):
        self.b  = radius
        self.Np = particles
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
        self.gamma2s = 4*PI*self.eta*self.b
        self.gamma3t = 4*PI*self.eta*self.b/5.
        self.gamma3a = 8*PI*self.eta*self.b/15.
        self.gamma3s = 19*PI*self.eta*self.b/30.
        
        self.gammaH = np.tile(np.concatenate([np.full(9, self.gamma2s),
                                              np.full(3, self.gamma3t),
                                              np.full(9, self.gamma3a),
                                              np.full(27, self.gamma3s)]),self.Np)
        
        
    def RBM(self, v, o, r, F, T, S, D):
        b = self.b
        Np = self.Np
        eta = self.eta
        
        FH, exitCode = self.get_FH(r, F, T, S, D)
        # print(FH[0:self.dim2s])
        
        VH_j = np.zeros(self.dimH)
        
        force_i = np.zeros(3) #forcex, forcey, forcez = 0,0,0 instead of array
        force_j = np.zeros(3)
        
        torque_i = np.zeros(3)
        torque_j = np.zeros(3)
        
        D_i = np.zeros(3)
        
        for i in range(Np):
            v_ = np.zeros([3])
            o_ = np.zeros([3])
            for j in range(Np):
                if j!=i:
                    xij = r[i]    - r[j] 
                    yij = r[i+Np]  - r[j+Np]
                    zij = r[i+2*Np]  - r[j+2*Np]
                    
                    force_j[:]  = (F[j],F[j+Np], F[j+2*Np])
                    torque_j[:] = (T[j],T[j+Np], T[j+2*Np])
                    
                    VH_j[0:self.dim2s]  = (S[j],
                                           S[j+Np],
                                           S[j+2*Np],
                                           S[j+3*Np],
                                           S[j+4*Np],
                                           S[j+5*Np],
                                           S[j+6*Np],
                                           S[j+7*Np],
                                           S[j+8*Np])
                    
                    VH_j[self.dim2s:self.dim2s+3]  = (D[j],D[j+Np],D[j+2*Np]) 
                    
                    FH_j = FH[self.dimH*j:self.dimH*(j+1)]
                    
                    v_ += (me.G1s1sF(xij,yij,zij, b,eta, force_j)
                           + 1./b * me.G1s2aT(xij,yij,zij, b,eta, torque_j)
                           - me.G1sHFH(xij,yij,zij, b,eta, FH_j)
                           + me.K1sHVH(xij,yij,zij, b,eta, VH_j))
                    
                    # print(me.G1sHFH(xij,yij,zij, b,eta, FH_j))
                    # print(me.K1sHVH(xij,yij,zij, b,eta, VH_j))
                    
                    o_ += 0.5/b*(me.G2a1sF(xij,yij,zij, b,eta, force_j)
                                 + 1./b * me.G2a2aT(xij,yij,zij, b,eta, torque_j)
                                 - me.G2aHFH(xij,yij,zij, b,eta, FH_j)
                                 + me.K2aHVH(xij,yij,zij, b,eta, VH_j))

                    
                else: ## j==i
                    force_i[:]  = (F[i], F[i+Np], F[i+2*Np])
                    torque_i[:] = (T[i], T[i+Np], T[i+2*Np])
                    D_i[:]      = (D[i], D[i+Np], D[i+2*Np])
                    
                    ## This is a solution for V = V^A + muTT*force
                    ## We know that V^A = 1/5*V^(3t) for squirmer model
                    v_ += self.g1s*force_i + 0.2*D_i
                    o_ += 0.5/(b*b) * self.g2a*torque_i
                    
            v[i]      += v_[0]
            v[i+Np]   += v_[1]
            v[i+2*Np] += v_[2]
            
            o[i]      += o_[0]
            o[i+Np]   += o_[1]
            o[i+2*Np] += o_[2]
        return    
    
        
        
    def get_FH(self, r, F, T, S, D):
        b = self.b
        Np = self.Np
        eta = self.eta
        
        self.r = r ##for construction of GHHFH
        
        force_j  = np.zeros(3)
        torque_j = np.zeros(3)
        
        ## dimH is dimension of 2s + 3t + 3a + 3s
        VH_j = np.zeros(self.dimH)
        
        VH = np.zeros([self.dimH*Np])
        KHHVH = np.zeros([self.dimH*Np])
        GH1sF = np.zeros([self.dimH*Np])
        GH2aT = np.zeros([self.dimH*Np])
        
        for i in range(Np):
            for j in range(Np):
                VH_j[0:self.dim2s]  = (S[j],
                                       S[j+Np],
                                       S[j+2*Np],
                                       S[j+3*Np],
                                       S[j+4*Np],
                                       S[j+5*Np],
                                       S[j+6*Np],
                                       S[j+7*Np],
                                       S[j+8*Np])

                VH_j[self.dim2s:self.dim2s+3]  = (D[j],D[j+Np],D[j+2*Np])
                VH[self.dimH*j:self.dimH*(j+1)] = VH_j
                
                if j!=i: ## off diagonals
                    xij = r[i]    - r[j] 
                    yij = r[i+Np]  - r[j+Np]
                    zij = r[i+2*Np]  - r[j+2*Np]
                    
                    force_j[:]  = (F[j],F[j+Np], F[j+2*Np])
                    torque_j[:] = (T[j],T[j+Np], T[j+2*Np])
                    
                    me.KHHVH(KHHVH, self.dimH, i, xij,yij,zij, b,eta, VH_j)
                    me.GH1sF(GH1sF, self.dimH, i, xij,yij,zij, b,eta, force_j)
                    me.GH2aT(GH2aT, self.dimH, i, xij,yij,zij, b,eta, torque_j)
                    
                    
                else: ## add diagonal elements to KHH etc, j==i
                    me.KoHHVH(KHHVH, self.dimH, i, b,eta, VH_j)  # don't foregt the minus sign!
                    
        rhs = KHHVH + GH1sF + 1./b * GH2aT 
        FH0 = -self.gammaH*VH  #start at the one-particle solution
        
        GHHFH = LinearOperator((self.dimH*Np, self.dimH*Np), matvec = self.GHHFH)
            
        return bicgstab(GHHFH, rhs, x0=FH0, tol=self.tol)
    
    
    def GHHFH(self, FH):  ##FH is array of dimension 20*Np
        b = self.b
        Np = self.Np
        eta = self.eta
        
        r = self.r
        
        GHHFH = np.zeros([self.dimH*Np])
        
        for i in range(Np):
            for j in range(Np):
                if j!=i: ## off diagonals
                    xij = r[i]    - r[j] 
                    yij = r[i+Np]  - r[j+Np]
                    zij = r[i+2*Np]  - r[j+2*Np]
                    
                    me.GHHFH(GHHFH, self.dimH, i, xij,yij,zij, b,eta, FH[self.dimH*j:self.dimH*(j+1)])
                    
                    
                else: ## add diagonal elements to KHH etc, j==i
                    
                    me.GoHHFH(GHHFH, self.dimH, i, b,eta, FH[self.dimH*j:self.dimH*(j+1)])
                    
        return GHHFH
                    