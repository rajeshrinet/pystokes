import numpy as np
from scipy.sparse.linalg import bicgstab, LinearOperator
#import freeSpaceME as me  ##changed dimensions below to only use symmetry (not tracelessness) of irreducible tensors
import _2sME as me   ##including symmetry factors  

PI = 3.14159265359

class linearSolve_krylov:
    
    def __init__(self, radius=1., particles=1, viscosity=1.0):
        self.b  = radius
        self.Np = particles
        self.eta = viscosity
        
        ## single-layer matrix elements for i=j
        self.g1s = 1./(6*PI*self.eta*self.b)
        self.g2a = 1./(4*PI*self.eta*self.b)
        
        
    def RBM(self, v, o, r, F, T, S):
        b = self.b
        Np = self.Np
        eta = self.eta
        
        FH, exitCode = self.get_FH(r, F, T, S)
        
        VH_j = np.zeros(6)
        
        for i in range(Np):
            v_ = np.zeros([3])
            o_ = np.zeros([3])
            for j in range(Np):
                if j!=i:
                    xij = r[i]    - r[j] 
                    yij = r[i+Np]  - r[j+Np]
                    zij = r[i+2*Np]  - r[j+2*Np]
                    
                    force_j  = np.array([F[j],F[j+Np], F[j+2*Np]])
                    torque_j = np.array([T[j],T[j+Np], T[j+2*Np]])
                    
                    VH_j[0:6]  = np.array([S[j],S[j+Np],S[j+2*Np],S[j+3*Np],S[j+4*Np],S[j+5*Np]]) ##would have to parse S differently, dimension 6
                    
                    FH_j = FH[6*j:6*(j+1)]
                    
                    v_ += (me.G1s1sF(xij,yij,zij, b,eta, force_j)
                           + 1./b * me.G1s2aT(xij,yij,zij, b,eta, torque_j)
                           - me.G1sHFH(xij,yij,zij, b,eta, FH_j)
                           + me.K1sHVH(xij,yij,zij, b,eta, VH_j))
                    
                    o_ += 0.5/b*(me.G2a1sF(xij,yij,zij, b,eta, force_j)
                                 + 1./b * me.G2a2aT(xij,yij,zij, b,eta, torque_j)
                                 - me.G2aHFH(xij,yij,zij, b,eta, FH_j)
                                 + me.K2aHVH(xij,yij,zij, b,eta, VH_j))

                    
                else: ## j==i
                    force_i  = np.array([F[i], F[i+Np], F[i+2*Np]])
                    torque_i = np.array([T[i], T[i+Np], T[i+2*Np]])
                    
                    ## This is a solution for V = V^A + muTT*force
                    ## We know that V^A = 1/5*V^(3t) for squirmer model
                    v_ += self.g1s*force_i
                    o_ += 0.5/(b*b) * self.g2a*torque_i
                    
            v[i]      += v_[0]
            v[i+Np]   += v_[1]
            v[i+2*Np] += v_[2]
            
            o[i]      += o_[0]
            o[i+Np]   += o_[1]
            o[i+2*Np] += o_[2]
        return    
    
        
        
    def get_FH(self, r, F, T, S):
        b = self.b
        Np = self.Np
        eta = self.eta
        
        self.r = r ##for construction of GHHFH
        
        force_j  = np.zeros(3)
        torque_j = np.zeros(3)
        
        ## 20 is dimension of 2s + 3t + 3a + 3s
        VH_j = np.zeros(6)
        
        KHHVH = np.zeros([6*Np])
        GH1sF = np.zeros([6*Np])
        GH2aT = np.zeros([6*Np])
        
        GoHH = np.tile(me.GoHHFH(b,eta, np.ones(6)), Np) ##for krylov starting point x0
        
        for i in range(Np):
            for j in range(Np):
                if j!=i: ## off diagonals
                    xij = r[i]    - r[j] 
                    yij = r[i+Np]  - r[j+Np]
                    zij = r[i+2*Np]  - r[j+2*Np]
                    
                    force_j  = np.array([F[j],F[j+Np], F[j+2*Np]])
                    torque_j = np.array([T[j],T[j+Np], T[j+2*Np]])
                    
                    VH_j[0:6]  = np.array([S[j],S[j+Np],S[j+2*Np],S[j+3*Np],S[j+4*Np],S[j+5*Np]])  ##would have to parse S differently, dimension 6
                    
                    KHHVH[6*i:6*(i+1)] += me.KHHVH(xij,yij,zij, b,eta, VH_j)
                    GH1sF[6*i:6*(i+1)] += me.GH1sF(xij,yij,zij, b,eta, force_j)
                    GH2aT[6*i:6*(i+1)] += me.GH2aT(xij,yij,zij, b,eta, torque_j)
                    
                    
                else: ## add diagonal elements to KHH etc, j==i
                    VH_j[0:6]  = np.array([S[j],S[j+Np],S[j+2*Np],S[j+3*Np],S[j+4*Np],S[j+5*Np]])  ##would have to parse S differently, dimension 6
                    
                    KHHVH[6*i:6*(i+1)] += - me.KoHHVH(b, eta, VH_j)
                    
        rhs = KHHVH + GH1sF + 1./b * GH2aT 
        x0 = rhs/GoHH
        
        Ax = LinearOperator((6*Np, 6*Np), matvec = self.GHHFH)
            
        return bicgstab(Ax, rhs, x0)
    
    
    def GHHFH(self, FH):  ##FH is array of dimension 20*Np
        b = self.b
        Np = self.Np
        eta = self.eta
        
        r = self.r
        
        vecGHHFH = np.zeros([6*Np])
        
        for i in range(Np):
            for j in range(Np):
                if j!=i: ## off diagonals
                    xij = r[i]    - r[j] 
                    yij = r[i+Np]  - r[j+Np]
                    zij = r[i+2*Np]  - r[j+2*Np]
                    
                    vecGHHFH[6*i:6*(i+1)] += me.GHHFH(xij,yij,zij, b,eta, FH[6*j:6*(j+1)])
                    
                    
                else: ## add diagonal elements to KHH etc, j==i
                    vecGHHFH[6*i:6*(i+1)] += me.GoHHFH(b, eta, FH[6*j:6*(j+1)])
                    
        return vecGHHFH
                    