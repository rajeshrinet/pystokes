import numpy as np
import freeSpaceMatrixME as me  ## instead of matelms by Mathematica 

PI = 3.14159265359

class linearSolve_direct:
    
    def __init__(self, radius=1., particles=1, viscosity=1.0):
        self.b  = radius
        self.Np = particles
        self.eta = viscosity
        
        ## single-layer matrix elements for i=j
        self.g1s = 1./(6*PI*self.eta*self.b)
        self.g2a = 1./(4*PI*self.eta*self.b)
        
#         self.g2s = 3./(20*PI*self.eta*self.b)
#         self.g3t = 1./(2*PI*self.eta*self.b)
#         self.g3a = 3./(2*PI*self.eta*self.b)
#         self.g3s = 6./(7*PI*self.eta*self.b)
        
#         self.GoHH = np.diag(np.block([np.full(5, self.g2s), np.full(3, self.g3t), np.full(5, self.g3a), np.full(7, self.g3s)]))
        
        
#         ## double-layer matrix elements for i=j
#         self.halfMinusk2s = 0.6
#         self.halfMinusk3t = 0.4
#         self.halfMinusk3a = 0.8
#         self.halfMinusk3s = 19./35.
        
#         self.KoHH = np.diag(np.block([np.full(5, self.halfMinusk2s), np.full(3, self.halfMinusk3t), np.full(5, self.halfMinusk3a), np.full(7, self.halfMinusk3s)]))
        
        
    def RBM(self, v, o, r, F, T, S, D):
        b = self.b
        Np = self.Np
        eta = self.eta
        
        FH = self.get_FH(r, F, T, S, D)
        
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
                    
                    S_j  = np.array([S[j],S[j+Np],S[j+2*Np],S[j+3*Np],S[j+4*Np]])
                    D_j  = np.array([D[j],D[j+Np],D[j+2*Np]])
                    VH_j = np.concatenate([S_j,D_j,np.zeros(12)]) 
                    
                    FH_j = FH[20*j:20*(j+1)]
                    
                    v_ += (np.dot(me.G1s1s(xij,yij,zij, b,eta), force_j)
                          + 1./b * np.dot(me.G1s2a(xij,yij,zij, b,eta), torque_j)
                          - np.dot(me.G1sH(xij,yij,zij, b,eta), FH_j)
                          + np.dot(me.K1sH(xij,yij,zij, b,eta), VH_j))
                    
                    o_ += 0.5/b*(np.dot(me.G2a1s(xij,yij,zij, b,eta), force_j)
                                 + 1./b * np.dot(me.G2a2a(xij,yij,zij, b,eta), torque_j)
                                 - np.dot(me.G2aH(xij,yij,zij, b,eta), FH_j)
                                 + np.dot(me.K2aH(xij,yij,zij, b,eta), VH_j))
                    
                else: ## j==i
                    force_i  = np.array([F[i], F[i+Np], F[i+2*Np]])
                    torque_i = np.array([T[i], T[i+Np], T[i+2*Np]])
                    D_i      = np.array([D[i], D[i+Np], D[i+2*Np]])
                    
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
        
        force  = np.zeros(3*Np)
        torque = np.zeros(3*Np)
        
        ## 20 is dimension of 2s + 3t + 3a + 3s
        VH = np.zeros(20*Np)
        
        GHH = np.zeros([20*Np,20*Np])
        KHH = np.zeros([20*Np,20*Np])
        GH1s = np.zeros([20*Np,3*Np])
        GH2a = np.zeros([20*Np,3*Np])
        
        for i in range(Np):
            S_i  = np.array([S[i],S[i+Np],S[i+2*Np],S[i+3*Np],S[i+4*Np]])
            D_i  = np.array([D[i],D[i+Np],D[i+2*Np]])
            
            VH[20*i:20*i+5]   = S_i
            VH[20*i+5:20*i+8] = D_i
            
            ## have to reorder F and T from F1x, F2x, F1y, F2y,... to F1x, F1y, F1z, F2x,...
            force[3*i:3*i+3]  = np.array([F[i],F[i+Np], F[i+2*Np]])
            torque[3*i:3*i+3] = np.array([T[i],T[i+Np], T[i+2*Np]])
            
            for j in range(Np):
                if j!=i: ## off diagonals
                    xij = r[i]    - r[j] 
                    yij = r[i+Np]  - r[j+Np]
                    zij = r[i+2*Np]  - r[j+2*Np]
                    
                    GHH[20*i:20*(i+1), 20*j:20*(j+1)] = me.GHH(xij,yij,zij, b,eta)
                    KHH[20*i:20*(i+1), 20*j:20*(j+1)] = me.KHH(xij,yij,zij, b,eta)
                    GH1s[20*i:20*(i+1), 3*j:3*(j+1)] = me.GH1s(xij,yij,zij, b,eta)
                    GH2a[20*i:20*(i+1), 3*j:3*(j+1)] = me.GH2a(xij,yij,zij, b,eta)
                      
                else: ## fill diagonal elements of GHH etc, j==i
                    GHH[20*j:20*(j+1), 20*j:20*(j+1)] = me.GoHH(b, eta)
                    KHH[20*j:20*(j+1), 20*j:20*(j+1)] = - me.KoHH(b, eta)
            
        rhs = np.dot(KHH, VH) + np.dot(GH1s, force) + 1./b * np.dot(GH2a, torque)
        
        return np.linalg.solve(GHH, rhs)
                    