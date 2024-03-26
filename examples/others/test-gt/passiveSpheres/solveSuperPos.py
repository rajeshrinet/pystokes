import numpy as np
from scipy.sparse.linalg import bicgstab, LinearOperator
#import freeSpaceME as me  ##changed dimensions below to only use symmetry (not tracelessness) of irreducible tensors
import superPosME as me   ##including symmetry factors  

PI = 3.14159265359

class linearSolve_krylov:
    
    def __init__(self, radius=1., particles=1, viscosity=1.0):
        self.b  = radius
        self.Np = particles
        self.eta = viscosity
        
        ## single-layer matrix elements for i=j
        self.g1s = 1./(6*PI*self.eta*self.b)
        self.g2a = 1./(4*PI*self.eta*self.b)
        
        
    def RBM(self, v, o, r, F, T):
        b = self.b
        Np = self.Np
        eta = self.eta
        
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
                    
                    
                    v_ += (me.G1s1sF(xij,yij,zij, b,eta, force_j)
                           + 1./b * me.G1s2aT(xij,yij,zij, b,eta, torque_j))
                    
                    o_ += 0.5/b*(me.G2a1sF(xij,yij,zij, b,eta, force_j)
                                 + 1./b * me.G2a2aT(xij,yij,zij, b,eta, torque_j))

                    
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
                    