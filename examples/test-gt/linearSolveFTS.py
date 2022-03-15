import numpy as np
import matelms as me

PI = 3.14159265359

class linearSolve_direct:
    
    def __init__(self, radius=1., particles=1, viscosity=1.0):
        self.b  = radius
        self.Np = particles
        self.eta = viscosity
        
        ## single-layer matrix elements for i=j
        self.g1s = 1./(6*PI*self.eta*self.b)
        self.g2a = 1./(4*PI*self.eta*self.b)
        
        self.g2s = 3./(20*PI*self.eta*self.b)
        
        self.GoHH = np.diag(np.block([np.full(5, self.g2s)]))
        
        ## double-layer matrix elements for i=j
        self.halfMinusk2s = 0.6
        
        self.KoHH = np.diag(np.block([np.full(5, self.halfMinusk2s)]))
        
        
    def get_F2s(self, r, F, T, S):
        b = self.b
        Np = self.Np
        eta = self.eta
        
        ## 5 is dimension of 2s
        VH = np.zeros(5*Np)
        
        
        GHH = np.zeros([5*Np,5*Np])   #there is only F2s, while V2s=0 -- keep option to retain V2s as system needs to be quadratic anyway
        KHH  = np.zeros([5*Np,5*Np])
        GH1s = np.zeros([5*Np,3*Np])
        GH2a = np.zeros([5*Np,3*Np])
        
        force  = np.zeros(3*Np)
        torque = np.zeros(3*Np)
        
        for i in range(Np):
            
            VH[5*i:5*i+5]   = np.array([S[i],S[i+Np],S[i+2*Np],S[i+3*Np],S[i+4*Np]])
            
            ## have to reorder F and T from F1x, F2x, F1y, F2y,... to F1x, F1y, F1z, F2x,...
            force[3*i:3*i+3]  = np.array([F[i],F[i+Np], F[i+2*Np]])
            torque[3*i:3*i+3] = np.array([T[i],T[i+Np], T[i+2*Np]])
            
            for j in range(Np):
                if j!=i: ## off diagonals
                    xij = r[i]    - r[j] 
                    yij = r[i+Np]  - r[j+Np]
                    zij = r[i+2*Np]  - r[j+2*Np]
                    
                    GHH[5*i:5*(i+1), 5*j:5*(j+1)] = me.G2s2s(xij,yij,zij, b,eta)
                    KHH[5*i:5*(i+1), 5*j:5*(j+1)] = me.K2s2s(xij,yij,zij, b,eta)
                    GH1s[5*i:5*(i+1), 3*j:3*(j+1)] = me.G2s1s(xij,yij,zij, b,eta)
                    GH2a[5*i:5*(i+1), 3*j:3*(j+1)] = me.G2s2a(xij,yij,zij, b,eta)
                    
                    
                else: ## fill diagonal elements of GHH etc, j==i
                    GHH[5*j:5*(j+1), 5*j:5*(j+1)] = self.GoHH
                    KHH[5*j:5*(j+1), 5*j:5*(j+1)] = - self.KoHH
            
        rhs = np.dot(KHH, VH) + np.dot(GH1s, force) + 1./b * np.dot(GH2a, torque)
            
        return np.linalg.solve(GHH, rhs)
    
    
    
    def RBM(self, v, o, r, F, T, S):
        b = self.b
        Np = self.Np
        eta = self.eta
        
        FH = self.get_F2s(r, F, T, S) # order is 1xx, 1xy, 1xz,... 2xx, 2xy,...
        
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
                    VH_j = S_j
                    
                    FH_j = FH[5*j:5*(j+1)]
                    
                    v_ += (np.dot(me.G1s1s(xij,yij,zij, b,eta), force_j)
                          + 1./b * np.dot(me.G1s2a(xij,yij,zij, b,eta), torque_j)
                          - np.dot(me.G1s2s(xij,yij,zij, b,eta), FH_j)
                          + np.dot(me.K1s2s(xij,yij,zij, b,eta), VH_j))
                    
                    o_ += 0.5/b*(np.dot(me.G2a1s(xij,yij,zij, b,eta), force_j)
                                 + 1./b * np.dot(me.G2a2a(xij,yij,zij, b,eta), torque_j)
                                 - np.dot(me.G2a2s(xij,yij,zij, b,eta), FH_j)
                                 + np.dot(me.K2a2s(xij,yij,zij, b,eta), VH_j))
                    
                else: ## j==i
                    force_i  = np.array([F[i], F[i+Np], F[i+2*Np]])
                    torque_i = np.array([T[i], T[i+Np], T[i+2*Np]])
                    
                    v_ += self.g1s*force_i
                    o_ += 0.5/(b*b) * self.g2a*torque_i
                    
            v[i]      += v_[0]
            v[i+Np]   += v_[1]
            v[i+2*Np] += v_[2]
            
            o[i]      += o_[0]
            o[i+Np]   += o_[1]
            o[i+2*Np] += o_[2]
        return
    
    