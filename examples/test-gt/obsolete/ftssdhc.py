import numpy as np
import matelms as me

PI = 3.14159265359


class FTShc:
    """
    Rigid body motion (RBM) - velocity and angular velocity
    
    FTS - Stokesian Dynamics for an unbounded fluid: F^H=F2s, V^H=0 (rigid particle)
    
    Here, we solve the linear system arising from the boundary integral equation of Stokes
    flow directly, by inverting G^HH. Here, the matrix elements were computed in Mathematica
    and imported as hard-coded versions.
    """
    
    def __init__(self, radius=1., particles=1, viscosity=1.0):
        self.b  = radius
        self.Np = particles
        self.eta = viscosity
        
        ## matrix elements for i=j
        self.G01s = 1./(6*PI*self.eta*self.b)
        self.G02a = 1./(4*PI*self.eta*self.b)
        
        ## friction coefficients for i=j
        self.g2s = 4*PI*self.eta*self.b
        self.g3t = 0.8*PI*self.eta*self.b
        
        
    ## direct solver for FTS Stokesian dynamics
    ## not entirely sure yet whether there should be symmetry factors of 2 
    ## whenever there is a dot product G2s2s F2s etc, since we are effectively 
    ## throwing away half of the terms due to symmetry
    def directSolve(self, v, o, r, F, T):
        b = self.b
        Np = self.Np
        eta = self.eta
        for i in range(Np):
            v_ = np.zeros([3])
            o_ = np.zeros([3])
            for j in range(Np):
                xij = r[i]    - r[j]
                yij = r[i+Np]  - r[j+Np]
                zij = r[i+2*Np]  - r[j+2*Np]
                if j!=i:
                    force  = np.array([F[j],F[j+Np], F[j+2*Np]])
                    torque = np.array([T[j],T[j+Np], T[j+2*Np]])
                    
                    rhs = np.zeros(5)
                    ## F2s_j is induced by traction on all other particles
                    for k in range(Np):
                        xjk = r[j]    - r[k]
                        yjk = r[j+Np]  - r[k+Np]
                        zjk = r[j+2*Np]  - r[k+2*Np]
                        if k!=j:
                            force_k  = np.array([F[k],F[k+Np], F[k+2*Np]])
                            torque_k = np.array([T[k],T[k+Np], T[k+2*Np]])
                            
                            rhs += (np.dot(me.G2s1s(xjk,yjk,zjk, b,eta), force_k) 
                                   + 1./b * np.dot(me.G2s2a(xjk,yjk,zjk, b,eta), torque_k))
                        else:
                            pass #otherwise have diagonal elements here
                                    
                    F2s = np.linalg.solve(me.G2s2s(xij,yij,zij, b,eta), rhs)
                    
                    v_ += (np.dot(me.G1s1s(xij,yij,zij, b,eta), force)
                          + 1./b * np.dot(me.G1s2a(xij,yij,zij, b,eta), torque)
                          - np.dot(me.G1s2s(xij,yij,zij, b,eta),F2s))
                    
                    o_ += 0.5/b*(np.dot(me.G2a1s(xij,yij,zij, b,eta), force)
                                 + 1./b * np.dot(me.G2a2a(xij,yij,zij, b,eta), torque)
                                 - np.dot(me.G2a2s(xij,yij,zij, b,eta),F2s))
                           
                else:
                    force  = np.array([F[j],F[j+Np], F[j+2*Np]])
                    torque = np.array([T[j],T[j+Np], T[j+2*Np]])
                    
                    v_ += self.G01s*force
                    o_ += 0.5/(b*b) * self.G02a*torque
                    
            v[i]      += v_[0]
            v[i+Np]   += v_[1]
            v[i+2*Np] += v_[2]
            
            o[i]      += o_[0]
            o[i+Np]   += o_[1]
            o[i+2*Np] += o_[2]
        return
