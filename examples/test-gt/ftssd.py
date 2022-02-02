import autograd.numpy as np
from autograd import grad

PI = 3.14159265359


class FTS:
    """
    Rigid body motion (RBM) - velocity and angular velocity
    
    FTS - Stokesian Dynamics for an unbounded fluid: F^H=F2s, V^H=0 (rigid particle)
    
    Here, we solve the linear system arising from the boundary integral equation of Stokes
    flow directly, by inverting G^HH. Here, the matrix elements are computed by automatic
    differentiation.
    """
    
    def __init__(self, radius=1., particles=1, viscosity=1.0):
        self.b  = radius
        self.Np = particles
        self.eta = viscosity
        
        ## matrix elements for i=j
        self.G01s = 1./(6*PI*self.eta*self.b)
        self.G02a = 1./(4*PI*self.eta*self.b)
        self.G02s = 3./(20*PI*self.eta*self.b)
        
        
    ## basics for i!=j, hard-coded version is faster (see below)   
    def G(self, xij,yij,zij, alpha,beta): #G_alpha beta
        eta = self.eta
        rij = np.array([xij,yij,zij])
        r = np.linalg.norm(rij)
        return ((np.identity(3)/r + np.outer(rij,rij)/r**3)/(8*eta*PI))[alpha,beta]
    
    def delG(self, xij,yij,zij, alpha,beta,gamma): #G_alpha beta, gamma = nabla_gamma G_alpha beta
        eta = self.eta
        rij = np.array([xij,yij,zij])
        r = np.linalg.norm(rij)
        t1 = -np.einsum('ij,k',np.identity(3),rij)/r**3
        t2 = (np.einsum('ik,j',np.identity(3),rij) 
               + np.einsum('jk,i',np.identity(3),rij))/r**3
        t3 = -3*np.einsum('i,j,k',rij,rij,rij)/r**5
        return ((t1 + t2 + t3)/(8*eta*PI))[alpha,beta,gamma]

    def lapG(self, xij,yij,zij, alpha,beta): # nabla^2 G_alpha beta
        eta = self.eta
        rij = np.array([xij,yij,zij])
        r = np.linalg.norm(rij)
        return ((np.identity(3)/r**3 - 3*np.outer(rij,rij)/r**5)/(4*eta*PI))[alpha,
                                                                       beta]
    
    
    
    def epsilon(self, i,j,k):
        return (i-j)*(j-k)*(k-i)/2.
    
    
    
    ## RBM matrix elements: G^LL for i!=j
    def G1s1s(self, xij,yij,zij, alpha,beta):
        eta = self.eta
        b = self.b
        return (self.G(xij,yij,zij, alpha,beta)
                +b**2/3.*self.lapG(xij,yij,zij, alpha,beta))

    def G1s2a(self, xij,yij,zij, alpha,beta):
        eta = self.eta
        b = self.b
        g1s2a=0.
        for nu in range(3):
            for eta in range(3):
                g1s2a += self.epsilon(beta,nu,eta)*self.delG(xij,yij,zij,alpha,eta,nu)
        return -0.5*b*g1s2a

    def G2a1s(self, xij,yij,zij, alpha,beta):
        eta = self.eta
        b = self.b
        g2a1s=0
        for nu in range(3):
            for eta in range(3):
                g2a1s += self.epsilon(alpha,nu,eta)*self.delG(xij,yij,zij,eta,beta,nu)
        return 0.5*g2a1s

    def G2a2a(self, xij,yij,zij, alpha,beta):
        eta = self.eta
        b = self.b
        g2a2a=0
        for mu in range(3):
            for kappa in range(3):
                g2a2a += self.epsilon(alpha,mu,kappa)*self.delG1s2a(xij,yij,zij, kappa,beta,mu)
        return 0.5*g2a2a

    ## auxiliary functions 
    def delG1s2a(self, xij,yij,zij, kappa,beta,mu):
        return grad(self.G1s2a, mu)(xij,yij,zij, kappa,beta)
        
    #def delG1s1s(self, xij,yij,zij, alpha,beta,gamma):
    #    return grad(self.G1s1s, gamma)(xij,yij,zij, alpha,beta) 
        
        
        
        
    ## G^LH and G^HL matrix elements for i!=j
    def G1s2s(self, xij,yij,zij, alpha,kappa1,beta):
        b = self.b
        g1s2s = 0
        g1s2s += (self.delG(xij,yij,zij, alpha,beta,kappa1)
                  + self.delG(xij,yij,zij, alpha,kappa1,beta))
        g1s2s += 4*b*b/15.*(self.dellapG(xij,yij,zij, alpha,beta,kappa1) 
                            + self.dellapG(xij,yij,zij, alpha,kappa1,beta))
        return -0.5*b*g1s2s
              
    def G2a2s(self, xij,yij,zij, alpha,kappa1,beta):
        ## used in the matrix elements that eg G2a2s = 1/2 * curl G1s2s
        ## which follows from Omega = 1/2 * curl V
        ## the finite size operators will vanish because of the curl
        b = self.b
        g2a2s=0
        for nu in range(3):
            for eta in range(3):
                g2a2s += self.epsilon(alpha,nu,eta)*(self.deldelG(xij,yij,zij, eta,beta,kappa1,nu)
                                                + self.deldelG(xij,yij,zij, eta,kappa1,beta,nu))
        return -0.25*b*g2a2s

    def G2s1s(self, xij,yij,zij, alpha,gamma1,beta):
        b = self.b
        g2s1s = 0
        g2s1s += (self.delG(xij,yij,zij, alpha,beta,gamma1)
                  + self.delG(xij,yij,zij, gamma1,beta,alpha))
        g2s1s += 4*b*b/15.*(self.dellapG(xij,yij,zij, alpha,beta,gamma1) 
                            + self.dellapG(xij,yij,zij, gamma1,beta,alpha))
        return 0.5*b*g2s1s

    def G2s2a(self, xij,yij,zij, alpha,gamma1,mu):
        b = self.b
        g2s2a=0
        for kappa1 in range(3):
            for beta in range(3):
                g2s2a += self.epsilon(beta,kappa1,mu)*(self.deldelG(xij,yij,zij, gamma1,beta,alpha,kappa1)
                                                + self.deldelG(xij,yij,zij, alpha,beta,gamma1,kappa1))
        return 0.25*b*b*g2s2a

    
    
    
    ## G^HH matrix element for i!=j
    def G2s2s(self, xij,yij,zij, alpha,gamma1,kappa1,beta):
        b = self.b
        g2s2s = (self.deldelG(xij,yij,zij, alpha,beta,gamma1,kappa1) 
                  + self.deldelG(xij,yij,zij, gamma1,beta,alpha,kappa1))
        g2s2s += (self.deldelG(xij,yij,zij, alpha,kappa1,gamma1,beta) 
                  + self.deldelG(xij,yij,zij, gamma1,kappa1,alpha,beta))
        g2s2s += b*b/5.*(self.deldellapG(xij,yij,zij, alpha,beta,gamma1,kappa1)
                         + self.deldellapG(xij,yij,zij, gamma1,beta,alpha,kappa1))
        g2s2s += b*b/5.*(self.deldellapG(xij,yij,zij, alpha,kappa1,gamma1,beta)
                         + self.deldellapG(xij,yij,zij, gamma1,kappa1,alpha,beta))
        return -0.25*b*b*g2s2s
              
    ## auxiliary functions    
    def dellapG(self, xij,yij,zij, alpha,beta,kappa1):
        return grad(self.lapG, kappa1)(xij,yij,zij, alpha,beta)

    def deldelG(self, xij,yij,zij, eta,beta,kappa1,nu):
        return grad(self.delG, nu)(xij,yij,zij, eta,beta,kappa1)
    
    def deldellapG(self, xij,yij,zij, alpha,beta,gamma1,kappa1):
        return grad(self.dellapG, kappa1)(xij,yij,zij, alpha,beta,gamma1)
    
    
    
    ## fill tensors for tensorsolve from indices above
    def tensorG1s1s(self, xij,yij,zij):
        g=np.zeros([3,3])
        for alpha in range(3):
            for beta in range(3):
                g[alpha,beta]=self.G1s1s(xij,yij,zij, alpha,beta)
        return g

    def tensorG1s2a(self, xij,yij,zij):
        g=np.zeros([3,3])
        for alpha in range(3):
            for beta in range(3):
                g[alpha,beta]=self.G1s2a(xij,yij,zij, alpha,beta)
        return g

    def tensorG2a1s(self, xij,yij,zij):
        g=np.zeros([3,3])
        for alpha in range(3):
            for beta in range(3):
                g[alpha,beta]=self.G2a1s(xij,yij,zij, alpha,beta)
        return g

    def tensorG2a2a(self, xij,yij,zij):
        g=np.zeros([3,3])
        for alpha in range(3):
            for beta in range(3):
                g[alpha,beta]=self.G2a2a(xij,yij,zij, alpha,beta)
        return g


    ## higher order
    def tensorG1s2s(self, xij,yij,zij):
        g=np.zeros([3,3,3])
        for alpha in range(3):
            for beta in range(3):
                for gamma in range(3):
                    g[alpha,beta,gamma]=self.G1s2s(xij,yij,zij, alpha,beta,gamma)
        return g

    def tensorG2a2s(self, xij,yij,zij):
        g=np.zeros([3,3,3])
        for alpha in range(3):
            for beta in range(3):
                for gamma in range(3):
                    g[alpha,beta,gamma]=self.G2a2s(xij,yij,zij, alpha,beta,gamma)
        return g

    def tensorG2s1s(self, xij,yij,zij):
        g=np.zeros([3,3,3])
        for alpha in range(3):
            for beta in range(3):
                for gamma in range(3):
                    g[alpha,beta,gamma]=self.G2s1s(xij,yij,zij, alpha,beta,gamma)
        return g

    def tensorG2s2a(self, xij,yij,zij):
        g=np.zeros([3,3,3])
        for alpha in range(3):
            for beta in range(3):
                for gamma in range(3):
                    g[alpha,beta,gamma]=self.G2s2a(xij,yij,zij, alpha,beta,gamma)
        return g

    def tensorG2s2s(self, xij,yij,zij):
        g=np.zeros([3,3,3,3])
        for alpha in range(3):
            for beta in range(3):
                for gamma in range(3):
                    for delta in range(3):
                        g[alpha,beta,gamma,delta]=self.G2s2s(xij,yij,zij, alpha,beta,gamma,delta)
        return g
    
    
    
    ##direct solver for FTS Stokesian dynamics
    def directSolve(self, v, o, r, F, T):
        b = self.b
        Np = self.Np
        for i in range(Np):
            v_ = np.zeros([3])
            o_ = np.zeros([3])
            for j in range(Np):
                xij = r[i]    - r[j]
                yij = r[i+Np]  - r[j+Np]
                zij = r[i+2*Np]  - r[j+2*Np]
                if i!=j:
                    force  = np.array([F[j],F[j+Np], F[j+2*Np]])
                    torque = np.array([T[j],T[j+Np], T[j+2*Np]])
                                    
                    lhs = self.tensorG2s2s(xij,yij,zij)
                    lhs_mat = np.reshape(lhs, (9,9))
                    lhs_mat_inv = np.linalg.pinv(lhs_mat)
                    lhs_inv = np.reshape(lhs_mat_inv, (3,3,3,3))
                    rhs=np.zeros([3,3])
                    ## F2s is induced by traction on all other particles
                    for k in range(Np):
                        xjk = r[j]    - r[k]
                        yjk = r[j+Np]  - r[k+Np]
                        zjk = r[j+2*Np]  - r[k+2*Np]
                        if k!=j:
                            force_k  = np.array([F[k],F[k+Np], F[k+2*Np]])
                            torque_k = np.array([T[k],T[k+Np], T[k+2*Np]])
                            
                            rhs += (np.dot(self.tensorG2s1s(xjk,yjk,zjk), force_k) 
                                   + 1./b * np.dot(self.tensorG2s2a(xjk,yjk,zjk), torque_k))
                        else:
                            pass #otherwise have diagonal elements here
                    F2s = np.einsum('ijkl, kl', lhs_inv, rhs)
                    
                    v_ += (np.dot(self.tensorG1s1s(xij,yij,zij), force)
                          + 1./b * np.dot(self.tensorG1s2a(xij,yij,zij), torque)
                          - np.einsum('ijk,jk',self.tensorG1s2s(xij,yij,zij),F2s))
                
                    o_ += (np.dot(self.tensorG2a1s(xij,yij,zij), force)
                                 + 1./b * np.dot(self.tensorG2a2a(xij,yij,zij), torque)
                                 - np.einsum('ijk,jk',self.tensorG2a2s(xij,yij,zij),F2s))
                    ## no factor 1/(2b) necessary for angular velocity, because we have
                    ## used in the matrix elements that eg G2a1s = 1/2 * curl G1s1s
                    ## which follows from Omega = 1/2 * curl V
                       
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