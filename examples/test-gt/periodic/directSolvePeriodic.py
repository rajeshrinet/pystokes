import numpy as np
from math import *
import matrixPeriodic_1_4 as me

PI = 3.14159265359


class Rbm:
    
    def __init__(self, radius=1., viscosity=1.0, boxSize=10, xi = 123456789):
        self.b  = radius
        self.eta = viscosity
        self.L = boxSize
        
        if xi==123456789:
            self.xi = np.sqrt(PI)/boxSize 
            #Nijboer and De Wette have shown that \pi^{1/2}/V^{1/3} is a good choice for cubic lattices 
        else:
            self.xi = xi 
        
        
        ## single-layer matrix elements for i=j (not gamma)
        self.g1s = 1./(6*PI*self.eta*self.b)
        self.g2a = 1./(4*PI*self.eta*self.b)
        
        ## total dim of higher modes 
        self.dimH = 20
        
        ## used dimensions of 2s
        self.dim2s = 5
        
        ## subtract M2(r=0), see Beenakker
        self.M20 = self.g1s*(1 - 6/np.sqrt(PI)*self.xi*self.b + 40/(3*np.sqrt(PI))*self.xi**3*self.b**3)
        
        
    def directSolve(self, v, o, F, T, S, D, xi0=123456789):
        b = self.b
        eta = self.eta
        xi = self.xi
        L = self.L
        M20 = self.M20
        
        if xi0 != 123456789:
            xi = xi0 
        
        VH = np.zeros(self.dimH)
        FH = self.get_FH(F, T, S, D)
        #FH = np.zeros(self.dimH)
                    
        VH[0:self.dim2s]  = S
        VH[self.dim2s:self.dim2s+3]  = D 
        
        ## interactions with periodic lattice
        v += (np.dot(me.G1s1s(L,xi, b,eta), F)
              + 1./b * np.dot(me.G1s2a(L,xi, b,eta), T)
              - np.dot(me.G1sH(L,xi, b,eta), FH)
              + np.dot(me.K1sH(L,xi, b,eta), VH))

        o += 0.5/b*(np.dot(me.G2a1s(L,xi, b,eta), F)
                     + 1./b * np.dot(me.G2a2a(L,xi, b,eta), T)
                     - np.dot(me.G2aH(L,xi, b,eta), FH)
                     + np.dot(me.K2aH(L,xi, b,eta), VH))
        
        ## self-interaction, subtract M2(r=0), g1sF is included in first term
        v += M20*F + 0.2*D 
        o += 0.5/(b*b) * self.g2a*T ##M2(r=0) for rotation?
        
        return
    
    
    
    def get_FH(self, F, T, S, D):
        b = self.b
        eta = self.eta
        xi = self.xi
        L = self.L

        ## dimH is dimension of 2s + 3t + 3a + 3s
        VH = np.zeros(self.dimH)
        
        VH[0:self.dim2s]  = S
        VH[self.dim2s:self.dim2s+3]  = D
        
        rhs = (np.dot(me.KHH(L,xi, b,eta), VH) 
               + np.dot(me.GH1s(L,xi, b,eta), F) 
               + 1./b * np.dot(me.GH2a(L,xi, b,eta), T))
        
        rhs += - np.dot(me.KoHH(b, eta), VH)
        
        GHH = me.GHH(L,xi, b,eta) + me.GoHH(b, eta)
        
        return np.linalg.solve(GHH,rhs)