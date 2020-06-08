#!python
"""Unittesting for the pystokes module. Run as python -m unittest pystokes.test."""
import sys
import pystokes
import unittest
import inspect
import numpy as np
import scipy as sp

class UnboundedTest(unittest.TestCase):
    
    def test_translation(self):
        r = np.array([0,0,0.])
        F = np.array([0,0,1.])

        a, Np, eta = 1, 1, 1 
        mu = 1/(6*np.pi*a*eta)
        
        V1 = mu*F
        V2 = 0*mu*F
        
        uRbm = pystokes.unbounded.Rbm(a, Np, eta)
        uRbm.mobilityTT(V2, r, F)
        
        diff = V1[2] - V2[2] 
        self.assertTrue((np.asarray(diff) < 0.001).all(),
                       msg=f"Stokes law for translation is not satisfied")


    def test_rotation(self):
        r = np.array([0,0,0.])
        T = np.array([0,0,1.])

        a, Np, eta = 1, 1, 1 
        mu = 1/(8*np.pi*a**3*eta)

        W1 = mu*T
        W2 = 0*mu*T
        
        uRbm = pystokes.unbounded.Rbm(a, Np, eta)
        uRbm.mobilityRR(W2, r, T)
        
        diff = W1[2] - W2[2] 
        self.assertTrue((np.asarray(diff) < 0.001).all(),
                       msg=f"Stokes law for rotation is not satisfied")


class WallBoundedTest(unittest.TestCase):
    
    def test_parallelTranslation(self):
        r = np.array([0,0,1.])
        F = np.array([0,1,0.])

        a, Np, eta = 1, 1, 1 
        mu = 1/(6*np.pi*a*eta)
        mu = mu*(1- 9./16 + 1/8)  # add the standard wall-correction

        V1 = mu*F
        V2 = 0*mu*F

        uRbm = pystokes.wallBounded.Rbm(a, Np, eta)
        uRbm.mobilityTT(V2, r, F)
        
        diff = V1[1] - V2[1] 
        self.assertTrue((np.asarray(diff) < 0.001).all(),
                       msg=f"Stokes law for translation || to wall is not satisfied")


    def test_perpTranslation(self):
        r = np.array([0,0,1.])
        F = np.array([0,0,1.])

        a, Np, eta = 1, 1, 1 
        mu = 1/(6*np.pi*a*eta)
        mu = mu*(1 - 9/8. + 1./2)  # add the standard wall-correction

        V1 = mu*F
        V2 = 0*mu*F

        uRbm = pystokes.wallBounded.Rbm(a, Np, eta)
        uRbm.mobilityTT(V2, r, F)
        
        diff = V1[2] - V2[2] 
        self.assertTrue((np.asarray(diff) < 0.001).all(),
                       msg=f"Stokes law for translation perp to wall is not satisfied")
    

if __name__ == '__main__':
    unittest.main()
