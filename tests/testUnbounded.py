#!python
"""Unittesting for the pystokes module. Run as python -m unittest pystokes.test."""
import sys
import pystokes
import unittest
import inspect
import numpy as np
import scipy as sp

class UnboundedTest(unittest.TestCase):
    
    def test_rbm(self):
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
                       msg=f"Stokes law is not satisfied")


if __name__ == '__main__':
    unittest.main()
