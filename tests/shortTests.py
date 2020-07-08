"""
Unittesting for pystokes
"""
import sys
import pystokes
import unittest
import inspect
import numpy as np
import scipy as sp
from pystokes.unbounded import Rbm
from pystokes.unbounded import Flow

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
                       "Stokes law for translation is not satisfied")


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
                       "Stokes law for rotation is not satisfied")



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
                       "Stokes law for translation || to wall is not satisfied")


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
                       "Stokes law for translation perp to wall is not satisfied")
   



class InterfaceTest(unittest.TestCase):
    

    def test_parallelTranslation(self):
        r = np.array([0,0,1.])
        F = np.array([0,1,0.])

        a, Np, eta = 1, 1, 1 
        mu = 1/(6*np.pi*a*eta)
        mu = mu*(1 + 3./8 + 1/16)  # add the standard wall-correction

        V1 = mu*F
        V2 = 0*mu*F

        uRbm = pystokes.interface.Rbm(a, Np, eta)
        uRbm.mobilityTT(V2, r, F)
        
        diff = V1[1] - V2[1] 
        self.assertTrue((np.asarray(diff) < 0.001).all(),
                       "Stokes law for translation || to wall is not satisfied")


    def test_perpTranslation(self):
        r = np.array([0,0,1.])
        F = np.array([0,0,1.])

        a, Np, eta = 1, 1, 1 
        mu = 1/(6*np.pi*a*eta)
        mu = mu*(1 - 3/4. + 1./8)  # add the standard wall-correction

        V1 = mu*F
        V2 = 0*mu*F

        uRbm = pystokes.interface.Rbm(a, Np, eta)
        uRbm.mobilityTT(V2, r, F)
        
        diff = V1[2] - V2[2] 
        self.assertTrue((np.asarray(diff) < 0.001).all(),
                       "Stokes law for translation perp to wall is not satisfied")
   



class PeriodicTest(unittest.TestCase):


    def test_effectiveMobility(self):
        a, eta, Np = 1.0, 1.0/6, 1
        v = np.zeros(3*Np)
        r = np.zeros(3*Np)
        F = np.zeros(3*Np); F[2]=-1
        
        ll = ((4*np.pi/3)**(1.0/3))/0.3   # length of simulation box
        pRbm = pystokes.periodic.Rbm(a, Np, eta, ll)

        pRbm.mobilityTT(v, r, F)

        mu=1.0/(6*np.pi*eta*a)
        diff = -v[2]/mu - 0.498
        self.assertTrue((np.asarray(diff) < 0.002).all(),
                       "Effective mobility does not match Zick & Homsy (1982)")


class ForceFieldTest(unittest.TestCase):


    def test_lennardJones(self):
        Np = 2
        r = np.zeros(3*Np);  r[0]=0; r[1]=3.2 # minimum of LJ is 3
        F = np.zeros(3*Np); 

        forces  = pystokes.forceFields.Forces(particles=Np)
        forces.lennardJones(F,r);

        diff = F 
        self.assertTrue((np.asarray(diff) < 0.0001).all(),
                       "Lennard Jones extend beyong its cut-off")

    
    def test_harmonicConfinement(self):
        Np = 1
        r = np.zeros(3*Np);  r[0]=0; r[1]=1; r[2]=2
        F = np.zeros(3*Np); 
        k = 5 # stiffness of trap 

        forces  = pystokes.forceFields.Forces(particles=Np)
        forces.harmonicConfinement(F,r, k);

        diff = F - k*np.arange(3) 
        self.assertTrue((np.asarray(diff) < 0.0001).all(),
                        "harmonicConfinement is not working")

    
    def test_opticalConfinement(self):
        Np = 1
        r = np.zeros(3*Np);  r[0]=0; r[1]=1; r[2]=2
        F = np.zeros(3*Np); 
        k = np.array([4.]) # stiffness of trap 

        forces  = pystokes.forceFields.Forces(particles=Np)
        forces.opticalConfinement(F,r,np.ones(3*Np), k);
        
        print('this', F, k*(np.arange(3*Np)-np.ones(3*Np)))

        diff = F + k*(np.arange(3*Np)-np.ones(3*Np)) 
        self.assertTrue((np.asarray(diff) < 0.0001).all(),
                        "Optical trapping is not right")


if __name__ == '__main__':
    unittest.main()
