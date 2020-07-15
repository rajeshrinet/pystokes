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

        a, N, eta = 1, 1, 1 
        mu = 1/(6*np.pi*a*eta)
        
        V1 = mu*F
        V2 = 0*mu*F
        
        uRbm = pystokes.unbounded.Rbm(a, N, eta)
        uRbm.mobilityTT(V2, r, F)
        
        diff = np.absolute(V1[2] - V2[2])
        self.assertTrue((np.asarray(diff) < 0.001).all(),
                       "Stokes law for translation is not satisfied")


    def test_rotation(self):
        r = np.array([0,0,0.])
        T = np.array([0,0,1.])

        a, N, eta = 1, 1, 1 
        mu = 1/(8*np.pi*a**3*eta)

        W1 = mu*T
        W2 = 0*mu*T
        
        uRbm = pystokes.unbounded.Rbm(a, N, eta)
        uRbm.mobilityRR(W2, r, T)
        
        diff = np.absolute(W1[2] - W2[2])
        self.assertTrue((np.asarray(diff) < 0.001).all(),
                       "Stokes law for rotation is not satisfied")



class WallBoundedTest(unittest.TestCase):
    

    def test_parallelTranslation(self):
        r = np.array([0,0,1.])
        F = np.array([0,1,0.])

        a, N, eta = 1, 1, 1 
        mu = 1/(6*np.pi*a*eta)
        mu = mu*(1- 9./16 + 1/8)  # add the standard wall-correction

        V1 = mu*F
        V2 = 0*mu*F

        uRbm = pystokes.wallBounded.Rbm(a, N, eta)
        uRbm.mobilityTT(V2, r, F)
        
        diff = np.absolute(V1[1] - V2[1])
        self.assertTrue((np.asarray(diff) < 0.001).all(),
                       "Stokes law for translation || to wall is not satisfied")


    def test_perpTranslation(self):
        r = np.array([0,0,1.])
        F = np.array([0,0,1.])

        a, N, eta = 1, 1, 1 
        mu = 1/(6*np.pi*a*eta)
        mu = mu*(1 - 9/8. + 1./2)  # add the standard wall-correction

        V1 = mu*F
        V2 = 0*mu*F

        uRbm = pystokes.wallBounded.Rbm(a, N, eta)
        uRbm.mobilityTT(V2, r, F)
        
        diff = np.absolute(V1[2] - V2[2]) 
        self.assertTrue((np.asarray(diff) < 0.001).all(),
                       "Stokes law for translation perp to wall is not satisfied")
   



class InterfaceTest(unittest.TestCase):
    

    def test_parallelTranslation(self):
        r = np.array([0,0,1.])
        F = np.array([0,1,0.])

        a, N, eta = 1, 1, 1 
        mu = 1/(6*np.pi*a*eta)
        mu = mu*(1 + 3./8 + 1/16)  # add the standard wall-correction

        V1 = mu*F
        V2 = 0*mu*F

        uRbm = pystokes.interface.Rbm(a, N, eta)
        uRbm.mobilityTT(V2, r, F)
        
        diff = np.absolute(V1[1] - V2[1])
        self.assertTrue((np.asarray(diff) < 0.001).all(),
                       "Stokes law for translation || to wall is not satisfied")


    def test_perpTranslation(self):
        r = np.array([0,0,1.])
        F = np.array([0,0,1.])

        a, N, eta = 1, 1, 1 
        mu = 1/(6*np.pi*a*eta)
        mu = mu*(1 - 3/4. + 1./8)  # add the standard wall-correction

        V1 = mu*F
        V2 = 0*mu*F

        uRbm = pystokes.interface.Rbm(a, N, eta)
        uRbm.mobilityTT(V2, r, F)
        
        diff = np.absolute(V1[1] - V2[1])
        self.assertTrue((np.asarray(diff) < 0.001).all(),
                       "Stokes law for translation perp to wall is not satisfied")
   



class PeriodicTest(unittest.TestCase):


    def test_effectiveMobility(self):
        a, eta, N = 1.0, 1.0/6, 1
        v = np.zeros(3*N)
        r = np.zeros(3*N)
        F = np.zeros(3*N); F[2]=-1
        
        ll = ((4*np.pi/3)**(1.0/3))/0.3   # length of simulation box
        pRbm = pystokes.periodic.Rbm(a, N, eta, ll)

        pRbm.mobilityTT(v, r, F)

        mu=1.0/(6*np.pi*eta*a)
        diff = -v[2]/mu - 0.498
        self.assertTrue((np.asarray(diff) < 0.002).all(),
                       "Effective mobility does not match Zick & Homsy (1982)")


class ForcesTest(unittest.TestCase):


    def test_lennardJones(self):
        N = 2
        r = np.zeros(3*N);  r[0]=0; r[1]=3.2 # minimum of LJ is 3
        F = np.zeros(3*N); 

        forces  = pystokes.forceFields.Forces(particles=N)
        forces.lennardJones(F,r);

        diff = np.absolute(F)
        self.assertTrue((np.asarray(diff) < 0.0001).all(),
                       "Lennard Jones extend beyong its cut-off")

    
    def test_harmonicConfinement(self):
        N = 1
        r = np.zeros(3*N);  r[0]=0; r[1]=1; r[2]=2
        F = np.zeros(3*N); 
        k = 5 # stiffness of trap 

        forces  = pystokes.forceFields.Forces(particles=N)
        forces.harmonicConfinement(F,r, k);

        diff = np.absolute(F + k*np.arange(3))
        self.assertTrue((np.asarray(diff) < 0.0001).all(),
                        "harmonicConfinement is not working")

    

    def test_opticalConfinement(self):
        N = 1
        r = np.zeros(3*N);  r[0]=0; r[1]=1; r[2]=2
        F = np.zeros(3*N); 
        k = np.array([4.]) # stiffness of trap 

        forces  = pystokes.forceFields.Forces(particles=N)
        forces.opticalConfinement(F,r,np.ones(3*N), k);
        

        diff = np.absolute(F + k*(np.arange(3*N)-np.ones(3*N)))
        self.assertTrue((np.asarray(diff) < 0.0001).all(),
                        "Optical trapping is not right")

    def test_spring(self):
        N = 2
        r = np.zeros(3*N);  r[0]=0; r[1]=3; 
        F = np.zeros(3*N); 

        bondLength=2
        springModulus=2

        forces  = pystokes.forceFields.Forces(particles=N)
        forces.spring(F, r, bondLength, springModulus);
        
        dx=r[1]-r[0]
        F1 = np.zeros(3*N);  F1[0]=springModulus*(dx-bondLength); F1[1]=-F[0]
        diff = np.absolute(F - F1)
        self.assertTrue((np.asarray(diff) < 0.0001).all(),
                        "Spring is not right")

    
    def test_bottomHeaviness(self):
        N = 1
        p = np.zeros(3*N);  p[0]=1.
        T = np.zeros(3*N); 

        torques  = pystokes.forceFields.Torques(particles=N)
        torques.bottomHeaviness(T, p);

        diff = np.absolute(T[1]-1)
        self.assertTrue((np.asarray(diff) < 0.0001).all(),
                        "Bottom heaviness is not right")
    



if __name__ == '__main__':
    unittest.main()
