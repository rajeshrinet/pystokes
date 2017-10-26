# Driver code for the problem of active particles in a 2D lattice of harmonic potentials
#
#

from __future__ import division
import numpy as np
import trap, sys
import scipy.io as sio

''' Parameters:
                Np= number of particles, vs=active velocity, 
                k = spring constant of the trap, S0 = stresslet strength
                A = vs/k;   B = S0/k'''

try:
    Np, A, a0, vs   = int(sys.argv[1]), float(sys.argv[2]), float(sys.argv[3]), float(sys.argv[4])
except:
    Np, A, B, vs   = 2, 4, 1, 1
    theta1, theta2 = 90, 300
    print 'No input given. Taking the default values!'

#theta1, theta2 = 90, 300
dim = 3                              # dimensionality of the problem 
eta = 1.0/6                          # viscosity of the fluid simulated
a   = 1                              # radius of the particle       
k = vs/A                             # stiffness of the trap
S0, D0  = 0.01, 0.01                 # strength of the stresslet and potDipole
ljrmin, ljeps  = 3, .01              # lennard-jones parameters
Tf, Npts       = 400000, 5000        # final time and number of points 


# instantiate the class trap for simulating active particles in a harmonic potential
rm = trap.trap(a, Np, vs, eta, dim, S0, D0, k, ljeps, ljrmin)

# module to initialise the system. 
def initialConfig(rp0, trapCentre, theta, a, a0, vs, k, Np):
    '''initialise the system'''
    rr = np.pi*vs*a/k;   #confinement radius
    t1 = np.pi/180       
    
    if Np==1: 
        rp0[0], rp0[1], rp0[2] = 0, 0, 8   # particle 1 Position
        rp0[3], rp0[4], rp0[5] = 0, 0, -1   # Orientation

    elif Np==3:
        t1 = np.pi/180
        trapCentre[0] =  0
        trapCentre[1] =  a0
        trapCentre[2] = -a0
        trapCentre[3] =  0
        trapCentre[4] =  a0
        trapCentre[5] =  a0
        
        theta[0] = 90*t1
        theta[1] = 90*t1
        theta[2] = 90*t1
        for i in range(Np):
            rp0[i     ] =   trapCentre[i     ] + rr*np.cos(theta[i])
            rp0[i+Np  ] =   trapCentre[i+Np  ] + rr*np.sin(theta[i])
            rp0[i+3*Np] = np.cos(theta[i])
            rp0[i+4*Np] = np.sin(theta[i])

    else:
        Np1d = np.int(np.round( (Np)**(1.0/2)))
        nnd = Np1d/2 - 0.5; h0=0
        Np= Np1d*Np1d
        
        #2D initial
        for i in range(Np1d):
            for j in range(Np1d):
                    ii                  = i*Np1d + j
                    trapCentre[ii]      = a0*(-nnd + i)                  
                    trapCentre[ii+Np]   = a0*(-nnd + j)               
                    trapCentre[ii+2*Np] = h0
        
        theta = np.ones(Np)*np.pi/2
        #theta = np.random.random(Np)*np.pi/2
        for i in range(Np):
            rp0[i     ] = trapCentre[i   ] + rr*np.cos(theta[i])
            rp0[i+Np  ] = trapCentre[i+Np] + rr*np.sin(theta[i])
            rp0[i+3*Np] = np.cos(theta[i])
            rp0[i+4*Np] = np.sin(theta[i])


#initialise the system
rp0        = np.zeros(6*Np)
trapCentre = np.zeros(3*Np)
theta      = np.zeros(Np)
initialConfig(rp0, trapCentre, theta, a, a0, vs, k, Np)
rm.initialise(rp0, trapCentre)       

# simulate the resulting system
rm.simulate(Tf, Npts)
