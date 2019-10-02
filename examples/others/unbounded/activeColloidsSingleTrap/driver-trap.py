# Driver code for the problem of active particles in a harmonic potential 
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
    Np, A, B, vs   = int(sys.argv[1]), float(sys.argv[2]), float(sys.argv[3]), float(sys.argv[4])
except:
    Np, A, B, vs   = 100, 4, 1, 1
    print 'No input given. Taking the default values!'

dim = 3                                # dimensionality of the problem 
eta = 1.0/6                            # viscosity of the fluid simulated
a   = 1                                # radius of the particle       
mu  = 1/(6*np.pi*eta*a)                # particle mobility
k = vs/A                               # stiffness of the trap
S0  = B*k                              # strength of the stresslet 
D0 = 0.01
ljrmin, ljeps  = 4.0, .01               # Lennard Jones Parameters
Tf, Npts       = 8000, 2000              # Final time and number of points on which integrator returns the data


# instantiate the class trap for simulating active particles in a harmonic potential
rm = trap.trap(a, Np, vs, eta, dim, S0, D0, k, ljeps, ljrmin)

# initialise the system. Current version supports either Np=2 or initialization on a sphere or on a cube
rm.initialise('sphere')

# simulate the resulting system
rm.simulate(Tf, Npts)
