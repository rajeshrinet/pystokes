from __future__ import division
import numpy as np
import opticalTrap
import scipy.io as sio 

#Parameters
a, Np = 1, 2                    # radius, number of particles               
ljrmin, ljeps  = 3.0, 0.0001    # Lennard jones parameters
sForm, tau = 1, 1/8              # signal specification

#dimensionless numbers
lmda1, lmda2, lmda3 = 9060/8, 8156/8, 1040/8            # see the notes

# instantiate the optical trap
rm = opticalTrap.Rbm(a, Np, ljeps, ljrmin, sForm, tau, lmda1, lmda2, lmda3)


# initialise the system. 
x0 = np.zeros(Np*3);       #memory allocation
x0[0], x0[1], x0[2]  = -2.5, 0.0, 0.0         #Initial condition of first particle
x0[3], x0[4], x0[5]  = 2.5, 0.0, 0.0         # second particle displaced by 5a     

## now initialise the system
rm.initialise(x0)


# simulate the resulting system
T, nsteps = 10*tau, 500
filename='Np=2'

rm.simulate(T, nsteps, filename)
