## Force Fields for the simulations of colloidal particles
This is a library to be used along with pystokes.


### Installation:
```
python setup.py install

```

### Example 1 : computing rigid body motion of particles.

```
import pystokes
import pyforces
import numpy as np

a, Np, dim = 1, 3, 3                 # radius and number of particles and dimension
v = np.zeros(dim*Np)                 # Memory allocation for velocity
r = np.zeros(dim*Np)                 # Position vector of the particles
F = np.zeros(dim*Np)                 # Forces on the particles

r[0], r[1], r[2] = -4, 0 , 4         # x-comp of PV of particles

rm = pystokes.rbm.Periodic(a, Np)    # instantiate the classes
ff = pyforces.ForceFields.Forces(Np)

print 'Initial velocity', v

ff.sedimentation(F, g=10)            # call the Sedimentation module of pyforces
rm.stokesletV(v, r, F)               # and StokesletV module of pystokes

print 'Updated velocity', v
```
