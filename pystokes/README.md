## PyStokes: Library to compute stokes flows around active colloids

In this library, we simulate the motion of spherical active colloids and compute external flow field produced by them. The activity of the colloids is represented by a slip velocity on their surface. Eq. (15a) of the [paper](https://arxiv.org/pdf/1603.05735.pdf) gives the expression for the fluid flow, while Eq. (23) gives the expression for the rigid body motion. These are given in terms of a sum over all the irreducible (*lσ*) modes of the active slip. In the code, we compute the contributions due to each irreducible mode separately for modularity. Here is a correspondence for the nomenclature

* *lσ = 1s* : stokeslet
* *lσ = 2s* : stresslet
* *lσ = 2a* : rotlet
* *lσ = 3s* : septlet
* *lσ = 3a* : vortlet
* *lσ = 3t* : potDipole 
* *lσ = 4a* : spinlet
 
### Installation:
```
python setup.py install
```

The main dependencies are

* Python 2.7
* Numpy  1.11
* Cython 0.24

It is suggested to have the given versions of them respectively.




### Examples


Examples codes are added in the [example](https://github.com/rajeshrinet/pystokes/tree/master/examples) directory. Examples include
* [Fluid flow](https://github.com/rajeshrinet/pystokes/tree/master/examples/streamplots): Computation of exterior flow produced by the leading slip modes of active colloids in [unbounded](https://github.com/rajeshrinet/pystokes/blob/master/examples/streamplots/notebooks/unboundedFlow.ipynb), and [periodic](https://github.com/rajeshrinet/pystokes/blob/master/examples/streamplots/notebooks/periodic.ipynb) geometries of Stokes flow.
* [Convective rolls of active colloids in a harmonic trap](https://github.com/rajeshrinet/pystokes/blob/master/examples/unbounded/activeColloidsSingleTrap/convectiveRolls.ipynb): Here, we show that one-body force and two body torque leads to sustained convective rolls of active colloids in a harmonic trap.
* [Synchronization of active colloids in a two-dimensional lattice of harmonic
  traps](https://github.com/rajeshrinet/pystokes/blob/master/examples/unbounded/activeColloidsLatticeTraps/holographicTrap.ipynb): In this example we study the dynamics of active colloids in lattices of harmonic traps.
* [Crowley instability](https://github.com/rajeshrinet/pystokes/blob/master/examples/periodic/passiveColloids/crowleyInstability.py): Demonstration of Crowley instability in a lattice of sedimenting passive colloids
* [Mobility of a sedimenting lattice](https://github.com/rajeshrinet/pystokes/blob/master/examples/periodic/passiveColloids/mobilitySedimentingLattice.ipynb): Computation of mobility of sedimenting lattice of passive colloids as a function of volume fraction. Comparison with exact results of Zick & Homsy (1982) is also made.
* [Mollified irreducible multipole approach (MIMA)](https://github.com/rajeshrinet/pystokes/tree/master/examples/mima): MIMA is a Stokes solver which resolves the fluid degrees of freedom. Computation of fluid flow in 2D and 3D domain of Stokes flow using MIMA has been considered here. 

Some representative examples are given below.


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

pRbm = pystokes.periodic.Rbm(a, Np, L)    # instantiate the classes
ff = pyforces.ForceFields.Forces(Np)

print 'Initial velocity', v

ff.sedimentation(F, g=10)            # call the Sedimentation module of pyforces
pRbm.stokesletV(v, r, F)               # and StokesletV module of pystokes

print 'Updated velocity', v
```

### Example 2 : computing flow due to particles.
```
import pystokes
import pyforces
import numpy as np

a, Np, dim = 1, 1000, 3              # Radius, No of particle and dimension
S0, D0 = 1, 1                        # Multipole Strength 

r = np.zeros(dim*Np)                 # position vector of particles
p = np.zeros(dim*Np)                 # orientation vector of particles
v = np.zeros(dim*Np)                 # velocity vector of particles
F = np.zeros(dim*Np)                 # forces on particles
S = np.zeros(5*Np)                   # stresslets on particles
D = np.zeros(dim*Np)                 # potential dipoles on particles

Nt = 10                              # number of field points
rt = np.random.rand(dim*Nt)*100      # field point locations
vt = np.zeros(dim*Nt)                # velocity at field points


flwpnts = pystokes.periodic.Flow(a, Np, Nt) # instantiate the classes
ff = pyforces.ForceFields.Forces(Np)

r[0: Np] = np.linspace(0,3*Np,Np)    # distribute particles on a line along the x-axis 
ff.sedimentation(F, g=10)            # add forces in the z-direction

p[0: Np] = 1                         # orient particles along x-axis

# parametrize the stresslets uni-axially
S[    0:  Np] = S0*(p[0:Np]*p[0:Np] - 1/3)
S[  Np :2*Np] = S0*(p[Np:2*Np]*p[Np:2*Np] - 1/3)
S[2*Np :3*Np] = S0*(p[0:Np]*p[Np:2*Np])
S[3*Np :4*Np] = S0*(p[0:Np]*p[2*Np:3*Np])
S[4*Np :5*Np] = S0*(p[Np:2*Np]*p[2*Np:3*Np])

# parametrize the dipoles uniaxially
D[::] = D0*p[::]

print 'Initial flow at rt' vt

flwpnts.stokesletV(vt, rt, r, F)     # add flow due to stokeslet
flwpnts.stressletV(vt, rt, r, S)     # add flow due to stresslet
flwpnts.potDipoleV(vt, rt, r, D)     # add flow due to potential dipole

print 'Updated flow at rt' vt
```


### Example 3 : Grid based simulation Technique.
```
import pystokes
import pyforces
import numpy as np

Lx, Ly, Lz  = 128, 128, 128          # extent of the box
Nx, Ny, Nz  = 128, 128, 128          # grid division 

a, Np, dim = 1, 1000, 3              # Radius, No of particle and dimension
S0, D0 = 1, 1                        # Multipole Strength 

r = np.zeros(dim*Np)                 # position vector of particles
V = np.zeros(dim*Np)                 # Velocity vector of particles
p = np.zeros(dim*Np)                 # orientation vector of particles
F = np.zeros(dim*Np)                 # forces on particles
S = np.zeros(5*Np)                   # stresslets on particles
D = np.zeros(dim*Np)                 # potential dipoles on particles
vv = np.zeros(dim*Nx*Ny*Nz)          # velocity at grid points

pmima = pystokes.mima.flow(a, Np, Lx, Ly, Lz, Nx, Ny, Nz)   # instantiate the classes
ff = pyforces.ForceFields.Forces(Np)

ff.sedimentation(F, g=10)                                   # call the Sedimentation module of pyforces
pmima.stokesletV(vv, r, F, sigma = 3.0, NN = 3*sigma)       # update grid velocity

print 'Initial velocity', V

pmima.interpolate(V, r, vv, sigma = 3.0, NN = 3*sigma)      # interpolate back velocity of the particles

print 'Final velocity', V
```
