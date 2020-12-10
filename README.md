![Imagel](https://raw.githubusercontent.com/rajeshrinet/pystokes/master/examples/banner.png)


## PyStokes: phoresis and Stokesian hydrodynamics in Python  [![Binder](https://mybinder.org/badge_logo.svg)](https://mybinder.org/v2/gh/rajeshrinet/pystokes/master?filepath=binder) ![Installation](https://github.com/rajeshrinet/pystokes/workflows/CI/badge.svg) ![Notebooks](https://github.com/rajeshrinet/pystokes/workflows/notebooks/badge.svg) [![Documentation Status](https://readthedocs.org/projects/pystokes/badge/?version=latest)](https://pystokes.readthedocs.io/en/latest/?badge=latest) [![DOI](https://joss.theoj.org/papers/10.21105/joss.02318/status.svg)](https://doi.org/10.21105/joss.02318) [![PyPI](https://img.shields.io/pypi/v/pystokes.svg)](https://pypi.python.org/pypi/pystokes) [![Python Version](https://img.shields.io/pypi/pyversions/pystokes)](https://pypi.org/project/pystokes) [![Downloads](https://pepy.tech/badge/pystokes)](https://pepy.tech/project/pystokes)  

[About](#about) 
| [Blog](https://rajeshrinet.github.io/pystokes-blog/)
| [News](#news) 
| [Installation](#installation) 
| [Documentation](https://pystokes.readthedocs.io/en/latest/) 
| [Examples](#examples) 
| [Publications ](#publications)
| [Gallery](https://github.com/rajeshrinet/pystokes/wiki/Gallery)
| [Support](#support) 
| [License](#license)



## About

[PyStokes](https://github.com/rajeshrinet/pystokes) is a numerical library for phoresis and Stokesian hydrodynamics in Python. It uses a grid-free method, combining the integral representation of Laplace and Stokes equations, spectral expansion, and Galerkin discretization, to compute phoretic and hydrodynamic interactions between spheres with slip boundary conditions on their surfaces. The library also computes suspension scale quantities, such as rheological response, energy dissipation and fluid flow. The computational cost is quadratic in the number of particles and upto 1e5 particles have been accommodated on multicore computers. The library has been used to model suspensions of **microorganisms**,  **synthetic autophoretic particles** and **self-propelling droplets**. 

Please read the PyStokes [paper](https://doi.org/10.21105/joss.02318) and [Wiki](https://github.com/rajeshrinet/pystokes/wiki) before you use PyStokes for your research. Included below are some examples from [PyStokes Gallery](https://github.com/rajeshrinet/pystokes/wiki/Gallery): 

### Periodic orbits of active particles

![Image](https://raw.githubusercontent.com/rajeshrinet/pystokes-misc/master/gallery/2_volvox.gif)

Our work shows that the oscillatory dynamics of a pair of active particles near
a boundary, best exemplified by the fascinating dance of the green algae
*Volvox*, can be understood in terms of Hamiltonian mechanics, even though the
system does not conserve energy. Read more in the [PyStokes Gallery](https://github.com/rajeshrinet/pystokes/wiki/Gallery).
<br>

### Crystallization at a plane no-slip surface
It is well-known that crystallization of colloids approximating hard spheres
is due, paradoxically, to the higher entropy of the ordered crystalline state
compared to that of  the disordered liquid state. Out of equilibrium, no such general
principle is available to rationalize crystallization. Here, we identify a new non-equilibrium mechanism, associated with entropy production rather than entropy gain, which drives crystallization of active colloids near plane walls. Read more in the [PyStokes Gallery](https://github.com/rajeshrinet/pystokes/wiki/Gallery).


![Crystallization of active colloids](https://raw.githubusercontent.com/rajeshrinet/pystokes/master/examples/crystallite.gif) 


## News
**26th July 2019** -- PyStokes can compute hydrodynamic *and* phoretic interactions in autophoretic suspensions.  


## Installation
You can take PyStokes for a spin **without installation**: [![Binder](https://mybinder.org/badge_logo.svg)](https://mybinder.org/v2/gh/rajeshrinet/pystokes/master?filepath=binder). Please be patient while [Binder](https://mybinder.org/v2/gh/rajeshrinet/pystokes/master?filepath=binder) loads.

### From a checkout of this repo

#### Install PyStokes and an extended list of dependencies using 
 
```bash
>> git clone https://github.com/rajeshrinet/pystokes.git
>> cd pystokes
>> pip install -r requirements.txt
>> python setup.py install
```


####  Install PyStokes and its dependencies in an [environment](https://github.com/rajeshrinet/pystokes/blob/master/environment.yml) named "pystokes" via [Anaconda](https://docs.conda.io/projects/continuumio-conda/en/latest/user-guide/install/index.html)


```bash
>> git clone https://github.com/rajeshrinet/pystokes.git
>> cd pystokes
>> make env
>> conda activate pystokes
>> make
```

### Via pip

Install the latest [PyPI](https://pypi.org/project/pystokes) version

```bash
>> pip install pystokes
```


### Testing
Test installation and running

```bash
>> cd tests
>> python shortTests.py
```

Long test of example notebooks 

```bash
>> cd tests
>> python notebookTests.py
```


## Examples


```Python
# Example 1: Flow field due to $2s$ mode of active slip
import pystokes, numpy as np, matplotlib.pyplot as plt

# particle radius, self-propulsion speed, number and fluid viscosity
b, eta, Np = 1.0, 1.0/6.0, 1

# initialize
r, p = np.array([0.0, 0.0, 3.4]), np.array([0.0, 1.0, 0])
V2s  = pystokes.utils.irreducibleTensors(2, p)

# space dimension , extent , discretization
dim, L, Ng = 3, 10, 64;

# instantiate the Flow class
flow = pystokes.wallBounded.Flow(radius=b, particles=Np, viscosity=eta, gridpoints=Ng*Ng)

# create grid, evaluate flow and plot
rr, vv = pystokes.utils.gridYZ(dim, L, Ng)
flow.flowField2s(vv, rr, r, V2s)  
pystokes.utils.plotStreamlinesYZsurf(vv, rr, r, offset=6-1, density=1.4, title='2s')
```

```Python
#Example 2: Phoretic field due to active surface flux of l=0 mode
import pystokes, numpy as np, matplotlib.pyplot as plt
# particle radius, fluid viscosity, and number of particles
b, eta, Np = 1.0, 1.0/6.0, 1

#initialise
r, p = np.array([0.0, 0.0, 5]), np.array([0.0, 0.0, 1])
J0 = np.ones(Np)  # strength of chemical monopolar flux

# space dimension , extent , discretization
dim, L, Ng = 3, 10, 64;

# instantiate the Flow class
phoreticField = pystokes.phoretic.unbounded.Field(radius=b, particles=Np, phoreticConstant=eta, gridpoints=Ng*Ng)

# create grid, evaluate phoretic field and plot
rr, vv = pystokes.utils.gridYZ(dim, L, Ng)
phoreticField.phoreticField0(vv, rr, r, J0)  
pystokes.utils.plotContoursYZ(vv, rr, r, density=.8, offset=1e-16,  title='l=0') 
```

Other examples include
* [Irreducible Active flows](https://github.com/rajeshrinet/pystokes/blob/master/examples/ex1-unboundedFlow.ipynb)
* [Effect of plane boundaries on active flows](https://github.com/rajeshrinet/pystokes/blob/master/examples/ex2-flowPlaneSurface.ipynb)
* [Active Brownian Hydrodynamics near a plane wall](https://github.com/rajeshrinet/pystokes/blob/master/examples/ex3-crystalNucleation.ipynb)
* [Flow-induced phase separation at a plane surface](https://github.com/rajeshrinet/pystokes/blob/master/examples/ex4-crystallization.ipynb)
* [Irreducible autophoretic fields](https://github.com/rajeshrinet/pystokes/blob/master/examples/ex5-phoreticField.ipynb)
* [Autophoretic arrest of flow-induced phase separation](https://github.com/rajeshrinet/pystokes/blob/master/examples/ex6-arrestedCluster.ipynb)


## Selected publications

* [PyStokes: phoresis and Stokesian hydrodynamics in Python](https://doi.org/10.21105/joss.02318), R Singh and R Adhikari, **Journal of Open Source Software**, 5(50), 2318, (2020). *(Please cite this paper if you use PyStokes in your research)*.

* [Controlled optofluidic crystallization of colloids tethered at interfaces](https://journals.aps.org/prl/abstract/10.1103/PhysRevLett.125.068001), A Caciagli, R Singh, D Joshi, R Adhikari, and E Eiser, **Physical Review Letters** 125 (6), 068001 (2020)

* [Periodic Orbits of Active Particles Induced by Hydrodynamic Monopoles](https://journals.aps.org/prl/abstract/10.1103/PhysRevLett.124.088003), A Bolitho, R Singh, R Adhikari, **Physical Review Letters** 124 (8), 088003 (2020)

* [Competing phoretic and hydrodynamic interactions in autophoretic colloidal suspensions](https://aip.scitation.org/doi/full/10.1063/1.5090179), R Singh, R Adhikari, and ME Cates, **The Journal of Chemical Physics** 151, 044901 (2019)

* [Generalized Stokes laws for active colloids and their applications](https://iopscience.iop.org/article/10.1088/2399-6528/aaab0d), R Singh and R Adhikari, **Journal of Physics Communications**, 2, 025025 (2018)


* [Flow-induced phase separation of active particles is controlled by boundary conditions](https://www.pnas.org/content/115/21/5403), S Thutupalli, D Geyer, R Singh, R Adhikari, and HA Stone, **Proceedings of the National Academy of Sciences**, 115, 5403 (2018)  

* [Universal hydrodynamic mechanisms for crystallization in active colloidal suspensions](https://doi.org/10.1103/PhysRevLett.117.228002), R Singh and R Adhikari,  **Physical Review Letters**, 117, 228002 (2016)

See full publication list [here](https://github.com/rajeshrinet/pystokes/wiki/Publications).

## Support

* For help with and questions about PyStokes, please post to the [pystokes-users](https://groups.google.com/forum/#!forum/pystokes) group.
* For bug reports and feature requests, please use the [issue tracker](https://github.com/rajeshrinet/pystokes/issues) on GitHub.

## License
We believe that openness and sharing improves the practice of science and increases the reach of its benefits. This code is released under the [MIT license](http://opensource.org/licenses/MIT). Our choice is guided by the excellent article on [Licensing for the scientist-programmer](http://www.ploscompbiol.org/article/info%3Adoi%2F10.1371%2Fjournal.pcbi.1002598). 
		
