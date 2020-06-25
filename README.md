## PyStokes: Phoresis and Stokesian hydrodynamics in Python  [![Binder](https://mybinder.org/badge_logo.svg)](https://mybinder.org/v2/gh/rajeshrinet/pystokes/master?filepath=binder) 
![Installation](https://github.com/rajeshrinet/pystokes/workflows/Installation/badge.svg)
![Notebooks](https://github.com/rajeshrinet/pystokes/workflows/Notebooks/badge.svg)
[![Documentation Status](https://readthedocs.org/projects/pystokes/badge/?version=latest)](https://pystokes.readthedocs.io/en/latest/?badge=latest)
[![PyPI](https://img.shields.io/pypi/v/pystokes.svg)](https://pypi.python.org/pypi/pystokes)
[![Python Version](https://img.shields.io/pypi/pyversions/pystokes)](https://pypi.org/project/pystokes)
[![Downloads](https://pepy.tech/badge/pystokes)](https://pepy.tech/project/pystokes)
![License](https://img.shields.io/github/license/rajeshrinet/pystokes) 

[About](#about) | [News](#news) | [Installation](#installation) |  [Documentation](https://pystokes.readthedocs.io/en/latest/) | [Examples](#examples) | [Publications ](#publications)| [Support](#support) | [License](#license)

![Imagel](https://raw.githubusercontent.com/rajeshrinet/pystokes/master/examples/banner.png)


## About

[PyStokes](https://github.com/rajeshrinet/pystokes) is a numerical library for phoresis and Stokesian hydrodynamics in Python. It uses a grid-free method, combining the integral representation of Laplace and Stokes equations, spectral expansion, and Galerkin discretization, to compute phoretic and hydrodynamic interactions between spheres with slip boundary conditions on their surfaces. The library also computes suspension scale quantities, such as rheological response, energy dissipation and fluid flow. The computational cost is quadratic in the number of particles and upto 1e5 particles have been accommodated on multicore computers. The library has been used to model suspensions of **microorganisms**,  **synthetic autophoretic particles** and **self-propelling droplets**. 

![Crystallization of active colloids](https://raw.githubusercontent.com/rajeshrinet/pystokes/master/examples/crystallite.gif)


## News
**26th July 2019** -- PyStokes can compute hydrodynamic *and* phoretic interactions in autophoretic suspensions.  


## Installation
You can take PyStokes for a spin **without installation**: [![Binder](https://mybinder.org/badge_logo.svg)](https://mybinder.org/v2/gh/rajeshrinet/pystokes/master?filepath=binder). Please be patient while [Binder](https://mybinder.org/v2/gh/rajeshrinet/pystokes/master?filepath=binder) loads.

### Via [Anaconda](https://docs.conda.io/projects/continuumio-conda/en/latest/user-guide/install/index.html)

Install PyStokes and its dependencies in a `pystokes` [environment](https://github.com/rajeshrinet/pystokes/blob/master/environment.yml):

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

### From a checkout of this repo

Install PyStokes and required dependencies using
 
```bash
>> git clone https://github.com/rajeshrinet/pystokes.git
>> cd pystokes
>> pip install -r requirements.txt
>> python setup.py install
```

### Testing
Short test of installation and running

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
phoreticField = pystokes.phoreticUnbounded.Field(radius=b, particles=Np, phoreticConstant=eta, gridpoints=Ng*Ng)

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


## Publications

* [Hydrodynamic and phoretic interactions of active particles in Python](https://arxiv.org/abs/1910.00909), Rajesh Singh and R. Adhikari, arXiv:1910.00909, 2019. *(Please cite this paper if you use PyStokes in your research)*.

* [Competing phoretic and hydrodynamic interactions in autophoretic colloidal suspensions](https://aip.scitation.org/doi/full/10.1063/1.5090179), Rajesh Singh, R. Adhikari, and M. E. Cates, **The Journal of Chemical Physics** 151, 044901 (2019)

* [Generalized Stokes laws for active colloids and their applications](https://iopscience.iop.org/article/10.1088/2399-6528/aaab0d), Rajesh Singh and R. Adhikari, **Journal of Physics Communications**, 2, 025025 (2018)


* [Flow-induced phase separation of active particles is controlled by boundary conditions](https://www.pnas.org/content/115/21/5403), Shashi Thutupalli, Delphine Geyer, Rajesh Singh, R. Adhikari, and Howard A. Stone, **Proceedings of the National Academy of Sciences**, 115, 5403 (2018)  

* [Universal hydrodynamic mechanisms for crystallization in active colloidal suspensions](https://doi.org/10.1103/PhysRevLett.117.228002), Rajesh Singh and R. Adhikari,  **Physical Review Letters**, 117, 228002 (2016)


## Support

* For help with and questions about PyStokes, please post to the [pystokes-users](https://groups.google.com/forum/#!forum/pystokes) group.
* For bug reports and feature requests, please use the [issue tracker](https://github.com/rajeshrinet/pystokes/issues) on GitHub.

## License
We believe that openness and sharing improves the practice of science and increases the reach of its benefits. This code is released under the [MIT license](http://opensource.org/licenses/MIT). Our choice is guided by the excellent article on [Licensing for the scientist-programmer](http://www.ploscompbiol.org/article/info%3Adoi%2F10.1371%2Fjournal.pcbi.1002598). 
