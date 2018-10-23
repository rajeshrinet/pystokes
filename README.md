## PyStokes 
[![Binder](https://mybinder.org/badge.svg)](https://mybinder.org/v2/gh/rajeshrinet/pystokes/master?filepath=examples%2FbinderDemo)


[PyStokes](https://gitlab.com/rajeshrinet/pystokes) is a Cython library for computing Stokes flows produced by spheres with Dirichlet (velocity) or Neumann (traction) boundary conditions on their surfaces. The solution of Stokes equation is represented as an integral of the velocity and traction over the particle boundaries. The boundary conditions are expanded in a basis of tensorial spherical harmonics and the minimal set of terms that includes all long-ranged hydrodynamic interactions is retained. A linear system of equations is obtained for the unknown coefficients of the traction (for Dirichlet boundary conditions) or the velocity (for Neumann  boundary conditions). The flow and many-body hydrodynamic interactions are obtained from this solution. 

There are open source libraries, most notably [Hydrolib](http://dirac.cnrs-orleans.fr/plone/software/hydrolib) and [libstokes](https://github.com/kichiki/libstokes), which could have been moulded for our purpose but, in spite of Havoc Pennington's [warnings](http://www106.pair.com/rhp/hacking.html), we decided to write our own library. The codebase is kept lean in line with the philosophy that 

>large codebases are just failed ecosystems. 

We hope our effort was worth the while!

### What does the library compute ? 

The [PyStokes](https://gitlab.com/rajeshrinet/pystokes) library computes:

* **Matrix elements** of the single and double layer boundary integral operators of the Stokes equation over N spheres in a **Galerkin** basis of **tensorial spherical harmonics**. 

* **Flows** that are associated with the expansion of the surface stress (in the single layer) or the surface velocity (in the double layer) in the **Galerkin** basis.

* **Rigid body motion** of the N spheres due to the hydrodynamic interaction mediated by the Stokes flow. 

### What geometries are supported ? 

The [PyStokes](https://gitlab.com/rajeshrinet/pystokes) library currently supports four geometries:

* For **unbounded domains**, the library computes matrix elements, flow, and rigid body motion include all long-ranged terms.

* For **wall-bounded domains**, the Lorentz reflection theorem is applied to the unbounded flows to construct flows that vanish at a plane wall. 

* For **sphere-bounded** domains, the library uses an asymptotic expansion of Oseen's Green's function that vanishes on a sphere. This functionality is planned. 

* For **periodic domains**, O'Brien's method is used to derive an unconditionally convergent expression for the flow. Ewald summation of the resulting terms is implemented to accelerate convergence using Beenakker's method.


###  What about linear solvers ? 

The [PyStokes](https://gitlab.com/rajeshrinet/pystokes) library does not directly compute solutions to the boundary integral equation but only evaluates the terms that arise from a global Galerkin discretization. To solve the linear system, the matrix elements
computed by the library must be passed to iterative linear solvers like [PyKrylov](https://github.com/dpo/pykrylov).

### What about fast multipole accelerations ? 

The [PyStokes](https://gitlab.com/rajeshrinet/pystokes) library defaults to direct summation for matrix elements, flows, and rigid body motion. Direct summation is an ``O(N^2)`` operation and, with current many-core architectures, is feasible for about ``N ~ 1e5``. For larger N, accelerated summation methods like the fast multipole is desirable. Future plans include  supporting one or more of the following kernel independent fast multipole methods:

* [KIFMM3d](http://www.mrl.nyu.edu/~harper/kifmm3d/documentation/publications.html) (Biros et al) - restricted to Green's functions of elliptic 2nd order pdes.
* [BBFMM3d](https://github.com/ruoxi-wang/BBFMM3D) (Darve et al) - general kernels but no parallelization.
* [ScalaFMM](http://scalfmm-public.gforge.inria.fr/doc/) (Inria) - both classic and Darve's kernel independent methods are implemented; parallel version; automatic periodic summations.
*  [FMM template library](https://github.com/ccecka/fmmtl) - well written parallel code, includes Stokes kernel and support for general tensor kernels.

### Where can I read up the theory behind this ? 

The library has a fairly comprehensive documentation which describes the method. Please cite the following papers if you publish research using [PyStokes](https://gitlab.com/rajeshrinet/pystokes).

* [Irreducible Representations Of Oscillatory And Swirling Flows In Active Soft Matter](http://journals.aps.org/prl/abstract/10.1103/PhysRevLett.112.118102), *Physical Review Letters, 112, 118102 (2014)* (Irreducible flows for a single sphere).

* [Many-body microhydrodynamics of colloidal particles with active boundary layers](https://iopscience.iop.org/1742-5468/2015/6/P06017), *Journal of Statistical Mechanics: Theory and Experiment, P06017 (2015)* (Galerkin solution of boundary integral equation for N spheres).

* [Generalized Stokes laws for active colloids and their applications](https://dx.doi.org/10.1088/2399-6528/aaab0d), *Journal of Physics Communications (2018)* (Generalization of Stokes laws to derive active forces and torques).

* [Universal hydrodynamic mechanisms for crystallization in active colloidal suspensions](https://doi.org/10.1103/PhysRevLett.117.228002) *Physical Review Letters, 117, 228002 (2016)* (Application of generalized Stokes laws to derive universal mechanisms for crystallization of active colloids). 

* [Flow-induced phase separation of active particles is controlled by boundary conditions](https://doi.org/10.1073/pnas.1718807115) *Proceedings of the National Academy of Sciences (2018)* (Application of generalized Stokes laws to derive mechanisms of flow-induced phase separation). 

* [Microhydrodynamics of active colloids](https://www.imsc.res.in/xmlui/handle/123456789/418) *Homi Bhabha National Institute* (A PhD thesis which presents the theory behind PyStokes and its applications. Read more on PyStokes in chapter 10).


### And all this is free ? 

Yes. Both as in speech and beer. We believe strongly that openness and sharing improves the practice of science and increases the reach of its benefits. This code is released under the [MIT license](http://opensource.org/licenses/MIT). Our choice is guided by the excellent article on [Licensing for the scientist-programmer](http://www.ploscompbiol.org/article/info%3Adoi%2F10.1371%2Fjournal.pcbi.1002598). 


### PyForces

PyForces is a library to be used along with PyStokes for computing body forces and torques on the colloids.
