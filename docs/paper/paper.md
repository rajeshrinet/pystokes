---
title: 'PyStokes: Phoresis and Stokesian hydrodynamics in Python'
tags:
  - Python
  - Cython
  - Active particles
  - Stokes flow
  - Hydrodynamic interactions
  - Phoretic interactions
authors:
  - name: Rajesh Singh
    orcid: 0000-0003-0266-9691
    affiliation: 1
  - name: Ronojoy Adhikari
    affiliation: 1
affiliations:
 - name: DAMTP, Centre for Mathematical Sciences, University of Cambridge, Wilberforce Road, Cambridge CB3 0WA, UK
   index: 1
date: 6 June 2020
bibliography: paper.bib
---

# Summary

The PyStokes library allows to study active particles in a Stokesian fluid [@kim2005]. Active particles are distinguished by their ability to produce flow, and thus motion, in the absence of external forces or torques on them. Examples of active particles include auto-phoretic synthetic particles [@ebbens2010pursuit] and microorganisms [@brennen1977]. An auto-phoretic particle self-propels due to non-equilibrium processes, such as chemical reactions, on its surface [@anderson1989colloid]. Examples of phoresis (phoretic field) are electrophoresis (electric field), diffusiophoresis (concentration of chemical field), thermophoresis (temperature field) etc [@anderson1989colloid]. A particle is "active" if it creates the phoretic field itself (for example, by a built-in chemical asymmetry on its surface), and the resulting phenomena is called self-phoresis [@ebbens2010pursuit; @zhang2017active]. 


![Library structure and equations that determine the hydrodynamic and phoretic interactions between active particles in a three-dimensional domain $V$. The equations need to be solved with boundary conditions on the surface $S_{i}$ of the particles. Particle indices are $i=1,\ldots,N$ and harmonic indices are $l=1,2,\ldots$ and $\sigma=s,a,t$. \label{fig:figS}](FigSchema.png)

Hydrodynamic and phoretic interactions between active particles in a viscous fluid are central to the understanding of their collective dynamics. 
Under experimentally relevant conditions, the motion of the fluid is governed by the Stokes equation and that of the phoretic field, if one is present, by the Laplace equation. 
The “activity” appears in these equations as boundary conditions on the particle surfaces that prescribe the slip velocity \begin{equation}\boldsymbol{v}^{\mathcal{A}}\end{equation} in the Stokes equation and flux \begin{equation}\j^{\mathcal A}\end{equation} of the phoretic field in the Laplace equation. 
The slip velocity and the phoretic flux are related by a linear constitutive law that can be derived from a detailed analysis of the boundary layer physics [@anderson1989colloid]. 
The Stokes and Laplace equations are coupled by this linear constitutive law only at the particle boundaries. 
The linearity of the governing equations and of the coupling boundary conditions allows for a formally exact solution of the problem of determining 
the force per unit area on the particle surfaces. This formally exact solution can be approximated to any desired degree of accuracy by a 
truncated series expansion in a complete basis of functions on the particle boundaries. This, in turn, leads to an efficient and accurate numerical 
method for computing hydrodynamic and phoretic interactions between active particles [@singh2015many; @singh2018generalized; @singh2019competing].



Figure \autoref{fig:figS} shows the the overall organization of the library. To solve Laplace equation, the library takes input a set of expansion coefficients \begin{equation}\mathbf{J}_{i}^{(l)}\end{equation} for the prescribed active flux \begin{equation}j^{\mathcal A}\end{equation} on the surface of the $i$-th particle and computes the expansion coefficients of the surface concentration. The active slip velocity \begin{equation}\boldsymbol{v}^{\mathcal{A}}\end{equation} is obtained from this using the linear coupling relation \begin{equation}\boldsymbol{v}^{\mathcal{A}}=\mu_{c}\boldsymbol{\nabla}_{s}c\end{equation} [@anderson1989colloid]. The library outputs the expansion coefficients \begin{equation}\mathbf{V}_{i}^{(l\sigma)}\end{equation} of the active slip. To solve Stokes equation, the library takes as input these expansion coefficients, which may also be specified independently, and any body forces \begin{equation}\mathbf{F}_{i}^{B}\end{equation} and body torques \begin{equation}\mathbf{T}_{i}^{B}\end{equation} acting on the particles and returns their rigid body motion in terms of the velocities \begin{equation}\mathbf{V}_{i}\end{equation} and angular velocities \begin{equation}\boldsymbol{\Omega}_{i}\end{equation}. The PyStokes library can be used to compute the hydrodynamically interacting motion of squirming particles where the slip can be specified independently of a phoretic field, or the dynamics of passive suspensions where the slip vanishes and forces and torques are prescribed. 

Hydrodynamic interactions with no-slip boundary conditions can be computed in terms in mobility matrices @kim2005]. There exist libraries which computes these matrices [@hinsen1995; @libstokes]. As described above, we model active particles by a slip boundary conditions and the resulting hydrodynamic interactions are in terms of propulsion tensors [@singh2015many]. To the best of our knowledge PyStokes is the only numerical implementation of propulsion matrices to model hydrodynamic interactions of active particles. The PyStokes library has been used to model suspensions of microorganisms [@bolitho2020; @singh2016crystallization], 
synthetic autophoretic particles [@singh2016crystallization; @singh2019competing] and 
self-propelling droplets [@thutupalli2018FIPS].
Our software implementation uses a polylgot programming approach that combines the readability of Python with the speed of Cython and retains the advantages of a high-level, 
dynamically typed, interpreted language without sacrificing performance.  


The principal features that set this method apart are (a) the restriction of independent fluid and phoretic degrees of freedom to the particle boundaries 
(b) the freedom from grids, both in the bulk of the fluid and on the particle boundaries and 
(c) the ability to handle, within the same numerical framework, a wide variety of geometries and boundary conditions, 
including unbounded volumes, volumes bounded by plane walls or interfaces, periodic volumes and, indeed, 
any geometry-boundary condition combination for which the Green's functions of the governing equations are simply evaluated.


# Methods

Our method relies on the reduction of linear elliptic partial differential equations to systems of linear algebraic equations using following steps.  

![Key mathematical steps underpinning the PyStokes codebase.\label{fig:example}](figure.png)

The first step is the representation of the solution of an elliptic partial differential equation (PDE) in a three-dimensional volume $V$ as an integral over the 
boundary of the surface $S$ [@fkg1930bandwertaufgaben; @ladyzhenskaya1969; @youngren1975stokes; @zick1982stokes; @pozrikidis1992; @muldowney1995spectral; @cheng2005heritage; @singh2015many]. 
For the Laplace equation, this is the classical theorem of Green [@jackson1962classical]; for the Stokes equation, it is the generalization obtained by 
Lorentz [@fkg1930bandwertaufgaben; @lorentz1896eene; @ladyzhenskaya1969]. The integral representation leads to a linear integral equation that provides a 
functional relation between the field and its flux on $S$. Thus, if the surface flux in the Laplace equation is specified, the surface concentration is determined 
by the solution of the Laplace boundary integral equation. Similarly, if the surface velocity in the Stokes equation is specified, the surface traction is determined 
by the solution of the Stokes boundary integral equation. This transformation of the governing PDEs is the most direct way of relating boundary conditions (surface flux, slip velocities) 
to boundary values (surface concentration, surface traction). It reduces the dimensionality of the problem from a three-dimensional one in $V$ to a two-dimensional one on $S$. 
The second step is the spectral expansion of the field and its flux in terms of global basis functions on $S$. We use the geometry-adapted tensorial spherical harmonics, 
which provide an unified way of expanding both scalar and vector quantities on the surface of a sphere. These functions are both complete and orthogonal and provide representations of 
the three-dimensional rotation group [@hess2015tensors]. Thus, symmetries of the active boundary conditions can be represented in a straightforward and transparent manner. 
The third step is the discretization of the integral equation using the procedure of Ritz and Galerkin [@boyd2001chebyshev; @finlayson1966method], which reduces it to an 
infinite-dimensional self-adjoint linear system in the expansion coefficients. This exploits the orthogonality of the basis functions on the sphere. The matrix elements of 
the linear system can be evaluated analytically in terms of the Green's functions of the respective elliptic equations. The fourth step is the truncation of the 
infinite-dimensional linear system to a finite-dimensional one that can be solved by standard methods of linear algebra adapted for self-adjoint systems [@saad2003iterative]. 
Analytical solution can be obtained by Jacobi iteration, which is equivalent to Smoluchowski's method of reflections. Numerical solutions can be obtained by the conjugate 
gradient method, at a cost quadratic in the number of unknowns. From this solution, we can reconstruct the field and the flux on the boundary, use these to determine the 
fields in the bulk, and from there, compute derived quantities. These steps have been elaborated in several 
papers [@singh2015many; @singh2016crystallization; @singh2017fluctuation; @singh2018generalized; @singh2019competing; @bolitho2020] and we do not repeat them in detail here. 


PyStokes can be used to compute fluid flow on a given set of grid points 
```python
Flow = pystokes.unbounded.Flow(radius=1, particles=1, viscosity=1, 
        gridpoints=4096) 
```
or rigid body motion of hydrodynamically interacting colloidal spheres
```python
Rbm = pystokes.unbounded.Rbm(radius=1, particles=1024, viscosity=1) 
```
in various geometries of Stokes flow, by replacing `unbounded` with `wallBounded`, `interface`, `periodic`, etc. The above instantiation can then be used to compute `Flow` and `Rbm` due to body forces, body torques, and each irreducible mode of the surface slip.  PyStokes can also be used to obtain phoretic field 
```python
phoreticField = pystokes.phoreticUnbounded.Field(radius=1, particles=1, 
                phoreticConstant=1, gridpoints=4096)
```
and phoretic motion (phoresis)
```python
phoresis = pystokes.phoreticUnbounded.Phoresis(radius=1, particles=1024, 
            phoreticConstant=1)
```
`pystokes.forceFields` contains implementation of force fields commonly used in colloidal systems for completeness. The arXiv preprint [@singh2019Hydrodynamic] of this article contains more detailed documentation and examples.
  

# Acknowledgements

We thank A Banerjee, ME Cates, S Date, A Donev, E Eiser, D Frenkel, S Ghose, R
Goldstein, J Hinch, A Laskar, AJC Ladd, RK Manna, I Pagonabarraga, DJ Pine, T
Pradeep, R Simon, HA Stone, G Subramanian, PB Sunil Kumar, and S Thutupalli 
for useful discussions; Rajeev Singh and Abhrajit Laskar for code contributions in the initial stages of development. 
This work was funded in parts by the European Research Council under the EU’s Horizon 2020 Program, Grant No. 740269; 
a Royal Society-SERB Newton International Fellowship to RS; and an Early Career Grant to RA from the Isaac Newton Trust.

# References
