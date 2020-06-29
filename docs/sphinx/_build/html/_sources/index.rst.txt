PyStokes API
==================================
.. image:: ../../examples/banner.png
  :width: 700
  :alt: PyRoss banner



`PyStokes <https://github.com/rajeshrinet/pystokes>`_ is a numerical library for Phoresis and Stokesian hydrodynamics in Python. It uses a grid-free method, combining the integral representation of Stokes and Laplace equations, spectral expansion, and Galerkin discretization, to compute phoretic and hydrodynamic interactions between spheres with slip boundary conditions on their surfaces. The library also computes suspension scale quantities, such as rheological response, energy dissipation and fluid flow. The computational cost is quadratic in the number of particles and upto 1e5 particles have been accommodated on multicore computers. The library has been used to model suspensions of **microorganisms**,  **synthetic autophoretic particles** and **self-propelling droplets**. 


Please see installation instructions and more details in the `README.md <https://github.com/rajeshrinet/pystokes/blob/master/README.md>`_ on GitHub. 




API Reference
=============

.. toctree::
   :maxdepth: 1

   
   unbounded
   interface
   wallBounded
   twoWalls
   periodic
   phoreticUnbounded
   phoreticWallBounded
   forceFields
   utils
