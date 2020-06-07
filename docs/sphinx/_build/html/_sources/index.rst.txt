PyStokes API
==================================
.. image:: ../../examples/banner.png
  :width: 800
  :alt: PyRoss banner



PyStokes is a numerical library for Stokesian hydrodynamics. It uses a grid-free method, combining the integral representation of Stokes equation, spectral expansion, and Galerkin discretization, to compute hydrodynamic interactions between spheres with slip boundary conditions on their surfaces. The library also computes suspension scale quantities, such as rheological response, energy dissipation and fluid flow. The computational cost is quadratic in the number of particles and upto 1e5 particles have been accommodated on multicore computers. The library has been used to model bacterial suspensions, active colloids and autophoretic particles.


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
   mima
   phoreticUnbounded
   phoreticWallBounded
   forceFields
   utils
