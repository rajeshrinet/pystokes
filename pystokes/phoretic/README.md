This folder contains file to study phoresis of active particles. 

Each file computed the phoretic motion (phoresis) and phoretic field given the colloidal configuration.
The word phoresis derives from Greek word *phoras* which means bearing or "to carry". 

The filenames, as described below, correspond to the boundary conditions in the flow. The boundary conditions are implemented using an appropriate Green's function of the Laplace equation. See details of theory at: [J. Chem. Phys. 151, 044901 (2019)](https://aip.scitation.org/doi/abs/10.1063/1.5090179)

* unbounded.pyx - the phoretic flux vanishes at infinity. 
* wallBounded.pyx - the phoretic flux vanishes at a plane wall, which is located at z=0, such that region of interest is the upper half space, z>0. 

--
Corresponding to each .pyx file, there is a .pxd file. [A .pxd file](https://cython.readthedocs.io/en/latest/src/tutorial/pxd_files.html) contains declaration of cdef classes, methods, etc.  
