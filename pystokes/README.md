This folder contains the core files of the PyStokes library.

Each file computes the rigid body motion (RBM) and flow given the colloidal configuration.
The filenames, as described below, correspond to the boundary conditions in the flow. The boundary conditions are implemented using an appropriate Green's function of Stokes flow. See details at: https://arxiv.org/abs/1910.00909

The main files are:
* unbounded.pyx - the fluid flow only vanishes at infinity. Implemented using Oseen tensor.
* periodic.pyx - Implemented using the Ewald sum of Oseen tensor. The box is of size L.
* interface.pyx - the normal component of the flow vanishes at the interface (z=0). Implemented using Blake, J. Biomech. 8 179 (1975).
* wallBounded.pyx - fluid flow near a plane wall at z=0. The flow vanishes (no-slip boundary condition) at the plane wall. That region of interest is the upper half space, z>0. Implemented using Lorentz-Blake tensor. See Blake, Proc. Camb. Phil. Soc. 70 303 (2017).
* twoWalls.pyx - two no-slip walls are at z=0 and z=H. The physical region is then `0<z<H'. Implemented using the approximate solution of Liron and Mochon. See Liron and Mochon, J. Eng. Math, 10 143 (1976).
* utils.pyx - has miscellaneous functionalities


## Force fields
* forceFields.pyx has implementation of various body forces and torques acting in colloidal systems.


## Phoretic fields 
The folder `phoretic` contains files to study phoresis of active particles. 

---
Corresponding to each .pyx file, there is a .pxd file. [A .pxd file](https://cython.readthedocs.io/en/latest/src/tutorial/pxd_files.html) contains declaration of cdef classes, methods, etc.  
