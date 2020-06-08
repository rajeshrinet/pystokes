This folder contains the core files of the PyStokes library.

Each file computes the rigid body motion (RBM) and flow given the colloidal configuration.
The filenames, as described below, correspond to the boundary conditions in the flow. The boundary conditions are implemented using an appropriate Green's function of Stokes flow. See details at: https://arxiv.org/abs/1910.00909

The main files are:
* unbounded.pyx - the fluid flow only vanishes at infinity. Implemented using Oseen tensor.
* periodic.pyx - Implemented using the Ewald sum of Oseen tensor. The box is of size L.
* interface.pyx - the normal component of the flow vanishes at the interface (z=0). Implemented using Blake, J. Biomech. 8 179 (1975).
* wallBounded.pyx - fluid flow near a plane wall at z=0. The flow vanishes (no-slip boundary condition) at the plane wall. That region of interest is the upper half space, z>0. Implemented using Lorentz-Blake tensor. See Blake, Proc. Camb. Phil. Soc. 70 303 (2017).
* twoWalls.pyx - two no-slip walls are at z=0 and z=H. The physical region is then `0<z<H'. Implemented using the approximate solution of Liron and Mochon. See Liron and Mochon, J. Eng. Math, 10 143 (1976).
* mima.pyx - Mollified irreducible multipole approach (MIMA) resolves the fluid flow and solves Stokes equation directly. For more details, see chapter 10 of the thesis: https://www.imsc.res.in/xmlui/handle/123456789/418 
* utils.pyx - has miscellaneous functionalities


## Force fields
* forceFields.pyx has implementation of various body forces and torques acting in colloidal systems.


## Phoretic fields
Each file computed the phoretic motion and phoretic field given the colloidal configuration.
The filenames, as described below, correspond to the boundary conditions in the flow. The boundary conditions are implemented using an appropriate Green's function of the Laplace equation. See details of theory at: [J. Chem. Phys. 151, 044901 (2019)](https://aip.scitation.org/doi/abs/10.1063/1.5090179)


* phoreticUnbounded.pyx - the phoretic flux vanishes at infinity. 
* phoreticWallBounded.pyx - the phoretic flux vanishes at a plane wall, which is located at z=0, such that region of interest is the upper half space, z>0. 


Corresponding to each .pyx file, there is a .pxd file. A .pxd file contains declaration of cdef classes, methods, etc. It is essential when calling PyStokes from another Cython file. Read more: https://cython.readthedocs.io/en/latest/src/tutorial/pxd_files.html


## tests 
The folder tests contain unit testing framework for PyStokes







