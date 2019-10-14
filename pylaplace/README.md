This folder contains the core files of the PyLaplace library.

Each file computed the phoretic motion and phoretic field given the colloidal configuration.
The filenames, as described below, correspond to the boundary conditions in the flow. The boundary conditions are implemented using an appropriate Green's function of the Laplace equation. See details at: https://arxiv.org/abs/1910.00909


The main files are:
* unbounded.pyx - the phoretic flux vanishes at infinity. 
* wallBounded.pyx - the phoretic flux vanishes at a plane wall, which is located at z=0, such that region of interest is the upper half space, z>0. 
* utils.pyx has miscellaneous functionalities

Corresponding to each .pyx file, there is a .pxd file, which contains declaration of classes, methods, etc. It is essential when calling PyLaplace from another Cython file.
