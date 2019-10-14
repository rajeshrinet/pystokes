This folder contains the core files of PyLaplace repo.

Each file computed the phoretic motion and phoretic field given the colloidal configuration.
The filenames, as described below, correspond to the boundary conditions in the flow. The boundary conditions are implemented using an appropriate Green's function of the Laplace equation. See details at: https://arxiv.org/abs/1910.00909


* unbounded.pyx - the phoretic flux vanishes at infinity. I
* wallBounded.pyx - the phoretic flux vanishes at a plane wall, which is located at z=0, such that region of interest is the upper half space, z>0. 
* utils.pyx has miscellaneous functionalities
