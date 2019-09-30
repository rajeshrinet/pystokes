This folder contains the core files of PyStokes repo.

Each file corresponds to the boundary conditions in the flow. Flow is computed in unbounded domain and near a plane surface.

utils.pyx has miscellaneous functionalities


### Where is the wall?
A single plane boundary (be it wall or a fluid-fluid interface) is always kept at z=0 in the library such that region of interest is the upper half space for z>0. 


