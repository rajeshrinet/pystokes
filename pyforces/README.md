This folder contains the core files of the PyForces library.

The file forceFields.pyx has implementation of various body forces and torques acting in colloidal systems. 

Corresponding to each .pyx file, there is a .pxd file, which contains declaration of classes, methods, etc. It is essential when calling PyForces from another Cython file.
