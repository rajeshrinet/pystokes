This folder contains file to study power dissipation of active particles. 
See Eq.(18) in R.Singh and R.Adhikari J. Phys. Com. 2 025025 (2018).

* unbounded.pyx - Implemented using Ossen tensor.
* wallBounded.pyx - Implemented using Lorentz-Blake tensor.

---
Corresponding to each .pyx file, there is a .pxd file. [A .pxd file](https://cython.readthedocs.io/en/latest/src/tutorial/pxd_files.html) contains declaration of cdef classes, methods, etc.  
