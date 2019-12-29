"""Some utilities codes which do not fit anywhere else,
but are essential in simulations of active colloids,
and plotting flow and phoretic fields
"""

import  numpy as np
cimport numpy as np
import matplotlib.pyplot as plt
cimport cython
from libc.math cimport sqrt, pow, log
from cython.parallel import prange


cpdef irreducibleTensors(l, p, Y0=?)
