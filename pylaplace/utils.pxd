"""Some utilities codes which do not fit anywhere else,
but are essential in simulations of active colloids,
"""

import  numpy as np
cimport numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm, colors
from mpl_toolkits.mplot3d import Axes3D
cimport cython
from libc.math cimport sqrt, pow, log
from cython.parallel import prange

cpdef couplingTensors(l, p, M0=?)
