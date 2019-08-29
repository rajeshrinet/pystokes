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
cdef double PI = 3.1415926535

DTYPE   = np.float
DTYP1   = np.int32
ctypedef np.float_t DTYPE_t 


@cython.wraparound(False)
@cython.nonecheck(False)
@cython.boundscheck(False)
cpdef couplingTensors(l, p, M0=1):
    """
    Uniaxial paramterization of the tensorial harmonics (Yl) of order l
    l  : tensorialHarmonics of order l
    p  : axis along which the mode is paramterized
    M0 : strength of the mode

    returns: Yl - tensorial harmonics of rank l
    """
    cdef int i, Np = int (np.size(p)/3)
    cdef double S0 = M0
    MM = np.zeros((2*l+1)*Np, dtype=DTYPE)

    cdef double [:] p1 = p
    cdef double [:] Y1 = MM


    if l==0:
        MM = -M0

    if l==1:
        MM = -M0*p
    
    if l==2:
        for i in prange(Np, nogil=True):
            Y1[i + 0*Np] = S0*(p1[i]*p1[i]           -(1.0/3))
            Y1[i + 1*Np] = S0*(p1[i + Np]*p1[i + Np] -(1.0/3))
            Y1[i + 2*Np] = S0*(p1[i]*p1[i + Np])
            Y1[i + 3*Np] = S0*(p1[i]*p1[i + 2*Np])
            Y1[i + 4*Np] = S0*(p1[i + Np]*p1[i + 2*Np])

    if l==3:
        for i in range(Np):
            MM[i]      = M0*(p[i]*p[i]*p[i]            - 3/5*p[i]);
            MM[i+Np]   = M0*(p[i+Np]*p[i+Np]*p[i+Np]   - 3/5*p[i+Np]);
            MM[i+2*Np] = M0*(p[i]*p[i]*p[i+Np]         - 1/5*p[i+Np]);
            MM[i+3*Np] = M0*(p[i]*p[i]*p[i+2*Np]       - 1/5*p[i+2*Np]);
            MM[i+4*Np] = M0*(p[i]*p[i+Np]*p[1+Np]       -1/5* p[i]);
            MM[i+5*Np] = M0*(p[i+Np]*p[i+Np]*p[i+2*Np]);
            MM[i+6*Np] = M0*(p[i+Np]*p[i+Np]*p[i+2*Np] -1/5*p[i+2*Np]);
    return MM




def plotPhoreticField(l, c0=1):
    """
    l   : mode of phoretic field
    vls0: strength of the mode
    """

    theta = np.linspace(0, np.pi, 128)
    phi = np.linspace(0, 2*np.pi, 128)
    theta, phi = np.meshgrid(theta, phi)

    x = np.sin(theta) * np.cos(phi)
    y = np.sin(theta) * np.sin(phi)
    z = np.cos(theta)
    
    speedVls = z
    
    # Set the aspect ratio to 1 so our sphere looks spherical
    fig = plt.figure(figsize=(4, 4))
    ax = fig.add_subplot(111, projection='3d')

    if l==0:
        speedVls = c0 + z*0
        ax.plot_surface(x, y, z,  rstride=1, cstride=1, facecolors=cm.RdBu_r(speedVls))
        ax.set_axis_off()
        plt.show()
    
    elif l==1:
        speedVls = c0*np.cos(theta)
        ax.plot_surface(x, y, z,  rstride=1, cstride=1, facecolors=cm.RdBu_r(speedVls))
        ax.set_axis_off()
        plt.show()
    
    elif l==2:
        speedVls = 4*c0*(np.cos(theta)*np.cos(theta) - 1.0/3)
        ax.plot_surface(x, y, z,  rstride=1, cstride=1, facecolors=cm.RdBu_r(speedVls))
        ax.set_axis_off()
        plt.show()
    
    elif l==3:
        speedVls = 4*c0*(np.cos(theta)*np.cos(theta)*np.cos(theta) - 3*np.cos(theta)/5.0)
        ax.plot_surface(x, y, z,  rstride=1, cstride=1, facecolors=cm.RdBu_r(speedVls))
        ax.set_axis_off()
        plt.show()

    else:
        print('Not yet implemented...')


