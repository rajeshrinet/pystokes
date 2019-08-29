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
cdef double PI = 3.1415926535

DTYPE   = np.float
DTYP1   = np.int32
ctypedef np.float_t DTYPE_t 


@cython.wraparound(False)
@cython.nonecheck(False)
@cython.boundscheck(False)
cpdef irreducibleTensors(l, p, Y0=1):
    """
    Uniaxial paramterization of the tensorial harmonics (Yl) of order l
    l  : tensorialHarmonics of order l
    p  : axis along which the mode is paramterized
    Y0 : strength of the mode
    
    returns: Yl - tensorialHarmonics of rank l
    """
    cdef int i, Np = int (np.size(p)/3)
    cdef double S0 = Y0
    YY = np.zeros((2*l+1)*Np, dtype=DTYPE)
    
    cdef double [:] p1 = p
    cdef double [:] Y1 = YY
    
    
    if l==0:
            YY = Y0
    
    if l==1:
            YY = Y0*p
    
    if l==2:
        for i in prange(Np, nogil=True):
            Y1[i + 0*Np] = S0*(p1[i]*p1[i]                   -(1.0/3))
            Y1[i + 1*Np] = S0*(p1[i + Np]*p1[i + Np] -(1.0/3))
            Y1[i + 2*Np] = S0*(p1[i]*p1[i + Np])
            Y1[i + 3*Np] = S0*(p1[i]*p1[i + 2*Np])
            Y1[i + 4*Np] = S0*(p1[i + Np]*p1[i + 2*Np])
    
    if l==3:
        for i in range(Np):
            YY[i]      = Y0*(p[i]*p[i]*p[i]                    - 3/5*p[i]);
            YY[i+Np]   = Y0*(p[i+Np]*p[i+Np]*p[i+Np]   - 3/5*p[i+Np]);
            YY[i+2*Np] = Y0*(p[i]*p[i]*p[i+Np]                 - 1/5*p[i+Np]);
            YY[i+3*Np] = Y0*(p[i]*p[i]*p[i+2*Np]       - 1/5*p[i+2*Np]);
            YY[i+4*Np] = Y0*(p[i]*p[i+Np]*p[1+Np]           -1/5* p[i]);
            YY[i+5*Np] = Y0*(p[i+Np]*p[i+Np]*p[i+2*Np]);
            YY[i+6*Np] = Y0*(p[i+Np]*p[i+Np]*p[i+2*Np] -1/5*p[i+2*Np]);
    return YY


def simulate(rp0, Tf, Npts, rhs, integrator='odeint', filename='this'):
    from scipy.io import savemat; from scipy.integrate import odeint

    time_points=np.linspace(0, Tf, Npts+1);
    def rhs0(rp, t): 
        return rhs(rp)
    
    if integrator=='odeint':
        u = odeint(rhs0, rp0, time_points, mxstep=5000000)
        savemat(filename, {'X':u, 't':time_points})

    elif integrator=='odespy-vode':
        import odespy
        solver = odespy.Vode(rhs0, method = 'bdf', atol=1E-7, rtol=1E-6, order=5, nsteps=10**6)
        solver.set_initial_condition(rp0)
        u, t = solver.solve(time_points) 
        savemat(filename, {'X':u, 't':time_points})

    else:
        print("Error: Integration failed! \n Please set integrator='odeint' to use the scipy odeint (Deafult). \n Use integrator='odespy-vode' to use vode from odespy (github.com/rajeshrinet/odespy). \n Alternatively, write your own integrator to evolve the system in time \n")
    return


def initialCondition(Np, h0=3.1):
    '''
    Assigns initial condition. 
    To DO: Need to include options to assign possibilites
    '''
    rp0 = np.zeros(3*Np) 
    radius = np.sqrt(np.arange(Np)/float(Np))
    golden_angle = np.pi * (3 - np.sqrt(5))
    theta = golden_angle * np.arange(Np)
    
    points = np.zeros((Np, 2)) 
    points[:,0] = np.cos(theta)
    points[:,1] = np.sin(theta)
    points *= radius.reshape((Np, 1)) 
    points = points*2.1*np.sqrt(Np)
    
    points[:,0] += 2*(2*np.random.random(Np)-np.ones(Np)) 
    points[:,1] += 2*(2*np.random.random(Np)-np.ones(Np)) 
    
    
    rp0[0:Np]          = 1.5*points[:, 0]
    rp0[Np:2*Np]   = 1.5*points[:, 1]
    rp0[2*Np:3*Np] = h0
    ##rp0[0:3*Np] = ar*(2*np.random.random(3*Np)-np.ones(3*Np)) + rp0[0:3*Np]
    #
    #rp0[3*Np:4*Np] = np.zeros(Np)
    #rp0[4*Np:5*Np] = np.zeros(Np)
    #rp0[5*Np:6*Np] = -np.ones(Np)
    return rp0 


