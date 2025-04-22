"""
Some utilities codes which do not fit anywhere else,
but are essential in simulations of active colloids,
and plotting flow and phoretic fields
"""

import  numpy as np
cimport numpy as np
cimport cython
from libc.math cimport sqrt, pow, log
from cython.parallel import prange
cdef double PI = 3.1415926535

DTYPE   = np.float64
DTYP1   = np.int32
ctypedef np.float_t DTYPE_t 


def gridXY(dim, L, Ng):
    """
    Returns the grid in XY direction centered around zero
    ...

    Parameters
    ----------
    L  : float  
        Length of the grid 
    Ng : int 
         Number of grid points

    Examples
    --------
    An example of creating grid 
    
    >>> import numpy as np, pystokes 
    >>> dim, L, Ng = 3, 10, 32
    >>>  rr, vv = pystokes.utils.gridXY(dim, L, Ng)
    """

    Nt = Ng*Ng
    rr, vv = np.zeros(dim*Nt), np.zeros(dim*Nt)
    X, Y = np.meshgrid(np.linspace(-L, L, Ng), np.linspace(-L, L, Ng))
    rr[0:2*Nt] = np.concatenate((X.reshape(Ng*Ng), Y.reshape(Ng*Ng)))
    return rr, vv


def gridYZ(dim, L, Ng):
    """
    Returns the grid in YZ direction centered around zero
    ...

    Parameters
    ----------
    L  : float  
        Length of the grid 
    Ng : int 
         Number of grid points

    Examples
    --------
    An example of creating grid 
    
    >>> import numpy as np, pystokes 
    >>> dim, L, Ng = 3, 10, 32
    >>>  rr, vv = pystokes.utils.gridYZ(dim, L, Ng)
    """

    Nt = Ng*Ng
    rr, vv = np.zeros(dim*Nt), np.zeros(dim*Nt)
    X, Y = np.meshgrid(np.linspace(-L, L, Ng), np.linspace(-0, L, Ng))
    rr[Nt:3*Nt] = np.concatenate((X.reshape(Ng*Ng), Y.reshape(Ng*Ng)))
    return rr, vv



@cython.wraparound(False)
@cython.nonecheck(False)
@cython.boundscheck(False)
cpdef irreducibleTensors(l, p, Y0=1):
    """
    Uniaxial paramterization of the tensorial harmonics (Yl) of order l
    ...

    Parameters
    ----------
    l  : int 
        Tensorial Harmonics of order l
    p  : np.rrray 
        An array of size 3
        Axis along which the mode is paramterized
    Y0 : float 
        Strength of the mode
    
    returns: Yl - tensorialHarmonics of rank l
    """
    cdef int i, N = int (np.size(p)/3)
    cdef double S0 = Y0
    YY = np.zeros((2*l+1)*N, dtype=DTYPE)
    
    cdef double [:] p1 = p
    cdef double [:] Y1 = YY
    
    
    if l==0:
            YY = Y0
    
    if l==1:
            YY = Y0*p
    
    if l==2:
        for i in prange(N, nogil=True):
            Y1[i + 0*N] = S0*(p1[i]*p1[i]                    -(1.0/3))
            Y1[i + 1*N] = S0*(p1[i + N]*p1[i + N] -(1.0/3))
            Y1[i + 2*N] = S0*(p1[i]*p1[i + N])
            Y1[i + 3*N] = S0*(p1[i]*p1[i + 2*N])
            Y1[i + 4*N] = S0*(p1[i + N]*p1[i + 2*N])
    
    if l==3:
        for i in range(N):
            YY[i]     = Y0*(p[i]*p[i]*p[i]                    - 3/5*p[i]);
            YY[i+N]   = Y0*(p[i+N]*p[i+N]*p[i+N]   - 3/5*p[i+N]);
            YY[i+2*N] = Y0*(p[i]*p[i]*p[i+N]                   - 1/5*p[i+N]);
            YY[i+3*N] = Y0*(p[i]*p[i]*p[i+2*N]     - 1/5*p[i+2*N]);
            YY[i+4*N] = Y0*(p[i]*p[i+N]*p[i+N]          -1/5* p[i]);
            YY[i+5*N] = Y0*(p[i]*p[i+N]*p[i+2*N]);
            YY[i+6*N] = Y0*(p[i+N]*p[i+N]*p[i+2*N] -1/5*p[i+2*N]);
    return YY




def simulate(rp0, Tf, Nts, rhs, integrator='odeint', filename='this.mat',
            Ti=0, maxNumSteps=100000, **kwargs):
    """
    Simulates using choice of integrator
    
    ...

    Parameters
    ----------
    rp0 : np.array 
        Initial condition 
    Tf  : int 
         Final time 
    Nts: int 
        Number of points to return data 
    rhs : Python Function
        Right hand side to integrate 
    integrator: string 
        Default is 'odeint' of scipy. 
        Please set integrator='odeint' to use 
        the scipy.integrate's odeint (Default)
        Use integrator='solve_ivp' to use ivp  
        Use integrator='odespy-vode' to use vode
        from odespy (github.com/rajeshrinet/odespy).
        Use integrator='odespy-rkf45' to use RKF45  
        from odespy (github.com/rajeshrinet/odespy).
        Use integrator='odespy-rk4' to use RK4 from 
        odespy (github.com/rajeshrinet/odespy).     
        Alternatively, write your own integrator to 
        evolve the system in time and store the data.
    filename: string 
        filename to write the data. 
        Deafult is 'this.mat'
    """

    def dxdtEval(rp, t): 
        """
        returns the right hand side
        """
        return rhs(rp)
   

    if integrator=='odeint':
        from scipy.integrate import odeint
        time_points=np.linspace(Ti, Tf, Nts+1);
        X = odeint(dxdtEval, rp0, time_points, mxstep=maxNumSteps, **kwargs)

    elif integrator=='solve_ivp':
        from scipy.integrate import solve_ivp
        time_points=np.linspace(Ti, Tf, Nts+1)                                                           
        X = solve_ivp(lambda t, xt: dxdtEval(xt,t), [0,Tf], rp0, 
                         t_eval=time_points, **kwargs).y.T

    elif integrator=='odespy-vode':
        import odespy
        time_points=np.linspace(Ti, Tf, Nts+1);
        solver = odespy.Vode(dxdtEval, method = 'bdf', 
        atol=1E-7, rtol=1E-6, order=5, nsteps=maxNumSteps)
        solver.set_initial_condition(rp0)
        X, t = solver.solve(time_points, **kwargs) 

    elif integrator=='odespy-rkf45':
        import odespy
        time_points=np.linspace(Ti, Tf, Nts+1);
        solver = odespy.RKF45(dxdtEval)
        solver.set_initial_condition(rp0)
        X, t = solver.solve(time_points, **kwargs) 

    elif integrator=='odespy-rk4':
        import odespy
        time_points=np.linspace(Ti, Tf, Nts+1);
        solver = odespy.RK4(dxdtEval)
        solver.set_initial_condition(rp0)
        X, t = solver.solve(time_points, **kwargs) 

    else:
        raise Exception("Error: Integration method not found! \n \
                        Please set integrator='odeint' to use \n \
                        the scipy.integrate's odeint (Default)\n \
                        Use integrator='solve_ivp' to use ivp  \
                        Use integrator='odespy-vode' to use vode \
                        from odespy (github.com/rajeshrinet/odespy).\n \
                        Use integrator='odespy-rkf45' to use RKF45  \
                        from odespy (github.com/rajeshrinet/odespy).\n \
                        Use integrator='odespy-rk4' to use RK4 from \
                        odespy (github.com/rajeshrinet/odespy).     \n \
                        Alternatively, write your own integrator to \
                        evolve the system in time and store the data.\n")

    from scipy.io import savemat
    savemat(filename, {'X':X, 't':time_points})
    return

def simulate_vo2rp(rp0, Tf, Nts, vo, filename='this.mat',
            Ti=0, maxNumSteps=100000, **kwargs):
    """
    Simulates using choice of integrator
    ...

    Parameters
    ----------
    rp0 : np.array 
        Initial condition 
    Tf	: int 
         Final time 
    Nts: int 
        Number of points to return data 
    vo : Python Function
        velcotity and angular velocity
    filename: string 
        filename to write the data. 
        Deafult is 'this.mat'
    """

    def dxdtEval(rp, t): 
        """
        returns velocity and angular velocity
        """
        return vo(rp)

    def dot(A, b):
        """
        A: Array size of (3, 3)
        b: Array size of 3
        returns A \dot b
        """
        retval = np.zeros(3)
        for i in range(3):
            for j in range(3):
                retval[i] += A[i, j] * b[j]
        return retval

    from scipy.integrate import solve_ivp
    from scipy.linalg import expm

    time_points=np.linspace(Ti, Tf, Nts+1)
    N = int(len(rp0) / 6)
    X = np.zeros((Nts + 1, len(rp0)))
    X[0] = rp0
    for n in range(Nts):
        X[n + 1] = solve_ivp(lambda t, xt: dxdtEval(xt,t), 
                             [time_points[n], time_points[n + 1]], X[n], 
                             t_eval=[time_points[n], time_points[n + 1]], **kwargs).y.T[1]
        P = X[n + 1, 3*N:6*N]
        P_old = X[n, 3*N:6*N]
        O = vo(X[n])[3*N:6*N]
        for p in range(N):
            Op = O[p::N]
            skewOdt = np.array([[0.0, -Op[2], Op[1]],
                              [Op[2], 0.0, -Op[0]],
                              [-Op[1], Op[0], 0.0]]) \
                    * (time_points[n + 1] - time_points[n])
            P[p::N] = dot(expm(skewOdt), P_old[p::N])
        X[n + 1, 3*N:6*N] = P
        
    from scipy.io import savemat
    savemat(filename, {'X':X, 't':time_points})
    return




def initialCondition(N, h0=3.1):
    '''
    Assigns initial condition. 
    To DO: Need to include options to assign possibilites
    '''
    rp0 = np.zeros(3*N) 
    radius = np.sqrt(np.arange(N)/float(N))
    golden_angle = np.pi * (3 - np.sqrt(5))
    theta = golden_angle * np.arange(N)
    
    points = np.zeros((N, 2)) 
    points[:,0] = np.cos(theta)
    points[:,1] = np.sin(theta)
    points *= radius.reshape((N, 1)) 
    points = points*2.1*np.sqrt(N)
    
    points[:,0] += 2*(2*np.random.random(N)-np.ones(N)) 
    points[:,1] += 2*(2*np.random.random(N)-np.ones(N)) 
    
    
    rp0[0:N]           = 1.5*points[:, 0]
    rp0[N:2*N]   = 1.5*points[:, 1]
    rp0[2*N:3*N] = h0
    ##rp0[0:3*N] = ar*(2*np.random.random(3*N)-np.ones(3*N)) + rp0[0:3*N]
    #
    #rp0[3*N:4*N] = np.zeros(N)
    #rp0[4*N:5*N] = np.zeros(N)
    #rp0[5*N:6*N] = -np.ones(N)
    return rp0 




def plotLogo():
    """Plots the PyStokes Logo"""
    import pystokes 
    import numpy as np, matplotlib.pyplot as plt
    import matplotlib.patches as patches
    ax=plt.gca();

    # particle radius, fluid viscosity, and number of particles
    b, eta, N = 1.0, 1.0/6.0, 4
    
    #initialise
    r = np.zeros(3*N);  r[2*N:3*N] = 2
    p = np.zeros(3*N);  p[2*N:3*N] = 1
    
    r[2*N]=2.8;  r[N:2*N]=np.linspace(-3.2, 3.2, N);
    # irreducible coeffcients
    F1s = pystokes.utils.irreducibleTensors(1, p)
    V2s = pystokes.utils.irreducibleTensors(2, p)
    V3t = pystokes.utils.irreducibleTensors(1, p) 
    
    # space dimension , extent , discretization
    dim, L, Ng = 3, 8, 100
    
    #Instantiate the Flow class
    wFlow = pystokes.unbounded.Flow(radius=b, particles=N, viscosity=eta, gridpoints=Ng*Ng)
    wFlow = pystokes.wallBounded.Flow(radius=b, particles=N, viscosity=eta, gridpoints=Ng*Ng)
    #iFlow = pystokes.interface.Flow(radius=b, particles=N, viscosity=eta, gridpoints=Ng*Ng)
    
    # create the grid
    rr, vv = pystokes.utils.gridYZ(dim, L, Ng); 
    wFlow.flowField1s(vv, rr, r, F1s)  
    
    plt.figure(figsize=(26, 8));  Nt=Ng*Ng
    
    yy, zz = rr[Nt:2*Nt].reshape(Ng, Ng), rr[2*Nt:3*Nt].reshape(Ng, Ng)
    vy, vz = vv[Nt:2*Nt].reshape(Ng, Ng), vv[2*Nt:3*Nt].reshape(Ng, Ng)
    density=0.75; arrowSize=4; mask=0.6; ms=36; offset=1e-6
    plt.streamplot(yy, zz, vy, vz, color="gray", arrowsize =arrowSize, density=density, linewidth=4.4, zorder=1)
    
    for i in range(N):
        ax.add_patch(patches.Circle((r[i],r[i+N]),1.1,facecolor='w',alpha=1,edgecolor="steelblue", linewidth=4))
    plt.grid()
    
    ww=0.3
    plt.ylim(-ww, np.max(zz))
    plt.axhspan(-ww, ww, facecolor='gray');
    plt.axis('off')
    plt.text(r[N]-.6, r[2*N]-.5, 'Py', fontsize=100);
    plt.text(r[N+1]-.66, r[2*N+1]-.64, 'St', fontsize=111);
    plt.text(r[N+2]-.78, r[2*N+2]-.64, 'ok', fontsize=111);
    plt.text(r[N+3]-.66, r[2*N+3]-.64, 'es', fontsize=111);




def plotStreamlinesXY(vv, rr, r, density=0.82, arrowSize=1.2, mask=1, ms=36, offset=1e-6, title='None'):
    """
    Plots streamlines in XY plane given the position and velocity; 
    vv: one dimensional arrays of velocity
    rr: one dimensional arrays of positions where velocity is computed
    """
    import matplotlib.pyplot as plt
    import matplotlib.patches as patches
    ax=plt.gca();

    N, Nt = int(np.size(r)/3), int(np.size(rr)/3);  Ng=int(np.sqrt(Nt))
    xx, yy = rr[0:Nt].reshape(Ng, Ng), rr[Nt:2*Nt].reshape(Ng, Ng)
    vx, vy = vv[0:Nt].reshape(Ng, Ng), vv[Nt:2*Nt].reshape(Ng, Ng)

    spd = np.hypot(vx, vy)
    rr  = np.hypot(xx-r[0], yy-r[1])
    spd[rr<mask]=0;  spd+=offset
    plt.pcolormesh(xx, yy, np.log(spd), cmap=plt.cm.gray_r, shading='auto')

    st=11;  sn=st*st;  a0=2;  ss=np.zeros((2, st*st))
    for i in range(st):
        for j in range(st):
            ii = i*st+j
            ss[0, ii]      = a0*(-5 + i)
            ss[1, ii]      = a0*(-5 + j)

    plt.streamplot(xx, yy, vx, vy, color="black", arrowsize =arrowSize, arrowstyle='->', start_points=ss.T, density=density, broken_streamlines=True, zorder=1)
    for i in range(N):
        ax.add_patch(patches.Circle((r[i],r[i+N]),1.1,facecolor='w',alpha=1,edgecolor="steelblue", linewidth=4))
        
    plt.xlim(np.min(xx), np.max(xx))
    plt.ylim(np.min(yy), np.max(yy))
    plt.axis('off')
    
    if title==str('None'):
        pass
    elif title==str('1s'):
        plt.title('$l\sigma=1s$', fontsize=26);
    elif title==str('2s'):
        plt.title('$l\sigma=2s$', fontsize=26);
    elif title==str('3t'):
        plt.title('$l\sigma=3t$', fontsize=26);
    else:
        plt.title(title, fontsize=26);




def plotStreamlinesYZ(vv, rr, r, density=0.795, arrowSize=1.2, mask=0.6, ms=36, offset=1e-6, title='None'):
    """
    Plots streamlines in YZ plane given the position and velocity; 
    vv: one dimensional arrays of velocity
    rr: one dimensional arrays of positions where velocity is computed
    """
    import matplotlib.pyplot as plt
    import matplotlib.patches as patches
    ax=plt.gca();
    N, Nt = int(np.size(r)/3), int(np.size(rr)/3);  Ng=int(np.sqrt(Nt))
    yy, zz = rr[Nt:2*Nt].reshape(Ng, Ng), rr[2*Nt:3*Nt].reshape(Ng, Ng)
    vy, vz = vv[Nt:2*Nt].reshape(Ng, Ng), vv[2*Nt:3*Nt].reshape(Ng, Ng)

    spd = np.hypot(vy, vz)
    rr  = np.hypot(yy-r[1], zz-r[2])
    spd[rr<mask]=0;  spd+=offset
    plt.pcolormesh(yy, zz, np.log(spd), cmap=plt.cm.gray_r, shading='auto')

    st=11;  sn=st*st;  a0=2;  ss=np.zeros((2, st*st))
    for i in range(st):
        for j in range(st):
            ii = i*st+j
            ss[0, ii]      = a0*(-5 + i)
            ss[1, ii]      = 1*(0 + j)
    plt.streamplot(yy, zz, vy, vz, color="black", arrowsize =arrowSize, arrowstyle='->', start_points=ss.T, density=density, zorder=1)
    for i in range(N):
        ax.add_patch(patches.Circle((r[i],r[i+2*N]),1.1,facecolor='w',alpha=1,edgecolor="steelblue", linewidth=4))

    plt.xlim(np.min(yy), np.max(yy))
    plt.ylim(np.min(zz), np.max(zz))
    plt.axis('off')
    
    if title==str('None'):
        pass
    elif title==str('1s'):
        plt.title(r'$l\sigma=1s$', fontsize=26);
    elif title==str('2s'):
        plt.title(r'$l\sigma=2s$', fontsize=26);
    elif title==str('3t'):
        plt.title(r'$l\sigma=3t$', fontsize=26);
    else:
        plt.title(title, fontsize=26);




def plotStreamlinesYZsurf(vv, rr, r, density=0.8, arrowSize=1.2, mask=0.6, ms=36, offset=1e-6, title='None'):
    """
    Plots streamlines in YZ plane given the position and velocity; The surface is also plotted
    vv: one dimensional arrays of velocity
    rr: one dimensional arrays of positions where velocity is computed
    """
    import matplotlib.pyplot as plt
    import matplotlib.patches as patches
    ax=plt.gca();
    N, Nt = int(np.size(r)/3), int(np.size(rr)/3);  Ng=int(np.sqrt(Nt))
    yy, zz = rr[Nt:2*Nt].reshape(Ng, Ng), rr[2*Nt:3*Nt].reshape(Ng, Ng)
    vy, vz = vv[Nt:2*Nt].reshape(Ng, Ng), vv[2*Nt:3*Nt].reshape(Ng, Ng)

    spd = np.hypot(vy, vz)
    rr  = np.hypot(yy-r[1], zz-r[2])
    spd[rr<mask]=0;  spd+=offset
    plt.pcolormesh(yy, zz, np.log(spd), cmap=plt.cm.gray_r, shading='auto')
    
    st=11;  sn=st*st;  a0=2;  ss=np.zeros((2, st*st))
    for i in range(st):
        for j in range(st):
            ii = i*st+j
            ss[0, ii]      = a0*(-5 + i)
            ss[1, ii]      = 1*(0 + j)
    plt.streamplot(yy, zz, vy, vz, color="black", arrowsize =arrowSize, arrowstyle='->', start_points=ss.T, density=density, zorder=1)
    for i in range(N):
        ax.add_patch(patches.Circle((r[i],r[i+2*N]),1.1,facecolor='w',alpha=1,edgecolor="steelblue", linewidth=4))

    plt.xlim(np.min(yy), np.max(yy))
    ww=0.3
    plt.ylim(-ww, np.max(zz))
    plt.axhspan(-ww, ww, facecolor='black');
    plt.axis('off')
    
    if title==str('None'):
        pass
    elif title==str('1s'):
        plt.title('$l\sigma=1s$', fontsize=26);
    elif title==str('2s'):
        plt.title('$l\sigma=2s$', fontsize=26);
    elif title==str('3t'):
        plt.title('$l\sigma=3t$', fontsize=26);
    else:
        plt.title(title, fontsize=26);





def plotContoursYZ(vv, rr, r, density=1.2, arrowSize=1.2, mask=0.6, ms=36, offset=1e-6, title='None'):
    """
    Plots streamlines in YZ plane given the position and velocity; 
    vv: one dimensional arrays of velocity
    rr: one dimensional arrays of positions where velocity is computed
    """
    import matplotlib.pyplot as plt
    import matplotlib.patches as patches
    ax=plt.gca();
    N, Nt = int(np.size(r)/3), int(np.size(rr)/3);  Ng=int(np.sqrt(Nt))
    yy, zz = rr[Nt:2*Nt].reshape(Ng, Ng), rr[2*Nt:3*Nt].reshape(Ng, Ng)

    spd = np.sqrt(vv[0:Nt]*vv[0:Nt]).reshape(Ng, Ng)
    spd+=offset
    cp=plt.pcolor(yy, zz, np.log(spd), cmap=plt.cm.gray_r)
    cp=plt.contour(yy, zz, np.log(spd), colors='#778899', linestyles='solid')

    plt.xlim(np.min(yy), np.max(yy))
    ww=0.
    plt.ylim(-ww, np.max(zz))
    plt.axis('off')
    if title==str('None'):
        pass 
    elif title==str('m=0'):
        plt.title('$m=0$', fontsize=26);
        #plt.ylabel('Unbounded domain', fontsize=20)
    elif title==str('m=1'):
        plt.title('$m=1$', fontsize=26);
    elif title==str('m=2'):
        plt.title('$m=2$', fontsize=26);
    else:
        plt.title(title, fontsize=26);

    for i in range(N):
        ax.add_patch(patches.Circle((r[i],r[i+2*N]),1,color='k',alpha=1))




def plotContoursYZsurf(vv, rr, r, density=1.2, arrowSize=1.2, mask=0.6, ms=36, offset=1e-6, title='None'):
    """
    Plots streamlines in YZ plane given the position and velocity; 

    vv: one dimensional arrays of velocity
    rr: one dimensional arrays of positions where velocity is computed
    """
    import matplotlib.pyplot as plt
    import matplotlib.patches as patches
    ax=plt.gca();
    N, Nt = int(np.size(r)/3), int(np.size(rr)/3);  Ng=int(np.sqrt(Nt))
    yy, zz = rr[Nt:2*Nt].reshape(Ng, Ng), rr[2*Nt:3*Nt].reshape(Ng, Ng)

    spd = np.sqrt(vv[0:Nt]*vv[0:Nt]).reshape(Ng, Ng)
    spd+=offset
    cp=plt.pcolor(yy, zz, np.log(spd), cmap=plt.cm.gray_r)
    cp=plt.contour(yy, zz, np.log(spd), colors='#778899', linestyles='solid')

    plt.xlim(np.min(yy), np.max(yy))
    ww=0.3
    plt.ylim(-ww, np.max(zz))
    plt.axhspan(-ww, ww, facecolor='black');
    plt.axis('off')
    for i in range(N):
        ax.add_patch(patches.Circle((r[i],r[i+2*N]),1,color='k',alpha=1))

    
    if title==str('None'):
        pass 
    elif title==str('l=0'):
        plt.title('$l=0$', fontsize=26);
        plt.ylabel('Plane no-flux wall', fontsize=20)
    elif title==str('l=1'):
        plt.title('$l=1$', fontsize=26);
    elif title==str('l=0 and l=1'):
        plt.title('$l=0$ and $l=1$', fontsize=26);
    else:
        plt.title(title, fontsize=26);
    #elif title==str('0'):
    #    plt.title('title, fontsize=26);
    #elif title==str('1'):
    #    plt.title('$l=1$', fontsize=26);



def plotTrajectory(twoBodyDynamics, T, bins=50):
    import matplotlib.pyplot as plt
    import matplotlib.patches as patches
    ax=plt.gca();
    f=plt.figure(figsize=(24, 11), edgecolor='gray', linewidth=4);  
    ax=f.add_subplot(221);
    x1, x2 = twoBodyDynamics(T[0])
    plt.hist(np.abs(x1-x2), bins=bins, density=True, color='gray', alpha=0.5)
    plt.title("Temperature  = %s"%T[0], fontsize=24)
    plt.ylabel('Separation of colloids', fontsize=28)
    plt.xticks(fontsize=24); plt.yticks(fontsize=24)
    plt.yticks([])
    
    ax=f.add_subplot(223);
    plt.plot(x1, '--', label="colloid 1", color="dimgray", lw=3.2)
    plt.plot(x2, '-', label="colloid 2", color="slategray", lw=3.2)
    plt.legend(fontsize=24, loc='upper right'); 
    plt.xticks(fontsize=24); 
    plt.ylabel('Position of colloids', fontsize=28)
    plt.xticks([])
    ax.yaxis.tick_right()
    plt.yticks(fontsize=24)
    
    ax=f.add_subplot(222);
    x1, x2 = twoBodyDynamics(T[1]);  data=np.abs(x1-x2)
    #plt.hist(np.abs(x1-x2), 64, density=True, color='gray')
    y,binEdges=np.histogram(data,bins=bins)
    bincenters = 0.5*(binEdges[1:]+binEdges[:-1])
    plt.hist(data, bins=bins, label='simulation', color='gray', alpha=0.5);
    plt.gca().set_xlim(left=2.0)
   
    aa=0 
    bincenters = np.sort(bincenters)
    for i in range(10):
        if bincenters[i]<1.9:
            aa+=1

    xx = bincenters[aa:];   rr=(xx**2 + 4*2.5*2.5)**(-1.5)
    yy = np.log(y[aa:]);    coefficients = np.polyfit(rr, yy, 1)
    polynomial = np.poly1d(coefficients); ys = polynomial(rr)
    plt.plot(xx[0:], np.exp(ys)[0:], '-', label='analytical', lw=3.2, color='dimgray')
    plt.title("Temperature  = %s"%T[1], fontsize=24)
    plt.xticks(fontsize=24); plt.yticks(fontsize=24)
    plt.yticks([]); plt.legend(fontsize=22)
    
    ax=f.add_subplot(224);
    plt.plot(x1, '--', label="colloid 1", color="dimgray", lw=2)
    plt.plot(x2, '-', label="colloid 2", color="slategray", lw=2)
    plt.legend(fontsize=24); plt.xticks(fontsize=24); plt.yticks(fontsize=24)
    plt.xticks([])
    ax.yaxis.tick_right()
    plt.yticks(fontsize=24)
    


def createCircle(R, alpha=1):
    import matplotlib.pyplot as plt
    return plt.Circle((0,0), radius= R, color='silver', alpha=alpha) 



def showShape(patch):
    import matplotlib.pyplot as plt
    ax=plt.gca()
    ax.add_patch(patch)
    plt.axis('scaled')
    #plt.show()



def plotConfigs(t=[0,1], ms=36, tau=1, filename='None'):
    import matplotlib.pyplot as plt
    tNew= np.asanyarray(tau)*t
    from scipy.io import loadmat 
    data  = loadmat(filename)
    X     = data['X']
    xx = int(np.size(X[0,:]));  N = int(xx/6);   
    color='gray'


    plt.figure(figsize=(28, 10), edgecolor='gray', linewidth=4)

    ll = np.abs(np.max(X[t[0], 0:N]))+5
    if np.size(t)==1:
        c = createCircle(ll, alpha=1);     showShape(c)
        plt.scatter(X[t[0], 0:N], X[t[0], N:2*N], s=ms, c=color, edgecolors='darkslategray', mew=4); 
        plt.xlim(-ll, ll); plt.ylim(-ll, ll); plt.axis('off');
    
    elif np.size(t)==2:
        plt.subplot(141); 
        c = createCircle(ll, alpha=1);     showShape(c)
        plt.scatter(X[t[0], 0:N], X[t[0], N:2*N], s=ms, c=color, edgecolors='darkslategray'); plt.axis('off');
        plt.subplot(142);                                                                     
        c = createCircle(ll, alpha=1);     showShape(c)
        plt.scatter(X[t[1], 0:N], X[t[1], N:2*N], s=ms, c=color, edgecolors='darkslategray'); plt.axis('off');
                                                                                              
    elif np.size(t)==3:                                                                       
        plt.subplot(141);                                                                     
        c = createCircle(ll, alpha=1);     showShape(c)
        plt.scatter(X[t[0], 0:N], X[t[0], N:2*N], s=ms, c=color, edgecolors='darkslategray'); plt.axis('off');
                                                                                              
        plt.subplot(142);                                                                     
        c = createCircle(ll, alpha=1);     showShape(c)
        plt.scatter(X[t[1], 0:N], X[t[1], N:2*N], s=ms, c=color, edgecolors='darkslategray'); plt.axis('off');
                                                                                              
        plt.subplot(143);                                                                     
        c = createCircle(ll, alpha=1);     showShape(c)
        plt.scatter(X[t[2], 0:N], X[t[2], N:2*N], s=ms, c=color, edgecolors='darkslategray'); plt.axis('off');

    elif np.size(t)==4:
        plt.subplot(141); 
        c = createCircle(ll, alpha=.1);     showShape(c)
        plt.scatter(X[t[0], 0:N], X[t[0], N:2*N], s=ms, c=color, edgecolors='darkslategray'); 
        plt.xlim(-ll, ll); plt.ylim(-ll, ll); plt.axis('off'); plt.title(r'Time=%d$\tau$'%tNew[0], fontsize=32)
        
        plt.subplot(142); 
        c = createCircle(ll, alpha=.1);     showShape(c)
        plt.scatter(X[t[1], 0:N], X[t[1], N:2*N], s=ms, c=color, edgecolors='darkslategray'); 
        plt.xlim(-ll, ll); plt.ylim(-ll, ll); plt.axis('off'); plt.title(r'Time=%d$\tau$'%tNew[1], fontsize=32)
        
        plt.subplot(143); 
        c = createCircle(ll, alpha=.1);     showShape(c)
        plt.scatter(X[t[2], 0:N], X[t[2], N:2*N], s=ms, c=color, edgecolors='darkslategray');
        plt.xlim(-ll, ll); plt.ylim(-ll, ll); plt.axis('off'); plt.title(r'Time=%d$\tau$'%tNew[2], fontsize=32)
        
        plt.subplot(144); 
        c = createCircle(ll, alpha=.1);     showShape(c)
        plt.scatter(X[t[3], 0:N], X[t[3], N:2*N], s=ms, c=color, edgecolors='darkslategray');
        plt.xlim(-ll, ll); plt.ylim(-ll, ll); plt.axis('off'); plt.title(r'Time=%d$\tau$'%tNew[3], fontsize=32)

        
    elif np.size(t)==5:
        plt.subplot(151); 
        c = createCircle(ll, alpha=1);     showShape(c)
        plt.scatter(X[t[0], 0:N], X[t[0], N:2*N], s=ms, c=color, edgecolors='darkslategray'); 
        plt.xlim(-ll, ll); plt.ylim(-ll, ll); plt.axis('off'); plt.title('Time=%d$\tau$'%tNew[0], fontsize=26)
        
        plt.subplot(152); 
        c = createCircle(ll, alpha=1);     showShape(c)
        plt.scatter(X[t[1], 0:N], X[t[1], N:2*N], s=ms, c=color, edgecolors='darkslategray'); 
        plt.xlim(-ll, ll); plt.ylim(-ll, ll); plt.axis('off'); plt.title('Time=%d$\tau$'%tNew[1], fontsize=26)
        
        plt.subplot(153); 
        c = createCircle(ll, alpha=1);     showShape(c)
        plt.scatter(X[t[2], 0:N], X[t[2], N:2*N], s=ms, c=color, edgecolors='darkslategray');
        plt.xlim(-ll, ll); plt.ylim(-ll, ll); plt.axis('off'); plt.title('Time=%d$\tau$'%tNew[2], fontsize=26)
        
        plt.subplot(154); 
        c = createCircle(ll, alpha=1);     showShape(c)
        plt.scatter(X[t[3], 0:N], X[t[3], N:2*N], s=ms, c=color, edgecolors='darkslategray');
        plt.xlim(-ll, ll); plt.ylim(-ll, ll); plt.axis('off'); plt.title('Time=%d$\tau$'%tNew[3], fontsize=26)
        
        plt.subplot(155); 
        c = createCircle(ll, alpha=1);     showShape(c)
        plt.scatter(X[t[4], 0:N], X[t[4], N:2*N], s=ms, c=color, edgecolors='darkslategray');
        plt.xlim(-ll, ll); plt.ylim(-ll, ll); plt.axis('off'); plt.title('Time=%d$\tau$'%tNew[4], fontsize=26)
    return 



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
    cdef int i, N = int (np.size(p)/3)
    cdef double S0 = M0
    MM = np.zeros((2*l+1)*N, dtype=DTYPE)

    cdef double [:] p1 = p
    cdef double [:] Y1 = MM


    if l==0:
        MM = -M0

    if l==1:
        MM = -M0*p
    
    if l==2:
        for i in prange(N, nogil=True):
            Y1[i + 0*N] = S0*(p1[i]*p1[i]            -(1.0/3))
            Y1[i + 1*N] = S0*(p1[i + N]*p1[i + N] -(1.0/3))
            Y1[i + 2*N] = S0*(p1[i]*p1[i + N])
            Y1[i + 3*N] = S0*(p1[i]*p1[i + 2*N])
            Y1[i + 4*N] = S0*(p1[i + N]*p1[i + 2*N])

    if l==3:
        for i in range(N):
            MM[i]      = M0*(p[i]*p[i]*p[i]            - 3/5*p[i]);
            MM[i+N]   = M0*(p[i+N]*p[i+N]*p[i+N]   - 3/5*p[i+N]);
            MM[i+2*N] = M0*(p[i]*p[i]*p[i+N]           - 1/5*p[i+N]);
            MM[i+3*N] = M0*(p[i]*p[i]*p[i+2*N]     - 1/5*p[i+2*N]);
            MM[i+4*N] = M0*(p[i]*p[i+N]*p[1+N]      -1/5* p[i]);
            MM[i+5*N] = M0*(p[i+N]*p[i+N]*p[i+2*N]);
            MM[i+6*N] = M0*(p[i+N]*p[i+N]*p[i+2*N] -1/5*p[i+2*N]);
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
    import matplotlib.pyplot as plt
    fig = plt.figure(figsize=(4, 4))
    ax = fig.add_subplot(111, projection='3d')

    if l==0:
        speedVls = c0 + z*0
        ax.plot_surface(x, y, z,  rstride=1, cstride=1, facecolors=plt.cm.RdBu_r(speedVls))
        ax.set_axis_off()
        plt.show()
    
    elif l==1:
        speedVls = c0*np.cos(theta)
        ax.plot_surface(x, y, z,  rstride=1, cstride=1, facecolors=plt.cm.RdBu_r(speedVls))
        ax.set_axis_off()
        plt.show()
    
    elif l==2:
        speedVls = 4*c0*(np.cos(theta)*np.cos(theta) - 1.0/3)
        ax.plot_surface(x, y, z,  rstride=1, cstride=1, facecolors=plt.cm.RdBu_r(speedVls))
        ax.set_axis_off()
        plt.show()
    
    elif l==3:
        speedVls = 4*c0*(np.cos(theta)*np.cos(theta)*np.cos(theta) - 3*np.cos(theta)/5.0)
        ax.plot_surface(x, y, z,  rstride=1, cstride=1, facecolors=plt.cm.RdBu_r(speedVls))
        ax.set_axis_off()
        plt.show()

    else:
        print('Not yet implemented...')





####------------Mean-Squared Displacement (MSD) ----------
cpdef MSD2d(double [:] d2, double [:] xt, double [:] yt):
    """
    Mean square displacement in two dimensions 
    xt  : time series of x-coordinate
    yt  : time series of y-coordinate
    d2  : square of displacement 
    Nt  : total number of data points 
    tdev: time difference
    returns: updates the dr2 vector
    """
    cdef int i, j, tdev=np.size(d2), Nt=np.size(xt)
    cdef double dr2, dx, dy
    for i in range(tdev): 
        dr2=0
        for j in range(Nt-i):
            dx = xt[i+j] - xt[j]
            dy = yt[i+j] - yt[j]
            dr2 += dx*dx + dy*dy 
        d2[i] = dr2/(Nt-i)
    return 



cpdef MSD1d(double [:] d2, double [:] xt):
    cdef int i, j, tdev=np.size(d2), Nt=np.size(xt)
    cdef double dr2, dx, dy, dz
    for i in range(tdev): 
        dr2=0
        for j in range(Nt-i):
            dx = xt[i+j] - xt[j]
            dr2 += dx*dx
        d2[i] = dr2/(Nt-i)
    return   



cpdef MSD3d(double [:] d2, double [:] xt, double [:] yt,  double [:] zt):
    cdef int i, j, tdev=np.size(d2), Nt=np.size(xt)
    cdef double dr2, dx, dy, dz
    for i in range(tdev): 
        dr2=0
        for j in range(Nt-i):
            dx = xt[i+j] - xt[j]
            dy = yt[i+j] - yt[j]
            dz = zt[i+j] - zt[j]
            dr2 += dx*dx + dy*dy + dz*dz
        d2[i] = dr2/(Nt-i)
    return  
