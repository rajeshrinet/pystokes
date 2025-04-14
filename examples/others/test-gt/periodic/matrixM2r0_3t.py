import numpy
from math import *
PI = 3.14159265359

def GM2_1s1s(xi, b,eta):
    return numpy.array([[(1/8)*(8*xi + (-160/9)*b**2*xi**3)/(PI**(3/2)*eta), 0, 0], [0, (1/8)*(8*xi + (-160/9)*b**2*xi**3)/(PI**(3/2)*eta), 0], [0, 0, (1/8)*(8*xi + (-160/9)*b**2*xi**3)/(PI**(3/2)*eta)]])

def GM2_2a1s(xi, b,eta):
    return numpy.array([[0, 0, 0], [0, 0, 0], [0, 0, 0]])

def GM2_1s2a(xi, b,eta):
    return numpy.array([[0, 0, 0], [0, 0, 0], [0, 0, 0]])

def GM2_2a2a(xi, b,eta):
    return numpy.array([[(10/3)*b**2*xi**3/(PI**(3/2)*eta), 0, 0], [0, (10/3)*b**2*xi**3/(PI**(3/2)*eta), 0], [0, 0, (10/3)*b**2*xi**3/(PI**(3/2)*eta)]])

def GM2_H1s(xi, b,eta):
    return numpy.array([[0, 0, 0], [0, 0, 0], [0, 0, 0]])

def GM2_H2a(xi, b,eta):
    return numpy.array([[0, 0, 0], [0, 0, 0], [0, 0, 0]])

def KM2_HH(xi, b,eta):
    return numpy.array([[0, 0, 0], [0, 0, 0], [0, 0, 0]])

def GM2_HH(xi, b,eta):
    return numpy.array([[0, 0, 0], [0, 0, 0], [0, 0, 0]])

def GM2_1sH(xi, b,eta):
    return numpy.array([[4*b**2*xi**3/(3*PI**(3/2)*eta), 0, 0], [0, 4*b**2*xi**3/(3*PI**(3/2)*eta), 0], [0, 0, 4*b**2*xi**3/(3*PI**(3/2)*eta)]])

def GM2_2aH(xi, b,eta):
    return numpy.array([[0, 0, 0], [0, 0, 0], [0, 0, 0]])

def KM2_1sH(xi, b,eta):
    return numpy.array([[8*b**3*xi**3/(5*numpy.sqrt(PI)), 0, 0], [0, 8*b**3*xi**3/(5*numpy.sqrt(PI)), 0], [0, 0, 8*b**3*xi**3/(5*numpy.sqrt(PI))]])

def KM2_2aH(xi, b,eta):
    return numpy.array([[0, 0, 0], [0, 0, 0], [0, 0, 0]])

