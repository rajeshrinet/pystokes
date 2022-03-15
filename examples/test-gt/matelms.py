##hard-coded matrix elements 

import numpy as np
PI = 3.14159265359


##
## define matrices eventually used in the direct-solver: hat indicates that (3t) is excluded
##

def hatGH1s(xij,yij,zij, b,eta):
    return np.block([[G2s1s(xij,yij,zij, b,eta)],
                     [G3a1s(xij,yij,zij, b,eta)],
                     [G3s1s(xij,yij,zij, b,eta)]])

def hatGH2a(xij,yij,zij, b,eta):
    return np.block([[G2s2a(xij,yij,zij, b,eta)],
                     [G3a2a(xij,yij,zij, b,eta)],
                     [G3s2a(xij,yij,zij, b,eta)]])

def hatGHH(xij,yij,zij, b,eta):
    return np.block([[G2s2s(xij,yij,zij, b,eta), G2s3a(xij,yij,zij, b,eta), G2s3s(xij,yij,zij, b,eta)],
                     [G3a2s(xij,yij,zij, b,eta), G3a3a(xij,yij,zij, b,eta), G3a3s(xij,yij,zij, b,eta)],
                     [G3s2s(xij,yij,zij, b,eta), G3s3a(xij,yij,zij, b,eta), G3s3s(xij,yij,zij, b,eta)]])

def hatKHH(xij,yij,zij, b,eta):
    nonzero = np.block([[K2s2s(xij,yij,zij, b,eta), K2s3t(xij,yij,zij, b,eta)],
                        [K3a2s(xij,yij,zij, b,eta), K3a3t(xij,yij,zij, b,eta)],
                        [K3s2s(xij,yij,zij, b,eta), K3s3t(xij,yij,zij, b,eta)]])
    return np.block([nonzero, np.zeros([17,9])])


def hatGH3t(xij,yij,zij, b,eta):
    return np.block([[G2s3t(xij,yij,zij, b,eta)],
                     [G3a3t(xij,yij,zij, b,eta)],
                     [G3s3t(xij,yij,zij, b,eta)]])


##
## and the more general ones containing (3t)
##


def G1sH(xij,yij,zij, b,eta):
    return np.block([G1s2s(xij,yij,zij, b,eta), G1s3t(xij,yij,zij, b,eta), G1s3a(xij,yij,zij, b,eta), G1s3s(xij,yij,zij, b,eta)])

def G2aH(xij,yij,zij, b,eta):
    return np.block([G2a2s(xij,yij,zij, b,eta), G2a3t(xij,yij,zij, b,eta), G2a3a(xij,yij,zij, b,eta), G2a3s(xij,yij,zij, b,eta)])

def K1sH(xij,yij,zij, b,eta):
    return np.block([K1s2s(xij,yij,zij, b,eta), K1s3t(xij,yij,zij, b,eta), np.zeros([3,12])])

def K2aH(xij,yij,zij, b,eta):
    return np.block([K2a2s(xij,yij,zij, b,eta), K2a3t(xij,yij,zij, b,eta), np.zeros([3,12])])

def GH1s(xij,yij,zij, b,eta):
    return np.block([[G2s1s(xij,yij,zij, b,eta)],[G3t1s(xij,yij,zij, b,eta)],[G3a1s(xij,yij,zij, b,eta)],[G3s1s(xij,yij,zij, b,eta)]])

def GH2a(xij,yij,zij, b,eta):
    return np.block([[G2s2a(xij,yij,zij, b,eta)],
                     [G3t2a(xij,yij,zij, b,eta)],
                     [G3a2a(xij,yij,zij, b,eta)],
                     [G3s2a(xij,yij,zij, b,eta)]])


def GHH(xij,yij,zij, b,eta):
    return np.block([[G2s2s(xij,yij,zij, b,eta), G2s3t(xij,yij,zij, b,eta), G2s3a(xij,yij,zij, b,eta), G2s3s(xij,yij,zij, b,eta)],
                     [G3t2s(xij,yij,zij, b,eta), G3t3t(xij,yij,zij, b,eta), G3t3a(xij,yij,zij, b,eta), G3t3s(xij,yij,zij, b,eta)],
                     [G3a2s(xij,yij,zij, b,eta), G3a3t(xij,yij,zij, b,eta), G3a3a(xij,yij,zij, b,eta), G3a3s(xij,yij,zij, b,eta)],
                     [G3s2s(xij,yij,zij, b,eta), G3s3t(xij,yij,zij, b,eta), G3s3a(xij,yij,zij, b,eta), G3s3s(xij,yij,zij, b,eta)]])


def KHH(xij,yij,zij, b,eta):
    nonzero = np.block([[K2s2s(xij,yij,zij, b,eta), K2s3t(xij,yij,zij, b,eta)],
                        [K3t2s(xij,yij,zij, b,eta), K3t3t(xij,yij,zij, b,eta)],
                        [K3a2s(xij,yij,zij, b,eta), K3a3t(xij,yij,zij, b,eta)],
                        [K3s2s(xij,yij,zij, b,eta), K3s3t(xij,yij,zij, b,eta)]])
    return np.block([nonzero, np.zeros([20,12])])


def halfMinusKHH(xij,yij,zij, b,eta):
    return 0.5*np.identity(20) - KHH(xij,yij,zij, b,eta)





################################
################################

##
## Matrix elements
##



##
## lowest order matrix elements (FTS)
##

def G1s1s(xij,yij,zij, b,eta):
    return np.array([[(6*xij**4 + 9*xij**2*(yij**2 + zij**2) + 3*(yij**2 + zij**2)**2 + 2*b**2*(-2*xij**2 + yij**2 + zij**2))/(24.*eta*PI*(xij**2 + yij**2 + zij**2)**2.5),
     (xij*yij*(-2*b**2 + xij**2 + yij**2 + zij**2))/(8.*eta*PI*(xij**2 + yij**2 + zij**2)**2.5),(xij*zij*(-2*b**2 + xij**2 + yij**2 + zij**2))/(8.*eta*PI*(xij**2 + yij**2 + zij**2)**2.5)],
     [(xij*yij*(-2*b**2 + xij**2 + yij**2 + zij**2))/(8.*eta*PI*(xij**2 + yij**2 + zij**2)**2.5),
     (2*b**2*(xij**2 - 2*yij**2 + zij**2) + 3*(xij**4 + 2*yij**4 + 3*yij**2*zij**2 + zij**4 + xij**2*(3*yij**2 + 2*zij**2)))/(24.*eta*PI*(xij**2 + yij**2 + zij**2)**2.5),
     (yij*zij*(-2*b**2 + xij**2 + yij**2 + zij**2))/(8.*eta*PI*(xij**2 + yij**2 + zij**2)**2.5)],
     [(xij*zij*(-2*b**2 + xij**2 + yij**2 + zij**2))/(8.*eta*PI*(xij**2 + yij**2 + zij**2)**2.5),(yij*zij*(-2*b**2 + xij**2 + yij**2 + zij**2))/(8.*eta*PI*(xij**2 + yij**2 + zij**2)**2.5),
     (2*b**2*(xij**2 + yij**2 - 2*zij**2) + 3*(xij**4 + yij**4 + 3*yij**2*zij**2 + 2*zij**4 + xij**2*(2*yij**2 + 3*zij**2)))/(24.*eta*PI*(xij**2 + yij**2 + zij**2)**2.5)]])



def G2a1s(xij,yij,zij, b,eta):
    return  np.array([[0,(b*zij)/(4.*eta*PI*(xij**2 + yij**2 + zij**2)**1.5),-0.25*(b*yij)/(eta*PI*(xij**2 + yij**2 + zij**2)**1.5)],
     [-0.25*(b*zij)/(eta*PI*(xij**2 + yij**2 + zij**2)**1.5),0,(b*xij)/(4.*eta*PI*(xij**2 + yij**2 + zij**2)**1.5)],
     [(b*yij)/(4.*eta*PI*(xij**2 + yij**2 + zij**2)**1.5),-0.25*(b*xij)/(eta*PI*(xij**2 + yij**2 + zij**2)**1.5),0]])


def G1s2a(xij,yij,zij, b,eta):
    return  np.array([[0.,(0.039788735772973836*b*zij)/(eta*(xij**2 + yij**2 + zij**2)**1.5),(-0.039788735772973836*b*yij)/(eta*(xij**2 + yij**2 + zij**2)**1.5)],
     [(-0.039788735772973836*b*zij)/(eta*(xij**2 + yij**2 + zij**2)**1.5),0.,(0.039788735772973836*b*xij)/(eta*(xij**2 + yij**2 + zij**2)**1.5)],
     [(0.039788735772973836*b*yij)/(eta*(xij**2 + yij**2 + zij**2)**1.5),(-0.039788735772973836*b*xij)/(eta*(xij**2 + yij**2 + zij**2)**1.5),0.]])



def G2a2a(xij,yij,zij, b,eta):
    return  np.array([[(-0.039788735772973836*b**2*(-2.*xij**2 + yij**2 + zij**2))/(eta*(xij**2 + yij**2 + zij**2)**2.5),
     (0.1193662073189215*b**2*xij*yij)/(eta*(xij**2 + yij**2 + zij**2)**2.5),
     (0.1193662073189215*b**2*xij*zij)/(eta*(xij**2 + yij**2 + zij**2)**2.5)],
     [(0.1193662073189215*b**2*xij*yij)/(eta*(xij**2 + yij**2 + zij**2)**2.5),
     (-0.039788735772973836*b**2*(xij**2 - 2.*yij**2 + zij**2))/(eta*(xij**2 + yij**2 + zij**2)**2.5),
     (0.1193662073189215*b**2*yij*zij)/(eta*(xij**2 + yij**2 + zij**2)**2.5)],
     [(0.1193662073189215*b**2*xij*zij)/(eta*(xij**2 + yij**2 + zij**2)**2.5),
     (0.1193662073189215*b**2*yij*zij)/(eta*(xij**2 + yij**2 + zij**2)**2.5),
     (-0.039788735772973836*b**2*(xij**2 + yij**2 - 2.*zij**2))/(eta*(xij**2 + yij**2 + zij**2)**2.5)]])



def G1s2s(xij,yij,zij, b,eta):
    return np.array([[(-0.12732395447351627*b*xij*(-0.625*xij**4 + 0.3125*yij**4 + 0.625*yij**2*zij**2 + 0.3125*zij**4 + b**2*(1.*xij**2 - 1.4999999999999998*yij**2 - 1.4999999999999998*zij**2) + 
     xij**2*(-0.3125*yij**2 - 0.3125*zij**2)))/(eta*(xij**2 + yij**2 + zij**2)**3.5),
     (0.06366197723675814*b*yij*(b**2*(-4.*xij**2 + 1.*yij**2 + 1.*zij**2) + xij**2*(1.875*xij**2 + 1.875*yij**2 + 1.875*zij**2)))/(eta*(xij**2 + yij**2 + zij**2)**3.5),
     (0.06366197723675814*b*zij*(b**2*(-4.*xij**2 + 1.*yij**2 + 1.*zij**2) + xij**2*(1.875*xij**2 + 1.875*yij**2 + 1.875*zij**2)))/(eta*(xij**2 + yij**2 + zij**2)**3.5),
     (0.06366197723675814*b*xij*(-0.625*xij**4 + 1.25*yij**4 + 0.625*yij**2*zij**2 - 0.625*zij**4 + xij**2*(0.625*yij**2 - 1.25*zij**2) + b**2*(1.*xij**2 - 4.*yij**2 + 1.*zij**2)))/(eta*(xij**2 + yij**2 + zij**2)**3.5),
     (0.1193662073189215*b*xij*yij*zij*(-2.6666666666666665*b**2 + 1.*xij**2 + 1.*yij**2 + 1.*zij**2))/(eta*(xij**2 + yij**2 + zij**2)**3.5)],
     [(0.06366197723675814*b*yij*(1.25*xij**4 - 0.625*yij**4 - 1.25*yij**2*zij**2 - 0.625*zij**4 + xij**2*(0.625*yij**2 + 0.625*zij**2) + b**2*(-4.*xij**2 + 1.*yij**2 + 1.*zij**2)))/
     (eta*(xij**2 + yij**2 + zij**2)**3.5),(0.06366197723675814*b*xij*(b**2*(1.*xij**2 - 4.*yij**2 + 1.*zij**2) + yij**2*(1.875*xij**2 + 1.875*yij**2 + 1.875*zij**2)))/(eta*(xij**2 + yij**2 + zij**2)**3.5),
     (0.1193662073189215*b*xij*yij*zij*(-2.6666666666666665*b**2 + 1.*xij**2 + 1.*yij**2 + 1.*zij**2))/(eta*(xij**2 + yij**2 + zij**2)**3.5),
     (0.19098593171027442*b*yij*(-0.20833333333333334*xij**4 + 0.4166666666666667*yij**4 + 0.20833333333333334*yij**2*zij**2 - 0.20833333333333334*zij**4 + 
     xij**2*(0.20833333333333334*yij**2 - 0.4166666666666667*zij**2) + b**2*(1.*xij**2 - 0.6666666666666667*yij**2 + 1.*zij**2)))/(eta*(xij**2 + yij**2 + zij**2)**3.5),
     (0.06366197723675814*b*zij*(b**2*(1.*xij**2 - 4.*yij**2 + 1.*zij**2) + yij**2*(1.875*xij**2 + 1.875*yij**2 + 1.875*zij**2)))/(eta*(xij**2 + yij**2 + zij**2)**3.5)],
     [(0.06366197723675814*b*zij*(1.25*xij**4 - 0.625*yij**4 - 1.25*yij**2*zij**2 - 0.625*zij**4 + xij**2*(0.625*yij**2 + 0.625*zij**2) + b**2*(-4.*xij**2 + 1.*yij**2 + 1.*zij**2)))/
     (eta*(xij**2 + yij**2 + zij**2)**3.5),(0.1193662073189215*b*xij*yij*zij*(-2.6666666666666665*b**2 + 1.*xij**2 + 1.*yij**2 + 1.*zij**2))/(eta*(xij**2 + yij**2 + zij**2)**3.5),
     (0.06366197723675814*b*xij*(b**2*(1.*xij**2 + 1.*yij**2 - 4.*zij**2) + zij**2*(1.875*xij**2 + 1.875*yij**2 + 1.875*zij**2)))/(eta*(xij**2 + yij**2 + zij**2)**3.5),
     (0.06366197723675814*b*zij*(-0.625*xij**4 + 1.25*yij**4 + 0.625*yij**2*zij**2 - 0.625*zij**4 + xij**2*(0.625*yij**2 - 1.25*zij**2) + b**2*(1.*xij**2 - 4.*yij**2 + 1.*zij**2)))/(eta*(xij**2 + yij**2 + zij**2)**3.5),
     (0.06366197723675814*b*yij*(b**2*(1.*xij**2 + 1.*yij**2 - 4.*zij**2) + zij**2*(1.875*xij**2 + 1.875*yij**2 + 1.875*zij**2)))/(eta*(xij**2 + yij**2 + zij**2)**3.5)]])



def G2a2s(xij,yij,zij, b,eta):
    return np.array([[0.,(0.1193662073189215*b**2*xij*zij)/(eta*(xij**2 + yij**2 + zij**2)**2.5),
     (-0.1193662073189215*b**2*xij*yij)/(eta*(xij**2 + yij**2 + zij**2)**2.5),
     (0.238732414637843*b**2*yij*zij)/(eta*(xij**2 + yij**2 + zij**2)**2.5),
     (-0.1193662073189215*b**2*(yij**2 - 1.*zij**2))/(eta*(xij**2 + yij**2 + zij**2)**2.5)],
     [(-0.238732414637843*b**2*xij*zij)/(eta*(xij**2 + yij**2 + zij**2)**2.5),
     (-0.1193662073189215*b**2*yij*zij)/(eta*(xij**2 + yij**2 + zij**2)**2.5),
     (0.1193662073189215*b**2*(xij**2 - 1.*zij**2))/(eta*(xij**2 + yij**2 + zij**2)**2.5),0.,
     (0.1193662073189215*b**2*xij*yij)/(eta*(xij**2 + yij**2 + zij**2)**2.5)],
     [(0.238732414637843*b**2*xij*yij)/(eta*(xij**2 + yij**2 + zij**2)**2.5),
     (-0.1193662073189215*b**2*(xij**2 - 1.*yij**2))/(eta*(xij**2 + yij**2 + zij**2)**2.5),
     (0.1193662073189215*b**2*yij*zij)/(eta*(xij**2 + yij**2 + zij**2)**2.5),
     (-0.238732414637843*b**2*xij*yij)/(eta*(xij**2 + yij**2 + zij**2)**2.5),
     (-0.1193662073189215*b**2*xij*zij)/(eta*(xij**2 + yij**2 + zij**2)**2.5)]])




def G2s1s(xij,yij,zij, b,eta):
    return np.array([[(0.12732395447351627*b*xij*(-0.625*xij**4 + 0.3125*yij**4 + 0.625*yij**2*zij**2 + 0.3125*zij**4 + b**2*(1.*xij**2 - 1.4999999999999998*yij**2 - 1.4999999999999998*zij**2) + 
     xij**2*(-0.3125*yij**2 - 0.3125*zij**2)))/(eta*(xij**2 + yij**2 + zij**2)**3.5),
     (-0.06366197723675814*b*yij*(1.25*xij**4 - 0.625*yij**4 - 1.25*yij**2*zij**2 - 0.625*zij**4 + xij**2*(0.625*yij**2 + 0.625*zij**2) + b**2*(-4.*xij**2 + 1.*yij**2 + 1.*zij**2)))/
     (eta*(xij**2 + yij**2 + zij**2)**3.5),(-0.06366197723675814*b*zij*(1.25*xij**4 - 0.625*yij**4 - 1.25*yij**2*zij**2 - 0.625*zij**4 + xij**2*(0.625*yij**2 + 0.625*zij**2) + 
     b**2*(-4.*xij**2 + 1.*yij**2 + 1.*zij**2)))/(eta*(xij**2 + yij**2 + zij**2)**3.5)],
     [(-0.06366197723675814*b*yij*(b**2*(-4.*xij**2 + 1.*yij**2 + 1.*zij**2) + xij**2*(1.875*xij**2 + 1.875*yij**2 + 1.875*zij**2)))/(eta*(xij**2 + yij**2 + zij**2)**3.5),
     (-0.06366197723675814*b*xij*(b**2*(1.*xij**2 - 4.*yij**2 + 1.*zij**2) + yij**2*(1.875*xij**2 + 1.875*yij**2 + 1.875*zij**2)))/(eta*(xij**2 + yij**2 + zij**2)**3.5),
     (-0.1193662073189215*b*xij*yij*zij*(-2.6666666666666665*b**2 + 1.*xij**2 + 1.*yij**2 + 1.*zij**2))/(eta*(xij**2 + yij**2 + zij**2)**3.5)],
     [(-0.06366197723675814*b*zij*(b**2*(-4.*xij**2 + 1.*yij**2 + 1.*zij**2) + xij**2*(1.875*xij**2 + 1.875*yij**2 + 1.875*zij**2)))/(eta*(xij**2 + yij**2 + zij**2)**3.5),
     (-0.1193662073189215*b*xij*yij*zij*(-2.6666666666666665*b**2 + 1.*xij**2 + 1.*yij**2 + 1.*zij**2))/(eta*(xij**2 + yij**2 + zij**2)**3.5),
     (-0.06366197723675814*b*xij*(b**2*(1.*xij**2 + 1.*yij**2 - 4.*zij**2) + zij**2*(1.875*xij**2 + 1.875*yij**2 + 1.875*zij**2)))/(eta*(xij**2 + yij**2 + zij**2)**3.5)],
     [(-0.06366197723675814*b*xij*(-0.625*xij**4 + 1.25*yij**4 + 0.625*yij**2*zij**2 - 0.625*zij**4 + xij**2*(0.625*yij**2 - 1.25*zij**2) + b**2*(1.*xij**2 - 4.*yij**2 + 1.*zij**2)))/
     (eta*(xij**2 + yij**2 + zij**2)**3.5),(-0.19098593171027442*b*yij*(-0.20833333333333334*xij**4 + 0.4166666666666667*yij**4 + 0.20833333333333334*yij**2*zij**2 - 0.20833333333333334*zij**4 + 
     xij**2*(0.20833333333333334*yij**2 - 0.4166666666666667*zij**2) + b**2*(1.*xij**2 - 0.6666666666666667*yij**2 + 1.*zij**2)))/(eta*(xij**2 + yij**2 + zij**2)**3.5),
     (-0.06366197723675814*b*zij*(-0.625*xij**4 + 1.25*yij**4 + 0.625*yij**2*zij**2 - 0.625*zij**4 + xij**2*(0.625*yij**2 - 1.25*zij**2) + b**2*(1.*xij**2 - 4.*yij**2 + 1.*zij**2)))/
     (eta*(xij**2 + yij**2 + zij**2)**3.5)],[(-0.1193662073189215*b*xij*yij*zij*(-2.6666666666666665*b**2 + 1.*xij**2 + 1.*yij**2 + 1.*zij**2))/(eta*(xij**2 + yij**2 + zij**2)**3.5),
     (-0.06366197723675814*b*zij*(b**2*(1.*xij**2 - 4.*yij**2 + 1.*zij**2) + yij**2*(1.875*xij**2 + 1.875*yij**2 + 1.875*zij**2)))/(eta*(xij**2 + yij**2 + zij**2)**3.5),
     (-0.06366197723675814*b*yij*(b**2*(1.*xij**2 + 1.*yij**2 - 4.*zij**2) + zij**2*(1.875*xij**2 + 1.875*yij**2 + 1.875*zij**2)))/(eta*(xij**2 + yij**2 + zij**2)**3.5)]])




def G2s2a(xij,yij,zij, b,eta):
    return np.array([[0.,(-0.1193662073189215*b**2*xij*zij)/(eta*(xij**2 + yij**2 + zij**2)**2.5),(0.1193662073189215*b**2*xij*yij)/(eta*(xij**2 + yij**2 + zij**2)**2.5)],
     [(0.05968310365946075*b**2*xij*zij)/(eta*(xij**2 + yij**2 + zij**2)**2.5),(-0.05968310365946075*b**2*yij*zij)/(eta*(xij**2 + yij**2 + zij**2)**2.5),
     (-0.05968310365946075*b**2*(xij**2 - 1.*yij**2))/(eta*(xij**2 + yij**2 + zij**2)**2.5)],
     [(-0.05968310365946075*b**2*xij*yij)/(eta*(xij**2 + yij**2 + zij**2)**2.5),(0.05968310365946075*b**2*(xij**2 - 1.*zij**2))/(eta*(xij**2 + yij**2 + zij**2)**2.5),
     (0.05968310365946075*b**2*yij*zij)/(eta*(xij**2 + yij**2 + zij**2)**2.5)],[(0.1193662073189215*b**2*yij*zij)/(eta*(xij**2 + yij**2 + zij**2)**2.5),0.,
     (-0.1193662073189215*b**2*xij*yij)/(eta*(xij**2 + yij**2 + zij**2)**2.5)],[(-0.05968310365946075*b**2*(yij**2 - 1.*zij**2))/(eta*(xij**2 + yij**2 + zij**2)**2.5),
     (0.05968310365946075*b**2*xij*yij)/(eta*(xij**2 + yij**2 + zij**2)**2.5),(-0.05968310365946075*b**2*xij*zij)/(eta*(xij**2 + yij**2 + zij**2)**2.5)]])




def G2s2s(xij,yij,zij, b,eta):
    return np.array([[(0.3819718634205488*b**2*(-0.4166666666666667*xij**6 - 0.10416666666666667*yij**6 - 0.3125*yij**4*zij**2 - 0.3125*yij**2*zij**4 - 0.10416666666666667*zij**6 + xij**4*(0.625*yij**2 + 0.625*zij**2) + 
     xij**2*(0.9375*yij**4 + 1.875*yij**2*zij**2 + 0.9375*zij**4) + b**2*(1.*xij**4 + 0.375*yij**4 + 0.75*yij**2*zij**2 + 0.375*zij**4 + xij**2*(-3.*yij**2 - 3.*zij**2))))/(eta*(xij**2 + yij**2 + zij**2)**4.5),
     (-0.716197243913529*b**2*xij*yij*(0.5*xij**4 - 0.3333333333333333*yij**4 - 0.6666666666666666*yij**2*zij**2 - 0.3333333333333333*zij**4 + xij**2*(0.16666666666666666*yij**2 + 0.16666666666666666*zij**2) + 
     b**2*(-1.3333333333333333*xij**2 + 1.*yij**2 + 1.*zij**2)))/(eta*(xij**2 + yij**2 + zij**2)**4.5),
     (-0.716197243913529*b**2*xij*zij*(0.5*xij**4 - 0.3333333333333333*yij**4 - 0.6666666666666666*yij**2*zij**2 - 0.3333333333333333*zij**4 + xij**2*(0.16666666666666666*yij**2 + 0.16666666666666666*zij**2) + 
     b**2*(-1.3333333333333333*xij**2 + 1.*yij**2 + 1.*zij**2)))/(eta*(xij**2 + yij**2 + zij**2)**4.5),
     (-0.19098593171027442*b**2*(-0.4166666666666667*xij**6 - 0.4166666666666667*yij**6 - 0.625*yij**4*zij**2 + 0.20833333333333334*zij**6 + xij**4*(1.875*yij**2 - 0.625*zij**2) + 
     xij**2*(1.875*yij**4 + 1.875*yij**2*zij**2) + b**2*(1.*xij**4 + 1.*yij**4 + 0.75*yij**2*zij**2 - 0.25*zij**4 + xij**2*(-6.75*yij**2 + 0.75*zij**2))))/(eta*(xij**2 + yij**2 + zij**2)**4.5),
     (0.1193662073189215*b**2*yij*zij*(-4.*xij**4 + yij**4 + 2.*yij**2*zij**2 + zij**4 + xij**2*(-3.*yij**2 - 3.*zij**2) + b**2*(12.*xij**2 - 2.*yij**2 - 2.*zij**2)))/(eta*(xij**2 + yij**2 + zij**2)**4.5)],
     [(-0.716197243913529*b**2*xij*yij*(0.5*xij**4 - 0.3333333333333333*yij**4 - 0.6666666666666666*yij**2*zij**2 - 0.3333333333333333*zij**4 + xij**2*(0.16666666666666666*yij**2 + 0.16666666666666666*zij**2) + 
     b**2*(-1.3333333333333333*xij**2 + 1.*yij**2 + 1.*zij**2)))/(eta*(xij**2 + yij**2 + zij**2)**4.5),
     (-0.19098593171027445*b**2*(-0.3125*xij**6 - 0.3125*yij**6 - 0.625*yij**4*zij**2 - 0.3125*yij**2*zij**4 + xij**4*(2.1875*yij**2 - 0.625*zij**2) + xij**2*(2.1875*yij**4 + 1.875*yij**2*zij**2 - 0.3125*zij**4) + 
     b**2*(1.*xij**4 + 1.*yij**4 + 0.7499999999999999*yij**2*zij**2 - 0.25*zij**4 + xij**2*(-6.75*yij**2 + 0.7499999999999999*zij**2))))/(eta*(xij**2 + yij**2 + zij**2)**4.5),
     (0.05968310365946075*b**2*yij*zij*(-9.*xij**4 + yij**4 + 2.*yij**2*zij**2 + zij**4 + xij**2*(-8.*yij**2 - 8.*zij**2) + b**2*(24.*xij**2 - 4.*yij**2 - 4.*zij**2)))/(eta*(xij**2 + yij**2 + zij**2)**4.5),
     (-0.716197243913529*b**2*xij*yij*(-0.3333333333333333*xij**4 + 0.5*yij**4 + 0.16666666666666666*yij**2*zij**2 - 0.3333333333333333*zij**4 + xij**2*(0.16666666666666666*yij**2 - 0.6666666666666666*zij**2) + 
     b**2*(1.*xij**2 - 1.3333333333333333*yij**2 + 1.*zij**2)))/(eta*(xij**2 + yij**2 + zij**2)**4.5),
     (-0.238732414637843*b**2*xij*zij*(-0.25*xij**4 + 2.25*yij**4 + 2.*yij**2*zij**2 - 0.25*zij**4 + xij**2*(2.*yij**2 - 0.5*zij**2) + b**2*(1.*xij**2 - 6.*yij**2 + 1.*zij**2)))/(eta*(xij**2 + yij**2 + zij**2)**4.5)],
     [(-0.716197243913529*b**2*xij*zij*(0.5*xij**4 - 0.3333333333333333*yij**4 - 0.6666666666666666*yij**2*zij**2 - 0.3333333333333333*zij**4 + xij**2*(0.16666666666666666*yij**2 + 0.16666666666666666*zij**2) + 
     b**2*(-1.3333333333333333*xij**2 + 1.*yij**2 + 1.*zij**2)))/(eta*(xij**2 + yij**2 + zij**2)**4.5),
     (0.05968310365946075*b**2*yij*zij*(-9.*xij**4 + yij**4 + 2.*yij**2*zij**2 + zij**4 + xij**2*(-8.*yij**2 - 8.*zij**2) + b**2*(24.*xij**2 - 4.*yij**2 - 4.*zij**2)))/(eta*(xij**2 + yij**2 + zij**2)**4.5),
     (-0.19098593171027445*b**2*(-0.3125*xij**6 - 0.3125*yij**4*zij**2 - 0.625*yij**2*zij**4 - 0.3125*zij**6 + xij**4*(-0.625*yij**2 + 2.1875*zij**2) + xij**2*(-0.3125*yij**4 + 1.875*yij**2*zij**2 + 2.1875*zij**4) + 
     b**2*(1.*xij**4 - 0.25*yij**4 + 0.7499999999999999*yij**2*zij**2 + 1.*zij**4 + xij**2*(0.7499999999999999*yij**2 - 6.75*zij**2))))/(eta*(xij**2 + yij**2 + zij**2)**4.5),
     (-0.238732414637843*b**2*xij*zij*(-0.5*xij**4 + 2.*yij**4 + 1.5*yij**2*zij**2 - 0.5*zij**4 + xij**2*(1.5*yij**2 - 1.*zij**2) + b**2*(1.*xij**2 - 6.*yij**2 + 1.*zij**2)))/(eta*(xij**2 + yij**2 + zij**2)**4.5),
     (-0.238732414637843*b**2*xij*yij*(-0.25*xij**4 - 0.25*yij**4 + 2.*yij**2*zij**2 + 2.25*zij**4 + b**2*(1.*xij**2 + 1.*yij**2 - 6.*zij**2) + xij**2*(-0.5*yij**2 + 2.*zij**2)))/(eta*(xij**2 + yij**2 + zij**2)**4.5)],
     [(-0.19098593171027442*b**2*(-0.4166666666666667*xij**6 - 0.4166666666666667*yij**6 - 0.625*yij**4*zij**2 + 0.20833333333333334*zij**6 + xij**4*(1.875*yij**2 - 0.625*zij**2) + 
     xij**2*(1.875*yij**4 + 1.875*yij**2*zij**2) + b**2*(1.*xij**4 + 1.*yij**4 + 0.75*yij**2*zij**2 - 0.25*zij**4 + xij**2*(-6.75*yij**2 + 0.75*zij**2))))/(eta*(xij**2 + yij**2 + zij**2)**4.5),
     (-0.716197243913529*b**2*xij*yij*(-0.3333333333333333*xij**4 + 0.5*yij**4 + 0.16666666666666666*yij**2*zij**2 - 0.3333333333333333*zij**4 + xij**2*(0.16666666666666666*yij**2 - 0.6666666666666666*zij**2) + 
     b**2*(1.*xij**2 - 1.3333333333333333*yij**2 + 1.*zij**2)))/(eta*(xij**2 + yij**2 + zij**2)**4.5),
     (-0.238732414637843*b**2*xij*zij*(-0.5*xij**4 + 2.*yij**4 + 1.5*yij**2*zij**2 - 0.5*zij**4 + xij**2*(1.5*yij**2 - 1.*zij**2) + b**2*(1.*xij**2 - 6.*yij**2 + 1.*zij**2)))/(eta*(xij**2 + yij**2 + zij**2)**4.5),
     (0.1432394487827058*b**2*(-0.2777777777777778*xij**6 - 1.1111111111111112*yij**6 + 1.6666666666666667*yij**4*zij**2 + 2.5*yij**2*zij**4 - 0.2777777777777778*zij**6 + 
     xij**4*(2.5*yij**2 - 0.8333333333333334*zij**2) + xij**2*(1.6666666666666667*yij**4 + 5.*yij**2*zij**2 - 0.8333333333333334*zij**4) + 
     b**2*(1.*xij**4 + 2.6666666666666665*yij**4 - 8.*yij**2*zij**2 + 1.*zij**4 + xij**2*(-8.*yij**2 + 2.*zij**2))))/(eta*(xij**2 + yij**2 + zij**2)**4.5),
     (-0.716197243913529*b**2*yij*zij*(-0.3333333333333333*xij**4 + 0.5*yij**4 + 0.16666666666666666*yij**2*zij**2 - 0.3333333333333333*zij**4 + xij**2*(0.16666666666666666*yij**2 - 0.6666666666666666*zij**2) + 
     b**2*(1.*xij**2 - 1.3333333333333333*yij**2 + 1.*zij**2)))/(eta*(xij**2 + yij**2 + zij**2)**4.5)],
     [(0.1193662073189215*b**2*yij*zij*(-4.*xij**4 + yij**4 + 2.*yij**2*zij**2 + zij**4 + xij**2*(-3.*yij**2 - 3.*zij**2) + b**2*(12.*xij**2 - 2.*yij**2 - 2.*zij**2)))/(eta*(xij**2 + yij**2 + zij**2)**4.5),
     (-0.238732414637843*b**2*xij*zij*(-0.25*xij**4 + 2.25*yij**4 + 2.*yij**2*zij**2 - 0.25*zij**4 + xij**2*(2.*yij**2 - 0.5*zij**2) + b**2*(1.*xij**2 - 6.*yij**2 + 1.*zij**2)))/(eta*(xij**2 + yij**2 + zij**2)**4.5),
     (-0.238732414637843*b**2*xij*yij*(-0.25*xij**4 - 0.25*yij**4 + 2.*yij**2*zij**2 + 2.25*zij**4 + b**2*(1.*xij**2 + 1.*yij**2 - 6.*zij**2) + xij**2*(-0.5*yij**2 + 2.*zij**2)))/(eta*(xij**2 + yij**2 + zij**2)**4.5),
     (-0.716197243913529*b**2*yij*zij*(-0.3333333333333333*xij**4 + 0.5*yij**4 + 0.16666666666666666*yij**2*zij**2 - 0.3333333333333333*zij**4 + xij**2*(0.16666666666666666*yij**2 - 0.6666666666666666*zij**2) + 
     b**2*(1.*xij**2 - 1.3333333333333333*yij**2 + 1.*zij**2)))/(eta*(xij**2 + yij**2 + zij**2)**4.5),
     (0.04774648292756861*b**2*(1.25*yij**6 - 8.75*yij**4*zij**2 - 8.75*yij**2*zij**4 + 1.25*zij**6 + xij**4*(1.25*yij**2 + 1.25*zij**2) + xij**2*(2.5*yij**4 - 7.5*yij**2*zij**2 + 2.5*zij**4) + 
     b**2*(1.*xij**4 - 4.*yij**4 + 27.*yij**2*zij**2 - 4.*zij**4 + xij**2*(-2.9999999999999996*yij**2 - 2.9999999999999996*zij**2))))/(eta*(xij**2 + yij**2 + zij**2)**4.5)]])



##
## matrix elements connecting RBM with induced force moments: G^LH
##

def G1s3t(xij,yij,zij, b,eta):
    return np.array([[(b**2*(2*xij**2 - yij**2 - zij**2))/(20.*eta*PI*(xij**2 + yij**2 + zij**2)**2.5),
     (3*b**2*xij*yij)/(20.*eta*PI*(xij**2 + yij**2 + zij**2)**2.5),(3*b**2*xij*zij)/(20.*eta*PI*(xij**2 + yij**2 + zij**2)**2.5)],
     [(3*b**2*xij*yij)/(20.*eta*PI*(xij**2 + yij**2 + zij**2)**2.5),
     -0.05*(b**2*(xij**2 - 2*yij**2 + zij**2))/(eta*PI*(xij**2 + yij**2 + zij**2)**2.5),
     (3*b**2*yij*zij)/(20.*eta*PI*(xij**2 + yij**2 + zij**2)**2.5)],
     [(3*b**2*xij*zij)/(20.*eta*PI*(xij**2 + yij**2 + zij**2)**2.5),(3*b**2*yij*zij)/(20.*eta*PI*(xij**2 + yij**2 + zij**2)**2.5),
     -0.05*(b**2*(xij**2 + yij**2 - 2*zij**2))/(eta*PI*(xij**2 + yij**2 + zij**2)**2.5)]])


def G1s3a(xij,yij,zij, b,eta):
    return np.array([[0,(b**2*xij*zij)/(3.*eta*PI*(xij**2 + yij**2 + zij**2)**2.5),
      -0.3333333333333333*(b**2*xij*yij)/(eta*PI*(xij**2 + yij**2 + zij**2)**2.5),(b**2*yij*zij)/(2.*eta*PI*(xij**2 + yij**2 + zij**2)**2.5),
      (b**2*(xij**2 - 5*yij**2 + 4*zij**2))/(18.*eta*PI*(xij**2 + yij**2 + zij**2)**2.5)],
     [-0.5*(b**2*xij*zij)/(eta*PI*(xij**2 + yij**2 + zij**2)**2.5),
      -0.16666666666666666*(b**2*yij*zij)/(eta*PI*(xij**2 + yij**2 + zij**2)**2.5),
      (b**2*(5*xij**2 - yij**2 - 4*zij**2))/(18.*eta*PI*(xij**2 + yij**2 + zij**2)**2.5),0,
      (b**2*xij*yij)/(3.*eta*PI*(xij**2 + yij**2 + zij**2)**2.5)],
     [(b**2*xij*yij)/(2.*eta*PI*(xij**2 + yij**2 + zij**2)**2.5),
      -0.05555555555555555*(b**2*(5*xij**2 - 4*yij**2 - zij**2))/(eta*PI*(xij**2 + yij**2 + zij**2)**2.5),
      (b**2*yij*zij)/(6.*eta*PI*(xij**2 + yij**2 + zij**2)**2.5),-0.5*(b**2*xij*yij)/(eta*PI*(xij**2 + yij**2 + zij**2)**2.5),
      -0.16666666666666666*(b**2*xij*zij)/(eta*PI*(xij**2 + yij**2 + zij**2)**2.5)]])


def G1s3s(xij,yij,zij, b,eta):
    return np.array([[(-50*b**4*(8*xij**4 - 24*xij**2*(yij**2 + zij**2) + 3*(yij**2 + zij**2)**2) + 
        21*b**2*(8*xij**6 - 8*xij**4*(yij**2 + zij**2) - 15*xij**2*(yij**2 + zij**2)**2 + (yij**2 + zij**2)**3))/
      (280.*eta*PI*(xij**2 + yij**2 + zij**2)**4.5),(b**2*xij*yij*
        (-250*b**2*(4*xij**2 - 3*(yij**2 + zij**2)) + 7*(52*xij**4 + 29*xij**2*(yij**2 + zij**2) - 23*(yij**2 + zij**2)**2)))/
      (280.*eta*PI*(xij**2 + yij**2 + zij**2)**4.5),(b**2*xij*zij*
        (-250*b**2*(4*xij**2 - 3*(yij**2 + zij**2)) + 7*(52*xij**4 + 29*xij**2*(yij**2 + zij**2) - 23*(yij**2 + zij**2)**2)))/
      (280.*eta*PI*(xij**2 + yij**2 + zij**2)**4.5),(50*b**4*(4*xij**4 + 4*yij**4 + 3*yij**2*zij**2 - zij**4 + 3*xij**2*(-9*yij**2 + zij**2)) - 
        7*b**2*(12*xij**6 + (4*yij**2 - zij**2)*(yij**2 + zij**2)**2 + xij**4*(-47*yij**2 + 23*zij**2) - 
           5*xij**2*(11*yij**4 + 9*yij**2*zij**2 - 2*zij**4)))/(280.*eta*PI*(xij**2 + yij**2 + zij**2)**4.5),
     (b**2*yij*zij*(98*xij**4 - 50*b**2*(6*xij**2 - yij**2 - zij**2) + 91*xij**2*(yij**2 + zij**2) - 7*(yij**2 + zij**2)**2))/
      (56.*eta*PI*(xij**2 + yij**2 + zij**2)**4.5),(b**2*xij*yij*
        (250*b**2*(3*xij**2 - 4*yij**2 + 3*zij**2) - 21*(13*xij**4 - 12*yij**4 + yij**2*zij**2 + 13*zij**4 + xij**2*(yij**2 + 26*zij**2))))/
      (280.*eta*PI*(xij**2 + yij**2 + zij**2)**4.5),(b**2*xij*zij*
        (250*b**2*(xij**2 - 6*yij**2 + zij**2) - 7*(13*xij**4 - 62*yij**4 - 49*yij**2*zij**2 + 13*zij**4 + xij**2*(-49*yij**2 + 26*zij**2))))/
      (280.*eta*PI*(xij**2 + yij**2 + zij**2)**4.5)],[-0.0035714285714285713*
      (b**2*xij*yij*(250*b**2*(4*xij**2 - 3*(yij**2 + zij**2)) + 21*(-12*xij**4 + xij**2*(yij**2 + zij**2) + 13*(yij**2 + zij**2)**2)))/
       (eta*PI*(xij**2 + yij**2 + zij**2)**4.5),(50*b**4*(4*xij**4 + 4*yij**4 + 3*yij**2*zij**2 - zij**4 + 3*xij**2*(-9*yij**2 + zij**2)) - 
        7*b**2*(4*xij**6 + (12*yij**2 - zij**2)*(yij**2 + zij**2)**2 + xij**4*(-55*yij**2 + 7*zij**2) + 
           xij**2*(-47*yij**4 - 45*yij**2*zij**2 + 2*zij**4)))/(280.*eta*PI*(xij**2 + yij**2 + zij**2)**4.5),
     (b**2*yij*zij*(434*xij**4 - 250*b**2*(6*xij**2 - yij**2 - zij**2) + 343*xij**2*(yij**2 + zij**2) - 91*(yij**2 + zij**2)**2))/
      (280.*eta*PI*(xij**2 + yij**2 + zij**2)**4.5),(b**2*xij*yij*
        (250*b**2*(3*xij**2 - 4*yij**2 + 3*zij**2) - 7*(23*xij**4 - 52*yij**4 - 29*yij**2*zij**2 + 23*zij**4 + xij**2*(-29*yij**2 + 46*zij**2))))/
      (280.*eta*PI*(xij**2 + yij**2 + zij**2)**4.5),(b**2*xij*zij*
        (50*b**2*(xij**2 - 6*yij**2 + zij**2) - 7*(xij**4 - 14*yij**4 - 13*yij**2*zij**2 + zij**4 + xij**2*(-13*yij**2 + 2*zij**2))))/
      (56.*eta*PI*(xij**2 + yij**2 + zij**2)**4.5),(-50*b**4*(3*xij**4 + 8*yij**4 - 24*yij**2*zij**2 + 3*zij**4 + 6*xij**2*(-4*yij**2 + zij**2)) + 
        21*b**2*(xij**6 + 8*yij**6 - 8*yij**4*zij**2 - 15*yij**2*zij**4 + zij**6 + 3*xij**4*(-5*yij**2 + zij**2) + 
           xij**2*(-8*yij**4 - 30*yij**2*zij**2 + 3*zij**4)))/(280.*eta*PI*(xij**2 + yij**2 + zij**2)**4.5),
     (b**2*yij*zij*(250*b**2*(3*xij**2 - 4*yij**2 + 3*zij**2) - 
          7*(23*xij**4 - 52*yij**4 - 29*yij**2*zij**2 + 23*zij**4 + xij**2*(-29*yij**2 + 46*zij**2))))/(280.*eta*PI*(xij**2 + yij**2 + zij**2)**4.5)]
     ,[-0.0035714285714285713*(b**2*xij*zij*(250*b**2*(4*xij**2 - 3*(yij**2 + zij**2)) + 
           21*(-12*xij**4 + xij**2*(yij**2 + zij**2) + 13*(yij**2 + zij**2)**2)))/(eta*PI*(xij**2 + yij**2 + zij**2)**4.5),
     (b**2*yij*zij*(434*xij**4 - 250*b**2*(6*xij**2 - yij**2 - zij**2) + 343*xij**2*(yij**2 + zij**2) - 91*(yij**2 + zij**2)**2))/
      (280.*eta*PI*(xij**2 + yij**2 + zij**2)**4.5),(50*b**4*(4*xij**4 - yij**4 + 3*yij**2*zij**2 + 4*zij**4 + 3*xij**2*(yij**2 - 9*zij**2)) - 
        7*b**2*(4*xij**6 + xij**4*(7*yij**2 - 55*zij**2) - (yij**2 - 12*zij**2)*(yij**2 + zij**2)**2 + 
           xij**2*(2*yij**4 - 45*yij**2*zij**2 - 47*zij**4)))/(280.*eta*PI*(xij**2 + yij**2 + zij**2)**4.5),
     (b**2*xij*zij*(250*b**2*(xij**2 - 6*yij**2 + zij**2) - 
          7*(13*xij**4 - 62*yij**4 - 49*yij**2*zij**2 + 13*zij**4 + xij**2*(-49*yij**2 + 26*zij**2))))/(280.*eta*PI*(xij**2 + yij**2 + zij**2)**4.5),
     (b**2*xij*yij*(50*b**2*(xij**2 + yij**2 - 6*zij**2) - 7*(xij**4 + yij**4 - 13*yij**2*zij**2 - 14*zij**4 + xij**2*(2*yij**2 - 13*zij**2))))/
      (56.*eta*PI*(xij**2 + yij**2 + zij**2)**4.5),(b**2*yij*zij*
        (250*b**2*(3*xij**2 - 4*yij**2 + 3*zij**2) - 21*(13*xij**4 - 12*yij**4 + yij**2*zij**2 + 13*zij**4 + xij**2*(yij**2 + 26*zij**2))))/
      (280.*eta*PI*(xij**2 + yij**2 + zij**2)**4.5),(-50*b**4*(xij**4 - 4*yij**4 + 27*yij**2*zij**2 - 4*zij**4 - 3*xij**2*(yij**2 + zij**2)) + 
        7*b**2*(xij**6 - 4*yij**6 + 55*yij**4*zij**2 + 47*yij**2*zij**4 - 12*zij**6 - 2*xij**4*(yij**2 + 5*zij**2) + 
           xij**2*(-7*yij**4 + 45*yij**2*zij**2 - 23*zij**4)))/(280.*eta*PI*(xij**2 + yij**2 + zij**2)**4.5)]])



def G2a3a(xij,yij,zij, b,eta):
    return np.array([[(b**3*(2*xij**3 - 3*xij*(yij**2 + zij**2)))/(2.*eta*PI*(xij**2 + yij**2 + zij**2)**3.5),
      -0.5*(b**3*yij*(-4*xij**2 + yij**2 + zij**2))/(eta*PI*(xij**2 + yij**2 + zij**2)**3.5),
      -0.5*(b**3*zij*(-4*xij**2 + yij**2 + zij**2))/(eta*PI*(xij**2 + yij**2 + zij**2)**3.5),
      -0.5*(b**3*xij*(xij**2 - 4*yij**2 + zij**2))/(eta*PI*(xij**2 + yij**2 + zij**2)**3.5),
      (5*b**3*xij*yij*zij)/(2.*eta*PI*(xij**2 + yij**2 + zij**2)**3.5)],
     [-0.5*(b**3*yij*(-4*xij**2 + yij**2 + zij**2))/(eta*PI*(xij**2 + yij**2 + zij**2)**3.5),
      -0.5*(b**3*xij*(xij**2 - 4*yij**2 + zij**2))/(eta*PI*(xij**2 + yij**2 + zij**2)**3.5),
      (5*b**3*xij*yij*zij)/(2.*eta*PI*(xij**2 + yij**2 + zij**2)**3.5),
      (b**3*yij*(-3*xij**2 + 2*yij**2 - 3*zij**2))/(2.*eta*PI*(xij**2 + yij**2 + zij**2)**3.5),
      -0.5*(b**3*zij*(xij**2 - 4*yij**2 + zij**2))/(eta*PI*(xij**2 + yij**2 + zij**2)**3.5)],
     [-0.5*(b**3*zij*(-4*xij**2 + yij**2 + zij**2))/(eta*PI*(xij**2 + yij**2 + zij**2)**3.5),
      (5*b**3*xij*yij*zij)/(2.*eta*PI*(xij**2 + yij**2 + zij**2)**3.5),
      -0.5*(b**3*xij*(xij**2 + yij**2 - 4*zij**2))/(eta*PI*(xij**2 + yij**2 + zij**2)**3.5),
      -0.5*(b**3*zij*(xij**2 - 4*yij**2 + zij**2))/(eta*PI*(xij**2 + yij**2 + zij**2)**3.5),
      -0.5*(b**3*yij*(xij**2 + yij**2 - 4*zij**2))/(eta*PI*(xij**2 + yij**2 + zij**2)**3.5)]])


def G2a3s(xij,yij,zij, b,eta):
    return np.array([[0,-0.25*(b**3*zij*(-4*xij**2 + yij**2 + zij**2))/(eta*PI*(xij**2 + yij**2 + zij**2)**3.5),
      (b**3*yij*(-4*xij**2 + yij**2 + zij**2))/(4.*eta*PI*(xij**2 + yij**2 + zij**2)**3.5),
      (5*b**3*xij*yij*zij)/(2.*eta*PI*(xij**2 + yij**2 + zij**2)**3.5),(-5*b**3*xij*(yij**2 - zij**2))/(4.*eta*PI*(xij**2 + yij**2 + zij**2)**3.5),
      (-3*b**3*zij*(xij**2 - 4*yij**2 + zij**2))/(4.*eta*PI*(xij**2 + yij**2 + zij**2)**3.5),
      (b**3*yij*(xij**2 - 4*yij**2 + 11*zij**2))/(4.*eta*PI*(xij**2 + yij**2 + zij**2)**3.5)],
     [(3*b**3*zij*(-4*xij**2 + yij**2 + zij**2))/(4.*eta*PI*(xij**2 + yij**2 + zij**2)**3.5),
      (-5*b**3*xij*yij*zij)/(2.*eta*PI*(xij**2 + yij**2 + zij**2)**3.5),
      (b**3*xij*(4*xij**2 - yij**2 - 11*zij**2))/(4.*eta*PI*(xij**2 + yij**2 + zij**2)**3.5),
      (b**3*zij*(xij**2 - 4*yij**2 + zij**2))/(4.*eta*PI*(xij**2 + yij**2 + zij**2)**3.5),
      (5*b**3*yij*(xij**2 - zij**2))/(4.*eta*PI*(xij**2 + yij**2 + zij**2)**3.5),0,
      -0.25*(b**3*xij*(xij**2 - 4*yij**2 + zij**2))/(eta*PI*(xij**2 + yij**2 + zij**2)**3.5)],
     [(-3*b**3*yij*(-4*xij**2 + yij**2 + zij**2))/(4.*eta*PI*(xij**2 + yij**2 + zij**2)**3.5),
      -0.25*(b**3*xij*(4*xij**2 - 11*yij**2 - zij**2))/(eta*PI*(xij**2 + yij**2 + zij**2)**3.5),
      (5*b**3*xij*yij*zij)/(2.*eta*PI*(xij**2 + yij**2 + zij**2)**3.5),
      -0.25*(b**3*yij*(11*xij**2 - 4*yij**2 + zij**2))/(eta*PI*(xij**2 + yij**2 + zij**2)**3.5),
      (-5*b**3*(xij**2 - yij**2)*zij)/(4.*eta*PI*(xij**2 + yij**2 + zij**2)**3.5),
      (3*b**3*xij*(xij**2 - 4*yij**2 + zij**2))/(4.*eta*PI*(xij**2 + yij**2 + zij**2)**3.5),
      (-5*b**3*xij*yij*zij)/(2.*eta*PI*(xij**2 + yij**2 + zij**2)**3.5)]])


##
## matrix elements connecting higher slip modes with force/traction: G^HL
##

def G3a1s(xij,yij,zij, b,eta):
    return np.array([[0.,(-0.238732414637843*b**2*xij*zij)/(eta*(xij**2 + yij**2 + zij**2)**2.5),
      (0.238732414637843*b**2*xij*yij)/(eta*(xij**2 + yij**2 + zij**2)**2.5)],
     [(0.1193662073189215*b**2*xij*zij)/(eta*(xij**2 + yij**2 + zij**2)**2.5),
      (-0.1193662073189215*b**2*yij*zij)/(eta*(xij**2 + yij**2 + zij**2)**2.5),
      (-0.1193662073189215*b**2*(xij**2 - 1.*yij**2))/(eta*(xij**2 + yij**2 + zij**2)**2.5)],
     [(-0.1193662073189215*b**2*xij*yij)/(eta*(xij**2 + yij**2 + zij**2)**2.5),
      (0.1193662073189215*b**2*(xij**2 - 1.*zij**2))/(eta*(xij**2 + yij**2 + zij**2)**2.5),
      (0.1193662073189215*b**2*yij*zij)/(eta*(xij**2 + yij**2 + zij**2)**2.5)],
     [(0.238732414637843*b**2*yij*zij)/(eta*(xij**2 + yij**2 + zij**2)**2.5),0.,
      (-0.238732414637843*b**2*xij*yij)/(eta*(xij**2 + yij**2 + zij**2)**2.5)],
     [(-0.1193662073189215*b**2*(yij**2 - 1.*zij**2))/(eta*(xij**2 + yij**2 + zij**2)**2.5),
      (0.1193662073189215*b**2*xij*yij)/(eta*(xij**2 + yij**2 + zij**2)**2.5),
      (-0.1193662073189215*b**2*xij*zij)/(eta*(xij**2 + yij**2 + zij**2)**2.5)]])


def G3a2a(xij,yij,zij, b,eta):
    return np.array([[(-0.238732414637843*b**3*(1.*xij**3 - 1.5*xij*yij**2 - 1.5*xij*zij**2))/(eta*(xij**2 + yij**2 + zij**2)**3.5),
      (-0.477464829275686*b**3*(1.*xij**2*yij - 0.25*yij**3 - 0.25*yij*zij**2))/(eta*(xij**2 + yij**2 + zij**2)**3.5),
      (-0.477464829275686*b**3*(1.*xij**2*zij - 0.25*yij**2*zij - 0.25*zij**3))/(eta*(xij**2 + yij**2 + zij**2)**3.5)],
     [(-0.477464829275686*b**3*(1.*xij**2*yij - 0.25*yij**3 - 0.25*yij*zij**2))/(eta*(xij**2 + yij**2 + zij**2)**3.5),
      (0.1193662073189215*b**3*xij*(xij**2 - 4.*yij**2 + zij**2))/(eta*(xij**2 + yij**2 + zij**2)**3.5),
      (-0.5968310365946076*b**3*xij*yij*zij)/(eta*(xij**2 + yij**2 + zij**2)**3.5)],
     [(-0.477464829275686*b**3*(1.*xij**2*zij - 0.25*yij**2*zij - 0.25*zij**3))/(eta*(xij**2 + yij**2 + zij**2)**3.5),
      (-0.5968310365946076*b**3*xij*yij*zij)/(eta*(xij**2 + yij**2 + zij**2)**3.5),
      (0.1193662073189215*b**3*xij*(xij**2 + yij**2 - 4.*zij**2))/(eta*(xij**2 + yij**2 + zij**2)**3.5)],
     [(0.1193662073189215*b**3*xij*(xij**2 - 4.*yij**2 + zij**2))/(eta*(xij**2 + yij**2 + zij**2)**3.5),
      (0.3580986219567645*b**3*(1.*xij**2*yij - 0.6666666666666666*yij**3 + 1.*yij*zij**2))/(eta*(xij**2 + yij**2 + zij**2)**3.5),
      (0.1193662073189215*b**3*zij*(xij**2 - 4.*yij**2 + zij**2))/(eta*(xij**2 + yij**2 + zij**2)**3.5)],
     [(-0.5968310365946076*b**3*xij*yij*zij)/(eta*(xij**2 + yij**2 + zij**2)**3.5),
      (0.1193662073189215*b**3*zij*(xij**2 - 4.*yij**2 + zij**2))/(eta*(xij**2 + yij**2 + zij**2)**3.5),
      (0.1193662073189215*b**3*yij*(xij**2 + yij**2 - 4.*zij**2))/(eta*(xij**2 + yij**2 + zij**2)**3.5)]])


def G3s1s(xij,yij,zij, b,eta):
    return np.array([[(-50*b**4*(8*xij**4 - 24*xij**2*(yij**2 + zij**2) + 3*(yij**2 + zij**2)**2) + 
         21*b**2*(8*xij**6 - 8*xij**4*(yij**2 + zij**2) - 15*xij**2*(yij**2 + zij**2)**2 + (yij**2 + zij**2)**3))/
       (280.*eta*PI*(xij**2 + yij**2 + zij**2)**4.5),-0.0035714285714285713*
       (b**2*xij*yij*(250*b**2*(4*xij**2 - 3*(yij**2 + zij**2)) + 21*(-12*xij**4 + xij**2*(yij**2 + zij**2) + 13*(yij**2 + zij**2)**2)))/
        (eta*PI*(xij**2 + yij**2 + zij**2)**4.5),-0.0035714285714285713*
       (b**2*xij*zij*(250*b**2*(4*xij**2 - 3*(yij**2 + zij**2)) + 21*(-12*xij**4 + xij**2*(yij**2 + zij**2) + 13*(yij**2 + zij**2)**2)))/
        (eta*PI*(xij**2 + yij**2 + zij**2)**4.5)],[(b**2*xij*yij*
         (-250*b**2*(4*xij**2 - 3*(yij**2 + zij**2)) + 7*(52*xij**4 + 29*xij**2*(yij**2 + zij**2) - 23*(yij**2 + zij**2)**2)))/
       (280.*eta*PI*(xij**2 + yij**2 + zij**2)**4.5),(50*b**4*(4*xij**4 + 4*yij**4 + 3*yij**2*zij**2 - zij**4 + 3*xij**2*(-9*yij**2 + zij**2)) - 
         7*b**2*(4*xij**6 + (12*yij**2 - zij**2)*(yij**2 + zij**2)**2 + xij**4*(-55*yij**2 + 7*zij**2) + 
            xij**2*(-47*yij**4 - 45*yij**2*zij**2 + 2*zij**4)))/(280.*eta*PI*(xij**2 + yij**2 + zij**2)**4.5),
      (b**2*yij*zij*(434*xij**4 - 250*b**2*(6*xij**2 - yij**2 - zij**2) + 343*xij**2*(yij**2 + zij**2) - 91*(yij**2 + zij**2)**2))/
       (280.*eta*PI*(xij**2 + yij**2 + zij**2)**4.5)],[(b**2*xij*zij*
         (-250*b**2*(4*xij**2 - 3*(yij**2 + zij**2)) + 7*(52*xij**4 + 29*xij**2*(yij**2 + zij**2) - 23*(yij**2 + zij**2)**2)))/
       (280.*eta*PI*(xij**2 + yij**2 + zij**2)**4.5),(b**2*yij*zij*
         (434*xij**4 - 250*b**2*(6*xij**2 - yij**2 - zij**2) + 343*xij**2*(yij**2 + zij**2) - 91*(yij**2 + zij**2)**2))/
       (280.*eta*PI*(xij**2 + yij**2 + zij**2)**4.5),(50*b**4*(4*xij**4 - yij**4 + 3*yij**2*zij**2 + 4*zij**4 + 3*xij**2*(yij**2 - 9*zij**2)) - 
         7*b**2*(4*xij**6 + xij**4*(7*yij**2 - 55*zij**2) - (yij**2 - 12*zij**2)*(yij**2 + zij**2)**2 + 
            xij**2*(2*yij**4 - 45*yij**2*zij**2 - 47*zij**4)))/(280.*eta*PI*(xij**2 + yij**2 + zij**2)**4.5)],
     [(50*b**4*(4*xij**4 + 4*yij**4 + 3*yij**2*zij**2 - zij**4 + 3*xij**2*(-9*yij**2 + zij**2)) - 
         7*b**2*(12*xij**6 + (4*yij**2 - zij**2)*(yij**2 + zij**2)**2 + xij**4*(-47*yij**2 + 23*zij**2) - 
            5*xij**2*(11*yij**4 + 9*yij**2*zij**2 - 2*zij**4)))/(280.*eta*PI*(xij**2 + yij**2 + zij**2)**4.5),
      (b**2*xij*yij*(250*b**2*(3*xij**2 - 4*yij**2 + 3*zij**2) - 
           7*(23*xij**4 - 52*yij**4 - 29*yij**2*zij**2 + 23*zij**4 + xij**2*(-29*yij**2 + 46*zij**2))))/(280.*eta*PI*(xij**2 + yij**2 + zij**2)**4.5),
      (b**2*xij*zij*(250*b**2*(xij**2 - 6*yij**2 + zij**2) - 
           7*(13*xij**4 - 62*yij**4 - 49*yij**2*zij**2 + 13*zij**4 + xij**2*(-49*yij**2 + 26*zij**2))))/(280.*eta*PI*(xij**2 + yij**2 + zij**2)**4.5)]
      ,[(b**2*yij*zij*(98*xij**4 - 50*b**2*(6*xij**2 - yij**2 - zij**2) + 91*xij**2*(yij**2 + zij**2) - 7*(yij**2 + zij**2)**2))/
       (56.*eta*PI*(xij**2 + yij**2 + zij**2)**4.5),(b**2*xij*zij*
         (50*b**2*(xij**2 - 6*yij**2 + zij**2) - 7*(xij**4 - 14*yij**4 - 13*yij**2*zij**2 + zij**4 + xij**2*(-13*yij**2 + 2*zij**2))))/
       (56.*eta*PI*(xij**2 + yij**2 + zij**2)**4.5),(b**2*xij*yij*
         (50*b**2*(xij**2 + yij**2 - 6*zij**2) - 7*(xij**4 + yij**4 - 13*yij**2*zij**2 - 14*zij**4 + xij**2*(2*yij**2 - 13*zij**2))))/
       (56.*eta*PI*(xij**2 + yij**2 + zij**2)**4.5)],[(b**2*xij*yij*
         (250*b**2*(3*xij**2 - 4*yij**2 + 3*zij**2) - 21*(13*xij**4 - 12*yij**4 + yij**2*zij**2 + 13*zij**4 + xij**2*(yij**2 + 26*zij**2))))/
       (280.*eta*PI*(xij**2 + yij**2 + zij**2)**4.5),(-50*b**4*(3*xij**4 + 8*yij**4 - 24*yij**2*zij**2 + 3*zij**4 + 6*xij**2*(-4*yij**2 + zij**2)) + 
         21*b**2*(xij**6 + 8*yij**6 - 8*yij**4*zij**2 - 15*yij**2*zij**4 + zij**6 + 3*xij**4*(-5*yij**2 + zij**2) + 
            xij**2*(-8*yij**4 - 30*yij**2*zij**2 + 3*zij**4)))/(280.*eta*PI*(xij**2 + yij**2 + zij**2)**4.5),
      (b**2*yij*zij*(250*b**2*(3*xij**2 - 4*yij**2 + 3*zij**2) - 
           21*(13*xij**4 - 12*yij**4 + yij**2*zij**2 + 13*zij**4 + xij**2*(yij**2 + 26*zij**2))))/(280.*eta*PI*(xij**2 + yij**2 + zij**2)**4.5)],
     [(b**2*xij*zij*(250*b**2*(xij**2 - 6*yij**2 + zij**2) - 
           7*(13*xij**4 - 62*yij**4 - 49*yij**2*zij**2 + 13*zij**4 + xij**2*(-49*yij**2 + 26*zij**2))))/(280.*eta*PI*(xij**2 + yij**2 + zij**2)**4.5),
      (b**2*yij*zij*(250*b**2*(3*xij**2 - 4*yij**2 + 3*zij**2) - 
           7*(23*xij**4 - 52*yij**4 - 29*yij**2*zij**2 + 23*zij**4 + xij**2*(-29*yij**2 + 46*zij**2))))/(280.*eta*PI*(xij**2 + yij**2 + zij**2)**4.5),
      (-50*b**4*(xij**4 - 4*yij**4 + 27*yij**2*zij**2 - 4*zij**4 - 3*xij**2*(yij**2 + zij**2)) + 
         7*b**2*(xij**6 - 4*yij**6 + 55*yij**4*zij**2 + 47*yij**2*zij**4 - 12*zij**6 - 2*xij**4*(yij**2 + 5*zij**2) + 
            xij**2*(-7*yij**4 + 45*yij**2*zij**2 - 23*zij**4)))/(280.*eta*PI*(xij**2 + yij**2 + zij**2)**4.5)]])


def G3s2a(xij,yij,zij, b,eta):
    return np.array([[0.,(-0.1193662073189215*b**3*zij*(-4.*xij**2 + 1.*yij**2 + 1.*zij**2))/(eta*(xij**2 + yij**2 + zij**2)**3.5),
      (0.1193662073189215*b**3*yij*(-4.*xij**2 + 1.*yij**2 + 1.*zij**2))/(eta*(xij**2 + yij**2 + zij**2)**3.5)],
     [(0.039788735772973836*b**3*zij*(-4.*xij**2 + yij**2 + zij**2))/(eta*(xij**2 + yij**2 + zij**2)**3.5),
      (0.3978873577297384*b**3*xij*yij*zij)/(eta*(xij**2 + yij**2 + zij**2)**3.5),
      (0.15915494309189535*b**3*(1.*xij**3 - 2.75*xij*yij**2 - 0.25*xij*zij**2))/(eta*(xij**2 + yij**2 + zij**2)**3.5)],
     [(-0.039788735772973836*b**3*yij*(-4.*xij**2 + yij**2 + zij**2))/(eta*(xij**2 + yij**2 + zij**2)**3.5),
      (-0.15915494309189535*b**3*(1.*xij**3 - 0.25*xij*yij**2 - 2.75*xij*zij**2))/(eta*(xij**2 + yij**2 + zij**2)**3.5),
      (-0.3978873577297384*b**3*xij*yij*zij)/(eta*(xij**2 + yij**2 + zij**2)**3.5)],
     [(-0.3978873577297384*b**3*xij*yij*zij)/(eta*(xij**2 + yij**2 + zij**2)**3.5),
      (-0.039788735772973836*b**3*zij*(xij**2 - 4.*yij**2 + zij**2))/(eta*(xij**2 + yij**2 + zij**2)**3.5),
      (0.039788735772973836*b**3*yij*(11.*xij**2 - 4.*yij**2 + zij**2))/(eta*(xij**2 + yij**2 + zij**2)**3.5)],
     [(0.1989436788648692*b**3*xij*(yij**2 - 1.*zij**2))/(eta*(xij**2 + yij**2 + zij**2)**3.5),
      (-0.1989436788648692*b**3*yij*(xij**2 - 1.*zij**2))/(eta*(xij**2 + yij**2 + zij**2)**3.5),
      (0.1989436788648692*b**3*(xij**2 - 1.*yij**2)*zij)/(eta*(xij**2 + yij**2 + zij**2)**3.5)],
     [(0.1193662073189215*b**3*zij*(1.*xij**2 - 4.*yij**2 + 1.*zij**2))/(eta*(xij**2 + yij**2 + zij**2)**3.5),0.,
      (-0.1193662073189215*b**3*xij*(xij**2 - 4.*yij**2 + zij**2))/(eta*(xij**2 + yij**2 + zij**2)**3.5)],
     [(-0.039788735772973836*b**3*(1.*xij**2*yij - 4.*yij**3 + 11.*yij*zij**2))/(eta*(xij**2 + yij**2 + zij**2)**3.5),
      (0.039788735772973836*b**3*xij*(xij**2 - 4.*yij**2 + zij**2))/(eta*(xij**2 + yij**2 + zij**2)**3.5),
      (0.3978873577297384*b**3*xij*yij*zij)/(eta*(xij**2 + yij**2 + zij**2)**3.5)]])




##
## matrix elements connecting higher slip modes with induced traction modes: G^HH (apart from (2s,2s), which can be found in FTS section above)
##

def G2s3t(xij,yij,zij, b,eta):
    return np.array([[(-3*b**3*(2*xij**3 - 3*xij*(yij**2 + zij**2)))/(20.*eta*PI*(xij**2 + yij**2 + zij**2)**3.5),
      (3*b**3*yij*(-4*xij**2 + yij**2 + zij**2))/(20.*eta*PI*(xij**2 + yij**2 + zij**2)**3.5),
      (3*b**3*zij*(-4*xij**2 + yij**2 + zij**2))/(20.*eta*PI*(xij**2 + yij**2 + zij**2)**3.5)],
     [(3*b**3*yij*(-4*xij**2 + yij**2 + zij**2))/(20.*eta*PI*(xij**2 + yij**2 + zij**2)**3.5),
      (3*b**3*xij*(xij**2 - 4*yij**2 + zij**2))/(20.*eta*PI*(xij**2 + yij**2 + zij**2)**3.5),
      (-3*b**3*xij*yij*zij)/(4.*eta*PI*(xij**2 + yij**2 + zij**2)**3.5)],
     [(3*b**3*zij*(-4*xij**2 + yij**2 + zij**2))/(20.*eta*PI*(xij**2 + yij**2 + zij**2)**3.5),
      (-3*b**3*xij*yij*zij)/(4.*eta*PI*(xij**2 + yij**2 + zij**2)**3.5),
      (3*b**3*xij*(xij**2 + yij**2 - 4*zij**2))/(20.*eta*PI*(xij**2 + yij**2 + zij**2)**3.5)],
     [(3*b**3*xij*(xij**2 - 4*yij**2 + zij**2))/(20.*eta*PI*(xij**2 + yij**2 + zij**2)**3.5),
      (3*b**3*yij*(3*xij**2 - 2*yij**2 + 3*zij**2))/(20.*eta*PI*(xij**2 + yij**2 + zij**2)**3.5),
      (3*b**3*zij*(xij**2 - 4*yij**2 + zij**2))/(20.*eta*PI*(xij**2 + yij**2 + zij**2)**3.5)],
     [(-3*b**3*xij*yij*zij)/(4.*eta*PI*(xij**2 + yij**2 + zij**2)**3.5),
      (3*b**3*zij*(xij**2 - 4*yij**2 + zij**2))/(20.*eta*PI*(xij**2 + yij**2 + zij**2)**3.5),
      (3*b**3*yij*(xij**2 + yij**2 - 4*zij**2))/(20.*eta*PI*(xij**2 + yij**2 + zij**2)**3.5)]])



def G2s3a(xij,yij,zij, b,eta):
    return np.array([[0,(b**3*zij*(-4*xij**2 + yij**2 + zij**2))/(6.*eta*PI*(xij**2 + yij**2 + zij**2)**3.5),
      -0.16666666666666666*(b**3*yij*(-4*xij**2 + yij**2 + zij**2))/(eta*PI*(xij**2 + yij**2 + zij**2)**3.5),
      (-5*b**3*xij*yij*zij)/(2.*eta*PI*(xij**2 + yij**2 + zij**2)**3.5),
      (b**3*xij*(xij**2 + 6*yij**2 - 9*zij**2))/(6.*eta*PI*(xij**2 + yij**2 + zij**2)**3.5)],
     [-0.25*(b**3*zij*(-4*xij**2 + yij**2 + zij**2))/(eta*PI*(xij**2 + yij**2 + zij**2)**3.5),
      (5*b**3*xij*yij*zij)/(12.*eta*PI*(xij**2 + yij**2 + zij**2)**3.5),
      (-5*b**3*xij*(xij**2 - yij**2 - 2*zij**2))/(12.*eta*PI*(xij**2 + yij**2 + zij**2)**3.5),
      (b**3*zij*(xij**2 - 4*yij**2 + zij**2))/(4.*eta*PI*(xij**2 + yij**2 + zij**2)**3.5),
      (5*b**3*yij*(-xij**2 + yij**2 - 2*zij**2))/(12.*eta*PI*(xij**2 + yij**2 + zij**2)**3.5)],
     [(b**3*yij*(-4*xij**2 + yij**2 + zij**2))/(4.*eta*PI*(xij**2 + yij**2 + zij**2)**3.5),
      (5*b**3*xij*(xij**2 - 2*yij**2 - zij**2))/(12.*eta*PI*(xij**2 + yij**2 + zij**2)**3.5),
      (-5*b**3*xij*yij*zij)/(12.*eta*PI*(xij**2 + yij**2 + zij**2)**3.5),
      (5*b**3*yij*(xij**2 - zij**2))/(4.*eta*PI*(xij**2 + yij**2 + zij**2)**3.5),
      (b**3*zij*(13*xij**2 + 8*yij**2 - 7*zij**2))/(12.*eta*PI*(xij**2 + yij**2 + zij**2)**3.5)],
     [(5*b**3*xij*yij*zij)/(2.*eta*PI*(xij**2 + yij**2 + zij**2)**3.5),
      -0.3333333333333333*(b**3*zij*(xij**2 - 4*yij**2 + zij**2))/(eta*PI*(xij**2 + yij**2 + zij**2)**3.5),
      -0.16666666666666666*(b**3*yij*(6*xij**2 + yij**2 - 9*zij**2))/(eta*PI*(xij**2 + yij**2 + zij**2)**3.5),0,
      (b**3*xij*(xij**2 - 4*yij**2 + zij**2))/(6.*eta*PI*(xij**2 + yij**2 + zij**2)**3.5)],
     [(-5*b**3*xij*(yij**2 - zij**2))/(4.*eta*PI*(xij**2 + yij**2 + zij**2)**3.5),
      (b**3*yij*(8*xij**2 - 7*yij**2 + 13*zij**2))/(12.*eta*PI*(xij**2 + yij**2 + zij**2)**3.5),
      (b**3*zij*(-8*xij**2 - 13*yij**2 + 7*zij**2))/(12.*eta*PI*(xij**2 + yij**2 + zij**2)**3.5),
      -0.25*(b**3*xij*(xij**2 - 4*yij**2 + zij**2))/(eta*PI*(xij**2 + yij**2 + zij**2)**3.5),
      (5*b**3*xij*yij*zij)/(12.*eta*PI*(xij**2 + yij**2 + zij**2)**3.5)]])



def G3a2s(xij,yij,zij, b,eta):
    return np.array([[0.,(-0.477464829275686*b**3*(1.*xij**2*zij - 0.25*yij**2*zij - 0.25*zij**3))/(eta*(xij**2 + yij**2 + zij**2)**3.5),
      (0.477464829275686*b**3*(1.*xij**2*yij - 0.25*yij**3 - 0.25*yij*zij**2))/(eta*(xij**2 + yij**2 + zij**2)**3.5),
      (-1.1936620731892151*b**3*xij*yij*zij)/(eta*(xij**2 + yij**2 + zij**2)**3.5),
      (0.5968310365946076*b**3*xij*(yij**2 - 1.*zij**2))/(eta*(xij**2 + yij**2 + zij**2)**3.5)],
     [(-0.1193662073189215*b**3*zij*(-4.*xij**2 + 1.*yij**2 + 1.*zij**2))/(eta*(xij**2 + yij**2 + zij**2)**3.5),0.,
      (-0.238732414637843*b**3*(1.*xij**3 - 1.5*xij*yij**2 - 1.5*xij*zij**2))/(eta*(xij**2 + yij**2 + zij**2)**3.5),
      (0.1193662073189215*b**3*zij*(1.*xij**2 - 4.*yij**2 + 1.*zij**2))/(eta*(xij**2 + yij**2 + zij**2)**3.5),
      (-0.3580986219567645*b**3*(1.*xij**2*yij - 0.6666666666666666*yij**3 + 1.*yij*zij**2))/(eta*(xij**2 + yij**2 + zij**2)**3.5)],
     [(0.1193662073189215*b**3*yij*(-4.*xij**2 + 1.*yij**2 + 1.*zij**2))/(eta*(xij**2 + yij**2 + zij**2)**3.5),
      (0.238732414637843*b**3*(1.*xij**3 - 1.5*xij*yij**2 - 1.5*xij*zij**2))/(eta*(xij**2 + yij**2 + zij**2)**3.5),0.,
      (0.5968310365946076*b**3*yij*(xij**2 - 1.*zij**2))/(eta*(xij**2 + yij**2 + zij**2)**3.5),
      (0.3580986219567645*b**3*(1.*xij**2*zij + 1.*yij**2*zij - 0.6666666666666666*zij**3))/(eta*(xij**2 + yij**2 + zij**2)**3.5)],
     [(1.1936620731892151*b**3*xij*yij*zij)/(eta*(xij**2 + yij**2 + zij**2)**3.5),
      (-0.1193662073189215*b**3*zij*(xij**2 - 4.*yij**2 + zij**2))/(eta*(xij**2 + yij**2 + zij**2)**3.5),
      (-0.5968310365946076*b**3*yij*(xij**2 - 1.*zij**2))/(eta*(xij**2 + yij**2 + zij**2)**3.5),0.,
      (0.1193662073189215*b**3*xij*(xij**2 - 4.*yij**2 + zij**2))/(eta*(xij**2 + yij**2 + zij**2)**3.5)],
     [(-0.5968310365946076*b**3*xij*(yij**2 - 1.*zij**2))/(eta*(xij**2 + yij**2 + zij**2)**3.5),
      (0.3580986219567645*b**3*(1.*xij**2*yij - 0.6666666666666666*yij**3 + 1.*yij*zij**2))/(eta*(xij**2 + yij**2 + zij**2)**3.5),
      (-0.3580986219567645*b**3*(1.*xij**2*zij + 1.*yij**2*zij - 0.6666666666666666*zij**3))/(eta*(xij**2 + yij**2 + zij**2)**3.5),
      (-0.1193662073189215*b**3*xij*(xij**2 - 4.*yij**2 + zij**2))/(eta*(xij**2 + yij**2 + zij**2)**3.5),0.]])



def G3a3a(xij,yij,zij, b,eta):
    return np.array([[-0.5*(b**4*(8*xij**4 - 24*xij**2*(yij**2 + zij**2) + 3*(yij**2 + zij**2)**2))/(eta*PI*(xij**2 + yij**2 + zij**2)**4.5),
      (-5*b**4*xij*yij*(4*xij**2 - 3*(yij**2 + zij**2)))/(2.*eta*PI*(xij**2 + yij**2 + zij**2)**4.5),
      (-5*b**4*xij*zij*(4*xij**2 - 3*(yij**2 + zij**2)))/(2.*eta*PI*(xij**2 + yij**2 + zij**2)**4.5),
      (b**4*(4*xij**4 + 4*yij**4 + 3*yij**2*zij**2 - zij**4 + 3*xij**2*(-9*yij**2 + zij**2)))/(2.*eta*PI*(xij**2 + yij**2 + zij**2)**4.5),
      (5*b**4*yij*zij*(-6*xij**2 + yij**2 + zij**2))/(2.*eta*PI*(xij**2 + yij**2 + zij**2)**4.5)],
     [(-5*b**4*xij*yij*(4*xij**2 - 3*(yij**2 + zij**2)))/(2.*eta*PI*(xij**2 + yij**2 + zij**2)**4.5),
      (b**4*(4*xij**4 + 4*yij**4 + 3*yij**2*zij**2 - zij**4 + 3*xij**2*(-9*yij**2 + zij**2)))/(2.*eta*PI*(xij**2 + yij**2 + zij**2)**4.5),
      (5*b**4*yij*zij*(-6*xij**2 + yij**2 + zij**2))/(2.*eta*PI*(xij**2 + yij**2 + zij**2)**4.5),
      (5*b**4*xij*yij*(3*xij**2 - 4*yij**2 + 3*zij**2))/(2.*eta*PI*(xij**2 + yij**2 + zij**2)**4.5),
      (5*b**4*xij*zij*(xij**2 - 6*yij**2 + zij**2))/(2.*eta*PI*(xij**2 + yij**2 + zij**2)**4.5)],
     [(-5*b**4*xij*zij*(4*xij**2 - 3*(yij**2 + zij**2)))/(2.*eta*PI*(xij**2 + yij**2 + zij**2)**4.5),
      (5*b**4*yij*zij*(-6*xij**2 + yij**2 + zij**2))/(2.*eta*PI*(xij**2 + yij**2 + zij**2)**4.5),
      (b**4*(4*xij**4 - yij**4 + 3*yij**2*zij**2 + 4*zij**4 + 3*xij**2*(yij**2 - 9*zij**2)))/(2.*eta*PI*(xij**2 + yij**2 + zij**2)**4.5),
      (5*b**4*xij*zij*(xij**2 - 6*yij**2 + zij**2))/(2.*eta*PI*(xij**2 + yij**2 + zij**2)**4.5),
      (5*b**4*xij*yij*(xij**2 + yij**2 - 6*zij**2))/(2.*eta*PI*(xij**2 + yij**2 + zij**2)**4.5)],
     [(b**4*(4*xij**4 + 4*yij**4 + 3*yij**2*zij**2 - zij**4 + 3*xij**2*(-9*yij**2 + zij**2)))/(2.*eta*PI*(xij**2 + yij**2 + zij**2)**4.5),
      (5*b**4*xij*yij*(3*xij**2 - 4*yij**2 + 3*zij**2))/(2.*eta*PI*(xij**2 + yij**2 + zij**2)**4.5),
      (5*b**4*xij*zij*(xij**2 - 6*yij**2 + zij**2))/(2.*eta*PI*(xij**2 + yij**2 + zij**2)**4.5),
      -0.5*(b**4*(3*xij**4 + 8*yij**4 - 24*yij**2*zij**2 + 3*zij**4 + 6*xij**2*(-4*yij**2 + zij**2)))/(eta*PI*(xij**2 + yij**2 + zij**2)**4.5),
      (5*b**4*yij*zij*(3*xij**2 - 4*yij**2 + 3*zij**2))/(2.*eta*PI*(xij**2 + yij**2 + zij**2)**4.5)],
     [(5*b**4*yij*zij*(-6*xij**2 + yij**2 + zij**2))/(2.*eta*PI*(xij**2 + yij**2 + zij**2)**4.5),
      (5*b**4*xij*zij*(xij**2 - 6*yij**2 + zij**2))/(2.*eta*PI*(xij**2 + yij**2 + zij**2)**4.5),
      (5*b**4*xij*yij*(xij**2 + yij**2 - 6*zij**2))/(2.*eta*PI*(xij**2 + yij**2 + zij**2)**4.5),
      (5*b**4*yij*zij*(3*xij**2 - 4*yij**2 + 3*zij**2))/(2.*eta*PI*(xij**2 + yij**2 + zij**2)**4.5),
      -0.5*(b**4*(xij**4 - 4*yij**4 + 27*yij**2*zij**2 - 4*zij**4 - 3*xij**2*(yij**2 + zij**2)))/(eta*PI*(xij**2 + yij**2 + zij**2)**4.5)]])



def G2s3s(xij,yij,zij, b,eta):
    return np.array([[(1.6370222718023522*b**3*xij*(-0.35*xij**6 - 0.56875*yij**6 - 1.70625*yij**4*zij**2 - 1.70625*yij**2*zij**4 - 0.56875*zij**6 + 
           xij**4*(1.2833333333333332*yij**2 + 1.2833333333333332*zij**2) + 
           xij**2*(1.0645833333333332*yij**4 + 2.1291666666666664*yij**2*zij**2 + 1.0645833333333332*zij**4) + 
           b**2*(1.*xij**4 + 1.8749999999999998*yij**4 + 3.7499999999999996*yij**2*zij**2 + 1.8749999999999998*zij**4 + 
              xij**2*(-5.*yij**2 - 5.*zij**2))))/(eta*(xij**2 + yij**2 + zij**2)**5.5),
      (4.9110668154070565*b**3*yij*(-0.337037037037037*xij**6 - 0.03726851851851851*yij**6 - 0.11180555555555555*yij**4*zij**2 - 
           0.11180555555555555*yij**2*zij**4 - 0.03726851851851851*zij**6 + xij**4*(0.13935185185185184*yij**2 + 0.13935185185185184*zij**2) + 
           xij**2*(0.4391203703703704*yij**4 + 0.8782407407407408*yij**2*zij**2 + 0.4391203703703704*zij**4) + 
           b**2*(1.*xij**4 + 0.125*yij**4 + 0.25*yij**2*zij**2 + 0.125*zij**4 + xij**2*(-1.5*yij**2 - 1.5*zij**2))))/
       (eta*(xij**2 + yij**2 + zij**2)**5.5),(4.9110668154070565*b**3*zij*
         (-0.337037037037037*xij**6 - 0.03726851851851851*yij**6 - 0.11180555555555555*yij**4*zij**2 - 0.11180555555555555*yij**2*zij**4 - 
           0.03726851851851851*zij**6 + xij**4*(0.13935185185185184*yij**2 + 0.13935185185185184*zij**2) + 
           xij**2*(0.4391203703703704*yij**4 + 0.8782407407407408*yij**2*zij**2 + 0.4391203703703704*zij**4) + 
           b**2*(1.*xij**4 + 0.125*yij**4 + 0.25*yij**2*zij**2 + 0.125*zij**4 + xij**2*(-1.5*yij**2 - 1.5*zij**2))))/
       (eta*(xij**2 + yij**2 + zij**2)**5.5),(-0.8185111359011761*b**3*xij*
         (-0.35*xij**6 - 1.4194444444444443*yij**6 - 2.5569444444444445*yij**4*zij**2 - 0.8555555555555556*yij**2*zij**4 + 
           0.28194444444444444*zij**6 + xij**4*(2.9847222222222225*yij**2 - 0.4180555555555555*zij**2) + 
           xij**2*(1.9152777777777779*yij**4 + 2.1291666666666664*yij**2*zij**2 + 0.2138888888888889*zij**4) + 
           b**2*(1.*xij**4 + 4.5*yij**4 + 3.7499999999999996*yij**2*zij**2 - 0.75*zij**4 + xij**2*(-10.25*yij**2 + 0.25*zij**2))))/
       (eta*(xij**2 + yij**2 + zij**2)**5.5),(8.59436692696235*b**3*xij*yij*zij*
         (-0.32407407407407407*xij**4 + 0.16203703703703703*yij**4 + 0.32407407407407407*yij**2*zij**2 + 0.16203703703703703*zij**4 + 
           b**2*(1.*xij**2 - 0.5*yij**2 - 0.5*zij**2) + xij**2*(-0.16203703703703703*yij**2 - 0.16203703703703703*zij**2)))/
       (eta*(xij**2 + yij**2 + zij**2)**5.5),(-3.6833001115552926*b**3*yij*
         (-0.337037037037037*xij**6 - 0.07777777777777778*yij**6 - 0.07129629629629629*yij**4*zij**2 + 0.09074074074074073*yij**2*zij**4 + 
           0.08425925925925926*zij**6 + xij**4*(0.38240740740740736*yij**2 - 0.5898148148148148*zij**2) + 
           xij**2*(0.6416666666666666*yij**4 + 0.4731481481481481*yij**2*zij**2 - 0.1685185185185185*zij**4) + 
           b**2*(1.*xij**4 + 0.2222222222222222*yij**4 + 0.05555555555555555*yij**2*zij**2 - 0.16666666666666666*zij**4 + 
              xij**2*(-2.2777777777777777*yij**2 + 0.8333333333333334*zij**2))))/(eta*(xij**2 + yij**2 + zij**2)**5.5),
      (-1.2277667038517641*b**3*zij*(-0.337037037037037*xij**6 - 0.40185185185185185*yij**6 - 0.7194444444444444*yij**4*zij**2 - 
           0.2333333333333333*yij**2*zij**4 + 0.08425925925925926*zij**6 + xij**4*(2.3268518518518517*yij**2 - 0.5898148148148148*zij**2) + 
           xij**2*(2.262037037037037*yij**4 + 2.0935185185185183*yij**2*zij**2 - 0.1685185185185185*zij**4) + 
           b**2*(1.*xij**4 + 1.*yij**4 + 0.8333333333333334*yij**2*zij**2 - 0.16666666666666666*zij**4 + 
              xij**2*(-8.5*yij**2 + 0.8333333333333334*zij**2))))/(eta*(xij**2 + yij**2 + zij**2)**5.5)],
     [(4.9110668154070565*b**3*yij*(-0.33055555555555555*xij**6 - 0.03888888888888889*yij**6 - 0.11666666666666668*yij**4*zij**2 - 
           0.11666666666666668*yij**2*zij**4 - 0.03888888888888889*zij**6 + xij**4*(0.15069444444444444*yij**2 + 0.15069444444444444*zij**2) + 
           xij**2*(0.4423611111111111*yij**4 + 0.8847222222222222*yij**2*zij**2 + 0.4423611111111111*zij**4) + 
           b**2*(1.*xij**4 + 0.125*yij**4 + 0.25*yij**2*zij**2 + 0.125*zij**4 + xij**2*(-1.5*yij**2 - 1.5*zij**2))))/
       (eta*(xij**2 + yij**2 + zij**2)**5.5),(-0.8185111359011761*b**3*xij*
         (-0.3111111111111111*xij**6 - 1.4291666666666667*yij**6 - 2.683333333333333*yij**4*zij**2 - 1.0791666666666666*yij**2*zij**4 + 
           0.175*zij**6 + xij**4*(3.0527777777777776*yij**2 - 0.4472222222222222*zij**2) + 
           xij**2*(1.9347222222222222*yij**4 + 1.973611111111111*yij**2*zij**2 + 0.03888888888888889*zij**4) + 
           b**2*(1.*xij**4 + 4.5*yij**4 + 3.75*yij**2*zij**2 - 0.7499999999999999*zij**4 + xij**2*(-10.249999999999998*yij**2 + 0.25*zij**2))))/
       (eta*(xij**2 + yij**2 + zij**2)**5.5),(8.594366926962348*b**3*xij*yij*zij*
         (-0.3333333333333333*xij**4 + 0.15277777777777776*yij**4 + 0.3055555555555555*yij**2*zij**2 + 0.15277777777777776*zij**4 + 
           b**2*(1.*xij**2 - 0.5*yij**2 - 0.5*zij**2) + xij**2*(-0.18055555555555555*yij**2 - 0.18055555555555555*zij**2)))/
       (eta*(xij**2 + yij**2 + zij**2)**5.5),(-3.6833001115552926*b**3*yij*
         (-0.3175925925925926*xij**6 - 0.0691358024691358*yij**6 - 0.09938271604938272*yij**4*zij**2 + 0.008641975308641974*yij**2*zij**4 + 
           0.03888888888888889*zij**6 + xij**4*(0.42993827160493825*yij**2 - 0.5962962962962963*zij**2) + 
           xij**2*(0.678395061728395*yij**4 + 0.4385802469135802*yij**2*zij**2 - 0.2398148148148148*zij**4) + 
           b**2*(1.*xij**4 + 0.2222222222222222*yij**4 + 0.05555555555555555*yij**2*zij**2 - 0.16666666666666663*zij**4 + 
              xij**2*(-2.2777777777777777*yij**2 + 0.8333333333333334*zij**2))))/(eta*(xij**2 + yij**2 + zij**2)**5.5),
      (-1.227766703851764*b**3*zij*(-0.2916666666666667*xij**6 - 0.2916666666666667*yij**6 - 0.5509259259259259*yij**4*zij**2 - 
           0.22685185185185186*yij**2*zij**4 + 0.032407407407407406*zij**6 + xij**4*(2.5277777777777777*yij**2 - 0.5509259259259259*zij**2) + 
           xij**2*(2.5277777777777777*yij**4 + 2.300925925925926*yij**2*zij**2 - 0.22685185185185186*zij**4) + 
           b**2*(1.*xij**4 + 1.*yij**4 + 0.8333333333333334*yij**2*zij**2 - 0.16666666666666666*zij**4 + 
              xij**2*(-8.5*yij**2 + 0.8333333333333334*zij**2))))/(eta*(xij**2 + yij**2 + zij**2)**5.5),
      (0.6138833519258821*b**3*xij*(-0.3111111111111111*xij**6 - 2.6444444444444444*yij**6 + 1.2055555555555555*yij**4*zij**2 + 
           3.5388888888888888*yij**2*zij**4 - 0.3111111111111111*zij**6 + xij**4*(3.5388888888888888*yij**2 - 0.9333333333333333*zij**2) + 
           xij**2*(1.2055555555555555*yij**4 + 7.0777777777777775*yij**2*zij**2 - 0.9333333333333333*zij**4) + 
           b**2*(1.*xij**4 + 8.*yij**4 - 11.999999999999998*yij**2*zij**2 + 1.*zij**4 + xij**2*(-11.999999999999998*yij**2 + 2.*zij**2))))/
       (eta*(xij**2 + yij**2 + zij**2)**5.5),(-4.297183463481175*b**3*xij*yij*zij*
         (-0.3055555555555555*xij**4 + 0.6666666666666666*yij**4 + 0.3611111111111111*yij**2*zij**2 - 0.3055555555555555*zij**4 + 
           xij**2*(0.3611111111111111*yij**2 - 0.611111111111111*zij**2) + b**2*(1.*xij**2 - 2.*yij**2 + 1.*zij**2)))/
       (eta*(xij**2 + yij**2 + zij**2)**5.5)],[(4.9110668154070565*b**3*zij*
         (-0.33055555555555555*xij**6 - 0.03888888888888889*yij**6 - 0.11666666666666668*yij**4*zij**2 - 0.11666666666666668*yij**2*zij**4 - 
           0.03888888888888889*zij**6 + xij**4*(0.15069444444444444*yij**2 + 0.15069444444444444*zij**2) + 
           xij**2*(0.4423611111111111*yij**4 + 0.8847222222222222*yij**2*zij**2 + 0.4423611111111111*zij**4) + 
           b**2*(1.*xij**4 + 0.125*yij**4 + 0.25*yij**2*zij**2 + 0.125*zij**4 + xij**2*(-1.5*yij**2 - 1.5*zij**2))))/
       (eta*(xij**2 + yij**2 + zij**2)**5.5),(8.594366926962348*b**3*xij*yij*zij*
         (-0.3333333333333333*xij**4 + 0.15277777777777776*yij**4 + 0.3055555555555555*yij**2*zij**2 + 0.15277777777777776*zij**4 + 
           b**2*(1.*xij**2 - 0.5*yij**2 - 0.5*zij**2) + xij**2*(-0.18055555555555555*yij**2 - 0.18055555555555555*zij**2)))/
       (eta*(xij**2 + yij**2 + zij**2)**5.5),(-0.8185111359011761*b**3*xij*
         (-0.3111111111111111*xij**6 + 0.175*yij**6 - 1.0791666666666666*yij**4*zij**2 - 2.683333333333333*yij**2*zij**4 - 
           1.4291666666666667*zij**6 + xij**4*(-0.4472222222222222*yij**2 + 3.0527777777777776*zij**2) + 
           xij**2*(0.03888888888888889*yij**4 + 1.973611111111111*yij**2*zij**2 + 1.9347222222222222*zij**4) + 
           b**2*(1.*xij**4 - 0.7499999999999999*yij**4 + 3.75*yij**2*zij**2 + 4.5*zij**4 + xij**2*(0.25*yij**2 - 10.249999999999998*zij**2))))/
       (eta*(xij**2 + yij**2 + zij**2)**5.5),(-1.2277667038517641*b**3*zij*
         (-0.36944444444444446*xij**6 - 0.2722222222222222*yij**6 - 0.4925925925925926*yij**4*zij**2 - 0.1685185185185185*yij**2*zij**4 + 
           0.05185185185185185*zij**6 + xij**4*(2.391666666666667*yij**2 - 0.6870370370370371*zij**2) + 
           xij**2*(2.488888888888889*yij**4 + 2.223148148148148*yij**2*zij**2 - 0.2657407407407408*zij**4) + 
           b**2*(1.*xij**4 + 1.*yij**4 + 0.8333333333333334*yij**2*zij**2 - 0.16666666666666666*zij**4 + 
              xij**2*(-8.5*yij**2 + 0.8333333333333334*zij**2))))/(eta*(xij**2 + yij**2 + zij**2)**5.5),
      (-1.227766703851764*b**3*yij*(-0.2916666666666667*xij**6 + 0.032407407407407406*yij**6 - 0.22685185185185186*yij**4*zij**2 - 
           0.5509259259259259*yij**2*zij**4 - 0.2916666666666667*zij**6 + xij**4*(-0.5509259259259259*yij**2 + 2.5277777777777777*zij**2) + 
           xij**2*(-0.22685185185185186*yij**4 + 2.300925925925926*yij**2*zij**2 + 2.5277777777777777*zij**4) + 
           b**2*(1.*xij**4 - 0.16666666666666666*yij**4 + 0.8333333333333334*yij**2*zij**2 + 1.*zij**4 + 
              xij**2*(0.8333333333333334*yij**2 - 8.5*zij**2))))/(eta*(xij**2 + yij**2 + zij**2)**5.5),
      (-4.297183463481174*b**3*xij*yij*zij*(-0.3611111111111111*xij**4 + 0.6111111111111112*yij**4 + 0.25*yij**2*zij**2 - 0.3611111111111111*zij**4 + 
           xij**2*(0.25*yij**2 - 0.7222222222222222*zij**2) + b**2*(1.*xij**2 - 2.*yij**2 + 1.*zij**2)))/(eta*(xij**2 + yij**2 + zij**2)**5.5),
      (0.20462778397529402*b**3*xij*(-0.3111111111111111*xij**6 + 1.6333333333333333*yij**6 - 14.933333333333334*yij**4*zij**2 - 
           14.35*yij**2*zij**4 + 2.216666666666667*zij**6 + xij**4*(1.011111111111111*yij**2 + 1.5944444444444443*zij**2) + 
           xij**2*(2.9555555555555553*yij**4 - 13.338888888888889*yij**2*zij**2 + 4.122222222222222*zij**4) + 
           b**2*(1.*xij**4 - 5.999999999999999*yij**4 + 50.99999999999999*yij**2*zij**2 - 5.999999999999999*zij**4 + xij**2*(-5.*yij**2 - 5.*zij**2)))
         )/(eta*(xij**2 + yij**2 + zij**2)**5.5)],[(-0.8185111359011761*b**3*xij*
         (-0.35*xij**6 - 1.5166666666666666*yij**6 - 2.6541666666666663*yij**4*zij**2 - 0.7583333333333333*yij**2*zij**4 + 
           0.37916666666666665*zij**6 + xij**4*(2.8874999999999997*yij**2 - 0.3208333333333333*zij**2) + 
           xij**2*(1.7208333333333334*yij**4 + 2.1291666666666664*yij**2*zij**2 + 0.4083333333333333*zij**4) + 
           b**2*(1.*xij**4 + 4.5*yij**4 + 3.7499999999999996*yij**2*zij**2 - 0.75*zij**4 + xij**2*(-10.25*yij**2 + 0.25*zij**2))))/
       (eta*(xij**2 + yij**2 + zij**2)**5.5),(-3.683300111555292*b**3*yij*
         (-0.3154320987654321*xij**6 - 0.07777777777777778*yij**6 - 0.09290123456790124*yij**4*zij**2 + 0.04753086419753086*yij**2*zij**4 + 
           0.06265432098765432*zij**6 + xij**4*(0.4256172839506173*yij**2 - 0.5682098765432099*zij**2) + 
           xij**2*(0.6632716049382716*yij**4 + 0.47314814814814815*yij**2*zij**2 - 0.19012345679012344*zij**4) + 
           b**2*(1.*xij**4 + 0.2222222222222222*yij**4 + 0.05555555555555555*yij**2*zij**2 - 0.16666666666666666*zij**4 + 
              xij**2*(-2.2777777777777777*yij**2 + 0.8333333333333333*zij**2))))/(eta*(xij**2 + yij**2 + zij**2)**5.5),
      (-1.2277667038517641*b**3*zij*(-0.40185185185185185*xij**6 - 0.3370370370370371*yij**6 - 0.5898148148148149*yij**4*zij**2 - 
           0.16851851851851854*yij**2*zij**4 + 0.08425925925925927*zij**6 + xij**4*(2.2620370370370373*yij**2 - 0.7194444444444444*zij**2) + 
           xij**2*(2.3268518518518517*yij**4 + 2.093518518518519*yij**2*zij**2 - 0.23333333333333336*zij**4) + 
           b**2*(1.*xij**4 + 1.*yij**4 + 0.8333333333333334*yij**2*zij**2 - 0.16666666666666666*zij**4 + 
              xij**2*(-8.5*yij**2 + 0.8333333333333334*zij**2))))/(eta*(xij**2 + yij**2 + zij**2)**5.5),
      (0.6138833519258821*b**3*xij*(-0.29814814814814816*xij**6 - 2.696296296296296*yij**6 + 1.114814814814815*yij**4*zij**2 + 
           3.5129629629629626*yij**2*zij**4 - 0.29814814814814816*zij**6 + xij**4*(3.5129629629629626*yij**2 - 0.8944444444444444*zij**2) + 
           xij**2*(1.114814814814815*yij**4 + 7.025925925925925*yij**2*zij**2 - 0.8944444444444444*zij**4) + 
           b**2*(1.*xij**4 + 8.*yij**4 - 12.*yij**2*zij**2 + 1.*zij**4 + xij**2*(-12.*yij**2 + 2.*zij**2))))/(eta*(xij**2 + yij**2 + zij**2)**5.5),
      (-4.297183463481175*b**3*xij*yij*zij*(-0.32407407407407407*xij**4 + 0.6481481481481481*yij**4 + 0.32407407407407407*yij**2*zij**2 - 
           0.32407407407407407*zij**4 + xij**2*(0.32407407407407407*yij**2 - 0.6481481481481481*zij**2) + b**2*(1.*xij**2 - 2.*yij**2 + 1.*zij**2)))/
       (eta*(xij**2 + yij**2 + zij**2)**5.5),(3.06941675962941*b**3*yij*
         (-0.30333333333333334*xij**6 - 0.18666666666666668*yij**6 + 0.6844444444444445*yij**4*zij**2 + 0.5677777777777778*yij**2*zij**4 - 
           0.30333333333333334*zij**6 + xij**4*(0.5677777777777778*yij**2 - 0.91*zij**2) + 
           xij**2*(0.6844444444444445*yij**4 + 1.1355555555555557*yij**2*zij**2 - 0.91*zij**4) + 
           b**2*(1.*xij**4 + 0.5333333333333333*yij**4 - 2.666666666666667*yij**2*zij**2 + 1.*zij**4 + xij**2*(-2.666666666666667*yij**2 + 2.*zij**2))
           ))/(eta*(xij**2 + yij**2 + zij**2)**5.5),(0.6138833519258821*b**3*zij*
         (-0.29814814814814816*xij**6 - 2.696296296296296*yij**6 + 1.114814814814815*yij**4*zij**2 + 3.5129629629629626*yij**2*zij**4 - 
           0.29814814814814816*zij**6 + xij**4*(3.5129629629629626*yij**2 - 0.8944444444444444*zij**2) + 
           xij**2*(1.114814814814815*yij**4 + 7.025925925925925*yij**2*zij**2 - 0.8944444444444444*zij**4) + 
           b**2*(1.*xij**4 + 8.*yij**4 - 12.*yij**2*zij**2 + 1.*zij**4 + xij**2*(-12.*yij**2 + 2.*zij**2))))/(eta*(xij**2 + yij**2 + zij**2)**5.5)],
     [(8.59436692696235*b**3*xij*yij*zij*(-0.3055555555555555*xij**4 + 0.18055555555555555*yij**4 + 0.3611111111111111*yij**2*zij**2 + 
           0.18055555555555555*zij**4 + b**2*(1.*xij**2 - 0.5*yij**2 - 0.5*zij**2) + xij**2*(-0.125*yij**2 - 0.125*zij**2)))/
       (eta*(xij**2 + yij**2 + zij**2)**5.5),(-1.2277667038517641*b**3*zij*
         (-0.2722222222222222*xij**6 - 0.36944444444444446*yij**6 - 0.687037037037037*yij**4*zij**2 - 0.2657407407407407*yij**2*zij**4 + 
           0.05185185185185185*zij**6 + xij**4*(2.4888888888888885*yij**2 - 0.4925925925925926*zij**2) + 
           xij**2*(2.3916666666666666*yij**4 + 2.223148148148148*yij**2*zij**2 - 0.1685185185185185*zij**4) + 
           b**2*(1.*xij**4 + 1.*yij**4 + 0.8333333333333333*yij**2*zij**2 - 0.16666666666666666*zij**4 + 
              xij**2*(-8.5*yij**2 + 0.8333333333333333*zij**2))))/(eta*(xij**2 + yij**2 + zij**2)**5.5),
      (-1.2277667038517641*b**3*yij*(-0.2722222222222222*xij**6 + 0.05185185185185185*yij**6 - 0.2657407407407407*yij**4*zij**2 - 
           0.687037037037037*yij**2*zij**4 - 0.36944444444444446*zij**6 + xij**4*(-0.4925925925925926*yij**2 + 2.4888888888888885*zij**2) + 
           xij**2*(-0.1685185185185185*yij**4 + 2.223148148148148*yij**2*zij**2 + 2.3916666666666666*zij**4) + 
           b**2*(1.*xij**4 - 0.16666666666666666*yij**4 + 0.8333333333333333*yij**2*zij**2 + 1.*zij**4 + 
              xij**2*(0.8333333333333333*yij**2 - 8.5*zij**2))))/(eta*(xij**2 + yij**2 + zij**2)**5.5),
      (-4.297183463481175*b**3*xij*yij*zij*(-0.3055555555555555*xij**4 + 0.6666666666666666*yij**4 + 0.3611111111111111*yij**2*zij**2 - 
           0.3055555555555555*zij**4 + xij**2*(0.3611111111111111*yij**2 - 0.611111111111111*zij**2) + b**2*(1.*xij**2 - 2.*yij**2 + 1.*zij**2)))/
       (eta*(xij**2 + yij**2 + zij**2)**5.5),(0.20462778397529402*b**3*xij*
         (-0.19444444444444442*xij**6 + 1.7499999999999998*yij**6 - 15.166666666666666*yij**4*zij**2 - 15.166666666666666*yij**2*zij**4 + 
           1.7499999999999998*zij**6 + xij**4*(1.361111111111111*yij**2 + 1.361111111111111*zij**2) + 
           xij**2*(3.3055555555555554*yij**4 - 13.805555555555555*yij**2*zij**2 + 3.3055555555555554*zij**4) + 
           b**2*(1.*xij**4 - 6.*yij**4 + 50.99999999999999*yij**2*zij**2 - 6.*zij**4 + xij**2*(-5.*yij**2 - 5.*zij**2))))/
       (eta*(xij**2 + yij**2 + zij**2)**5.5),(0.6138833519258821*b**3*zij*
         (-0.3111111111111111*xij**6 - 2.6444444444444444*yij**6 + 1.2055555555555555*yij**4*zij**2 + 3.5388888888888888*yij**2*zij**4 - 
           0.3111111111111111*zij**6 + xij**4*(3.5388888888888888*yij**2 - 0.9333333333333333*zij**2) + 
           xij**2*(1.2055555555555555*yij**4 + 7.0777777777777775*yij**2*zij**2 - 0.9333333333333333*zij**4) + 
           b**2*(1.*xij**4 + 8.*yij**4 - 11.999999999999998*yij**2*zij**2 + 1.*zij**4 + xij**2*(-11.999999999999998*yij**2 + 2.*zij**2))))/
       (eta*(xij**2 + yij**2 + zij**2)**5.5),(0.6138833519258821*b**3*yij*
         (-0.23333333333333334*xij**6 + 0.4148148148148148*yij**6 - 4.07037037037037*yij**4*zij**2 - 2.5796296296296295*yij**2*zij**4 + 
           1.9055555555555554*zij**6 + xij**4*(-0.05185185185185185*yij**2 + 1.438888888888889*zij**2) + 
           xij**2*(0.5962962962962962*yij**4 - 2.6314814814814818*yij**2*zij**2 + 3.577777777777778*zij**4) + 
           b**2*(1.*xij**4 - 1.3333333333333335*yij**4 + 13.666666666666666*yij**2*zij**2 - 6.*zij**4 + 
              xij**2*(-0.33333333333333337*yij**2 - 5.*zij**2))))/(eta*(xij**2 + yij**2 + zij**2)**5.5)]])



def G3a3s(xij,yij,zij, b,eta):
    return np.array([[0.,(-1.5915494309189535*b**4*xij*zij*(1.*xij**2 - 0.75*yij**2 - 0.75*zij**2))/(eta*(xij**2 + yij**2 + zij**2)**4.5),
      (1.5915494309189535*b**4*xij*yij*(1.*xij**2 - 0.75*yij**2 - 0.75*zij**2))/(eta*(xij**2 + yij**2 + zij**2)**4.5),
      (0.7957747154594768*b**4*yij*zij*(-6.*xij**2 + 1.*yij**2 + 1.*zij**2))/(eta*(xij**2 + yij**2 + zij**2)**4.5),
      (0.3978873577297384*b**4*(-1.*yij**4 + zij**4 + xij**2*(6.*yij**2 - 6.*zij**2)))/(eta*(xij**2 + yij**2 + zij**2)**4.5),
      (1.1936620731892151*b**4*xij*zij*(xij**2 - 6.*yij**2 + zij**2))/(eta*(xij**2 + yij**2 + zij**2)**4.5),
      (-0.3978873577297384*b**4*xij*yij*(xij**2 - 6.*yij**2 + 15.*zij**2))/(eta*(xij**2 + yij**2 + zij**2)**4.5)],
     [(-1.7904931097838226*b**4*zij*(-1.3333333333333333*xij**3 + 1.*xij*yij**2 + 1.*xij*zij**2))/(eta*(xij**2 + yij**2 + zij**2)**4.5),
      (-0.1989436788648692*b**4*yij*zij*(-6.*xij**2 + yij**2 + zij**2))/(eta*(xij**2 + yij**2 + zij**2)**4.5),
      (-0.1989436788648692*b**4*(4.*xij**4 + yij**4 + 3.*yij**2*zij**2 + 2.*zij**4 + xij**2*(-9.*yij**2 - 15.*zij**2)))/
       (eta*(xij**2 + yij**2 + zij**2)**4.5),(0.1989436788648692*b**4*xij*zij*(xij**2 - 6.*yij**2 + zij**2))/(eta*(xij**2 + yij**2 + zij**2)**4.5),
      (-1.3926057520540842*b**4*(xij**3*yij - 1.*xij*yij**3))/(eta*(xij**2 + yij**2 + zij**2)**4.5),
      (1.7904931097838226*b**4*zij*(1.*xij**2*yij - 1.3333333333333333*yij**3 + 1.*yij*zij**2))/(eta*(xij**2 + yij**2 + zij**2)**4.5),
      (0.1989436788648692*b**4*(xij**4 + 4.*yij**4 - 15.*yij**2*zij**2 + 2.*zij**4 + xij**2*(-9.*yij**2 + 3.*zij**2)))/
       (eta*(xij**2 + yij**2 + zij**2)**4.5)],[(1.7904931097838226*b**4*yij*(-1.3333333333333333*xij**3 + 1.*xij*yij**2 + 1.*xij*zij**2))/
       (eta*(xij**2 + yij**2 + zij**2)**4.5),(0.1989436788648692*b**4*
         (4.*xij**4 + 2.*yij**4 + 3.*yij**2*zij**2 + zij**4 + xij**2*(-15.*yij**2 - 9.*zij**2)))/(eta*(xij**2 + yij**2 + zij**2)**4.5),
      (0.1989436788648692*b**4*yij*zij*(-6.*xij**2 + yij**2 + zij**2))/(eta*(xij**2 + yij**2 + zij**2)**4.5),
      (2.5862678252432993*b**4*xij*yij*(1.*xij**2 - 0.6153846153846154*yij**2 - 1.1538461538461537*zij**2))/(eta*(xij**2 + yij**2 + zij**2)**4.5),
      (1.3926057520540842*b**4*(xij**3*zij - 1.*xij*zij**3))/(eta*(xij**2 + yij**2 + zij**2)**4.5),
      (-0.5968310365946076*b**4*(xij**4 - 6.*xij**2*yij**2 + 6.*yij**2*zij**2 - 1.*zij**4))/(eta*(xij**2 + yij**2 + zij**2)**4.5),
      (1.5915494309189535*b**4*yij*zij*(1.875*xij**2 + 1.*yij**2 - 1.625*zij**2))/(eta*(xij**2 + yij**2 + zij**2)**4.5)],
     [(-1.1936620731892151*b**4*yij*zij*(-6.*xij**2 + 1.*yij**2 + 1.*zij**2))/(eta*(xij**2 + yij**2 + zij**2)**4.5),
      (-0.7957747154594768*b**4*xij*zij*(xij**2 - 6.*yij**2 + zij**2))/(eta*(xij**2 + yij**2 + zij**2)**4.5),
      (-2.3873241463784303*b**4*xij*yij*(1.*xij**2 - 0.16666666666666666*yij**2 - 2.5*zij**2))/(eta*(xij**2 + yij**2 + zij**2)**4.5),
      (1.5915494309189535*b**4*yij*zij*(-0.75*xij**2 + 1.*yij**2 - 0.75*zij**2))/(eta*(xij**2 + yij**2 + zij**2)**4.5),
      (0.3978873577297384*b**4*(xij**4 - 6.*xij**2*yij**2 + 6.*yij**2*zij**2 - 1.*zij**4))/(eta*(xij**2 + yij**2 + zij**2)**4.5),0.,
      (1.1936620731892151*b**4*xij*yij*(1.*xij**2 - 1.3333333333333333*yij**2 + 1.*zij**2))/(eta*(xij**2 + yij**2 + zij**2)**4.5)],
     [(-0.5968310365946076*b**4*(-1.*yij**4 + zij**4 + xij**2*(6.*yij**2 - 6.*zij**2)))/(eta*(xij**2 + yij**2 + zij**2)**4.5),
      (1.5915494309189535*b**4*xij*yij*(1.*xij**2 - 1.625*yij**2 + 1.875*zij**2))/(eta*(xij**2 + yij**2 + zij**2)**4.5),
      (-1.5915494309189535*b**4*xij*zij*(1.*xij**2 + 1.875*yij**2 - 1.625*zij**2))/(eta*(xij**2 + yij**2 + zij**2)**4.5),
      (-0.1989436788648692*b**4*(2.*xij**4 + 4.*yij**4 - 9.*yij**2*zij**2 + zij**4 + xij**2*(-15.*yij**2 + 3.*zij**2)))/
       (eta*(xij**2 + yij**2 + zij**2)**4.5),(-1.3926057520540842*b**4*(yij**3*zij - 1.*yij*zij**3))/(eta*(xij**2 + yij**2 + zij**2)**4.5),
      (-1.7904931097838226*b**4*xij*(1.*xij**2*yij - 1.3333333333333333*yij**3 + 1.*yij*zij**2))/(eta*(xij**2 + yij**2 + zij**2)**4.5),
      (-0.1989436788648692*b**4*xij*zij*(xij**2 - 6.*yij**2 + zij**2))/(eta*(xij**2 + yij**2 + zij**2)**4.5)]])



def G3s2s(xij,yij,zij, b,eta):
    return np.array([[0.,0.,0.,0.,0.],[0.,0.,0.,0.,0.],[0.,0.,0.,0.,0.],[0.,0.,0.,0.,0.],[0.,0.,0.,0.,0.],[0.,0.,0.,0.,0.],[0.,0.,0.,0.,0.]])



def G3s3a(xij,yij,zij, b,eta):
    return np.array([[0,(5*b**4*xij*zij*(4*xij**2 - 3*(yij**2 + zij**2)))/(6.*eta*PI*(xij**2 + yij**2 + zij**2)**4.5),
      (-5*b**4*xij*yij*(4*xij**2 - 3*(yij**2 + zij**2)))/(6.*eta*PI*(xij**2 + yij**2 + zij**2)**4.5),
      (-5*b**4*yij*zij*(-6*xij**2 + yij**2 + zij**2))/(2.*eta*PI*(xij**2 + yij**2 + zij**2)**4.5),
      -0.16666666666666666*(b**4*(4*xij**4 - 6*yij**4 + 3*yij**2*zij**2 + 9*zij**4 + xij**2*(33*yij**2 - 57*zij**2)))/
        (eta*PI*(xij**2 + yij**2 + zij**2)**4.5)],[(-5*b**4*xij*zij*(4*xij**2 - 3*(yij**2 + zij**2)))/(6.*eta*PI*(xij**2 + yij**2 + zij**2)**4.5),
      0,(b**4*(8*xij**4 - 24*xij**2*(yij**2 + zij**2) + 3*(yij**2 + zij**2)**2))/(6.*eta*PI*(xij**2 + yij**2 + zij**2)**4.5),
      (-5*b**4*xij*zij*(xij**2 - 6*yij**2 + zij**2))/(3.*eta*PI*(xij**2 + yij**2 + zij**2)**4.5),
      (5*b**4*xij*yij*(2*xij**2 - 5*yij**2 + 9*zij**2))/(6.*eta*PI*(xij**2 + yij**2 + zij**2)**4.5)],
     [(5*b**4*xij*yij*(4*xij**2 - 3*(yij**2 + zij**2)))/(6.*eta*PI*(xij**2 + yij**2 + zij**2)**4.5),
      -0.16666666666666666*(b**4*(8*xij**4 - 24*xij**2*(yij**2 + zij**2) + 3*(yij**2 + zij**2)**2))/(eta*PI*(xij**2 + yij**2 + zij**2)**4.5),0,
      (-5*b**4*xij*yij*(6*xij**2 - yij**2 - 15*zij**2))/(6.*eta*PI*(xij**2 + yij**2 + zij**2)**4.5),
      (-5*b**4*xij*zij*(3*xij**2 + 3*yij**2 - 4*zij**2))/(3.*eta*PI*(xij**2 + yij**2 + zij**2)**4.5)],
     [(5*b**4*yij*zij*(-6*xij**2 + yij**2 + zij**2))/(3.*eta*PI*(xij**2 + yij**2 + zij**2)**4.5),
      (5*b**4*xij*zij*(xij**2 - 6*yij**2 + zij**2))/(6.*eta*PI*(xij**2 + yij**2 + zij**2)**4.5),
      (5*b**4*xij*yij*(5*xij**2 - 2*yij**2 - 9*zij**2))/(6.*eta*PI*(xij**2 + yij**2 + zij**2)**4.5),
      (5*b**4*yij*zij*(-3*xij**2 + 4*yij**2 - 3*zij**2))/(6.*eta*PI*(xij**2 + yij**2 + zij**2)**4.5),
      -0.16666666666666666*(b**4*(3*xij**4 + 8*yij**4 - 24*yij**2*zij**2 + 3*zij**4 + 6*xij**2*(-4*yij**2 + zij**2)))/
        (eta*PI*(xij**2 + yij**2 + zij**2)**4.5)],[(5*b**4*(-yij**4 + zij**4 + 6*xij**2*(yij**2 - zij**2)))/
       (6.*eta*PI*(xij**2 + yij**2 + zij**2)**4.5),(-5*b**4*xij*yij*(3*xij**2 - 4*yij**2 + 3*zij**2))/(6.*eta*PI*(xij**2 + yij**2 + zij**2)**4.5),
      (5*b**4*xij*zij*(3*xij**2 + 3*yij**2 - 4*zij**2))/(6.*eta*PI*(xij**2 + yij**2 + zij**2)**4.5),
      (5*b**4*(xij**2 - zij**2)*(xij**2 - 6*yij**2 + zij**2))/(6.*eta*PI*(xij**2 + yij**2 + zij**2)**4.5),
      (-5*b**4*yij*zij*(3*xij**2 + 3*yij**2 - 4*zij**2))/(6.*eta*PI*(xij**2 + yij**2 + zij**2)**4.5)],
     [(5*b**4*xij*zij*(xij**2 - 6*yij**2 + zij**2))/(2.*eta*PI*(xij**2 + yij**2 + zij**2)**4.5),
      (5*b**4*yij*zij*(3*xij**2 - 4*yij**2 + 3*zij**2))/(3.*eta*PI*(xij**2 + yij**2 + zij**2)**4.5),
      (b**4*(-6*xij**4 + 4*yij**4 - 57*yij**2*zij**2 + 9*zij**4 + 3*xij**2*(11*yij**2 + zij**2)))/(6.*eta*PI*(xij**2 + yij**2 + zij**2)**4.5),0,
      (-5*b**4*xij*yij*(3*xij**2 - 4*yij**2 + 3*zij**2))/(6.*eta*PI*(xij**2 + yij**2 + zij**2)**4.5)],
     [(-5*b**4*xij*yij*(xij**2 - 6*yij**2 + 15*zij**2))/(6.*eta*PI*(xij**2 + yij**2 + zij**2)**4.5),
      (b**4*(2*xij**4 + 12*yij**4 - 51*yij**2*zij**2 + 7*zij**4 + xij**2*(-21*yij**2 + 9*zij**2)))/(6.*eta*PI*(xij**2 + yij**2 + zij**2)**4.5),
      (5*b**4*yij*zij*(3*xij**2 + 3*yij**2 - 4*zij**2))/(3.*eta*PI*(xij**2 + yij**2 + zij**2)**4.5),
      (5*b**4*xij*yij*(3*xij**2 - 4*yij**2 + 3*zij**2))/(6.*eta*PI*(xij**2 + yij**2 + zij**2)**4.5),0]])



def G3s3t(xij,yij,zij, b,eta):
    return np.array([[(3*b**4*(8*xij**4 - 24*xij**2*(yij**2 + zij**2) + 3*(yij**2 + zij**2)**2))/(20.*eta*PI*(xij**2 + yij**2 + zij**2)**4.5),
      (3*b**4*xij*yij*(4*xij**2 - 3*(yij**2 + zij**2)))/(4.*eta*PI*(xij**2 + yij**2 + zij**2)**4.5),
      (3*b**4*xij*zij*(4*xij**2 - 3*(yij**2 + zij**2)))/(4.*eta*PI*(xij**2 + yij**2 + zij**2)**4.5)],
     [(3*b**4*xij*yij*(4*xij**2 - 3*(yij**2 + zij**2)))/(4.*eta*PI*(xij**2 + yij**2 + zij**2)**4.5),
      (-3*b**4*(4*xij**4 + 4*yij**4 + 3*yij**2*zij**2 - zij**4 + 3*xij**2*(-9*yij**2 + zij**2)))/(20.*eta*PI*(xij**2 + yij**2 + zij**2)**4.5),
      (-3*b**4*yij*zij*(-6*xij**2 + yij**2 + zij**2))/(4.*eta*PI*(xij**2 + yij**2 + zij**2)**4.5)],
     [(3*b**4*xij*zij*(4*xij**2 - 3*(yij**2 + zij**2)))/(4.*eta*PI*(xij**2 + yij**2 + zij**2)**4.5),
      (-3*b**4*yij*zij*(-6*xij**2 + yij**2 + zij**2))/(4.*eta*PI*(xij**2 + yij**2 + zij**2)**4.5),
      (-3*b**4*(4*xij**4 - yij**4 + 3*yij**2*zij**2 + 4*zij**4 + 3*xij**2*(yij**2 - 9*zij**2)))/(20.*eta*PI*(xij**2 + yij**2 + zij**2)**4.5)],
     [(-3*b**4*(4*xij**4 + 4*yij**4 + 3*yij**2*zij**2 - zij**4 + 3*xij**2*(-9*yij**2 + zij**2)))/(20.*eta*PI*(xij**2 + yij**2 + zij**2)**4.5),
      (-3*b**4*xij*yij*(3*xij**2 - 4*yij**2 + 3*zij**2))/(4.*eta*PI*(xij**2 + yij**2 + zij**2)**4.5),
      (-3*b**4*xij*zij*(xij**2 - 6*yij**2 + zij**2))/(4.*eta*PI*(xij**2 + yij**2 + zij**2)**4.5)],
     [(-3*b**4*yij*zij*(-6*xij**2 + yij**2 + zij**2))/(4.*eta*PI*(xij**2 + yij**2 + zij**2)**4.5),
      (-3*b**4*xij*zij*(xij**2 - 6*yij**2 + zij**2))/(4.*eta*PI*(xij**2 + yij**2 + zij**2)**4.5),
      (-3*b**4*xij*yij*(xij**2 + yij**2 - 6*zij**2))/(4.*eta*PI*(xij**2 + yij**2 + zij**2)**4.5)],
     [(-3*b**4*xij*yij*(3*xij**2 - 4*yij**2 + 3*zij**2))/(4.*eta*PI*(xij**2 + yij**2 + zij**2)**4.5),
      (3*b**4*(3*xij**4 + 8*yij**4 - 24*yij**2*zij**2 + 3*zij**4 + 6*xij**2*(-4*yij**2 + zij**2)))/(20.*eta*PI*(xij**2 + yij**2 + zij**2)**4.5),
      (3*b**4*yij*zij*(-3*xij**2 + 4*yij**2 - 3*zij**2))/(4.*eta*PI*(xij**2 + yij**2 + zij**2)**4.5)],
     [(-3*b**4*xij*zij*(xij**2 - 6*yij**2 + zij**2))/(4.*eta*PI*(xij**2 + yij**2 + zij**2)**4.5),
      (3*b**4*yij*zij*(-3*xij**2 + 4*yij**2 - 3*zij**2))/(4.*eta*PI*(xij**2 + yij**2 + zij**2)**4.5),
      (3*b**4*(xij**4 - 4*yij**4 + 27*yij**2*zij**2 - 4*zij**4 - 3*xij**2*(yij**2 + zij**2)))/(20.*eta*PI*(xij**2 + yij**2 + zij**2)**4.5)]])



def G3s3s(xij,yij,zij, b,eta):
    return np.array([[(-450*b**6*(16*xij**6 - 120*xij**4*(yij**2 + zij**2) + 90*xij**2*(yij**2 + zij**2)**2 - 5*(yij**2 + zij**2)**3) + 
         21*b**4*(112*xij**8 - 712*xij**6*(yij**2 + zij**2) - 218*xij**4*(yij**2 + zij**2)**2 + 573*xij**2*(yij**2 + zij**2)**3 - 
            33*(yij**2 + zij**2)**4))/(280.*eta*PI*(xij**2 + yij**2 + zij**2)**6.5),
      -0.125*(b**4*xij*yij*(-232*xij**6 + 340*xij**4*(yij**2 + zij**2) + 431*xij**2*(yij**2 + zij**2)**2 - 141*(yij**2 + zij**2)**3 + 
            90*b**2*(8*xij**4 - 20*xij**2*(yij**2 + zij**2) + 5*(yij**2 + zij**2)**2)))/(eta*PI*(xij**2 + yij**2 + zij**2)**6.5),
      -0.125*(b**4*xij*zij*(-232*xij**6 + 340*xij**4*(yij**2 + zij**2) + 431*xij**2*(yij**2 + zij**2)**2 - 141*(yij**2 + zij**2)**3 + 
            90*b**2*(8*xij**4 - 20*xij**2*(yij**2 + zij**2) + 5*(yij**2 + zij**2)**2)))/(eta*PI*(xij**2 + yij**2 + zij**2)**6.5),
      (450*b**6*(8*xij**6 - (6*yij**2 - zij**2)*(yij**2 + zij**2)**2 - 4*xij**4*(29*yij**2 + zij**2) + 
            xij**2*(101*yij**4 + 90*yij**2*zij**2 - 11*zij**4)) + 
         7*b**4*(-168*xij**8 + 24*xij**6*(92*yij**2 - 3*zij**2) - xij**2*(1937*yij**2 - 218*zij**2)*(yij**2 + zij**2)**2 + 
            (122*yij**2 - 23*zij**2)*(yij**2 + zij**2)**3 + xij**4*(317*yij**4 + 654*yij**2*zij**2 + 337*zij**4)))/
       (280.*eta*PI*(xij**2 + yij**2 + zij**2)**6.5),-0.125*
       (b**4*yij*zij*(-456*xij**6 + 4*xij**4*(yij**2 + zij**2) + 431*xij**2*(yij**2 + zij**2)**2 - 29*(yij**2 + zij**2)**3 + 
            90*b**2*(16*xij**4 - 16*xij**2*(yij**2 + zij**2) + (yij**2 + zij**2)**2)))/(eta*PI*(xij**2 + yij**2 + zij**2)**6.5),
      (3*b**4*xij*yij*(-58*xij**6 + xij**4*(141*yij**2 - 83*zij**2) - (58*yij**2 - 33*zij**2)*(yij**2 + zij**2)**2 + 
           xij**2*(141*yij**4 + 149*yij**2*zij**2 + 8*zij**4) + 90*b**2*(2*xij**4 + 2*yij**4 + yij**2*zij**2 - zij**4 + xij**2*(-7*yij**2 + zij**2))))
        /(8.*eta*PI*(xij**2 + yij**2 + zij**2)**6.5),(b**4*xij*zij*
         (-58*xij**6 + xij**4*(589*yij**2 - 83*zij**2) - 3*(80*yij**2 - 11*zij**2)*(yij**2 + zij**2)**2 + 
           xij**2*(407*yij**4 + 415*yij**2*zij**2 + 8*zij**4) + 
           90*b**2*(2*xij**4 + 8*yij**4 + 7*yij**2*zij**2 - zij**4 + xij**2*(-23*yij**2 + zij**2))))/(8.*eta*PI*(xij**2 + yij**2 + zij**2)**6.5)],
     [-0.125*(b**4*xij*yij*(-232*xij**6 + 340*xij**4*(yij**2 + zij**2) + 431*xij**2*(yij**2 + zij**2)**2 - 141*(yij**2 + zij**2)**3 + 
            90*b**2*(8*xij**4 - 20*xij**2*(yij**2 + zij**2) + 5*(yij**2 + zij**2)**2)))/(eta*PI*(xij**2 + yij**2 + zij**2)**6.5),
      (1350*b**6*(8*xij**6 - (6*yij**2 - zij**2)*(yij**2 + zij**2)**2 - 4*xij**4*(29*yij**2 + zij**2) + 
            xij**2*(101*yij**4 + 90*yij**2*zij**2 - 11*zij**4)) - 
         7*b**4*(488*xij**8 + xij**2*(5807*yij**2 - 578*zij**2)*(yij**2 + zij**2)**2 - (362*yij**2 - 53*zij**2)*(yij**2 + zij**2)**3 + 
            xij**6*(-6668*yij**2 + 292*zij**2) - xij**4*(987*yij**4 + 1814*yij**2*zij**2 + 827*zij**4)))/(840.*eta*PI*(xij**2 + yij**2 + zij**2)**6.5)
       ,(b**4*yij*zij*(1392*xij**6 + 32*xij**4*(yij**2 + zij**2) - 1277*xij**2*(yij**2 + zij**2)**2 + 83*(yij**2 + zij**2)**3 - 
           270*b**2*(16*xij**4 - 16*xij**2*(yij**2 + zij**2) + (yij**2 + zij**2)**2)))/(24.*eta*PI*(xij**2 + yij**2 + zij**2)**6.5),
      (b**4*xij*yij*(-514*xij**6 + xij**4*(1293*yij**2 - 779*zij**2) - (514*yij**2 - 249*zij**2)*(yij**2 + zij**2)**2 + 
           xij**2*(1293*yij**4 + 1277*yij**2*zij**2 - 16*zij**4) + 
           810*b**2*(2*xij**4 + 2*yij**4 + yij**2*zij**2 - zij**4 + xij**2*(-7*yij**2 + zij**2))))/(24.*eta*PI*(xij**2 + yij**2 + zij**2)**6.5),
      (b**4*xij*zij*(-166*xij**6 + xij**4*(1819*yij**2 - 253*zij**2) - (684*yij**2 - 79*zij**2)*(yij**2 + zij**2)**2 + 
           xij**2*(1301*yij**4 + 1293*yij**2*zij**2 - 8*zij**4) + 
           270*b**2*(2*xij**4 + 8*yij**4 + 7*yij**2*zij**2 - zij**4 + xij**2*(-23*yij**2 + zij**2))))/(24.*eta*PI*(xij**2 + yij**2 + zij**2)**6.5),
      (-450*b**6*(6*xij**6 - 8*yij**6 + 4*yij**4*zij**2 + 11*yij**2*zij**4 - zij**6 + xij**4*(-101*yij**2 + 11*zij**2) + 
            2*xij**2*(58*yij**4 - 45*yij**2*zij**2 + 2*zij**4)) + 
         7*b**4*(122*xij**8 + xij**6*(-1937*yij**2 + 343*zij**2) - (yij**2 + zij**2)**2*(168*yij**4 - 264*yij**2*zij**2 + 23*zij**4) + 
            xij**4*(317*yij**4 - 3656*yij**2*zij**2 + 297*zij**4) + xij**2*(2208*yij**6 + 654*yij**4*zij**2 - 1501*yij**2*zij**4 + 53*zij**6)))/
       (280.*eta*PI*(xij**2 + yij**2 + zij**2)**6.5),(b**4*yij*zij*
         (270*b**2*(8*xij**4 + 2*yij**4 + yij**2*zij**2 - zij**4 + xij**2*(-23*yij**2 + 7*zij**2)) - 
           7*(96*xij**6 + xij**4*(-187*yij**2 + 179*zij**2) + xij**2*(-257*yij**4 - 187*yij**2*zij**2 + 70*zij**4) + 
              13*(2*yij**6 + 3*yij**4*zij**2 - zij**6))))/(24.*eta*PI*(xij**2 + yij**2 + zij**2)**6.5)],
     [-0.125*(b**4*xij*zij*(-232*xij**6 + 340*xij**4*(yij**2 + zij**2) + 431*xij**2*(yij**2 + zij**2)**2 - 141*(yij**2 + zij**2)**3 + 
            90*b**2*(8*xij**4 - 20*xij**2*(yij**2 + zij**2) + 5*(yij**2 + zij**2)**2)))/(eta*PI*(xij**2 + yij**2 + zij**2)**6.5),
      (b**4*yij*zij*(1392*xij**6 + 32*xij**4*(yij**2 + zij**2) - 1277*xij**2*(yij**2 + zij**2)**2 + 83*(yij**2 + zij**2)**3 - 
           270*b**2*(16*xij**4 - 16*xij**2*(yij**2 + zij**2) + (yij**2 + zij**2)**2)))/(24.*eta*PI*(xij**2 + yij**2 + zij**2)**6.5),
      (1350*b**6*(8*xij**6 + (yij**2 - 6*zij**2)*(yij**2 + zij**2)**2 - 4*xij**4*(yij**2 + 29*zij**2) + 
            xij**2*(-11*yij**4 + 90*yij**2*zij**2 + 101*zij**4)) - 
         7*b**4*(488*xij**8 + 4*xij**6*(73*yij**2 - 1667*zij**2) - xij**2*(578*yij**2 - 5807*zij**2)*(yij**2 + zij**2)**2 + 
            (53*yij**2 - 362*zij**2)*(yij**2 + zij**2)**3 - xij**4*(827*yij**4 + 1814*yij**2*zij**2 + 987*zij**4)))/
       (840.*eta*PI*(xij**2 + yij**2 + zij**2)**6.5),(b**4*xij*zij*
         (270*b**2*(2*xij**4 + 8*yij**4 + 7*yij**2*zij**2 - zij**4 + xij**2*(-23*yij**2 + zij**2)) - 
           7*(26*xij**6 - 187*xij**2*yij**2*(yij**2 + zij**2) + (96*yij**2 - 13*zij**2)*(yij**2 + zij**2)**2 + xij**4*(-257*yij**2 + 39*zij**2))))/
       (24.*eta*PI*(xij**2 + yij**2 + zij**2)**6.5),(b**4*xij*yij*
         (-166*xij**6 + (79*yij**2 - 684*zij**2)*(yij**2 + zij**2)**2 + xij**4*(-253*yij**2 + 1819*zij**2) + 
           xij**2*(-8*yij**4 + 1293*yij**2*zij**2 + 1301*zij**4) + 
           270*b**2*(2*xij**4 - yij**4 + 7*yij**2*zij**2 + 8*zij**4 + xij**2*(yij**2 - 23*zij**2))))/(24.*eta*PI*(xij**2 + yij**2 + zij**2)**6.5),
      (b**4*yij*zij*(-240*xij**6 + xij**4*(407*yij**2 - 447*zij**2) - (58*yij**2 - 33*zij**2)*(yij**2 + zij**2)**2 + 
           xij**2*(589*yij**4 + 415*yij**2*zij**2 - 174*zij**4) + 
           90*b**2*(8*xij**4 + 2*yij**4 + yij**2*zij**2 - zij**4 + xij**2*(-23*yij**2 + 7*zij**2))))/(8.*eta*PI*(xij**2 + yij**2 + zij**2)**6.5),
      (-1350*b**6*(2*xij**6 + 2*yij**6 - 15*yij**4*zij**2 - 15*yij**2*zij**4 + 2*zij**6 - 15*xij**4*(yij**2 + zij**2) - 
            15*xij**2*(yij**4 - 12*yij**2*zij**2 + zij**4)) + 
         7*b**4*(122*xij**8 - xij**6*(737*yij**2 + 857*zij**2) + (yij**2 + zij**2)**2*(122*yij**4 - 1101*yij**2*zij**2 + 142*zij**4) - 
            2*xij**4*(859*yij**4 - 4577*yij**2*zij**2 + 969*zij**4) + xij**2*(-737*yij**6 + 9154*yij**4*zij**2 + 9074*yij**2*zij**4 - 817*zij**6)))/
       (840.*eta*PI*(xij**2 + yij**2 + zij**2)**6.5)],[(450*b**6*
          (8*xij**6 - (6*yij**2 - zij**2)*(yij**2 + zij**2)**2 - 4*xij**4*(29*yij**2 + zij**2) + xij**2*(101*yij**4 + 90*yij**2*zij**2 - 11*zij**4))\
          + 7*b**4*(-168*xij**8 + 24*xij**6*(92*yij**2 - 3*zij**2) - xij**2*(1937*yij**2 - 218*zij**2)*(yij**2 + zij**2)**2 + 
            (122*yij**2 - 23*zij**2)*(yij**2 + zij**2)**3 + xij**4*(317*yij**4 + 654*yij**2*zij**2 + 337*zij**4)))/
       (280.*eta*PI*(xij**2 + yij**2 + zij**2)**6.5),(b**4*xij*yij*
         (-514*xij**6 + xij**4*(1293*yij**2 - 779*zij**2) - (514*yij**2 - 249*zij**2)*(yij**2 + zij**2)**2 + 
           xij**2*(1293*yij**4 + 1277*yij**2*zij**2 - 16*zij**4) + 
           810*b**2*(2*xij**4 + 2*yij**4 + yij**2*zij**2 - zij**4 + xij**2*(-7*yij**2 + zij**2))))/(24.*eta*PI*(xij**2 + yij**2 + zij**2)**6.5),
      (b**4*xij*zij*(270*b**2*(2*xij**4 + 8*yij**4 + 7*yij**2*zij**2 - zij**4 + xij**2*(-23*yij**2 + zij**2)) - 
           7*(26*xij**6 - 187*xij**2*yij**2*(yij**2 + zij**2) + (96*yij**2 - 13*zij**2)*(yij**2 + zij**2)**2 + xij**4*(-257*yij**2 + 39*zij**2))))/
       (24.*eta*PI*(xij**2 + yij**2 + zij**2)**6.5),(-1350*b**6*
          (6*xij**6 - 8*yij**6 + 4*yij**4*zij**2 + 11*yij**2*zij**4 - zij**6 + xij**4*(-101*yij**2 + 11*zij**2) + 
            2*xij**2*(58*yij**4 - 45*yij**2*zij**2 + 2*zij**4)) + 
         7*b**4*(362*xij**8 + xij**6*(-5807*yij**2 + 1033*zij**2) - (yij**2 + zij**2)**2*(488*yij**4 - 684*yij**2*zij**2 + 53*zij**4) + 
            xij**4*(987*yij**4 - 11036*yij**2*zij**2 + 927*zij**4) + xij**2*(6668*yij**6 + 1814*yij**4*zij**2 - 4651*yij**2*zij**4 + 203*zij**6)))/
       (840.*eta*PI*(xij**2 + yij**2 + zij**2)**6.5),(b**4*yij*zij*
         (-684*xij**6 + xij**4*(1301*yij**2 - 1289*zij**2) - (166*yij**2 - 79*zij**2)*(yij**2 + zij**2)**2 + 
           xij**2*(1819*yij**4 + 1293*yij**2*zij**2 - 526*zij**4) + 
           270*b**2*(8*xij**4 + 2*yij**4 + yij**2*zij**2 - zij**4 + xij**2*(-23*yij**2 + 7*zij**2))))/(24.*eta*PI*(xij**2 + yij**2 + zij**2)**6.5),
      -0.125*(b**4*xij*yij*(-141*xij**6 - 232*yij**6 + 340*yij**4*zij**2 + 431*yij**2*zij**4 - 141*zij**6 + xij**4*(431*yij**2 - 423*zij**2) + 
            xij**2*(340*yij**4 + 862*yij**2*zij**2 - 423*zij**4) + 
            90*b**2*(5*xij**4 + 8*yij**4 - 20*yij**2*zij**2 + 5*zij**4 + 10*xij**2*(-2*yij**2 + zij**2))))/(eta*PI*(xij**2 + yij**2 + zij**2)**6.5),
      (b**4*xij*zij*(83*xij**6 + 1392*yij**6 + 32*yij**4*zij**2 - 1277*yij**2*zij**4 + 83*zij**6 + xij**4*(-1277*yij**2 + 249*zij**2) + 
           xij**2*(32*yij**4 - 2554*yij**2*zij**2 + 249*zij**4) - 
           270*b**2*(xij**4 + 16*yij**4 - 16*yij**2*zij**2 + zij**4 + 2*xij**2*(-8*yij**2 + zij**2))))/(24.*eta*PI*(xij**2 + yij**2 + zij**2)**6.5)],
     [-0.125*(b**4*yij*zij*(-456*xij**6 + 4*xij**4*(yij**2 + zij**2) + 431*xij**2*(yij**2 + zij**2)**2 - 29*(yij**2 + zij**2)**3 + 
            90*b**2*(16*xij**4 - 16*xij**2*(yij**2 + zij**2) + (yij**2 + zij**2)**2)))/(eta*PI*(xij**2 + yij**2 + zij**2)**6.5),
      (b**4*xij*zij*(-166*xij**6 + xij**4*(1819*yij**2 - 253*zij**2) - (684*yij**2 - 79*zij**2)*(yij**2 + zij**2)**2 + 
           xij**2*(1301*yij**4 + 1293*yij**2*zij**2 - 8*zij**4) + 
           270*b**2*(2*xij**4 + 8*yij**4 + 7*yij**2*zij**2 - zij**4 + xij**2*(-23*yij**2 + zij**2))))/(24.*eta*PI*(xij**2 + yij**2 + zij**2)**6.5),
      (b**4*xij*yij*(-166*xij**6 + (79*yij**2 - 684*zij**2)*(yij**2 + zij**2)**2 + xij**4*(-253*yij**2 + 1819*zij**2) + 
           xij**2*(-8*yij**4 + 1293*yij**2*zij**2 + 1301*zij**4) + 
           270*b**2*(2*xij**4 - yij**4 + 7*yij**2*zij**2 + 8*zij**4 + xij**2*(yij**2 - 23*zij**2))))/(24.*eta*PI*(xij**2 + yij**2 + zij**2)**6.5),
      (b**4*yij*zij*(-684*xij**6 + xij**4*(1301*yij**2 - 1289*zij**2) - (166*yij**2 - 79*zij**2)*(yij**2 + zij**2)**2 + 
           xij**2*(1819*yij**4 + 1293*yij**2*zij**2 - 526*zij**4) + 
           270*b**2*(8*xij**4 + 2*yij**4 + yij**2*zij**2 - zij**4 + xij**2*(-23*yij**2 + 7*zij**2))))/(24.*eta*PI*(xij**2 + yij**2 + zij**2)**6.5),
      (-270*b**6*(2*xij**6 + 2*yij**6 - 15*yij**4*zij**2 - 15*yij**2*zij**4 + 2*zij**6 - 15*xij**4*(yij**2 + zij**2) - 
            15*xij**2*(yij**4 - 12*yij**2*zij**2 + zij**4)) + 
         7*b**4*(22*xij**8 - 157*xij**6*(yij**2 + zij**2) + (yij**2 + zij**2)**2*(22*yij**4 - 201*yij**2*zij**2 + 22*zij**4) - 
            2*xij**4*(179*yij**4 - 937*yij**2*zij**2 + 179*zij**4) + xij**2*(-157*yij**6 + 1874*yij**4*zij**2 + 1874*yij**2*zij**4 - 157*zij**6)))/
       (168.*eta*PI*(xij**2 + yij**2 + zij**2)**6.5),-0.125*
       (b**4*xij*zij*(-29*xij**6 - 456*yij**6 + 4*yij**4*zij**2 + 431*yij**2*zij**4 - 29*zij**6 + xij**4*(431*yij**2 - 87*zij**2) + 
            xij**2*(4*yij**4 + 862*yij**2*zij**2 - 87*zij**4) + 
            90*b**2*(xij**4 + 16*yij**4 - 16*yij**2*zij**2 + zij**4 + 2*xij**2*(-8*yij**2 + zij**2))))/(eta*PI*(xij**2 + yij**2 + zij**2)**6.5),
      (b**4*xij*yij*(79*xij**6 - 166*yij**6 + 1819*yij**4*zij**2 + 1301*yij**2*zij**4 - 684*zij**6 - 2*xij**4*(4*yij**2 + 263*zij**2) + 
           xij**2*(-253*yij**4 + 1293*yij**2*zij**2 - 1289*zij**4) - 
           270*b**2*(xij**4 - 2*yij**4 + 23*yij**2*zij**2 - 8*zij**4 - xij**2*(yij**2 + 7*zij**2))))/(24.*eta*PI*(xij**2 + yij**2 + zij**2)**6.5)],
     [(3*b**4*xij*yij*(-58*xij**6 + xij**4*(141*yij**2 - 83*zij**2) - (58*yij**2 - 33*zij**2)*(yij**2 + zij**2)**2 + 
           xij**2*(141*yij**4 + 149*yij**2*zij**2 + 8*zij**4) + 90*b**2*(2*xij**4 + 2*yij**4 + yij**2*zij**2 - zij**4 + xij**2*(-7*yij**2 + zij**2))))
        /(8.*eta*PI*(xij**2 + yij**2 + zij**2)**6.5),(-450*b**6*
          (6*xij**6 - 8*yij**6 + 4*yij**4*zij**2 + 11*yij**2*zij**4 - zij**6 + xij**4*(-101*yij**2 + 11*zij**2) + 
            2*xij**2*(58*yij**4 - 45*yij**2*zij**2 + 2*zij**4)) + 
         7*b**4*(122*xij**8 + xij**6*(-1937*yij**2 + 343*zij**2) - (yij**2 + zij**2)**2*(168*yij**4 - 264*yij**2*zij**2 + 23*zij**4) + 
            xij**4*(317*yij**4 - 3656*yij**2*zij**2 + 297*zij**4) + xij**2*(2208*yij**6 + 654*yij**4*zij**2 - 1501*yij**2*zij**4 + 53*zij**6)))/
       (280.*eta*PI*(xij**2 + yij**2 + zij**2)**6.5),(b**4*yij*zij*
         (-240*xij**6 + xij**4*(407*yij**2 - 447*zij**2) - (58*yij**2 - 33*zij**2)*(yij**2 + zij**2)**2 + 
           xij**2*(589*yij**4 + 415*yij**2*zij**2 - 174*zij**4) + 
           90*b**2*(8*xij**4 + 2*yij**4 + yij**2*zij**2 - zij**4 + xij**2*(-23*yij**2 + 7*zij**2))))/(8.*eta*PI*(xij**2 + yij**2 + zij**2)**6.5),
      -0.125*(b**4*xij*yij*(-141*xij**6 - 232*yij**6 + 340*yij**4*zij**2 + 431*yij**2*zij**4 - 141*zij**6 + xij**4*(431*yij**2 - 423*zij**2) + 
            xij**2*(340*yij**4 + 862*yij**2*zij**2 - 423*zij**4) + 
            90*b**2*(5*xij**4 + 8*yij**4 - 20*yij**2*zij**2 + 5*zij**4 + 10*xij**2*(-2*yij**2 + zij**2))))/(eta*PI*(xij**2 + yij**2 + zij**2)**6.5),
      -0.125*(b**4*xij*zij*(-29*xij**6 - 456*yij**6 + 4*yij**4*zij**2 + 431*yij**2*zij**4 - 29*zij**6 + xij**4*(431*yij**2 - 87*zij**2) + 
            xij**2*(4*yij**4 + 862*yij**2*zij**2 - 87*zij**4) + 
            90*b**2*(xij**4 + 16*yij**4 - 16*yij**2*zij**2 + zij**4 + 2*xij**2*(-8*yij**2 + zij**2))))/(eta*PI*(xij**2 + yij**2 + zij**2)**6.5),
      (450*b**6*(5*xij**6 - 16*yij**6 + 120*yij**4*zij**2 - 90*yij**2*zij**4 + 5*zij**6 + 15*xij**4*(-6*yij**2 + zij**2) + 
            15*xij**2*(8*yij**4 - 12*yij**2*zij**2 + zij**4)) - 
         21*b**4*(33*xij**8 - 112*yij**8 + 712*yij**6*zij**2 + 218*yij**4*zij**4 - 573*yij**2*zij**6 + 33*zij**8 + 
            xij**6*(-573*yij**2 + 132*zij**2) + xij**4*(218*yij**4 - 1719*yij**2*zij**2 + 198*zij**4) + 
            xij**2*(712*yij**6 + 436*yij**4*zij**2 - 1719*yij**2*zij**4 + 132*zij**6)))/(280.*eta*PI*(xij**2 + yij**2 + zij**2)**6.5),
      -0.125*(b**4*yij*zij*(-141*xij**6 - 232*yij**6 + 340*yij**4*zij**2 + 431*yij**2*zij**4 - 141*zij**6 + xij**4*(431*yij**2 - 423*zij**2) + 
            xij**2*(340*yij**4 + 862*yij**2*zij**2 - 423*zij**4) + 
            90*b**2*(5*xij**4 + 8*yij**4 - 20*yij**2*zij**2 + 5*zij**4 + 10*xij**2*(-2*yij**2 + zij**2))))/(eta*PI*(xij**2 + yij**2 + zij**2)**6.5)],
     [(b**4*xij*zij*(-58*xij**6 + xij**4*(589*yij**2 - 83*zij**2) - 3*(80*yij**2 - 11*zij**2)*(yij**2 + zij**2)**2 + 
           xij**2*(407*yij**4 + 415*yij**2*zij**2 + 8*zij**4) + 
           90*b**2*(2*xij**4 + 8*yij**4 + 7*yij**2*zij**2 - zij**4 + xij**2*(-23*yij**2 + zij**2))))/(8.*eta*PI*(xij**2 + yij**2 + zij**2)**6.5),
      (b**4*yij*zij*(270*b**2*(8*xij**4 + 2*yij**4 + yij**2*zij**2 - zij**4 + xij**2*(-23*yij**2 + 7*zij**2)) - 
           7*(96*xij**6 + xij**4*(-187*yij**2 + 179*zij**2) + xij**2*(-257*yij**4 - 187*yij**2*zij**2 + 70*zij**4) + 
              13*(2*yij**6 + 3*yij**4*zij**2 - zij**6))))/(24.*eta*PI*(xij**2 + yij**2 + zij**2)**6.5),
      (-1350*b**6*(2*xij**6 + 2*yij**6 - 15*yij**4*zij**2 - 15*yij**2*zij**4 + 2*zij**6 - 15*xij**4*(yij**2 + zij**2) - 
            15*xij**2*(yij**4 - 12*yij**2*zij**2 + zij**4)) + 
         7*b**4*(122*xij**8 - xij**6*(737*yij**2 + 857*zij**2) + (yij**2 + zij**2)**2*(122*yij**4 - 1101*yij**2*zij**2 + 142*zij**4) - 
            2*xij**4*(859*yij**4 - 4577*yij**2*zij**2 + 969*zij**4) + xij**2*(-737*yij**6 + 9154*yij**4*zij**2 + 9074*yij**2*zij**4 - 817*zij**6)))/
       (840.*eta*PI*(xij**2 + yij**2 + zij**2)**6.5),(b**4*xij*zij*
         (83*xij**6 + 1392*yij**6 + 32*yij**4*zij**2 - 1277*yij**2*zij**4 + 83*zij**6 + xij**4*(-1277*yij**2 + 249*zij**2) + 
           xij**2*(32*yij**4 - 2554*yij**2*zij**2 + 249*zij**4) - 
           270*b**2*(xij**4 + 16*yij**4 - 16*yij**2*zij**2 + zij**4 + 2*xij**2*(-8*yij**2 + zij**2))))/(24.*eta*PI*(xij**2 + yij**2 + zij**2)**6.5),
      (b**4*xij*yij*(79*xij**6 - 166*yij**6 + 1819*yij**4*zij**2 + 1301*yij**2*zij**4 - 684*zij**6 - 2*xij**4*(4*yij**2 + 263*zij**2) + 
           xij**2*(-253*yij**4 + 1293*yij**2*zij**2 - 1289*zij**4) - 
           270*b**2*(xij**4 - 2*yij**4 + 23*yij**2*zij**2 - 8*zij**4 - xij**2*(yij**2 + 7*zij**2))))/(24.*eta*PI*(xij**2 + yij**2 + zij**2)**6.5),
      -0.125*(b**4*yij*zij*(-141*xij**6 - 232*yij**6 + 340*yij**4*zij**2 + 431*yij**2*zij**4 - 141*zij**6 + xij**4*(431*yij**2 - 423*zij**2) + 
            xij**2*(340*yij**4 + 862*yij**2*zij**2 - 423*zij**4) + 
            90*b**2*(5*xij**4 + 8*yij**4 - 20*yij**2*zij**2 + 5*zij**4 + 10*xij**2*(-2*yij**2 + zij**2))))/(eta*PI*(xij**2 + yij**2 + zij**2)**6.5),
      (1350*b**6*(xij**6 + 8*yij**6 - 116*yij**4*zij**2 + 101*yij**2*zij**4 - 6*zij**6 - xij**4*(11*yij**2 + 4*zij**2) + 
            xij**2*(-4*yij**4 + 90*yij**2*zij**2 - 11*zij**4)) - 
         7*b**4*(53*xij**8 + 488*yij**8 - 6668*yij**6*zij**2 - 987*yij**4*zij**4 + 5807*yij**2*zij**6 - 362*zij**8 - 
            xij**6*(578*yij**2 + 203*zij**2) + xij**4*(-827*yij**4 + 4651*yij**2*zij**2 - 927*zij**4) + 
            xij**2*(292*yij**6 - 1814*yij**4*zij**2 + 11036*yij**2*zij**4 - 1033*zij**6)))/(840.*eta*PI*(xij**2 + yij**2 + zij**2)**6.5)]])



##
## matrix elements connecting RBM to higher slip modes: K^LH
##


def K1s2s(xij,yij,zij, b,eta):
    return np.array([[(b**2*xij*(2*xij**4 - yij**4 - zij**2*(-18 + zij**2) - 2*yij**2*(-9 + zij**2) + xij**2*(-12 + yij**2 + zij**2)))/
       (3.*(xij**2 + yij**2 + zij**2)**3.5),(b**2*yij*(xij**4 + xij**2*(-8 + yij**2 + zij**2) + 2*(yij**2 + zij**2)))/
       (xij**2 + yij**2 + zij**2)**3.5,(b**2*zij*(xij**4 + xij**2*(-8 + yij**2 + zij**2) + 2*(yij**2 + zij**2)))/
       (xij**2 + yij**2 + zij**2)**3.5,-0.3333333333333333*
       (b**2*xij*(xij**4 - 2*yij**4 - xij**2*(6 + yij**2 - 2*zij**2) - yij**2*(-24 + zij**2) + zij**2*(-6 + zij**2)))/
        (xij**2 + yij**2 + zij**2)**3.5,(b**2*xij*yij*zij*(-10 + xij**2 + yij**2 + zij**2))/(xij**2 + yij**2 + zij**2)**3.5],
     [-0.3333333333333333*(b**2*yij*(-2*xij**4 + yij**4 + zij**2*(-6 + zij**2) + 2*yij**2*(-3 + zij**2) - xij**2*(-24 + yij**2 + zij**2)))/
        (xij**2 + yij**2 + zij**2)**3.5,(b**2*xij*(yij**4 + xij**2*(2 + yij**2) + 2*zij**2 + yij**2*(-8 + zij**2)))/
       (xij**2 + yij**2 + zij**2)**3.5,(b**2*xij*yij*zij*(-10 + xij**2 + yij**2 + zij**2))/(xij**2 + yij**2 + zij**2)**3.5,
      (b**2*yij*(-xij**4 + 2*yij**4 + xij**2*(18 + yij**2 - 2*zij**2) - zij**2*(-18 + zij**2) + yij**2*(-12 + zij**2)))/
       (3.*(xij**2 + yij**2 + zij**2)**3.5),(b**2*zij*(yij**4 + xij**2*(2 + yij**2) + 2*zij**2 + yij**2*(-8 + zij**2)))/
       (xij**2 + yij**2 + zij**2)**3.5],[-0.3333333333333333*
       (b**2*zij*(-2*xij**4 + yij**4 + zij**2*(-6 + zij**2) + 2*yij**2*(-3 + zij**2) - xij**2*(-24 + yij**2 + zij**2)))/
        (xij**2 + yij**2 + zij**2)**3.5,(b**2*xij*yij*zij*(-10 + xij**2 + yij**2 + zij**2))/(xij**2 + yij**2 + zij**2)**3.5,
      (b**2*xij*(zij**2*(-8 + zij**2) + xij**2*(2 + zij**2) + yij**2*(2 + zij**2)))/(xij**2 + yij**2 + zij**2)**3.5,
      -0.3333333333333333*(b**2*zij*(xij**4 - 2*yij**4 - xij**2*(6 + yij**2 - 2*zij**2) - yij**2*(-24 + zij**2) + zij**2*(-6 + zij**2)))/
        (xij**2 + yij**2 + zij**2)**3.5,(b**2*yij*(zij**2*(-8 + zij**2) + xij**2*(2 + zij**2) + yij**2*(2 + zij**2)))/
       (xij**2 + yij**2 + zij**2)**3.5]])



def K1s3t(xij,yij,zij, b,eta):
    return np.array([[(3*b**3*(2*xij**2 - yij**2 - zij**2))/(50.*(xij**2 + yij**2 + zij**2)**2.5),
      (9*b**3*xij*yij)/(50.*(xij**2 + yij**2 + zij**2)**2.5),(9*b**3*xij*zij)/(50.*(xij**2 + yij**2 + zij**2)**2.5)],
     [(9*b**3*xij*yij)/(50.*(xij**2 + yij**2 + zij**2)**2.5),(-3*b**3*(xij**2 - 2*yij**2 + zij**2))/(50.*(xij**2 + yij**2 + zij**2)**2.5),
      (9*b**3*yij*zij)/(50.*(xij**2 + yij**2 + zij**2)**2.5)],
     [(9*b**3*xij*zij)/(50.*(xij**2 + yij**2 + zij**2)**2.5),(9*b**3*yij*zij)/(50.*(xij**2 + yij**2 + zij**2)**2.5),
      (-3*b**3*(xij**2 + yij**2 - 2*zij**2))/(50.*(xij**2 + yij**2 + zij**2)**2.5)]])



def K2a2s(xij,yij,zij, b,eta):
    return np.array([[0,(b**3*xij*zij)/(xij**2 + yij**2 + zij**2)**2.5,-((b**3*xij*yij)/(xij**2 + yij**2 + zij**2)**2.5),
      (2*b**3*yij*zij)/(xij**2 + yij**2 + zij**2)**2.5,-((b**3*(yij**2 - zij**2))/(xij**2 + yij**2 + zij**2)**2.5)],
     [(-2*b**3*xij*zij)/(xij**2 + yij**2 + zij**2)**2.5,-((b**3*yij*zij)/(xij**2 + yij**2 + zij**2)**2.5),
      (b**3*(xij**2 - zij**2))/(xij**2 + yij**2 + zij**2)**2.5,0,(b**3*xij*yij)/(xij**2 + yij**2 + zij**2)**2.5],
     [(2*b**3*xij*yij)/(xij**2 + yij**2 + zij**2)**2.5,-((b**3*(xij**2 - yij**2))/(xij**2 + yij**2 + zij**2)**2.5),
      (b**3*yij*zij)/(xij**2 + yij**2 + zij**2)**2.5,(-2*b**3*xij*yij)/(xij**2 + yij**2 + zij**2)**2.5,
      -((b**3*xij*zij)/(xij**2 + yij**2 + zij**2)**2.5)]])


##
## matrix elements connecting higher slip modes with themselves: K^HH
##


def K2s2s(xij,yij,zij, b,eta):
    return np.array([[(6*b**5*(8*xij**4 - 24*xij**2*(yij**2 + zij**2) + 3*(yij**2 + zij**2)**2) - 
         5*b**3*(4*xij**6 - 6*xij**4*(yij**2 + zij**2) - 9*xij**2*(yij**2 + zij**2)**2 + (yij**2 + zij**2)**3))/
       (15.*(xij**2 + yij**2 + zij**2)**4.5),-((b**3*xij*yij*
           (3*xij**4 + xij**2*(yij**2 + zij**2) - 2*(yij**2 + zij**2)**2 + b**2*(-8*xij**2 + 6*(yij**2 + zij**2))))/
         (xij**2 + yij**2 + zij**2)**4.5),-((b**3*xij*zij*
           (3*xij**4 + xij**2*(yij**2 + zij**2) - 2*(yij**2 + zij**2)**2 + b**2*(-8*xij**2 + 6*(yij**2 + zij**2))))/
         (xij**2 + yij**2 + zij**2)**4.5),(-6*b**5*(4*xij**4 + 4*yij**4 + 3*yij**2*zij**2 - zij**4 + 3*xij**2*(-9*yij**2 + zij**2)) + 
         5*b**3*(2*xij**6 + 2*yij**6 + 3*yij**4*zij**2 - zij**6 - 9*xij**2*yij**2*(yij**2 + zij**2) + xij**4*(-9*yij**2 + 3*zij**2)))/
       (15.*(xij**2 + yij**2 + zij**2)**4.5),(b**3*yij*zij*
         (-4*xij**4 + 2*b**2*(6*xij**2 - yij**2 - zij**2) - 3*xij**2*(yij**2 + zij**2) + (yij**2 + zij**2)**2))/(xij**2 + yij**2 + zij**2)**4.5
                     ],[-((b**3*xij*yij*(3*xij**4 + xij**2*(yij**2 + zij**2) - 2*(yij**2 + zij**2)**2 + b**2*(-8*xij**2 + 6*(yij**2 + zij**2))))/
         (xij**2 + yij**2 + zij**2)**4.5),(-4*b**5*(4*xij**4 + 4*yij**4 + 3*yij**2*zij**2 - zij**4 + 3*xij**2*(-9*yij**2 + zij**2)) + 
         5*b**3*(xij**6 + yij**2*(yij**2 + zij**2)**2 + xij**4*(-7*yij**2 + 2*zij**2) + xij**2*(-7*yij**4 - 6*yij**2*zij**2 + zij**4)))/
       (10.*(xij**2 + yij**2 + zij**2)**4.5),(b**3*yij*zij*
         (-9*xij**4 + 4*b**2*(6*xij**2 - yij**2 - zij**2) - 8*xij**2*(yij**2 + zij**2) + (yij**2 + zij**2)**2))/
       (2.*(xij**2 + yij**2 + zij**2)**4.5),-((b**3*xij*yij*
           (-2*xij**4 + 3*yij**4 + yij**2*zij**2 - 2*zij**4 + xij**2*(yij**2 - 4*zij**2) + b**2*(6*xij**2 - 8*yij**2 + 6*zij**2)))/
         (xij**2 + yij**2 + zij**2)**4.5),(b**3*xij*zij*
         (xij**4 - 9*yij**4 - 8*yij**2*zij**2 + zij**4 - 4*b**2*(xij**2 - 6*yij**2 + zij**2) + xij**2*(-8*yij**2 + 2*zij**2)))/
       (2.*(xij**2 + yij**2 + zij**2)**4.5)],[-((b**3*xij*zij*
           (3*xij**4 + xij**2*(yij**2 + zij**2) - 2*(yij**2 + zij**2)**2 + b**2*(-8*xij**2 + 6*(yij**2 + zij**2))))/
         (xij**2 + yij**2 + zij**2)**4.5),(b**3*yij*zij*
         (-9*xij**4 + 4*b**2*(6*xij**2 - yij**2 - zij**2) - 8*xij**2*(yij**2 + zij**2) + (yij**2 + zij**2)**2))/
       (2.*(xij**2 + yij**2 + zij**2)**4.5),(-4*b**5*(4*xij**4 - yij**4 + 3*yij**2*zij**2 + 4*zij**4 + 3*xij**2*(yij**2 - 9*zij**2)) + 
         5*b**3*(xij**6 + xij**4*(2*yij**2 - 7*zij**2) + zij**2*(yij**2 + zij**2)**2 + xij**2*(yij**4 - 6*yij**2*zij**2 - 7*zij**4)))/
       (10.*(xij**2 + yij**2 + zij**2)**4.5),(b**3*xij*zij*
         (xij**4 - 4*yij**4 - 3*yij**2*zij**2 + zij**4 - 2*b**2*(xij**2 - 6*yij**2 + zij**2) + xij**2*(-3*yij**2 + 2*zij**2)))/
       (xij**2 + yij**2 + zij**2)**4.5,(b**3*xij*yij*(xij**4 + yij**4 - 8*yij**2*zij**2 - 9*zij**4 - 4*b**2*(xij**2 + yij**2 - 6*zij**2) + 
           2*xij**2*(yij**2 - 4*zij**2)))/(2.*(xij**2 + yij**2 + zij**2)**4.5)],
     [(-6*b**5*(4*xij**4 + 4*yij**4 + 3*yij**2*zij**2 - zij**4 + 3*xij**2*(-9*yij**2 + zij**2)) + 
         5*b**3*(2*xij**6 + 2*yij**6 + 3*yij**4*zij**2 - zij**6 - 9*xij**2*yij**2*(yij**2 + zij**2) + xij**4*(-9*yij**2 + 3*zij**2)))/
       (15.*(xij**2 + yij**2 + zij**2)**4.5),-((b**3*xij*yij*
           (-2*xij**4 + 3*yij**4 + yij**2*zij**2 - 2*zij**4 + xij**2*(yij**2 - 4*zij**2) + b**2*(6*xij**2 - 8*yij**2 + 6*zij**2)))/
         (xij**2 + yij**2 + zij**2)**4.5),(b**3*xij*zij*
         (xij**4 - 4*yij**4 - 3*yij**2*zij**2 + zij**4 - 2*b**2*(xij**2 - 6*yij**2 + zij**2) + xij**2*(-3*yij**2 + 2*zij**2)))/
       (xij**2 + yij**2 + zij**2)**4.5,(6*b**5*(3*xij**4 + 8*yij**4 - 24*yij**2*zij**2 + 3*zij**4 + 6*xij**2*(-4*yij**2 + zij**2)) - 
         5*b**3*(xij**6 + 4*yij**6 - 6*yij**4*zij**2 - 9*yij**2*zij**4 + zij**6 + xij**4*(-9*yij**2 + 3*zij**2) - 
            3*xij**2*(2*yij**4 + 6*yij**2*zij**2 - zij**4)))/(15.*(xij**2 + yij**2 + zij**2)**4.5),
      -((b**3*yij*zij*(-2*xij**4 + 3*yij**4 + yij**2*zij**2 - 2*zij**4 + xij**2*(yij**2 - 4*zij**2) + b**2*(6*xij**2 - 8*yij**2 + 6*zij**2)))/
         (xij**2 + yij**2 + zij**2)**4.5)],[(b**3*yij*zij*
         (-4*xij**4 + 2*b**2*(6*xij**2 - yij**2 - zij**2) - 3*xij**2*(yij**2 + zij**2) + (yij**2 + zij**2)**2))/(xij**2 + yij**2 + zij**2)**4.5
       ,(b**3*xij*zij*(xij**4 - 9*yij**4 - 8*yij**2*zij**2 + zij**4 - 4*b**2*(xij**2 - 6*yij**2 + zij**2) + xij**2*(-8*yij**2 + 2*zij**2)))/
       (2.*(xij**2 + yij**2 + zij**2)**4.5),(b**3*xij*yij*
         (xij**4 + yij**4 - 8*yij**2*zij**2 - 9*zij**4 - 4*b**2*(xij**2 + yij**2 - 6*zij**2) + 2*xij**2*(yij**2 - 4*zij**2)))/
       (2.*(xij**2 + yij**2 + zij**2)**4.5),-((b**3*yij*zij*
           (-2*xij**4 + 3*yij**4 + yij**2*zij**2 - 2*zij**4 + xij**2*(yij**2 - 4*zij**2) + b**2*(6*xij**2 - 8*yij**2 + 6*zij**2)))/
         (xij**2 + yij**2 + zij**2)**4.5),(4*b**5*(xij**4 - 4*yij**4 + 27*yij**2*zij**2 - 4*zij**4 - 3*xij**2*(yij**2 + zij**2)) + 
         5*b**3*(yij**6 - 7*yij**4*zij**2 - 7*yij**2*zij**4 + zij**6 + xij**4*(yij**2 + zij**2) + 2*xij**2*(yij**4 - 3*yij**2*zij**2 + zij**4))
         )/(10.*(xij**2 + yij**2 + zij**2)**4.5)]])



def K2s3t(xij,yij,zij, b,eta):
    return np.array([[(-9*b**4*(2*xij**3 - 3*xij*(yij**2 + zij**2)))/(50.*(xij**2 + yij**2 + zij**2)**3.5),
      (9*b**4*yij*(-4*xij**2 + yij**2 + zij**2))/(50.*(xij**2 + yij**2 + zij**2)**3.5),
      (9*b**4*zij*(-4*xij**2 + yij**2 + zij**2))/(50.*(xij**2 + yij**2 + zij**2)**3.5)],
     [(9*b**4*yij*(-4*xij**2 + yij**2 + zij**2))/(50.*(xij**2 + yij**2 + zij**2)**3.5),
      (9*b**4*xij*(xij**2 - 4*yij**2 + zij**2))/(50.*(xij**2 + yij**2 + zij**2)**3.5),
      (-9*b**4*xij*yij*zij)/(10.*(xij**2 + yij**2 + zij**2)**3.5)],
     [(9*b**4*zij*(-4*xij**2 + yij**2 + zij**2))/(50.*(xij**2 + yij**2 + zij**2)**3.5),
      (-9*b**4*xij*yij*zij)/(10.*(xij**2 + yij**2 + zij**2)**3.5),
      (9*b**4*xij*(xij**2 + yij**2 - 4*zij**2))/(50.*(xij**2 + yij**2 + zij**2)**3.5)],
     [(9*b**4*xij*(xij**2 - 4*yij**2 + zij**2))/(50.*(xij**2 + yij**2 + zij**2)**3.5),
      (9*b**4*yij*(3*xij**2 - 2*yij**2 + 3*zij**2))/(50.*(xij**2 + yij**2 + zij**2)**3.5),
      (9*b**4*zij*(xij**2 - 4*yij**2 + zij**2))/(50.*(xij**2 + yij**2 + zij**2)**3.5)],
     [(-9*b**4*xij*yij*zij)/(10.*(xij**2 + yij**2 + zij**2)**3.5),
      (9*b**4*zij*(xij**2 - 4*yij**2 + zij**2))/(50.*(xij**2 + yij**2 + zij**2)**3.5),
      (9*b**4*yij*(xij**2 + yij**2 - 4*zij**2))/(50.*(xij**2 + yij**2 + zij**2)**3.5)]])



def K3a2s(xij,yij,zij, b,eta):
    return np.array([[0,(b**4*zij*(-4*xij**2 + yij**2 + zij**2))/(xij**2 + yij**2 + zij**2)**3.5,
      -((b**4*yij*(-4*xij**2 + yij**2 + zij**2))/(xij**2 + yij**2 + zij**2)**3.5),(-10*b**4*xij*yij*zij)/(xij**2 + yij**2 + zij**2)**3.5,
      (5*b**4*xij*(yij**2 - zij**2))/(xij**2 + yij**2 + zij**2)**3.5],
     [-((b**4*zij*(-4*xij**2 + yij**2 + zij**2))/(xij**2 + yij**2 + zij**2)**3.5),0,
      -((b**4*(2*xij**3 - 3*xij*(yij**2 + zij**2)))/(xij**2 + yij**2 + zij**2)**3.5),
      (b**4*zij*(xij**2 - 4*yij**2 + zij**2))/(xij**2 + yij**2 + zij**2)**3.5,
      (b**4*yij*(-3*xij**2 + 2*yij**2 - 3*zij**2))/(xij**2 + yij**2 + zij**2)**3.5],
     [(b**4*yij*(-4*xij**2 + yij**2 + zij**2))/(xij**2 + yij**2 + zij**2)**3.5,
      (b**4*(2*xij**3 - 3*xij*(yij**2 + zij**2)))/(xij**2 + yij**2 + zij**2)**3.5,0,
      (5*b**4*yij*(xij**2 - zij**2))/(xij**2 + yij**2 + zij**2)**3.5,
      (b**4*zij*(3*xij**2 + 3*yij**2 - 2*zij**2))/(xij**2 + yij**2 + zij**2)**3.5],
     [(10*b**4*xij*yij*zij)/(xij**2 + yij**2 + zij**2)**3.5,-((b**4*zij*(xij**2 - 4*yij**2 + zij**2))/(xij**2 + yij**2 + zij**2)**3.5),
      (-5*b**4*yij*(xij**2 - zij**2))/(xij**2 + yij**2 + zij**2)**3.5,0,
      (b**4*xij*(xij**2 - 4*yij**2 + zij**2))/(xij**2 + yij**2 + zij**2)**3.5],
     [(-5*b**4*xij*(yij**2 - zij**2))/(xij**2 + yij**2 + zij**2)**3.5,
      (b**4*yij*(3*xij**2 - 2*yij**2 + 3*zij**2))/(xij**2 + yij**2 + zij**2)**3.5,
      (b**4*zij*(-3*xij**2 - 3*yij**2 + 2*zij**2))/(xij**2 + yij**2 + zij**2)**3.5,
      -((b**4*xij*(xij**2 - 4*yij**2 + zij**2))/(xij**2 + yij**2 + zij**2)**3.5),0]])




def K3s2s(xij,yij,zij, b,eta):
    return np.array([[(-60*b**6*xij*(8*xij**4 - 40*xij**2*(yij**2 + zij**2) + 15*(yij**2 + zij**2)**2) + 
         7*b**4*xij*(24*xij**6 - 88*xij**4*(yij**2 + zij**2) - 73*xij**2*(yij**2 + zij**2)**2 + 39*(yij**2 + zij**2)**3))/
       (35.*(xij**2 + yij**2 + zij**2)**5.5),(-180*b**6*yij*(8*xij**4 - 12*xij**2*(yij**2 + zij**2) + (yij**2 + zij**2)**2) + 
         7*b**4*yij*(68*xij**6 - 31*xij**4*(yij**2 + zij**2) - 91*xij**2*(yij**2 + zij**2)**2 + 8*(yij**2 + zij**2)**3))/
       (35.*(xij**2 + yij**2 + zij**2)**5.5),(-180*b**6*zij*(8*xij**4 - 12*xij**2*(yij**2 + zij**2) + (yij**2 + zij**2)**2) + 
         7*b**4*zij*(68*xij**6 - 31*xij**4*(yij**2 + zij**2) - 91*xij**2*(yij**2 + zij**2)**2 + 8*(yij**2 + zij**2)**3))/
       (35.*(xij**2 + yij**2 + zij**2)**5.5),(b**4*xij*(-84*xij**6 - 77*xij**4*(-9*yij**2 + zij**2) - 91*(4*yij**2 - zij**2)*(yij**2 + zij**2)**2 + 
           7*xij**2*(59*yij**4 + 73*yij**2*zij**2 + 14*zij**4) + 
           60*b**2*(4*xij**4 + xij**2*(-41*yij**2 + zij**2) + 3*(6*yij**4 + 5*yij**2*zij**2 - zij**4))))/(35.*(xij**2 + yij**2 + zij**2)**5.5),
      (b**4*xij*yij*zij*(22*xij**4 + 9*xij**2*(yij**2 + zij**2) - 13*(yij**2 + zij**2)**2 + 36*b**2*(-2*xij**2 + yij**2 + zij**2)))/
       (xij**2 + yij**2 + zij**2)**5.5],[(-540*b**6*yij*(8*xij**4 - 12*xij**2*(yij**2 + zij**2) + (yij**2 + zij**2)**2) + 
         7*b**4*yij*(208*xij**6 - 86*xij**4*(yij**2 + zij**2) - 271*xij**2*(yij**2 + zij**2)**2 + 23*(yij**2 + zij**2)**3))/
       (105.*(xij**2 + yij**2 + zij**2)**5.5),(b**4*xij*(180*b**2*
            (4*xij**4 + xij**2*(-41*yij**2 + zij**2) + 3*(6*yij**4 + 5*yij**2*zij**2 - zij**4)) - 
           7*(32*xij**6 + 3*(49*yij**2 - 6*zij**2)*(yij**2 + zij**2)**2 + xij**4*(-314*yij**2 + 46*zij**2) - 
              xij**2*(199*yij**4 + 203*yij**2*zij**2 + 4*zij**4))))/(105.*(xij**2 + yij**2 + zij**2)**5.5),
      (b**4*xij*yij*zij*(24*xij**4 + 13*xij**2*(yij**2 + zij**2) - 11*(yij**2 + zij**2)**2 + 36*b**2*(-2*xij**2 + yij**2 + zij**2)))/
       (xij**2 + yij**2 + zij**2)**5.5,(b**4*yij*(180*b**2*(18*xij**4 + 4*yij**4 + yij**2*zij**2 - 3*zij**4 + xij**2*(-41*yij**2 + 15*zij**2)) - 
           7*(146*xij**6 + (36*yij**2 - 29*zij**2)*(yij**2 + zij**2)**2 + xij**4*(-197*yij**2 + 263*zij**2) + 
              xij**2*(-307*yij**4 - 219*yij**2*zij**2 + 88*zij**4))))/(105.*(xij**2 + yij**2 + zij**2)**5.5),
      (b**4*zij*(180*b**2*(6*xij**4 + 6*yij**4 + 5*yij**2*zij**2 - zij**4 + xij**2*(-51*yij**2 + 5*zij**2)) - 
           7*(42*xij**6 + (57*yij**2 - 8*zij**2)*(yij**2 + zij**2)**2 + xij**4*(-384*yij**2 + 76*zij**2) + 
              xij**2*(-369*yij**4 - 343*yij**2*zij**2 + 26*zij**4))))/(105.*(xij**2 + yij**2 + zij**2)**5.5)],
     [(-540*b**6*zij*(8*xij**4 - 12*xij**2*(yij**2 + zij**2) + (yij**2 + zij**2)**2) + 
         7*b**4*zij*(208*xij**6 - 86*xij**4*(yij**2 + zij**2) - 271*xij**2*(yij**2 + zij**2)**2 + 23*(yij**2 + zij**2)**3))/
       (105.*(xij**2 + yij**2 + zij**2)**5.5),(b**4*xij*yij*zij*
         (24*xij**4 + 13*xij**2*(yij**2 + zij**2) - 11*(yij**2 + zij**2)**2 + 36*b**2*(-2*xij**2 + yij**2 + zij**2)))/(xij**2 + yij**2 + zij**2)**5.5,
      (b**4*xij*(180*b**2*(4*xij**4 + xij**2*(yij**2 - 41*zij**2) - 3*(yij**4 - 5*yij**2*zij**2 - 6*zij**4)) - 
           7*(32*xij**6 + xij**4*(46*yij**2 - 314*zij**2) - 3*(6*yij**2 - 49*zij**2)*(yij**2 + zij**2)**2 - 
              xij**2*(4*yij**4 + 203*yij**2*zij**2 + 199*zij**4))))/(105.*(xij**2 + yij**2 + zij**2)**5.5),
      (b**4*zij*(180*b**2*(6*xij**4 + 6*yij**4 + 5*yij**2*zij**2 - zij**4 + xij**2*(-51*yij**2 + 5*zij**2)) - 
           7*(62*xij**6 + 13*(4*yij**2 - zij**2)*(yij**2 + zij**2)**2 + xij**4*(-349*yij**2 + 111*zij**2) + 
              xij**2*(-359*yij**4 - 323*yij**2*zij**2 + 36*zij**4))))/(105.*(xij**2 + yij**2 + zij**2)**5.5),
      (b**4*yij*(180*b**2*(6*xij**4 - yij**4 + 5*yij**2*zij**2 + 6*zij**4 + xij**2*(5*yij**2 - 51*zij**2)) - 
           7*(42*xij**6 + 4*xij**4*(19*yij**2 - 96*zij**2) - (8*yij**2 - 57*zij**2)*(yij**2 + zij**2)**2 + 
              xij**2*(26*yij**4 - 343*yij**2*zij**2 - 369*zij**4))))/(105.*(xij**2 + yij**2 + zij**2)**5.5)],
     [(b**4*xij*(180*b**2*(4*xij**4 + xij**2*(-41*yij**2 + zij**2) + 3*(6*yij**4 + 5*yij**2*zij**2 - zij**4)) - 
           7*(36*xij**6 + (146*yij**2 - 29*zij**2)*(yij**2 + zij**2)**2 + xij**4*(-307*yij**2 + 43*zij**2) - 
              xij**2*(197*yij**4 + 219*yij**2*zij**2 + 22*zij**4))))/(105.*(xij**2 + yij**2 + zij**2)**5.5),
      (b**4*yij*(180*b**2*(18*xij**4 + 4*yij**4 + yij**2*zij**2 - 3*zij**4 + xij**2*(-41*yij**2 + 15*zij**2)) - 
           7*(147*xij**6 + 2*(16*yij**2 - 9*zij**2)*(yij**2 + zij**2)**2 + xij**4*(-199*yij**2 + 276*zij**2) + 
              xij**2*(-314*yij**4 - 203*yij**2*zij**2 + 111*zij**4))))/(105.*(xij**2 + yij**2 + zij**2)**5.5),
      (b**4*zij*(180*b**2*(6*xij**4 + 6*yij**4 + 5*yij**2*zij**2 - zij**4 + xij**2*(-51*yij**2 + 5*zij**2)) - 
           7*(57*xij**6 + 2*(21*yij**2 - 4*zij**2)*(yij**2 + zij**2)**2 + xij**4*(-369*yij**2 + 106*zij**2) + 
              xij**2*(-384*yij**4 - 343*yij**2*zij**2 + 41*zij**4))))/(105.*(xij**2 + yij**2 + zij**2)**5.5),
      (-540*b**6*xij*(xij**4 + 8*yij**4 - 12*yij**2*zij**2 + zij**4 + 2*xij**2*(-6*yij**2 + zij**2)) + 
         7*b**4*xij*(23*xij**6 + 208*yij**6 - 86*yij**4*zij**2 - 271*yij**2*zij**4 + 23*zij**6 + xij**4*(-271*yij**2 + 69*zij**2) + 
            xij**2*(-86*yij**4 - 542*yij**2*zij**2 + 69*zij**4)))/(105.*(xij**2 + yij**2 + zij**2)**5.5),
      (b**4*xij*yij*zij*(-11*xij**4 + 24*yij**4 + 13*yij**2*zij**2 - 11*zij**4 + xij**2*(13*yij**2 - 22*zij**2) + 
           36*b**2*(xij**2 - 2*yij**2 + zij**2)))/(xij**2 + yij**2 + zij**2)**5.5],
     [(b**4*xij*yij*zij*(2*xij**2 - yij**2 - zij**2)*(-108*b**2 + 35*(xij**2 + yij**2 + zij**2)))/(3.*(xij**2 + yij**2 + zij**2)**5.5),
      (b**4*zij*(36*b**2*(6*xij**4 + 6*yij**4 + 5*yij**2*zij**2 - zij**4 + xij**2*(-51*yij**2 + 5*zij**2)) - 
           7*(9*xij**6 + (9*yij**2 - zij**2)*(yij**2 + zij**2)**2 + xij**4*(-78*yij**2 + 17*zij**2) + 
              xij**2*(-78*yij**4 - 71*yij**2*zij**2 + 7*zij**4))))/(21.*(xij**2 + yij**2 + zij**2)**5.5),
      (b**4*yij*(36*b**2*(6*xij**4 - yij**4 + 5*yij**2*zij**2 + 6*zij**4 + xij**2*(5*yij**2 - 51*zij**2)) - 
           7*(9*xij**6 + xij**4*(17*yij**2 - 78*zij**2) - (yij**2 - 9*zij**2)*(yij**2 + zij**2)**2 + xij**2*(7*yij**4 - 71*yij**2*zij**2 - 78*zij**4))
           ))/(21.*(xij**2 + yij**2 + zij**2)**5.5),(b**4*xij*yij*zij*(xij**2 - 2*yij**2 + zij**2)*(108*b**2 - 35*(xij**2 + yij**2 + zij**2)))/
       (3.*(xij**2 + yij**2 + zij**2)**5.5),(b**4*xij*(-36*b**2*(xij**4 - 6*yij**4 + 51*yij**2*zij**2 - 6*zij**4 - 5*xij**2*(yij**2 + zij**2)) + 
           7*(xij**6 - 9*yij**6 + 78*yij**4*zij**2 + 78*yij**2*zij**4 - 9*zij**6 - 7*xij**4*(yij**2 + zij**2) + 
              xij**2*(-17*yij**4 + 71*yij**2*zij**2 - 17*zij**4))))/(21.*(xij**2 + yij**2 + zij**2)**5.5)],
     [(b**4*yij*(60*b**2*(18*xij**4 + 4*yij**4 + yij**2*zij**2 - 3*zij**4 + xij**2*(-41*yij**2 + 15*zij**2)) - 
           7*(52*xij**6 + (12*yij**2 - 13*zij**2)*(yij**2 + zij**2)**2 + xij**4*(-59*yij**2 + 91*zij**2) + 
              xij**2*(-99*yij**4 - 73*yij**2*zij**2 + 26*zij**4))))/(35.*(xij**2 + yij**2 + zij**2)**5.5),
      (-180*b**6*xij*(xij**4 + 8*yij**4 - 12*yij**2*zij**2 + zij**4 + 2*xij**2*(-6*yij**2 + zij**2)) + 
         7*b**4*xij*(8*xij**6 + 68*yij**6 - 31*yij**4*zij**2 - 91*yij**2*zij**4 + 8*zij**6 + xij**4*(-91*yij**2 + 24*zij**2) + 
            xij**2*(-31*yij**4 - 182*yij**2*zij**2 + 24*zij**4)))/(35.*(xij**2 + yij**2 + zij**2)**5.5),
      (b**4*xij*yij*zij*(-13*xij**4 + 22*yij**4 + 9*yij**2*zij**2 - 13*zij**4 + xij**2*(9*yij**2 - 26*zij**2) + 
           36*b**2*(xij**2 - 2*yij**2 + zij**2)))/(xij**2 + yij**2 + zij**2)**5.5,
      (-60*b**6*yij*(15*xij**4 + 8*yij**4 - 40*yij**2*zij**2 + 15*zij**4 + xij**2*(-40*yij**2 + 30*zij**2)) + 
         7*b**4*yij*(39*xij**6 + 24*yij**6 - 88*yij**4*zij**2 - 73*yij**2*zij**4 + 39*zij**6 + xij**4*(-73*yij**2 + 117*zij**2) + 
            xij**2*(-88*yij**4 - 146*yij**2*zij**2 + 117*zij**4)))/(35.*(xij**2 + yij**2 + zij**2)**5.5),
      (-180*b**6*zij*(xij**4 + 8*yij**4 - 12*yij**2*zij**2 + zij**4 + 2*xij**2*(-6*yij**2 + zij**2)) + 
         7*b**4*zij*(8*xij**6 + 68*yij**6 - 31*yij**4*zij**2 - 91*yij**2*zij**4 + 8*zij**6 + xij**4*(-91*yij**2 + 24*zij**2) + 
            xij**2*(-31*yij**4 - 182*yij**2*zij**2 + 24*zij**4)))/(35.*(xij**2 + yij**2 + zij**2)**5.5)],
     [(b**4*zij*(180*b**2*(6*xij**4 + 6*yij**4 + 5*yij**2*zij**2 - zij**4 + xij**2*(-51*yij**2 + 5*zij**2)) - 
           7*(52*xij**6 + (62*yij**2 - 13*zij**2)*(yij**2 + zij**2)**2 + xij**4*(-359*yij**2 + 91*zij**2) + 
              xij**2*(-349*yij**4 - 323*yij**2*zij**2 + 26*zij**4))))/(105.*(xij**2 + yij**2 + zij**2)**5.5),
      (b**4*xij*yij*zij*(-11*xij**4 + 24*yij**4 + 13*yij**2*zij**2 - 11*zij**4 + xij**2*(13*yij**2 - 22*zij**2) + 
           36*b**2*(xij**2 - 2*yij**2 + zij**2)))/(xij**2 + yij**2 + zij**2)**5.5,
      (b**4*xij*(-180*b**2*(xij**4 - 6*yij**4 + 51*yij**2*zij**2 - 6*zij**4 - 5*xij**2*(yij**2 + zij**2)) + 
           7*(8*xij**6 - 42*yij**6 + 384*yij**4*zij**2 + 369*yij**2*zij**4 - 57*zij**6 - xij**4*(26*yij**2 + 41*zij**2) + 
              xij**2*(-76*yij**4 + 343*yij**2*zij**2 - 106*zij**4))))/(105.*(xij**2 + yij**2 + zij**2)**5.5),
      (-540*b**6*zij*(xij**4 + 8*yij**4 - 12*yij**2*zij**2 + zij**4 + 2*xij**2*(-6*yij**2 + zij**2)) + 
         7*b**4*zij*(23*xij**6 + 208*yij**6 - 86*yij**4*zij**2 - 271*yij**2*zij**4 + 23*zij**6 + xij**4*(-271*yij**2 + 69*zij**2) + 
            xij**2*(-86*yij**4 - 542*yij**2*zij**2 + 69*zij**4)))/(105.*(xij**2 + yij**2 + zij**2)**5.5),
      (b**4*yij*(-180*b**2*(3*xij**4 - 4*yij**4 + 41*yij**2*zij**2 - 18*zij**4 - xij**2*(yij**2 + 15*zij**2)) + 
           7*(18*xij**6 - 32*yij**6 + 314*yij**4*zij**2 + 199*yij**2*zij**4 - 147*zij**6 + xij**4*(4*yij**2 - 111*zij**2) + 
              xij**2*(-46*yij**4 + 203*yij**2*zij**2 - 276*zij**4))))/(105.*(xij**2 + yij**2 + zij**2)**5.5)]])



def K3s3t(xij,yij,zij, b,eta):
    return np.array([[(9*b**5*(8*xij**4 - 24*xij**2*(yij**2 + zij**2) + 3*(yij**2 + zij**2)**2))/(50.*(xij**2 + yij**2 + zij**2)**4.5),
      (9*b**5*xij*yij*(4*xij**2 - 3*(yij**2 + zij**2)))/(10.*(xij**2 + yij**2 + zij**2)**4.5),
      (9*b**5*xij*zij*(4*xij**2 - 3*(yij**2 + zij**2)))/(10.*(xij**2 + yij**2 + zij**2)**4.5)],
     [(9*b**5*xij*yij*(4*xij**2 - 3*(yij**2 + zij**2)))/(10.*(xij**2 + yij**2 + zij**2)**4.5),
      (-9*b**5*(4*xij**4 + 4*yij**4 + 3*yij**2*zij**2 - zij**4 + 3*xij**2*(-9*yij**2 + zij**2)))/(50.*(xij**2 + yij**2 + zij**2)**4.5),
      (-9*b**5*yij*zij*(-6*xij**2 + yij**2 + zij**2))/(10.*(xij**2 + yij**2 + zij**2)**4.5)],
     [(9*b**5*xij*zij*(4*xij**2 - 3*(yij**2 + zij**2)))/(10.*(xij**2 + yij**2 + zij**2)**4.5),
      (-9*b**5*yij*zij*(-6*xij**2 + yij**2 + zij**2))/(10.*(xij**2 + yij**2 + zij**2)**4.5),
      (-9*b**5*(4*xij**4 - yij**4 + 3*yij**2*zij**2 + 4*zij**4 + 3*xij**2*(yij**2 - 9*zij**2)))/(50.*(xij**2 + yij**2 + zij**2)**4.5)],
     [(-9*b**5*(4*xij**4 + 4*yij**4 + 3*yij**2*zij**2 - zij**4 + 3*xij**2*(-9*yij**2 + zij**2)))/(50.*(xij**2 + yij**2 + zij**2)**4.5),
      (-9*b**5*xij*yij*(3*xij**2 - 4*yij**2 + 3*zij**2))/(10.*(xij**2 + yij**2 + zij**2)**4.5),
      (-9*b**5*xij*zij*(xij**2 - 6*yij**2 + zij**2))/(10.*(xij**2 + yij**2 + zij**2)**4.5)],
     [(-9*b**5*yij*zij*(-6*xij**2 + yij**2 + zij**2))/(10.*(xij**2 + yij**2 + zij**2)**4.5),
      (-9*b**5*xij*zij*(xij**2 - 6*yij**2 + zij**2))/(10.*(xij**2 + yij**2 + zij**2)**4.5),
      (-9*b**5*xij*yij*(xij**2 + yij**2 - 6*zij**2))/(10.*(xij**2 + yij**2 + zij**2)**4.5)],
     [(-9*b**5*xij*yij*(3*xij**2 - 4*yij**2 + 3*zij**2))/(10.*(xij**2 + yij**2 + zij**2)**4.5),
      (9*b**5*(3*xij**4 + 8*yij**4 - 24*yij**2*zij**2 + 3*zij**4 + 6*xij**2*(-4*yij**2 + zij**2)))/(50.*(xij**2 + yij**2 + zij**2)**4.5),
      (9*b**5*yij*zij*(-3*xij**2 + 4*yij**2 - 3*zij**2))/(10.*(xij**2 + yij**2 + zij**2)**4.5)],
     [(-9*b**5*xij*zij*(xij**2 - 6*yij**2 + zij**2))/(10.*(xij**2 + yij**2 + zij**2)**4.5),
      (9*b**5*yij*zij*(-3*xij**2 + 4*yij**2 - 3*zij**2))/(10.*(xij**2 + yij**2 + zij**2)**4.5),
      (9*b**5*(xij**4 - 4*yij**4 + 27*yij**2*zij**2 - 4*zij**4 - 3*xij**2*(yij**2 + zij**2)))/(50.*(xij**2 + yij**2 + zij**2)**4.5)]])



##
## matrix elements that are zero in free space: G^(3t,lsigma), K^(3t, lsigma), G^(la,3t), K^(la,3t)
##

def G3t1s(xij,yij,zij, b,eta):
    return np.zeros([3,3])

def G3t2a(xij,yij,zij, b,eta):
    return np.zeros([3,3])

def G3t2s(xij,yij,zij, b,eta):
    return np.zeros([3,5])

def G3t3t(xij,yij,zij, b,eta):
    return np.zeros([3,3])

def G3t3a(xij,yij,zij, b,eta):
    return np.zeros([3,5])

def G3t3s(xij,yij,zij, b,eta):
    return np.zeros([3,7])


def K3t2s(xij,yij,zij, b,eta):
    return np.zeros([3,5])

def K3t3t(xij,yij,zij, b,eta):
    return np.zeros([3,3])

def K3t3a(xij,yij,zij, b,eta):
    return np.zeros([3,5])

def K3t3s(xij,yij,zij, b,eta):
    return np.zeros([3,7])


def G2a3t(xij,yij,zij, b,eta):
    return np.zeros([3,3])

def G3a3t(xij,yij,zij, b,eta):
    return np.zeros([5,3])


def K2a3t(xij,yij,zij, b,eta):
    return np.zeros([3,3])

def K3a3t(xij,yij,zij, b,eta):
    return np.zeros([5,3])
