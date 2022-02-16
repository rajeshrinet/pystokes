##hard-coded matrix elements 

import numpy as np
PI = 3.14159265359


##
## define matrices eventually used in the direct-solver
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
    return np.block([[K2s2s(xij,yij,zij, b,eta), np.zeros([5,14])],
                     [np.zeros([14,5]), np.zeros([14,14])]])

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
    return np.block([K1s2s(xij,yij,zij, b,eta), K1s3t(xij,yij,zij, b,eta), np.zeros([3,14])])

def K2aH(xij,yij,zij, b,eta):
    return np.block([K2a2s(xij,yij,zij, b,eta), K2a3t(xij,yij,zij, b,eta), np.zeros([3,14])])

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
                        [K3t2s(xij,yij,zij, b,eta), K3t3t(xij,yij,zij, b,eta)]])
    return np.block([[nonzero, np.zeros([8,14])],
                     [np.zeros([14,8]), np.zeros([14,14])]])

def halfMinusKHH(xij,yij,zij, b,eta):
    return 0.5*np.identity(22) - KHH(xij,yij,zij, b,eta)


##
## Matrix elements
##



##
## FTS Stokesian dynamics matrix elements
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
    return np.array([[(-30*b**4*(8*xij**4 - 24*xij**2*(yij**2 + zij**2) + 3*(yij**2 + zij**2)**2) + 
         7*b**2*(16*xij**6 - 12*xij**4*(yij**2 + zij**2) - 27*xij**2*(yij**2 + zij**2)**2 + (yij**2 + zij**2)**3))/
       (168.*eta*PI*(xij**2 + yij**2 + zij**2)**4.5),(b**2*xij*yij*
         (-150*b**2*(4*xij**2 - 3*(yij**2 + zij**2)) + 7*(32*xij**4 + 19*xij**2*(yij**2 + zij**2) - 13*(yij**2 + zij**2)**2)))/
       (168.*eta*PI*(xij**2 + yij**2 + zij**2)**4.5),(b**2*xij*zij*
         (-150*b**2*(4*xij**2 - 3*(yij**2 + zij**2)) + 7*(32*xij**4 + 19*xij**2*(yij**2 + zij**2) - 13*(yij**2 + zij**2)**2)))/
       (168.*eta*PI*(xij**2 + yij**2 + zij**2)**4.5),(90*b**4*
          (4*xij**4 + 4*yij**4 + 3*yij**2*zij**2 - zij**4 + 3*xij**2*(-9*yij**2 + zij**2)) - 
         7*b**2*(20*xij**6 + (8*yij**2 - zij**2)*(yij**2 + zij**2)**2 + xij**4*(-87*yij**2 + 39*zij**2) - 
            9*xij**2*(11*yij**4 + 9*yij**2*zij**2 - 2*zij**4)))/(504.*eta*PI*(xij**2 + yij**2 + zij**2)**4.5),
      (b**2*yij*zij*(98*xij**4 - 50*b**2*(6*xij**2 - yij**2 - zij**2) + 91*xij**2*(yij**2 + zij**2) - 7*(yij**2 + zij**2)**2))/
       (56.*eta*PI*(xij**2 + yij**2 + zij**2)**4.5),(90*b**4*(4*xij**4 - yij**4 + 3*yij**2*zij**2 + 4*zij**4 + 3*xij**2*(yij**2 - 9*zij**2)) - 
         7*b**2*(20*xij**6 + xij**4*(39*yij**2 - 87*zij**2) - (yij**2 - 8*zij**2)*(yij**2 + zij**2)**2 + 
            9*xij**2*(2*yij**4 - 9*yij**2*zij**2 - 11*zij**4)))/(504.*eta*PI*(xij**2 + yij**2 + zij**2)**4.5),
      (b**2*xij*yij*(50*b**2*(3*xij**2 - 4*yij**2 + 3*zij**2) + 
           7*(-7*xij**4 + 8*yij**4 + yij**2*zij**2 - 7*zij**4 + xij**2*(yij**2 - 14*zij**2))))/(56.*eta*PI*(xij**2 + yij**2 + zij**2)**4.5),
      (b**2*xij*zij*(150*b**2*(xij**2 - 6*yij**2 + zij**2) - 
           7*(7*xij**4 - 38*yij**4 - 31*yij**2*zij**2 + 7*zij**4 + xij**2*(-31*yij**2 + 14*zij**2))))/
       (168.*eta*PI*(xij**2 + yij**2 + zij**2)**4.5),(b**2*xij*yij*
         (150*b**2*(xij**2 + yij**2 - 6*zij**2) - 7*(7*xij**4 + 7*yij**4 - 31*yij**2*zij**2 - 38*zij**4 + xij**2*(14*yij**2 - 31*zij**2))))/
       (168.*eta*PI*(xij**2 + yij**2 + zij**2)**4.5)],[(b**2*xij*yij*
         (-50*b**2*(4*xij**2 - 3*(yij**2 + zij**2)) + 7*(8*xij**4 + xij**2*(yij**2 + zij**2) - 7*(yij**2 + zij**2)**2)))/
       (56.*eta*PI*(xij**2 + yij**2 + zij**2)**4.5),(90*b**4*
          (4*xij**4 + 4*yij**4 + 3*yij**2*zij**2 - zij**4 + 3*xij**2*(-9*yij**2 + zij**2)) - 
         7*b**2*(8*xij**6 + (20*yij**2 - zij**2)*(yij**2 + zij**2)**2 + xij**4*(-99*yij**2 + 15*zij**2) + 
            xij**2*(-87*yij**4 - 81*yij**2*zij**2 + 6*zij**4)))/(504.*eta*PI*(xij**2 + yij**2 + zij**2)**4.5),
      (b**2*yij*zij*(-150*b**2*(6*xij**2 - yij**2 - zij**2) + 7*(38*xij**4 + 31*xij**2*(yij**2 + zij**2) - 7*(yij**2 + zij**2)**2)))/
       (168.*eta*PI*(xij**2 + yij**2 + zij**2)**4.5),(b**2*xij*yij*
         (150*b**2*(3*xij**2 - 4*yij**2 + 3*zij**2) - 7*
            (13*xij**4 - 32*yij**4 - 19*yij**2*zij**2 + 13*zij**4 + xij**2*(-19*yij**2 + 26*zij**2))))/
       (168.*eta*PI*(xij**2 + yij**2 + zij**2)**4.5),(b**2*xij*zij*
         (50*b**2*(xij**2 - 6*yij**2 + zij**2) - 7*(xij**4 - 14*yij**4 - 13*yij**2*zij**2 + zij**4 + xij**2*(-13*yij**2 + 2*zij**2))))/
       (56.*eta*PI*(xij**2 + yij**2 + zij**2)**4.5),(b**2*xij*yij*
         (150*b**2*(xij**2 + yij**2 - 6*zij**2) - 7*(7*xij**4 + 7*yij**4 - 31*yij**2*zij**2 - 38*zij**4 + xij**2*(14*yij**2 - 31*zij**2))))/
       (168.*eta*PI*(xij**2 + yij**2 + zij**2)**4.5),(-30*b**4*
          (3*xij**4 + 8*yij**4 - 24*yij**2*zij**2 + 3*zij**4 + 6*xij**2*(-4*yij**2 + zij**2)) + 
         7*b**2*(xij**6 + 16*yij**6 - 12*yij**4*zij**2 - 27*yij**2*zij**4 + zij**6 + 3*xij**4*(-9*yij**2 + zij**2) - 
            3*xij**2*(4*yij**4 + 18*yij**2*zij**2 - zij**4)))/(168.*eta*PI*(xij**2 + yij**2 + zij**2)**4.5),
      (b**2*yij*zij*(150*b**2*(3*xij**2 - 4*yij**2 + 3*zij**2) - 
           7*(13*xij**4 - 32*yij**4 - 19*yij**2*zij**2 + 13*zij**4 + xij**2*(-19*yij**2 + 26*zij**2))))/
       (168.*eta*PI*(xij**2 + yij**2 + zij**2)**4.5),(-90*b**4*
          (xij**4 - 4*yij**4 + 27*yij**2*zij**2 - 4*zij**4 - 3*xij**2*(yij**2 + zij**2)) + 
         7*b**2*(xij**6 - 20*yij**6 + 87*yij**4*zij**2 + 99*yij**2*zij**4 - 8*zij**6 - 6*xij**4*(3*yij**2 + zij**2) - 
            3*xij**2*(13*yij**4 - 27*yij**2*zij**2 + 5*zij**4)))/(504.*eta*PI*(xij**2 + yij**2 + zij**2)**4.5)],
     [(b**2*xij*zij*(-50*b**2*(4*xij**2 - 3*(yij**2 + zij**2)) + 7*(8*xij**4 + xij**2*(yij**2 + zij**2) - 7*(yij**2 + zij**2)**2)))/
       (56.*eta*PI*(xij**2 + yij**2 + zij**2)**4.5),(b**2*yij*zij*
         (-150*b**2*(6*xij**2 - yij**2 - zij**2) + 7*(38*xij**4 + 31*xij**2*(yij**2 + zij**2) - 7*(yij**2 + zij**2)**2)))/
       (168.*eta*PI*(xij**2 + yij**2 + zij**2)**4.5),(90*b**4*
          (4*xij**4 - yij**4 + 3*yij**2*zij**2 + 4*zij**4 + 3*xij**2*(yij**2 - 9*zij**2)) - 
         7*b**2*(8*xij**6 + 3*xij**4*(5*yij**2 - 33*zij**2) - (yij**2 - 20*zij**2)*(yij**2 + zij**2)**2 + 
            xij**2*(6*yij**4 - 81*yij**2*zij**2 - 87*zij**4)))/(504.*eta*PI*(xij**2 + yij**2 + zij**2)**4.5),
      (b**2*xij*zij*(150*b**2*(xij**2 - 6*yij**2 + zij**2) - 
           7*(7*xij**4 - 38*yij**4 - 31*yij**2*zij**2 + 7*zij**4 + xij**2*(-31*yij**2 + 14*zij**2))))/
       (168.*eta*PI*(xij**2 + yij**2 + zij**2)**4.5),(b**2*xij*yij*
         (50*b**2*(xij**2 + yij**2 - 6*zij**2) - 7*(xij**4 + yij**4 - 13*yij**2*zij**2 - 14*zij**4 + xij**2*(2*yij**2 - 13*zij**2))))/
       (56.*eta*PI*(xij**2 + yij**2 + zij**2)**4.5),(b**2*xij*zij*
         (150*b**2*(3*xij**2 + 3*yij**2 - 4*zij**2) - 7*
            (13*xij**4 + 13*yij**4 - 19*yij**2*zij**2 - 32*zij**4 + xij**2*(26*yij**2 - 19*zij**2))))/
       (168.*eta*PI*(xij**2 + yij**2 + zij**2)**4.5),(b**2*yij*zij*
         (50*b**2*(3*xij**2 - 4*yij**2 + 3*zij**2) + 7*(-7*xij**4 + 8*yij**4 + yij**2*zij**2 - 7*zij**4 + xij**2*(yij**2 - 14*zij**2))))/
       (56.*eta*PI*(xij**2 + yij**2 + zij**2)**4.5),(-90*b**4*(xij**4 - 4*yij**4 + 27*yij**2*zij**2 - 4*zij**4 - 3*xij**2*(yij**2 + zij**2)) + 
         7*b**2*(xij**6 - 8*yij**6 + 99*yij**4*zij**2 + 87*yij**2*zij**4 - 20*zij**6 - 6*xij**4*(yij**2 + 3*zij**2) - 
            3*xij**2*(5*yij**4 - 27*yij**2*zij**2 + 13*zij**4)))/(504.*eta*PI*(xij**2 + yij**2 + zij**2)**4.5),
      (b**2*yij*zij*(150*b**2*(3*xij**2 + 3*yij**2 - 4*zij**2) - 
           7*(13*xij**4 + 13*yij**4 - 19*yij**2*zij**2 - 32*zij**4 + xij**2*(26*yij**2 - 19*zij**2))))/
       (168.*eta*PI*(xij**2 + yij**2 + zij**2)**4.5)]])



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
      (5*b**3*xij*yij*zij)/(2.*eta*PI*(xij**2 + yij**2 + zij**2)**3.5),
      (-5*b**3*xij*(yij**2 - zij**2))/(4.*eta*PI*(xij**2 + yij**2 + zij**2)**3.5),
      (-5*b**3*xij*yij*zij)/(2.*eta*PI*(xij**2 + yij**2 + zij**2)**3.5),
      (-3*b**3*zij*(xij**2 - 4*yij**2 + zij**2))/(4.*eta*PI*(xij**2 + yij**2 + zij**2)**3.5),
      (b**3*yij*(xij**2 - 4*yij**2 + 11*zij**2))/(4.*eta*PI*(xij**2 + yij**2 + zij**2)**3.5),
      -0.25*(b**3*zij*(xij**2 + 11*yij**2 - 4*zij**2))/(eta*PI*(xij**2 + yij**2 + zij**2)**3.5)],
     [(3*b**3*zij*(-4*xij**2 + yij**2 + zij**2))/(4.*eta*PI*(xij**2 + yij**2 + zij**2)**3.5),
      (-5*b**3*xij*yij*zij)/(2.*eta*PI*(xij**2 + yij**2 + zij**2)**3.5),
      (b**3*xij*(4*xij**2 - yij**2 - 11*zij**2))/(4.*eta*PI*(xij**2 + yij**2 + zij**2)**3.5),
      (b**3*zij*(xij**2 - 4*yij**2 + zij**2))/(4.*eta*PI*(xij**2 + yij**2 + zij**2)**3.5),
      (5*b**3*yij*(xij**2 - zij**2))/(4.*eta*PI*(xij**2 + yij**2 + zij**2)**3.5),
      (b**3*zij*(11*xij**2 + yij**2 - 4*zij**2))/(4.*eta*PI*(xij**2 + yij**2 + zij**2)**3.5),0,
      -0.25*(b**3*xij*(xij**2 - 4*yij**2 + zij**2))/(eta*PI*(xij**2 + yij**2 + zij**2)**3.5),
      (5*b**3*xij*yij*zij)/(2.*eta*PI*(xij**2 + yij**2 + zij**2)**3.5)],
     [(-3*b**3*yij*(-4*xij**2 + yij**2 + zij**2))/(4.*eta*PI*(xij**2 + yij**2 + zij**2)**3.5),
      -0.25*(b**3*xij*(4*xij**2 - 11*yij**2 - zij**2))/(eta*PI*(xij**2 + yij**2 + zij**2)**3.5),
      (5*b**3*xij*yij*zij)/(2.*eta*PI*(xij**2 + yij**2 + zij**2)**3.5),
      -0.25*(b**3*yij*(11*xij**2 - 4*yij**2 + zij**2))/(eta*PI*(xij**2 + yij**2 + zij**2)**3.5),
      (-5*b**3*(xij**2 - yij**2)*zij)/(4.*eta*PI*(xij**2 + yij**2 + zij**2)**3.5),
      -0.25*(b**3*yij*(xij**2 + yij**2 - 4*zij**2))/(eta*PI*(xij**2 + yij**2 + zij**2)**3.5),
      (3*b**3*xij*(xij**2 - 4*yij**2 + zij**2))/(4.*eta*PI*(xij**2 + yij**2 + zij**2)**3.5),
      (-5*b**3*xij*yij*zij)/(2.*eta*PI*(xij**2 + yij**2 + zij**2)**3.5),
      (b**3*xij*(xij**2 + yij**2 - 4*zij**2))/(4.*eta*PI*(xij**2 + yij**2 + zij**2)**3.5)]])


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
    return np.array([[(-30*b**4*(8*xij**4 - 24*xij**2*(yij**2 + zij**2) + 3*(yij**2 + zij**2)**2) + 
         7*b**2*(16*xij**6 - 12*xij**4*(yij**2 + zij**2) - 27*xij**2*(yij**2 + zij**2)**2 + (yij**2 + zij**2)**3))/
       (168.*eta*PI*(xij**2 + yij**2 + zij**2)**4.5),(b**2*xij*yij*
         (-50*b**2*(4*xij**2 - 3*(yij**2 + zij**2)) + 7*(8*xij**4 + xij**2*(yij**2 + zij**2) - 7*(yij**2 + zij**2)**2)))/
       (56.*eta*PI*(xij**2 + yij**2 + zij**2)**4.5),(b**2*xij*zij*
         (-50*b**2*(4*xij**2 - 3*(yij**2 + zij**2)) + 7*(8*xij**4 + xij**2*(yij**2 + zij**2) - 7*(yij**2 + zij**2)**2)))/
       (56.*eta*PI*(xij**2 + yij**2 + zij**2)**4.5)],[(b**2*xij*yij*
         (-150*b**2*(4*xij**2 - 3*(yij**2 + zij**2)) + 7*(32*xij**4 + 19*xij**2*(yij**2 + zij**2) - 13*(yij**2 + zij**2)**2)))/
       (168.*eta*PI*(xij**2 + yij**2 + zij**2)**4.5),(90*b**4*
          (4*xij**4 + 4*yij**4 + 3*yij**2*zij**2 - zij**4 + 3*xij**2*(-9*yij**2 + zij**2)) - 
         7*b**2*(8*xij**6 + (20*yij**2 - zij**2)*(yij**2 + zij**2)**2 + xij**4*(-99*yij**2 + 15*zij**2) + 
            xij**2*(-87*yij**4 - 81*yij**2*zij**2 + 6*zij**4)))/(504.*eta*PI*(xij**2 + yij**2 + zij**2)**4.5),
      (b**2*yij*zij*(-150*b**2*(6*xij**2 - yij**2 - zij**2) + 7*(38*xij**4 + 31*xij**2*(yij**2 + zij**2) - 7*(yij**2 + zij**2)**2)))/
       (168.*eta*PI*(xij**2 + yij**2 + zij**2)**4.5)],[(b**2*xij*zij*
         (-150*b**2*(4*xij**2 - 3*(yij**2 + zij**2)) + 7*(32*xij**4 + 19*xij**2*(yij**2 + zij**2) - 13*(yij**2 + zij**2)**2)))/
       (168.*eta*PI*(xij**2 + yij**2 + zij**2)**4.5),(b**2*yij*zij*
         (-150*b**2*(6*xij**2 - yij**2 - zij**2) + 7*(38*xij**4 + 31*xij**2*(yij**2 + zij**2) - 7*(yij**2 + zij**2)**2)))/
       (168.*eta*PI*(xij**2 + yij**2 + zij**2)**4.5),(90*b**4*
          (4*xij**4 - yij**4 + 3*yij**2*zij**2 + 4*zij**4 + 3*xij**2*(yij**2 - 9*zij**2)) - 
         7*b**2*(8*xij**6 + 3*xij**4*(5*yij**2 - 33*zij**2) - (yij**2 - 20*zij**2)*(yij**2 + zij**2)**2 + 
            xij**2*(6*yij**4 - 81*yij**2*zij**2 - 87*zij**4)))/(504.*eta*PI*(xij**2 + yij**2 + zij**2)**4.5)],
     [(90*b**4*(4*xij**4 + 4*yij**4 + 3*yij**2*zij**2 - zij**4 + 3*xij**2*(-9*yij**2 + zij**2)) - 
         7*b**2*(20*xij**6 + (8*yij**2 - zij**2)*(yij**2 + zij**2)**2 + xij**4*(-87*yij**2 + 39*zij**2) - 
            9*xij**2*(11*yij**4 + 9*yij**2*zij**2 - 2*zij**4)))/(504.*eta*PI*(xij**2 + yij**2 + zij**2)**4.5),
      (b**2*xij*yij*(150*b**2*(3*xij**2 - 4*yij**2 + 3*zij**2) - 
           7*(13*xij**4 - 32*yij**4 - 19*yij**2*zij**2 + 13*zij**4 + xij**2*(-19*yij**2 + 26*zij**2))))/
       (168.*eta*PI*(xij**2 + yij**2 + zij**2)**4.5),(b**2*xij*zij*
         (150*b**2*(xij**2 - 6*yij**2 + zij**2) - 7*(7*xij**4 - 38*yij**4 - 31*yij**2*zij**2 + 7*zij**4 + xij**2*(-31*yij**2 + 14*zij**2))))/
       (168.*eta*PI*(xij**2 + yij**2 + zij**2)**4.5)],[(b**2*yij*zij*
         (98*xij**4 - 50*b**2*(6*xij**2 - yij**2 - zij**2) + 91*xij**2*(yij**2 + zij**2) - 7*(yij**2 + zij**2)**2))/
       (56.*eta*PI*(xij**2 + yij**2 + zij**2)**4.5),(b**2*xij*zij*
         (50*b**2*(xij**2 - 6*yij**2 + zij**2) - 7*(xij**4 - 14*yij**4 - 13*yij**2*zij**2 + zij**4 + xij**2*(-13*yij**2 + 2*zij**2))))/
       (56.*eta*PI*(xij**2 + yij**2 + zij**2)**4.5),(b**2*xij*yij*
         (50*b**2*(xij**2 + yij**2 - 6*zij**2) - 7*(xij**4 + yij**4 - 13*yij**2*zij**2 - 14*zij**4 + xij**2*(2*yij**2 - 13*zij**2))))/
       (56.*eta*PI*(xij**2 + yij**2 + zij**2)**4.5)],[(90*b**4*
          (4*xij**4 - yij**4 + 3*yij**2*zij**2 + 4*zij**4 + 3*xij**2*(yij**2 - 9*zij**2)) - 
         7*b**2*(20*xij**6 + xij**4*(39*yij**2 - 87*zij**2) - (yij**2 - 8*zij**2)*(yij**2 + zij**2)**2 + 
            9*xij**2*(2*yij**4 - 9*yij**2*zij**2 - 11*zij**4)))/(504.*eta*PI*(xij**2 + yij**2 + zij**2)**4.5),
      (b**2*xij*yij*(150*b**2*(xij**2 + yij**2 - 6*zij**2) - 
           7*(7*xij**4 + 7*yij**4 - 31*yij**2*zij**2 - 38*zij**4 + xij**2*(14*yij**2 - 31*zij**2))))/
       (168.*eta*PI*(xij**2 + yij**2 + zij**2)**4.5),(b**2*xij*zij*
         (150*b**2*(3*xij**2 + 3*yij**2 - 4*zij**2) - 7*
            (13*xij**4 + 13*yij**4 - 19*yij**2*zij**2 - 32*zij**4 + xij**2*(26*yij**2 - 19*zij**2))))/
       (168.*eta*PI*(xij**2 + yij**2 + zij**2)**4.5)],[(b**2*xij*yij*
         (50*b**2*(3*xij**2 - 4*yij**2 + 3*zij**2) + 7*(-7*xij**4 + 8*yij**4 + yij**2*zij**2 - 7*zij**4 + xij**2*(yij**2 - 14*zij**2))))/
       (56.*eta*PI*(xij**2 + yij**2 + zij**2)**4.5),(-30*b**4*
          (3*xij**4 + 8*yij**4 - 24*yij**2*zij**2 + 3*zij**4 + 6*xij**2*(-4*yij**2 + zij**2)) + 
         7*b**2*(xij**6 + 16*yij**6 - 12*yij**4*zij**2 - 27*yij**2*zij**4 + zij**6 + 3*xij**4*(-9*yij**2 + zij**2) - 
            3*xij**2*(4*yij**4 + 18*yij**2*zij**2 - zij**4)))/(168.*eta*PI*(xij**2 + yij**2 + zij**2)**4.5),
      (b**2*yij*zij*(50*b**2*(3*xij**2 - 4*yij**2 + 3*zij**2) + 
           7*(-7*xij**4 + 8*yij**4 + yij**2*zij**2 - 7*zij**4 + xij**2*(yij**2 - 14*zij**2))))/(56.*eta*PI*(xij**2 + yij**2 + zij**2)**4.5)],
     [(b**2*xij*zij*(150*b**2*(xij**2 - 6*yij**2 + zij**2) - 
           7*(7*xij**4 - 38*yij**4 - 31*yij**2*zij**2 + 7*zij**4 + xij**2*(-31*yij**2 + 14*zij**2))))/
       (168.*eta*PI*(xij**2 + yij**2 + zij**2)**4.5),(b**2*yij*zij*
         (150*b**2*(3*xij**2 - 4*yij**2 + 3*zij**2) - 7*
            (13*xij**4 - 32*yij**4 - 19*yij**2*zij**2 + 13*zij**4 + xij**2*(-19*yij**2 + 26*zij**2))))/
       (168.*eta*PI*(xij**2 + yij**2 + zij**2)**4.5),(-90*b**4*
          (xij**4 - 4*yij**4 + 27*yij**2*zij**2 - 4*zij**4 - 3*xij**2*(yij**2 + zij**2)) + 
         7*b**2*(xij**6 - 8*yij**6 + 99*yij**4*zij**2 + 87*yij**2*zij**4 - 20*zij**6 - 6*xij**4*(yij**2 + 3*zij**2) - 
            3*xij**2*(5*yij**4 - 27*yij**2*zij**2 + 13*zij**4)))/(504.*eta*PI*(xij**2 + yij**2 + zij**2)**4.5)],
     [(b**2*xij*yij*(150*b**2*(xij**2 + yij**2 - 6*zij**2) - 
           7*(7*xij**4 + 7*yij**4 - 31*yij**2*zij**2 - 38*zij**4 + xij**2*(14*yij**2 - 31*zij**2))))/
       (168.*eta*PI*(xij**2 + yij**2 + zij**2)**4.5),(-90*b**4*
          (xij**4 - 4*yij**4 + 27*yij**2*zij**2 - 4*zij**4 - 3*xij**2*(yij**2 + zij**2)) + 
         7*b**2*(xij**6 - 20*yij**6 + 87*yij**4*zij**2 + 99*yij**2*zij**4 - 8*zij**6 - 6*xij**4*(3*yij**2 + zij**2) - 
            3*xij**2*(13*yij**4 - 27*yij**2*zij**2 + 5*zij**4)))/(504.*eta*PI*(xij**2 + yij**2 + zij**2)**4.5),
      (b**2*yij*zij*(150*b**2*(3*xij**2 + 3*yij**2 - 4*zij**2) - 
           7*(13*xij**4 + 13*yij**4 - 19*yij**2*zij**2 - 32*zij**4 + xij**2*(26*yij**2 - 19*zij**2))))/
       (168.*eta*PI*(xij**2 + yij**2 + zij**2)**4.5)]])


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
     [(0.3978873577297384*b**3*xij*yij*zij)/(eta*(xij**2 + yij**2 + zij**2)**3.5),
      (-0.039788735772973836*b**3*zij*(11.*xij**2 + yij**2 - 4.*zij**2))/(eta*(xij**2 + yij**2 + zij**2)**3.5),
      (0.039788735772973836*b**3*yij*(xij**2 + yij**2 - 4.*zij**2))/(eta*(xij**2 + yij**2 + zij**2)**3.5)],
     [(0.1193662073189215*b**3*zij*(1.*xij**2 - 4.*yij**2 + 1.*zij**2))/(eta*(xij**2 + yij**2 + zij**2)**3.5),0.,
      (-0.1193662073189215*b**3*xij*(xij**2 - 4.*yij**2 + zij**2))/(eta*(xij**2 + yij**2 + zij**2)**3.5)],
     [(-0.039788735772973836*b**3*(1.*xij**2*yij - 4.*yij**3 + 11.*yij*zij**2))/(eta*(xij**2 + yij**2 + zij**2)**3.5),
      (0.039788735772973836*b**3*xij*(xij**2 - 4.*yij**2 + zij**2))/(eta*(xij**2 + yij**2 + zij**2)**3.5),
      (0.3978873577297384*b**3*xij*yij*zij)/(eta*(xij**2 + yij**2 + zij**2)**3.5)],
     [(0.039788735772973836*b**3*zij*(xij**2 + 11.*yij**2 - 4.*zij**2))/(eta*(xij**2 + yij**2 + zij**2)**3.5),
      (-0.3978873577297384*b**3*xij*yij*zij)/(eta*(xij**2 + yij**2 + zij**2)**3.5),
      (-0.039788735772973836*b**3*xij*(xij**2 + yij**2 - 4.*zij**2))/(eta*(xij**2 + yij**2 + zij**2)**3.5)]])




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
    return np.array([[(1.6370222718023522*b**3*xij*(-0.38888888888888884*xij**6 - 0.5104166666666666*yij**6 - 1.5312499999999998*yij**4*zij**2 - 
           1.5312499999999998*yij**2*zij**4 - 0.5104166666666666*zij**6 + xij**4*(1.2638888888888888*yij**2 + 1.2638888888888888*zij**2) + 
           xij**2*(1.142361111111111*yij**4 + 2.284722222222222*yij**2*zij**2 + 1.142361111111111*zij**4) + 
           b**2*(1.*xij**4 + 1.8749999999999998*yij**4 + 3.7499999999999996*yij**2*zij**2 + 1.8749999999999998*zij**4 + 
              xij**2*(-5.*yij**2 - 5.*zij**2))))/(eta*(xij**2 + yij**2 + zij**2)**5.5),
      (4.9110668154070565*b**3*yij*(-0.345679012345679*xij**6 - 0.03510802469135803*yij**6 - 0.10532407407407407*yij**4*zij**2 - 
           0.10532407407407407*yij**2*zij**4 - 0.03510802469135803*zij**6 + xij**4*(0.12422839506172839*yij**2 + 0.12422839506172839*zij**2) + 
           xij**2*(0.4347993827160494*yij**4 + 0.8695987654320988*yij**2*zij**2 + 0.4347993827160494*zij**4) + 
           b**2*(1.*xij**4 + 0.125*yij**4 + 0.25*yij**2*zij**2 + 0.125*zij**4 + xij**2*(-1.5*yij**2 - 1.5*zij**2))))/
       (eta*(xij**2 + yij**2 + zij**2)**5.5),(4.9110668154070565*b**3*zij*
         (-0.345679012345679*xij**6 - 0.03510802469135803*yij**6 - 0.10532407407407407*yij**4*zij**2 - 0.10532407407407407*yij**2*zij**4 - 
           0.03510802469135803*zij**6 + xij**4*(0.12422839506172839*yij**2 + 0.12422839506172839*zij**2) + 
           xij**2*(0.4347993827160494*yij**4 + 0.8695987654320988*yij**2*zij**2 + 0.4347993827160494*zij**4) + 
           b**2*(1.*xij**4 + 0.125*yij**4 + 0.25*yij**2*zij**2 + 0.125*zij**4 + xij**2*(-1.5*yij**2 - 1.5*zij**2))))/
       (eta*(xij**2 + yij**2 + zij**2)**5.5),(-0.818511135901176*b**3*xij*
         (-0.32407407407407407*xij**6 - 1.4583333333333333*yij**6 - 2.673611111111111*yij**4*zij**2 - 0.9722222222222222*yij**2*zij**4 + 
           0.24305555555555555*zij**6 + xij**4*(2.997685185185185*yij**2 - 0.40509259259259256*zij**2) + 
           xij**2*(1.8634259259259258*yij**4 + 2.025462962962963*yij**2*zij**2 + 0.16203703703703703*zij**4) + 
           b**2*(1.*xij**4 + 4.5*yij**4 + 3.7499999999999996*yij**2*zij**2 - 0.75*zij**4 + xij**2*(-10.25*yij**2 + 0.25*zij**2))))/
       (eta*(xij**2 + yij**2 + zij**2)**5.5),(8.59436692696235*b**3*xij*yij*zij*
         (-0.32407407407407407*xij**4 + 0.16203703703703703*yij**4 + 0.32407407407407407*yij**2*zij**2 + 0.16203703703703703*zij**4 + 
           b**2*(1.*xij**2 - 0.5*yij**2 - 0.5*zij**2) + xij**2*(-0.16203703703703703*yij**2 - 0.16203703703703703*zij**2)))/
       (eta*(xij**2 + yij**2 + zij**2)**5.5),(-0.818511135901176*b**3*xij*
         (-0.32407407407407407*xij**6 + 0.24305555555555555*yij**6 - 0.9722222222222222*yij**4*zij**2 - 2.673611111111111*yij**2*zij**4 - 
           1.4583333333333333*zij**6 + xij**4*(-0.40509259259259256*yij**2 + 2.997685185185185*zij**2) + 
           xij**2*(0.16203703703703703*yij**4 + 2.025462962962963*yij**2*zij**2 + 1.8634259259259258*zij**4) + 
           b**2*(1.*xij**4 - 0.75*yij**4 + 3.7499999999999996*yij**2*zij**2 + 4.5*zij**4 + xij**2*(0.25*yij**2 - 10.25*zij**2))))/
       (eta*(xij**2 + yij**2 + zij**2)**5.5),(-3.683300111555292*b**3*yij*
         (-0.30246913580246915*xij**6 - 0.08641975308641975*yij**6 - 0.09722222222222222*yij**4*zij**2 + 0.06481481481481481*yij**2*zij**4 + 
           0.07561728395061729*zij**6 + xij**4*(0.4429012345679012*yij**2 - 0.529320987654321*zij**2) + 
           xij**2*(0.6589506172839505*yij**4 + 0.5077160493827161*yij**2*zij**2 - 0.15123456790123457*zij**4) + 
           b**2*(1.*xij**4 + 0.22222222222222224*yij**4 + 0.05555555555555556*yij**2*zij**2 - 0.16666666666666666*zij**4 + 
              xij**2*(-2.2777777777777777*yij**2 + 0.8333333333333334*zij**2))))/(eta*(xij**2 + yij**2 + zij**2)**5.5),
      (-1.2277667038517641*b**3*zij*(-0.30246913580246915*xij**6 - 0.4104938271604938*yij**6 - 0.7453703703703703*yij**4*zij**2 - 
           0.25925925925925924*yij**2*zij**4 + 0.07561728395061729*zij**6 + xij**4*(2.3873456790123457*yij**2 - 0.529320987654321*zij**2) + 
           xij**2*(2.279320987654321*yij**4 + 2.128086419753086*yij**2*zij**2 - 0.15123456790123457*zij**4) + 
           b**2*(1.*xij**4 + 1.*yij**4 + 0.8333333333333334*yij**2*zij**2 - 0.16666666666666666*zij**4 + 
              xij**2*(-8.5*yij**2 + 0.8333333333333334*zij**2))))/(eta*(xij**2 + yij**2 + zij**2)**5.5),
      (-1.2277667038517641*b**3*yij*(-0.30246913580246915*xij**6 + 0.07561728395061729*yij**6 - 0.25925925925925924*yij**4*zij**2 - 
           0.7453703703703703*yij**2*zij**4 - 0.4104938271604938*zij**6 + xij**4*(-0.529320987654321*yij**2 + 2.3873456790123457*zij**2) + 
           xij**2*(-0.15123456790123457*yij**4 + 2.128086419753086*yij**2*zij**2 + 2.279320987654321*zij**4) + 
           b**2*(1.*xij**4 - 0.16666666666666666*yij**4 + 0.8333333333333334*yij**2*zij**2 + 1.*zij**4 + 
              xij**2*(0.8333333333333334*yij**2 - 8.5*zij**2))))/(eta*(xij**2 + yij**2 + zij**2)**5.5)],
     [(4.9110668154070565*b**3*yij*(-0.35648148148148145*xij**6 - 0.032407407407407406*yij**6 - 0.09722222222222221*yij**4*zij**2 - 
           0.09722222222222221*yij**2*zij**4 - 0.032407407407407406*zij**6 + 
           xij**4*(0.10532407407407408*yij**2 + 0.10532407407407408*zij**2) + 
           xij**2*(0.42939814814814814*yij**4 + 0.8587962962962963*yij**2*zij**2 + 0.42939814814814814*zij**4) + 
           b**2*(1.*xij**4 + 0.125*yij**4 + 0.25*yij**2*zij**2 + 0.125*zij**4 + xij**2*(-1.5*yij**2 - 1.5*zij**2))))/
       (eta*(xij**2 + yij**2 + zij**2)**5.5),(-0.818511135901176*b**3*xij*
         (-0.32407407407407407*xij**6 - 1.3773148148148147*yij**6 - 2.5925925925925926*yij**4*zij**2 - 1.0532407407407407*yij**2*zij**4 + 
           0.16203703703703703*zij**6 + xij**4*(3.0787037037037037*yij**2 - 0.4861111111111111*zij**2) + 
           xij**2*(2.025462962962963*yij**4 + 2.025462962962963*yij**2*zij**2) + 
           b**2*(1.*xij**4 + 4.5*yij**4 + 3.7499999999999996*yij**2*zij**2 - 0.75*zij**4 + xij**2*(-10.25*yij**2 + 0.25*zij**2))))/
       (eta*(xij**2 + yij**2 + zij**2)**5.5),(8.594366926962348*b**3*xij*yij*zij*
         (-0.3395061728395062*xij**4 + 0.14660493827160495*yij**4 + 0.2932098765432099*yij**2*zij**2 + 0.14660493827160495*zij**4 + 
           b**2*(1.*xij**2 - 0.5*yij**2 - 0.5*zij**2) + xij**2*(-0.19290123456790126*yij**2 - 0.19290123456790126*zij**2)))/
       (eta*(xij**2 + yij**2 + zij**2)**5.5),(-3.6833001115552917*b**3*yij*
         (-0.30606995884773663*xij**6 - 0.0720164609053498*yij**6 - 0.1080246913580247*yij**4*zij**2 + 0.0360082304526749*zij**6 + 
           xij**4*(0.45010288065843623*yij**2 - 0.5761316872427984*zij**2) + 
           xij**2*(0.6841563786008231*yij**4 + 0.45010288065843623*yij**2*zij**2 - 0.2340534979423868*zij**4) + 
           b**2*(1.*xij**4 + 0.2222222222222222*yij**4 + 0.05555555555555555*yij**2*zij**2 - 0.16666666666666666*zij**4 + 
              xij**2*(-2.2777777777777777*yij**2 + 0.8333333333333334*zij**2))))/(eta*(xij**2 + yij**2 + zij**2)**5.5),
      (-1.227766703851764*b**3*zij*(-0.2916666666666667*xij**6 - 0.2916666666666667*yij**6 - 0.5509259259259259*yij**4*zij**2 - 
           0.22685185185185186*yij**2*zij**4 + 0.032407407407407406*zij**6 + xij**4*(2.5277777777777777*yij**2 - 0.5509259259259259*zij**2) + 
           xij**2*(2.5277777777777777*yij**4 + 2.300925925925926*yij**2*zij**2 - 0.22685185185185186*zij**4) + 
           b**2*(1.*xij**4 + 1.*yij**4 + 0.8333333333333334*yij**2*zij**2 - 0.16666666666666666*zij**4 + 
              xij**2*(-8.5*yij**2 + 0.8333333333333334*zij**2))))/(eta*(xij**2 + yij**2 + zij**2)**5.5),
      (-1.227766703851764*b**3*yij*(-0.33487654320987653*xij**6 + 0.043209876543209874*yij**6 - 0.19444444444444445*yij**4*zij**2 - 
           0.5185185185185185*yij**2*zij**4 - 0.2808641975308642*zij**6 + xij**4*(-0.6265432098765432*yij**2 + 2.4521604938271606*zij**2) + 
           xij**2*(-0.24845679012345678*yij**4 + 2.257716049382716*yij**2*zij**2 + 2.506172839506173*zij**4) + 
           b**2*(1.*xij**4 - 0.16666666666666669*yij**4 + 0.8333333333333333*yij**2*zij**2 + 1.*zij**4 + 
              xij**2*(0.8333333333333333*yij**2 - 8.5*zij**2))))/(eta*(xij**2 + yij**2 + zij**2)**5.5),
      (0.6138833519258821*b**3*xij*(-0.25925925925925924*xij**6 - 2.8518518518518516*yij**6 + 0.8425925925925926*yij**4*zij**2 + 
           3.435185185185185*yij**2*zij**4 - 0.25925925925925924*zij**6 + xij**4*(3.435185185185185*yij**2 - 0.7777777777777778*zij**2) + 
           xij**2*(0.8425925925925926*yij**4 + 6.87037037037037*yij**2*zij**2 - 0.7777777777777778*zij**4) + 
           b**2*(1.*xij**4 + 8.*yij**4 - 12.*yij**2*zij**2 + 1.*zij**4 + xij**2*(-12.*yij**2 + 2.*zij**2))))/
       (eta*(xij**2 + yij**2 + zij**2)**5.5),(-4.297183463481174*b**3*xij*yij*zij*
         (-0.2932098765432099*xij**4 + 0.6790123456790124*yij**4 + 0.3858024691358025*yij**2*zij**2 - 0.2932098765432099*zij**4 + 
           xij**2*(0.3858024691358025*yij**2 - 0.5864197530864198*zij**2) + b**2*(1.*xij**2 - 2.*yij**2 + 1.*zij**2)))/
       (eta*(xij**2 + yij**2 + zij**2)**5.5),(0.20462778397529402*b**3*xij*
         (-0.25925925925925924*xij**6 + 2.009259259259259*yij**6 - 14.712962962962962*yij**4*zij**2 - 15.037037037037036*yij**2*zij**4 + 
           1.6851851851851851*zij**6 + xij**4*(1.4907407407407407*yij**2 + 1.1666666666666667*zij**2) + 
           xij**2*(3.759259259259259*yij**4 - 13.546296296296296*yij**2*zij**2 + 3.111111111111111*zij**4) + 
           b**2*(1.*xij**4 - 6.*yij**4 + 51.*yij**2*zij**2 - 6.*zij**4 + xij**2*(-5.*yij**2 - 5.*zij**2))))/
       (eta*(xij**2 + yij**2 + zij**2)**5.5)],[(4.9110668154070565*b**3*zij*
         (-0.35648148148148145*xij**6 - 0.032407407407407406*yij**6 - 0.09722222222222221*yij**4*zij**2 - 0.09722222222222221*yij**2*zij**4 - 
           0.032407407407407406*zij**6 + xij**4*(0.10532407407407408*yij**2 + 0.10532407407407408*zij**2) + 
           xij**2*(0.42939814814814814*yij**4 + 0.8587962962962963*yij**2*zij**2 + 0.42939814814814814*zij**4) + 
           b**2*(1.*xij**4 + 0.125*yij**4 + 0.25*yij**2*zij**2 + 0.125*zij**4 + xij**2*(-1.5*yij**2 - 1.5*zij**2))))/
       (eta*(xij**2 + yij**2 + zij**2)**5.5),(8.594366926962348*b**3*xij*yij*zij*
         (-0.3395061728395062*xij**4 + 0.14660493827160495*yij**4 + 0.2932098765432099*yij**2*zij**2 + 0.14660493827160495*zij**4 + 
           b**2*(1.*xij**2 - 0.5*yij**2 - 0.5*zij**2) + xij**2*(-0.19290123456790126*yij**2 - 0.19290123456790126*zij**2)))/
       (eta*(xij**2 + yij**2 + zij**2)**5.5),(-0.818511135901176*b**3*xij*
         (-0.32407407407407407*xij**6 + 0.16203703703703703*yij**6 - 1.0532407407407407*yij**4*zij**2 - 2.5925925925925926*yij**2*zij**4 - 
           1.3773148148148147*zij**6 + xij**4*(-0.4861111111111111*yij**2 + 3.0787037037037037*zij**2) + 
           xij**2*(2.025462962962963*yij**2*zij**2 + 2.025462962962963*zij**4) + 
           b**2*(1.*xij**4 - 0.75*yij**4 + 3.7499999999999996*yij**2*zij**2 + 4.5*zij**4 + xij**2*(0.25*yij**2 - 10.25*zij**2))))/
       (eta*(xij**2 + yij**2 + zij**2)**5.5),(-1.227766703851764*b**3*zij*
         (-0.33487654320987653*xij**6 - 0.2808641975308642*yij**6 - 0.5185185185185185*yij**4*zij**2 - 0.19444444444444445*yij**2*zij**4 + 
           0.043209876543209874*zij**6 + xij**4*(2.4521604938271606*yij**2 - 0.6265432098765432*zij**2) + 
           xij**2*(2.506172839506173*yij**4 + 2.257716049382716*yij**2*zij**2 - 0.24845679012345678*zij**4) + 
           b**2*(1.*xij**4 + 1.*yij**4 + 0.8333333333333333*yij**2*zij**2 - 0.16666666666666669*zij**4 + 
              xij**2*(-8.5*yij**2 + 0.8333333333333333*zij**2))))/(eta*(xij**2 + yij**2 + zij**2)**5.5),
      (-1.227766703851764*b**3*yij*(-0.2916666666666667*xij**6 + 0.032407407407407406*yij**6 - 0.22685185185185186*yij**4*zij**2 - 
           0.5509259259259259*yij**2*zij**4 - 0.2916666666666667*zij**6 + xij**4*(-0.5509259259259259*yij**2 + 2.5277777777777777*zij**2) + 
           xij**2*(-0.22685185185185186*yij**4 + 2.300925925925926*yij**2*zij**2 + 2.5277777777777777*zij**4) + 
           b**2*(1.*xij**4 - 0.16666666666666666*yij**4 + 0.8333333333333334*yij**2*zij**2 + 1.*zij**4 + 
              xij**2*(0.8333333333333334*yij**2 - 8.5*zij**2))))/(eta*(xij**2 + yij**2 + zij**2)**5.5),
      (-3.6833001115552917*b**3*zij*(-0.30606995884773663*xij**6 + 0.0360082304526749*yij**6 - 0.1080246913580247*yij**2*zij**4 - 
           0.0720164609053498*zij**6 + xij**4*(-0.5761316872427984*yij**2 + 0.45010288065843623*zij**2) + 
           xij**2*(-0.2340534979423868*yij**4 + 0.45010288065843623*yij**2*zij**2 + 0.6841563786008231*zij**4) + 
           b**2*(1.*xij**4 - 0.16666666666666666*yij**4 + 0.05555555555555555*yij**2*zij**2 + 0.2222222222222222*zij**4 + 
              xij**2*(0.8333333333333334*yij**2 - 2.2777777777777777*zij**2))))/(eta*(xij**2 + yij**2 + zij**2)**5.5),
      (-4.297183463481175*b**3*xij*yij*zij*(-0.32407407407407407*xij**4 + 0.6481481481481481*yij**4 + 0.32407407407407407*yij**2*zij**2 - 
           0.32407407407407407*zij**4 + xij**2*(0.32407407407407407*yij**2 - 0.6481481481481481*zij**2) + 
           b**2*(1.*xij**2 - 2.*yij**2 + 1.*zij**2)))/(eta*(xij**2 + yij**2 + zij**2)**5.5),
      (0.20462778397529402*b**3*xij*(-0.25925925925925924*xij**6 + 1.6851851851851851*yij**6 - 15.037037037037036*yij**4*zij**2 - 
           14.712962962962962*yij**2*zij**4 + 2.009259259259259*zij**6 + xij**4*(1.1666666666666667*yij**2 + 1.4907407407407407*zij**2) + 
           xij**2*(3.111111111111111*yij**4 - 13.546296296296296*yij**2*zij**2 + 3.759259259259259*zij**4) + 
           b**2*(1.*xij**4 - 6.*yij**4 + 51.*yij**2*zij**2 - 6.*zij**4 + xij**2*(-5.*yij**2 - 5.*zij**2))))/
       (eta*(xij**2 + yij**2 + zij**2)**5.5),(-4.297183463481174*b**3*xij*yij*zij*
         (-0.2932098765432099*xij**4 - 0.2932098765432099*yij**4 + 0.3858024691358025*yij**2*zij**2 + 0.6790123456790124*zij**4 + 
           b**2*(1.*xij**2 + 1.*yij**2 - 2.*zij**2) + xij**2*(-0.5864197530864198*yij**2 + 0.3858024691358025*zij**2)))/
       (eta*(xij**2 + yij**2 + zij**2)**5.5)],[(-0.8185111359011761*b**3*xij*
         (-0.38888888888888884*xij**6 - 1.361111111111111*yij**6 - 2.381944444444444*yij**4*zij**2 - 0.6805555555555555*yij**2*zij**4 + 
           0.34027777777777773*zij**6 + xij**4*(2.9652777777777777*yij**2 - 0.43749999999999994*zij**2) + 
           xij**2*(1.9930555555555554*yij**4 + 2.284722222222222*yij**2*zij**2 + 0.29166666666666663*zij**4) + 
           b**2*(1.*xij**4 + 4.5*yij**4 + 3.7499999999999996*yij**2*zij**2 - 0.75*zij**4 + xij**2*(-10.25*yij**2 + 0.25*zij**2))))/
       (eta*(xij**2 + yij**2 + zij**2)**5.5),(-3.6833001115552917*b**3*yij*
         (-0.32407407407407407*xij**6 - 0.07201646090534979*yij**6 - 0.09002057613168725*yij**4*zij**2 + 0.03600823045267489*yij**2*zij**4 + 
           0.05401234567901234*zij**6 + xij**4*(0.41409465020576125*yij**2 - 0.5941358024691358*zij**2) + 
           xij**2*(0.6661522633744855*yij**4 + 0.4501028806584362*yij**2*zij**2 - 0.21604938271604937*zij**4) + 
           b**2*(1.*xij**4 + 0.2222222222222222*yij**4 + 0.05555555555555555*yij**2*zij**2 - 0.16666666666666663*zij**4 + 
              xij**2*(-2.2777777777777777*yij**2 + 0.8333333333333334*zij**2))))/(eta*(xij**2 + yij**2 + zij**2)**5.5),
      (-1.2277667038517641*b**3*zij*(-0.4104938271604938*xij**6 - 0.3024691358024691*yij**6 - 0.529320987654321*yij**4*zij**2 - 
           0.15123456790123455*yij**2*zij**4 + 0.07561728395061727*zij**6 + xij**4*(2.279320987654321*yij**2 - 0.7453703703703703*zij**2) + 
           xij**2*(2.3873456790123457*yij**4 + 2.128086419753086*yij**2*zij**2 - 0.25925925925925924*zij**4) + 
           b**2*(1.*xij**4 + 1.*yij**4 + 0.8333333333333334*yij**2*zij**2 - 0.16666666666666666*zij**4 + 
              xij**2*(-8.5*yij**2 + 0.8333333333333334*zij**2))))/(eta*(xij**2 + yij**2 + zij**2)**5.5),
      (0.613883351925882*b**3*xij*(-0.2808641975308642*xij**6 - 2.7654320987654324*yij**6 + 0.9938271604938271*yij**4*zij**2 + 
           3.478395061728395*yij**2*zij**4 - 0.2808641975308642*zij**6 + xij**4*(3.478395061728395*yij**2 - 0.8425925925925927*zij**2) + 
           xij**2*(0.9938271604938271*yij**4 + 6.95679012345679*yij**2*zij**2 - 0.8425925925925927*zij**4) + 
           b**2*(1.*xij**4 + 8.*yij**4 - 12.*yij**2*zij**2 + 1.*zij**4 + xij**2*(-12.*yij**2 + 2.*zij**2))))/
       (eta*(xij**2 + yij**2 + zij**2)**5.5),(-4.297183463481175*b**3*xij*yij*zij*
         (-0.32407407407407407*xij**4 + 0.6481481481481481*yij**4 + 0.32407407407407407*yij**2*zij**2 - 0.32407407407407407*zij**4 + 
           xij**2*(0.32407407407407407*yij**2 - 0.6481481481481481*zij**2) + b**2*(1.*xij**2 - 2.*yij**2 + 1.*zij**2)))/
       (eta*(xij**2 + yij**2 + zij**2)**5.5),(0.204627783975294*b**3*xij*
         (-0.4537037037037038*xij**6 + 1.814814814814815*yij**6 - 14.324074074074076*yij**4*zij**2 - 13.675925925925927*yij**2*zij**4 + 
           2.4629629629629632*zij**6 + xij**4*(0.9074074074074076*yij**2 + 1.5555555555555556*zij**2) + 
           xij**2*(3.175925925925926*yij**4 - 12.768518518518519*yij**2*zij**2 + 4.472222222222223*zij**4) + 
           b**2*(1.*xij**4 - 6.000000000000001*yij**4 + 51.00000000000001*yij**2*zij**2 - 6.000000000000001*zij**4 + 
              xij**2*(-5.000000000000001*yij**2 - 5.000000000000001*zij**2))))/(eta*(xij**2 + yij**2 + zij**2)**5.5),
      (3.06941675962941*b**3*yij*(-0.2722222222222222*xij**6 - 0.2074074074074074*yij**6 + 0.674074074074074*yij**4*zij**2 + 
           0.6092592592592593*yij**2*zij**4 - 0.2722222222222222*zij**6 + xij**4*(0.6092592592592593*yij**2 - 0.8166666666666667*zij**2) + 
           xij**2*(0.674074074074074*yij**4 + 1.2185185185185186*yij**2*zij**2 - 0.8166666666666667*zij**4) + 
           b**2*(1.*xij**4 + 0.5333333333333333*yij**4 - 2.6666666666666665*yij**2*zij**2 + 1.*zij**4 + 
              xij**2*(-2.6666666666666665*yij**2 + 2.*zij**2))))/(eta*(xij**2 + yij**2 + zij**2)**5.5),
      (0.613883351925882*b**3*zij*(-0.2808641975308642*xij**6 - 2.7654320987654324*yij**6 + 0.9938271604938271*yij**4*zij**2 + 
           3.478395061728395*yij**2*zij**4 - 0.2808641975308642*zij**6 + xij**4*(3.478395061728395*yij**2 - 0.8425925925925927*zij**2) + 
           xij**2*(0.9938271604938271*yij**4 + 6.95679012345679*yij**2*zij**2 - 0.8425925925925927*zij**4) + 
           b**2*(1.*xij**4 + 8.*yij**4 - 12.*yij**2*zij**2 + 1.*zij**4 + xij**2*(-12.*yij**2 + 2.*zij**2))))/
       (eta*(xij**2 + yij**2 + zij**2)**5.5),(0.6138833519258821*b**3*yij*
         (-0.32407407407407407*xij**6 + 0.43209876543209874*yij**6 - 3.996913580246914*yij**4*zij**2 - 2.484567901234568*yij**2*zij**4 + 
           1.9444444444444444*zij**6 + xij**4*(-0.21604938271604937*yij**2 + 1.2962962962962963*zij**2) + 
           xij**2*(0.5401234567901234*yij**4 - 2.7006172839506175*yij**2*zij**2 + 3.5648148148148144*zij**4) + 
           b**2*(1.*xij**4 - 1.333333333333333*yij**4 + 13.666666666666666*yij**2*zij**2 - 6.*zij**4 + 
              xij**2*(-0.33333333333333326*yij**2 - 5.*zij**2))))/(eta*(xij**2 + yij**2 + zij**2)**5.5)],
     [(8.59436692696235*b**3*xij*yij*zij*(-0.32407407407407407*xij**4 + 0.16203703703703703*yij**4 + 0.32407407407407407*yij**2*zij**2 + 
           0.16203703703703703*zij**4 + b**2*(1.*xij**2 - 0.5*yij**2 - 0.5*zij**2) + 
           xij**2*(-0.16203703703703703*yij**2 - 0.16203703703703703*zij**2)))/(eta*(xij**2 + yij**2 + zij**2)**5.5),
      (-1.227766703851764*b**3*zij*(-0.2808641975308642*xij**6 - 0.33487654320987653*yij**6 - 0.6265432098765432*yij**4*zij**2 - 
           0.24845679012345678*yij**2*zij**4 + 0.04320987654320988*zij**6 + xij**4*(2.506172839506173*yij**2 - 0.5185185185185186*zij**2) + 
           xij**2*(2.4521604938271606*yij**4 + 2.257716049382716*yij**2*zij**2 - 0.19444444444444445*zij**4) + 
           b**2*(1.*xij**4 + 1.*yij**4 + 0.8333333333333334*yij**2*zij**2 - 0.16666666666666669*zij**4 + 
              xij**2*(-8.5*yij**2 + 0.8333333333333334*zij**2))))/(eta*(xij**2 + yij**2 + zij**2)**5.5),
      (-1.227766703851764*b**3*yij*(-0.2808641975308642*xij**6 + 0.04320987654320988*yij**6 - 0.24845679012345678*yij**4*zij**2 - 
           0.6265432098765432*yij**2*zij**4 - 0.33487654320987653*zij**6 + xij**4*(-0.5185185185185186*yij**2 + 2.506172839506173*zij**2) + 
           xij**2*(-0.19444444444444445*yij**4 + 2.257716049382716*yij**2*zij**2 + 2.4521604938271606*zij**4) + 
           b**2*(1.*xij**4 - 0.16666666666666669*yij**4 + 0.8333333333333334*yij**2*zij**2 + 1.*zij**4 + 
              xij**2*(0.8333333333333334*yij**2 - 8.5*zij**2))))/(eta*(xij**2 + yij**2 + zij**2)**5.5),
      (-4.297183463481174*b**3*xij*yij*zij*(-0.2932098765432099*xij**4 + 0.6790123456790124*yij**4 + 0.3858024691358025*yij**2*zij**2 - 
           0.2932098765432099*zij**4 + xij**2*(0.3858024691358025*yij**2 - 0.5864197530864198*zij**2) + 
           b**2*(1.*xij**2 - 2.*yij**2 + 1.*zij**2)))/(eta*(xij**2 + yij**2 + zij**2)**5.5),
      (0.20462778397529402*b**3*xij*(-0.19444444444444442*xij**6 + 1.7499999999999998*yij**6 - 15.166666666666666*yij**4*zij**2 - 
           15.166666666666666*yij**2*zij**4 + 1.7499999999999998*zij**6 + xij**4*(1.361111111111111*yij**2 + 1.361111111111111*zij**2) + 
           xij**2*(3.3055555555555554*yij**4 - 13.805555555555555*yij**2*zij**2 + 3.3055555555555554*zij**4) + 
           b**2*(1.*xij**4 - 6.*yij**4 + 50.99999999999999*yij**2*zij**2 - 6.*zij**4 + xij**2*(-5.*yij**2 - 5.*zij**2))))/
       (eta*(xij**2 + yij**2 + zij**2)**5.5),(-4.297183463481174*b**3*xij*yij*zij*
         (-0.2932098765432099*xij**4 - 0.2932098765432099*yij**4 + 0.3858024691358025*yij**2*zij**2 + 0.6790123456790124*zij**4 + 
           b**2*(1.*xij**2 + 1.*yij**2 - 2.*zij**2) + xij**2*(-0.5864197530864198*yij**2 + 0.3858024691358025*zij**2)))/
       (eta*(xij**2 + yij**2 + zij**2)**5.5),(0.6138833519258821*b**3*zij*
         (-0.25925925925925924*xij**6 - 2.8518518518518516*yij**6 + 0.8425925925925926*yij**4*zij**2 + 3.435185185185185*yij**2*zij**4 - 
           0.25925925925925924*zij**6 + xij**4*(3.435185185185185*yij**2 - 0.7777777777777778*zij**2) + 
           xij**2*(0.8425925925925926*yij**4 + 6.87037037037037*yij**2*zij**2 - 0.7777777777777778*zij**4) + 
           b**2*(1.*xij**4 + 8.*yij**4 - 12.*yij**2*zij**2 + 1.*zij**4 + xij**2*(-12.*yij**2 + 2.*zij**2))))/
       (eta*(xij**2 + yij**2 + zij**2)**5.5),(0.6138833519258821*b**3*yij*
         (-0.21604938271604937*xij**6 + 0.43209876543209874*yij**6 + 1.4043209876543208*xij**4*zij**2 - 4.104938271604938*yij**4*zij**2 - 
           2.700617283950617*yij**2*zij**4 + 1.8364197530864197*zij**6 + 
           xij**2*(0.6481481481481481*yij**4 - 2.700617283950617*yij**2*zij**2 + 3.45679012345679*zij**4) + 
           b**2*(1.*xij**4 - 1.3333333333333333*yij**4 + 13.666666666666666*yij**2*zij**2 - 6.*zij**4 + 
              xij**2*(-0.3333333333333333*yij**2 - 4.999999999999999*zij**2))))/(eta*(xij**2 + yij**2 + zij**2)**5.5),
      (0.6138833519258821*b**3*zij*(-0.21604938271604937*xij**6 + 1.4043209876543208*xij**4*yij**2 + 1.8364197530864197*yij**6 - 
           2.700617283950617*yij**4*zij**2 - 4.104938271604938*yij**2*zij**4 + 0.43209876543209874*zij**6 + 
           xij**2*(3.45679012345679*yij**4 - 2.700617283950617*yij**2*zij**2 + 0.6481481481481481*zij**4) + 
           b**2*(1.*xij**4 - 6.*yij**4 + 13.666666666666666*yij**2*zij**2 - 1.3333333333333333*zij**4 + 
              xij**2*(-4.999999999999999*yij**2 - 0.3333333333333333*zij**2))))/(eta*(xij**2 + yij**2 + zij**2)**5.5)]])



def G3a3s(xij,yij,zij, b,eta):
    return np.array([[0.,(-1.5915494309189535*b**4*xij*zij*(1.*xij**2 - 0.75*yij**2 - 0.75*zij**2))/(eta*(xij**2 + yij**2 + zij**2)**4.5),
      (1.5915494309189535*b**4*xij*yij*(1.*xij**2 - 0.75*yij**2 - 0.75*zij**2))/(eta*(xij**2 + yij**2 + zij**2)**4.5),
      (0.7957747154594768*b**4*yij*zij*(-6.*xij**2 + 1.*yij**2 + 1.*zij**2))/(eta*(xij**2 + yij**2 + zij**2)**4.5),
      (0.3978873577297384*b**4*(-1.*yij**4 + zij**4 + xij**2*(6.*yij**2 - 6.*zij**2)))/(eta*(xij**2 + yij**2 + zij**2)**4.5),
      (-0.7957747154594768*b**4*yij*zij*(-6.*xij**2 + 1.*yij**2 + 1.*zij**2))/(eta*(xij**2 + yij**2 + zij**2)**4.5),
      (1.1936620731892151*b**4*xij*zij*(xij**2 - 6.*yij**2 + zij**2))/(eta*(xij**2 + yij**2 + zij**2)**4.5),
      (-0.3978873577297384*b**4*xij*yij*(xij**2 - 6.*yij**2 + 15.*zij**2))/(eta*(xij**2 + yij**2 + zij**2)**4.5),
      (0.3978873577297384*b**4*xij*zij*(xij**2 + 15.*yij**2 - 6.*zij**2))/(eta*(xij**2 + yij**2 + zij**2)**4.5)],
     [(-1.7904931097838226*b**4*zij*(-1.3333333333333333*xij**3 + 1.*xij*yij**2 + 1.*xij*zij**2))/(eta*(xij**2 + yij**2 + zij**2)**4.5),
      (-0.1989436788648692*b**4*yij*zij*(-6.*xij**2 + yij**2 + zij**2))/(eta*(xij**2 + yij**2 + zij**2)**4.5),
      (-0.1989436788648692*b**4*(4.*xij**4 + yij**4 + 3.*yij**2*zij**2 + 2.*zij**4 + xij**2*(-9.*yij**2 - 15.*zij**2)))/
       (eta*(xij**2 + yij**2 + zij**2)**4.5),(0.1989436788648692*b**4*xij*zij*(xij**2 - 6.*yij**2 + zij**2))/
       (eta*(xij**2 + yij**2 + zij**2)**4.5),(-1.3926057520540842*b**4*(xij**3*yij - 1.*xij*yij**3))/(eta*(xij**2 + yij**2 + zij**2)**4.5),
      (-2.5862678252432993*b**4*xij*zij*(1.*xij**2 - 1.1538461538461537*yij**2 - 0.6153846153846154*zij**2))/
       (eta*(xij**2 + yij**2 + zij**2)**4.5),(1.7904931097838226*b**4*zij*(1.*xij**2*yij - 1.3333333333333333*yij**3 + 1.*yij*zij**2))/
       (eta*(xij**2 + yij**2 + zij**2)**4.5),(0.1989436788648692*b**4*
         (xij**4 + 4.*yij**4 - 15.*yij**2*zij**2 + 2.*zij**4 + xij**2*(-9.*yij**2 + 3.*zij**2)))/(eta*(xij**2 + yij**2 + zij**2)**4.5),
      (2.5862678252433*b**4*yij*zij*(-1.1538461538461537*xij**2 + 1.*yij**2 - 0.6153846153846153*zij**2))/
       (eta*(xij**2 + yij**2 + zij**2)**4.5)],[(1.7904931097838226*b**4*yij*(-1.3333333333333333*xij**3 + 1.*xij*yij**2 + 1.*xij*zij**2))/
       (eta*(xij**2 + yij**2 + zij**2)**4.5),(0.1989436788648692*b**4*
         (4.*xij**4 + 2.*yij**4 + 3.*yij**2*zij**2 + zij**4 + xij**2*(-15.*yij**2 - 9.*zij**2)))/(eta*(xij**2 + yij**2 + zij**2)**4.5),
      (0.1989436788648692*b**4*yij*zij*(-6.*xij**2 + yij**2 + zij**2))/(eta*(xij**2 + yij**2 + zij**2)**4.5),
      (2.5862678252432993*b**4*xij*yij*(1.*xij**2 - 0.6153846153846154*yij**2 - 1.1538461538461537*zij**2))/
       (eta*(xij**2 + yij**2 + zij**2)**4.5),(1.3926057520540842*b**4*(xij**3*zij - 1.*xij*zij**3))/(eta*(xij**2 + yij**2 + zij**2)**4.5),
      (-0.1989436788648692*b**4*xij*yij*(xij**2 + yij**2 - 6.*zij**2))/(eta*(xij**2 + yij**2 + zij**2)**4.5),
      (-0.5968310365946076*b**4*(xij**4 - 6.*xij**2*yij**2 + 6.*yij**2*zij**2 - 1.*zij**4))/(eta*(xij**2 + yij**2 + zij**2)**4.5),
      (1.5915494309189535*b**4*yij*zij*(1.875*xij**2 + 1.*yij**2 - 1.625*zij**2))/(eta*(xij**2 + yij**2 + zij**2)**4.5),
      (-0.1989436788648692*b**4*(xij**4 + 2.*yij**4 - 15.*yij**2*zij**2 + 4.*zij**4 + xij**2*(3.*yij**2 - 9.*zij**2)))/
       (eta*(xij**2 + yij**2 + zij**2)**4.5)],[(-1.1936620731892151*b**4*yij*zij*(-6.*xij**2 + 1.*yij**2 + 1.*zij**2))/
       (eta*(xij**2 + yij**2 + zij**2)**4.5),(-0.7957747154594768*b**4*xij*zij*(xij**2 - 6.*yij**2 + zij**2))/
       (eta*(xij**2 + yij**2 + zij**2)**4.5),(-2.3873241463784303*b**4*xij*yij*(1.*xij**2 - 0.16666666666666666*yij**2 - 2.5*zij**2))/
       (eta*(xij**2 + yij**2 + zij**2)**4.5),(1.5915494309189535*b**4*yij*zij*(-0.75*xij**2 + 1.*yij**2 - 0.75*zij**2))/
       (eta*(xij**2 + yij**2 + zij**2)**4.5),(0.3978873577297384*b**4*(xij**4 - 6.*xij**2*yij**2 + 6.*yij**2*zij**2 - 1.*zij**4))/
       (eta*(xij**2 + yij**2 + zij**2)**4.5),(-0.3978873577297384*b**4*yij*zij*(15.*xij**2 + yij**2 - 6.*zij**2))/
       (eta*(xij**2 + yij**2 + zij**2)**4.5),0.,(1.1936620731892151*b**4*xij*yij*(1.*xij**2 - 1.3333333333333333*yij**2 + 1.*zij**2))/
       (eta*(xij**2 + yij**2 + zij**2)**4.5),(0.7957747154594768*b**4*xij*zij*(xij**2 - 6.*yij**2 + zij**2))/
       (eta*(xij**2 + yij**2 + zij**2)**4.5)],[(-0.5968310365946076*b**4*(-1.*yij**4 + zij**4 + xij**2*(6.*yij**2 - 6.*zij**2)))/
       (eta*(xij**2 + yij**2 + zij**2)**4.5),(1.5915494309189535*b**4*xij*yij*(1.*xij**2 - 1.625*yij**2 + 1.875*zij**2))/
       (eta*(xij**2 + yij**2 + zij**2)**4.5),(-1.5915494309189535*b**4*xij*zij*(1.*xij**2 + 1.875*yij**2 - 1.625*zij**2))/
       (eta*(xij**2 + yij**2 + zij**2)**4.5),(-0.1989436788648692*b**4*
         (2.*xij**4 + 4.*yij**4 - 9.*yij**2*zij**2 + zij**4 + xij**2*(-15.*yij**2 + 3.*zij**2)))/(eta*(xij**2 + yij**2 + zij**2)**4.5),
      (-1.3926057520540842*b**4*(yij**3*zij - 1.*yij*zij**3))/(eta*(xij**2 + yij**2 + zij**2)**4.5),
      (0.1989436788648692*b**4*(2.*xij**4 + yij**4 - 9.*yij**2*zij**2 + 4.*zij**4 + xij**2*(3.*yij**2 - 15.*zij**2)))/
       (eta*(xij**2 + yij**2 + zij**2)**4.5),(-1.7904931097838226*b**4*xij*(1.*xij**2*yij - 1.3333333333333333*yij**3 + 1.*yij*zij**2))/
       (eta*(xij**2 + yij**2 + zij**2)**4.5),(-0.1989436788648692*b**4*xij*zij*(xij**2 - 6.*yij**2 + zij**2))/
       (eta*(xij**2 + yij**2 + zij**2)**4.5),(0.1989436788648692*b**4*xij*yij*(xij**2 + yij**2 - 6.*zij**2))/
       (eta*(xij**2 + yij**2 + zij**2)**4.5)]])



def G3s2s(xij,yij,zij, b,eta):
    return np.array([[0.,0.,0.,0.,0.],[0.,0.,0.,0.,0.],[0.,0.,0.,0.,0.],[0.,0.,0.,0.,0.],[0.,0.,0.,0.,0.],[0.,0.,0.,0.,0.],
     [0.,0.,0.,0.,0.],[0.,0.,0.,0.,0.],[0.,0.,0.,0.,0.]])



def G3s3a(xij,yij,zij, b,eta):
    return np.array([[0,(5*b**4*xij*zij*(4*xij**2 - 3*(yij**2 + zij**2)))/(6.*eta*PI*(xij**2 + yij**2 + zij**2)**4.5),
      (-5*b**4*xij*yij*(4*xij**2 - 3*(yij**2 + zij**2)))/(6.*eta*PI*(xij**2 + yij**2 + zij**2)**4.5),
      (-5*b**4*yij*zij*(-6*xij**2 + yij**2 + zij**2))/(2.*eta*PI*(xij**2 + yij**2 + zij**2)**4.5),
      -0.16666666666666666*(b**4*(4*xij**4 - 6*yij**4 + 3*yij**2*zij**2 + 9*zij**4 + xij**2*(33*yij**2 - 57*zij**2)))/
        (eta*PI*(xij**2 + yij**2 + zij**2)**4.5)],[(-5*b**4*xij*zij*(4*xij**2 - 3*(yij**2 + zij**2)))/
       (6.*eta*PI*(xij**2 + yij**2 + zij**2)**4.5),0,(b**4*(8*xij**4 - 24*xij**2*(yij**2 + zij**2) + 3*(yij**2 + zij**2)**2))/
       (6.*eta*PI*(xij**2 + yij**2 + zij**2)**4.5),(-5*b**4*xij*zij*(xij**2 - 6*yij**2 + zij**2))/(3.*eta*PI*(xij**2 + yij**2 + zij**2)**4.5),
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
       (6.*eta*PI*(xij**2 + yij**2 + zij**2)**4.5),(-5*b**4*xij*yij*(3*xij**2 - 4*yij**2 + 3*zij**2))/
       (6.*eta*PI*(xij**2 + yij**2 + zij**2)**4.5),(5*b**4*xij*zij*(3*xij**2 + 3*yij**2 - 4*zij**2))/
       (6.*eta*PI*(xij**2 + yij**2 + zij**2)**4.5),(5*b**4*(xij**2 - zij**2)*(xij**2 - 6*yij**2 + zij**2))/
       (6.*eta*PI*(xij**2 + yij**2 + zij**2)**4.5),(-5*b**4*yij*zij*(3*xij**2 + 3*yij**2 - 4*zij**2))/
       (6.*eta*PI*(xij**2 + yij**2 + zij**2)**4.5)],[(-5*b**4*yij*zij*(-6*xij**2 + yij**2 + zij**2))/
       (3.*eta*PI*(xij**2 + yij**2 + zij**2)**4.5),(-5*b**4*xij*zij*(5*xij**2 - 9*yij**2 - 2*zij**2))/
       (6.*eta*PI*(xij**2 + yij**2 + zij**2)**4.5),(-5*b**4*xij*yij*(xij**2 + yij**2 - 6*zij**2))/(6.*eta*PI*(xij**2 + yij**2 + zij**2)**4.5),
      (-5*b**4*yij*zij*(15*xij**2 + yij**2 - 6*zij**2))/(6.*eta*PI*(xij**2 + yij**2 + zij**2)**4.5),
      (b**4*(7*xij**4 + 2*yij**4 - 21*yij**2*zij**2 + 12*zij**4 + xij**2*(9*yij**2 - 51*zij**2)))/(6.*eta*PI*(xij**2 + yij**2 + zij**2)**4.5)],
     [(5*b**4*xij*zij*(xij**2 - 6*yij**2 + zij**2))/(2.*eta*PI*(xij**2 + yij**2 + zij**2)**4.5),
      (5*b**4*yij*zij*(3*xij**2 - 4*yij**2 + 3*zij**2))/(3.*eta*PI*(xij**2 + yij**2 + zij**2)**4.5),
      (b**4*(-6*xij**4 + 4*yij**4 - 57*yij**2*zij**2 + 9*zij**4 + 3*xij**2*(11*yij**2 + zij**2)))/(6.*eta*PI*(xij**2 + yij**2 + zij**2)**4.5),
      0,(-5*b**4*xij*yij*(3*xij**2 - 4*yij**2 + 3*zij**2))/(6.*eta*PI*(xij**2 + yij**2 + zij**2)**4.5)],
     [(-5*b**4*xij*yij*(xij**2 - 6*yij**2 + 15*zij**2))/(6.*eta*PI*(xij**2 + yij**2 + zij**2)**4.5),
      (b**4*(2*xij**4 + 12*yij**4 - 51*yij**2*zij**2 + 7*zij**4 + xij**2*(-21*yij**2 + 9*zij**2)))/(6.*eta*PI*(xij**2 + yij**2 + zij**2)**4.5),
      (5*b**4*yij*zij*(3*xij**2 + 3*yij**2 - 4*zij**2))/(3.*eta*PI*(xij**2 + yij**2 + zij**2)**4.5),
      (5*b**4*xij*yij*(3*xij**2 - 4*yij**2 + 3*zij**2))/(6.*eta*PI*(xij**2 + yij**2 + zij**2)**4.5),0],
     [(5*b**4*xij*zij*(xij**2 + 15*yij**2 - 6*zij**2))/(6.*eta*PI*(xij**2 + yij**2 + zij**2)**4.5),
      (5*b**4*yij*zij*(-3*xij**2 + 4*yij**2 - 3*zij**2))/(3.*eta*PI*(xij**2 + yij**2 + zij**2)**4.5),
      -0.16666666666666666*(b**4*(2*xij**4 + 7*yij**4 - 51*yij**2*zij**2 + 12*zij**4 + 3*xij**2*(3*yij**2 - 7*zij**2)))/
        (eta*PI*(xij**2 + yij**2 + zij**2)**4.5),(5*b**4*xij*zij*(xij**2 - 6*yij**2 + zij**2))/(3.*eta*PI*(xij**2 + yij**2 + zij**2)**4.5),
      (5*b**4*xij*yij*(xij**2 + yij**2 - 6*zij**2))/(6.*eta*PI*(xij**2 + yij**2 + zij**2)**4.5)]])



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
     [(-3*b**4*(4*xij**4 + 4*yij**4 + 3*yij**2*zij**2 - zij**4 + 3*xij**2*(-9*yij**2 + zij**2)))/
       (20.*eta*PI*(xij**2 + yij**2 + zij**2)**4.5),(-3*b**4*xij*yij*(3*xij**2 - 4*yij**2 + 3*zij**2))/
       (4.*eta*PI*(xij**2 + yij**2 + zij**2)**4.5),(-3*b**4*xij*zij*(xij**2 - 6*yij**2 + zij**2))/(4.*eta*PI*(xij**2 + yij**2 + zij**2)**4.5)],
     [(-3*b**4*yij*zij*(-6*xij**2 + yij**2 + zij**2))/(4.*eta*PI*(xij**2 + yij**2 + zij**2)**4.5),
      (-3*b**4*xij*zij*(xij**2 - 6*yij**2 + zij**2))/(4.*eta*PI*(xij**2 + yij**2 + zij**2)**4.5),
      (-3*b**4*xij*yij*(xij**2 + yij**2 - 6*zij**2))/(4.*eta*PI*(xij**2 + yij**2 + zij**2)**4.5)],
     [(-3*b**4*(4*xij**4 - yij**4 + 3*yij**2*zij**2 + 4*zij**4 + 3*xij**2*(yij**2 - 9*zij**2)))/
       (20.*eta*PI*(xij**2 + yij**2 + zij**2)**4.5),(-3*b**4*xij*yij*(xij**2 + yij**2 - 6*zij**2))/(4.*eta*PI*(xij**2 + yij**2 + zij**2)**4.5),
      (-3*b**4*xij*zij*(3*xij**2 + 3*yij**2 - 4*zij**2))/(4.*eta*PI*(xij**2 + yij**2 + zij**2)**4.5)],
     [(-3*b**4*xij*yij*(3*xij**2 - 4*yij**2 + 3*zij**2))/(4.*eta*PI*(xij**2 + yij**2 + zij**2)**4.5),
      (3*b**4*(3*xij**4 + 8*yij**4 - 24*yij**2*zij**2 + 3*zij**4 + 6*xij**2*(-4*yij**2 + zij**2)))/
       (20.*eta*PI*(xij**2 + yij**2 + zij**2)**4.5),(3*b**4*yij*zij*(-3*xij**2 + 4*yij**2 - 3*zij**2))/
       (4.*eta*PI*(xij**2 + yij**2 + zij**2)**4.5)],[(-3*b**4*xij*zij*(xij**2 - 6*yij**2 + zij**2))/
       (4.*eta*PI*(xij**2 + yij**2 + zij**2)**4.5),(3*b**4*yij*zij*(-3*xij**2 + 4*yij**2 - 3*zij**2))/
       (4.*eta*PI*(xij**2 + yij**2 + zij**2)**4.5),(3*b**4*(xij**4 - 4*yij**4 + 27*yij**2*zij**2 - 4*zij**4 - 3*xij**2*(yij**2 + zij**2)))/
       (20.*eta*PI*(xij**2 + yij**2 + zij**2)**4.5)],[(-3*b**4*xij*yij*(xij**2 + yij**2 - 6*zij**2))/
       (4.*eta*PI*(xij**2 + yij**2 + zij**2)**4.5),(3*b**4*(xij**4 - 4*yij**4 + 27*yij**2*zij**2 - 4*zij**4 - 3*xij**2*(yij**2 + zij**2)))/
       (20.*eta*PI*(xij**2 + yij**2 + zij**2)**4.5),(-3*b**4*yij*zij*(3*xij**2 + 3*yij**2 - 4*zij**2))/
       (4.*eta*PI*(xij**2 + yij**2 + zij**2)**4.5)]])



def G3s3s(xij,yij,zij, b,eta):
    return np.array([[(-90*b**6*(16*xij**6 - 120*xij**4*(yij**2 + zij**2) + 90*xij**2*(yij**2 + zij**2)**2 - 5*(yij**2 + zij**2)**3) + 
         35*b**4*(16*xij**8 - 88*xij**6*(yij**2 + zij**2) - 38*xij**4*(yij**2 + zij**2)**2 + 63*xij**2*(yij**2 + zij**2)**3 - 
            3*(yij**2 + zij**2)**4))/(56.*eta*PI*(xij**2 + yij**2 + zij**2)**6.5),
      (-5*b**4*xij*yij*(-152*xij**6 + 188*xij**4*(yij**2 + zij**2) + 265*xij**2*(yij**2 + zij**2)**2 - 75*(yij**2 + zij**2)**3 + 
           54*b**2*(8*xij**4 - 20*xij**2*(yij**2 + zij**2) + 5*(yij**2 + zij**2)**2)))/(24.*eta*PI*(xij**2 + yij**2 + zij**2)**6.5),
      (-5*b**4*xij*zij*(-152*xij**6 + 188*xij**4*(yij**2 + zij**2) + 265*xij**2*(yij**2 + zij**2)**2 - 75*(yij**2 + zij**2)**3 + 
           54*b**2*(8*xij**4 - 20*xij**2*(yij**2 + zij**2) + 5*(yij**2 + zij**2)**2)))/(24.*eta*PI*(xij**2 + yij**2 + zij**2)**6.5),
      (270*b**6*(8*xij**6 - (6*yij**2 - zij**2)*(yij**2 + zij**2)**2 - 4*xij**4*(29*yij**2 + zij**2) + 
            xij**2*(101*yij**4 + 90*yij**2*zij**2 - 11*zij**4)) - 
         7*b**4*(104*xij**8 + 3*xij**2*(377*yij**2 - 38*zij**2)*(yij**2 + zij**2)**2 - 3*(22*yij**2 - 3*zij**2)*(yij**2 + zij**2)**3 + 
            xij**6*(-1364*yij**2 + 76*zij**2) - xij**4*(271*yij**4 + 422*yij**2*zij**2 + 151*zij**4)))/
       (168.*eta*PI*(xij**2 + yij**2 + zij**2)**6.5),(5*b**4*yij*zij*
         (96*xij**6 + 8*xij**4*(yij**2 + zij**2) - 83*xij**2*(yij**2 + zij**2)**2 + 5*(yij**2 + zij**2)**3 - 
           18*b**2*(16*xij**4 - 16*xij**2*(yij**2 + zij**2) + (yij**2 + zij**2)**2)))/(8.*eta*PI*(xij**2 + yij**2 + zij**2)**6.5),
      (270*b**6*(8*xij**6 + (yij**2 - 6*zij**2)*(yij**2 + zij**2)**2 - 4*xij**4*(yij**2 + 29*zij**2) + 
            xij**2*(-11*yij**4 + 90*yij**2*zij**2 + 101*zij**4)) - 
         7*b**4*(104*xij**8 + 4*xij**6*(19*yij**2 - 341*zij**2) - 3*xij**2*(38*yij**2 - 377*zij**2)*(yij**2 + zij**2)**2 + 
            3*(3*yij**2 - 22*zij**2)*(yij**2 + zij**2)**3 - xij**4*(151*yij**4 + 422*yij**2*zij**2 + 271*zij**4)))/
       (168.*eta*PI*(xij**2 + yij**2 + zij**2)**6.5),(5*b**4*xij*yij*
         (-34*xij**6 + xij**4*(87*yij**2 - 53*zij**2) - (34*yij**2 - 15*zij**2)*(yij**2 + zij**2)**2 + 
           xij**2*(87*yij**4 + 83*yij**2*zij**2 - 4*zij**4) + 
           54*b**2*(2*xij**4 + 2*yij**4 + yij**2*zij**2 - zij**4 + xij**2*(-7*yij**2 + zij**2))))/(8.*eta*PI*(xij**2 + yij**2 + zij**2)**6.5),
      (5*b**4*xij*zij*(-34*xij**6 + xij**4*(367*yij**2 - 53*zij**2) - 3*(44*yij**2 - 5*zij**2)*(yij**2 + zij**2)**2 + 
           xij**2*(269*yij**4 + 265*yij**2*zij**2 - 4*zij**4) + 
           54*b**2*(2*xij**4 + 8*yij**4 + 7*yij**2*zij**2 - zij**4 + xij**2*(-23*yij**2 + zij**2))))/
       (24.*eta*PI*(xij**2 + yij**2 + zij**2)**6.5),(5*b**4*xij*yij*
         (-34*xij**6 + 3*(5*yij**2 - 44*zij**2)*(yij**2 + zij**2)**2 + xij**4*(-53*yij**2 + 367*zij**2) + 
           xij**2*(-4*yij**4 + 265*yij**2*zij**2 + 269*zij**4) + 
           54*b**2*(2*xij**4 - yij**4 + 7*yij**2*zij**2 + 8*zij**4 + xij**2*(yij**2 - 23*zij**2))))/
       (24.*eta*PI*(xij**2 + yij**2 + zij**2)**6.5)],[(-5*b**4*xij*yij*
         (-152*xij**6 + 188*xij**4*(yij**2 + zij**2) + 265*xij**2*(yij**2 + zij**2)**2 - 75*(yij**2 + zij**2)**3 + 
           54*b**2*(8*xij**4 - 20*xij**2*(yij**2 + zij**2) + 5*(yij**2 + zij**2)**2)))/(24.*eta*PI*(xij**2 + yij**2 + zij**2)**6.5),
      (270*b**6*(8*xij**6 - (6*yij**2 - zij**2)*(yij**2 + zij**2)**2 - 4*xij**4*(29*yij**2 + zij**2) + 
            xij**2*(101*yij**4 + 90*yij**2*zij**2 - 11*zij**4)) - 
         7*b**4*(104*xij**8 + 3*xij**2*(377*yij**2 - 38*zij**2)*(yij**2 + zij**2)**2 - 3*(22*yij**2 - 3*zij**2)*(yij**2 + zij**2)**3 + 
            xij**6*(-1364*yij**2 + 76*zij**2) - xij**4*(271*yij**4 + 422*yij**2*zij**2 + 151*zij**4)))/
       (168.*eta*PI*(xij**2 + yij**2 + zij**2)**6.5),(5*b**4*yij*zij*
         (96*xij**6 + 8*xij**4*(yij**2 + zij**2) - 83*xij**2*(yij**2 + zij**2)**2 + 5*(yij**2 + zij**2)**3 - 
           18*b**2*(16*xij**4 - 16*xij**2*(yij**2 + zij**2) + (yij**2 + zij**2)**2)))/(8.*eta*PI*(xij**2 + yij**2 + zij**2)**6.5),
      (5*b**4*xij*yij*(-34*xij**6 + xij**4*(87*yij**2 - 53*zij**2) - (34*yij**2 - 15*zij**2)*(yij**2 + zij**2)**2 + 
           xij**2*(87*yij**4 + 83*yij**2*zij**2 - 4*zij**4) + 
           54*b**2*(2*xij**4 + 2*yij**4 + yij**2*zij**2 - zij**4 + xij**2*(-7*yij**2 + zij**2))))/(8.*eta*PI*(xij**2 + yij**2 + zij**2)**6.5),
      (5*b**4*xij*zij*(-34*xij**6 + xij**4*(367*yij**2 - 53*zij**2) - 3*(44*yij**2 - 5*zij**2)*(yij**2 + zij**2)**2 + 
           xij**2*(269*yij**4 + 265*yij**2*zij**2 - 4*zij**4) + 
           54*b**2*(2*xij**4 + 8*yij**4 + 7*yij**2*zij**2 - zij**4 + xij**2*(-23*yij**2 + zij**2))))/
       (24.*eta*PI*(xij**2 + yij**2 + zij**2)**6.5),(5*b**4*xij*yij*
         (-34*xij**6 + 3*(5*yij**2 - 44*zij**2)*(yij**2 + zij**2)**2 + xij**4*(-53*yij**2 + 367*zij**2) + 
           xij**2*(-4*yij**4 + 265*yij**2*zij**2 + 269*zij**4) + 
           54*b**2*(2*xij**4 - yij**4 + 7*yij**2*zij**2 + 8*zij**4 + xij**2*(yij**2 - 23*zij**2))))/
       (24.*eta*PI*(xij**2 + yij**2 + zij**2)**6.5),(-270*b**6*
          (6*xij**6 - 8*yij**6 + 4*yij**4*zij**2 + 11*yij**2*zij**4 - zij**6 + xij**4*(-101*yij**2 + 11*zij**2) + 
            2*xij**2*(58*yij**4 - 45*yij**2*zij**2 + 2*zij**4)) + 
         7*b**4*(66*xij**8 - 3*xij**6*(377*yij**2 - 63*zij**2) - (yij**2 + zij**2)**2*(104*yij**4 - 132*yij**2*zij**2 + 9*zij**4) + 
            xij**4*(271*yij**4 - 2148*yij**2*zij**2 + 171*zij**4) + xij**2*(1364*yij**6 + 422*yij**4*zij**2 - 903*yij**2*zij**4 + 39*zij**6)))/
       (168.*eta*PI*(xij**2 + yij**2 + zij**2)**6.5),(5*b**4*yij*zij*
         (-132*xij**6 + xij**4*(269*yij**2 - 249*zij**2) - (34*yij**2 - 15*zij**2)*(yij**2 + zij**2)**2 + 
           xij**2*(367*yij**4 + 265*yij**2*zij**2 - 102*zij**4) + 
           54*b**2*(8*xij**4 + 2*yij**4 + yij**2*zij**2 - zij**4 + xij**2*(-23*yij**2 + 7*zij**2))))/
       (24.*eta*PI*(xij**2 + yij**2 + zij**2)**6.5),(-270*b**6*
          (2*xij**6 + 2*yij**6 - 15*yij**4*zij**2 - 15*yij**2*zij**4 + 2*zij**6 - 15*xij**4*(yij**2 + zij**2) - 
            15*xij**2*(yij**4 - 12*yij**2*zij**2 + zij**4)) + 
         7*b**4*(22*xij**8 - 157*xij**6*(yij**2 + zij**2) + (yij**2 + zij**2)**2*(22*yij**4 - 201*yij**2*zij**2 + 22*zij**4) - 
            2*xij**4*(179*yij**4 - 937*yij**2*zij**2 + 179*zij**4) + 
            xij**2*(-157*yij**6 + 1874*yij**4*zij**2 + 1874*yij**2*zij**4 - 157*zij**6)))/(168.*eta*PI*(xij**2 + yij**2 + zij**2)**6.5)],
     [(-5*b**4*xij*zij*(-152*xij**6 + 188*xij**4*(yij**2 + zij**2) + 265*xij**2*(yij**2 + zij**2)**2 - 75*(yij**2 + zij**2)**3 + 
           54*b**2*(8*xij**4 - 20*xij**2*(yij**2 + zij**2) + 5*(yij**2 + zij**2)**2)))/(24.*eta*PI*(xij**2 + yij**2 + zij**2)**6.5),
      (5*b**4*yij*zij*(96*xij**6 + 8*xij**4*(yij**2 + zij**2) - 83*xij**2*(yij**2 + zij**2)**2 + 5*(yij**2 + zij**2)**3 - 
           18*b**2*(16*xij**4 - 16*xij**2*(yij**2 + zij**2) + (yij**2 + zij**2)**2)))/(8.*eta*PI*(xij**2 + yij**2 + zij**2)**6.5),
      (270*b**6*(8*xij**6 + (yij**2 - 6*zij**2)*(yij**2 + zij**2)**2 - 4*xij**4*(yij**2 + 29*zij**2) + 
            xij**2*(-11*yij**4 + 90*yij**2*zij**2 + 101*zij**4)) - 
         7*b**4*(104*xij**8 + 4*xij**6*(19*yij**2 - 341*zij**2) - 3*xij**2*(38*yij**2 - 377*zij**2)*(yij**2 + zij**2)**2 + 
            3*(3*yij**2 - 22*zij**2)*(yij**2 + zij**2)**3 - xij**4*(151*yij**4 + 422*yij**2*zij**2 + 271*zij**4)))/
       (168.*eta*PI*(xij**2 + yij**2 + zij**2)**6.5),(5*b**4*xij*zij*
         (-34*xij**6 + xij**4*(367*yij**2 - 53*zij**2) - 3*(44*yij**2 - 5*zij**2)*(yij**2 + zij**2)**2 + 
           xij**2*(269*yij**4 + 265*yij**2*zij**2 - 4*zij**4) + 
           54*b**2*(2*xij**4 + 8*yij**4 + 7*yij**2*zij**2 - zij**4 + xij**2*(-23*yij**2 + zij**2))))/
       (24.*eta*PI*(xij**2 + yij**2 + zij**2)**6.5),(5*b**4*xij*yij*
         (-34*xij**6 + 3*(5*yij**2 - 44*zij**2)*(yij**2 + zij**2)**2 + xij**4*(-53*yij**2 + 367*zij**2) + 
           xij**2*(-4*yij**4 + 265*yij**2*zij**2 + 269*zij**4) + 
           54*b**2*(2*xij**4 - yij**4 + 7*yij**2*zij**2 + 8*zij**4 + xij**2*(yij**2 - 23*zij**2))))/
       (24.*eta*PI*(xij**2 + yij**2 + zij**2)**6.5),(5*b**4*xij*zij*
         (-34*xij**6 + (15*yij**2 - 34*zij**2)*(yij**2 + zij**2)**2 + xij**4*(-53*yij**2 + 87*zij**2) + 
           xij**2*(-4*yij**4 + 83*yij**2*zij**2 + 87*zij**4) + 
           54*b**2*(2*xij**4 - yij**4 + yij**2*zij**2 + 2*zij**4 + xij**2*(yij**2 - 7*zij**2))))/(8.*eta*PI*(xij**2 + yij**2 + zij**2)**6.5),
      (5*b**4*yij*zij*(-132*xij**6 + xij**4*(269*yij**2 - 249*zij**2) - (34*yij**2 - 15*zij**2)*(yij**2 + zij**2)**2 + 
           xij**2*(367*yij**4 + 265*yij**2*zij**2 - 102*zij**4) + 
           54*b**2*(8*xij**4 + 2*yij**4 + yij**2*zij**2 - zij**4 + xij**2*(-23*yij**2 + 7*zij**2))))/
       (24.*eta*PI*(xij**2 + yij**2 + zij**2)**6.5),(-270*b**6*
          (2*xij**6 + 2*yij**6 - 15*yij**4*zij**2 - 15*yij**2*zij**4 + 2*zij**6 - 15*xij**4*(yij**2 + zij**2) - 
            15*xij**2*(yij**4 - 12*yij**2*zij**2 + zij**4)) + 
         7*b**4*(22*xij**8 - 157*xij**6*(yij**2 + zij**2) + (yij**2 + zij**2)**2*(22*yij**4 - 201*yij**2*zij**2 + 22*zij**4) - 
            2*xij**4*(179*yij**4 - 937*yij**2*zij**2 + 179*zij**4) + 
            xij**2*(-157*yij**6 + 1874*yij**4*zij**2 + 1874*yij**2*zij**4 - 157*zij**6)))/(168.*eta*PI*(xij**2 + yij**2 + zij**2)**6.5),
      (5*b**4*yij*zij*(-132*xij**6 + (15*yij**2 - 34*zij**2)*(yij**2 + zij**2)**2 + xij**4*(-249*yij**2 + 269*zij**2) + 
           xij**2*(-102*yij**4 + 265*yij**2*zij**2 + 367*zij**4) + 
           54*b**2*(8*xij**4 - yij**4 + yij**2*zij**2 + 2*zij**4 + xij**2*(7*yij**2 - 23*zij**2))))/
       (24.*eta*PI*(xij**2 + yij**2 + zij**2)**6.5)],[(270*b**6*
          (8*xij**6 - (6*yij**2 - zij**2)*(yij**2 + zij**2)**2 - 4*xij**4*(29*yij**2 + zij**2) + 
            xij**2*(101*yij**4 + 90*yij**2*zij**2 - 11*zij**4)) - 
         7*b**4*(104*xij**8 + 3*xij**2*(377*yij**2 - 38*zij**2)*(yij**2 + zij**2)**2 - 3*(22*yij**2 - 3*zij**2)*(yij**2 + zij**2)**3 + 
            xij**6*(-1364*yij**2 + 76*zij**2) - xij**4*(271*yij**4 + 422*yij**2*zij**2 + 151*zij**4)))/
       (168.*eta*PI*(xij**2 + yij**2 + zij**2)**6.5),(5*b**4*xij*yij*
         (-34*xij**6 + xij**4*(87*yij**2 - 53*zij**2) - (34*yij**2 - 15*zij**2)*(yij**2 + zij**2)**2 + 
           xij**2*(87*yij**4 + 83*yij**2*zij**2 - 4*zij**4) + 
           54*b**2*(2*xij**4 + 2*yij**4 + yij**2*zij**2 - zij**4 + xij**2*(-7*yij**2 + zij**2))))/(8.*eta*PI*(xij**2 + yij**2 + zij**2)**6.5),
      (5*b**4*xij*zij*(-34*xij**6 + xij**4*(367*yij**2 - 53*zij**2) - 3*(44*yij**2 - 5*zij**2)*(yij**2 + zij**2)**2 + 
           xij**2*(269*yij**4 + 265*yij**2*zij**2 - 4*zij**4) + 
           54*b**2*(2*xij**4 + 8*yij**4 + 7*yij**2*zij**2 - zij**4 + xij**2*(-23*yij**2 + zij**2))))/
       (24.*eta*PI*(xij**2 + yij**2 + zij**2)**6.5),(-270*b**6*
          (6*xij**6 - 8*yij**6 + 4*yij**4*zij**2 + 11*yij**2*zij**4 - zij**6 + xij**4*(-101*yij**2 + 11*zij**2) + 
            2*xij**2*(58*yij**4 - 45*yij**2*zij**2 + 2*zij**4)) + 
         7*b**4*(66*xij**8 - 3*xij**6*(377*yij**2 - 63*zij**2) - (yij**2 + zij**2)**2*(104*yij**4 - 132*yij**2*zij**2 + 9*zij**4) + 
            xij**4*(271*yij**4 - 2148*yij**2*zij**2 + 171*zij**4) + xij**2*(1364*yij**6 + 422*yij**4*zij**2 - 903*yij**2*zij**4 + 39*zij**6)))/
       (168.*eta*PI*(xij**2 + yij**2 + zij**2)**6.5),(5*b**4*yij*zij*
         (-132*xij**6 + xij**4*(269*yij**2 - 249*zij**2) - (34*yij**2 - 15*zij**2)*(yij**2 + zij**2)**2 + 
           xij**2*(367*yij**4 + 265*yij**2*zij**2 - 102*zij**4) + 
           54*b**2*(8*xij**4 + 2*yij**4 + yij**2*zij**2 - zij**4 + xij**2*(-23*yij**2 + 7*zij**2))))/
       (24.*eta*PI*(xij**2 + yij**2 + zij**2)**6.5),(-270*b**6*
          (2*xij**6 + 2*yij**6 - 15*yij**4*zij**2 - 15*yij**2*zij**4 + 2*zij**6 - 15*xij**4*(yij**2 + zij**2) - 
            15*xij**2*(yij**4 - 12*yij**2*zij**2 + zij**4)) + 
         7*b**4*(22*xij**8 - 157*xij**6*(yij**2 + zij**2) + (yij**2 + zij**2)**2*(22*yij**4 - 201*yij**2*zij**2 + 22*zij**4) - 
            2*xij**4*(179*yij**4 - 937*yij**2*zij**2 + 179*zij**4) + 
            xij**2*(-157*yij**6 + 1874*yij**4*zij**2 + 1874*yij**2*zij**4 - 157*zij**6)))/(168.*eta*PI*(xij**2 + yij**2 + zij**2)**6.5),
      (-5*b**4*xij*yij*(-75*xij**6 - 152*yij**6 + 188*yij**4*zij**2 + 265*yij**2*zij**4 - 75*zij**6 + 5*xij**4*(53*yij**2 - 45*zij**2) + 
           xij**2*(188*yij**4 + 530*yij**2*zij**2 - 225*zij**4) + 
           54*b**2*(5*xij**4 + 8*yij**4 - 20*yij**2*zij**2 + 5*zij**4 + 10*xij**2*(-2*yij**2 + zij**2))))/
       (24.*eta*PI*(xij**2 + yij**2 + zij**2)**6.5),(5*b**4*xij*zij*
         (5*xij**6 + 96*yij**6 + 8*yij**4*zij**2 - 83*yij**2*zij**4 + 5*zij**6 + xij**4*(-83*yij**2 + 15*zij**2) + 
           xij**2*(8*yij**4 - 166*yij**2*zij**2 + 15*zij**4) - 
           18*b**2*(xij**4 + 16*yij**4 - 16*yij**2*zij**2 + zij**4 + 2*xij**2*(-8*yij**2 + zij**2))))/
       (8.*eta*PI*(xij**2 + yij**2 + zij**2)**6.5),(5*b**4*xij*yij*
         (15*xij**6 - 34*yij**6 + 367*yij**4*zij**2 + 269*yij**2*zij**4 - 132*zij**6 - 2*xij**4*(2*yij**2 + 51*zij**2) + 
           xij**2*(-53*yij**4 + 265*yij**2*zij**2 - 249*zij**4) - 
           54*b**2*(xij**4 - 2*yij**4 + 23*yij**2*zij**2 - 8*zij**4 - xij**2*(yij**2 + 7*zij**2))))/
       (24.*eta*PI*(xij**2 + yij**2 + zij**2)**6.5)],[(5*b**4*yij*zij*
         (96*xij**6 + 8*xij**4*(yij**2 + zij**2) - 83*xij**2*(yij**2 + zij**2)**2 + 5*(yij**2 + zij**2)**3 - 
           18*b**2*(16*xij**4 - 16*xij**2*(yij**2 + zij**2) + (yij**2 + zij**2)**2)))/(8.*eta*PI*(xij**2 + yij**2 + zij**2)**6.5),
      (5*b**4*xij*zij*(-34*xij**6 + xij**4*(367*yij**2 - 53*zij**2) - 3*(44*yij**2 - 5*zij**2)*(yij**2 + zij**2)**2 + 
           xij**2*(269*yij**4 + 265*yij**2*zij**2 - 4*zij**4) + 
           54*b**2*(2*xij**4 + 8*yij**4 + 7*yij**2*zij**2 - zij**4 + xij**2*(-23*yij**2 + zij**2))))/
       (24.*eta*PI*(xij**2 + yij**2 + zij**2)**6.5),(5*b**4*xij*yij*
         (-34*xij**6 + 3*(5*yij**2 - 44*zij**2)*(yij**2 + zij**2)**2 + xij**4*(-53*yij**2 + 367*zij**2) + 
           xij**2*(-4*yij**4 + 265*yij**2*zij**2 + 269*zij**4) + 
           54*b**2*(2*xij**4 - yij**4 + 7*yij**2*zij**2 + 8*zij**4 + xij**2*(yij**2 - 23*zij**2))))/
       (24.*eta*PI*(xij**2 + yij**2 + zij**2)**6.5),(5*b**4*yij*zij*
         (-132*xij**6 + xij**4*(269*yij**2 - 249*zij**2) - (34*yij**2 - 15*zij**2)*(yij**2 + zij**2)**2 + 
           xij**2*(367*yij**4 + 265*yij**2*zij**2 - 102*zij**4) + 
           54*b**2*(8*xij**4 + 2*yij**4 + yij**2*zij**2 - zij**4 + xij**2*(-23*yij**2 + 7*zij**2))))/
       (24.*eta*PI*(xij**2 + yij**2 + zij**2)**6.5),(-270*b**6*
          (2*xij**6 + 2*yij**6 - 15*yij**4*zij**2 - 15*yij**2*zij**4 + 2*zij**6 - 15*xij**4*(yij**2 + zij**2) - 
            15*xij**2*(yij**4 - 12*yij**2*zij**2 + zij**4)) + 
         7*b**4*(22*xij**8 - 157*xij**6*(yij**2 + zij**2) + (yij**2 + zij**2)**2*(22*yij**4 - 201*yij**2*zij**2 + 22*zij**4) - 
            2*xij**4*(179*yij**4 - 937*yij**2*zij**2 + 179*zij**4) + 
            xij**2*(-157*yij**6 + 1874*yij**4*zij**2 + 1874*yij**2*zij**4 - 157*zij**6)))/(168.*eta*PI*(xij**2 + yij**2 + zij**2)**6.5),
      (5*b**4*yij*zij*(-132*xij**6 + (15*yij**2 - 34*zij**2)*(yij**2 + zij**2)**2 + xij**4*(-249*yij**2 + 269*zij**2) + 
           xij**2*(-102*yij**4 + 265*yij**2*zij**2 + 367*zij**4) + 
           54*b**2*(8*xij**4 - yij**4 + yij**2*zij**2 + 2*zij**4 + xij**2*(7*yij**2 - 23*zij**2))))/
       (24.*eta*PI*(xij**2 + yij**2 + zij**2)**6.5),(5*b**4*xij*zij*
         (5*xij**6 + 96*yij**6 + 8*yij**4*zij**2 - 83*yij**2*zij**4 + 5*zij**6 + xij**4*(-83*yij**2 + 15*zij**2) + 
           xij**2*(8*yij**4 - 166*yij**2*zij**2 + 15*zij**4) - 
           18*b**2*(xij**4 + 16*yij**4 - 16*yij**2*zij**2 + zij**4 + 2*xij**2*(-8*yij**2 + zij**2))))/
       (8.*eta*PI*(xij**2 + yij**2 + zij**2)**6.5),(5*b**4*xij*yij*
         (15*xij**6 - 34*yij**6 + 367*yij**4*zij**2 + 269*yij**2*zij**4 - 132*zij**6 - 2*xij**4*(2*yij**2 + 51*zij**2) + 
           xij**2*(-53*yij**4 + 265*yij**2*zij**2 - 249*zij**4) - 
           54*b**2*(xij**4 - 2*yij**4 + 23*yij**2*zij**2 - 8*zij**4 - xij**2*(yij**2 + 7*zij**2))))/
       (24.*eta*PI*(xij**2 + yij**2 + zij**2)**6.5),(5*b**4*xij*zij*
         (15*xij**6 - 132*yij**6 + 269*yij**4*zij**2 + 367*yij**2*zij**4 - 34*zij**6 - 2*xij**4*(51*yij**2 + 2*zij**2) + 
           xij**2*(-249*yij**4 + 265*yij**2*zij**2 - 53*zij**4) - 
           54*b**2*(xij**4 - 8*yij**4 + 23*yij**2*zij**2 - 2*zij**4 - xij**2*(7*yij**2 + zij**2))))/
       (24.*eta*PI*(xij**2 + yij**2 + zij**2)**6.5)],[(270*b**6*
          (8*xij**6 + (yij**2 - 6*zij**2)*(yij**2 + zij**2)**2 - 4*xij**4*(yij**2 + 29*zij**2) + 
            xij**2*(-11*yij**4 + 90*yij**2*zij**2 + 101*zij**4)) - 
         7*b**4*(104*xij**8 + 4*xij**6*(19*yij**2 - 341*zij**2) - 3*xij**2*(38*yij**2 - 377*zij**2)*(yij**2 + zij**2)**2 + 
            3*(3*yij**2 - 22*zij**2)*(yij**2 + zij**2)**3 - xij**4*(151*yij**4 + 422*yij**2*zij**2 + 271*zij**4)))/
       (168.*eta*PI*(xij**2 + yij**2 + zij**2)**6.5),(5*b**4*xij*yij*
         (-34*xij**6 + 3*(5*yij**2 - 44*zij**2)*(yij**2 + zij**2)**2 + xij**4*(-53*yij**2 + 367*zij**2) + 
           xij**2*(-4*yij**4 + 265*yij**2*zij**2 + 269*zij**4) + 
           54*b**2*(2*xij**4 - yij**4 + 7*yij**2*zij**2 + 8*zij**4 + xij**2*(yij**2 - 23*zij**2))))/
       (24.*eta*PI*(xij**2 + yij**2 + zij**2)**6.5),(5*b**4*xij*zij*
         (-34*xij**6 + (15*yij**2 - 34*zij**2)*(yij**2 + zij**2)**2 + xij**4*(-53*yij**2 + 87*zij**2) + 
           xij**2*(-4*yij**4 + 83*yij**2*zij**2 + 87*zij**4) + 
           54*b**2*(2*xij**4 - yij**4 + yij**2*zij**2 + 2*zij**4 + xij**2*(yij**2 - 7*zij**2))))/(8.*eta*PI*(xij**2 + yij**2 + zij**2)**6.5),
      (-270*b**6*(2*xij**6 + 2*yij**6 - 15*yij**4*zij**2 - 15*yij**2*zij**4 + 2*zij**6 - 15*xij**4*(yij**2 + zij**2) - 
            15*xij**2*(yij**4 - 12*yij**2*zij**2 + zij**4)) + 
         7*b**4*(22*xij**8 - 157*xij**6*(yij**2 + zij**2) + (yij**2 + zij**2)**2*(22*yij**4 - 201*yij**2*zij**2 + 22*zij**4) - 
            2*xij**4*(179*yij**4 - 937*yij**2*zij**2 + 179*zij**4) + 
            xij**2*(-157*yij**6 + 1874*yij**4*zij**2 + 1874*yij**2*zij**4 - 157*zij**6)))/(168.*eta*PI*(xij**2 + yij**2 + zij**2)**6.5),
      (5*b**4*yij*zij*(-132*xij**6 + (15*yij**2 - 34*zij**2)*(yij**2 + zij**2)**2 + xij**4*(-249*yij**2 + 269*zij**2) + 
           xij**2*(-102*yij**4 + 265*yij**2*zij**2 + 367*zij**4) + 
           54*b**2*(8*xij**4 - yij**4 + yij**2*zij**2 + 2*zij**4 + xij**2*(7*yij**2 - 23*zij**2))))/
       (24.*eta*PI*(xij**2 + yij**2 + zij**2)**6.5),(-270*b**6*
          (6*xij**6 - yij**6 + 11*yij**4*zij**2 + 4*yij**2*zij**4 - 8*zij**6 + xij**4*(11*yij**2 - 101*zij**2) + 
            2*xij**2*(2*yij**4 - 45*yij**2*zij**2 + 58*zij**4)) + 
         7*b**4*(66*xij**8 + 3*xij**6*(63*yij**2 - 377*zij**2) - (yij**2 + zij**2)**2*(9*yij**4 - 132*yij**2*zij**2 + 104*zij**4) + 
            xij**4*(171*yij**4 - 2148*yij**2*zij**2 + 271*zij**4) + xij**2*(39*yij**6 - 903*yij**4*zij**2 + 422*yij**2*zij**4 + 1364*zij**6)))/
       (168.*eta*PI*(xij**2 + yij**2 + zij**2)**6.5),(5*b**4*xij*yij*
         (15*xij**6 - 34*yij**6 + 367*yij**4*zij**2 + 269*yij**2*zij**4 - 132*zij**6 - 2*xij**4*(2*yij**2 + 51*zij**2) + 
           xij**2*(-53*yij**4 + 265*yij**2*zij**2 - 249*zij**4) - 
           54*b**2*(xij**4 - 2*yij**4 + 23*yij**2*zij**2 - 8*zij**4 - xij**2*(yij**2 + 7*zij**2))))/
       (24.*eta*PI*(xij**2 + yij**2 + zij**2)**6.5),(5*b**4*xij*zij*
         (15*xij**6 - 132*yij**6 + 269*yij**4*zij**2 + 367*yij**2*zij**4 - 34*zij**6 - 2*xij**4*(51*yij**2 + 2*zij**2) + 
           xij**2*(-249*yij**4 + 265*yij**2*zij**2 - 53*zij**4) - 
           54*b**2*(xij**4 - 8*yij**4 + 23*yij**2*zij**2 - 2*zij**4 - xij**2*(7*yij**2 + zij**2))))/
       (24.*eta*PI*(xij**2 + yij**2 + zij**2)**6.5),(5*b**4*xij*yij*
         (5*xij**6 + 5*yij**6 - 83*yij**4*zij**2 + 8*yij**2*zij**4 + 96*zij**6 + xij**4*(15*yij**2 - 83*zij**2) + 
           xij**2*(15*yij**4 - 166*yij**2*zij**2 + 8*zij**4) - 
           18*b**2*(xij**4 + yij**4 - 16*yij**2*zij**2 + 16*zij**4 + 2*xij**2*(yij**2 - 8*zij**2))))/
       (8.*eta*PI*(xij**2 + yij**2 + zij**2)**6.5)],[(5*b**4*xij*yij*
         (-34*xij**6 + xij**4*(87*yij**2 - 53*zij**2) - (34*yij**2 - 15*zij**2)*(yij**2 + zij**2)**2 + 
           xij**2*(87*yij**4 + 83*yij**2*zij**2 - 4*zij**4) + 
           54*b**2*(2*xij**4 + 2*yij**4 + yij**2*zij**2 - zij**4 + xij**2*(-7*yij**2 + zij**2))))/(8.*eta*PI*(xij**2 + yij**2 + zij**2)**6.5),
      (-270*b**6*(6*xij**6 - 8*yij**6 + 4*yij**4*zij**2 + 11*yij**2*zij**4 - zij**6 + xij**4*(-101*yij**2 + 11*zij**2) + 
            2*xij**2*(58*yij**4 - 45*yij**2*zij**2 + 2*zij**4)) + 
         7*b**4*(66*xij**8 - 3*xij**6*(377*yij**2 - 63*zij**2) - (yij**2 + zij**2)**2*(104*yij**4 - 132*yij**2*zij**2 + 9*zij**4) + 
            xij**4*(271*yij**4 - 2148*yij**2*zij**2 + 171*zij**4) + xij**2*(1364*yij**6 + 422*yij**4*zij**2 - 903*yij**2*zij**4 + 39*zij**6)))/
       (168.*eta*PI*(xij**2 + yij**2 + zij**2)**6.5),(5*b**4*yij*zij*
         (-132*xij**6 + xij**4*(269*yij**2 - 249*zij**2) - (34*yij**2 - 15*zij**2)*(yij**2 + zij**2)**2 + 
           xij**2*(367*yij**4 + 265*yij**2*zij**2 - 102*zij**4) + 
           54*b**2*(8*xij**4 + 2*yij**4 + yij**2*zij**2 - zij**4 + xij**2*(-23*yij**2 + 7*zij**2))))/
       (24.*eta*PI*(xij**2 + yij**2 + zij**2)**6.5),(-5*b**4*xij*yij*
         (-75*xij**6 - 152*yij**6 + 188*yij**4*zij**2 + 265*yij**2*zij**4 - 75*zij**6 + 5*xij**4*(53*yij**2 - 45*zij**2) + 
           xij**2*(188*yij**4 + 530*yij**2*zij**2 - 225*zij**4) + 
           54*b**2*(5*xij**4 + 8*yij**4 - 20*yij**2*zij**2 + 5*zij**4 + 10*xij**2*(-2*yij**2 + zij**2))))/
       (24.*eta*PI*(xij**2 + yij**2 + zij**2)**6.5),(5*b**4*xij*zij*
         (5*xij**6 + 96*yij**6 + 8*yij**4*zij**2 - 83*yij**2*zij**4 + 5*zij**6 + xij**4*(-83*yij**2 + 15*zij**2) + 
           xij**2*(8*yij**4 - 166*yij**2*zij**2 + 15*zij**4) - 
           18*b**2*(xij**4 + 16*yij**4 - 16*yij**2*zij**2 + zij**4 + 2*xij**2*(-8*yij**2 + zij**2))))/
       (8.*eta*PI*(xij**2 + yij**2 + zij**2)**6.5),(5*b**4*xij*yij*
         (15*xij**6 - 34*yij**6 + 367*yij**4*zij**2 + 269*yij**2*zij**4 - 132*zij**6 - 2*xij**4*(2*yij**2 + 51*zij**2) + 
           xij**2*(-53*yij**4 + 265*yij**2*zij**2 - 249*zij**4) - 
           54*b**2*(xij**4 - 2*yij**4 + 23*yij**2*zij**2 - 8*zij**4 - xij**2*(yij**2 + 7*zij**2))))/
       (24.*eta*PI*(xij**2 + yij**2 + zij**2)**6.5),(90*b**6*
          (5*xij**6 - 16*yij**6 + 120*yij**4*zij**2 - 90*yij**2*zij**4 + 5*zij**6 + 15*xij**4*(-6*yij**2 + zij**2) + 
            15*xij**2*(8*yij**4 - 12*yij**2*zij**2 + zij**4)) - 
         35*b**4*(3*xij**8 - 16*yij**8 + 88*yij**6*zij**2 + 38*yij**4*zij**4 - 63*yij**2*zij**6 + 3*zij**8 + xij**6*(-63*yij**2 + 12*zij**2) + 
            xij**4*(38*yij**4 - 189*yij**2*zij**2 + 18*zij**4) + xij**2*(88*yij**6 + 76*yij**4*zij**2 - 189*yij**2*zij**4 + 12*zij**6)))/
       (56.*eta*PI*(xij**2 + yij**2 + zij**2)**6.5),(-5*b**4*yij*zij*
         (-75*xij**6 - 152*yij**6 + 188*yij**4*zij**2 + 265*yij**2*zij**4 - 75*zij**6 + 5*xij**4*(53*yij**2 - 45*zij**2) + 
           xij**2*(188*yij**4 + 530*yij**2*zij**2 - 225*zij**4) + 
           54*b**2*(5*xij**4 + 8*yij**4 - 20*yij**2*zij**2 + 5*zij**4 + 10*xij**2*(-2*yij**2 + zij**2))))/
       (24.*eta*PI*(xij**2 + yij**2 + zij**2)**6.5),(270*b**6*
          (xij**6 + 8*yij**6 - 116*yij**4*zij**2 + 101*yij**2*zij**4 - 6*zij**6 - xij**4*(11*yij**2 + 4*zij**2) + 
            xij**2*(-4*yij**4 + 90*yij**2*zij**2 - 11*zij**4)) - 
         7*b**4*(9*xij**8 + 104*yij**8 - 1364*yij**6*zij**2 - 271*yij**4*zij**4 + 1131*yij**2*zij**6 - 66*zij**8 - 
            3*xij**6*(38*yij**2 + 13*zij**2) + xij**4*(-151*yij**4 + 903*yij**2*zij**2 - 171*zij**4) + 
            xij**2*(76*yij**6 - 422*yij**4*zij**2 + 2148*yij**2*zij**4 - 189*zij**6)))/(168.*eta*PI*(xij**2 + yij**2 + zij**2)**6.5)],
     [(5*b**4*xij*zij*(-34*xij**6 + xij**4*(367*yij**2 - 53*zij**2) - 3*(44*yij**2 - 5*zij**2)*(yij**2 + zij**2)**2 + 
           xij**2*(269*yij**4 + 265*yij**2*zij**2 - 4*zij**4) + 
           54*b**2*(2*xij**4 + 8*yij**4 + 7*yij**2*zij**2 - zij**4 + xij**2*(-23*yij**2 + zij**2))))/
       (24.*eta*PI*(xij**2 + yij**2 + zij**2)**6.5),(5*b**4*yij*zij*
         (-132*xij**6 + xij**4*(269*yij**2 - 249*zij**2) - (34*yij**2 - 15*zij**2)*(yij**2 + zij**2)**2 + 
           xij**2*(367*yij**4 + 265*yij**2*zij**2 - 102*zij**4) + 
           54*b**2*(8*xij**4 + 2*yij**4 + yij**2*zij**2 - zij**4 + xij**2*(-23*yij**2 + 7*zij**2))))/
       (24.*eta*PI*(xij**2 + yij**2 + zij**2)**6.5),(-270*b**6*
          (2*xij**6 + 2*yij**6 - 15*yij**4*zij**2 - 15*yij**2*zij**4 + 2*zij**6 - 15*xij**4*(yij**2 + zij**2) - 
            15*xij**2*(yij**4 - 12*yij**2*zij**2 + zij**4)) + 
         7*b**4*(22*xij**8 - 157*xij**6*(yij**2 + zij**2) + (yij**2 + zij**2)**2*(22*yij**4 - 201*yij**2*zij**2 + 22*zij**4) - 
            2*xij**4*(179*yij**4 - 937*yij**2*zij**2 + 179*zij**4) + 
            xij**2*(-157*yij**6 + 1874*yij**4*zij**2 + 1874*yij**2*zij**4 - 157*zij**6)))/(168.*eta*PI*(xij**2 + yij**2 + zij**2)**6.5),
      (5*b**4*xij*zij*(5*xij**6 + 96*yij**6 + 8*yij**4*zij**2 - 83*yij**2*zij**4 + 5*zij**6 + xij**4*(-83*yij**2 + 15*zij**2) + 
           xij**2*(8*yij**4 - 166*yij**2*zij**2 + 15*zij**4) - 
           18*b**2*(xij**4 + 16*yij**4 - 16*yij**2*zij**2 + zij**4 + 2*xij**2*(-8*yij**2 + zij**2))))/
       (8.*eta*PI*(xij**2 + yij**2 + zij**2)**6.5),(5*b**4*xij*yij*
         (15*xij**6 - 34*yij**6 + 367*yij**4*zij**2 + 269*yij**2*zij**4 - 132*zij**6 - 2*xij**4*(2*yij**2 + 51*zij**2) + 
           xij**2*(-53*yij**4 + 265*yij**2*zij**2 - 249*zij**4) - 
           54*b**2*(xij**4 - 2*yij**4 + 23*yij**2*zij**2 - 8*zij**4 - xij**2*(yij**2 + 7*zij**2))))/
       (24.*eta*PI*(xij**2 + yij**2 + zij**2)**6.5),(5*b**4*xij*zij*
         (15*xij**6 - 132*yij**6 + 269*yij**4*zij**2 + 367*yij**2*zij**4 - 34*zij**6 - 2*xij**4*(51*yij**2 + 2*zij**2) + 
           xij**2*(-249*yij**4 + 265*yij**2*zij**2 - 53*zij**4) - 
           54*b**2*(xij**4 - 8*yij**4 + 23*yij**2*zij**2 - 2*zij**4 - xij**2*(7*yij**2 + zij**2))))/
       (24.*eta*PI*(xij**2 + yij**2 + zij**2)**6.5),(-5*b**4*yij*zij*
         (-75*xij**6 - 152*yij**6 + 188*yij**4*zij**2 + 265*yij**2*zij**4 - 75*zij**6 + 5*xij**4*(53*yij**2 - 45*zij**2) + 
           xij**2*(188*yij**4 + 530*yij**2*zij**2 - 225*zij**4) + 
           54*b**2*(5*xij**4 + 8*yij**4 - 20*yij**2*zij**2 + 5*zij**4 + 10*xij**2*(-2*yij**2 + zij**2))))/
       (24.*eta*PI*(xij**2 + yij**2 + zij**2)**6.5),(270*b**6*
          (xij**6 + 8*yij**6 - 116*yij**4*zij**2 + 101*yij**2*zij**4 - 6*zij**6 - xij**4*(11*yij**2 + 4*zij**2) + 
            xij**2*(-4*yij**4 + 90*yij**2*zij**2 - 11*zij**4)) - 
         7*b**4*(9*xij**8 + 104*yij**8 - 1364*yij**6*zij**2 - 271*yij**4*zij**4 + 1131*yij**2*zij**6 - 66*zij**8 - 
            3*xij**6*(38*yij**2 + 13*zij**2) + xij**4*(-151*yij**4 + 903*yij**2*zij**2 - 171*zij**4) + 
            xij**2*(76*yij**6 - 422*yij**4*zij**2 + 2148*yij**2*zij**4 - 189*zij**6)))/(168.*eta*PI*(xij**2 + yij**2 + zij**2)**6.5),
      (5*b**4*yij*zij*(15*xij**6 - 34*yij**6 + 87*yij**4*zij**2 + 87*yij**2*zij**4 - 34*zij**6 - 4*xij**4*(yij**2 + zij**2) + 
           xij**2*(-53*yij**4 + 83*yij**2*zij**2 - 53*zij**4) - 
           54*b**2*(xij**4 - 2*yij**4 + 7*yij**2*zij**2 - 2*zij**4 - xij**2*(yij**2 + zij**2))))/(8.*eta*PI*(xij**2 + yij**2 + zij**2)**6.5)],
     [(5*b**4*xij*yij*(-34*xij**6 + 3*(5*yij**2 - 44*zij**2)*(yij**2 + zij**2)**2 + xij**4*(-53*yij**2 + 367*zij**2) + 
           xij**2*(-4*yij**4 + 265*yij**2*zij**2 + 269*zij**4) + 
           54*b**2*(2*xij**4 - yij**4 + 7*yij**2*zij**2 + 8*zij**4 + xij**2*(yij**2 - 23*zij**2))))/
       (24.*eta*PI*(xij**2 + yij**2 + zij**2)**6.5),(-270*b**6*
          (2*xij**6 + 2*yij**6 - 15*yij**4*zij**2 - 15*yij**2*zij**4 + 2*zij**6 - 15*xij**4*(yij**2 + zij**2) - 
            15*xij**2*(yij**4 - 12*yij**2*zij**2 + zij**4)) + 
         7*b**4*(22*xij**8 - 157*xij**6*(yij**2 + zij**2) + (yij**2 + zij**2)**2*(22*yij**4 - 201*yij**2*zij**2 + 22*zij**4) - 
            2*xij**4*(179*yij**4 - 937*yij**2*zij**2 + 179*zij**4) + 
            xij**2*(-157*yij**6 + 1874*yij**4*zij**2 + 1874*yij**2*zij**4 - 157*zij**6)))/(168.*eta*PI*(xij**2 + yij**2 + zij**2)**6.5),
      (5*b**4*yij*zij*(-132*xij**6 + (15*yij**2 - 34*zij**2)*(yij**2 + zij**2)**2 + xij**4*(-249*yij**2 + 269*zij**2) + 
           xij**2*(-102*yij**4 + 265*yij**2*zij**2 + 367*zij**4) + 
           54*b**2*(8*xij**4 - yij**4 + yij**2*zij**2 + 2*zij**4 + xij**2*(7*yij**2 - 23*zij**2))))/
       (24.*eta*PI*(xij**2 + yij**2 + zij**2)**6.5),(5*b**4*xij*yij*
         (15*xij**6 - 34*yij**6 + 367*yij**4*zij**2 + 269*yij**2*zij**4 - 132*zij**6 - 2*xij**4*(2*yij**2 + 51*zij**2) + 
           xij**2*(-53*yij**4 + 265*yij**2*zij**2 - 249*zij**4) - 
           54*b**2*(xij**4 - 2*yij**4 + 23*yij**2*zij**2 - 8*zij**4 - xij**2*(yij**2 + 7*zij**2))))/
       (24.*eta*PI*(xij**2 + yij**2 + zij**2)**6.5),(5*b**4*xij*zij*
         (15*xij**6 - 132*yij**6 + 269*yij**4*zij**2 + 367*yij**2*zij**4 - 34*zij**6 - 2*xij**4*(51*yij**2 + 2*zij**2) + 
           xij**2*(-249*yij**4 + 265*yij**2*zij**2 - 53*zij**4) - 
           54*b**2*(xij**4 - 8*yij**4 + 23*yij**2*zij**2 - 2*zij**4 - xij**2*(7*yij**2 + zij**2))))/
       (24.*eta*PI*(xij**2 + yij**2 + zij**2)**6.5),(5*b**4*xij*yij*
         (5*xij**6 + 5*yij**6 - 83*yij**4*zij**2 + 8*yij**2*zij**4 + 96*zij**6 + xij**4*(15*yij**2 - 83*zij**2) + 
           xij**2*(15*yij**4 - 166*yij**2*zij**2 + 8*zij**4) - 
           18*b**2*(xij**4 + yij**4 - 16*yij**2*zij**2 + 16*zij**4 + 2*xij**2*(yij**2 - 8*zij**2))))/
       (8.*eta*PI*(xij**2 + yij**2 + zij**2)**6.5),(270*b**6*
          (xij**6 + 8*yij**6 - 116*yij**4*zij**2 + 101*yij**2*zij**4 - 6*zij**6 - xij**4*(11*yij**2 + 4*zij**2) + 
            xij**2*(-4*yij**4 + 90*yij**2*zij**2 - 11*zij**4)) - 
         7*b**4*(9*xij**8 + 104*yij**8 - 1364*yij**6*zij**2 - 271*yij**4*zij**4 + 1131*yij**2*zij**6 - 66*zij**8 - 
            3*xij**6*(38*yij**2 + 13*zij**2) + xij**4*(-151*yij**4 + 903*yij**2*zij**2 - 171*zij**4) + 
            xij**2*(76*yij**6 - 422*yij**4*zij**2 + 2148*yij**2*zij**4 - 189*zij**6)))/(168.*eta*PI*(xij**2 + yij**2 + zij**2)**6.5),
      (5*b**4*yij*zij*(15*xij**6 - 34*yij**6 + 87*yij**4*zij**2 + 87*yij**2*zij**4 - 34*zij**6 - 4*xij**4*(yij**2 + zij**2) + 
           xij**2*(-53*yij**4 + 83*yij**2*zij**2 - 53*zij**4) - 
           54*b**2*(xij**4 - 2*yij**4 + 7*yij**2*zij**2 - 2*zij**4 - xij**2*(yij**2 + zij**2))))/(8.*eta*PI*(xij**2 + yij**2 + zij**2)**6.5),
      (270*b**6*(xij**6 - 6*yij**6 + 101*yij**4*zij**2 - 116*yij**2*zij**4 + 8*zij**6 - xij**4*(4*yij**2 + 11*zij**2) + 
            xij**2*(-11*yij**4 + 90*yij**2*zij**2 - 4*zij**4)) - 
         7*b**4*(9*xij**8 - 66*yij**8 + 1131*yij**6*zij**2 - 271*yij**4*zij**4 - 1364*yij**2*zij**6 + 104*zij**8 - 
            3*xij**6*(13*yij**2 + 38*zij**2) + xij**4*(-171*yij**4 + 903*yij**2*zij**2 - 151*zij**4) + 
            xij**2*(-189*yij**6 + 2148*yij**4*zij**2 - 422*yij**2*zij**4 + 76*zij**6)))/(168.*eta*PI*(xij**2 + yij**2 + zij**2)**6.5)]])



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
    return np.array([[(-36*b**6*xij*(8*xij**4 - 40*xij**2*(yij**2 + zij**2) + 15*(yij**2 + zij**2)**2) + 
         7*b**4*xij*(16*xij**6 - 52*xij**4*(yij**2 + zij**2) - 47*xij**2*(yij**2 + zij**2)**2 + 21*(yij**2 + zij**2)**3))/
       (21.*(xij**2 + yij**2 + zij**2)**5.5),(-108*b**6*yij*(8*xij**4 - 12*xij**2*(yij**2 + zij**2) + (yij**2 + zij**2)**2) + 
         7*b**4*yij*(44*xij**6 - 13*xij**4*(yij**2 + zij**2) - 53*xij**2*(yij**2 + zij**2)**2 + 4*(yij**2 + zij**2)**3))/
       (21.*(xij**2 + yij**2 + zij**2)**5.5),(-108*b**6*zij*(8*xij**4 - 12*xij**2*(yij**2 + zij**2) + (yij**2 + zij**2)**2) + 
         7*b**4*zij*(44*xij**6 - 13*xij**4*(yij**2 + zij**2) - 53*xij**2*(yij**2 + zij**2)**2 + 4*(yij**2 + zij**2)**3))/
       (21.*(xij**2 + yij**2 + zij**2)**5.5),(b**4*xij*(36*b**2*
            (4*xij**4 + xij**2*(-41*yij**2 + zij**2) + 3*(6*yij**4 + 5*yij**2*zij**2 - zij**4)) - 
           7*(8*xij**6 + 7*(4*yij**2 - zij**2)*(yij**2 + zij**2)**2 + xij**4*(-61*yij**2 + 9*zij**2) - 
              xij**2*(41*yij**4 + 47*yij**2*zij**2 + 6*zij**4))))/(21.*(xij**2 + yij**2 + zij**2)**5.5),
      (b**4*xij*yij*zij*(2*xij**2 - yij**2 - zij**2)*(-108*b**2 + 35*(xij**2 + yij**2 + zij**2)))/(3.*(xij**2 + yij**2 + zij**2)**5.5)],
     [-0.015873015873015872*(b**4*yij*(-896*xij**6 + 322*xij**4*(yij**2 + zij**2) + 1127*xij**2*(yij**2 + zij**2)**2 - 
            91*(yij**2 + zij**2)**3 + 324*b**2*(8*xij**4 - 12*xij**2*(yij**2 + zij**2) + (yij**2 + zij**2)**2)))/
        (xij**2 + yij**2 + zij**2)**5.5,(b**4*xij*(-35*(4*xij**6 - 25*xij**2*yij**2*(yij**2 + zij**2) + 
              (17*yij**2 - 2*zij**2)*(yij**2 + zij**2)**2 + xij**4*(-38*yij**2 + 6*zij**2)) + 
           108*b**2*(4*xij**4 + xij**2*(-41*yij**2 + zij**2) + 3*(6*yij**4 + 5*yij**2*zij**2 - zij**4))))/(63.*(xij**2 + yij**2 + zij**2)**5.5)
       ,(b**4*xij*yij*zij*(220*xij**4 - 324*b**2*(2*xij**2 - yij**2 - zij**2) + 125*xij**2*(yij**2 + zij**2) - 95*(yij**2 + zij**2)**2))/
       (9.*(xij**2 + yij**2 + zij**2)**5.5),(b**4*yij*(108*b**2 - 35*(xij**2 + yij**2 + zij**2))*
         (18*xij**4 + 4*yij**4 + yij**2*zij**2 - 3*zij**4 + xij**2*(-41*yij**2 + 15*zij**2)))/(63.*(xij**2 + yij**2 + zij**2)**5.5),
      (b**4*zij*(108*b**2*(6*xij**4 + 6*yij**4 + 5*yij**2*zij**2 - zij**4 + xij**2*(-51*yij**2 + 5*zij**2)) - 
           7*(26*xij**6 - 8*xij**4*(29*yij**2 - 6*zij**2) + (31*yij**2 - 4*zij**2)*(yij**2 + zij**2)**2 + 
              xij**2*(-227*yij**4 - 209*yij**2*zij**2 + 18*zij**4))))/(63.*(xij**2 + yij**2 + zij**2)**5.5)],
     [-0.015873015873015872*(b**4*zij*(-896*xij**6 + 322*xij**4*(yij**2 + zij**2) + 1127*xij**2*(yij**2 + zij**2)**2 - 
            91*(yij**2 + zij**2)**3 + 324*b**2*(8*xij**4 - 12*xij**2*(yij**2 + zij**2) + (yij**2 + zij**2)**2)))/
        (xij**2 + yij**2 + zij**2)**5.5,(b**4*xij*yij*zij*
         (220*xij**4 - 324*b**2*(2*xij**2 - yij**2 - zij**2) + 125*xij**2*(yij**2 + zij**2) - 95*(yij**2 + zij**2)**2))/
       (9.*(xij**2 + yij**2 + zij**2)**5.5),(b**4*xij*(-35*
            (4*xij**6 + xij**4*(6*yij**2 - 38*zij**2) - 25*xij**2*zij**2*(yij**2 + zij**2) - (2*yij**2 - 17*zij**2)*(yij**2 + zij**2)**2) + 
           108*b**2*(4*xij**4 + xij**2*(yij**2 - 41*zij**2) - 3*(yij**4 - 5*yij**2*zij**2 - 6*zij**4))))/(63.*(xij**2 + yij**2 + zij**2)**5.5),
      (b**4*zij*(108*b**2*(6*xij**4 + 6*yij**4 + 5*yij**2*zij**2 - zij**4 + xij**2*(-51*yij**2 + 5*zij**2)) - 
           7*(38*xij**6 + 7*(4*yij**2 - zij**2)*(yij**2 + zij**2)**2 + xij**4*(-211*yij**2 + 69*zij**2) + 
              xij**2*(-221*yij**4 - 197*yij**2*zij**2 + 24*zij**4))))/(63.*(xij**2 + yij**2 + zij**2)**5.5),
      (b**4*yij*(108*b**2*(6*xij**4 - yij**4 + 5*yij**2*zij**2 + 6*zij**4 + xij**2*(5*yij**2 - 51*zij**2)) - 
           7*(26*xij**6 + 8*xij**4*(6*yij**2 - 29*zij**2) - (4*yij**2 - 31*zij**2)*(yij**2 + zij**2)**2 + 
              xij**2*(18*yij**4 - 209*yij**2*zij**2 - 227*zij**4))))/(63.*(xij**2 + yij**2 + zij**2)**5.5)],
     [(b**4*xij*(108*b**2 - 35*(xij**2 + yij**2 + zij**2))*
         (4*xij**4 + xij**2*(-41*yij**2 + zij**2) + 3*(6*yij**4 + 5*yij**2*zij**2 - zij**4)))/(63.*(xij**2 + yij**2 + zij**2)**5.5),
      (b**4*yij*(108*b**2*(18*xij**4 + 4*yij**4 + yij**2*zij**2 - 3*zij**4 + xij**2*(-41*yij**2 + 15*zij**2)) - 
           35*(17*xij**6 + 4*yij**6 + 6*yij**4*zij**2 - 2*zij**6 + xij**4*(-25*yij**2 + 32*zij**2) + 
              xij**2*(-38*yij**4 - 25*yij**2*zij**2 + 13*zij**4))))/(63.*(xij**2 + yij**2 + zij**2)**5.5),
      (b**4*zij*(108*b**2*(6*xij**4 + 6*yij**4 + 5*yij**2*zij**2 - zij**4 + xij**2*(-51*yij**2 + 5*zij**2)) - 
           7*(31*xij**6 + 2*(13*yij**2 - 2*zij**2)*(yij**2 + zij**2)**2 + xij**4*(-227*yij**2 + 58*zij**2) + 
              xij**2*(-232*yij**4 - 209*yij**2*zij**2 + 23*zij**4))))/(63.*(xij**2 + yij**2 + zij**2)**5.5),
      (-324*b**6*xij*(xij**4 + 8*yij**4 - 12*yij**2*zij**2 + zij**4 + 2*xij**2*(-6*yij**2 + zij**2)) + 
         7*b**4*xij*(13*xij**6 + 128*yij**6 - 46*yij**4*zij**2 - 161*yij**2*zij**4 + 13*zij**6 + xij**4*(-161*yij**2 + 39*zij**2) + 
            xij**2*(-46*yij**4 - 322*yij**2*zij**2 + 39*zij**4)))/(63.*(xij**2 + yij**2 + zij**2)**5.5),
      (b**4*xij*yij*zij*(324*b**2*(xij**2 - 2*yij**2 + zij**2) - 
           5*(19*xij**4 - 44*yij**4 - 25*yij**2*zij**2 + 19*zij**4 + xij**2*(-25*yij**2 + 38*zij**2))))/(9.*(xij**2 + yij**2 + zij**2)**5.5)],
     [(b**4*xij*yij*zij*(2*xij**2 - yij**2 - zij**2)*(-108*b**2 + 35*(xij**2 + yij**2 + zij**2)))/(3.*(xij**2 + yij**2 + zij**2)**5.5),
      (b**4*zij*(36*b**2*(6*xij**4 + 6*yij**4 + 5*yij**2*zij**2 - zij**4 + xij**2*(-51*yij**2 + 5*zij**2)) - 
           7*(9*xij**6 + (9*yij**2 - zij**2)*(yij**2 + zij**2)**2 + xij**4*(-78*yij**2 + 17*zij**2) + 
              xij**2*(-78*yij**4 - 71*yij**2*zij**2 + 7*zij**4))))/(21.*(xij**2 + yij**2 + zij**2)**5.5),
      (b**4*yij*(36*b**2*(6*xij**4 - yij**4 + 5*yij**2*zij**2 + 6*zij**4 + xij**2*(5*yij**2 - 51*zij**2)) - 
           7*(9*xij**6 + xij**4*(17*yij**2 - 78*zij**2) - (yij**2 - 9*zij**2)*(yij**2 + zij**2)**2 + 
              xij**2*(7*yij**4 - 71*yij**2*zij**2 - 78*zij**4))))/(21.*(xij**2 + yij**2 + zij**2)**5.5),
      (b**4*xij*yij*zij*(xij**2 - 2*yij**2 + zij**2)*(108*b**2 - 35*(xij**2 + yij**2 + zij**2)))/(3.*(xij**2 + yij**2 + zij**2)**5.5),
      (b**4*xij*(-36*b**2*(xij**4 - 6*yij**4 + 51*yij**2*zij**2 - 6*zij**4 - 5*xij**2*(yij**2 + zij**2)) + 
           7*(xij**6 - 9*yij**6 + 78*yij**4*zij**2 + 78*yij**2*zij**4 - 9*zij**6 - 7*xij**4*(yij**2 + zij**2) + 
              xij**2*(-17*yij**4 + 71*yij**2*zij**2 - 17*zij**4))))/(21.*(xij**2 + yij**2 + zij**2)**5.5)],
     [(b**4*xij*(108*b**2 - 35*(xij**2 + yij**2 + zij**2))*
         (4*xij**4 + xij**2*(yij**2 - 41*zij**2) - 3*(yij**4 - 5*yij**2*zij**2 - 6*zij**4)))/(63.*(xij**2 + yij**2 + zij**2)**5.5),
      (b**4*yij*(108*b**2*(6*xij**4 - yij**4 + 5*yij**2*zij**2 + 6*zij**4 + xij**2*(5*yij**2 - 51*zij**2)) - 
           7*(31*xij**6 + xij**4*(58*yij**2 - 227*zij**2) - 2*(2*yij**2 - 13*zij**2)*(yij**2 + zij**2)**2 + 
              xij**2*(23*yij**4 - 209*yij**2*zij**2 - 232*zij**4))))/(63.*(xij**2 + yij**2 + zij**2)**5.5),
      (b**4*zij*(108*b**2*(18*xij**4 - 3*yij**4 + yij**2*zij**2 + 4*zij**4 + xij**2*(15*yij**2 - 41*zij**2)) - 
           35*(17*xij**6 - 2*yij**6 + 6*yij**2*zij**4 + 4*zij**6 + xij**4*(32*yij**2 - 25*zij**2) + 
              xij**2*(13*yij**4 - 25*yij**2*zij**2 - 38*zij**4))))/(63.*(xij**2 + yij**2 + zij**2)**5.5),
      (b**4*xij*(-108*b**2*(xij**4 - 6*yij**4 + 51*yij**2*zij**2 - 6*zij**4 - 5*xij**2*(yij**2 + zij**2)) + 
           7*(7*xij**6 - 28*yij**6 + 221*yij**4*zij**2 + 211*yij**2*zij**4 - 38*zij**6 - 2*xij**4*(7*yij**2 + 12*zij**2) + 
              xij**2*(-49*yij**4 + 197*yij**2*zij**2 - 69*zij**4))))/(63.*(xij**2 + yij**2 + zij**2)**5.5),
      (b**4*xij*yij*zij*(324*b**2*(xij**2 + yij**2 - 2*zij**2) - 
           5*(19*xij**4 + 19*yij**4 - 25*yij**2*zij**2 - 44*zij**4 + xij**2*(38*yij**2 - 25*zij**2))))/(9.*(xij**2 + yij**2 + zij**2)**5.5)],
     [(b**4*yij*(36*b**2*(18*xij**4 + 4*yij**4 + yij**2*zij**2 - 3*zij**4 + xij**2*(-41*yij**2 + 15*zij**2)) - 
           7*(28*xij**6 + (8*yij**2 - 7*zij**2)*(yij**2 + zij**2)**2 + xij**4*(-41*yij**2 + 49*zij**2) + 
              xij**2*(-61*yij**4 - 47*yij**2*zij**2 + 14*zij**4))))/(21.*(xij**2 + yij**2 + zij**2)**5.5),
      (-108*b**6*xij*(xij**4 + 8*yij**4 - 12*yij**2*zij**2 + zij**4 + 2*xij**2*(-6*yij**2 + zij**2)) + 
         7*b**4*xij*(4*xij**6 + 44*yij**6 - 13*yij**4*zij**2 - 53*yij**2*zij**4 + 4*zij**6 + xij**4*(-53*yij**2 + 12*zij**2) + 
            xij**2*(-13*yij**4 - 106*yij**2*zij**2 + 12*zij**4)))/(21.*(xij**2 + yij**2 + zij**2)**5.5),
      (b**4*xij*yij*zij*(xij**2 - 2*yij**2 + zij**2)*(108*b**2 - 35*(xij**2 + yij**2 + zij**2)))/(3.*(xij**2 + yij**2 + zij**2)**5.5),
      (-36*b**6*yij*(15*xij**4 + 8*yij**4 - 40*yij**2*zij**2 + 15*zij**4 + xij**2*(-40*yij**2 + 30*zij**2)) + 
         7*b**4*yij*(21*xij**6 + 16*yij**6 - 52*yij**4*zij**2 - 47*yij**2*zij**4 + 21*zij**6 + xij**4*(-47*yij**2 + 63*zij**2) + 
            xij**2*(-52*yij**4 - 94*yij**2*zij**2 + 63*zij**4)))/(21.*(xij**2 + yij**2 + zij**2)**5.5),
      (-108*b**6*zij*(xij**4 + 8*yij**4 - 12*yij**2*zij**2 + zij**4 + 2*xij**2*(-6*yij**2 + zij**2)) + 
         7*b**4*zij*(4*xij**6 + 44*yij**6 - 13*yij**4*zij**2 - 53*yij**2*zij**4 + 4*zij**6 + xij**4*(-53*yij**2 + 12*zij**2) + 
            xij**2*(-13*yij**4 - 106*yij**2*zij**2 + 12*zij**4)))/(21.*(xij**2 + yij**2 + zij**2)**5.5)],
     [(b**4*zij*(108*b**2*(6*xij**4 + 6*yij**4 + 5*yij**2*zij**2 - zij**4 + xij**2*(-51*yij**2 + 5*zij**2)) - 
           7*(28*xij**6 + (38*yij**2 - 7*zij**2)*(yij**2 + zij**2)**2 + xij**4*(-221*yij**2 + 49*zij**2) + 
              xij**2*(-211*yij**4 - 197*yij**2*zij**2 + 14*zij**4))))/(63.*(xij**2 + yij**2 + zij**2)**5.5),
      (b**4*xij*yij*zij*(324*b**2*(xij**2 - 2*yij**2 + zij**2) - 
           5*(19*xij**4 - 44*yij**4 - 25*yij**2*zij**2 + 19*zij**4 + xij**2*(-25*yij**2 + 38*zij**2))))/(9.*(xij**2 + yij**2 + zij**2)**5.5),
      (b**4*xij*(-108*b**2*(xij**4 - 6*yij**4 + 51*yij**2*zij**2 - 6*zij**4 - 5*xij**2*(yij**2 + zij**2)) + 
           7*(4*xij**6 - 26*yij**6 + 232*yij**4*zij**2 + 227*yij**2*zij**4 - 31*zij**6 - xij**4*(18*yij**2 + 23*zij**2) + 
              xij**2*(-48*yij**4 + 209*yij**2*zij**2 - 58*zij**4))))/(63.*(xij**2 + yij**2 + zij**2)**5.5),
      (-324*b**6*zij*(xij**4 + 8*yij**4 - 12*yij**2*zij**2 + zij**4 + 2*xij**2*(-6*yij**2 + zij**2)) + 
         7*b**4*zij*(13*xij**6 + 128*yij**6 - 46*yij**4*zij**2 - 161*yij**2*zij**4 + 13*zij**6 + xij**4*(-161*yij**2 + 39*zij**2) + 
            xij**2*(-46*yij**4 - 322*yij**2*zij**2 + 39*zij**4)))/(63.*(xij**2 + yij**2 + zij**2)**5.5),
      (b**4*yij*(-108*b**2*(3*xij**4 - 4*yij**4 + 41*yij**2*zij**2 - 18*zij**4 - xij**2*(yij**2 + 15*zij**2)) + 
           35*(2*xij**6 - 4*yij**6 - 13*xij**4*zij**2 + 38*yij**4*zij**2 + 25*yij**2*zij**4 - 17*zij**6 + 
              xij**2*(-6*yij**4 + 25*yij**2*zij**2 - 32*zij**4))))/(63.*(xij**2 + yij**2 + zij**2)**5.5)],
     [(b**4*yij*(108*b**2*(6*xij**4 - yij**4 + 5*yij**2*zij**2 + 6*zij**4 + xij**2*(5*yij**2 - 51*zij**2)) - 
           7*(28*xij**6 + xij**4*(49*yij**2 - 221*zij**2) - (7*yij**2 - 38*zij**2)*(yij**2 + zij**2)**2 + 
              xij**2*(14*yij**4 - 197*yij**2*zij**2 - 211*zij**4))))/(63.*(xij**2 + yij**2 + zij**2)**5.5),
      (b**4*xij*(-108*b**2*(xij**4 - 6*yij**4 + 51*yij**2*zij**2 - 6*zij**4 - 5*xij**2*(yij**2 + zij**2)) + 
           7*(4*xij**6 - 31*yij**6 + 227*yij**4*zij**2 + 232*yij**2*zij**4 - 26*zij**6 - xij**4*(23*yij**2 + 18*zij**2) + 
              xij**2*(-58*yij**4 + 209*yij**2*zij**2 - 48*zij**4))))/(63.*(xij**2 + yij**2 + zij**2)**5.5),
      (b**4*xij*yij*zij*(324*b**2*(xij**2 + yij**2 - 2*zij**2) - 
           5*(19*xij**4 + 19*yij**4 - 25*yij**2*zij**2 - 44*zij**4 + xij**2*(38*yij**2 - 25*zij**2))))/(9.*(xij**2 + yij**2 + zij**2)**5.5),
      (b**4*yij*(108*b**2 - 35*(xij**2 + yij**2 + zij**2))*
         (-3*xij**4 + 4*yij**4 - 41*yij**2*zij**2 + 18*zij**4 + xij**2*(yij**2 + 15*zij**2)))/(63.*(xij**2 + yij**2 + zij**2)**5.5),
      (b**4*zij*(-108*b**2*(3*xij**4 - 18*yij**4 + 41*yij**2*zij**2 - 4*zij**4 - xij**2*(15*yij**2 + zij**2)) + 
           35*(2*xij**6 - 13*xij**4*yij**2 - 17*yij**6 + 25*yij**4*zij**2 + 38*yij**2*zij**4 - 4*zij**6 + 
              xij**2*(-32*yij**4 + 25*yij**2*zij**2 - 6*zij**4))))/(63.*(xij**2 + yij**2 + zij**2)**5.5)]])



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
     [(-9*b**5*(4*xij**4 - yij**4 + 3*yij**2*zij**2 + 4*zij**4 + 3*xij**2*(yij**2 - 9*zij**2)))/(50.*(xij**2 + yij**2 + zij**2)**4.5),
      (-9*b**5*xij*yij*(xij**2 + yij**2 - 6*zij**2))/(10.*(xij**2 + yij**2 + zij**2)**4.5),
      (-9*b**5*xij*zij*(3*xij**2 + 3*yij**2 - 4*zij**2))/(10.*(xij**2 + yij**2 + zij**2)**4.5)],
     [(-9*b**5*xij*yij*(3*xij**2 - 4*yij**2 + 3*zij**2))/(10.*(xij**2 + yij**2 + zij**2)**4.5),
      (9*b**5*(3*xij**4 + 8*yij**4 - 24*yij**2*zij**2 + 3*zij**4 + 6*xij**2*(-4*yij**2 + zij**2)))/(50.*(xij**2 + yij**2 + zij**2)**4.5),
      (9*b**5*yij*zij*(-3*xij**2 + 4*yij**2 - 3*zij**2))/(10.*(xij**2 + yij**2 + zij**2)**4.5)],
     [(-9*b**5*xij*zij*(xij**2 - 6*yij**2 + zij**2))/(10.*(xij**2 + yij**2 + zij**2)**4.5),
      (9*b**5*yij*zij*(-3*xij**2 + 4*yij**2 - 3*zij**2))/(10.*(xij**2 + yij**2 + zij**2)**4.5),
      (9*b**5*(xij**4 - 4*yij**4 + 27*yij**2*zij**2 - 4*zij**4 - 3*xij**2*(yij**2 + zij**2)))/(50.*(xij**2 + yij**2 + zij**2)**4.5)],
     [(-9*b**5*xij*yij*(xij**2 + yij**2 - 6*zij**2))/(10.*(xij**2 + yij**2 + zij**2)**4.5),
      (9*b**5*(xij**4 - 4*yij**4 + 27*yij**2*zij**2 - 4*zij**4 - 3*xij**2*(yij**2 + zij**2)))/(50.*(xij**2 + yij**2 + zij**2)**4.5),
      (-9*b**5*yij*zij*(3*xij**2 + 3*yij**2 - 4*zij**2))/(10.*(xij**2 + yij**2 + zij**2)**4.5)]])



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
    return np.zeros([3,9])


def K3t2s(xij,yij,zij, b,eta):
    return np.zeros([3,5])

def K3t3t(xij,yij,zij, b,eta):
    return np.zeros([3,3])

def K3t3a(xij,yij,zij, b,eta):
    return np.zeros([3,5])

def K3t3s(xij,yij,zij, b,eta):
    return np.zeros([3,9])


def G2a3t(xij,yij,zij, b,eta):
    return np.zeros([3,3])

def G3a3t(xij,yij,zij, b,eta):
    return np.zeros([5,3])


def K2a3t(xij,yij,zij, b,eta):
    return np.zeros([3,3])

def K3a3t(xij,yij,zij, b,eta):
    return np.zeros([5,3])
