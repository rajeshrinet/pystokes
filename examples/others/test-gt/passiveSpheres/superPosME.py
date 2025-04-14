import numpy
PI = 3.14159265359

def G1s1sF(xij,yij,zij, b,eta, force):
    return numpy.array([xij*yij*(-2*b**2 + xij**2 + yij**2 + zij**2)*force[1]/(8*PI*eta*(xij**2 + yij**2 + zij**2)**(5/2)) + xij*zij*(-2*b**2 + xij**2 + yij**2 + zij**2)*force[2]/(8*PI*eta*(xij**2 + yij**2 + zij**2)**(5/2)) + (-2*b**2*(2*xij**2 - yij**2 - zij**2) + 3*(xij**2 + yij**2 + zij**2)*(2*xij**2 + yij**2 + zij**2))*force[0]/(24*PI*eta*(xij**2 + yij**2 + zij**2)**(5/2)), xij*yij*(-2*b**2 + xij**2 + yij**2 + zij**2)*force[0]/(8*PI*eta*(xij**2 + yij**2 + zij**2)**(5/2)) + yij*zij*(-2*b**2 + xij**2 + yij**2 + zij**2)*force[2]/(8*PI*eta*(xij**2 + yij**2 + zij**2)**(5/2)) + (-2*b**2*(-xij**2 + 2*yij**2 - zij**2) + 3*(xij**2 + yij**2 + zij**2)*(xij**2 + 2*yij**2 + zij**2))*force[1]/(24*PI*eta*(xij**2 + yij**2 + zij**2)**(5/2)), xij*zij*(-2*b**2 + xij**2 + yij**2 + zij**2)*force[0]/(8*PI*eta*(xij**2 + yij**2 + zij**2)**(5/2)) + yij*zij*(-2*b**2 + xij**2 + yij**2 + zij**2)*force[1]/(8*PI*eta*(xij**2 + yij**2 + zij**2)**(5/2)) + (-2*b**2*(-xij**2 - yij**2 + 2*zij**2) + 3*(xij**2 + yij**2 + zij**2)*(xij**2 + yij**2 + 2*zij**2))*force[2]/(24*PI*eta*(xij**2 + yij**2 + zij**2)**(5/2))])

def G2a1sF(xij,yij,zij, b,eta, force):
    return numpy.array([-b*yij*force[2]/(4*PI*eta*(xij**2 + yij**2 + zij**2)**(3/2)) + b*zij*force[1]/(4*PI*eta*(xij**2 + yij**2 + zij**2)**(3/2)), b*xij*force[2]/(4*PI*eta*(xij**2 + yij**2 + zij**2)**(3/2)) - b*zij*force[0]/(4*PI*eta*(xij**2 + yij**2 + zij**2)**(3/2)), -b*xij*force[1]/(4*PI*eta*(xij**2 + yij**2 + zij**2)**(3/2)) + b*yij*force[0]/(4*PI*eta*(xij**2 + yij**2 + zij**2)**(3/2))])

def G1s2aT(xij,yij,zij, b,eta, torque):
    return numpy.array([-0.125*b*yij*torque[2]/(PI*eta*(xij**2 + yij**2 + zij**2)**(3/2)) + 0.125*b*zij*torque[1]/(PI*eta*(xij**2 + yij**2 + zij**2)**(3/2)), 0.125*b*xij*torque[2]/(PI*eta*(xij**2 + yij**2 + zij**2)**(3/2)) - 0.125*b*zij*torque[0]/(PI*eta*(xij**2 + yij**2 + zij**2)**(3/2)), -0.125*b*xij*torque[1]/(PI*eta*(xij**2 + yij**2 + zij**2)**(3/2)) + 0.125*b*yij*torque[0]/(PI*eta*(xij**2 + yij**2 + zij**2)**(3/2))])

def G2a2aT(xij,yij,zij, b,eta, torque):
    return numpy.array([0.375*b**2*xij*yij*torque[1]/(PI*eta*(xij**2 + yij**2 + zij**2)**(5/2)) + 0.375*b**2*xij*zij*torque[2]/(PI*eta*(xij**2 + yij**2 + zij**2)**(5/2)) + b**2*(0.25*xij**2 - 0.125*yij**2 - 0.125*zij**2)*torque[0]/(PI*eta*(xij**2 + yij**2 + zij**2)**(5/2)), 0.375*b**2*xij*yij*torque[0]/(PI*eta*(xij**2 + yij**2 + zij**2)**(5/2)) + 0.375*b**2*yij*zij*torque[2]/(PI*eta*(xij**2 + yij**2 + zij**2)**(5/2)) + b**2*(-0.125*xij**2 + 0.25*yij**2 - 0.125*zij**2)*torque[1]/(PI*eta*(xij**2 + yij**2 + zij**2)**(5/2)), 0.375*b**2*xij*zij*torque[0]/(PI*eta*(xij**2 + yij**2 + zij**2)**(5/2)) + 0.375*b**2*yij*zij*torque[1]/(PI*eta*(xij**2 + yij**2 + zij**2)**(5/2)) + b**2*(-0.125*xij**2 - 0.125*yij**2 + 0.25*zij**2)*torque[2]/(PI*eta*(xij**2 + yij**2 + zij**2)**(5/2))])

