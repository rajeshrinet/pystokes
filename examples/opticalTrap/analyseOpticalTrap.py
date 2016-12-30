from pylab import *
import scipy.io as sio
from scipy.signal import square

data  = sio.loadmat('Np=2.mat')
X     = data['X']
tm    = data['t']



data  = sio.loadmat('Np=2.mat')
X     = data['X']
tm    = data['t']
tau = (data['tau']).reshape(1)
lmda1 = (data['lmda1']).reshape(1)
lmda2 = (data['lmda2']).reshape(1)
lmda3 = (data['lmda3']).reshape(1)


NN = np.size(X[:, 1]); t = tm.reshape(NN)
F = lmda3*square(2*pi*t/tau)


# Now plot the results
f = plt.figure(num=None, figsize=(18, 6), dpi=80, facecolor='w', edgecolor='k')
f.add_subplot(1, 2, 1) 
plot(t, 0.002*F, t, X[:,0]- np.mean(X[10:,0]), linewidth=2)
xlabel(r'$t/\tau$', fontsize=20)
ylabel('position and force of first particle',fontsize=16);
xlim([5*tau, 8*tau]); #ylim([-1.2, 1.2])
legend( ('Force', 'Particle1'), loc='lower left')

f.add_subplot(1, 2, 2) 
plot(t, 0.002*F , t , X[:, 0] - np.mean(X[10:,0]), t , X[:, 3] - np.mean(X[10:,3]), linewidth=2) 
xlabel(r'$t/\tau$', fontsize=20);
xlim([5*tau, 8*tau]); #ylim([-1.2, 1.2])
ylabel('Position about the mean',fontsize=16)
legend( ('Force', 'Particle1', 'Particle2'), loc='lower left');
show()
