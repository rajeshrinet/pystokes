import os

owd = os.getcwd()

os.chdir('pystokes/')
os.system('python setup.py install')

os.chdir(owd)
os.chdir('pyforces/')
os.system('python setup.py install')

os.chdir(owd)
os.chdir('examples/unbounded/activeColloidsLatticeTraps/')
os.system('python setup.py install')

os.chdir(owd)

