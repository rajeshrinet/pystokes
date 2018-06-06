import os

owd = os.getcwd()

os.chdir('pystokes/')
os.system('/srv/conda/bin/python setup.py install')

os.chdir(owd)
os.chdir('pyforces/')
os.system('/srv/conda/bin/python setup.py install')

os.chdir(owd)
os.chdir('examples/unbounded/activeColloidsLatticeTraps/')
os.system('/srv/conda/bin/python setup.py install')

os.chdir(owd)

