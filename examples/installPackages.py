import os
os.chdir(../pystokes/)
python setup.py install
os.chdir(../pyforces/)
python setup.py install
os.chdir(unbounded/activeColloidsLatticeTraps/)
python setup.py install
