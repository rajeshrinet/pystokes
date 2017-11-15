# change the first line to the absolute path of python binary of
# anaconda/epdpython/Sage if you don't want to use the default version
# of python

#PYTHON = python
PYTHON = ~/software/anaconda/bin/python

all: opticalTrap.so

opticalTrap.so: opticalTrap.pyx
	$(PYTHON) setup.py build_ext --inplace

clean:
	rm -rf build opticalTrap.so opticalTrap.c opticalTrap.html
