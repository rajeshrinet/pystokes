
#PYTHON = python
PYTHON = ~/software/anaconda/bin/python

all: trap.so

trap.so: trap.pyx
	$(PYTHON) setup.py build_ext --inplace

clean:
	rm -rf build trap.so trap.c trap.html
