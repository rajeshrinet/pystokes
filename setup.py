import numpy, os, sys, os.path, tempfile, subprocess, shutil
#try:
#    from setuptools import setup, Extension, find_packages
#except ImportError:
from distutils.core import setup
from distutils.extension import Extension
import Cython.Compiler.Options
from Cython.Build import cythonize



def checkOpenmpSupport():
    """ Adapted from https://stackoverflow.com/questions/16549893/programatically-testing-for-openmp-support-from-a-python-setup-script
    """ 
    ompTest = \
    r"""
    #include <omp.h>
    #include <stdio.h>
    int main() {
    #pragma omp parallel
    printf("Thread %d, Total number of threads %d\n", omp_get_thread_num(), omp_get_num_threads());
    }
    """
    tmpdir = tempfile.mkdtemp()
    curdir = os.getcwd()
    os.chdir(tmpdir)

    filename = r'test.c'
    with open(filename, 'w') as file:
        file.write(ompTest)
    with open(os.devnull, 'w') as fnull:
        result = subprocess.call(['cc', '-fopenmp', filename],
                                 stdout=fnull, stderr=fnull)

    os.chdir(curdir)
    shutil.rmtree(tmpdir) 
    if result == 0:
        return True
    else:
        return False

if checkOpenmpSupport() == True:
    ompArgs = ['-fopenmp']
else:
    ompArgs = None 




#installation of PyForces
setup(
    name='pyforces',
    version='1.0.0',
    url='https://gitlab.com/rajeshrinet/pystokes',
    author = 'The PyStokes team',
    author_email = 'PyStokes@googlegroups.com',
    license='MIT',
    description='force fields for computing Stokes flows',
    long_description='pyforces is the library for force fields used in computing Stokes flows',
    platforms='tested on LINUX',
    ext_modules=cythonize([ Extension("pyforces/*", ["pyforces/*.pyx"],
        include_dirs=[numpy.get_include()],
        extra_compile_args=ompArgs,
        extra_link_args=ompArgs,
        )],
        compiler_directives={"language_level": sys.version_info[0]},
        ),
    libraries=[],
    #zip_safe = True,
    packages=['pyforces'],
)




#installation of PyLaplace
setup(
    name='pylaplace',
    version='1.0.0',
    url='https://gitlab.com/rajeshrinet/pystokes',
    author = 'The PyStokes team',
    author_email = 'PyStokes@googlegroups.com',
    license='MIT',
    description='Solving Laplace equation in python',
    long_description='python library for numerical simulation of long-ranged interactions between colloids given by the Laplace equation',
    platforms='tested on LINUX',
    ext_modules=cythonize([ Extension("pylaplace/*", ["pylaplace/*.pyx"],
        include_dirs=[numpy.get_include()],
        extra_compile_args=ompArgs,
        extra_link_args=ompArgs,
        )],
        compiler_directives={"language_level": sys.version_info[0]},
        ),
    libraries=[],
    #zip_safe = True,
    packages=['pylaplace'],
)




#installation of PyStokes
setup(
    name='pystokes',
    version='1.0.0',
    url='https://gitlab.com/rajeshrinet/pystokes',
    author = 'The PyStokes team',
    author_email = 'PyStokes@googlegroups.com',
    license='MIT',
    description='python library for computing Stokes flows',
    long_description='pystokes is a library for computing Stokes flows in various geometries',
    platforms='tested on LINUX',
    ext_modules=cythonize([ Extension("pystokes/*", ["pystokes/*.pyx"],
        include_dirs=[numpy.get_include()],
        extra_compile_args=ompArgs,
        extra_link_args=ompArgs,
        )],
        compiler_directives={"language_level": sys.version_info[0]},
        ),
    libraries=[],
    #zip_safe = True,
    packages=['pystokes'],
)

