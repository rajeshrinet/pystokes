import numpy, os, sys, os.path, tempfile, subprocess, shutil
import os, sys
from setuptools import setup, Extension
from Cython.Build import cythonize
import Cython.Compiler.Options
Cython.Compiler.Options.annotate=True


def checkOpenmpSupport():
    ''' Adapted from https://stackoverflow.com/questions/16549893/programatically-testing-for-openmp-support-from-a-python-setup-script
    ''' 
    ompTest = \
    r'''
    #include <omp.h>
    #include <stdio.h>
    int main() {
    #pragma omp parallel
    printf('Thread %d, Total number of threads %d\n', omp_get_thread_num(), omp_get_num_threads());
    }
    '''
    tmpdir = tempfile.mkdtemp()
    curdir = os.getcwd()
    os.chdir(tmpdir)

    filename = r'test.c'
    try:
        with open(filename, 'w') as file:
            file.write(ompTest)
        with open(os.devnull, 'w') as fnull:
            result = subprocess.call(['cc', '-fopenmp', filename],
                                     stdout=fnull, stderr=fnull)
    except:
        print('Failed to test for OpenMP support. Assuming unavailability');
        result = -1;
    
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


with open("README.md", "r") as fh:
    long_description = fh.read()


setup(
    name='pystokes',
    version='2.1.8',
    url='https://github.com/rajeshrinet/pystokes',
    author = 'The PyStokes team',
    author_email = 'PyStokes@googlegroups.com',
    license='MIT',
    description='Stokesian hydrodynamics in Python',
    long_description=long_description,
    long_description_content_type='text/markdown',
    platforms='tested on LINUX',
    ext_modules=cythonize([ Extension('pystokes/*', ['pystokes/*.pyx'],
        include_dirs=[numpy.get_include()],
        extra_compile_args=ompArgs,
        extra_link_args=ompArgs,
        )],
        compiler_directives={'language_level': sys.version_info[0]},
        ),
    libraries=[],
    packages=['pystokes'],
    package_data={'pystokes': ['*.pxd']},
    include_package_data=True,
    classifiers=[
        'License :: OSI Approved :: MIT License',
        'Programming Language :: Python',
        'Programming Language :: Python :: 2',
        'Programming Language :: Python :: 2.7',
        'Programming Language :: Python :: 3',
        'Programming Language :: Python :: 3.7',
        ],
    install_requires=['cython','numpy','scipy','matplotlib']
)

