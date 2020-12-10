import numpy, os, sys, os.path, tempfile, subprocess, shutil
import os, sys, re
from setuptools import setup, Extension
from Cython.Build import cythonize
import Cython.Compiler.Options
Cython.Compiler.Options.annotate=True


def checkOpenmpSupport():
    ''' 
    Function to check openMP support
    Adapted from https://stackoverflow.com/questions/16549893/ 
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


### Checking for openMP support
if checkOpenmpSupport() == True:
    ompArgs = ['-fopenmp']
else:
    ompArgs = None 


with open("README.md", "r") as fh:
    long_description = fh.read()


### Setting version names
cwd = os.path.abspath(os.path.dirname(__file__))
with open(os.path.join(cwd, 'pystokes', '__init__.py')) as fp:
    for line in fp:
        m = re.search(r'^\s*__version__\s*=\s*([\'"])([^\'"]+)\1\s*$', line)
        if m:
            version = m.group(2)
            break
    else:
        raise RuntimeError('Unable to find own __version__ string')


### Extensions to be compiled 
extension1 = Extension('pystokes/*', ['pystokes/*.pyx'],
        include_dirs=[numpy.get_include()],
        extra_compile_args=ompArgs,
        extra_link_args=ompArgs,
        )
extension2 = Extension('pystokes/phoretic/*', ['pystokes/phoretic/*.pyx'],
        include_dirs=[numpy.get_include()],
        extra_compile_args=ompArgs,
        extra_link_args=ompArgs,
        )


setup(
    name='pystokes',
    version=version,
    url='https://github.com/rajeshrinet/pystokes',
    author = 'The PyStokes team',
    author_email = 'PyStokes@googlegroups.com',
    license='MIT',
    description='Phoresis and Stokesian hydrodynamics in Python',
    long_description=long_description,
    long_description_content_type='text/markdown',
    platforms='tested on macOS, windows, and LINUX',
    ext_modules=cythonize([extension1, extension2 ],
        compiler_directives={'language_level': sys.version_info[0],
                            'linetrace': True},
        ),
    libraries=[],
    packages=['pystokes', 'pystokes/phoretic'],
    package_data={'pystokes': ['*.pxd'], 'pystokes/phoretic': ['*.pxd']},
    include_package_data=True,
    setup_requires=['wheel'],
    classifiers=[
        'License :: OSI Approved :: MIT License',
        "Operating System :: OS Independent",
        'Programming Language :: Python',
        'Programming Language :: Python :: 2',
        'Programming Language :: Python :: 2.7',
        'Programming Language :: Python :: 3',
        'Programming Language :: Python :: 3.7',
        ],
    install_requires=['cython','numpy','scipy','matplotlib']
)

